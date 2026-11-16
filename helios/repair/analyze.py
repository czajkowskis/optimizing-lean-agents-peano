"""Analyze the repair pilot (see helios/repair/DESIGN.md).

Inputs: <data>/pilot_attempts.jsonl, <data>/selection.json, <out>/compiled.jsonl.
Prints the report and writes <out>/analysis.json.

  H1   Is repair worth it at all? LOCAL vs FRESH on successes per million generated tokens.
       FRESH costs nothing new: each problem's baseline success rate and mean tokens per
       attempt come from its 256 original attempts.
  H2   The main test: is LOCAL's advantage over BACK larger when the error line has high
       entropy? Difference-in-differences of per-attempt success rates, averaged over problems,
       with a problem-bootstrap CI and a permutation test that shuffles strata within problems.
  Policies  A=FRESH, B=LOCAL, C-back, C-fresh, D (random LOCAL/BACK in C-back's proportion,
       computed exactly), E=FEEDBACK. Compared on attempts where every arm has all samples.
  Decision  go / stop / rethink, as written in DESIGN.md.

Usage: python helios/repair/analyze.py --data helios/repair/data --out amu_work/repair
"""
import argparse
import collections
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--compiled", default=None, help="default: <out>/compiled.jsonl")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--perm", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

attempts = {}
for l in open(os.path.join(args.data, "pilot_attempts.jsonl"), encoding="utf-8"):
    a = json.loads(l)
    attempts[a["attempt"]] = a
sel = json.load(open(os.path.join(args.data, "selection.json")))
probs = sel["problems"]

samples = collections.defaultdict(lambda: collections.defaultdict(list))   # attempt -> arm -> [(ok, new tokens, prompt tokens)]
for l in open(args.compiled or os.path.join(args.out, "compiled.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    samples[r["attempt"]][r["arm"]].append((bool(r["complete"]), r["new_tokens"], r["prompt_len"]))

ARMS = ["local", "back", "feedback"]
n_samp = max((len(v) for s in samples.values() for v in s.values()), default=0)
full = [a for a in attempts if all(len(samples[a][arm]) == n_samp for arm in ARMS)]
rb = [a for a in attempts if all(len(samples[a][arm]) == n_samp for arm in ("local", "back"))]
print(f"{len(attempts)} pilot attempts, {n_samp} samples per arm; complete LOCAL+BACK: {len(rb)}, all three arms: {len(full)}")


def att_stats(a, arm):
    """(success rate, mean new tokens, mean prompt tokens) of one attempt under one arm."""
    if arm == "fresh":
        p = probs[attempts[a]["problem"]]
        return p["rate"], p["fresh_mean_tokens"], 0.0
    s = np.array(samples[a][arm], dtype=float)
    return s[:, 0].mean(), s[:, 1].mean(), s[:, 2].mean()


# --- per-arm summary -------------------------------------------------------------------------
report = {"n_attempts": len(attempts), "n_samples_per_arm": n_samp, "n_local_back": len(rb), "n_all_arms": len(full)}
print(f"\n{'arm':9} {'attempts':>8} {'success':>8} {'new tok':>8} {'prompt tok':>10} {'succ/Mtok':>9}")
report["arms"] = {}
for arm in ARMS + ["fresh"]:
    ids = [a for a in attempts if arm == "fresh" or len(samples[a][arm]) == n_samp]
    if not ids:
        continue
    st = np.array([att_stats(a, arm) for a in ids])
    spm = st[:, 0].sum() / st[:, 1].sum() * 1e6
    report["arms"][arm] = {"attempts": len(ids), "success": st[:, 0].mean(), "new_tokens": st[:, 1].mean(),
                           "prompt_tokens": st[:, 2].mean(), "succ_per_Mtok": spm}
    print(f"{arm:9} {len(ids):8} {st[:, 0].mean():8.3f} {st[:, 1].mean():8.0f} {st[:, 2].mean():10.0f} {spm:9.2f}")


# --- step 1 (LOCAL only is enough) -------------------------------------------------------------
loc = [a for a in attempts if len(samples[a]["local"]) >= 1]
if loc:
    def h1_stat(ids):
        """LOCAL minus FRESH successes per million new tokens, over the given attempts."""
        L = np.array([att_stats(a, "local") for a in ids]); F = np.array([att_stats(a, "fresh") for a in ids])
        return L[:, 0].sum() / L[:, 1].sum() * 1e6 - F[:, 0].sum() / F[:, 1].sum() * 1e6

    def strat_diff(ids, strata):
        """Mean over problems of LOCAL success (high stratum) - (low stratum)."""
        by_p = collections.defaultdict(lambda: {"high": [], "low": []})
        for a, s in zip(ids, strata):
            by_p[attempts[a]["problem"]][s].append(att_stats(a, "local")[0])
        per = [np.mean(v["high"]) - np.mean(v["low"]) for v in by_p.values() if v["high"] and v["low"]]
        return float(np.mean(per)) if per else float("nan")

    by_pl = collections.defaultdict(list)
    for a in loc:
        by_pl[attempts[a]["problem"]].append(a)
    keys = list(by_pl)
    h1_0, sd_0 = h1_stat(loc), strat_diff(loc, [attempts[a]["stratum"] for a in loc])
    h1_b, sd_b = [], []
    for _ in range(args.boot):
        ids = [a for k in rng.choice(len(keys), size=len(keys), replace=True) for a in by_pl[keys[k]]]
        h1_b.append(h1_stat(ids)); sd_b.append(strat_diff(ids, [attempts[a]["stratum"] for a in ids]))
    perm = []
    for _ in range(args.perm):
        sh = {}
        for k in keys:
            sh.update(zip(by_pl[k], rng.permutation([attempts[a]["stratum"] for a in by_pl[k]])))
        perm.append(strat_diff(loc, [sh[a] for a in loc]))
    q = lambda v, lo_, hi_: (float(np.nanquantile(v, lo_)), float(np.nanquantile(v, hi_)))
    n_succ = int(sum(x[0] for a in loc for x in samples[a]["local"]))
    n_tot = sum(len(samples[a]["local"]) for a in loc)
    rate_by = {s: float(np.mean([att_stats(a, "local")[0] for a in loc if attempts[a]["stratum"] == s])) for s in ("high", "low")}
    solved_new = sorted({attempts[a]["problem"] for a in loc if any(x[0] for x in samples[a]["local"])})
    report["step1"] = {"local_successes": n_succ, "local_samples": n_tot, "local_rate_by_stratum": rate_by,
                       "H1_local_minus_fresh_succ_per_Mtok": h1_0, "H1_ci80": q(h1_b, .1, .9), "H1_ci95": q(h1_b, .025, .975),
                       "high_minus_low_local": sd_0, "high_minus_low_ci95": q(sd_b, .025, .975),
                       "high_minus_low_perm_p_one_sided": float((np.sum(np.array(perm) >= sd_0) + 1) / (len(perm) + 1)),
                       "problems_with_a_local_repair": solved_new}
    s1 = report["step1"]
    print(f"\nSTEP 1 (LOCAL only): {n_succ} of {n_tot} LOCAL samples pass Lean, on {len(solved_new)} problems")
    print(f"  H1  LOCAL minus FRESH successes per million tokens: {h1_0:+.2f}, "
          f"80% CI [{s1['H1_ci80'][0]:+.2f}, {s1['H1_ci80'][1]:+.2f}], 95% CI [{s1['H1_ci95'][0]:+.2f}, {s1['H1_ci95'][1]:+.2f}]")
    print(f"  LOCAL success by entropy stratum: high {rate_by['high']:.3f}, low {rate_by['low']:.3f}; "
          f"within-problem high minus low {sd_0:+.3f}, 95% CI [{s1['high_minus_low_ci95'][0]:+.3f}, {s1['high_minus_low_ci95'][1]:+.3f}], "
          f"permutation p (one-sided) {s1['high_minus_low_perm_p_one_sided']:.4f}")
    print("  (descriptive: without BACK there is no control for how repairable each attempt is anyway)")
    if not rb:
        if s1["H1_ci80"][0] > 0:
            d1 = ("GO TO STEP 2: resampling from Lean's error line beats fresh attempts per token. "
                  "Add BACK (run.sh --step 2) to test whether entropy should choose the repair.")
        elif s1["H1_ci80"][1] < 0:
            d1 = ("STOP: resampling from the error line does worse than fresh attempts per token, so choosing a repair "
                  "point by entropy can't pay off here. Write up as a negative result; skip steps 2 and 3.")
        else:
            d1 = ("INCONCLUSIVE: LOCAL vs FRESH per token is within noise. Step 2 is cheap enough to run anyway, "
                  "or rerun step 1 with more samples first.")
        report["decision"] = d1
        print("\nDECISION: " + d1)

# --- policies --------------------------------------------------------------------------------
def policy_rows(ids):
    """per attempt: {policy: (success, tokens)}; D is the exact expectation."""
    p_high = np.mean([attempts[a]["stratum"] == "high" for a in ids])
    rows = []
    for a in ids:
        L, Bk, F = (att_stats(a, arm) for arm in ("local", "back", "fresh"))
        E = att_stats(a, "feedback") if HAS_FB else (np.nan, np.nan, np.nan)
        high = attempts[a]["stratum"] == "high"
        rows.append({"problem": attempts[a]["problem"],
                     "A": F[:2], "B": L[:2], "C-back": (L if high else Bk)[:2], "C-fresh": (L if high else F)[:2],
                     "C-feedback": (L if high else E)[:2],
                     "D": (p_high * L[0] + (1 - p_high) * Bk[0], p_high * L[1] + (1 - p_high) * Bk[1]), "E": E[:2]})
    return rows


HAS_FB = bool(full)           # step 3 done: FEEDBACK policies can be evaluated
POL_IDS = full if HAS_FB else rb
POL = ["A", "B", "C-back", "C-fresh", "C-feedback", "D", "E"] if HAS_FB else ["A", "B", "C-back", "C-fresh", "D"]


def pol_metrics(rows):
    out = {}
    for p in POL:
        v = np.array([r[p] for r in rows])
        out[p] = {"success": v[:, 0].mean(), "tokens": v[:, 1].mean(), "succ_per_Mtok": v[:, 0].sum() / v[:, 1].sum() * 1e6}
    return out


def boot_by_problem(rows, stat):
    by_p = collections.defaultdict(list)
    for r in rows:
        by_p[r["problem"]].append(r)
    keys = list(by_p)
    vals = []
    for _ in range(args.boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        vals.append(stat([r for k in pick for r in by_p[keys[k]]]))
    return np.array(vals)


if POL_IDS:
    rows = policy_rows(POL_IDS)
    pm = pol_metrics(rows)
    report["policies"] = pm
    print(f"\npolicies on the {len(POL_IDS)} attempts with {'every arm' if HAS_FB else 'LOCAL and BACK (no FEEDBACK yet)'}:")
    print(f"  {'policy':10} {'success':>8} {'tokens':>8} {'succ/Mtok':>9}")
    for p in POL:
        print(f"  {p:10} {pm[p]['success']:8.3f} {pm[p]['tokens']:8.0f} {pm[p]['succ_per_Mtok']:9.2f}")
    comps = [("C-back", "B"), ("C-back", "D"), ("C-fresh", "B"), ("B", "A")]
    if HAS_FB:
        comps += [("C-back", "E"), ("C-feedback", "B"), ("C-feedback", "E"), ("E", "A")]
    def diffs(rs):
        m = pol_metrics(rs)
        return {f"{x}-{y}": (m[x]["succ_per_Mtok"] - m[y]["succ_per_Mtok"], m[x]["success"] - m[y]["success"]) for x, y in comps}

    boots = boot_by_problem(rows, diffs)
    report["policy_diffs"] = {}
    print("\n  difference              succ/Mtok [80% CI] [95% CI]                 success rate [95% CI]")
    for x, y in comps:
        k = f"{x}-{y}"
        d = np.array([b[k] for b in boots])
        pt = (pm[x]["succ_per_Mtok"] - pm[y]["succ_per_Mtok"], pm[x]["success"] - pm[y]["success"])
        q = lambda col, lo, hi: (float(np.quantile(d[:, col], lo)), float(np.quantile(d[:, col], hi)))
        report["policy_diffs"][k] = {"succ_per_Mtok": pt[0], "ci80": q(0, .1, .9), "ci95": q(0, .025, .975),
                                     "success": pt[1], "success_ci95": q(1, .025, .975)}
        print(f"  {k:12} {pt[0]:+9.2f} [{q(0, .1, .9)[0]:+.2f}, {q(0, .1, .9)[1]:+.2f}] [{q(0, .025, .975)[0]:+.2f}, {q(0, .025, .975)[1]:+.2f}]"
              f"   {pt[1]:+.3f} [{q(1, .025, .975)[0]:+.3f}, {q(1, .025, .975)[1]:+.3f}]")


# --- H2: difference in differences -----------------------------------------------------------
_gap = {}


def did(ids, strata):
    """Mean over problems of (LOCAL-BACK | high) - (LOCAL-BACK | low); problems lacking a stratum are skipped."""
    by_p = collections.defaultdict(lambda: {"high": [], "low": []})
    for a, s in zip(ids, strata):
        if a not in _gap:
            _gap[a] = att_stats(a, "local")[0] - att_stats(a, "back")[0]
        by_p[attempts[a]["problem"]][s].append(_gap[a])
    per = [np.mean(v["high"]) - np.mean(v["low"]) for v in by_p.values() if v["high"] and v["low"]]
    return float(np.mean(per)) if per else float("nan"), len(per)


if rb:
    strata = [attempts[a]["stratum"] for a in rb]
    d0, n_p = did(rb, strata)
    gap = {s: float(np.mean([att_stats(a, "local")[0] - att_stats(a, "back")[0] for a in rb if attempts[a]["stratum"] == s]))
           for s in ("high", "low")}
    # problem bootstrap
    by_p = collections.defaultdict(list)
    for a in rb:
        by_p[attempts[a]["problem"]].append(a)
    keys = list(by_p)
    bs = []
    for _ in range(args.boot):
        ids = [a for k in rng.choice(len(keys), size=len(keys), replace=True) for a in by_p[keys[k]]]
        bs.append(did(ids, [attempts[a]["stratum"] for a in ids])[0])
    bs = np.array(bs)
    # permutation: shuffle stratum labels within each problem
    perm = []
    for _ in range(args.perm):
        sh = {}
        for k in keys:
            labs = [attempts[a]["stratum"] for a in by_p[k]]
            sh.update(zip(by_p[k], rng.permutation(labs)))
        perm.append(did(rb, [sh[a] for a in rb])[0])
    p_one = float((np.sum(np.array(perm) >= d0) + 1) / (len(perm) + 1))
    report["H2"] = {"did": d0, "problems": n_p, "local_minus_back": gap, "ci80": (float(np.quantile(bs, .1)), float(np.quantile(bs, .9))),
                    "ci95": (float(np.quantile(bs, .025)), float(np.quantile(bs, .975))), "perm_p_one_sided": p_one}
    print(f"\nH2: LOCAL minus BACK success rate: high entropy {gap['high']:+.3f}, low entropy {gap['low']:+.3f}")
    print(f"    difference in differences (mean over {n_p} problems) {d0:+.3f}, "
          f"95% CI [{report['H2']['ci95'][0]:+.3f}, {report['H2']['ci95'][1]:+.3f}], within-problem permutation p (one-sided) = {p_one:.4f}")

# --- secondary -------------------------------------------------------------------------------
if rb:
    report["by_kind"], report["by_split"] = {}, {}
    print("\nby error kind (LOCAL and BACK success rate, n attempts):")
    for k, n in collections.Counter(attempts[a]["kind"] for a in rb).most_common():
        ids = [a for a in rb if attempts[a]["kind"] == k]
        v = {arm: float(np.mean([att_stats(a, arm)[0] for a in ids])) for arm in ("local", "back")}
        report["by_kind"][k] = {**v, "n": n}
        print(f"  {k:20} n={n:4}  LOCAL {v['local']:.3f}  BACK {v['back']:.3f}")
    for sp in ("train", "test"):
        ids = [a for a in rb if attempts[a]["split"] == sp]
        if ids:
            report["by_split"][sp] = {"n": len(ids), "did": did(ids, [attempts[a]["stratum"] for a in ids])[0]}
    print("H2 by split: " + ", ".join(f"{k} DiD {v['did']:+.3f} (n={v['n']})" for k, v in report["by_split"].items()))
    report["pass_at_k"] = {}
    for arm in ARMS:
        ids = [a for a in attempts if len(samples[a][arm]) == n_samp]
        if ids:
            report["pass_at_k"][arm] = [float(np.mean([any(x[0] for x in samples[a][arm][:k]) for a in ids])) for k in range(1, n_samp + 1)]
    print("pass@k per attempt: " + "; ".join(f"{arm} " + " ".join(f"{v:.3f}" for v in vs) for arm, vs in report["pass_at_k"].items()))
    solved_by = collections.defaultdict(set)
    for a in attempts:
        for arm in ARMS:
            if any(x[0] for x in samples[a][arm]):
                solved_by[arm].add(attempts[a]["problem"])
    report["problems_with_a_repair"] = {arm: sorted(v) for arm, v in solved_by.items()}

# --- decision ----------------------------------------------------------------------------------
if rb:
    pd_ = report["policy_diffs"]
    h1 = pd_["B-A"]["succ_per_Mtok"] > 0
    lo, hi = report["H2"]["ci95"]
    interaction = report["H2"]["did"] > 0 and lo > 0 and report["H2"]["perm_p_one_sided"] < 0.05
    # A gate can only beat always-LOCAL if the low-entropy alternative beats LOCAL there (a crossover).
    winners = [c for c in ("C-back", "C-feedback", "C-fresh") if f"{c}-B" in pd_ and pd_[f"{c}-B"]["ci80"][0] > 0]
    go = interaction and "C-back" in winners and pd_["C-back-D"]["ci80"][0] > 0
    nxt = "Proceed to Stage 2." if HAS_FB else "Run step 3 (FEEDBACK, 4 samples) to confirm and to compare with the strong baseline."
    if go:
        verdict = f"GO: entropy gating beats both always-LOCAL and its random control (80% CI above 0). {nxt}"
    elif interaction and winners:
        verdict = (f"GO (other gate): the entropy interaction holds and {', '.join(winners)} beats always-LOCAL, but C-back does not. {nxt}")
    elif interaction:
        verdict = ("INTERACTION WITHOUT CROSSOVER: LOCAL repair works better on high-entropy lines, but LOCAL is still the best "
                   "choice on low-entropy lines, so a gate doesn't change the decision. Entropy predicts repairability; it "
                   "doesn't choose the repair. Report it that way"
                   + ("; Stage 2 only if the predictive use (e.g. ranking which failed attempts to repair first) is worth testing."
                      if HAS_FB else "; step 3 only to test whether FEEDBACK is the better alternative on low-entropy lines (C-feedback)."))
    elif lo < 0 < hi and abs(report["H2"]["did"]) < 0.02:
        verdict = ("STOP: LOCAL vs BACK does not depend on the entropy stratum. Write up as a negative result"
                   + ("." if HAS_FB else "; skip step 3."))
    else:
        verdict = "INCONCLUSIVE: neither the go nor the stop rule is met. Look at the CIs before deciding."
    if HAS_FB and all(pm["E"]["succ_per_Mtok"] > pm[p]["succ_per_Mtok"] for p in POL if p != "E"):
        verdict += " ALSO: FEEDBACK beats every resampling policy per token; rethink toward gating when to ask for feedback."
    if not h1:
        verdict += " NOTE: LOCAL does not beat FRESH per token (H1 fails), so the gating question is moot in this setting."
    report["decision"] = verdict
    print("\nDECISION: " + verdict)

json.dump(report, open(os.path.join(args.out, "analysis.json"), "w"), indent=1, default=float)
