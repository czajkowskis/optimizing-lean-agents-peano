"""Choose the failed attempts for the repair pilot and write everything the GPU side needs.

Runs where the original whole-proof results live (the laptop, or Helios scratch). Follows
helios/repair/DESIGN.md:
  * the same error-line mapping as results/wp85_all/scripts/error_lines85.py (Lean's first
    error -> a line of the model's last ```lean4 block -> its tokens);
  * the gate is the error line's max token entropy, z-scored against the accepted proof lines
    before it; the threshold is the median over ALL eligible failed attempts (all problems),
    fixed here, before any generation;
  * Stage 1 problems: solved by 1..65% of the 256 baseline attempts; per problem up to N attempts
    above and N below the threshold, seeded.

Per selected attempt it writes the cut points (token indices into output_ids) for LOCAL (start of
the error line) and BACK (start of the line 3 proof lines earlier), and the FEEDBACK prompt text,
built exactly like the prover's correction round (gp_seed_correction, history_mode=last_one,
with_cot, error_thres=True), rendered with the model's chat template.

Usage:
  python helios/repair/choose.py --repo . --tokenizer Goedel-LM/Goedel-Prover-V2-8B --out helios/repair/data
"""
import argparse
import ast
import collections
import difflib
import json
import os
import re
import shutil
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=".")
ap.add_argument("--tokenizer", default="Goedel-LM/Goedel-Prover-V2-8B")
ap.add_argument("--out", default=os.path.join(HERE, "data"))
ap.add_argument("--per-stratum", type=int, default=20, help="attempts per problem per gate stratum")
ap.add_argument("--max-rate", type=float, default=0.65, help="exclude problems the baseline solves more often")
ap.add_argument("--back-lines", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--check-per-problem", type=int, default=2, help="original attempts per verdict for the compile check")
args = ap.parse_args()

REPO = os.path.abspath(args.repo)


def load_function(path, name):
    """Take one function's source from a file and exec it, without importing the file's module
    (prover/utils.py pulls in pandas, jload, ...). Keeps the prover's exact implementation."""
    src = open(path, encoding="utf-8").read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {}
    exec(compile(ast.Module([node], []), path, "exec"), ns)
    return ns[name]


get_error_str = load_function(os.path.join(REPO, "prover", "utils.py"), "get_error_str")
from transformers import AutoTokenizer  # noqa: E402

MAX_LEN = 40960                      # model context = max_tokens of the original run
FEEDBACK_MAX_PROMPT = MAX_LEN * 3 // 4   # the prover drops prompts longer than this
RUNS = [("old20", "results/helios_wp/results/putnam20_whole_proof_8b", "results/helios_wp/token_ids/putnam20_whole_proof_8b")] + \
       [("new65", f"results/helios_wp85/results/putnam85_rest_wp_8b_s{i}", "results/helios_wp85/token_ids/putnam85_rest_wp_8b") for i in range(5)]
CORRECTION_TEMPLATE = open(os.path.join(REPO, "prover/template/prover/gp_seed_correction.md"), encoding="utf-8").read()

split_txt = open(os.path.join(REPO, "dataset/putnam_rewrite_solved_train_test_split.txt"), encoding="utf-8").read()
TRAIN = set(re.findall(r"putnam_\w+", split_txt.split("Test")[0]))
stmt_lines = {}
for l in open(os.path.join(REPO, "dataset", "putnam_rewrite_solved.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    stmt_lines[r["problem_id"]] = r["formal_statement"].rstrip("\n").count("\n") + 1

tok = AutoTokenizer.from_pretrained(args.tokenizer)
PIECES = np.array([tok.decode([i]) for i in range(len(tok))], dtype=object)   # token id -> its own text
asc = lambda s: re.sub(r"[^\x21-\x7e]", "", s)    # visible ASCII only (as in error_lines85.py)


def as_list(x):
    return x if isinstance(x, list) else ast.literal_eval(x)


def kind(msg):
    m = msg.split("\n")[0]
    for k, p in [("unknown identifier", r"unknown (identifier|constant)"), ("unsolved goals", r"unsolved goals"),
                 ("linarith failed", r"linarith failed"), ("type mismatch", r"type mismatch"),
                 ("rewrite failed", r"rewrite' failed|motive is not type correct"), ("omega failed", r"omega"),
                 ("simp failed", r"simp made no progress|simp failed")]:
        if re.search(p, m):
            return k
    return "other"


skip = collections.Counter()
eligible = []          # failed attempts usable for the pilot
originals = collections.defaultdict(lambda: {"complete": [], "failed": []})   # for the compile check
attempts_per_problem = collections.Counter()
solved_per_problem = collections.Counter()
fresh_tokens = collections.defaultdict(list)   # output tokens of every original attempt (FRESH arm cost)

for run_i, (group, R, IDS) in enumerate(RUNS):
    inf = json.load(open(os.path.join(REPO, R, "to_inference_codes.json"), encoding="utf-8"))
    comp = {c["name"]: c for c in json.load(open(os.path.join(REPO, R, "code_compilation_repl.json"), encoding="utf-8"))}
    for rec_i, rec in enumerate(inf):
        pid = rec["origin_problem_id"]
        c = comp.get(rec["problem_id"])
        attempts_per_problem[pid] += 1
        fresh_tokens[pid].append(int(rec["detailed_cost"]["output_tokens"]))
        if not c:
            skip["no compile result"] += 1; continue
        cr = c["compilation_result"]
        complete = bool(cr.get("complete"))
        solved_per_problem[pid] += complete
        ids_name = os.path.basename(rec["detailed_cost"]["token_ids_file"])
        f = os.path.join(REPO, IDS, ids_name)
        orig = {"attempt": ids_name[:-4], "problem": pid, "code": c["code"], "complete": complete,
                "first_error_line": min((e["pos"]["line"] for e in (cr.get("errors") or []) if isinstance(e, dict) and e.get("pos")), default=None)}
        originals[pid]["complete" if complete else "failed"].append(orig)
        if complete:
            continue
        z = np.load(f)
        prompt_ids, output_ids = z["prompt_ids"], z["output_ids"]
        if len(prompt_ids) + len(output_ids) >= MAX_LEN - 8:
            skip["hit the token limit"] += 1; continue
        st = np.load(f[:-4] + ".stats.npz")
        pieces = PIECES[np.asarray(output_ids)].tolist()
        if len(pieces) != len(st["entropy"]):
            skip["length mismatch"] += 1; continue
        text = "".join(pieces)
        blocks = list(re.finditer(r"```lean4?\n(.*?)```", text, re.S))
        if not blocks:
            skip["no lean block"] += 1; continue
        b0, b1 = blocks[-1].start(1), blocks[-1].end(1)
        block_lines = blocks[-1].group(1).split("\n")
        starts = np.cumsum([0] + [len(x) + 1 for x in block_lines])[:-1] + b0   # char where each block line starts
        lens = np.array([len(p) for p in pieces])
        tok_ends = np.cumsum(lens) - 1
        tok_starts = tok_ends - lens + 1
        tok_line = np.searchsorted(starts, tok_ends, side="right") - 1
        in_block = (tok_ends >= b0) & (tok_ends < b1)
        H = st["entropy"].astype(np.float32)
        code_lines = c["code"].split("\n")
        k0 = stmt_lines.get(pid, 0)
        sm = difflib.SequenceMatcher(None, [asc(x) for x in code_lines], [asc(x) for x in block_lines], autojunk=False)
        code2block = {}
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                code2block.update({i1 + t: j1 + t for t in range(i2 - i1)})
        body = [code2block[i] for i in range(k0, len(code_lines)) if i in code2block and asc(code_lines[i])]
        lines = []
        for j in body:
            m = in_block & (tok_line == j)
            if m.any():
                lines.append({"line": j, "maxH": float(H[m].max())})
        if not lines:
            skip["no matched proof lines"] += 1; continue
        errs = [e for e in (cr.get("errors") or []) if isinstance(e, dict) and (e.get("pos") or {}).get("line")]
        if not errs:
            skip["failed, no positioned error"] += 1; continue
        e = min(errs, key=lambda e: (e["pos"]["line"], e["pos"].get("column", 0)))
        ci = e["pos"]["line"] - 1
        if ci < k0:
            skip["error in statement lines"] += 1; continue
        if ci not in code2block:
            skip["error line not in model block"] += 1; continue
        idx = [i for i, x in enumerate(lines) if x["line"] == code2block[ci]]
        if not idx:
            skip["error line has no tokens"] += 1; continue
        err = idx[0]
        if err < args.back_lines:
            skip[f"error within the first {args.back_lines} proof lines"] += 1; continue
        prev = np.array([x["maxH"] for x in lines[:err]])
        if prev.std() == 0:
            skip["no entropy variation before the error"] += 1; continue

        def cut_at(block_line):
            """Token index where block_line starts; None if no token starts exactly there."""
            hit = np.flatnonzero(tok_starts == starts[block_line])
            return int(hit[0]) if len(hit) else None

        local_line, back_line = lines[err]["line"], lines[err - args.back_lines]["line"]
        cut_local, cut_back = cut_at(local_line), cut_at(back_line)
        if cut_local is None or cut_back is None:
            skip["cut not on a token boundary"] += 1; continue

        eligible.append({
            "attempt": ids_name[:-4], "problem": pid, "group": group, "split": "train" if pid in TRAIN else "test",
            "ids_file": os.path.relpath(f, REPO).replace(os.sep, "/"), "_src": (run_i, rec_i),
            "lean4_code": rec["lean4_code"], "orig_code": c["code"],
            "kind": kind(e.get("data", "")), "error_msg": e.get("data", "")[:300],
            "n_proof_lines": len(lines), "err_index": err,
            "gate_z": float((lines[err]["maxH"] - prev.mean()) / prev.std()),
            "prompt_len": int(len(prompt_ids)), "output_len": int(len(output_ids)),
            "cut_local": cut_local, "cut_back": cut_back,
        })

print(f"{len(eligible)} eligible failed attempts; skipped: {dict(skip)}")
z_all = np.array([a["gate_z"] for a in eligible])
thr = float(np.median(z_all))
thr_train = float(np.median([a["gate_z"] for a in eligible if a["split"] == "train"]))
print(f"gate threshold (median z over all eligible attempts): {thr:.4f}   (train-only median: {thr_train:.4f})")

rate = {p: solved_per_problem[p] / attempts_per_problem[p] for p in attempts_per_problem}
problems = sorted(p for p, r in rate.items() if 0 < r <= args.max_rate)
print(f"{len(problems)} Stage-1 problems (solved by >0 and <= {args.max_rate:.0%} of attempts)")

rng = np.random.default_rng(args.seed)
selected, per_problem = [], {}
for p in problems:
    pool = [a for a in eligible if a["problem"] == p]
    hi = [a for a in pool if a["gate_z"] > thr]
    lo = [a for a in pool if a["gate_z"] <= thr]
    take = []
    for name, s in (("high", hi), ("low", lo)):
        pick = rng.choice(len(s), size=min(args.per_stratum, len(s)), replace=False) if s else []
        for i in sorted(pick):
            take.append({**s[i], "stratum": name})
    per_problem[p] = {"rate": rate[p], "attempts": attempts_per_problem[p], "successes": solved_per_problem[p],
                      "fresh_mean_tokens": float(np.mean(fresh_tokens[p])), "split": "train" if p in TRAIN else "test",
                      "eligible": len(pool), "high": len(hi), "low": len(lo),
                      "selected_high": sum(a["stratum"] == "high" for a in take),
                      "selected_low": sum(a["stratum"] == "low" for a in take)}
    selected += take

# FEEDBACK prompt, only for the selected attempts, exactly as the prover's correction round builds it.
by_run = collections.defaultdict(list)
for a in selected:
    by_run[a["_src"][0]].append(a)
for run_i, items in by_run.items():
    _, R, _ = RUNS[run_i]
    inf = json.load(open(os.path.join(REPO, R, "to_inference_codes.json"), encoding="utf-8"))
    comp = {c["name"]: c for c in json.load(open(os.path.join(REPO, R, "code_compilation_repl.json"), encoding="utf-8"))}
    for a in items:
        rec = inf[a["_src"][1]]
        c = comp[rec["problem_id"]]
        history = as_list(rec["messages_history_list"])
        err_str = get_error_str(c["code"], c["compilation_result"].get("errors", []), True)
        messages = [history[-1], {"role": "assistant", "content": rec["model_output"]},
                    {"role": "user", "content": CORRECTION_TEMPLATE.format(prev_round_num=0, error_message_for_prev_round=err_str,
                                                                          informal_prefix="")}]
        a["feedback_prompt"] = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        a["feedback_prompt_len"] = len(tok(a["feedback_prompt"], add_special_tokens=False)["input_ids"])
        a["feedback_ok"] = a["feedback_prompt_len"] <= FEEDBACK_MAX_PROMPT
    del inf, comp
for a in selected:
    a.pop("_src")

os.makedirs(os.path.join(args.out, "ids"), exist_ok=True)
for a in selected:
    shutil.copyfile(os.path.join(REPO, a["ids_file"]), os.path.join(args.out, "ids", a["attempt"] + ".npz"))
    a["ids_file"] = f"ids/{a['attempt']}.npz"
with open(os.path.join(args.out, "pilot_attempts.jsonl"), "w", encoding="utf-8") as fh:
    for a in selected:
        fh.write(json.dumps(a, ensure_ascii=False) + "\n")

# Compile check: unchanged original attempts of the Stage-1 problems, with their Helios verdicts.
check = []
for p in problems:
    for verdict in ("complete", "failed"):
        pool = originals[p][verdict]
        for i in rng.choice(len(pool), size=min(args.check_per_problem, len(pool)), replace=False):
            check.append(pool[i])
with open(os.path.join(args.out, "compile_check.jsonl"), "w", encoding="utf-8") as fh:
    for o in check:
        fh.write(json.dumps(o, ensure_ascii=False) + "\n")

meta = {"threshold": thr, "threshold_train_only": thr_train, "n_eligible": len(eligible), "skipped": dict(skip),
        "n_selected": len(selected), "problems": per_problem, "args": vars(args),
        "feedback_too_long": sum(not a["feedback_ok"] for a in selected),
        "kinds": dict(collections.Counter(a["kind"] for a in selected)),
        "eligible_gate_z_quantiles": {q: float(np.quantile(z_all, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}}
json.dump(meta, open(os.path.join(args.out, "selection.json"), "w"), indent=1)
print(f"selected {len(selected)} attempts "
      f"({sum(a['stratum'] == 'high' for a in selected)} high / {sum(a['stratum'] == 'low' for a in selected)} low), "
      f"{meta['feedback_too_long']} with a FEEDBACK prompt over {FEEDBACK_MAX_PROMPT} tokens; "
      f"{len(check)} originals for the compile check")
for p, v in per_problem.items():
    print(f"  {p:18} rate {v['rate']:.3f}  eligible {v['eligible']:4}  high {v['high']:4}  low {v['low']:4}  "
          f"-> {v['selected_high']}+{v['selected_low']}")
