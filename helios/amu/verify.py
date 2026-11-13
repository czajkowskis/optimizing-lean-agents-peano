#!/usr/bin/env python3
"""Checks for a re-scoring run done away from Helios.

  python helios/amu/verify.py check amu_data/check
      Compares the stats you computed for the 200 check attempts (amu_data/check/ids/*.stats.npz)
      with the ones computed on Helios (amu_data/check/reference/). Different GPUs give slightly
      different bf16 numbers; this says whether the difference is small enough to pool the runs.

  python helios/amu/verify.py done amu_data/token_ids/putnam85_rest_wp_8b
      Checks that every attempt has a complete, finite .stats.npz.

Needs only numpy.
"""
import glob
import os
import sys

import numpy as np

KEYS = ("entropy", "varentropy", "nll")


def ids_files(d):
    return sorted(f for f in glob.glob(os.path.join(d, "*.npz")) if not f.endswith(".stats.npz"))


def stats_of(ids_path):
    return ids_path[: -len(".npz")] + ".stats.npz"


def check(root):
    refs = sorted(glob.glob(os.path.join(root, "reference", "*.stats.npz")))
    if not refs:
        sys.exit(f"no reference files in {root}/reference")
    tok = {k: ([], []) for k in ("entropy", "nll")}
    per_attempt, missing, bad_len = [], [], []
    for r in refs:
        mine = os.path.join(root, "ids", os.path.basename(r))
        if not os.path.exists(mine):
            missing.append(os.path.basename(r)); continue
        a, b = np.load(mine), np.load(r)
        if len(a["nll"]) != len(b["nll"]):
            bad_len.append(os.path.basename(r)); continue
        if not len(b["nll"]):
            continue
        for k in tok:
            tok[k][0].append(a[k].astype(np.float32)); tok[k][1].append(b[k].astype(np.float32))
        per_attempt.append((a["entropy"].astype(np.float32).mean(), b["entropy"].astype(np.float32).mean(),
                            a["nll"].astype(np.float32).mean(), b["nll"].astype(np.float32).mean()))
    print(f"compared {len(per_attempt)} of {len(refs)} attempts; missing {len(missing)}, length mismatch {len(bad_len)}")
    if missing:
        print("  missing (not re-scored yet?):", missing[:5], "..." if len(missing) > 5 else "")
    if bad_len:
        print("  LENGTH MISMATCH (wrong files?):", bad_len[:5])
    if not per_attempt:
        sys.exit(1)
    ok = not bad_len
    for k, (x, y) in tok.items():
        x, y = np.concatenate(x), np.concatenate(y)
        d = np.abs(x - y)
        r = np.corrcoef(x, y)[0, 1]
        print(f"token-level {k:8}: {len(x):,} tokens, Pearson r {r:.5f}, median |diff| {np.median(d):.4f}, "
              f"p99 |diff| {np.quantile(d, .99):.4f}, max {d.max():.3f}")
        ok &= r > 0.99
    pa = np.array(per_attempt)
    dh, dn = np.abs(pa[:, 0] - pa[:, 1]), np.abs(pa[:, 2] - pa[:, 3])
    print(f"per-attempt mean entropy: median |diff| {np.median(dh):.5f}, max {dh.max():.5f} nats "
          f"(typical attempt mean is ~{np.median(pa[:, 1]):.2f})")
    print(f"per-attempt mean NLL:     median |diff| {np.median(dn):.5f}, max {dn.max():.5f} nats")
    ok &= np.median(dh) < 0.01
    print("\nVERDICT:", "looks consistent with Helios" if ok else "DIFFERENT from Helios - send this output to Szymon before going on")
    print("(Please send this whole output back either way.)")
    if not ok:
        sys.exit(1)


def done(d):
    files = ids_files(d)
    missing, empty, bad = [], 0, []
    for f in files:
        s = stats_of(f)
        if not os.path.exists(s):
            missing.append(os.path.basename(f)); continue
        n_out = len(np.load(f)["output_ids"])
        try:
            st = np.load(s)
            lens = {len(st[k]) for k in KEYS}
            finite = all(np.isfinite(st[k].astype(np.float32)).all() for k in KEYS)
        except Exception as e:  # truncated or corrupt file
            bad.append(f"{os.path.basename(s)}: {e}"); continue
        if lens != {n_out} or not finite:
            bad.append(f"{os.path.basename(s)}: lengths {lens} vs {n_out} output tokens, finite={finite}")
        empty += n_out == 0
    print(f"{len(files)} attempts, {len(files) - len(missing)} with stats, {len(missing)} missing, {len(bad)} bad, {empty} empty outputs")
    for m in missing[:10]:
        print("  missing:", m)
    for b in bad[:10]:
        print("  bad:", b)
    if missing or bad:
        print("\nNOT DONE: resubmit the job (finished files are skipped). Delete 'bad' .stats.npz files first.")
        sys.exit(1)
    print("\nALL DONE")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("check", "done"):
        sys.exit(__doc__)
    (check if sys.argv[1] == "check" else done)(sys.argv[2])
