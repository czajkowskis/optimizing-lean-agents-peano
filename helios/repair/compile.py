"""Compile the pilot's proofs with the prover's own Lean REPL scheduler (lean_compiler/repl_scheduler.py).

Two modes:
  check    Recompile unchanged original attempts (data/compile_check.jsonl) and compare with the
           verdicts from the original Helios run. Run this first: if this Lean setup disagrees
           with Helios, repair results aren't comparable with the baseline. Exit code 3 on failure.
  repairs  Build the code for every generated sample (gen_*.jsonl) exactly as the prover does
           (last ```lean4 block -> statement check -> import filter) and compile it.
           Results are appended to <out>/compiled.jsonl; a rerun skips what is already there.

Must run from the repository root on Linux, with Lean + the mathlib4 submodule built
(see helios/repair/setup_lean.sh). Needs numpy, pexpect, transformers (tokenizer only).

Usage:
  python helios/repair/compile.py check   --data helios/repair/data --out amu_work/repair --workers 32
  python helios/repair/compile.py repairs --data helios/repair/data --out amu_work/repair --workers 32
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
from prover_funcs import code_to_compile  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["check", "repairs"])
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=32)
ap.add_argument("--model", default="Goedel-LM/Goedel-Prover-V2-8B", help="for the tokenizer")
ap.add_argument("--chunk", type=int, default=1000, help="proofs per scheduler call (progress is saved after each)")
args = ap.parse_args()
os.chdir(REPO)   # the scheduler's Lean workspace path ("mathlib4/") is relative to the repo root
from lean_compiler.repl_scheduler import scheduler  # noqa: E402


def first_error_line(cr):
    errs = [e for e in (cr.get("errors") or []) if isinstance(e, dict) and e.get("pos")]
    return min((e["pos"]["line"] for e in errs), default=None)


def compile_all(items):
    """items: list of {"name", "code"} -> {name: compilation_result}"""
    res = scheduler(items, num_workers=min(args.workers, max(1, len(items))))
    return {r["name"]: r["compilation_result"] for r in res}


if args.mode == "check":
    orig = [json.loads(l) for l in open(os.path.join(args.data, "compile_check.jsonl"), encoding="utf-8")]
    got = compile_all([{"name": o["attempt"], "code": o["code"]} for o in orig])
    agree = lost = gained = sys_err = 0
    rows = []
    for o in orig:
        cr = got.get(o["attempt"], {"complete": False, "system_errors": "missing"})
        now = bool(cr.get("complete"))
        agree += now == o["complete"]
        lost += o["complete"] and not now
        gained += now and not o["complete"]
        sys_err += cr.get("system_errors") is not None
        rows.append({"attempt": o["attempt"], "problem": o["problem"], "helios_complete": o["complete"], "here_complete": now,
                     "helios_first_error_line": o["first_error_line"], "here_first_error_line": first_error_line(cr),
                     "system_errors": cr.get("system_errors")})
    os.makedirs(args.out, exist_ok=True)
    json.dump(rows, open(os.path.join(args.out, "compile_check.json"), "w"), indent=1)
    same_line = sum(r["helios_first_error_line"] == r["here_first_error_line"] for r in rows if not r["helios_complete"])
    n_failed = sum(not r["helios_complete"] for r in rows)
    print(f"compile check: {len(orig)} original attempts, verdict agrees on {agree}; "
          f"Helios-complete now failing: {lost}; Helios-failed now complete: {gained}; system errors: {sys_err}; "
          f"same first error line on {same_line}/{n_failed} failed ones")
    ok = lost == 0 and gained == 0 and sys_err == 0
    print("COMPILE_CHECK_OK" if ok else "COMPILE_CHECK_FAILED: this Lean setup does not reproduce the Helios verdicts")
    sys.exit(0 if ok else 3)

# --- repairs -------------------------------------------------------------------------------
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.model)
attempts = {json.loads(l)["attempt"]: json.loads(l) for l in open(os.path.join(args.data, "pilot_attempts.jsonl"), encoding="utf-8")}
out_path = os.path.join(args.out, "compiled.jsonl")
done = set()
if os.path.exists(out_path):
    for l in open(out_path, encoding="utf-8"):
        try:
            done.add(json.loads(l)["id"])
        except (json.JSONDecodeError, KeyError):
            pass

prefix_text = {}


def response_of(g):
    """The full assistant output the sample stands for: kept prefix + new text (repair arms),
    or just the new text (FEEDBACK, a fresh attempt from the correction prompt)."""
    if g["arm"] == "feedback":
        return g["text"]
    a = attempts[g["attempt"]]
    key = (g["attempt"], g["arm"])
    if key not in prefix_text:
        ids = np.load(os.path.join(args.data, a["ids_file"]))["output_ids"].tolist()
        cut = a["cut_local"] if g["arm"] == "local" else a["cut_back"]
        prefix_text[key] = tok.decode(ids[:cut], skip_special_tokens=True)
    return prefix_text[key] + g["text"]


todo, precomputed = [], []
for f in sorted(glob.glob(os.path.join(args.out, "gen_*.jsonl"))):
    for l in open(f, encoding="utf-8"):
        try:
            g = json.loads(l)
        except json.JSONDecodeError:
            continue
        if g["id"] in done:
            continue
        done.add(g["id"])
        code, why = code_to_compile(attempts[g["attempt"]]["lean4_code"], response_of(g))
        base = {k: g[k] for k in ("id", "attempt", "problem", "arm", "sample", "prompt_len", "new_tokens", "finish_reason")}
        if code is None:
            precomputed.append({**base, "complete": False, "not_compiled": why, "first_error_line": None, "first_error": None})
        else:
            todo.append((base, code))

print(f"{len(todo)} samples to compile, {len(precomputed)} failed before compiling (no Lean block / bad statement)", flush=True)
with open(out_path, "a", encoding="utf-8") as fh:
    for r in precomputed:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for i in range(0, len(todo), args.chunk):
        chunk = todo[i : i + args.chunk]
        got = compile_all([{"name": b["id"], "code": c} for b, c in chunk])
        for b, c in chunk:
            cr = got.get(b["id"], {"complete": False, "system_errors": "missing from scheduler output"})
            errs = [e for e in (cr.get("errors") or []) if isinstance(e, dict)]
            fe = min(errs, key=lambda e: ((e.get("pos") or {}).get("line", 1e9), (e.get("pos") or {}).get("column", 0)), default=None)
            fh.write(json.dumps({**b, "complete": bool(cr.get("complete")), "not_compiled": None,
                                 "system_errors": cr.get("system_errors"), "first_error_line": first_error_line(cr),
                                 "first_error": (fe or {}).get("data", "")[:300] or None, "code": c}, ensure_ascii=False) + "\n")
        fh.flush()
        print(f"[{min(i + args.chunk, len(todo))}/{len(todo)}] compiled", flush=True)
print("COMPILE_OK")
