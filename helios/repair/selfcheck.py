"""Offline checks of the pilot's data and code paths, run where the original results live (no GPU, no Lean).

  1. decoding an attempt's stored output_ids gives the model_output the prover saw
  2. prover_funcs.code_to_compile(lean4_code, model_output) rebuilds exactly the code the
     original run compiled (so repaired proofs go through the same extraction)
  3. for every selected attempt and both cut points, decode(prefix) + decode(rest) == decode(all),
     and the prefix ends with a newline (the cut is on a line boundary)
  4. the prefix up to the LOCAL cut ends just before the error line's text

Usage: python helios/repair/selfcheck.py --repo . --data helios/repair/data [--n 300]
"""
import argparse
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prover_funcs import code_to_compile  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=".")
ap.add_argument("--data", default=os.path.join(HERE, "data"))
ap.add_argument("--tokenizer", default="Goedel-LM/Goedel-Prover-V2-8B")
ap.add_argument("--n", type=int, default=300, help="original attempts to check for 1 and 2")
args = ap.parse_args()
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.tokenizer)
fails = []

# 1 + 2 on a random sample of original records from one run
R = os.path.join(args.repo, "results/helios_wp85/results/putnam85_rest_wp_8b_s0")
IDS = os.path.join(args.repo, "results/helios_wp85/token_ids/putnam85_rest_wp_8b")
inf = json.load(open(os.path.join(R, "to_inference_codes.json"), encoding="utf-8"))
comp = {c["name"]: c for c in json.load(open(os.path.join(R, "code_compilation_repl.json"), encoding="utf-8"))}
random.seed(0)
dec_ok = dec_trunc = code_ok = code_err = n2 = 0
for rec in random.sample(inf, min(args.n, len(inf))):
    z = np.load(os.path.join(IDS, os.path.basename(rec["detailed_cost"]["token_ids_file"])))
    ids = z["output_ids"].tolist()
    d = tok.decode(ids, skip_special_tokens=True)
    if d == rec["model_output"]:
        dec_ok += 1
    elif len(z["prompt_ids"]) + len(ids) >= 40960 - 8 and d.rstrip("�") == rec["model_output"]:
        dec_trunc += 1   # cut off at the token limit mid-character; such attempts are excluded from the pilot
    c = comp.get(rec["problem_id"])
    if c is None:
        continue
    code, why = code_to_compile(rec["lean4_code"], rec["model_output"])
    n2 += 1
    if code == c["code"] or (code is None and why == "no lean block" and not c["code"].strip()):
        code_ok += 1
    elif code is None and why.startswith("**Error**") and not c["compilation_result"].get("complete"):
        code_err += 1    # the prover compiled its "**Error** ..." string, which fails; we mark it failed directly
n1 = min(args.n, len(inf))
print(f"1. decode(output_ids) == model_output: {dec_ok}/{n1}, plus {dec_trunc} truncated mid-character at the token limit")
print(f"2. rebuilt compile code == original:   {code_ok}/{n2}, plus {code_err} statement-check errors failed either way")
ok12 = dec_ok + dec_trunc == n1 and code_ok + code_err == n2

# 3 + 4 on every selected attempt
att = [json.loads(l) for l in open(os.path.join(args.data, "pilot_attempts.jsonl"), encoding="utf-8")]
ok3 = ok4 = 0
for a in att:
    ids = np.load(os.path.join(args.data, a["ids_file"]))["output_ids"].tolist()
    whole = tok.decode(ids, skip_special_tokens=True)
    good = True
    for cut in (a["cut_local"], a["cut_back"]):
        pre, rest = tok.decode(ids[:cut], skip_special_tokens=True), tok.decode(ids[cut:], skip_special_tokens=True)
        good &= (pre + rest == whole) and pre.endswith("\n")
    ok3 += good
    if not good:
        fails.append(a["attempt"])
    # the error line is the first line after the LOCAL prefix; its code text should appear in the original compiled code
    nxt = tok.decode(ids[a["cut_local"]:], skip_special_tokens=True).split("\n", 1)[0]
    ok4 += nxt.strip() != "" and nxt.strip() in a["orig_code"]
print(f"3. clean cuts on line boundaries:      {ok3}/{len(att)}")
print(f"4. LOCAL cut is followed by a code line from the compiled proof: {ok4}/{len(att)}")
if fails:
    print("   bad cuts:", fails[:10])
print("SELFCHECK_OK" if ok12 and ok3 == len(att) and ok4 == len(att) else "SELFCHECK: see counts above")
