"""Generate repair samples for the pilot (LOCAL, BACK, FEEDBACK arms) with vLLM, on one GPU.

Sampling matches the original whole-proof run: Goedel-Prover-V2-8B, bf16, temperature 1.0,
top_p 0.95, a 40 960-token context. LOCAL and BACK continue the model's own output from a cut
point (stored token ids, no re-tokenization) and get at most --max-new-repair new tokens. FEEDBACK
is a new attempt from the correction prompt and may use the rest of the context, like the original.

Every (attempt, arm, sample) has a fixed seed. Results are appended to <out>/gen_<shard>.jsonl as
they finish; a rerun skips what is already there.

Usage (one process per GPU):
  python helios/repair/generate.py --data helios/repair/data --out amu_work/repair --shard 0 --nshards 4
  python helios/repair/generate.py ... --limit-attempts 20 --samples 1      # smoke test
"""
import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Goedel-LM/Goedel-Prover-V2-8B")
ap.add_argument("--arms", default="local,back,feedback")
ap.add_argument("--samples", type=int, default=4)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit-attempts", type=int, default=0, help="only the first N attempts (smoke test); spread over problems")
ap.add_argument("--max-new-repair", type=int, default=8192)
ap.add_argument("--max-len", type=int, default=40960)
ap.add_argument("--chunk", type=int, default=512, help="requests per vLLM call (progress is saved after each)")
ap.add_argument("--gpu-mem", type=float, default=0.90)
args = ap.parse_args()

ARMS = args.arms.split(",")
attempts = [json.loads(l) for l in open(os.path.join(args.data, "pilot_attempts.jsonl"), encoding="utf-8")]
if args.limit_attempts:
    # round-robin over problems so a smoke test touches several problems and both strata
    by_p = {}
    for a in attempts:
        by_p.setdefault(a["problem"], []).append(a)
    order = []
    while len(order) < min(args.limit_attempts, len(attempts)):
        for p in sorted(by_p):
            if by_p[p]:
                order.append(by_p[p].pop(0))
    attempts = order[: args.limit_attempts]
attempts = attempts[args.shard :: args.nshards]

os.makedirs(args.out, exist_ok=True)
out_path = os.path.join(args.out, f"gen_{args.shard}.jsonl")
done = set()
# Read every shard's file, not just ours: earlier steps may have run with a different number of
# shards (e.g. steps 1-2 on 1 GPU, step 3 on 4), so our attempts' samples can be in any of them.
for f in sorted(glob.glob(os.path.join(args.out, "gen_*.jsonl"))):
    for l in open(f, encoding="utf-8"):
        try:
            done.add(json.loads(l)["id"])
        except (json.JSONDecodeError, KeyError):
            pass   # a line cut off by a time limit (or still being written); that request is simply redone


def seed_of(rid):
    return int.from_bytes(hashlib.sha256(rid.encode()).digest()[:4], "little") & 0x7FFFFFFF


requests, skipped = [], {"feedback prompt too long": 0}
for a in attempts:
    z = np.load(os.path.join(args.data, a["ids_file"]))
    prompt_ids, output_ids = z["prompt_ids"].tolist(), z["output_ids"].tolist()
    for arm in ARMS:
        if arm == "feedback" and not a["feedback_ok"]:
            skipped["feedback prompt too long"] += 1
            continue
        for s in range(args.samples):
            rid = f"{a['attempt']}.{arm}.{s}"
            if rid in done:
                continue
            if arm in ("local", "back"):
                cut = a["cut_local"] if arm == "local" else a["cut_back"]
                prompt = {"prompt_token_ids": prompt_ids + output_ids[:cut]}
                plen = len(prompt["prompt_token_ids"])
                max_new = min(args.max_new_repair, args.max_len - plen)
            else:
                prompt = a["feedback_prompt"]
                plen = a["feedback_prompt_len"]
                max_new = args.max_len - plen
            requests.append({"id": rid, "attempt": a["attempt"], "problem": a["problem"], "arm": arm, "sample": s,
                             "seed": seed_of(rid), "prompt_len": plen, "max_new": max_new, "prompt": prompt})

# Short continuations first (samples of one attempt stay adjacent, so they share the cached
# prefix), then the long FEEDBACK attempts, so a chunk doesn't wait on a mix of both.
requests.sort(key=lambda r: r["arm"] == "feedback")
print(f"shard {args.shard}/{args.nshards}: {len(attempts)} attempts, {len(requests)} requests to run, "
      f"{len(done)} already done, skipped {skipped}", flush=True)
if not requests:
    print("GENERATE_OK (nothing to do)")
    raise SystemExit(0)

from vllm import LLM, SamplingParams  # noqa: E402  (import late: slow, and not needed when done)

llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_len, enable_prefix_caching=True,
          gpu_memory_utilization=args.gpu_mem, seed=0)
t0, n_tok, n_done = time.time(), 0, 0
with open(out_path, "a", encoding="utf-8") as fh:
    for i in range(0, len(requests), args.chunk):
        chunk = requests[i : i + args.chunk]
        params = [SamplingParams(temperature=1.0, top_p=0.95, max_tokens=r["max_new"], seed=r["seed"]) for r in chunk]
        outs = llm.generate([r["prompt"] for r in chunk], params, use_tqdm=False)
        for r, o in zip(chunk, outs):
            c = o.outputs[0]
            rec = {k: v for k, v in r.items() if k != "prompt"}
            rec.update({"new_tokens": len(c.token_ids), "finish_reason": c.finish_reason, "text": c.text})
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_tok += len(c.token_ids)
        fh.flush()
        n_done += len(chunk)
        dt = time.time() - t0
        print(f"[{n_done}/{len(requests)}] {n_tok / dt:.0f} new tok/s, {dt / 60:.1f} min", flush=True)
print("GENERATE_OK")
