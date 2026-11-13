"""Exact per-token entropy for saved prover attempts, by re-scoring.

Generation (prover/query/api.py with TOKEN_IDS_DIR set) saves one `<uuid>.npz` per
attempt holding `prompt_ids` and `output_ids`. This script runs one forward pass of
the same model over prompt+output and writes `<uuid>.stats.npz` next to it with, for
every output token t (all from the model's raw distribution, i.e. temperature 1):

    entropy[t]     H = -sum_v p_v log p_v               (nats, full vocabulary)
    varentropy[t]  sum_v p_v (log p_v)^2 - H^2
    nll[t]         -log p(sampled token)

At temperature 1, mean(nll) estimates mean(entropy); the sampler's top-p 0.95 makes
it slightly lower. The script prints both as a sanity check.

Usage (one process per GPU; files are split by hash so shards never overlap):
    python helios/entropy/rescore.py --ids-dir $SCRATCH/token_ids/putnam20_whole_proof_8b \
        --shard 0 --num-shards 4
"""
import argparse
import glob
import os
import sys
import time
import zlib

import numpy as np


def stats_path(ids_path):
    return ids_path[: -len(".npz")] + ".stats.npz"


def token_stats(logits, targets):
    """logits: (T, V) float32, targets: (T,) long -> (entropy, varentropy, nll)."""
    import torch

    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    plogp = p * logp
    entropy = -plogp.sum(-1)
    varentropy = (plogp * logp).sum(-1) - entropy**2
    nll = -logp.gather(-1, targets[:, None]).squeeze(-1)
    return entropy, varentropy.clamp_min(0), nll


def rescore_file(model, path, device, chunk):
    import torch

    data = np.load(path)
    prompt, output = data["prompt_ids"], data["output_ids"]
    n_out = len(output)
    if n_out == 0 or len(prompt) == 0:
        empty = np.zeros(0, dtype=np.float16)
        return dict(entropy=empty, varentropy=empty, nll=empty)

    ids = torch.from_numpy(np.concatenate([prompt, output]).astype(np.int64))[None].to(device)
    with torch.no_grad():
        # No KV cache: a single forward pass does not need it, and it saves ~6 GB on 40k-token attempts.
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        # Position i predicts token i+1, so output token t is predicted at len(prompt)-1+t.
        hidden = hidden[len(prompt) - 1 : len(prompt) - 1 + n_out]
        targets = ids[0, len(prompt) :]
        parts = {"entropy": [], "varentropy": [], "nll": []}
        for s in range(0, n_out, chunk):
            logits = model.lm_head(hidden[s : s + chunk]).float()
            e, v, n = token_stats(logits, targets[s : s + chunk])
            parts["entropy"].append(e)
            parts["varentropy"].append(v)
            parts["nll"].append(n)
    return {k: torch.cat(v).cpu().numpy().astype(np.float16) for k, v in parts.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids-dir", required=True)
    ap.add_argument("--model", default="Goedel-LM/Goedel-Prover-V2-8B")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=1024, help="output positions per lm_head chunk")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many files (0 = all)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    files = sorted(
        f for f in glob.glob(os.path.join(args.ids_dir, "*.npz"))
        if not f.endswith(".stats.npz")
        and zlib.crc32(os.path.basename(f).encode()) % args.num_shards == args.shard
    )
    todo = [f for f in files if not os.path.exists(stats_path(f))]
    if args.limit:
        todo = todo[: args.limit]
    print(f"shard {args.shard}/{args.num_shards}: {len(files)} files, {len(todo)} to score", flush=True)
    if not todo:
        return

    device = args.device
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()

    t0, n_tokens = time.time(), 0
    for i, path in enumerate(todo, 1):
        try:
            stats = rescore_file(model, path, device, args.chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"OOM, skipped: {path}", file=sys.stderr, flush=True)
            continue
        tmp = stats_path(path) + ".tmp.npz"
        np.savez_compressed(tmp, **stats)
        os.replace(tmp, stats_path(path))
        n_tokens += len(stats["nll"])
        if (i % 50 == 0 or i == len(todo)) and len(stats["nll"]):
            e = stats["entropy"].astype(np.float32)
            nll = stats["nll"].astype(np.float32)
            print(
                f"[{i}/{len(todo)}] {n_tokens / (time.time() - t0):.0f} tok/s | last file: "
                f"mean H={e.mean():.3f} mean NLL={nll.mean():.3f} (expect NLL slightly < H)",
                flush=True,
            )


if __name__ == "__main__":
    main()
