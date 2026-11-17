# Repair pilot on a Slurm GPU cluster

Does the model's entropy on the line where a proof breaks tell us how to repair it? The design, the hypotheses and the decision rule are in [DESIGN.md](DESIGN.md). This page is how to run it.

## Run it

Run on the cluster's login node, from a clone of the `amu-rescore` branch with at least 40 GB free. If the AMU re-scoring clone is still there, use it: the model download in `amu_work/hf_cache` is reused.

*Using an AI coding agent?* Give it this prompt: "Follow `helios/repair/AGENT.md` from the `amu-rescore` branch of https://github.com/czajkowskis/optimizing-lean-agents-peano".

**Everything in one command** (the default way to run it):

```bash
bash helios/repair/run.sh --all $OPTS
```

This submits the whole pilot as one chain of Slurm jobs, each stage starting when the one before it succeeds. The stages are: the Lean check, the smoke test, step 1, step 2 and step 3, cheapest first (see "Cost ladder" in [DESIGN.md](DESIGN.md)). It takes about 8–14 A100-hours in total.

Every stage runs, whatever the earlier `DECISION:` lines say. The chain stops by itself only on a real failure:
- the Lean check doesn't reproduce the Helios verdicts;
- the smoke gate finds the repaired proofs not closing their Lean block;
- a job fails.

On AMU, `OPTS="--partition gpu_csi --gres gpu:nvidia_a100:1 --account <account>"`: these are the settings the re-scoring run used. Add `--cpu-partition <name>` if the automatically chosen CPU partition is wrong (`sinfo -s` lists them). `--gpus N` sets the number of GPUs for step 3 (default 4).

Each stage leaves `amu_work/repair*/report_<stage>.txt`, ending in a `DECISION:` line, and `<stage>_results.tar.gz` in the clone's root folder. When the chain is done, send `smoke_results.tar.gz` and `step3_results.tar.gz` to Szymon. The last archive contains everything from steps 1–3.

**One stage at a time** (to stop as soon as a DECISION says so, and save GPU time):

| Stage | Command | Cost |
|---|---|---|
| smoke | `bash helios/repair/run.sh --smoke $OPTS` | minutes of GPU, plus the one-time Lean setup |
| 1 | `bash helios/repair/run.sh --step 1 $OPTS` | under 1 A100-hour (LOCAL only) |
| 2 | `bash helios/repair/run.sh --step 2 $OPTS` | under 1 A100-hour more (adds BACK) |
| 3 | `bash helios/repair/run.sh --step 3 $OPTS --gpus 4` | about 6–12 A100-hours more (adds FEEDBACK, 4 samples) |

## What happens

`run.sh` takes 10–30 minutes on the login node. It:

1. builds a venv in `amu_work/venv_repair` with vLLM 0.21.0, torch 2.11.0 and transformers 5.9.0, the versions of the original Helios run;
2. downloads the model, unless it's already there;
3. installs Lean with elan into `~/.elan`, checks out the `mathlib4` submodule and downloads prebuilt Mathlib.

Then it submits chained jobs. Each stage (`smoke`, `step1`, `step2`, `step3`) has three of them:

| Job | Where | What it does |
|---|---|---|
| `repair-lean` | CPU | Builds the Lean REPL, then recompiles 87 **unchanged** original attempts. They must get the same verdicts as on Helios, or the remaining jobs are cancelled. Runs only once. |
| `repair-gen-<stage>` | GPU array | Generates the stage's LOCAL, BACK and FEEDBACK samples (`generate.py`). |
| `repair-compile-<stage>` | CPU | Compiles every sample with the prover's own Lean scheduler (`compile.py`). |
| `repair-analyze-<stage>` | CPU | Writes the report and the decision (`analyze.py`), then packs the results. For the smoke stage it also runs the smoke gate (`SMOKE_OK` or `SMOKE_FAILED`). |

**If anything fails or hits a time limit,** run the same command again. The venv, Lean, a passed check, and every finished sample and compile result are kept, so stages that are already done finish in minutes.

## Smoke-test checklist (for Szymon)

The checks below use `amu_work/repair_smoke/report_smoke.txt` and the logs. The first and third are enforced automatically.

- [ ] `repair-lean` log ends with `COMPILE_CHECK_OK`.
- [ ] The arms table shows real token counts per arm. Use them to update the budget in DESIGN.md.
- [ ] The `repair-analyze-smoke` log ends with `SMOKE_OK`: at most half of the LOCAL and BACK samples lack a Lean block, and Lean system errors are rare.
- [ ] The tok/s in the `repair-gen-smoke` log is as expected.

## Files

| File | Role |
|---|---|
| `choose.py` | Runs where the original attempts live (the laptop). Fixes the entropy gate, picks the attempts, computes the cut points and builds the FEEDBACK prompts. Writes `data/`. |
| `data/pilot_attempts.jsonl` | One line per selected attempt: gate z-score, stratum, cut points, FEEDBACK prompt, original code. |
| `data/ids/` | The selected attempts' `prompt_ids` and `output_ids`. |
| `data/compile_check.jsonl` | Original attempts with their Helios verdicts, for the Lean check. |
| `data/selection.json` | The threshold, skip counts, per-problem pools and baseline rates. |
| `generate.py`, `compile.py`, `analyze.py` | The three steps. |
| `prover_funcs.py` | The prover's own code-extraction functions, loaded from its source files. |
| `run.sh`, `job.sh` | The Slurm wrapper. |
