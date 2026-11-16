# Repair pilot on a Slurm GPU cluster

Does the model's entropy on the line where a proof breaks tell us how to repair it? The design, the hypotheses and the decision rule are in [DESIGN.md](DESIGN.md). This page is how to run it.

## Run it

Run on the cluster's login node, from a clone of the `amu-rescore` branch with at least 40 GB free. If the AMU re-scoring clone is still there, use it: the model download in `amu_work/hf_cache` is reused.

The pilot runs in steps, cheapest first (see "Cost ladder" in [DESIGN.md](DESIGN.md)). After each step, read the `DECISION:` line at the end of `amu_work/repair/report.txt` (`repair_smoke/` for the smoke test). Run the next step only if the decision says so.

| Step | Command | Cost |
|---|---|---|
| smoke | `bash helios/repair/run.sh --smoke $OPTS` | minutes of GPU, plus the one-time Lean setup |
| 1 | `bash helios/repair/run.sh --step 1 $OPTS` | under 1 A100-hour (LOCAL only) |
| 2 | `bash helios/repair/run.sh --step 2 $OPTS` | under 1 A100-hour more (adds BACK) |
| 3 | `bash helios/repair/run.sh --step 3 $OPTS --gpus 4` | about 6–12 A100-hours more (adds FEEDBACK, 4 samples) |

On AMU, `OPTS="--partition gpu_csi --gres gpu:nvidia_a100:1 --account <account>"`: these are the settings the re-scoring run used. Add `--cpu-partition <name>` if the automatically chosen CPU partition is wrong (`sinfo -s` lists them).

When the analyze job finishes, send `repair_smoke_results.tar.gz` (or `repair_results.tar.gz` for steps 1–3) to Szymon.

## What happens

`run.sh` takes 10–30 minutes on the login node. It:

1. builds a venv in `amu_work/venv_repair` with vLLM 0.21.0, torch 2.11.0 and transformers 5.9.0, the versions of the original Helios run;
2. downloads the model, unless it's already there;
3. installs Lean with elan into `~/.elan`, checks out the `mathlib4` submodule and downloads prebuilt Mathlib.

Then it submits four chained jobs:

| Job | Where | What it does |
|---|---|---|
| `repair-lean` | CPU | Builds the Lean REPL, then recompiles 87 **unchanged** original attempts. They must get the same verdicts as on Helios, or the remaining jobs are cancelled. Runs only once. |
| `repair-gen` | GPU array | Generates LOCAL, BACK and FEEDBACK samples (`generate.py`). |
| `repair-compile` | CPU | Compiles every sample with the prover's own Lean scheduler (`compile.py`). |
| `repair-analyze` | CPU | Writes the report and the go / stop decision (`analyze.py`), then packs the results. |

**If anything fails or hits a time limit,** run the same command again. The venv, Lean, a passed check, and every finished sample and compile result are kept.

## Smoke-test checklist (for Szymon)

The checks below use `amu_work/repair_smoke/report.txt` and the logs:

- [ ] `repair-lean` log ends with `COMPILE_CHECK_OK`.
- [ ] The arms table shows real token counts per arm. Use them to update the budget in DESIGN.md.
- [ ] Few samples in `compiled.jsonl` have `not_compiled: "no lean block"`. Many would mean continuations aren't closing the proof block.
- [ ] The tok/s in the `repair-gen` log is as expected.

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
