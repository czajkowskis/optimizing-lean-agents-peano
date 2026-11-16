# Instructions for an AI coding agent: entropy-gated repair pilot

You are helping a researcher (the **operator**, who started you) run a GPU experiment on their Slurm cluster for Szymon Czajkowski. Follow these steps in order. Ask the operator only where a step says so. Keep a short log of what you did; your reports (steps 5 and 8) are built from it.

## Goal

The language model Goedel-Prover-V2-8B writes Lean proofs. For 872 of its saved failed proofs, the experiment asks the model to repair each one in several ways. It then checks every repaired proof with Lean and measures whether the model's uncertainty (entropy) on the broken line predicts which repair works. The design is in `helios/repair/DESIGN.md`. You don't need it to run things.

Everything is automated by `helios/repair/run.sh`. It runs in **steps, cheapest first**, and each step ends with a `DECISION:` line that says whether the next step is worth running.

| Step | Command flag | GPU cost | Runs only if |
|---|---|---|---|
| smoke | `--smoke` | minutes | always first |
| 1 | `--step 1` | under 1 A100-hour | the smoke test passed |
| 2 | `--step 2` | under 1 A100-hour | step 1's DECISION starts with `GO TO STEP 2` or `INCONCLUSIVE` |
| 3 | `--step 3 --gpus 4` | about 6–12 A100-hours | step 2's DECISION starts with `GO` or `INTERACTION WITHOUT CROSSOVER`, **and the operator approves** |

Your job is to:
- pick the right cluster settings;
- run the smoke test and then the steps;
- read each DECISION and stop where it says to stop;
- fix routine failures;
- report back after every step.

**Done means:** the last step you were allowed to run has finished, its `report.txt` ends with a `DECISION:` line, and its results archive exists.

## Rules

1. **Do not change** anything in `helios/repair/data/`, `helios/repair/*.py`, `lean_compiler/`, `prover/` or `mathlib4/`. Do not change the model, its precision (bf16), the sampling settings, or the pinned versions (vLLM 0.21.0, transformers 5.9.0). The results must stay comparable with the original runs on another cluster.
2. **Do not push** to git, open pull requests, or upload anything anywhere. The operator sends the results.
3. **Do not skip ahead.** Never run a step whose earlier step's DECISION says to stop, and never run step 3 without the operator's explicit approval.
4. **Ask the operator before:**
   - using more than 20 GPU-hours in total, including reruns;
   - deleting anything outside the clone's `amu_work/` folder;
   - using a partition or account you aren't sure they may use.
5. **Never** run the model or compile proofs on the login node. All GPU and Lean work goes through `run.sh`, which submits Slurm jobs. The only exceptions are setting up the environment, which `run.sh` does itself, and reading files.
6. **Never** have two runs going at the same time. `run.sh` refuses while earlier `repair*` jobs are queued or running; don't work around that.
7. **Don't handle passwords, tokens or keys.** The model download from Hugging Face needs no login.
8. `run.sh` installs Lean with elan into `~/.elan` (about 1–2 GB), because the proof checker expects it there. This is expected; you don't need to ask about it. If `~/.elan` already exists, it is reused.

## Step 1: find out about the cluster (read-only)

Run these on the login node and note the results:

```bash
sinfo -o '%P %G %l %D %t' | head -40            # partitions, their GPUs (GRES), time limits
sacctmgr -nP show assoc user=$USER format=account,partition 2>/dev/null | sort -u   # accounts you can charge
curl -sI https://huggingface.co | head -1       # internet on the login node? expect HTTP 200
curl -sI https://github.com | head -1           # needed for Lean and Mathlib
python3 --version; module avail python 2>&1 | head -20
df -h $HOME                                      # ~/.elan needs about 2 GB
```

**If this is the AMU cluster** and the re-scoring run was done here earlier, its settings are known to work: partition `gpu_csi`, `--gres gpu:nvidia_a100:1`. The earlier clone is probably at `/work/<user>/optimizing-lean-agents-peano`. Reusing it saves the 16 GB model download. Ask the operator for the account if it isn't obvious.

**Choose the GPU partition:**
- Use GPUs with **at least 40 GB of memory and bf16 support**: A100, H100, H200, GH200, L40S, or A6000/A40.
- **Avoid** V100, P100, T4 and 24 GB cards.
- The time limit must be at least 12 h.
- If the GRES column shows types (for example `gpu:a100:4`), you'll pass `--gres gpu:a100:1`.

**Choose the CPU partition:** proof checking runs on CPU nodes, with 32 cores and 128 GB for up to 12 h. `run.sh` picks the first partition without GPUs. If that one is wrong (debug-only, too short, not allowed), pass `--cpu-partition <name>`. If no CPU partition fits, pass the GPU partition's name; the jobs then request no GPU there.

**Choose the account:** if `sacctmgr` shows exactly one account, use it. If it shows several, ask the operator. If it shows none, try without `--account`.

**If nothing fits** (no internet on the login node, no GPU of at least 40 GB, under 40 GB of free project space), stop and tell the operator what is missing.

## Step 2: get the code

**Reusing an earlier clone:**

```bash
cd <existing clone>
git status --short          # expect nothing, or only amu_work/ and amu_stats.tar.gz
git fetch origin && git checkout amu-rescore && git pull --ff-only
```

If `git status` shows changes to tracked files, don't discard them: ask the operator.

**Fresh clone,** from a folder with at least 40 GB free:

```bash
git clone -b amu-rescore https://github.com/czajkowskis/optimizing-lean-agents-peano.git
cd optimizing-lean-agents-peano
```

Check that `helios/repair/data/pilot_attempts.jsonl` exists and has 872 lines (`wc -l`).

Set your options once; the examples below use them:

```bash
OPTS="--partition <P> --gres <gpu:type:1> [--account <A>] [--cpu-partition <C>]"
```

## Step 3: smoke test

```bash
bash helios/repair/run.sh --smoke $OPTS
```

On the login node this takes 10–30 minutes. It builds the venv in `amu_work/venv_repair`, downloads the model if needed, installs Lean and downloads prebuilt Mathlib. Then it submits four chained jobs: `repair-lean`, then `repair-gen`, then `repair-compile`, then `repair-analyze`.

**If the login output says `cache get failed`,** prebuilt Mathlib isn't available, and the `repair-lean` job will build it from source. That can take several hours of CPU time, within its 8 h limit. Tell the operator; it's a one-time cost.

Watch the jobs:

```bash
squeue -u $USER
cat amu_work/logs/repair-lean-*.out | tail -20
```

The `repair-lean` job recompiles 87 unchanged original proofs and compares the verdicts with the original run:
- **`COMPILE_CHECK_OK`:** good. The other jobs start by themselves.
- **`COMPILE_CHECK_FAILED`:** the remaining jobs are cancelled automatically. **Stop here.** Report the last 30 lines of the `repair-lean` log and `amu_work/repair_lean_check/compile_check.json` to the operator, to forward to Szymon. Don't retry with changed settings.

When `repair-analyze` finishes, check the smoke results:

```bash
cat amu_work/repair_smoke/report.txt
grep -c '"no lean block"' amu_work/repair_smoke/compiled.jsonl    # of 57-60 samples
tail -3 amu_work/logs/repair-gen-*.out                           # tok/s
```

The smoke test **passed** if all of these hold:
- the Lean check printed `COMPILE_CHECK_OK`;
- `report.txt` has an arms table with token counts for `local`, `back` and `feedback`;
- the `repair-compile` log ends with `COMPILE_OK`;
- at most about half of the `local` and `back` samples are `no lean block`.

Ignore the smoke test's `DECISION` line: 20 attempts are far too few to decide anything.

If the smoke test didn't pass, see step 7. Then go to step 5 and report it before going on.

## Step 4: run the steps

Run them one at a time. Wait for each `repair-analyze` job to finish, then read the decision:

```bash
bash helios/repair/run.sh --step 1 $OPTS
# ... wait until squeue shows no repair* jobs ...
grep '^DECISION' amu_work/repair/report.txt
```

Then follow the decision:

| After | DECISION starts with | Do |
|---|---|---|
| step 1 | `GO TO STEP 2` or `INCONCLUSIVE` | report (step 5), then run `--step 2` |
| step 1 | `STOP` | report (step 5) and finish (step 8) |
| step 2 | `GO` or `INTERACTION WITHOUT CROSSOVER` | report (step 5) and **ask the operator** whether to run step 3 (about 6–12 GPU-hours) |
| step 2 | `STOP` or `INCONCLUSIVE` | report (step 5) and finish (step 8) |
| step 3 | anything | report (step 5) and finish (step 8) |

Step 3 is run with 4 GPUs: `bash helios/repair/run.sh --step 3 $OPTS --gpus 4`.

Steps 1–3 all write to `amu_work/repair/`. Each step adds to what's there, so never delete that folder between steps.

## Step 5: report after each step

Give the operator this summary to send to Szymon, with the results archive (`repair_smoke_results.tar.gz` for the smoke test, `repair_results.tar.gz` for steps 1–3, in the clone's root folder):

```
Step: <smoke / 1 / 2 / 3>
Result archive: <absolute path>, <size>
Cluster / GPU partition / GPU type / CPU partition: <...>
Lean check: <the "compile check: ..." line and COMPILE_CHECK_OK/FAILED>
Mathlib: <prebuilt download worked / built from source, how long>
Speed: <typical new tok/s per GPU from the repair-gen logs>
GPU time used this step and in total: <from sacct, see below>
Reruns needed: <none, or what failed and how it was fixed>
report.txt (verbatim, from the arms table to the DECISION line):
<paste>
```

GPU time:

```bash
sacct -u $USER -S $(date -d '-14 days' +%F) --name=repair-gen -X --format=JobID%18,State,Elapsed,AllocTRES%40
```

## Step 6: watch running jobs

Check every 30–60 minutes; there's no need to poll more often.

```bash
squeue -u $USER
tail -n 2 amu_work/logs/repair-gen-*.out          # progress: [k/n] new tok/s
tail -n 2 amu_work/logs/repair-compile-*.out      # progress: [k/n] compiled
```

## Step 7: fix failures

The universal fix is to **run the same `run.sh` command again** from the same folder, once `squeue -u $USER` shows no `repair*` jobs. Everything finished is kept: the venv, Lean, a passed Lean check, every generated sample, and every compiled proof.

| Symptom (where) | What to do |
|---|---|
| A job hit its time limit (`DUE TO TIME LIMIT` in a log, or `TIMEOUT` in `sacct -X -j <id>`) | Rerun. For `repair-gen` in step 3, add `--gpus 8` if the logs show under about 2,000 tok/s. |
| `repair-lean` fails building Mathlib or the REPL (a `lake` error, not `COMPILE_CHECK_FAILED`) | Report the last 50 lines of the log to the operator. Don't edit `mathlib4/`. |
| `COMPILE_CHECK_FAILED` | Stop and report (see step 3). |
| `CUDA out of memory` in a `repair-gen` log | Rerun with a partition or GRES that gives 80 GB GPUs. |
| `repair-gen` fails at startup with a CUDA, driver or `nvcc` error | Report the log to the operator. Don't change the vLLM or torch versions. |
| sbatch rejects a job (invalid account, partition or gres, or a QOS error) | Fix the option using what you learned in step 1; ask the operator if unsure. Then rerun. |
| `model download failed` | The login node has no internet. Ask the operator: the model must be downloaded elsewhere and copied into `amu_work/hf_cache`. |
| `the venv is broken` | `rm -rf amu_work/venv_repair` and rerun. |
| Jobs wait in the queue for a long time | That's normal. Report the expected start time (`squeue --start -u $USER`) to the operator if it's more than a day away. Don't switch to unsuitable GPUs to start sooner. |
| Anything else | Read the job's log in `amu_work/logs/`. If it's a cluster problem (node failure, preemption), rerun. Otherwise report the log to the operator. |

**Rerun at most 3 times** for the same problem; after that, report to the operator.

## Step 8: finish

Tell the operator:
- which step was the last one run, and why the experiment stopped there (the DECISION line, or the operator's choice);
- the total GPU time used;
- that all results are in the archives already reported.

After the operator has sent the archives, they may delete `amu_work/venv_repair` (about 10 GB). They can delete the whole clone only if no further runs are planned. `~/.elan` can be removed with `rm -rf ~/.elan` if nothing else on the cluster uses Lean.
