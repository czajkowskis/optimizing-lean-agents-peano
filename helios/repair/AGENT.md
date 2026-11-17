# Instructions for an AI coding agent: entropy-gated repair pilot

You are helping a researcher (the **operator**, who started you) run a GPU experiment on their Slurm cluster for Szymon Czajkowski. Follow these steps in order. Ask the operator only where a step says so. Keep a short log of what you did; your reports (steps 5 and 8) are built from it.

## Goal

The language model Goedel-Prover-V2-8B writes Lean proofs. For 872 of its saved failed proofs, the experiment asks the model to repair each one in several ways. It then checks every repaired proof with Lean and measures whether the model's uncertainty (entropy) on the broken line predicts which repair works. The design is in `helios/repair/DESIGN.md`. You don't need it to run things.

Everything is automated by `helios/repair/run.sh --all`. **One command submits all stages as a chain of Slurm jobs**, each starting when the one before it succeeds:

| Stage | What it generates | GPU cost |
|---|---|---|
| Lean check | nothing: recompiles 87 original proofs and compares them with the original verdicts | 0 (CPU) |
| smoke | 20 attempts × 1 sample per arm, plus an automatic smoke gate | minutes |
| step 1 | LOCAL × 2 samples, 872 attempts | under 1 A100-hour |
| step 2 | adds BACK × 2 samples | under 1 A100-hour |
| step 3 | adds FEEDBACK and raises every arm to 4 samples (4 GPUs) | about 6–12 A100-hours |

**All stages run, whatever the earlier `DECISION:` lines say.** Those lines are for Szymon to read afterwards. The chain stops by itself only if something is broken: the Lean check fails, the smoke gate fails, or a job fails. Expect about 8–14 GPU-hours in total.

Your job is to:
- pick the right cluster settings;
- start the run with one command;
- watch it, and fix routine failures by rerunning the same command;
- report back.

**Done means:** `report_step3.txt` ends with a `DECISION:` line and `step3_results.tar.gz` exists in the clone's root folder, or the chain stopped on a failure you have reported.

## Rules

1. **Do not change** anything in `helios/repair/data/`, `helios/repair/*.py`, `lean_compiler/`, `prover/` or `mathlib4/`. Do not change the model, its precision (bf16), the sampling settings, or the pinned versions (vLLM 0.21.0, transformers 5.9.0). The results must stay comparable with the original runs on another cluster.
2. **Do not push** to git, open pull requests, or upload anything anywhere. The operator sends the results.
3. **Use `--all`.** Don't run the stages one by one with `--smoke` or `--step`, and don't stop the chain because of a `DECISION:` line: the researcher wants every stage run.
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

## Step 3: start the whole run

```bash
bash helios/repair/run.sh --all $OPTS
```

On the login node this takes 10–30 minutes. It builds the venv in `amu_work/venv_repair`, downloads the model if needed, installs Lean and downloads prebuilt Mathlib. Then it submits the whole chain and prints the job ids: the Lean check, then for each of smoke, step1, step2 and step3 a `repair-gen-<stage>`, a `repair-compile-<stage>` and a `repair-analyze-<stage>` job. Note the ids in your log.

**If the login output says `cache get failed`,** prebuilt Mathlib isn't available, and the `repair-lean` job will build it from source. That can take several hours of CPU time, within its 8 h limit. Tell the operator; it's a one-time cost.

## Step 4: the two automatic gates

The chain checks itself twice early on. Look at both as soon as they finish.

**1. The Lean check** (`amu_work/logs/repair-lean-*.out`) recompiles 87 unchanged original proofs and compares the verdicts with the original run.
- **`COMPILE_CHECK_OK`:** good. The chain continues by itself.
- **`COMPILE_CHECK_FAILED`:** every later job is cancelled automatically. **Stop here.** Report the last 30 lines of the log and `amu_work/repair_lean_check/compile_check.json` (step 5). Don't retry with changed settings.

**2. The smoke gate** (end of `amu_work/logs/repair-analyze-smoke-*.out`) checks that the repaired proofs mostly close their Lean code block and that Lean ran without system errors.
- **`SMOKE_OK`:** good. Steps 1–3 follow by themselves.
- **`SMOKE_FAILED`:** steps 1–3 are cancelled automatically. **Stop here** and report the log and `smoke_results.tar.gz` (step 5).

Ignore the smoke stage's `DECISION:` line: 20 attempts are far too few to decide anything.

Steps 1–3 all write to `amu_work/repair/`, each adding to what's there. Never delete that folder.

## Step 5: report

Report to the operator **twice**: once after the smoke gate (`SMOKE_OK`, or a failure), so Szymon knows the setup works, and once at the end. If the chain stopped on a failure, report then instead.

Each finished stage leaves `<stage>_results.tar.gz` in the clone's root folder and `amu_work/repair*/report_<stage>.txt`. The last archive contains everything before it, so at the end only `step3_results.tar.gz` (plus `smoke_results.tar.gz`) needs to be sent.

```
Last stage finished: <Lean check / smoke / step1 / step2 / step3>, or where and why the chain stopped
Result archives: <absolute paths and sizes>
Cluster / GPU partition / GPU type / CPU partition: <...>
Lean check: <the "compile check: ..." line and COMPILE_CHECK_OK/FAILED>
Smoke gate: <the "smoke gate: ..." line and SMOKE_OK/FAILED>
Mathlib: <prebuilt download worked / built from source, how long>
Speed: <typical new tok/s per GPU from the repair-gen logs>
GPU time used: <total, from sacct, see below>
Reruns needed: <none, or what failed and how it was fixed>
DECISION lines: <the DECISION line of each report_step*.txt>
report_<last step>.txt (verbatim, from the arms table to the DECISION line):
<paste>
```

GPU time:

```bash
sacct -u $USER -S $(date -d '-14 days' +%F) -X --format=JobID%18,JobName%22,State,Elapsed,AllocTRES%40 | grep -E 'JobID|repair-gen'
```

## Step 6: watch running jobs

Check every 30–60 minutes; there's no need to poll more often.

```bash
squeue -u $USER
tail -n 2 amu_work/logs/repair-gen-*.out          # progress: [k/n] new tok/s
tail -n 2 amu_work/logs/repair-compile-*.out      # progress: [k/n] compiled
```

## Step 7: fix failures

The universal fix is to **run the same `run.sh --all` command again** from the same folder, once `squeue -u $USER` shows no `repair*` jobs. Everything finished is kept: the venv, Lean, a passed Lean check, every generated sample, and every compiled proof. The rerun resubmits the whole chain, but stages that are already done finish in minutes, because nothing is generated or compiled twice.

| Symptom (where) | What to do |
|---|---|
| A job hit its time limit (`DUE TO TIME LIMIT` in a log, or `TIMEOUT` in `sacct -X -j <id>`) | Rerun. For `repair-gen` in step 3, add `--gpus 8` if the logs show under about 2,000 tok/s. |
| `repair-lean` fails building Mathlib or the REPL (a `lake` error, not `COMPILE_CHECK_FAILED`) | Report the last 50 lines of the log to the operator. Don't edit `mathlib4/`. |
| `COMPILE_CHECK_FAILED` | Stop and report (see step 4). |
| `SMOKE_FAILED` | Stop and report (see step 4). Don't rerun: the same samples would fail the same way. |
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
- whether all stages finished, or where and why the chain stopped;
- the total GPU time used;
- that all results are in the archives already reported.

After the operator has sent the archives, they may delete `amu_work/venv_repair` (about 10 GB). They can delete the whole clone only if no further runs are planned. `~/.elan` can be removed with `rm -rf ~/.elan` if nothing else on the cluster uses Lean.
