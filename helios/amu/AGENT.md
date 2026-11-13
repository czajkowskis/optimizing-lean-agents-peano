# Instructions for an AI coding agent: entropy re-scoring run

You are helping a researcher (the **operator**, who started you) run a GPU job on their Slurm cluster for Szymon Czajkowski. Follow these steps in order. Ask the operator only where a step says so. Keep a short log of what you did; your final report (step 7) is built from it.

## Goal

Run one forward pass of the language model Goedel-Prover-V2-8B over 16,640 saved proof attempts. For every token, save the model's entropy and the surprisal of the chosen token. Everything is automated by `helios/amu/run.sh`. Your job is to:
- pick the right cluster settings,
- start the run,
- watch it, and fix routine failures,
- report back.

**Done means:** the file `amu_stats.tar.gz` exists (about 1 GB) and the collect log ends with `DONE`.

**Expected cost:** about 4 GPU-hours on H100 or GH200, 10–12 on A100, plus a 15-minute check job. It runs as 4 jobs with 1 GPU each, so about 1–3 h of wall time plus queue time.

## Rules

1. **Do not change** anything in `amu_data/`, `helios/entropy/rescore.py` or `helios/amu/verify.py`. Do not change the model, its precision (bf16), or the transformers version (5.9.0). The results must stay comparable with runs on another cluster.
2. **Do not push** to git, open pull requests, or upload anything anywhere. The operator sends the result file.
3. **Ask the operator before:**
   - using more than 30 GPU-hours in total, including reruns;
   - deleting anything outside the clone's `amu_work/` folder;
   - using a partition or account you aren't sure they may use.
4. **Never** run the model on the login node. All GPU work goes through `run.sh`, which submits Slurm jobs.
5. **Never** have two runs of the re-scoring at the same time. `run.sh` refuses while earlier jobs are queued or running; don't work around that.
6. **Don't handle passwords, tokens or keys.** The model download from Hugging Face needs no login.

## Step 1: find out about the cluster (read-only)

Run these on the login node and note the results:

```bash
sinfo -o '%P %G %l %D %t' | head -40            # partitions, their GPUs (GRES), time limits
sacctmgr -nP show assoc user=$USER format=account,partition 2>/dev/null | sort -u   # accounts you can charge
curl -sI https://huggingface.co | head -1       # does the login node have internet? expect HTTP 200
python3 --version; module avail python 2>&1 | head -20
```

Also look for project or scratch storage with **at least 30 GB free**. Check `df -h` on `$HOME`, `$SCRATCH`, `/scratch`, `/project*` and `/work*`, and the cluster documentation if it's linked in the login message (`cat /etc/motd`). Home directories are often too small.

**Choose the partition:**
- Use GPUs with **at least 40 GB of memory and bf16 support**: A100, H100, H200, GH200, L40S (48 GB) or A6000/A40 (48 GB).
- **Avoid** V100, P100, T4 and 24 GB cards: no bf16, too slow, or too little memory for the longest 41k-token attempts.
- If the GRES column shows types (for example `gpu:a100:4`), you'll pass `--gres gpu:a100:1`.
- The partition's time limit must be **at least 6 h**.

**Choose the account:** if `sacctmgr` shows exactly one account, use it. If it shows several, ask the operator which one to charge. If it shows none, try without `--account`.

**If nothing fits** (no internet on the login node, no GPU of at least 40 GB, no 30 GB of space), stop and tell the operator what is missing.

## Step 2: clone and start

From the folder with enough space:

```bash
git clone -b amu-rescore https://github.com/czajkowskis/optimizing-lean-agents-peano.git
cd optimizing-lean-agents-peano
bash helios/amu/run.sh --partition <P> [--account <A>] [--gres gpu:<type>:1]
```

`run.sh` takes 10–20 minutes on the login node. It:
1. builds a venv in `amu_work/venv`;
2. downloads the model (16 GB) into `amu_work/hf_cache`;
3. submits three chained jobs: `rescore-check`, then `rescore` (an array of 4), then `rescore-collect`.

If it needs Python 3.10 or newer and none is on the PATH, it tries `module load python/3.1x`. If that fails, load a suitable module yourself and rerun with `PYTHON=$(which python3.11) bash helios/amu/run.sh ...`.

## Step 3: watch the check job

```bash
squeue -u $USER
cat amu_work/logs/check-*.out
```

The check re-scores 200 attempts that were already scored on the Helios cluster and compares the results.
- **`VERDICT: looks consistent with Helios`:** good. The main jobs start by themselves. Note the `tok/s` numbers in the log.
- **`VERDICT: DIFFERENT from Helios`:** the main jobs are cancelled automatically. **Stop here.** Report the full check log to the operator, to forward to Szymon. Don't retry with changed settings.
- **`PyTorch cannot use the GPU`:** see step 5.

## Step 4: watch the main run

```bash
squeue -u $USER
tail -n 3 amu_work/logs/rescore-*.out          # progress lines: [k/n] tok/s
```

Check every 30–60 minutes; there's no need to poll more often. When `rescore-collect` has finished, read `amu_work/logs/collect-*.out`:
- **Ends with `DONE: send .../amu_stats.tar.gz`:** go to step 6.
- **Says attempts are missing:** see step 5.

## Step 5: fix failures

The universal fix is to **run the same `run.sh` command again** from the same folder, once `squeue -u $USER` shows no `rescore*` jobs. Everything finished is kept: the venv, the model, a passed check, and every attempt already scored.

| Symptom (where) | What to do |
|---|---|
| A `rescore` task hit its time limit (`DUE TO TIME LIMIT` in a log, or `TIMEOUT` in `sacct -X -j <id>`) | Rerun. If the logs show under about 4,000 tok/s, add `--gpus 8`. |
| `OOM, skipped: ...` in a rescore log | That GPU is too small for the longest attempts. Rerun with `--partition`/`--gres` pointing at bigger GPUs (80 GB if possible). Only the skipped attempts are processed. |
| `PyTorch cannot use the GPU` (check log) | The driver is older than PyTorch's CUDA. Run `nvidia-smi` in a short GPU job to read the "CUDA Version". Then `rm -rf amu_work/venv` and rerun with `--torch-index https://download.pytorch.org/whl/cu121`, or `cu118` for drivers older than CUDA 12.1. |
| sbatch rejects the job (invalid account, partition or gres, or a QOS error) | Fix the option using what you learned in step 1; ask the operator if unsure. Then rerun. |
| `model download failed` | The login node has no internet. Ask the operator: the model must be downloaded elsewhere and copied into `amu_work/hf_cache`. |
| A job failed for another reason | Read its log in `amu_work/logs/`. If it's a cluster problem (node failure, preemption), rerun. Otherwise report the log to the operator. |
| Jobs wait in the queue for a long time | That's normal. Report the expected start time (`squeue --start -u $USER`) to the operator if it's more than a day away. Don't switch to unsuitable GPUs to start sooner. |

**Rerun at most 3 times** for the same problem; after that, report to the operator.

## Step 6: final check

```bash
ls -lh amu_stats.tar.gz                                         # about 1 GB
tar tzf amu_stats.tar.gz | grep -c 'stats.npz$'                 # expect 16840 (16640 attempts + 200 check)
sacct -u $USER -S $(date -d '-7 days' +%F) --name=rescore-check,rescore,rescore-collect -X \
      --format=JobID%18,JobName%16,State,Elapsed,AllocTRES%40     # GPU time used
```

## Step 7: report to the operator

Give the operator this summary to send to Szymon, together with the file:

```
Result file: <absolute path to amu_stats.tar.gz>, <size>, <number of stats files>
Cluster / partition / GPU type: <...>
Speed: <typical tok/s per GPU from the logs>
GPU time used: <total, from sacct>
Reruns needed: <none, or what failed and how it was fixed>
OOM-skipped attempts: <none, or the count>
Check job output (verbatim):
<paste the last ~12 lines of amu_work/logs/check-*.out, from "compared ..." to the VERDICT>
```

The operator sends `amu_stats.tar.gz` (or a link to it) and this summary to Szymon. After that, you can tell the operator they may delete the clone. That frees about 25 GB, mostly the venv and the model.
