#!/bin/bash
# One command for the repair pilot on a Slurm cluster (see helios/repair/README.md):
#
#   bash helios/repair/run.sh --all [options]     # everything below in one go: smoke, then steps 1, 2, 3 as chained
#                                                 # Slurm jobs; a failed stage (or a failed smoke check) cancels the rest
#   bash helios/repair/run.sh --smoke [options]   # 20 attempts x 1 sample per arm: checks the pipeline, measures tokens
#   bash helios/repair/run.sh --step 1 [options]  # LOCAL only, 2 samples per attempt     (~5M tokens, <1 GPU-hour)
#   bash helios/repair/run.sh --step 2 [options]  # adds BACK, 2 samples per attempt      (~5M more)
#   bash helios/repair/run.sh --step 3 [options]  # all arms incl. FEEDBACK, 4 samples   (the full design, ~60M more)
# Each step reuses everything the earlier steps generated and compiled. Run one at a time, a step only if the
# previous DECISION line says so (DESIGN.md, "Cost ladder"), or all at once with --all, which runs every step.
#
# Options: --partition P (GPU) --cpu-partition P --account A --gres gpu:1 --gpus N (default 1; 4 for step 3, and
#          with --all it sets step 3 only)
#
# Sets up (once) a vLLM venv, the model and Lean + mathlib4, then submits chained jobs:
#   1. lean-check (CPU):  build the Lean REPL if needed and recompile original attempts; must
#                         reproduce the Helios verdicts, or everything after it is cancelled
#   2. generate (GPU):    an array of --gpus one-GPU jobs: LOCAL, BACK and FEEDBACK samples
#   3. compile (CPU):     compile every sample with the prover's Lean scheduler
#   4. analyze (CPU):     the report, ending with the go / stop decision
# Safe to run again: finished setup, the passed check, and every finished sample are kept.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

PARTITION="" CPU_PARTITION="" ACCOUNT="" GRES="gpu:1" NGPU="" SMOKE=0 STEP="" ALL=0
while [ $# -gt 0 ]; do
  case $1 in
    --all) ALL=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --step) STEP=$2; shift 2 ;;
    --partition) PARTITION=$2; shift 2 ;;
    --cpu-partition) CPU_PARTITION=$2; shift 2 ;;
    --account) ACCOUNT=$2; shift 2 ;;
    --gres) GRES=$2; shift 2 ;;
    --gpus) NGPU=$2; shift 2 ;;
    -h|--help) sed -n '2,/^set -euo/{/^#/p}' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)"; exit 2 ;;
  esac
done
WORK=$REPO/amu_work
ROUT=$WORK/repair
if [ $ALL = 1 ]; then
  NGPU3=${NGPU:-4}        # --gpus applies to step 3; the smoke test and steps 1-2 use 1 GPU
elif [ $SMOKE = 1 ]; then
  LABEL=smoke GEN_ARGS="--limit-attempts 20 --samples 1 --arms local,back,feedback" NGPU=1 ROUT=$WORK/repair_smoke
else
  case $STEP in
    1) LABEL=step1 GEN_ARGS="--samples 2 --arms local" ;;
    2) LABEL=step2 GEN_ARGS="--samples 2 --arms local,back" ;;
    3) LABEL=step3 GEN_ARGS="--samples 4 --arms local,back,feedback"; NGPU=${NGPU:-4} ;;
    *) echo "choose --all, --smoke or --step 1|2|3 (see --help)"; exit 2 ;;
  esac
fi
NGPU=${NGPU:-1}

export ROUT   # job.sh needs it; each stage of --all overrides it at submission
export RVENV=${RVENV:-$WORK/venv_repair}
export HF_HOME=${HF_HOME:-$WORK/hf_cache}
mkdir -p "$WORK/logs" "$ROUT" "$WORK/repair_smoke"
echo '*' > "$WORK/.gitignore"
say() { printf '\n== %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

say "checks"
command -v sbatch >/dev/null || die "sbatch not found: run this on the cluster's login node"
N=$(wc -l < helios/repair/data/pilot_attempts.jsonl)
[ "$(ls helios/repair/data/ids | wc -l)" = "$N" ] || die "helios/repair/data is incomplete; is the clone complete?"
echo "$N pilot attempts; output in $ROUT"

say "python environment for vLLM ($RVENV)"
if [ ! -x "$RVENV/bin/python" ]; then
  PY=${PYTHON:-}
  if [ -z "$PY" ]; then
    for c in python3.12 python3.11 python3.10 python3; do
      if command -v $c >/dev/null && $c -c 'import sys; sys.exit(not ((3, 10) <= sys.version_info < (3, 13)))' 2>/dev/null; then PY=$c; break; fi
    done
  fi
  [ -n "$PY" ] || die "no Python 3.10-3.12 found; rerun with PYTHON=/path/to/python3.11 bash helios/repair/run.sh ..."
  echo "using $($PY --version) at $(command -v $PY)"
  $PY -m venv "$RVENV"
  "$RVENV/bin/pip" install -q --upgrade pip
  # the versions of the original Helios run (uv.lock): vLLM 0.21.0 brings torch 2.11.0
  "$RVENV/bin/pip" install -q "vllm==0.21.0" "transformers==5.9.0" numpy pexpect huggingface_hub
fi
"$RVENV/bin/python" -c 'import vllm, torch, transformers, pexpect; print("vllm", vllm.__version__, "| torch", torch.__version__, "| transformers", transformers.__version__)' \
  || die "the venv is broken; delete $RVENV and run again"

say "model (Goedel-LM/Goedel-Prover-V2-8B, into $HF_HOME)"
"$RVENV/bin/python" -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Goedel-LM/Goedel-Prover-V2-8B"))' \
  || die "model download failed; does the login node have internet?"

say "Lean (elan in ~/.elan, as lean_compiler/repl_scheduler.py expects) + mathlib4"
if [ ! -x "$HOME/.elan/bin/lake" ]; then
  curl https://elan.lean-lang.org/elan-init.sh -sSf | sh -s -- -y --no-modify-path --default-toolchain none
fi
[ -f mathlib4/lean-toolchain ] || git submodule update --init mathlib4
# downloading prebuilt Mathlib files needs internet, so it runs here; building the REPL runs in the lean-check job
(cd mathlib4 && PATH=$HOME/.elan/bin:$PATH lake exe cache get) || echo "cache get failed: the lean-check job will build Mathlib from source (hours)"

say "slurm settings"
if [ -z "$PARTITION" ]; then
  PARTITION=$(sinfo -h -o '%P %G' | awk 'tolower($2) ~ /gpu/ {gsub(/\*/, "", $1); print $1; exit}')
  [ -n "$PARTITION" ] || die "could not find a GPU partition; rerun with --partition <name>"
fi
if [ -z "$CPU_PARTITION" ]; then
  CPU_PARTITION=$(sinfo -h -o '%P %G' | awk 'tolower($2) !~ /gpu/ {gsub(/\*/, "", $1); print $1; exit}')
  [ -n "$CPU_PARTITION" ] || CPU_PARTITION=$PARTITION
fi
echo "GPU partition $PARTITION ($GRES, $NGPU jobs), CPU partition $CPU_PARTITION, account ${ACCOUNT:-(your default)}"
A=(${ACCOUNT:+--account="$ACCOUNT"})
SB_GPU=(sbatch --parsable --partition="$PARTITION" --gres="$GRES" --cpus-per-task=8 --mem=64G --export=ALL ${A[@]+"${A[@]}"})
SB_CPU=(sbatch --parsable --partition="$CPU_PARTITION" --cpus-per-task=32 --mem=128G --export=ALL ${A[@]+"${A[@]}"})

say "submitting"
if squeue -h -u "${USER:-$(whoami)}" -o '%j' 2>/dev/null | grep -q '^repair'; then
  squeue -u "${USER:-$(whoami)}" -o '%.12i %.16j %.10T %.12M %R' | grep -E 'JOBID|repair'
  die "repair jobs from an earlier run are still queued or running; wait for them (or scancel them) and run again"
fi
DEPS=""   # job ids the next stage waits for (all must succeed)
if [ -f "$WORK/lean_check_passed" ]; then
  echo "Lean check already passed earlier: skipping it"
else
  J=$("${SB_CPU[@]}" --job-name=repair-lean --time=08:00:00 --output="$WORK/logs/repair-lean-%j.out" helios/repair/job.sh lean)
  echo "lean-check job: $J"
  DEPS=$J
fi

# submit_stage LABEL OUTDIR NGPU GEN_ARGS...: generate -> compile -> analyze, after $DEPS; sets DEPS for the next stage
submit_stage() {
  local label=$1 out=$2 ngpu=$3; shift 3
  local dep=(); [ -n "$DEPS" ] && dep=(--dependency=afterok:$DEPS --kill-on-invalid-dep=yes)
  echo "-- $label: generation $*, $ngpu GPU job(s), output $out"
  local g c a
  g=$(ROUT=$out "${SB_GPU[@]}" ${dep[@]+"${dep[@]}"} --job-name=repair-gen-$label --time=12:00:00 --array=0-$((ngpu - 1)) \
      --output="$WORK/logs/repair-gen-$label-%A_%a.out" helios/repair/job.sh generate "$ngpu" "$@")
  c=$(ROUT=$out "${SB_CPU[@]}" --dependency=afterany:$g --kill-on-invalid-dep=yes --job-name=repair-compile-$label --time=12:00:00 \
      --output="$WORK/logs/repair-compile-$label-%j.out" helios/repair/job.sh compile)
  a=$(ROUT=$out "${SB_CPU[@]}" --dependency=afterok:$c --kill-on-invalid-dep=yes --cpus-per-task=2 --mem=16G \
      --job-name=repair-analyze-$label --time=01:00:00 --output="$WORK/logs/repair-analyze-$label-%j.out" helios/repair/job.sh analyze "$label")
  echo "   generate $g (array of $ngpu), compile $c, analyze $a"
  # the next stage starts only if this stage's generation and analysis both succeeded
  DEPS="$g:$a"
}

if [ $ALL = 1 ]; then
  submit_stage smoke "$WORK/repair_smoke" 1 --limit-attempts 20 --samples 1 --arms local,back,feedback
  submit_stage step1 "$WORK/repair" 1 --samples 2 --arms local
  submit_stage step2 "$WORK/repair" 1 --samples 2 --arms local,back
  submit_stage step3 "$WORK/repair" "${NGPU3:-4}" --samples 4 --arms local,back,feedback
else
  submit_stage "$LABEL" "$ROUT" "$NGPU" $GEN_ARGS
fi

cat <<EOF

Submitted. Watch with: squeue -u \$USER     Logs: $WORK/logs/repair-*
Each stage writes report_<stage>.txt (ending in a DECISION line) and <stage>_results.tar.gz in $REPO.
Send the result archives to Szymon. If a job fails or times out, run the same command again: finished work is kept.
EOF
