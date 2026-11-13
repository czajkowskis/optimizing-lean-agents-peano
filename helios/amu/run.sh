#!/bin/bash
# One command to re-score the saved prover attempts on a Slurm GPU cluster (see helios/amu/README.md):
#
#   bash helios/amu/run.sh [--partition P] [--account A] [--gres gpu:1] [--gpus 4] [--torch-index URL]
#
# Sets up a Python venv and the model (once), then submits three chained Slurm jobs:
#   1. check:   re-score 200 attempts already scored on Helios and compare (stops here if they differ)
#   2. rescore: an array of --gpus one-GPU jobs over the 16,640 attempts
#   3. collect: verify every attempt has stats and pack them into amu_stats.tar.gz
# Safe to run again at any time: finished setup steps, checks and files are skipped.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

PARTITION="" ACCOUNT="" GRES="gpu:1" NGPU=4 TORCH_INDEX=""
while [ $# -gt 0 ]; do
  case $1 in
    --partition) PARTITION=$2; shift 2 ;;
    --account) ACCOUNT=$2; shift 2 ;;
    --gres) GRES=$2; shift 2 ;;
    --gpus) NGPU=$2; shift 2 ;;
    --torch-index) TORCH_INDEX=$2; shift 2 ;;
    -h|--help) sed -n 2,11p "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)"; exit 2 ;;
  esac
done

WORK=$REPO/amu_work
DATA=amu_data/token_ids/putnam85_rest_wp_8b
export VENV=${VENV:-$WORK/venv}
export HF_HOME=${HF_HOME:-$WORK/hf_cache}
mkdir -p "$WORK/logs"
echo '*' > "$WORK/.gitignore"
say() { printf '\n== %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

say "checks"
command -v sbatch >/dev/null || die "sbatch not found: run this on the cluster's login node"
[ "$(ls "$DATA" | grep -vc stats)" = 16640 ] || die "expected 16640 attempts in $DATA; is the clone complete?"
df -h "$WORK" | tail -1 | awk '{print "free disk here: " $4 " (need about 30G for the venv and the model)"}'

say "python environment ($VENV)"
if [ ! -x "$VENV/bin/python" ]; then
  PY=${PYTHON:-}
  if [ -z "$PY" ]; then
    for c in python3.13 python3.12 python3.11 python3.10 python3; do
      if command -v $c >/dev/null && $c -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then PY=$c; break; fi
    done
  fi
  if [ -z "$PY" ] && command -v module >/dev/null; then
    m=$(module -t avail python 2>&1 | grep -iE '^python/3\.1[0-9]' | sort -V | tail -1 || true)
    [ -n "$m" ] && { echo "loading module $m"; module load "$m"; PY=python3; }
  fi
  [ -n "$PY" ] || die "no Python >= 3.10 found; rerun with PYTHON=/path/to/python3.10+ bash helios/amu/run.sh"
  echo "using $($PY --version) at $(command -v $PY)"
  $PY -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q torch ${TORCH_INDEX:+--index-url "$TORCH_INDEX"}
  "$VENV/bin/pip" install -q "transformers==5.9.0" numpy safetensors huggingface_hub
fi
"$VENV/bin/python" -c 'import torch, transformers; print("torch", torch.__version__, "| transformers", transformers.__version__)' \
  || die "the venv is broken; delete $VENV and run again"

say "model (Goedel-LM/Goedel-Prover-V2-8B, about 16 GB, into $HF_HOME)"
"$VENV/bin/python" -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Goedel-LM/Goedel-Prover-V2-8B"))' \
  || die "model download failed; does the login node have internet?"

say "slurm settings"
if [ -z "$PARTITION" ]; then
  PARTITION=$(sinfo -h -o '%P %G' | awk 'tolower($2) ~ /gpu/ {gsub(/\*/, "", $1); print $1; exit}')
  [ -n "$PARTITION" ] || die "could not find a GPU partition; rerun with --partition <name> (see sinfo -s)"
  echo "no --partition given, using the first partition with GPUs: $PARTITION"
fi
echo "partition $PARTITION, gres $GRES, account ${ACCOUNT:-(your default)}, $NGPU GPUs for the main run"
SB=(sbatch --parsable --partition="$PARTITION" --gres="$GRES" --cpus-per-task=8 --mem=64G --export=ALL ${ACCOUNT:+--account="$ACCOUNT"})

if "$VENV/bin/python" helios/amu/verify.py done "$DATA" >/dev/null 2>&1; then
  say "all attempts already have stats: packing only"
  bash helios/amu/job.sh collect
  exit 0
fi

say "submitting"
if squeue -h -u "${USER:-$(whoami)}" -o '%j' 2>/dev/null | grep -q '^rescore'; then
  squeue -u "${USER:-$(whoami)}" -o '%.12i %.16j %.10T %.12M %R' | grep -E 'JOBID|rescore'
  die "rescore jobs from an earlier run are still queued or running; wait for them (or scancel them) and run again"
fi
DEP=()
if [ -f "$WORK/check_passed" ]; then
  echo "check already passed earlier: skipping it"
else
  J1=$("${SB[@]}" --job-name=rescore-check --time=01:00:00 --output="$WORK/logs/check-%j.out" helios/amu/job.sh check)
  echo "check job:   $J1"
  DEP=(--dependency=afterok:$J1 --kill-on-invalid-dep=yes)
fi
J2=$("${SB[@]}" "${DEP[@]}" --job-name=rescore --time=06:00:00 --array=0-$((NGPU - 1)) \
     --output="$WORK/logs/rescore-%A_%a.out" helios/amu/job.sh rescore "$NGPU")
echo "rescore job: $J2 (array of $NGPU)"
J3=$("${SB[@]}" --dependency=afterany:$J2 --kill-on-invalid-dep=yes --job-name=rescore-collect --time=00:30:00 \
     --output="$WORK/logs/collect-%j.out" helios/amu/job.sh collect)
echo "collect job: $J3"

cat <<EOF

Submitted. Nothing else to do until the jobs finish (usually a few hours, plus queue time).
  watch:   squeue -u \$USER
  logs:    ls $WORK/logs/
  result:  $REPO/amu_stats.tar.gz appears when everything is done. Send that file to Szymon.
If a job fails or times out, just run the same command again: finished work is skipped.
EOF
