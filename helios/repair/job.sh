#!/bin/bash
# Slurm job body for helios/repair/run.sh (do not submit by hand). Modes:
#   lean                  build the Lean REPL (if needed) and recompile original attempts vs. Helios verdicts
#   generate <n> [args]   generate shard $SLURM_ARRAY_TASK_ID of n (extra args go to generate.py)
#   compile               compile every generated sample
#   analyze               write the report and pack the results
set -euo pipefail
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO"
WORK=$REPO/amu_work
ROUT=${ROUT:?run through helios/repair/run.sh}
source "${RVENV:-$WORK/venv_repair}/bin/activate"
export HF_HOME=${HF_HOME:-$WORK/hf_cache} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH=$HOME/.elan/bin:$PATH
DATA=helios/repair/data

case ${1:-} in
  lean)
    (cd mathlib4 && lake build repl 2>&1 | tail -3) || (cd mathlib4 && lake build 2>&1 | tail -3)
    python helios/repair/compile.py check --data "$DATA" --out "$WORK/repair_lean_check" --workers "${SLURM_CPUS_PER_TASK:-32}"
    touch "$WORK/lean_check_passed"
    ;;
  generate)
    shift; n=${1:?number of shards}; shift
    echo "host $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo none)"
    python helios/repair/generate.py --data "$DATA" --out "$ROUT" --shard "$SLURM_ARRAY_TASK_ID" --nshards "$n" "$@"
    ;;
  compile)
    python helios/repair/compile.py repairs --data "$DATA" --out "$ROUT" --workers "${SLURM_CPUS_PER_TASK:-32}"
    ;;
  analyze)
    python helios/repair/analyze.py --data "$DATA" --out "$ROUT" | tee "$ROUT/report.txt"
    name=$(basename "$ROUT")
    parts=("$name" logs); [ -d "$WORK/repair_lean_check" ] && parts+=(repair_lean_check)
    tar czf "$REPO/${name}_results.tar.gz" -C "$WORK" "${parts[@]}"
    ls -lh "$REPO/${name}_results.tar.gz"
    echo "DONE: send $REPO/${name}_results.tar.gz to Szymon."
    ;;
  *) echo "usage: job.sh lean | generate <n> [args] | compile | analyze"; exit 2 ;;
esac
