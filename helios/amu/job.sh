#!/bin/bash
# Slurm job body for helios/amu/run.sh (do not submit by hand). Modes:
#   check            re-score the 200 check attempts and compare them with the Helios stats
#   rescore <shards> re-score shard $SLURM_ARRAY_TASK_ID of the 16,640 attempts
#   collect          verify that every attempt has stats, then pack them into amu_stats.tar.gz
set -euo pipefail
# Slurm runs a copy of this script from its spool dir, so find the repo via the submit dir.
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO"
WORK=$REPO/amu_work
DATA=amu_data/token_ids/putnam85_rest_wp_8b
source "${VENV:-$WORK/venv}/bin/activate"
export HF_HOME=${HF_HOME:-$WORK/hf_cache} HF_HUB_OFFLINE=1

gpu_info() {
  echo "host $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo none)"
  python -c 'import sys, torch; ok = torch.cuda.is_available(); print("torch sees CUDA:", ok); sys.exit(0 if ok else 1)' \
    || { echo "ERROR: PyTorch cannot use the GPU. Delete amu_work/venv and rerun run.sh with --torch-index https://download.pytorch.org/whl/cu121 (or the cuXXX matching nvidia-smi)."; exit 1; }
}

case ${1:-} in
  check)
    gpu_info
    python helios/entropy/rescore.py --ids-dir amu_data/check/ids
    python helios/amu/verify.py check amu_data/check
    touch "$WORK/check_passed"
    ;;
  rescore)
    gpu_info
    python helios/entropy/rescore.py --ids-dir "$DATA" --shard "$SLURM_ARRAY_TASK_ID" --num-shards "${2:?number of shards}"
    echo RESCORE_OK
    ;;
  collect)
    if ! python helios/amu/verify.py done "$DATA"; then
      echo "Some attempts are missing stats (see above). Run 'bash helios/amu/run.sh' again with the same options."
      exit 1
    fi
    { find amu_data -name '*.stats.npz' -not -path '*/reference/*'; echo amu_work/logs; } > "$WORK/stats_files.txt"
    tar czf amu_stats.tar.gz -T "$WORK/stats_files.txt"
    ls -lh amu_stats.tar.gz
    echo "DONE: send $REPO/amu_stats.tar.gz to Szymon."
    ;;
  *) echo "usage: job.sh check | rescore <shards> | collect"; exit 2 ;;
esac
