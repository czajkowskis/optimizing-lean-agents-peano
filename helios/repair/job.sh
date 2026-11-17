#!/bin/bash
# Slurm job body for helios/repair/run.sh (do not submit by hand). Modes:
#   lean                  build the Lean REPL (if needed) and recompile original attempts vs. Helios verdicts
#   generate <n> [args]   generate shard $SLURM_ARRAY_TASK_ID of n (extra args go to generate.py)
#   compile               compile every generated sample
#   analyze [label]       write report_<label>.txt and <label>_results.tar.gz; for "smoke" also the smoke gate
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
    label=${2:-$(basename "$ROUT")}
    python helios/repair/analyze.py --data "$DATA" --out "$ROUT" | tee "$ROUT/report_$label.txt"
    cp "$ROUT/report_$label.txt" "$ROUT/report.txt"
    cp "$ROUT/analysis.json" "$ROUT/analysis_$label.json"
    parts=("$(basename "$ROUT")" logs); [ -d "$WORK/repair_lean_check" ] && parts+=(repair_lean_check)
    tar czf "$REPO/${label}_results.tar.gz" -C "$WORK" "${parts[@]}"
    ls -lh "$REPO/${label}_results.tar.gz"
    echo "DONE: send $REPO/${label}_results.tar.gz to Szymon."
    if [ "$label" = smoke ]; then
      # Smoke gate (exit 1 cancels later stages in --all): repair continuations must mostly close the Lean block.
      python - "$ROUT/compiled.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
rb = [r for r in rows if r["arm"] in ("local", "back")]
no_block = sum(r.get("not_compiled") == "no lean block" for r in rb)
sys_err = sum(bool(r.get("system_errors")) for r in rows)
print(f"smoke gate: {len(rows)} samples compiled or checked; LOCAL/BACK without a Lean block {no_block}/{len(rb)}; "
      f"Lean system errors {sys_err}")
ok = rb and no_block <= len(rb) / 2 and sys_err <= len(rows) / 10
print("SMOKE_OK" if ok else "SMOKE_FAILED: later stages are cancelled; send this log and the smoke archive to Szymon")
sys.exit(0 if ok else 1)
PY
    fi
    ;;
  *) echo "usage: job.sh lean | generate <n> [args] | compile | analyze"; exit 2 ;;
esac
