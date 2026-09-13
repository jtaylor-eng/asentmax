#!/usr/bin/env bash
# submit_impl_cmp.sh [smoke|full]
# Apples-to-apples Stieltjes implementation comparison (branch torch_triton_comp):
#   stieltjes (triton kernel) vs stieltjes_eager, sort + copy, seed 1,
#   at the Table-1 Stieltjes-selected LR per task. One Slurm array per task.
set -euo pipefail
source "$(dirname "$0")/env.sh"
MODE=${1:-full}
METHODS=(stieltjes stieltjes_eager)
declare -A LR=( [sort]="4e-4" [copy]="5e-4" )
declare -A WALL=( [sort]="6:00:00" [copy]="4:00:00" )
SEED=1; STEPS=""
if [ "$MODE" = smoke ]; then STEPS=1500; for t in "${!WALL[@]}"; do WALL[$t]="1:00:00"; done; fi
mkdir -p "$RESULTS_ROOT/slurm" "$RESULTS_ROOT/manifests"

cd "$REPO"; git diff --quiet || { echo "ERROR: dirty tree in $REPO; commit/stash first" >&2; exit 1; }
COMMIT=$(git rev-parse --short HEAD)
TASKS=(${TASKS:-sort copy})
for task in "${TASKS[@]}"; do
  MAN="$RESULTS_ROOT/manifests/implcmp_${MODE}_${task}_${COMMIT}.txt"; : > "$MAN"
  for m in "${METHODS[@]}"; do echo "$task $m $SEED ${LR[$task]}" >> "$MAN"; done
  N=$(wc -l < "$MAN")
  jid=$(sbatch --parsable --account=PAS2836 --job-name="implcmp_${MODE}_${task}" --time="${WALL[$task]}" \
    --nodes=1 --ntasks-per-node=1 --cpus-per-task=6 --gpus-per-node=1 --mem=48G \
    --array="1-${N}" --output="$RESULTS_ROOT/slurm/implcmp_${MODE}_${task}_%A_%a.log" \
    --export=ALL,MANIFEST="$MAN",MAX_STEPS_OVERRIDE="$STEPS" \
    "$REPO/experiments/osc/array_worker.sbatch")
  echo "submitted $task: job $jid x$N  (manifest $MAN)"
done
