#!/usr/bin/env bash
# submit.sh [smoke|full] [methods...]
# Builds the (task, method, seed, lr) manifest and submits ONE Slurm job array per task
# (each task has its own walltime). Each array element runs osc/run_one.sh.
#   smoke: MAX_STEPS=1500, 1 seed, 1 lr, all tasks x methods, 2h walltime.
#   full : paper step budgets, seeds {1,2}, per-task LR grids (Option B: mqmtar 2 LRs).
set -euo pipefail
source "$(dirname "$0")/env.sh"
MODE=${1:-full}; shift || true
METHODS=("$@"); [ ${#METHODS[@]} -eq 0 ] && METHODS=(softmax asentmax stieltjes)
mkdir -p "$RESULTS_ROOT/slurm" "$RESULTS_ROOT/manifests"

declare -A LRS=( [sort]="2e-4 4e-4 8e-4" [reverse]="2e-4 4e-4 8e-4" [copy]="5e-4 1e-3 2e-3" [mqmtar]="1e-4 2e-4" )
# walltime per single run (train + ladder) on A100-40GB, with margin
declare -A WALL=( [sort]="6:00:00" [reverse]="8:00:00" [copy]="4:00:00" [mqmtar]="16:00:00" )
SEEDS="1 2"; STEPS=""
if [ "$MODE" = smoke ]; then
  SEEDS="1"; STEPS=1500
  for t in "${!LRS[@]}"; do LRS[$t]=$(echo ${LRS[$t]} | awk '{print $2}'); WALL[$t]="2:00:00"; done
fi

cd "$REPO"; git diff --quiet || { echo "ERROR: dirty tree in $REPO; commit/stash first" >&2; exit 1; }
COMMIT=$(git rev-parse --short HEAD)
TASKS=(${TASKS:-sort reverse copy mqmtar})
for task in "${TASKS[@]}"; do
  MAN="$RESULTS_ROOT/manifests/${MODE}_${task}_${COMMIT}.txt"; : > "$MAN"
  for m in "${METHODS[@]}"; do for s in $SEEDS; do for lr in ${LRS[$task]}; do
    echo "$task $m $s $lr" >> "$MAN"
  done; done; done
  N=$(wc -l < "$MAN")
  jid=$(sbatch --parsable --account=PAS2836 --job-name="t1_${MODE}_${task}" --time="${WALL[$task]}" \
    --nodes=1 --ntasks-per-node=1 --cpus-per-task=6 --gpus-per-node=1 --mem=48G \
    --array="1-${N}" --output="$RESULTS_ROOT/slurm/${MODE}_${task}_%A_%a.log" \
    --export=ALL,MANIFEST="$MAN",MAX_STEPS_OVERRIDE="$STEPS" \
    "$REPO/synthetic/osc/array_worker.sbatch")
  echo "submitted $task: job $jid x$N  (manifest $MAN)"
done
