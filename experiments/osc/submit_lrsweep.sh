#!/usr/bin/env bash
# submit_lrsweep.sh stage1 | stage2 | reladder | manifest <file> [walltime]
# LR sweep (Sep 13, context/reproduction_0913.md): extend the 0907 grids past their edges so the
# best LR is bracketed on both sides. Softmax + Stieltjes only; ASEntmax is out of scope here.
#   stage1  : 2 seeds x the new bracketing LRs per (task, method) — see LRS below.
#   reladder: eval-only. Re-run the ladder (Triton kernel) on the existing eager-trained Stieltjes
#             checkpoints whose ladders were capped/OOM: copy s1 5e-4/1e-3 (64x), mqmtar 2e-4 (256x/1024x).
#   stage2  : seed 3 at the top-2 LRs per cell; manifest lines live in manifests/lrsweep_stage2_<task>.txt
#             (written by hand from the stage-1 selection; see reproduction_0913.md).
#   manifest: submit an arbitrary manifest file (one "task method seed lr" per line) as one array.
# One Slurm array per task, each element runs run_one.sh (idempotent: done runs exit fast).
set -euo pipefail
source "$(dirname "$0")/env.sh"
MODE=${1:-stage1}
mkdir -p "$RESULTS_ROOT/slurm" "$RESULTS_ROOT/manifests"

# stage-1 grid: (task, method) -> LRs to add. Existing (0907) grids for reference:
#   sort/reverse {2e-4,4e-4,8e-4}  copy {5e-4,1e-3,2e-3}  mqmtar {1e-4,2e-4}
declare -A LRS=(
  [sort/softmax]="1.6e-3 3.2e-3"      [sort/stieltjes]="1.6e-3 3.2e-3"
  [reverse/softmax]="1.6e-3 3.2e-3"   [reverse/stieltjes]="1.6e-3 3.2e-3"
  [copy/softmax]="2.5e-4 4e-3"        [copy/stieltjes]="2.5e-4 4e-3"
  [mqmtar/softmax]="5e-5 4e-4"        [mqmtar/stieltjes]="4e-4 8e-4"
)
# walltime per single run (train + full ladder) on A100-40GB, with margin (stieltjes is the slow one)
declare -A WALL=( [sort]="6:00:00" [reverse]="8:00:00" [copy]="5:00:00" [mqmtar]="16:00:00" )
SEEDS="1 2"

cd "$REPO"; git diff --quiet || { echo "ERROR: dirty tree in $REPO; commit/stash first" >&2; exit 1; }
COMMIT=$(git rev-parse --short HEAD)

submit_array() {  # name task manifest walltime
  local name=$1 task=$2 man=$3 wall=$4 N
  N=$(wc -l < "$man"); [ "$N" -gt 0 ] || { echo "skip $task: empty manifest"; return; }
  jid=$(sbatch --parsable --account=PAS2836 --job-name="${name}_${task}" --time="$wall" \
    --nodes=1 --ntasks-per-node=1 --cpus-per-task=6 --gpus-per-node=1 --mem=48G \
    --array="1-${N}" --output="$RESULTS_ROOT/slurm/${name}_${task}_%A_%a.log" \
    --export=ALL,MANIFEST="$man",MAX_STEPS_OVERRIDE="" \
    "$REPO/experiments/osc/array_worker.sbatch")
  echo "submitted $name/$task: job $jid x$N  (manifest $man)"
}

TASKS=(${TASKS:-sort reverse copy mqmtar})
case $MODE in
  stage1)
    for task in "${TASKS[@]}"; do
      MAN="$RESULTS_ROOT/manifests/lrsweep_stage1_${task}_${COMMIT}.txt"; : > "$MAN"
      for m in softmax stieltjes; do for s in $SEEDS; do for lr in ${LRS[$task/$m]}; do
        echo "$task $m $s $lr" >> "$MAN"
      done; done; done
      submit_array lrs1 "$task" "$MAN" "${WALL[$task]}"
    done ;;
  reladder)
    MAN="$RESULTS_ROOT/manifests/lrsweep_reladder_copy_${COMMIT}.txt"
    printf "copy stieltjes 1 5e-4\ncopy stieltjes 1 1e-3\n" > "$MAN"; submit_array lrsrl copy "$MAN" "3:00:00"
    MAN="$RESULTS_ROOT/manifests/lrsweep_reladder_mqmtar_${COMMIT}.txt"
    printf "mqmtar stieltjes 1 2e-4\nmqmtar stieltjes 2 2e-4\n" > "$MAN"; submit_array lrsrl mqmtar "$MAN" "8:00:00" ;;
  stage2)
    for task in "${TASKS[@]}"; do
      MAN="$RESULTS_ROOT/manifests/lrsweep_stage2_${task}.txt"
      [ -f "$MAN" ] || { echo "no $MAN; write it first" >&2; continue; }
      submit_array lrs2 "$task" "$MAN" "${WALL[$task]}"
    done ;;
  manifest)
    MAN=$2; task=$(awk 'NR==1{print $1}' "$MAN")
    submit_array lrsm "$task" "$MAN" "${3:-${WALL[$task]}}" ;;
  *) echo "usage: $0 stage1|stage2|reladder|manifest <file> [walltime]" >&2; exit 1 ;;
esac
