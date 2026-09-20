#!/usr/bin/env bash
# submit_variants.sh smoke | full | manifest <file> [walltime]
# Stieltjes-variant batch (Sep 19; decisions: "sound" option, guessed LRs, no LR sweep):
#   stieltjes_q16  dense q=16            3 seeds x 4 tasks x 1 LR
#   wstieltjes     windowed q=4, d=2     3 seeds x 4 tasks x 1 LR
#   aswstieltjes   windowed + AS scale   3 seeds x 4 tasks x 1 LR
#   entmax         1.5-entmax control    2 seeds x 4 tasks x 1 LR
#   asentmax       mqmtar only, lr 4e-4  3 seeds   (0907 row never left the plateau at 1e-4/2e-4)
# LRs: the swept q=4 Stieltjes optimum (0913) for the Stieltjes rows; ASEntmax's 0907 optimum for
# the entmax-family rows (they NaN'd at >= 8e-4 on sort/copy). Not bracketed: one-LR existence
# claims only; any surprising win gets a sweep before it is quoted.
#   smoke : 1500 steps, seed 1, every (task, method), 2 h walltime — kernel compile + train + ladder on the A100
#   full  : the batch above (run_one.sh is idempotent: completed runs exit fast, so re-submitting is safe)
set -euo pipefail
source "$(dirname "$0")/env.sh"
MODE=${1:-smoke}
mkdir -p "$RESULTS_ROOT/slurm" "$RESULTS_ROOT/manifests"

declare -A LR=(
  [sort/stieltjes_q16]=3.2e-3   [reverse/stieltjes_q16]=3.2e-3   [copy/stieltjes_q16]=5e-4   [mqmtar/stieltjes_q16]=4e-4
  [sort/wstieltjes]=1.6e-3      [reverse/wstieltjes]=1.6e-3      [copy/wstieltjes]=5e-4      [mqmtar/wstieltjes]=4e-4
  [sort/aswstieltjes]=4e-4      [reverse/aswstieltjes]=4e-4      [copy/aswstieltjes]=5e-4    [mqmtar/aswstieltjes]=2e-4
  [sort/entmax]=4e-4            [reverse/entmax]=4e-4            [copy/entmax]=5e-4          [mqmtar/entmax]=2e-4
  [mqmtar/asentmax]=4e-4
)
declare -A SEEDS=( [stieltjes_q16]="1 2 3" [wstieltjes]="1 2 3" [aswstieltjes]="1 2 3" [entmax]="1 2" [asentmax]="1 2 3" )
declare -A WALL=( [sort]="6:00:00" [reverse]="8:00:00" [copy]="5:00:00" [mqmtar]="16:00:00" )

cd "$REPO"; git diff --quiet || { echo "ERROR: dirty tree in $REPO; commit/stash first" >&2; exit 1; }
COMMIT=$(git rev-parse --short HEAD)

submit_array() {  # name task manifest walltime [max_steps]
  local name=$1 task=$2 man=$3 wall=$4 steps=${5:-} N
  N=$(wc -l < "$man"); [ "$N" -gt 0 ] || { echo "skip $task: empty manifest"; return; }
  jid=$(sbatch --parsable --account=PAS2836 --job-name="${name}_${task}" --time="$wall" \
    --nodes=1 --ntasks-per-node=1 --cpus-per-task=6 --gpus-per-node=1 --mem=48G \
    --array="1-${N}" --output="$RESULTS_ROOT/slurm/${name}_${task}_%A_%a.log" \
    --export=ALL,MANIFEST="$man",MAX_STEPS_OVERRIDE="$steps" \
    "$REPO/experiments/osc/array_worker.sbatch")
  echo "submitted $name/$task: job $jid x$N  (manifest $man)"
}

TASKS=(${TASKS:-sort reverse copy mqmtar})
METHODS=(${METHODS:-stieltjes_q16 wstieltjes aswstieltjes entmax asentmax})
case $MODE in
  smoke|full)
    # smoke runs live under their own results root so their TRAIN_DONE/ladder markers can never
    # short-circuit the real runs (run_one.sh keys everything on $RESULTS_ROOT/<task>/<method>/s<seed>_lr<lr>).
    [ "$MODE" = smoke ] && export RESULTS_ROOT="${RESULTS_ROOT}_smoke" && mkdir -p "$RESULTS_ROOT/slurm" "$RESULTS_ROOT/manifests"
    for task in "${TASKS[@]}"; do
      MAN="$RESULTS_ROOT/manifests/variants_${MODE}_${task}_${COMMIT}.txt"; : > "$MAN"
      for m in "${METHODS[@]}"; do
        lr=${LR[$task/$m]:-}; [ -n "$lr" ] || continue
        seeds=${SEEDS[$m]}; [ "$MODE" = smoke ] && seeds=1
        for s in $seeds; do echo "$task $m $s $lr" >> "$MAN"; done
      done
      if [ "$MODE" = smoke ]; then submit_array vsm "$task" "$MAN" "2:00:00" 1500
      else submit_array var "$task" "$MAN" "${WALL[$task]}"; fi
    done ;;
  manifest)
    MAN=$2; task=$(awk 'NR==1{print $1}' "$MAN")
    submit_array varm "$task" "$MAN" "${3:-${WALL[$task]}}" ;;
  *) echo "usage: $0 smoke|full|manifest <file> [walltime]" >&2; exit 1 ;;
esac
