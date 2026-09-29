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
  # DEP=afterok:<id>[:<id>...] chains this array on earlier jobs (the SBATCH_DEPENDENCY env var was
  # NOT honoured here on Sep 19; pass it explicitly).
  jid=$(sbatch --parsable --account=PAS2836 --job-name="${name}_${task}" --time="$wall" \
    ${DEP:+--dependency="$DEP"} \
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
  followup)
    # Sep 20 follow-ups (see context/variants_0919_plan.md §6):
    #  reverse resume : the 9 Stieltjes-variant runs that hit the 8 h wall at ~205k/234k steps (12 h wall)
    #  mqmtar w20k    : asentmax + aswstieltjes with 20k warmup, 3 seeds at 2e-4 (the only config that ever escaped)
    #  matched LR     : aswstieltjes at ASEntmax's actual optimum, sort 2e-4 / copy 1e-3, 3 seeds
    MAN="$RESULTS_ROOT/manifests/variants_followup_reverse_${COMMIT}.txt"; : > "$MAN"
    for s in 1 2 3; do echo "reverse stieltjes_q16 $s 3.2e-3"; echo "reverse wstieltjes $s 1.6e-3"; echo "reverse aswstieltjes $s 4e-4"; done >> "$MAN"
    submit_array varf reverse "$MAN" "12:00:00"
    MAN="$RESULTS_ROOT/manifests/variants_followup_mqmtar_${COMMIT}.txt"; : > "$MAN"
    for s in 1 2 3; do echo "mqmtar asentmax_w20k $s 2e-4"; echo "mqmtar aswstieltjes_w20k $s 2e-4"; done >> "$MAN"
    submit_array varf mqmtar "$MAN" "${WALL[mqmtar]}"
    MAN="$RESULTS_ROOT/manifests/variants_followup_sort_${COMMIT}.txt"; : > "$MAN"
    for s in 1 2 3; do echo "sort aswstieltjes $s 2e-4"; done >> "$MAN"
    submit_array varf sort "$MAN" "${WALL[sort]}"
    MAN="$RESULTS_ROOT/manifests/variants_followup_copy_${COMMIT}.txt"; : > "$MAN"
    for s in 1 2 3; do echo "copy aswstieltjes $s 1e-3"; done >> "$MAN"
    submit_array varf copy "$MAN" "${WALL[copy]}" ;;
  asfix)
    # Sep 23: ASEntmax-on-MQMTAR fix. Zero-initialised adaptive scale (asentmax_zi: scaler = 1.05 for
    # every query at step 0 instead of median 1.7 / p95 5-11x under the HF init), 3 seeds, at both
    # warmups; plus the plain-entmax control at the same LR/warmup (currently 1/2 escaped).
    MAN="$RESULTS_ROOT/manifests/asfix_mqmtar_${COMMIT}.txt"; : > "$MAN"
    for s in 1 2 3; do echo "mqmtar asentmax_zi_w20k $s 2e-4"; echo "mqmtar asentmax_zi $s 2e-4"; echo "mqmtar entmax_w20k $s 2e-4"; done >> "$MAN"
    submit_array asf mqmtar "$MAN" "${WALL[mqmtar]}" ;;
  qsweep)
    # Sep 23: q in {2, 8} for the four Stieltjes rows (q=4 exists for stieltjes/wstieltjes; run for the
    # two AS rows), 1 seed, 1 LR per (task, row). Dense/windowed at the swept q=4 optima (0913); AS rows
    # at the ASEntmax-matched values (sort 2e-4, reverse 4e-4, copy 1e-3, mqmtar 2e-4 + 20k warmup).
    declare -A QLR=(
      [sort/stieltjes]=3.2e-3    [reverse/stieltjes]=3.2e-3    [copy/stieltjes]=5e-4    [mqmtar/stieltjes]=4e-4
      [sort/wstieltjes]=1.6e-3   [reverse/wstieltjes]=1.6e-3   [copy/wstieltjes]=5e-4   [mqmtar/wstieltjes]=4e-4
      [sort/asstieltjes]=2e-4    [reverse/asstieltjes]=4e-4    [copy/asstieltjes]=1e-3  [mqmtar/asstieltjes]=2e-4
      [sort/aswstieltjes]=2e-4   [reverse/aswstieltjes]=4e-4   [copy/aswstieltjes]=1e-3 [mqmtar/aswstieltjes]=2e-4
    )
    for task in "${TASKS[@]}"; do
      MAN="$RESULTS_ROOT/manifests/qsweep_${task}_${COMMIT}.txt"; : > "$MAN"
      for row in stieltjes wstieltjes asstieltjes aswstieltjes; do
        lr=${QLR[$task/$row]}
        qs="2 8"; [[ $row == as* ]] && qs="2 4 8"
        for q in $qs; do
          m="${row}_q${q}"
          # AS rows on mqmtar: 20k warmup (10k never escaped the plateau for any AS-scaled map)
          [[ $task == mqmtar && $row == as* ]] && m="${m}_w20k"
          echo "$task $m 1 $lr" >> "$MAN"
        done
      done
      submit_array qsw "$task" "$MAN" "${WALL[$task]}"
    done ;;
  q2follow)
    # Sep 29 (reproduction_0929.md §3.2/§1): the single-seed q2 windowed cells and the ASEntmax fix, with seeds.
    #  aswstieltjes_q2     sort + mqmtar, seeds 2 3 (seed 1 exists)           -> is sort 4x 62 / mqmtar 88/45 real
    #  aswstieltjes_q2_d{1,4}  sort + mqmtar, seeds 1 2                       -> does window width matter at q2
    #  asentmax_zi_w20k at 4e-4, mqmtar, seeds 1 2 3                          -> hotter LR = earlier escape?
    MAN="$RESULTS_ROOT/manifests/q2follow_sort_${COMMIT}.txt"; : > "$MAN"
    for s in 2 3; do echo "sort aswstieltjes_q2 $s 2e-4"; done >> "$MAN"
    for d in 1 4; do for s in 1 2; do echo "sort aswstieltjes_q2_d$d $s 2e-4"; done; done >> "$MAN"
    submit_array q2f sort "$MAN" "${WALL[sort]}"
    MAN="$RESULTS_ROOT/manifests/q2follow_mqmtar_${COMMIT}.txt"; : > "$MAN"
    for s in 2 3; do echo "mqmtar aswstieltjes_q2_w20k $s 2e-4"; done >> "$MAN"
    for d in 1 4; do for s in 1 2; do echo "mqmtar aswstieltjes_q2_d${d}_w20k $s 2e-4"; done; done >> "$MAN"
    for s in 1 2 3; do echo "mqmtar asentmax_zi_w20k $s 4e-4"; done >> "$MAN"
    submit_array q2f mqmtar "$MAN" "${WALL[mqmtar]}" ;;
  *) echo "usage: $0 smoke|full|followup|asfix|qsweep|q2follow|manifest <file> [walltime]" >&2; exit 1 ;;
esac
