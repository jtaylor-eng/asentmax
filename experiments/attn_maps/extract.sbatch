#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=attn_maps
#SBATCH --time=2:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --gpus-per-node=1 --mem=48G
#SBATCH --output=/fs/scratch/PAS2836/%u/asentmax_results/table1/slurm/attn_maps_%j.log
# Attention-map extraction for the best run per mapping family x task (manifest.tsv, from
# select_runs.py), at ID and one OOD length. Output: $RESULTS_ROOT/attn_maps/<task>__<method>_s<seed>_lr<lr>__<stem>.npz
# Usage: sbatch experiments/attn_maps/extract.sbatch [manifest.tsv]   (STEMS_<task> env vars override the lengths)
set -uo pipefail
module reset >/dev/null 2>&1 || true
export SCRATCH=/fs/scratch/PAS2836/$USER
source $SCRATCH/asentmax/experiments/osc/env.sh
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"
export DATA_PATH="$DATA_ROOT"
MANIFEST=${1:-$REPO/experiments/attn_maps/manifest.tsv}
OUT=$RESULTS_ROOT/attn_maps; mkdir -p $OUT
# ID + one OOD length per task (sort/reverse 4x: first length where the rows separate; copy/mqmtar 64x)
declare -A STEMS=(
  [sort]="${STEMS_sort:-test_0_64 test_2_256}"
  [reverse]="${STEMS_reverse:-test_0_64 test_3_256}"
  [copy]="${STEMS_copy:-test_0_64 test_6_4096}"
  [mqmtar]="${STEMS_mqmtar:-test_0_64 test_5_4096}"
)
echo "job=${SLURM_JOB_ID:-local} host=$(hostname) commit=$(git -C $REPO rev-parse --short HEAD) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
n=0; fail=0
while IFS=$'\t' read -r task family method seed lr kind primary rest; do
  [ "$task" = task ] && continue
  ckdir=$RESULTS_ROOT/$task/$method/s${seed}_lr${lr}/checkpoints
  if [ "$kind" = last ]; then ck=$ckdir/last.ckpt; else ck=$(find $ckdir -name "epoch=*.ckpt" | head -1); fi
  [ -f "$ck" ] || { echo "!! no ckpt for $task $method s$seed $lr ($kind)"; fail=$((fail+1)); continue; }
  echo "== $task $family $method s$seed lr$lr  $(basename $ck)  primary=$primary"
  python3 $REPO/experiments/attn_maps/extract.py --ckpt "$ck" --task $task --method $method \
    --tag "${method}_s${seed}_lr${lr}" --stems ${STEMS[$task]} --out-dir $OUT --samples 4 --map-samples 2 \
    2>&1 | grep -E "^\[|exists|Error|error|Traceback|no test split" || { echo "!! FAILED $task $method s$seed"; fail=$((fail+1)); }
  n=$((n+1))
done < "$MANIFEST"
cp "$MANIFEST" $OUT/manifest.tsv
echo "=== DONE runs=$n failed=$fail  $(ls $OUT/*.npz | wc -l) npz, $(du -sh $OUT | cut -f1)"
