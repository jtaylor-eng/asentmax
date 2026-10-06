#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=attn_tail
#SBATCH --time=3:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --gpus-per-node=1 --mem=64G
#SBATCH --output=/fs/scratch/PAS2836/%u/asentmax_results/table1/slurm/attn_tail_%j.log
# Long-context tail statistics (tail.py) for copy + mqmtar at every ladder length, best run per family
# (manifest_tail.tsv: validation pick; remaining validation ties -> best test ladder, pick=best_ladder).
# Output: $RESULTS_ROOT/attn_tail/<task>__<method>_s<seed>_lr<lr>.npz
set -uo pipefail
module reset >/dev/null 2>&1 || true
export SCRATCH=/fs/scratch/PAS2836/$USER
source $SCRATCH/asentmax/experiments/osc/env.sh
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"
# stage the two datasets to node-local NVMe (65k-token lines; random seeks on GPFS are slow)
for rel in copy/20M_V-32_tr-32-64_val_64-512_t-64-4096 mqmtar/50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4; do
  mkdir -p $TMPDIR/data/$rel && cp $DATA_ROOT/$rel/test_* $TMPDIR/data/$rel/
done
export DATA_PATH=$TMPDIR/data
MANIFEST=${1:-$REPO/experiments/attn_maps/manifest_tail.tsv}
OUT=$RESULTS_ROOT/attn_tail; mkdir -p $OUT
declare -A STEMS=(
  [copy]="test_0_64 test_1_128 test_2_256 test_3_512 test_4_1024 test_5_2048 test_6_4096"
  [mqmtar]="test_0_64 test_1_128 test_2_256 test_3_512 test_4_1024 test_5_4096 test_6_16384 test_7_65536"
)
echo "job=${SLURM_JOB_ID:-local} host=$(hostname) commit=$(git -C $REPO rev-parse --short HEAD) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
n=0; fail=0
while IFS=$'\t' read -r task family method seed lr kind primary rest; do
  [ "$task" = task ] && continue
  [ "$primary" = 1 ] || continue
  ckdir=$RESULTS_ROOT/$task/$method/s${seed}_lr${lr}/checkpoints
  if [ "$kind" = last ]; then ck=$ckdir/last.ckpt; else ck=$(find $ckdir -name "epoch=*.ckpt" | head -1); fi
  [ -f "$ck" ] || { echo "!! no ckpt for $task $method s$seed $lr ($kind)"; fail=$((fail+1)); continue; }
  echo "== $task $family $method s$seed lr$lr  $(basename $ck)"
  python3 $REPO/experiments/attn_maps/tail.py --ckpt "$ck" --task $task --method $method \
    --tag "${method}_s${seed}_lr${lr}" --stems ${STEMS[$task]} --out-dir $OUT --samples 8 \
    2>&1 | grep -E "^\[|cached|wrote|Error|error|Traceback" || { echo "!! FAILED $task $method s$seed"; fail=$((fail+1)); }
  n=$((n+1))
done < "$MANIFEST"
cp "$MANIFEST" $OUT/manifest.tsv
echo "=== DONE runs=$n failed=$fail  $(ls $OUT/*.npz | wc -l) npz, $(du -sh $OUT | cut -f1)"
