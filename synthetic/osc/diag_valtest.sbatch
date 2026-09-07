#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=t1_diag
#SBATCH --time=0:40:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --gpus-per-node=1 --mem=32G
#SBATCH --output=/fs/scratch/PAS2836/%u/asentmax_results/table1/slurm/diag_%j.log
# Diagnose val/test disagreement on reverse: eval one ckpt on test_3_256 with batch 4 / 32 / 1,
# and on the validation_3_256 file (via file_subset on test split with a symlinked copy).
set -uo pipefail
module reset >/dev/null 2>&1 || true
source /fs/scratch/PAS2836/$USER/asentmax/synthetic/osc/env.sh
source "$VENV/bin/activate"; cd "$PROJECT_ROOT"
RUN=$RESULTS_ROOT/reverse/asentmax/s1_lr8e-4
D=$DATA_ROOT/reverse/30M_tr-32-64_val_64-256_t-64-512
# temp dataset dir with validation_3_256 exposed as a test file
T=$TMPDIR/diagdata/reverse/x; mkdir -p $T
for f in test_3_256 test_1_96; do cp $D/$f.src $D/$f.trg $T/; done
cp $D/validation_3_256.src $T/test_9_256v.src; cp $D/validation_3_256.trg $T/test_9_256v.trg
cp $D/train.src $T/train.src; cp $D/train.trg $T/train.trg   # provider needs a train split
export DATA_PATH=$TMPDIR/diagdata
OV=(model.net.entmax_alpha=1.5 ++model.net.attn_implementation=eager ++model.net.use_fast_attn=True ++model.net.attn_scale_type=adapt-softplus-tanh ++model.net.attn_scale_proj_bias=True ++model.net.apply_rotary=False ++model.net.apply_nape=True)
for ck in "epoch=0-step=11718.ckpt" "last.ckpt"; do
for spec in "test_3_256 4" "test_3_256 32" "test_3_256 1" "test_9_256v 32" "test_9_256v 4"; do
  set -- $spec; stem=$1; bs=$2
  python3 src/eval.py experiment=entmax/reverse logger=csv task_name=diag_${ck%%.*}_${stem}_bs$bs +seed=1 \
    'data.data_provider.path=${oc.env:DATA_PATH}/reverse/x/' "++data.data_provider.file_subset.test=[$stem]" \
    data.batch_config.test.size=$bs "${OV[@]}" ckpt_path="'$RUN/checkpoints/$ck'" > $TMPDIR/e.log 2>&1
  acc=$(tail -1 $(find logs/diag_${ck%%.*}_${stem}_bs$bs -name metrics.csv | tail -1) | cut -d, -f4)
  echo "ckpt=$ck file=$stem bs=$bs acc=$acc"
done; done
echo DIAG_DONE
