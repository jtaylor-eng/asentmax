#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=t1_gpusmoke
#SBATCH --time=0:40:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --gpus-per-node=1 --mem=24G
#SBATCH --output=/fs/scratch/PAS2836/%u/asentmax_results/table1/slurm/gpusmoke_%j.log
# GPU stack smoke: tiny sort dataset -> 40 train steps + 1-batch eval for softmax(flash), asentmax(adasplash), stieltjes(eager).
set -uo pipefail
module reset >/dev/null 2>&1 || true
source /fs/scratch/PAS2836/$USER/asentmax/synthetic/osc/env.sh
source "$VENV/bin/activate"; cd "$PROJECT_ROOT"
nvidia-smi -L; python -c "import torch;print(torch.cuda.get_device_name(0), torch.version.cuda)"
export DATA_PATH=$TMPDIR/data; mkdir -p "$DATA_PATH/sort/tiny"
python3 "$REPO/scripts/generate_data.py" --task_type sort --out_dir "$DATA_PATH/sort/tiny" --train_size 20000 --dev_size 16 --test_size 16 \
  --seq_len 48 --vary_len 16 --vocab_size 32 --mdps_seq_len "64 128 256 512" --mdps_vary_len "0 0 0 0" > /dev/null
python3 "$REPO/synthetic/tests/test_stieltjes_eager.py" 2>&1 | tail -3
D='${oc.env:DATA_PATH}/sort/tiny/'
declare -A OV
OV[softmax]="model.net.entmax_alpha=1.0 ++model.net.attn_implementation=flash_attention_2 ++model.net.use_fast_attn=True ++model.net.attn_scale_type=null ++model.net.apply_rotary=False ++model.net.apply_nape=True"
OV[asentmax]="model.net.entmax_alpha=1.5 ++model.net.attn_implementation=eager ++model.net.use_fast_attn=True ++model.net.attn_scale_type=adapt-softplus-tanh ++model.net.attn_scale_proj_bias=True ++model.net.apply_rotary=False ++model.net.apply_nape=True"
OV[stieltjes]="model.net.entmax_alpha=1.0 ++model.net.attn_type=stieltjes ++model.net.stieltjes_q=4.0 ++model.net.attn_implementation=eager ++model.net.use_fast_attn=False ++model.net.attn_scale_type=null ++model.net.apply_rotary=False ++model.net.apply_nape=True"
for m in softmax asentmax stieltjes; do
  OUT=$TMPDIR/smoke_$m; mkdir -p $OUT
  t0=$(date +%s)
  python3 src/train.py experiment=entmax/sort logger=csv_wandb_offline task_name=gpusmoke_$m test=False seed=1 model.optimizer.lr=4e-4 \
    data.data_provider.path="$D" "++callbacks.model_checkpoint.dirpath=$OUT/ckpt" ++trainer.max_steps=100 trainer.max_epochs=1 \
    trainer.val_check_interval=50 ++logger.wandb.project=asentmax-table1-smoke ${OV[$m]} > $OUT/train.log 2>&1; rc=$?
  t1=$(date +%s)
  echo "[$m] train rc=$rc in $((t1-t0))s; ckpts: $(ls $OUT/ckpt 2>/dev/null | tr '\n' ' ')"
  [ $rc -ne 0 ] && { tail -25 $OUT/train.log; continue; }
  python3 src/eval.py experiment=entmax/sort logger=csv task_name=gpusmoke_eval_$m +seed=1 data.data_provider.path="$D" \
    "++data.data_provider.file_subset.test=[test_2_256]" ${OV[$m]} ckpt_path="'$OUT/ckpt/last.ckpt'" > $OUT/eval.log 2>&1; rc=$?
  echo "[$m] eval@256 rc=$rc acc=$(tail -1 $(find logs/gpusmoke_eval_$m -name metrics.csv | tail -1) | cut -d, -f4)"
  [ $rc -ne 0 ] && tail -25 $OUT/eval.log
  grep -oE "[0-9.]+it/s" $OUT/train.log | tail -1 | sed "s/^/[$m] train speed: /"
done
ls logs/gpusmoke_softmax/runs/*/wandb 2>/dev/null | head -3
echo SMOKE_DONE
