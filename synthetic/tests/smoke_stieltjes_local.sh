#!/usr/bin/env bash
# Local smoke test of the Stieltjes attention row: ~60 train steps + eval at 64/128 on the sort data.
set -eu
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PROJECT_ROOT="$(pwd)" DATA_PATH="$(pwd)/data" HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
DATA='${oc.env:DATA_PATH}/sort/40M-tr_32-64_vt_64-4096_vocab_32_FULL/'
OUT=/tmp/stieltjes_smoke; rm -rf "$OUT"; mkdir -p "$OUT"
OV=(model.net.entmax_alpha=1.0 ++model.net.attn_type=stieltjes ++model.net.stieltjes_q=4.0
    ++model.net.attn_implementation=eager ++model.net.use_fast_attn=False
    ++model.net.attn_scale_type=null ++model.net.apply_rotary=False ++model.net.apply_nape=True)
python3 src/train.py experiment=entmax/sort logger=csv task_name=stieltjes_smoke test=False seed=1 \
  model.optimizer.lr=4e-4 data.data_provider.path="$DATA" "++callbacks.model_checkpoint.dirpath=$OUT/ckpt" \
  ++trainer.max_steps=60 trainer.max_epochs=1 trainer.val_check_interval=30 ++trainer.limit_val_batches=2 \
  "${OV[@]}" > "$OUT/train.log" 2>&1 && echo "TRAIN OK" || { echo "TRAIN FAILED"; tail -40 "$OUT/train.log"; exit 1; }
ls "$OUT/ckpt"
python3 src/eval.py experiment=entmax/sort logger=csv task_name=stieltjes_smoke_eval +seed=1 \
  data.data_provider.path="$DATA" "++data.data_provider.file_subset.test=[test_1_128]" ++trainer.limit_test_batches=2 \
  "${OV[@]}" ckpt_path="'$OUT/ckpt/last.ckpt'" > "$OUT/eval.log" 2>&1 && echo "EVAL OK" || { echo "EVAL FAILED"; tail -40 "$OUT/eval.log"; exit 1; }
grep -h "test/acc" $(find logs/stieltjes_smoke_eval -name metrics.csv | tail -1) | head -2
grep -E "loss_step" $(find logs/stieltjes_smoke -name metrics.csv | sort | tail -1) | head -1
