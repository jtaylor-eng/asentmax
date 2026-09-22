#!/usr/bin/env bash
# Local smoke test of the Stieltjes rows through the real entrypoints: ~60 train steps + eval at 128
# on the sort data, for each METHOD (default: all Stieltjes variants + entmax). The Hydra overrides
# come from the same case block as experiments/osc/run_one.sh (kept in sync by hand; diff if in doubt).
#   bash experiments/tests/smoke_stieltjes_local.sh [method ...]
set -eu
cd "$(dirname "$0")/../../synthetic"
source .venv/bin/activate
export PROJECT_ROOT="$(pwd)" DATA_PATH="$(pwd)/data" HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
DATA='${oc.env:DATA_PATH}/sort/40M-tr_32-64_vt_64-4096_vocab_32_FULL/'
METHODS=("$@"); [ ${#METHODS[@]} -eq 0 ] && METHODS=(stieltjes stieltjes_q16 wstieltjes aswstieltjes entmax)
for method in "${METHODS[@]}"; do
  case $method in
    stieltjes|stieltjes_q16|wstieltjes|aswstieltjes|asstieltjes)
      SQ=4.0; SW=0; SCALE=(++model.net.attn_scale_type=null)
      [[ $method == stieltjes_q16 ]] && SQ=16.0
      [[ $method == *wstieltjes ]] && SW=2.0
      [[ $method == as* ]] && SCALE=(++model.net.attn_scale_type=adapt-softplus-tanh ++model.net.attn_scale_proj_bias=True)
      OV=(model.net.entmax_alpha=1.0 ++model.net.attn_type=stieltjes ++model.net.stieltjes_q=$SQ ++model.net.stieltjes_window=$SW
          ++model.net.stieltjes_num_iter=30 ++model.net.stieltjes_impl=triton
          ++model.net.attn_implementation=eager ++model.net.use_fast_attn=False
          "${SCALE[@]}" ++model.net.apply_rotary=False ++model.net.apply_nape=True) ;;
    entmax)
      OV=(model.net.entmax_alpha=1.5 ++model.net.attn_implementation=eager ++model.net.use_fast_attn=True
          ++model.net.attn_scale_type=null ++model.net.apply_rotary=False ++model.net.apply_nape=True) ;;
    *) echo "unknown method $method"; exit 1 ;;
  esac
  OUT=/tmp/stieltjes_smoke_$method; rm -rf "$OUT" "logs/smoke_$method" "logs/smoke_${method}_eval"; mkdir -p "$OUT"
  python3 src/train.py experiment=entmax/sort logger=csv task_name=smoke_$method test=False seed=1 \
    model.optimizer.lr=4e-4 data.data_provider.path="$DATA" "++callbacks.model_checkpoint.dirpath=$OUT/ckpt" \
    ++trainer.max_steps=60 trainer.max_epochs=1 trainer.val_check_interval=30 ++trainer.limit_val_batches=2 \
    "${OV[@]}" > "$OUT/train.log" 2>&1 && echo "$method TRAIN OK" || { echo "$method TRAIN FAILED"; tail -40 "$OUT/train.log"; exit 1; }
  python3 src/eval.py experiment=entmax/sort logger=csv task_name=smoke_${method}_eval +seed=1 \
    data.data_provider.path="$DATA" "++data.data_provider.file_subset.test=[test_1_128]" ++trainer.limit_test_batches=2 \
    "${OV[@]}" ckpt_path="'$OUT/ckpt/last.ckpt'" > "$OUT/eval.log" 2>&1 && echo "$method EVAL OK" || { echo "$method EVAL FAILED"; tail -40 "$OUT/eval.log"; exit 1; }
  loss=$(grep -oE "train/loss_step[^,]*" $(find logs/smoke_$method -name metrics.csv | sort | tail -1) | head -1)
  python3 - "$(find logs/smoke_$method -name metrics.csv | sort | tail -1)" "$(find logs/smoke_${method}_eval -name metrics.csv | sort | tail -1)" <<'PY'
import sys, csv
def last(path, prefix):
    v = None
    for row in csv.DictReader(open(path)):
        for k, x in row.items():
            if k.startswith(prefix) and x: v = float(x)
    return v
print(f"    last train/loss_step={last(sys.argv[1], 'train/loss_step'):.4f}  test/acc_epoch@128={last(sys.argv[2], 'test/acc_epoch')}")
PY
done
