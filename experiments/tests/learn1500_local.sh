#!/usr/bin/env bash
# Local 1500-step learning check for the Stieltjes rows (sort data, lr 3.2e-3, 200-step warmup so
# the loss actually moves). Prints per-method wall time, loss trajectory and val acc. ~4 min/method on a 4070 Ti.
#   bash experiments/tests/learn1500_local.sh [method ...]
set -u
cd "$(dirname "$0")/../../synthetic"
source .venv/bin/activate
export PROJECT_ROOT="$(pwd)" DATA_PATH="$(pwd)/data" HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
DATA='${oc.env:DATA_PATH}/sort/40M-tr_32-64_vt_64-4096_vocab_32_FULL/'
METHODS=("$@"); [ ${#METHODS[@]} -eq 0 ] && METHODS=(stieltjes stieltjes_q16 wstieltjes aswstieltjes entmax)
for m in "${METHODS[@]}"; do
  case $m in
    entmax) OV=(model.net.entmax_alpha=1.5 ++model.net.attn_implementation=eager ++model.net.use_fast_attn=True ++model.net.attn_scale_type=null) ;;
    *) SQ=4.0; SW=0; SCALE=(++model.net.attn_scale_type=null)
       [ "$m" = stieltjes_q16 ] && SQ=16.0; [[ $m == *wstieltjes ]] && SW=2.0
       [[ $m == as* ]] && SCALE=(++model.net.attn_scale_type=adapt-softplus-tanh ++model.net.attn_scale_proj_bias=True)
       OV=(model.net.entmax_alpha=1.0 ++model.net.attn_type=stieltjes ++model.net.stieltjes_q=$SQ ++model.net.stieltjes_window=$SW
           ++model.net.stieltjes_num_iter=30 ++model.net.stieltjes_impl=triton ++model.net.attn_implementation=eager
           ++model.net.use_fast_attn=False "${SCALE[@]}") ;;
  esac
  rm -rf "logs/t1500_$m" "/tmp/t1500_$m"
  t0=$(date +%s)
  python3 src/train.py experiment=entmax/sort logger=csv task_name=t1500_$m test=False seed=1 model.optimizer.lr=3.2e-3 \
    model.scheduler.instance.num_warmup_steps=200 data.data_provider.path="$DATA" ++callbacks.model_checkpoint.dirpath=/tmp/t1500_$m \
    ++trainer.max_steps=1500 trainer.max_epochs=1 trainer.val_check_interval=500 ++trainer.limit_val_batches=5 \
    "${OV[@]}" ++model.net.apply_rotary=False ++model.net.apply_nape=True > /tmp/t1500_$m.log 2>&1
  rc=$?; wall=$(( $(date +%s) - t0 ))
  if [ $rc -ne 0 ]; then echo "$m FAILED rc=$rc"; tail -20 /tmp/t1500_$m.log; continue; fi
  python3 - "$m" "$wall" "$(find logs/t1500_$m -name metrics.csv | sort | tail -1)" <<'PY'
import sys, csv
m, wall, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
loss, val = [], []
for r in csv.DictReader(open(path)):
    if r.get("train/loss_step"): loss.append((int(r["step"]), float(r["train/loss_step"])))
    for k, v in r.items():
        if k.startswith("val/acc") and v: val.append((int(r["step"]), k, float(v)))
pts = [loss[i] for i in range(0, len(loss), max(1, len(loss) // 6))] + loss[-1:]
print(f"{m:14s} wall={wall}s  loss: " + " ".join(f"{s}:{l:.3f}" for s, l in pts))
print(f"{'':14s} val: " + " ".join(f"{s}:{k.split('/')[-1]}={v:.3f}" for s, k, v in val))
PY
done
