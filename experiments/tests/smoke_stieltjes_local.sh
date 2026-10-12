#!/usr/bin/env bash
# Local smoke test of Table-1 method rows through the real entrypoints: ~60 train steps + eval at 128
# on the sort data, for each METHOD. The Hydra overrides are produced by the SAME case block as
# experiments/osc/run_one.sh (extracted at run time, so the two cannot drift).
#   bash experiments/tests/smoke_stieltjes_local.sh [method ...]
set -eu
RUN_ONE="$(cd "$(dirname "$0")/../osc" && pwd)/run_one.sh"
cd "$(dirname "$0")/../../synthetic"
source .venv/bin/activate
export PROJECT_ROOT="$(pwd)" DATA_PATH="$(pwd)/data" HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
DATA='${oc.env:DATA_PATH}/sort/40M-tr_32-64_vt_64-4096_vocab_32_FULL/'
METHODS=("$@"); [ ${#METHODS[@]} -eq 0 ] && METHODS=(stieltjes stieltjes_q16 wstieltjes aswstieltjes entmax)
# The method block of run_one.sh: from the "method overrides" header through the OV+= line.
METHOD_BLOCK=$(sed -n '/^# ---------------- method overrides/,/^OV+=(/p' "$RUN_ONE")
for method in "${METHODS[@]}"; do
  log() { :; }
  eval "$METHOD_BLOCK"
  OUT=/tmp/stieltjes_smoke_$method; rm -rf "$OUT" "logs/smoke_$method" "logs/smoke_${method}_eval"; mkdir -p "$OUT"
  python3 src/train.py experiment=entmax/sort logger=csv task_name=smoke_$method test=False seed=1 \
    model.optimizer.lr=4e-4 data.data_provider.path="$DATA" "++callbacks.model_checkpoint.dirpath=$OUT/ckpt" \
    ++trainer.max_steps=60 trainer.max_epochs=1 trainer.val_check_interval=30 ++trainer.limit_val_batches=2 \
    "${OV[@]}" > "$OUT/train.log" 2>&1 && echo "$method TRAIN OK" || { echo "$method TRAIN FAILED"; tail -40 "$OUT/train.log"; exit 1; }
  python3 src/eval.py experiment=entmax/sort logger=csv task_name=smoke_${method}_eval +seed=1 \
    data.data_provider.path="$DATA" "++data.data_provider.file_subset.test=[test_1_128]" ++trainer.limit_test_batches=2 \
    "${OV[@]}" ckpt_path="'$OUT/ckpt/last.ckpt'" > "$OUT/eval.log" 2>&1 && echo "$method EVAL OK" || { echo "$method EVAL FAILED"; tail -40 "$OUT/eval.log"; exit 1; }
  # echo the resolved knobs so a wrong override is visible at a glance
  grep -oE "stieltjes_q: [0-9.]+|stieltjes_window: [0-9.]+|attn_scale_type: [a-z-]+|attn_scale_zero_init: [0-9.]+|num_warmup_steps: [0-9]+|entmax_alpha: [0-9.]+" \
    "$(find logs/smoke_$method -name config_tree.log | head -1)" | sort -u | tr '\n' ' ' | sed 's/^/    /'; echo
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
