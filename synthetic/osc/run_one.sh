#!/usr/bin/env bash
# run_one.sh <task> <method> <seed> <lr> [MAX_STEPS]
# Train one model with the paper protocol, then run the OOD eval ladder from the
# best-by-validation checkpoint (early-stop: once a length scores exactly 0.0, skip the rest).
# Idempotent: TRAIN_DONE / ladder.tsv markers let a requeued job resume.
set -uo pipefail
source "$(dirname "$0")/env.sh"
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"

task=$1; method=$2; seed=$3; lr=$4; MAX_STEPS_OVERRIDE=${5:-}
tag="${task}/${method}/s${seed}_lr${lr}"
RUN="$RESULTS_ROOT/$tag"
mkdir -p "$RUN"
LOG="$RUN/run.log"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== run_one $tag on $(hostname) job=${SLURM_JOB_ID:-local} commit=$(git -C "$REPO" rev-parse --short HEAD)"

# ---------------- per-task settings (paper Table 5 / Appendix G) ----------------
case $task in
  sort)    DATA='${oc.env:DATA_PATH}/sort/40M-tr_32-64_vt_64-4096_vocab_32/';           STEPS=312500
           STEMS=(test_0_64 test_1_128 test_2_256 test_3_512); LABELS=(ID 2x 4x 8x) ;;
  reverse) DATA='${oc.env:DATA_PATH}/reverse/30M_tr-32-64_val_64-256_t-64-512/';        STEPS=234375
           STEMS=(test_0_64 test_1_96 test_2_128 test_3_256 test_4_512); LABELS=(ID 1.5x 2x 4x 8x) ;;
  copy)    DATA='${oc.env:DATA_PATH}/copy/20M_V-32_tr-32-64_val_64-512_t-64-4096/';     STEPS=156250
           STEMS=(test_0_64 test_1_128 test_2_256 test_3_512 test_4_1024 test_5_2048 test_6_4096); LABELS=(ID 2x 4x 8x 16x 32x 64x) ;;
  mqmtar)  DATA='${oc.env:DATA_PATH}/mqmtar/50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4/'; STEPS=390625
           STEMS=(test_0_64 test_1_128 test_2_256 test_4_1024 test_5_4096 test_6_16384 test_7_65536); LABELS=(ID 2x 4x 16x 64x 256x 1024x) ;;
  *) log "unknown task $task"; exit 1 ;;
esac
[ -n "$MAX_STEPS_OVERRIDE" ] && STEPS=$MAX_STEPS_OVERRIDE

# ---------------- method overrides (all NAPE) ----------------
case $method in
  softmax)   OV=(model.net.entmax_alpha=1.0 ++model.net.attn_implementation=flash_attention_2 ++model.net.use_fast_attn=True
                 ++model.net.attn_scale_type=null ++model.net.apply_rotary=False ++model.net.apply_nape=True) ;;
  asentmax)  OV=(model.net.entmax_alpha=1.5 ++model.net.attn_implementation=eager ++model.net.use_fast_attn=True
                 ++model.net.attn_scale_type=adapt-softplus-tanh ++model.net.attn_scale_proj_bias=True
                 ++model.net.apply_rotary=False ++model.net.apply_nape=True) ;;
  stieltjes) OV=(model.net.entmax_alpha=1.0 ++model.net.attn_type=stieltjes ++model.net.stieltjes_q=4.0 ++model.net.stieltjes_num_iter=30
                 ++model.net.attn_implementation=eager ++model.net.use_fast_attn=False
                 ++model.net.attn_scale_type=null ++model.net.apply_rotary=False ++model.net.apply_nape=True) ;;
  asstieltjes) OV=(model.net.entmax_alpha=1.0 ++model.net.attn_type=stieltjes ++model.net.stieltjes_q=4.0 ++model.net.stieltjes_num_iter=30
                 ++model.net.attn_implementation=eager ++model.net.use_fast_attn=False
                 ++model.net.attn_scale_type=adapt-softplus-tanh ++model.net.attn_scale_proj_bias=True
                 ++model.net.apply_rotary=False ++model.net.apply_nape=True) ;;
  *) log "unknown method $method"; exit 1 ;;
esac
# Stieltjes eager attention can't prefill 16k/65k in memory (O(N^2)); cap its ladder at 4096.
if [[ $method == *stieltjes ]] && [ "$task" = mqmtar ]; then
  STEMS=(test_0_64 test_1_128 test_2_256 test_4_1024 test_5_4096); LABELS=(ID 2x 4x 16x 64x)
fi

CKPT_DIR="$RUN/checkpoints"
# ---------------- 1) train ----------------
if [ -f "$RUN/TRAIN_DONE" ]; then
  log "training already done"
else
  RESUME=()
  if [ -f "$CKPT_DIR/last.ckpt" ]; then
    # resume: dataset is exactly one epoch; Lightning advances the epoch counter on a
    # mid-epoch resume with a non-resumable loader, so allow max_epochs>1 (max_steps still binds)
    RESUME=(trainer.max_epochs=3); log "resuming from $CKPT_DIR/last.ckpt"
  else
    RESUME=(trainer.max_epochs=1)
  fi
  log "TRAIN start steps=$STEPS lr=$lr seed=$seed"
  SMOKE_EXTRA=()
  if [ -n "$MAX_STEPS_OVERRIDE" ]; then
    # short smoke run: validate a few times so the checkpoint callback fires
    SMOKE_EXTRA=(trainer.val_check_interval=$((STEPS / 3)) ++trainer.limit_val_batches=2)
  fi
  python3 src/train.py "experiment=entmax/$task" logger=csv_wandb_offline task_name="t1_${task}_${method}_s${seed}_lr${lr}" \
    test=False seed=$seed model.optimizer.lr=$lr data.data_provider.path="$DATA" \
    "++callbacks.model_checkpoint.dirpath=$CKPT_DIR" ++trainer.max_steps=$STEPS "${RESUME[@]}" "${SMOKE_EXTRA[@]}" \
    "++logger.wandb.project=asentmax-table1" "++logger.wandb.name=${task}-${method}-s${seed}-lr${lr}" \
    "++logger.wandb.group=${task}-${method}" "++logger.wandb.tags=[${task},${method},seed${seed}]" \
    "${OV[@]}" >> "$RUN/train.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then log "TRAIN FAILED rc=$rc"; exit $rc; fi
  touch "$RUN/TRAIN_DONE"; log "TRAIN done"
fi

# ---------------- 2) pick best checkpoint (paper protocol: best on validation monitor) ----------------
BEST=$(find "$CKPT_DIR" -name "*.ckpt" ! -name last.ckpt | head -1)
if [ -z "$BEST" ]; then
  if [ -f "$CKPT_DIR/last.ckpt" ]; then BEST="$CKPT_DIR/last.ckpt"; log "WARNING: no best ckpt, using last"
  else log "ERROR: no checkpoint in $CKPT_DIR"; exit 3; fi
fi
log "eval ckpt: $BEST"
echo "$BEST" > "$RUN/best_ckpt.txt"

# ---------------- 3) OOD ladder with early-stop on exact 0.0 ----------------
LADDER="$RUN/ladder.tsv"
if [ -f "$LADDER" ] && [ "$(wc -l < "$LADDER")" -ge "${#STEMS[@]}" ] && ! grep -q ERR "$LADDER"; then log "ladder already done"; else
: > "$LADDER"
stopped=0
for i in "${!STEMS[@]}"; do
  stem=${STEMS[$i]}; label=${LABELS[$i]}; n=${stem##*_}
  if [ $stopped -eq 1 ]; then
    printf "%s\t%s\t%s\tSKIPPED\n" "$label" "$n" "$stem" >> "$LADDER"; log "  $label ($n): SKIPPED"; continue
  fi
  # mqmtar at >=16384: batch 1 to bound KV/prefill memory
  EXTRA=(); if [ "$n" -ge 16384 ]; then EXTRA=(data.batch_config.test.size=1); fi
  python3 src/eval.py "experiment=entmax/$task" logger=csv task_name="t1e_${task}_${method}_s${seed}_lr${lr}_${stem}" \
    +seed=$seed data.data_provider.path="$DATA" "++data.data_provider.file_subset.test=[$stem]" \
    "${EXTRA[@]}" "${OV[@]}" ckpt_path="'$BEST'" > "$RUN/eval_${stem}.log" 2>&1
  csv=$(find "$PROJECT_ROOT/logs/t1e_${task}_${method}_s${seed}_lr${lr}_${stem}" -name metrics.csv 2>/dev/null | sort | tail -1)
  acc=$(python3 - "$csv" <<'PY'
import sys, csv
acc=None
try:
    for row in csv.DictReader(open(sys.argv[1])):
        for k,v in row.items():
            if k.startswith("test/acc_epoch") and v: acc=float(v)
except Exception: pass
print("ERR" if acc is None else f"{acc*100:.1f}")
PY
)
  printf "%s\t%s\t%s\t%s\n" "$label" "$n" "$stem" "$acc" >> "$LADDER"; log "  $label ($n): $acc%"
  if [ "$acc" = "ERR" ]; then log "ERROR: eval failed at $stem (see $RUN/eval_${stem}.log)"; exit 4; fi
  [ "$acc" = "0.0" ] && stopped=1
done
fi
log "=== DONE $tag"
