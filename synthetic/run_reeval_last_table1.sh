#!/usr/bin/env bash
# Re-evaluate Table 1 from the FULLY-TRAINED last.ckpt for all 8 model x task combos,
# under ONE consistent protocol (checkpoint-selection fix). Preserves prior best-ckpt
# ladder.tsv files; writes new results to ladder_last.tsv alongside them.
#
# Per-length eval ladder with early-stop: once a model scores EXACTLY 0.0 at a length,
# skip all longer lengths. Test sets are 100 samples. MQMTAR ladder capped at 4096 (64x).
set -u  # NOT -e: ladder must continue past non-zero evals and handle stops itself.

cd "$(dirname "$0")"
source .venv/bin/activate
export PROJECT_ROOT="$(pwd)"
export DATA_PATH="$(pwd)/data"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

RUN_LOG="$(pwd)/reeval_last.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RUN_LOG"; }

# ---------------- per-task data paths (literal, with ${oc.env:DATA_PATH}) ----------------
SORT_DATA='${oc.env:DATA_PATH}/sort/40M-tr_32-64_vt_64-4096_vocab_32_FULL/'
REVERSE_DATA='${oc.env:DATA_PATH}/reverse/30M_tr-32-64_val_64-256_t-64-512/'
COPY_DATA='${oc.env:DATA_PATH}/copy/20M_V-32_tr-32-64_val_64-512_t-64-4096/'
MQMTAR_DATA='${oc.env:DATA_PATH}/mqmtar/50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4/'

# model dir (relative) per tag
declare -A MODELDIR=(
  [sort_softmax]="results_table1_sort/softmax"
  [sort_asentmax]="results_table1_sort/asentmax"
  [reverse_softmax]="results_table1_reverse_copy/reverse_softmax"
  [reverse_asentmax]="results_table1_reverse_copy/reverse_asentmax"
  [copy_softmax]="results_table1_reverse_copy/copy_softmax"
  [copy_asentmax]="results_table1_reverse_copy/copy_asentmax"
  [mqmtar_softmax]="results_table1_reverse_copy/mqmtar_softmax"
  [mqmtar_asentmax]="results_table1_reverse_copy/mqmtar_asentmax"
)

# eval ladders (stems / labels / values) per task
SORT_STEMS=(test_0_64 test_1_128 test_2_256 test_3_512)
SORT_LABELS=(ID 2x 4x 8x); SORT_VALUES=(64 128 256 512)
REVERSE_STEMS=(test_0_64 test_1_96 test_2_128 test_3_256 test_4_512)
REVERSE_LABELS=(ID 1.5x 2x 4x 8x); REVERSE_VALUES=(64 96 128 256 512)
COPY_STEMS=(test_0_64 test_1_128 test_2_256 test_3_512 test_4_1024 test_5_2048 test_6_4096)
COPY_LABELS=(ID 2x 4x 8x 16x 32x 64x); COPY_VALUES=(64 128 256 512 1024 2048 4096)
MQMTAR_STEMS=(test_0_64 test_1_128 test_2_256 test_3_1024 test_4_4096)
MQMTAR_LABELS=(ID 2x 4x 16x 64x); MQMTAR_VALUES=(64 128 256 1024 4096)

# ---- attention-method overrides (NAPE) ----
softmax_overrides=(
  model.net.entmax_alpha=1.0
  ++model.net.attn_implementation=flash_attention_2
  ++model.net.use_fast_attn=True
  ++model.net.attn_scale_type=null
  ++model.net.apply_rotary=False
  ++model.net.apply_nape=True
)
asentmax_overrides=(
  model.net.entmax_alpha=1.5
  ++model.net.attn_implementation=eager
  ++model.net.use_fast_attn=True
  ++model.net.apply_rotary=False
  ++model.net.apply_nape=True
  ++model.net.attn_scale_type=adapt-softplus-tanh
  ++model.net.attn_scale_proj_bias=True
)

# eval_length <task> <method> <ovname> <stem> ; prints acc 0..1 (or ERR)
eval_length() {
  local task="$1"; local method="$2"; local -n ov2=$3; local stem="$4"
  local exp="entmax/$task"
  local tag="${task}_${method}"
  local data seed up
  up=${task^^}
  local dref="${up}_DATA"; data="${!dref}"
  case "$task" in
    sort)    seed=1 ;;
    reverse) seed=4 ;;
    copy)    seed=1 ;;
    mqmtar)  seed=1 ;;
  esac
  local ckpt="$PROJECT_ROOT/${MODELDIR[$tag]}/checkpoints/last.ckpt"
  if [ ! -f "$ckpt" ]; then echo "ERR"; return; fi
  local extra=()
  if [ "$task" = "mqmtar" ]; then extra+=(data.data_loader_kwargs.num_workers=2); fi
  python3 src/eval.py \
    "experiment=$exp" \
    logger=csv \
    task_name="${tag}_reeval_${stem}" \
    +seed=$seed \
    data.data_provider.path="$data" \
    "++data.data_provider.file_subset.test=[$stem]" \
    "${ov2[@]}" \
    "${extra[@]}" \
    ckpt_path="'$ckpt'" > "$PROJECT_ROOT/${MODELDIR[$tag]}/reeval_${stem}.log" 2>&1
  local csv
  csv=$(find "logs/${tag}_reeval_${stem}" -name metrics.csv 2>/dev/null | sort | tail -1)
  python3 - "$csv" <<'PY'
import sys, csv
path = sys.argv[1] if len(sys.argv) > 1 else ""
acc = None
if path:
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                for k, v in row.items():
                    if k.startswith("test/acc_epoch") and v not in ("", None):
                        acc = float(v)
    except Exception:
        pass
print(acc if acc is not None else "ERR")
PY
}

# run_ladder <task> <method> <ovname>
run_ladder() {
  local task="$1"; local method="$2"; local ovname="$3"
  local tag="${task}_${method}"
  local up=${task^^}
  local -n STEMS="${up}_STEMS"; local -n LABELS="${up}_LABELS"; local -n VALUES="${up}_VALUES"
  local out="$PROJECT_ROOT/${MODELDIR[$tag]}/ladder_last.tsv"
  log "[$tag] === REEVAL LADDER (last.ckpt, early-stop on 0.0) ==="
  : > "$out"
  local stopped=0
  for i in "${!STEMS[@]}"; do
    if [ "$stopped" -eq 1 ]; then
      log "[$tag] len ${VALUES[$i]} (${LABELS[$i]}): SKIPPED (early-stop)"
      printf "%s\t%s\t%s\tSKIPPED\n" "${LABELS[$i]}" "${VALUES[$i]}" "${STEMS[$i]}" >> "$out"
      continue
    fi
    local acc; acc=$(eval_length "$task" "$method" "$ovname" "${STEMS[$i]}")
    local pct="ERR"
    if [ "$acc" != "ERR" ]; then pct=$(python3 -c "print(f'{float('$acc')*100:.1f}')"); fi
    log "[$tag] len ${VALUES[$i]} (${LABELS[$i]}): exact_match = ${pct}%"
    printf "%s\t%s\t%s\t%s\n" "${LABELS[$i]}" "${VALUES[$i]}" "${STEMS[$i]}" "$pct" >> "$out"
    if [ "$acc" != "ERR" ]; then
      local iszero; iszero=$(python3 -c "print('1' if float('$acc')==0.0 else '0')")
      [ "$iszero" = "1" ] && stopped=1
    fi
  done
  log "[$tag] === LADDER DONE ($out) ==="
}

# ---------------- main ----------------
log "########## TABLE 1 RE-EVAL FROM last.ckpt (all 8 models) ##########"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Order: cheapest/highest-value first. reverse_softmax is FIX 1 (should jump to ~100 ID).
for spec in \
  "reverse softmax softmax_overrides" \
  "sort softmax softmax_overrides" \
  "sort asentmax asentmax_overrides" \
  "reverse asentmax asentmax_overrides" \
  "copy softmax softmax_overrides" \
  "copy asentmax asentmax_overrides" \
  "mqmtar softmax softmax_overrides" \
  "mqmtar asentmax asentmax_overrides" ; do
  set -- $spec
  run_ladder "$1" "$2" "$3"
done

log "########## RE-EVAL COMPLETE ##########"
