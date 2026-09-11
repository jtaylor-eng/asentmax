#!/usr/bin/env bash
# Full reproduction of Table 1 — Reverse (L=6), Copy (L=2), MQMTAR (L=4) blocks, Softmax vs ASEntmax.
# Full paper fidelity: per-task step schedule, batch 128, NAPE positional encoding.
# Per-length eval ladder with early-stop: once a model hits 0.0 exact-match at a length,
# skip the next longer length. Test sets capped at 100 samples (data generated that way).
# NOTE: MQMTAR eval ladder is capped at 4096 (64x) to avoid OOM / multi-hour evals at 16384/65536.
#       Checkpoints (best + last) are saved per model so the longer OOD lengths can be run later.
set -u  # (NOT -e: ladder must continue past non-zero evals and handle stops itself)

cd "$(dirname "$0")/../synthetic"   # scripts relocated to repo-root experiments/; run tree is still synthetic/
source .venv/bin/activate
export PROJECT_ROOT="$(pwd)"
export DATA_PATH="$(pwd)/data"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

RESULTS_DIR="${RESULTS_DIR:-$(pwd)/results_table1_reverse_copy}"
mkdir -p "$RESULTS_DIR"
RUN_LOG="$RESULTS_DIR/run.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RUN_LOG"; }

# ---------------- per-task definitions ----------------
# Each task: data path, step budget, seed, lr, eval length stems + labels.
REV_DATA='${oc.env:DATA_PATH}/reverse/30M_tr-32-64_val_64-256_t-64-512/'
COPY_DATA='${oc.env:DATA_PATH}/copy/20M_V-32_tr-32-64_val_64-512_t-64-4096/'
MQMTAR_DATA='${oc.env:DATA_PATH}/mqmtar/50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4/'

# reverse: ID/1.5x/2x/4x/8x  = 64/96/128/256/512
REVERSE_STEMS=(test_0_64 test_1_96 test_2_128 test_3_256 test_4_512)
REVERSE_LABELS=(ID 1.5x 2x 4x 8x)
REVERSE_VALUES=(64 96 128 256 512)
# copy: ID/2x/4x/8x/16x/32x/64x = 64/128/256/512/1024/2048/4096
COPY_STEMS=(test_0_64 test_1_128 test_2_256 test_3_512 test_4_1024 test_5_2048 test_6_4096)
COPY_LABELS=(ID 2x 4x 8x 16x 32x 64x)
COPY_VALUES=(64 128 256 512 1024 2048 4096)
# mqmtar: ID/2x/4x/16x/64x = 64/128/256/1024/4096 (data also has 16384/65536 test files,
# but the ladder is CAPPED at 4096 to avoid OOM; run those separately later from saved ckpts).
MQMTAR_STEMS=(test_0_64 test_1_128 test_2_256 test_3_1024 test_4_4096)
MQMTAR_LABELS=(ID 2x 4x 16x 64x)
MQMTAR_VALUES=(64 128 256 1024 4096)

# ---- attention-method overrides (shared across tasks; NAPE) ----
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

# train_model <task> <method> <ovname>
train_model() {
  local task="$1"; local method="$2"; local -n ov=$3
  local exp="entmax/$task"
  local data lr seed steps
  case "$task" in
    reverse) data="$REV_DATA";  lr=4e-4; seed=4; steps=234375 ;;
    copy)    data="$COPY_DATA"; lr=1e-3; seed=1; steps=156250 ;;
    mqmtar)  data="$MQMTAR_DATA"; lr=2e-4; seed=1; steps=390625 ;;
    *) log "unknown task $task"; return 1 ;;
  esac
  # MQMTAR validation trimmed to 3 loaders (64/128/256) for speed+memory; monitor the largest (idx_2).
  local extra_train=()
  if [ "$task" = "mqmtar" ]; then
    extra_train+=("++callbacks.model_checkpoint.monitor=val/acc/dataloader_idx_2")
  fi
  steps="${MAX_STEPS:-$steps}"
  local tag="${task}_${method}"
  local ckpt_dir="$RESULTS_DIR/$tag/checkpoints"
  mkdir -p "$RESULTS_DIR/$tag"
  if [ -f "$RESULTS_DIR/$tag/TRAIN_DONE" ]; then
    log "[$tag] training already complete, skipping."
    return 0
  fi
  # Data-readiness guard (esp. MQMTAR, whose 50M gen may still be running): wait for the
  # train index (.idx) to exist and stop growing before starting. Bounded to ~2h of waiting.
  local data_dir; data_dir=$(python3 -c "import os;print(os.path.expandvars('$data'.replace('\${oc.env:DATA_PATH}', os.environ['DATA_PATH'])))")
  local train_src="$data_dir/train.src"
  local waited=0
  while [ ! -f "$train_src" ] && [ $waited -lt 7200 ]; do
    log "[$tag] waiting for training data at $train_src ..."; sleep 60; waited=$((waited+60))
  done
  if [ -f "$train_src" ]; then
    # wait until file size is stable for two consecutive checks (gen finished writing)
    local s1 s2
    s1=$(stat -c%s "$train_src" 2>/dev/null || echo 0); sleep 20
    s2=$(stat -c%s "$train_src" 2>/dev/null || echo 0)
    while [ "$s1" != "$s2" ] && [ $waited -lt 7200 ]; do
      log "[$tag] training data still being written ($s1 -> $s2), waiting ..."
      s1=$s2; sleep 30; waited=$((waited+30)); s2=$(stat -c%s "$train_src" 2>/dev/null || echo 0)
    done
  fi
  log "[$tag] === TRAINING START (max_steps=$steps, lr=$lr, seed=$seed) ==="
  python3 src/train.py \
    "experiment=$exp" \
    logger=csv \
    task_name="$tag" \
    test=False \
    seed=$seed \
    model.optimizer.lr=$lr \
    data.data_provider.path="$data" \
    "++callbacks.model_checkpoint.dirpath=$ckpt_dir" \
    ++trainer.max_steps=$steps \
    trainer.max_epochs=1 \
    "${extra_train[@]}" \
    "${ov[@]}" >> "$RESULTS_DIR/$tag/train.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    touch "$RESULTS_DIR/$tag/TRAIN_DONE"
    log "[$tag] === TRAINING DONE ==="
  else
    log "[$tag] !!! TRAINING FAILED rc=$rc (see $RESULTS_DIR/$tag/train.log)"
  fi
  return $rc
}

# eval_length <task> <method> <ovname> <stem>; prints accuracy 0..1 (or ERR)
eval_length() {
  local task="$1"; local method="$2"; local -n ov2=$3; local stem="$4"
  local exp="entmax/$task"
  local data seed
  case "$task" in
    reverse) data="$REV_DATA";  seed=4 ;;
    copy)    data="$COPY_DATA"; seed=1 ;;
    mqmtar)  data="$MQMTAR_DATA"; seed=1 ;;
  esac
  local tag="${task}_${method}"
  local ckpt
  ckpt=$(find "$RESULTS_DIR/$tag/checkpoints" -name "*.ckpt" ! -name "last.ckpt" | head -1)
  if [ -z "$ckpt" ]; then ckpt="$RESULTS_DIR/$tag/checkpoints/last.ckpt"; fi
  python3 src/eval.py \
    "experiment=$exp" \
    logger=csv \
    task_name="${tag}_eval_${stem}" \
    +seed=$seed \
    data.data_provider.path="$data" \
    "++data.data_provider.file_subset.test=[$stem]" \
    "${ov2[@]}" \
    ckpt_path="'$ckpt'" > "$RESULTS_DIR/$tag/eval_${stem}.log" 2>&1
  local csv
  csv=$(find "logs/${tag}_eval_${stem}" -name metrics.csv 2>/dev/null | sort | tail -1)
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
  log "[$tag] === EVAL LADDER START (early-stop on 0.0) ==="
  : > "$RESULTS_DIR/$tag/ladder.tsv"
  local stopped=0
  for i in "${!STEMS[@]}"; do
    if [ "$stopped" -eq 1 ]; then
      log "[$tag] length ${VALUES[$i]} (${LABELS[$i]}): SKIPPED (early-stop after prior 0.0)"
      printf "%s\t%s\t%s\tSKIPPED\n" "${LABELS[$i]}" "${VALUES[$i]}" "${STEMS[$i]}" >> "$RESULTS_DIR/$tag/ladder.tsv"
      continue
    fi
    local acc
    acc=$(eval_length "$task" "$method" "$ovname" "${STEMS[$i]}")
    local pct="ERR"
    if [ "$acc" != "ERR" ]; then
      pct=$(python3 -c "print(f'{float('$acc')*100:.1f}')")
    fi
    log "[$tag] length ${VALUES[$i]} (${LABELS[$i]}): exact_match = ${pct}%"
    printf "%s\t%s\t%s\t%s\n" "${LABELS[$i]}" "${VALUES[$i]}" "${STEMS[$i]}" "$pct" >> "$RESULTS_DIR/$tag/ladder.tsv"
    if [ "$acc" != "ERR" ]; then
      local iszero; iszero=$(python3 -c "print('1' if float('$acc')==0.0 else '0')")
      [ "$iszero" = "1" ] && stopped=1
    fi
  done
  log "[$tag] === EVAL LADDER DONE ==="
}

# ---------------- main ----------------
log "########## TABLE 1: REVERSE + COPY + MQMTAR REPRODUCTION ##########"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

for task in reverse copy mqmtar; do
  train_model "$task" softmax  softmax_overrides
  run_ladder  "$task" softmax  softmax_overrides
  train_model "$task" asentmax asentmax_overrides
  run_ladder  "$task" asentmax asentmax_overrides
done

# ---------------- assemble final tables ----------------
log "########## ASSEMBLING TABLES ##########"
python3 - "$RESULTS_DIR" <<'PY' | tee -a "$RUN_LOG"
import os, sys
rd = sys.argv[1]
specs = {
  "reverse": (["ID","1.5x","2x","4x","8x"],
              {"softmax":["100.0","36.0","0.0","0.0","0.0"],
               "asentmax":["100.0","100.0","99.8","96.4","56.7"]}),
  "copy":    (["ID","2x","4x","8x","16x","32x","64x"],
              {"softmax":["100.0","100.0","99.9","99.9","99.4","96.1","85.5"],
               "asentmax":["100.0","100.0","99.9","99.7","99.4","96.3","86.6"]}),
  # mqmtar ladder capped at 64x (4096); paper full row goes to 1024x. Values shown are paper's
  # for the evaluated columns only (ID/2x/4x/16x/64x).
  "mqmtar":  (["ID","2x","4x","16x","64x"],
              {"softmax":["100.0","100.0","100.0","99.5","97.8"],
               "asentmax":["100.0","100.0","100.0","99.7","99.6"]}),
}
def read(tag):
    d={}; p=os.path.join(rd,tag,"ladder.tsv")
    if os.path.exists(p):
        for line in open(p):
            parts=line.rstrip("\n").split("\t")
            if len(parts)>=4: d[parts[0]]=parts[3]
    return d
for task,(labels,paper) in specs.items():
    print(f"\n=== Table 1 — {task.capitalize()}: reproduced (this run) vs paper ===")
    def cell(v): return "skip" if v=="SKIPPED" else v
    hdr=f"{'Method':<12}"+"".join(f"{l:>8}" for l in labels)
    print(hdr); print("-"*len(hdr))
    for name in ["softmax","asentmax"]:
        d=read(f"{task}_{name}")
        print(f"{name:<12}"+"".join(f"{cell(d.get(l,'-')):>8}" for l in labels))
        print(f"{'  (paper)':<12}"+"".join(f"{v:>8}" for v in paper[name]))
print("\nNote: 'skip' = SKIPPED by early-stop rule; '-' = missing.")
PY
log "########## ALL DONE ##########"
