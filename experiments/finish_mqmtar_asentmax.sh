#!/usr/bin/env bash
# Finish the interrupted run: resume mqmtar_asentmax from last.ckpt to completion,
# run its eval ladder (capped at 4096), then reassemble all three Table-1 blocks
# from the existing ladder.tsv files. The other 5 models are already done and untouched.
set -u
cd "$(dirname "$0")/../synthetic"   # scripts relocated to repo-root experiments/; run tree is still synthetic/
source .venv/bin/activate
export PROJECT_ROOT="$(pwd)"
export DATA_PATH="$(pwd)/data"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

RESULTS_DIR="$(pwd)/results_table1_reverse_copy"
RUN_LOG="$RESULTS_DIR/run.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RUN_LOG"; }

MQMTAR_DATA='${oc.env:DATA_PATH}/mqmtar/50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4/'
tag="mqmtar_asentmax"
ckpt_dir="$RESULTS_DIR/$tag/checkpoints"

asentmax_overrides=(
  model.net.entmax_alpha=1.5
  ++model.net.attn_implementation=eager
  ++model.net.use_fast_attn=True
  ++model.net.apply_rotary=False
  ++model.net.apply_nape=True
  ++model.net.attn_scale_type=adapt-softplus-tanh
  ++model.net.attn_scale_proj_bias=True
)

# mqmtar ladder capped at 64x (4096)
STEMS=(test_0_64 test_1_128 test_2_256 test_3_1024 test_4_4096)
LABELS=(ID 2x 4x 16x 64x)
VALUES=(64 128 256 1024 4096)

# ---------- 1) RESUME TRAINING ----------
log "########## FINISH RUN: resume $tag ##########"
if [ -f "$RESULTS_DIR/$tag/TRAIN_DONE" ]; then
  log "[$tag] already marked done, skipping training."
else
  log "[$tag] === RESUME TRAINING (target max_steps=390625, workers=2) ==="
  # NOTE: max_epochs must be > 1 on resume. The dataset is exactly 1 epoch (50M/128=390625 batches);
  # resuming from a mid-epoch checkpoint with a non-resumable loader makes Lightning advance the
  # epoch counter, so max_epochs=1 would stop instantly. max_steps=390625 remains the true hard stop
  # (binds before epoch 2), and the skip-sampler skips the 97,660 already-trained batches.
  python3 src/train.py \
    'experiment=entmax/mqmtar' \
    logger=csv \
    task_name="$tag" \
    test=False \
    seed=1 \
    model.optimizer.lr=2e-4 \
    data.data_provider.path="$MQMTAR_DATA" \
    "++callbacks.model_checkpoint.dirpath=$ckpt_dir" \
    ++callbacks.model_checkpoint.monitor=val/acc/dataloader_idx_2 \
    data.data_loader_kwargs.num_workers=2 \
    ++trainer.max_steps=390625 \
    trainer.max_epochs=3 \
    "${asentmax_overrides[@]}" >> "$RESULTS_DIR/$tag/train.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    touch "$RESULTS_DIR/$tag/TRAIN_DONE"
    log "[$tag] === TRAINING DONE ==="
  else
    log "[$tag] !!! TRAINING FAILED AGAIN rc=$rc (see $tag/train.log) — checkpoint preserved, rerun this script to resume."
    exit $rc
  fi
fi

# ---------- 2) EVAL LADDER ----------
eval_length() {  # <stem> -> prints acc 0..1 or ERR
  local stem="$1"
  local ckpt
  ckpt=$(find "$ckpt_dir" -name "*.ckpt" ! -name "last.ckpt" | head -1)
  if [ -z "$ckpt" ]; then ckpt="$ckpt_dir/last.ckpt"; fi
  python3 src/eval.py \
    'experiment=entmax/mqmtar' \
    logger=csv \
    task_name="${tag}_eval_${stem}" \
    +seed=1 \
    data.data_provider.path="$MQMTAR_DATA" \
    "++data.data_provider.file_subset.test=[$stem]" \
    data.data_loader_kwargs.num_workers=2 \
    "${asentmax_overrides[@]}" \
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

log "[$tag] === EVAL LADDER START (early-stop on 0.0) ==="
: > "$RESULTS_DIR/$tag/ladder.tsv"
stopped=0
for i in "${!STEMS[@]}"; do
  if [ "$stopped" -eq 1 ]; then
    log "[$tag] length ${VALUES[$i]} (${LABELS[$i]}): SKIPPED (early-stop after prior 0.0)"
    printf "%s\t%s\t%s\tSKIPPED\n" "${LABELS[$i]}" "${VALUES[$i]}" "${STEMS[$i]}" >> "$RESULTS_DIR/$tag/ladder.tsv"
    continue
  fi
  acc=$(eval_length "${STEMS[$i]}")
  pct="ERR"
  [ "$acc" != "ERR" ] && pct=$(python3 -c "print(f'{float('$acc')*100:.1f}')")
  log "[$tag] length ${VALUES[$i]} (${LABELS[$i]}): exact_match = ${pct}%"
  printf "%s\t%s\t%s\t%s\n" "${LABELS[$i]}" "${VALUES[$i]}" "${STEMS[$i]}" "$pct" >> "$RESULTS_DIR/$tag/ladder.tsv"
  if [ "$acc" != "ERR" ]; then
    iszero=$(python3 -c "print('1' if float('$acc')==0.0 else '0')")
    [ "$iszero" = "1" ] && stopped=1
  fi
done
log "[$tag] === EVAL LADDER DONE ==="

# ---------- 3) REASSEMBLE ALL THREE TABLES ----------
log "########## REASSEMBLING TABLES ##########"
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
log "########## FINISH RUN COMPLETE ##########"
