#!/usr/bin/env bash
# Overnight reproduction of Table 1 — Sort (L=2) block, Softmax vs ASEntmax.
# Full paper fidelity: 312,500-step schedule each, batch 128, NAPE positional encoding.
# Per-length eval ladder with early-stop: once a model hits 0.0 exact-match at a length,
# skip the next longer length. Test sets capped at 100 samples (data was generated that way).
set -u  # (NOT -e: we want the ladder to continue past a non-zero eval and handle stops ourselves)

cd "$(dirname "$0")"
source .venv/bin/activate
export PROJECT_ROOT="$(pwd)"
export DATA_PATH="$(pwd)/data"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

DATA_REL='${oc.env:DATA_PATH}/sort/40M-tr_32-64_vt_64-4096_vocab_32_FULL/'
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/results_table1_sort}"
mkdir -p "$RESULTS_DIR"
RUN_LOG="$RESULTS_DIR/run.log"

# length index -> (file stem, multiplier label). Sort in Table 1 goes ID/2x/4x/8x.
LEN_STEMS=(test_0_64 test_1_128 test_2_256 test_3_512)
LEN_LABELS=(ID 2x 4x 8x)
LEN_VALUES=(64 128 256 512)

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RUN_LOG"; }

# ---- method definitions (Hydra overrides beyond the shared experiment=entmax/sort) ----
# Shared: NAPE, seed=4, lr=4e-4 (paper's sort recipe).
softmax_overrides=(
  model.net.entmax_alpha=1.0
  ++model.net.attn_implementation=flash_attention_2
  ++model.net.attn_scale_type=null
  ++model.net.apply_rotary=False
  ++model.net.apply_nape=True
)
asentmax_overrides=(
  model.net.entmax_alpha=1.5
  ++model.net.apply_rotary=False
  ++model.net.apply_nape=True
  ++model.net.attn_scale_type=adapt-softplus-tanh
  ++model.net.attn_scale_proj_bias=True
)

MAX_STEPS="${MAX_STEPS:-312500}"

train_model() {
  local name="$1"; shift
  local -n ov=$1
  local ckpt_dir="$RESULTS_DIR/$name/checkpoints"
  if [ -f "$ckpt_dir/last.ckpt" ] && [ -f "$RESULTS_DIR/$name/TRAIN_DONE" ]; then
    log "[$name] training already complete, skipping."
    return 0
  fi
  log "[$name] === TRAINING START (max_steps=$MAX_STEPS) ==="
  python3 src/train.py \
    'experiment=entmax/sort' \
    logger=csv \
    task_name="$name" \
    test=False \
    seed=4 \
    model.optimizer.lr=4e-4 \
    data.data_provider.path="$DATA_REL" \
    "++callbacks.model_checkpoint.dirpath=$ckpt_dir" \
    ++trainer.max_steps=$MAX_STEPS \
    trainer.max_epochs=1 \
    "${ov[@]}" >> "$RESULTS_DIR/$name/train.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    touch "$RESULTS_DIR/$name/TRAIN_DONE"
    log "[$name] === TRAINING DONE ==="
  else
    log "[$name] !!! TRAINING FAILED rc=$rc (see $RESULTS_DIR/$name/train.log)"
  fi
  return $rc
}

# eval one (model, length-index); prints the exact-match accuracy (0..1) to stdout
eval_length() {
  local name="$1"; local idx="$2"; local -n ov2=$3
  local stem="${LEN_STEMS[$idx]}"
  local ckpt
  ckpt=$(find "$RESULTS_DIR/$name/checkpoints" -name "*.ckpt" ! -name "last.ckpt" | head -1)
  if [ -z "$ckpt" ]; then ckpt="$RESULTS_DIR/$name/checkpoints/last.ckpt"; fi
  local elog="$RESULTS_DIR/$name/eval_${stem}.log"
  python3 src/eval.py \
    'experiment=entmax/sort' \
    logger=csv \
    task_name="${name}_eval_${stem}" \
    +seed=4 \
    data.data_provider.path="$DATA_REL" \
    "++data.data_provider.file_subset.test=[$stem]" \
    "${ov2[@]}" \
    ckpt_path="'$ckpt'" > "$elog" 2>&1
  # pull test/acc_epoch from the eval CSV (newest run subdir wins)
  local csv
  csv=$(find "logs/${name}_eval_${stem}" -name metrics.csv 2>/dev/null | sort | tail -1)
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

run_ladder() {
  local name="$1"; local -n ov3=$2
  log "[$name] === EVAL LADDER START (early-stop on 0.0) ==="
  : > "$RESULTS_DIR/$name/ladder.tsv"
  local stopped=0
  for i in "${!LEN_STEMS[@]}"; do
    if [ "$stopped" -eq 1 ]; then
      log "[$name] length ${LEN_VALUES[$i]} (${LEN_LABELS[$i]}): SKIPPED (early-stop after prior 0.0)"
      printf "%s\t%s\t%s\tSKIPPED\n" "${LEN_LABELS[$i]}" "${LEN_VALUES[$i]}" "${LEN_STEMS[$i]}" >> "$RESULTS_DIR/$name/ladder.tsv"
      continue
    fi
    local acc
    acc=$(eval_length "$name" "$i" ov3)
    local pct="ERR"
    if [ "$acc" != "ERR" ]; then
      pct=$(python3 -c "print(f'{float('$acc')*100:.1f}')")
    fi
    log "[$name] length ${LEN_VALUES[$i]} (${LEN_LABELS[$i]}): exact_match = ${pct}%"
    printf "%s\t%s\t%s\t%s\n" "${LEN_LABELS[$i]}" "${LEN_VALUES[$i]}" "${LEN_STEMS[$i]}" "$pct" >> "$RESULTS_DIR/$name/ladder.tsv"
    # early stop rule: exact 0.0
    if [ "$acc" != "ERR" ]; then
      local iszero
      iszero=$(python3 -c "print('1' if float('$acc')==0.0 else '0')")
      [ "$iszero" = "1" ] && stopped=1
    fi
  done
  log "[$name] === EVAL LADDER DONE ==="
}

# ---------------- main ----------------
mkdir -p "$RESULTS_DIR/softmax" "$RESULTS_DIR/asentmax"
log "########## OVERNIGHT TABLE 1 SORT REPRODUCTION ##########"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

train_model softmax  softmax_overrides
run_ladder  softmax  softmax_overrides

train_model asentmax asentmax_overrides
run_ladder  asentmax asentmax_overrides

# ---------------- assemble final table ----------------
log "########## ASSEMBLING TABLE ##########"
python3 - "$RESULTS_DIR" <<'PY' | tee -a "$RUN_LOG"
import os, sys
rd = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), "results_table1_sort")
labels = ["ID","2x","4x","8x"]
paper = {"softmax":["100.0","0.0","0.0","0.0"],
         "asentmax":["100.0","100.0","79.7","0.0"]}
def read(name):
    d={}
    p=os.path.join(rd,name,"ladder.tsv")
    if os.path.exists(p):
        for line in open(p):
            parts=line.rstrip("\n").split("\t")
            if len(parts)>=4: d[parts[0]]=parts[3]
    return d
print("\n=== Table 1 — Sort (L=2): reproduced (this run) vs paper ===")
hdr = f"{'Method':<12}" + "".join(f"{l:>8}" for l in labels)
print(hdr); print("-"*len(hdr))
for name in ["softmax","asentmax"]:
    d=read(name)
    row=f"{name:<12}" + "".join(f"{d.get(l,'-'):>8}" for l in labels)
    print(row)
    prow=f"{'  (paper)':<12}" + "".join(f"{v:>8}" for v in paper[name])
    print(prow)
print("\nNote: '-' = SKIPPED by early-stop rule or missing.")
PY

log "########## ALL DONE ##########"
