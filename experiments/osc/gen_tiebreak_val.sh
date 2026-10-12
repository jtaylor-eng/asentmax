#!/usr/bin/env bash
# gen_tiebreak_val.sh — long-length VALIDATION splits for LR-sweep tie-breaking (Sep 13).
# On copy/mqmtar the primary monitor (val exact-match @8x) saturates at 1.0 for several runs, so
# runs tied at 1.0 are ranked by a post-hoc validation ladder at the next lengths up:
#   copy   : 1024 (16x), 2048 (32x)      mqmtar : 1024 (16x), 4096 (64x)
# These are generated with a DIFFERENT RNG seed (4243) from the Table-1 data (seed 42), so they are
# not the test files. (Each split re-draws its own kv vocabulary in generate_data.py, so a fresh seed
# gives fresh kv pairs, the same way train/val/test differ from each other.)
# They are stored as test_1<i>_val<len> so they land in the TEST split of the data provider (eval.py
# only runs the test dataloader; file_subset.test=[...] selects one file) and are NOT picked up as
# validation loaders during training (datagen.sbatch dropped those on purpose: they slow training).
# 100 samples each, like every other eval split here. Idempotent. Runs on a login node in seconds.
set -euo pipefail
source "$(dirname "$0")/env.sh"; source "$VENV/bin/activate"; cd "$REPO"
SEED=4243
gen() {  # out_dir  "len1 len2"  task-args...
  local out=$1 lens=$2; shift 2
  local tmp; tmp=$(mktemp -d "$SCRATCH_ROOT/tiebreak_gen.XXXX")
  local n=0; for L in $lens; do n=$((n+1)); done
  python3 scripts/generate_data.py --out_dir "$tmp" --seed $SEED --train_size 1 --dev_size 100 --test_size 1 \
    --mdps_seq_len "$lens" --mdps_vary_len "$(printf '0 %.0s' $(seq $n))" "$@" > "$tmp/gen.log" 2>&1
  local i=0; for L in $lens; do
    for ext in src trg; do cp "$tmp/validation_${i}_${L}.$ext" "$out/test_1${i}_val${L}.$ext"; done
    echo "wrote $out/test_1${i}_val${L}.{src,trg} ($(wc -l < "$out/test_1${i}_val${L}.src") lines, $(awk '{print NF}' "$out/test_1${i}_val${L}.src" | sort -n | tail -1) max src tokens)"
    i=$((i+1))
  done
  rm -rf "$tmp"
}
C="$DATA_ROOT/copy/20M_V-32_tr-32-64_val_64-512_t-64-4096"
M="$DATA_ROOT/mqmtar/50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4"
[ -f "$C/test_11_val2048.trg" ] && echo "copy tiebreak splits exist" || \
  gen "$C" "1024 2048" --task_type copy --seq_len 48 --vary_len 16 --vocab_size 32
[ -f "$M/test_11_val4096.trg" ] && echo "mqmtar tiebreak splits exist" || \
  gen "$M" "1024 4096" --task_type mqmtar --vocab_size 10000 --seq_len 48 --vary_len 16 --abc_size 256 --k_len 2 --v_len 2 --num_q 4 --num_kv 0.8
