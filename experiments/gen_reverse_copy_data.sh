#!/usr/bin/env bash
# Generate reverse (30M) + copy (20M) datasets matching the repo's data-config path names.
# Eval sets capped at 100 samples (dev + test), per the sort run.
set -eu
cd "$(dirname "$0")/../synthetic"   # scripts relocated to repo-root experiments/; run tree is still synthetic/
source .venv/bin/activate
DATA_ROOT="$(pwd)/data"
GEN=../scripts/generate_data.py

REV_DIR="$DATA_ROOT/reverse/30M_tr-32-64_val_64-256_t-64-512"
COPY_DIR="$DATA_ROOT/copy/20M_V-32_tr-32-64_val_64-512_t-64-4096"

echo "[$(date '+%H:%M:%S')] === REVERSE 30M ==="
mkdir -p "$REV_DIR"
python3 "$GEN" \
    --task_type reverse \
    --out_dir "$REV_DIR" \
    --train_size 30000000 \
    --dev_size 100 --test_size 100 \
    --seq_len 48 --vary_len 16 --vocab_size 32 \
    --mdps_seq_len "64 96 128 256 512" \
    --mdps_vary_len "0 0 0 0 0"

echo "[$(date '+%H:%M:%S')] === COPY 20M ==="
mkdir -p "$COPY_DIR"
python3 "$GEN" \
    --task_type copy \
    --out_dir "$COPY_DIR" \
    --train_size 20000000 \
    --dev_size 100 --test_size 100 \
    --seq_len 48 --vary_len 16 --vocab_size 32 \
    --mdps_seq_len "64 128 256 512 1024 2048 4096" \
    --mdps_vary_len "0 0 0 0 0 0 0"

# Copy: keep validation cheap during training — drop val loaders for 1024/2048/4096
# so validation only runs at 64/128/256/512 (monitor = val/acc/dataloader_idx_3 = 512).
# TEST still has all 7 lengths for the eval ladder.
echo "[$(date '+%H:%M:%S')] Trimming copy validation files >512 ..."
rm -f "$COPY_DIR"/validation_4_1024.* "$COPY_DIR"/validation_5_2048.* "$COPY_DIR"/validation_6_4096.*

echo "[$(date '+%H:%M:%S')] === DONE ==="
echo "reverse files:"; ls "$REV_DIR" | grep -E '\.(src|trg)$'
echo "copy files:";    ls "$COPY_DIR" | grep -E '\.(src|trg)$'
