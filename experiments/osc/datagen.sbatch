#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=t1_datagen
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-3
#SBATCH --output=/fs/scratch/PAS2836/%u/asentmax_results/table1/slurm/datagen_%a_%j.log
# Generates the 4 Table-1 datasets (paper sizes; 100-sample val/test per length).
# One array task per task. MQMTAR gets a validation split at 512 (8x) for paper-protocol
# checkpoint selection, plus test splits out to 65536 (1024x).
set -euo pipefail
source /fs/scratch/PAS2836/$USER/asentmax/experiments/osc/env.sh
source "$VENV/bin/activate"
cd "$REPO"
GEN=scripts/generate_data.py
TASKS=(sort reverse copy mqmtar)
task=${TASKS[$SLURM_ARRAY_TASK_ID]}
echo "[$(date)] datagen $task on $(hostname)"

case $task in
  sort)
    OUT="$DATA_ROOT/sort/40M-tr_32-64_vt_64-4096_vocab_32"
    ARGS=(--task_type sort --train_size 40000000 --seq_len 48 --vary_len 16 --vocab_size 32
          --mdps_seq_len "64 128 256 512" --mdps_vary_len "0 0 0 0") ;;
  reverse)
    OUT="$DATA_ROOT/reverse/30M_tr-32-64_val_64-256_t-64-512"
    ARGS=(--task_type reverse --train_size 30000000 --seq_len 48 --vary_len 16 --vocab_size 32
          --mdps_seq_len "64 96 128 256 512" --mdps_vary_len "0 0 0 0 0") ;;
  copy)
    OUT="$DATA_ROOT/copy/20M_V-32_tr-32-64_val_64-512_t-64-4096"
    ARGS=(--task_type copy --train_size 20000000 --seq_len 48 --vary_len 16 --vocab_size 32
          --mdps_seq_len "64 128 256 512 1024 2048 4096" --mdps_vary_len "0 0 0 0 0 0 0") ;;
  mqmtar)
    OUT="$DATA_ROOT/mqmtar/50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4"
    ARGS=(--task_type mqmtar --train_size 50000000 --vocab_size 10000 --seq_len 48 --vary_len 16
          --abc_size 256 --k_len 2 --v_len 2 --num_q 4 --num_kv 0.8
          --mdps_seq_len "64 128 256 512 1024 4096 16384 65536" --mdps_vary_len "0 0 0 0 0 0 0 0") ;;
esac

if [ -f "$OUT.DATA_DONE" ]; then echo "already done"; exit 0; fi
mkdir -p "$OUT"
python3 $GEN --out_dir "$OUT" --dev_size 100 --test_size 100 "${ARGS[@]}"

# Trim validation loaders that are never monitored and would slow training:
#   copy: keep val at 64/128/256/512 (idx_3 = 512 = 8x monitor); drop 1024/2048/4096
#   mqmtar: keep val 64/128/256/512 (idx_3 = 512 = 8x monitor); drop 1024+
case $task in
  copy)   rm -f "$OUT"/validation_4_1024.* "$OUT"/validation_5_2048.* "$OUT"/validation_6_4096.* ;;
  mqmtar) rm -f "$OUT"/validation_4_1024.* "$OUT"/validation_5_4096.* "$OUT"/validation_6_16384.* "$OUT"/validation_7_65536.* ;;
esac
ls -la "$OUT" | grep -E "\.(src|trg)$"
touch "$OUT.DATA_DONE"
echo "[$(date)] done"
