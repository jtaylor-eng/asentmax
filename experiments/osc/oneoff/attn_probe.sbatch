#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=attn_probe
#SBATCH --time=1:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --gpus-per-node=1 --mem=48G
#SBATCH --output=/fs/scratch/PAS2836/%u/asentmax_results/table1/slurm/probe_%j.log
# Attention-support probe on the MQMTAR plateau checkpoints (reproduction_0920.md §3.4 mechanism test):
# stuck ASEntmax (HF-init scale) vs stuck AS-windowed vs escaped windowed / dense q4 / plain entmax.
set -uo pipefail
module reset >/dev/null 2>&1 || true
export SCRATCH=/fs/scratch/PAS2836/$USER
source $SCRATCH/asentmax/experiments/osc/env.sh
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"
export DATA_PATH="$DATA_ROOT"
DATA='${oc.env:DATA_PATH}/mqmtar/50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4/'
OUT=$RESULTS_ROOT/probe; mkdir -p $OUT
probe() {  # method run_dir label
  local ck=$RESULTS_ROOT/mqmtar/$2/checkpoints/last.ckpt
  [ -f "$ck" ] || { echo "== $3: no ckpt $ck"; return; }
  echo "== $3  ($2, last.ckpt)"
  python3 $REPO/experiments/osc/oneoff/attn_support_probe.py --ckpt "$ck" --task mqmtar --method $1 --data "$DATA" --batches 4 --out $OUT/$3.json 2>&1 | grep -E "^layer|Error|Traceback|wrote"
}
probe asentmax_w20k        asentmax_w20k/s1_lr2e-4        asentmax_w20k_s1_STUCK
probe asentmax             asentmax/s2_lr4e-4             asentmax_4e-4_s2_STUCK
probe aswstieltjes         aswstieltjes/s1_lr2e-4         aswstieltjes_10k_s1_STUCK
probe aswstieltjes_w20k    aswstieltjes_w20k/s1_lr2e-4    aswstieltjes_w20k_s1_ESCAPED
probe wstieltjes           wstieltjes/s3_lr4e-4           wstieltjes_s3_ESCAPED
probe stieltjes            stieltjes/s3_lr4e-4            stieltjes_q4_s3_ESCAPED
probe entmax               entmax/s1_lr2e-4               entmax_s1_STUCK
probe entmax               entmax/s2_lr2e-4               entmax_s2_ESCAPED_late
probe stieltjes_q16        stieltjes_q16/s1_lr4e-4        stieltjes_q16_s1_STUCK
