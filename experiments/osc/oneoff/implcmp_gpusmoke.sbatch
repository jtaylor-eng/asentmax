#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=implcmp_gpusmoke
#SBATCH --time=0:40:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=6 --gpus-per-node=1 --mem=32G
#SBATCH --output=/fs/scratch/PAS2836/%u/asentmax_results/table1/slurm/implcmp_gpusmoke_%j.log
set -uo pipefail
module reset >/dev/null 2>&1 || true
source /fs/scratch/PAS2836/$USER/asentmax/experiments/osc/env.sh
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"
nvidia-smi -L
echo "== kernel test (ALiBi, fp32/bf16)"; python3 tests/test_stieltjes_triton_alibi.py 2>&1 | grep -E "^torch|OK|FAIL|Error" ; echo "rc=${PIPESTATUS[0]}"
echo "== layer test"; python3 tests/test_stieltjes_impl_layer.py 2>&1 | grep -E "^torch|OK|FAIL|Error" ; echo "rc=${PIPESTATUS[0]}"
echo "== eager test"; python3 tests/test_stieltjes_eager.py 2>&1 | tail -3
for impl in eager triton; do
  echo "== train smoke $impl"
  t0=$(date +%s)
  python3 src/train.py experiment=entmax/copy logger=csv task_name=implsmoke_$impl test=False seed=1 model.optimizer.lr=5e-4 \
    ++trainer.max_steps=200 trainer.max_epochs=1 trainer.val_check_interval=100 ++trainer.limit_val_batches=2 \
    ++callbacks.model_checkpoint.dirpath=$TMPDIR/implsmoke_$impl \
    model.net.entmax_alpha=1.0 ++model.net.attn_type=stieltjes ++model.net.stieltjes_q=4.0 ++model.net.stieltjes_num_iter=30 \
    ++model.net.stieltjes_impl=$impl ++model.net.attn_implementation=eager ++model.net.use_fast_attn=False \
    ++model.net.attn_scale_type=null ++model.net.apply_rotary=False ++model.net.apply_nape=True > $TMPDIR/implsmoke_$impl.log 2>&1
  echo "rc=$? elapsed=$(( $(date +%s) - t0 ))s"
  grep -E "Traceback|Error" $TMPDIR/implsmoke_$impl.log | head -3
  f=$(find logs/implsmoke_$impl -name metrics.csv | sort | tail -1)
  python3 - "$f" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows[-2:]:
    print({k: v for k, v in r.items() if v and ('loss' in k or k == 'step')})
PY
  rm -rf logs/implsmoke_$impl
done
