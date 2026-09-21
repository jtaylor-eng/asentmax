#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=stieltjes_kbench
#SBATCH --time=1:30:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --gpus-per-node=1 --mem=24G
#SBATCH --output=/fs/scratch/PAS2836/%u/asentmax_results/table1/slurm/kbench_%j.log
# Old vs new Triton Stieltjes kernel on the A100 (+ tile ablation) and the kernel-level equivalence test.
set -uo pipefail
module reset >/dev/null 2>&1 || true
export SCRATCH=/fs/scratch/PAS2836/$USER
source $SCRATCH/asentmax/experiments/osc/env.sh
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"
nvidia-smi -L
export PYTHONPATH="$REPO/experiments/osc/oneoff:${PYTHONPATH:-}"
echo "== bench"; python3 "$REPO/experiments/osc/oneoff/kernel_bench.py"
echo "== equivalence test (triton vs eager), default tiles"; python3 tests/test_stieltjes_triton_alibi.py 2>&1 | grep -E "forcing|FAIL|ALL OK|SOME|long-N"
for t in 64x32 128x64; do echo "== equivalence test, tiles $t"; STIELTJES_TILES=$t python3 tests/test_stieltjes_triton_alibi.py 2>&1 | grep -E "forcing|FAIL|ALL OK|SOME"; done
echo "== layer test"; python3 tests/test_stieltjes_impl_layer.py 2>&1 | grep -E "FAIL|ALL OK|SOME"
