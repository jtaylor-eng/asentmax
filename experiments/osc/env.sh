# Shared paths/env for the OSC Table-1 reproduction. `source osc/env.sh`
export OSC_USER="${USER}"
export SCRATCH_ROOT="/fs/scratch/PAS2836/${OSC_USER}"
export REPO="${SCRATCH_ROOT}/asentmax"                 # git checkout (this repo)
export VENV="${SCRATCH_ROOT}/venvs/asentmax"
export DATA_ROOT="${SCRATCH_ROOT}/asentmax_data"        # generated datasets (~35 GB)
export RESULTS_ROOT="${SCRATCH_ROOT}/asentmax_results/table1"
export ESS_ROOT="/fs/ess/PAS2836/${OSC_USER}/asentmax_results"   # durable tarballs of summaries

export HF_HOME="${SCRATCH_ROOT}/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export UV_CACHE_DIR="${SCRATCH_ROOT}/.uv-cache"
export PIP_CACHE_DIR="${SCRATCH_ROOT}/.pip-cache"
export TRITON_CACHE_DIR="${SCRATCH_ROOT}/.triton-cache"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_ROOT}/.inductor-cache"

# The synthetic sub-project expects these
export PROJECT_ROOT="${REPO}/synthetic"
export DATA_PATH="${DATA_ROOT}"
