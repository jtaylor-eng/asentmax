#!/usr/bin/env bash
# One-time OSC environment setup (run on an ASCEND LOGIN node; has internet).
#   bash osc/setup_env.sh
# Creates a uv venv on scratch with torch 2.5.1+cu121, flash-attn 2.6.3 (prebuilt sm80 wheel),
# adasplash (Triton), wandb; stages the gated google/gemma-2-2b config for offline use.
set -euo pipefail
source "$(dirname "$0")/env.sh"

mkdir -p "$SCRATCH_ROOT" "$DATA_ROOT" "$RESULTS_ROOT" "$HF_HOME"
module reset >/dev/null 2>&1 || true

if [ ! -x "$VENV/bin/python" ]; then
  echo "== creating venv at $VENV"
  uv venv --python 3.12 "$VENV"
fi
source "$VENV/bin/activate"

echo "== installing requirements"
# torch first (cu121 wheels), then the rest; flash-attn from the prebuilt wheel matching
# torch 2.5 / cu12 / cxx11abi FALSE / cp312 (same wheel family used locally).
uv pip install --python "$VENV/bin/python" torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "$VENV/bin/python" \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
uv pip install --python "$VENV/bin/python" -r "$REPO/synthetic/requirements.txt" wandb hydra-colorlog

echo "== HF: model config is vendored at experiments/hf/gemma-2-2b (no network needed)"

echo "== import check"
python - <<'PY'
import torch, triton, flash_attn, transformers, lightning, wandb, adasplash, entmax
print("torch", torch.__version__, "triton", triton.__version__, "flash_attn", flash_attn.__version__,
      "transformers", transformers.__version__, "lightning", lightning.__version__, "wandb", wandb.__version__)
from transformers import AutoConfig
c = AutoConfig.from_pretrained(__import__("os").environ["REPO"]+"/experiments/hf/gemma-2-2b"); print("vendored config OK:", c.model_type)
PY
echo "== DONE"
