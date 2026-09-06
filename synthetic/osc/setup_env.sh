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

echo "== staging HF config (offline)"
SNAP="$HF_HOME/hub/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf"
mkdir -p "$SNAP" "$HF_HOME/hub/models--google--gemma-2-2b/refs"
if [ ! -f "$SNAP/config.json" ]; then
  echo "config.json missing at $SNAP — copy it from the desktop:"
  echo "  scp ~/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/*/config.json ascend:$SNAP/"
  exit 2
fi
echo c5ebcd40d208330abc697524c919956e692655cf > "$HF_HOME/hub/models--google--gemma-2-2b/refs/main"

echo "== import check"
python - <<'PY'
import torch, triton, flash_attn, transformers, lightning, wandb, adasplash, entmax
print("torch", torch.__version__, "triton", triton.__version__, "flash_attn", flash_attn.__version__,
      "transformers", transformers.__version__, "lightning", lightning.__version__, "wandb", wandb.__version__)
from transformers import AutoConfig
c = AutoConfig.from_pretrained("google/gemma-2-2b"); print("HF offline config OK:", c.model_type)
PY
echo "== DONE"
