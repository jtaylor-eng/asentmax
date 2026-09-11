#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --job-name=stieltjes_kdiag
#SBATCH --time=0:20:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --gpus-per-node=1 --mem=24G
#SBATCH --output=/fs/scratch/PAS2836/%u/asentmax_results/table1/slurm/stieltjes_kdiag_%j.log
# Diagnose the A100 failure of tests/test_stieltjes_triton_alibi.py at (B=1,H=4,N=1024,D=64) fp32.
set -uo pipefail
module reset >/dev/null 2>&1 || true
source /fs/scratch/PAS2836/$USER/asentmax/synthetic/osc/env.sh
source "$VENV/bin/activate"
cd "$PROJECT_ROOT"
nvidia-smi -L
python3 - <<'PY'
import sys, torch, triton, traceback
sys.path.insert(0, ".")
import src.kernels.adasplash.triton_stieltjes as ts
from src.attention.stieltjes_eager import stieltjes_normalize
dev = torch.device("cuda:0")
props = triton.runtime.driver.active.utils.get_device_properties(0)
print("smem:", props["max_shared_mem"], "picked blocks D=64 fp32:", ts._pick_blocks(64, 4, dev), "bf16:", ts._pick_blocks(64, 2, dev))

def eager(q, k, v, Q=4.0):
    N = q.shape[-2]
    s = (q.float() @ k.float().transpose(-1, -2)) / (q.shape[-1] ** 0.5)
    s = s.masked_fill(~torch.tril(torch.ones(N, N, dtype=torch.bool, device=s.device)), torch.finfo(s.dtype).min)
    return stieltjes_normalize(s, q=Q, num_iter=30).to(v.dtype) @ v

def rel(a, b): return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30)).item()

def run(B, H, N, D, dtype, blocks=None):
    torch.manual_seed(0)
    if blocks is not None:
        ts._pick_blocks = lambda *_: blocks
    q0 = torch.randn(B, H, N, D, device=dev) * 2; k0 = torch.randn(B, H, N, D, device=dev) * 2; v0 = torch.randn(B, H, N, D, device=dev)
    do = torch.randn(B, H, N, D, device=dev)
    out = {}
    for name, fn in [("eager", eager), ("triton", lambda q, k, v: ts.stieltjes_attention(q, k, v, causal=True, sm_scale=1 / D ** 0.5, stieltjes_q=4.0, num_iter=30, normalize=True, ift_grad=True))]:
        q = q0.detach().clone().to(dtype).requires_grad_(True); k = k0.detach().clone().to(dtype).requires_grad_(True); v = v0.detach().clone().to(dtype).requires_grad_(True)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        try:
            o = fn(q, k, v); o.backward(do.to(dtype)); torch.cuda.synchronize()
            out[name] = (o, q.grad, k.grad, v.grad); print(f"  {name:6s} ok  peak={torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
        except Exception as e:
            print(f"  {name:6s} FAILED: {type(e).__name__}: {str(e)[:300]}"); traceback.print_exc(limit=3); out[name] = None
    if out.get("eager") and out.get("triton"):
        print("  rel err o/dq/dk/dv:", [f"{rel(a, b):.1e}" for a, b in zip(out["triton"], out["eager"])])

print("== failing config, default tiles"); run(1, 4, 1024, 64, torch.float32)
print("== same, bf16"); run(1, 4, 1024, 64, torch.bfloat16)
print("== D=64 fp32 N=256"); run(1, 4, 256, 64, torch.float32)
print("== D=64 fp32 N=1024, tiles 64x64"); run(1, 4, 1024, 64, torch.float32, (64, 64))
print("== D=64 fp32 N=1024, tiles 32x32"); run(1, 4, 1024, 64, torch.float32, (32, 32))
print("== D=64 bf16 N=4096, default tiles"); run(1, 4, 4096, 64, torch.bfloat16)
PY
