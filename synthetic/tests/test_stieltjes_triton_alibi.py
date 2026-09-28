"""Kernel-level check: fused Triton Stieltjes (dense + windowed, ALiBi) == eager
stieltjes_normalize with make_bias_window, forward and backward, fp32 + bf16, plus
the unnormalised row-sum residual |S-1| (a wrong λ cannot hide behind p = w/S).
Run from synthetic/:  .venv/bin/python tests/test_stieltjes_triton_alibi.py
"""
import sys, os, torch
sys.path.insert(0, ".")
import src.kernels.adasplash.triton_stieltjes as _tsm
from src.kernels.adasplash.triton_stieltjes import stieltjes_attention, stieltjes_solver_residual
from src.attention.stieltjes_eager import stieltjes_normalize
from src.models.architectures.sparse_gemma import make_bias_window, get_nape_slopes

# STIELTJES_TILES=64x32 forces a tile size (to validate a non-default tile on a given GPU)
if os.environ.get("STIELTJES_TILES"):
    _bm, _bn = map(int, os.environ["STIELTJES_TILES"].split("x"))
    _tsm._pick_blocks = lambda D, e, d: (_bm, _bn)
    print("forcing tiles", (_bm, _bn))

torch.manual_seed(0)
dev = "cuda"
ok = True

def eager(q, k, v, slopes, Q, d, causal=True):
    N = q.shape[-2]
    s = (q.float() @ k.float().transpose(-1, -2)) / (q.shape[-1] ** 0.5)
    if slopes is not None:
        s = s + make_bias_window(slopes, N, N, s.device, s.dtype)
    if causal:
        m = torch.tril(torch.ones(N, N, dtype=torch.bool, device=s.device))
        s = s.masked_fill(~m, torch.finfo(s.dtype).min)
    p = stieltjes_normalize(s, q=Q, num_iter=30, window=d).to(v.dtype)
    return p @ v

def rel(a, b): return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30)).item()

CASES = [(4.0, None), (16.0, None), (4.0, 2.0), (4.0, 1.0), (16.0, 1.5), (2.0, None), (8.0, None), (2.0, 2.0), (8.0, 2.0)]
SHAPES = [(2, 8, 64, 32), (2, 16, 64, 16), (1, 8, 200, 32), (1, 4, 1024, 64), (3, 16, 96, 16)]
for dtype in [torch.float32, torch.bfloat16]:
    for Q, d in CASES:
        for (B, H, N, D) in SHAPES:
            for use_alibi in [False, True]:
                slopes = get_nape_slopes(H, H // 2).to(dev) if use_alibi else None
                q0 = torch.randn(B, H, N, D, device=dev) * 2; k0 = torch.randn(B, H, N, D, device=dev) * 2; v0 = torch.randn(B, H, N, D, device=dev)
                do = torch.randn(B, H, N, D, device=dev)
                def run(fn):
                    q = q0.detach().clone().to(dtype).requires_grad_(True); k = k0.detach().clone().to(dtype).requires_grad_(True); v = v0.detach().clone().to(dtype).requires_grad_(True)
                    o = fn(q, k, v); o.backward(do.to(dtype)); return o, q.grad, k.grad, v.grad
                oe = run(lambda q, k, v: eager(q, k, v, slopes, Q, d))
                ot = run(lambda q, k, v: stieltjes_attention(q, k, v, causal=True, sm_scale=1 / D ** 0.5, stieltjes_q=Q,
                                                              num_iter=30, window=d, alibi_slopes=slopes))
                res = stieltjes_solver_residual(q0.to(dtype), k0.to(dtype), v0.to(dtype), causal=True, sm_scale=1 / D ** 0.5,
                                                stieltjes_q=Q, num_iter=30, window=d, alibi_slopes=slopes).max().item()
                errs = [rel(a, b) for a, b in zip(ot, oe)]
                tol = 1e-4 if dtype == torch.float32 else 3e-2
                good = all(e < tol for e in errs) and res < 1e-4 and not any(torch.isnan(t).any() for t in ot)
                ok &= good
                print(f"{str(dtype)[6:]:8s} q={Q:<4} d={str(d):<4} B={B} H={H:2d} N={N:4d} D={D:2d} alibi={use_alibi!s:5s} "
                      f"o={errs[0]:.1e} dq={errs[1]:.1e} dk={errs[2]:.1e} dv={errs[3]:.1e} |S-1|={res:.1e} {'OK' if good else 'FAIL'}")

# long-N solver check (no eager reference; residual + finiteness only)
for Q, d in CASES:
    B, H, N, D = 1, 2, 16384, 64
    q = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16) * 2; k = torch.randn_like(q) * 2; v = torch.randn_like(q)
    slopes = get_nape_slopes(H, H // 2).to(dev)
    res = stieltjes_solver_residual(q, k, v, causal=True, stieltjes_q=Q, num_iter=30, window=d, alibi_slopes=slopes).max().item()
    o = stieltjes_attention(q, k, v, causal=True, stieltjes_q=Q, num_iter=30, window=d, alibi_slopes=slopes)
    good = res < 1e-3 and torch.isfinite(o).all().item()
    ok &= good
    print(f"long-N bf16 q={Q:<4} d={str(d):<4} N={N} |S-1|={res:.1e} finite={torch.isfinite(o).all().item()} {'OK' if good else 'FAIL'}")
print("ALL OK" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
