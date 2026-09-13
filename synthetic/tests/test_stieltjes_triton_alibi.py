"""Kernel-level check: fused Triton Stieltjes (normalize+ift, ALiBi) == eager
stieltjes_normalize with make_bias_window, forward and backward, fp32 + bf16.
Run from synthetic/:  .venv/bin/python tests/test_stieltjes_triton_alibi.py
"""
import sys, torch
sys.path.insert(0, ".")
from src.kernels.adasplash.triton_stieltjes import stieltjes_attention
from src.attention.stieltjes_eager import stieltjes_normalize
from src.models.architectures.sparse_gemma import make_bias_window, get_nape_slopes

torch.manual_seed(0)
dev = "cuda"
ok = True

def eager(q, k, v, slopes, Q, causal=True):
    N = q.shape[-2]
    s = (q.float() @ k.float().transpose(-1, -2)) / (q.shape[-1] ** 0.5)
    if slopes is not None:
        s = s + make_bias_window(slopes, N, N, s.device, s.dtype)
    if causal:
        m = torch.tril(torch.ones(N, N, dtype=torch.bool, device=s.device))
        s = s.masked_fill(~m, torch.finfo(s.dtype).min)
    p = stieltjes_normalize(s, q=Q, num_iter=30).to(v.dtype)
    return p @ v

def rel(a, b): return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30)).item()

for dtype in [torch.float32, torch.bfloat16]:
    for (B, H, N, D) in [(2, 8, 64, 32), (2, 16, 64, 16), (1, 8, 200, 32), (1, 4, 1024, 64), (3, 16, 96, 16)]:
        for use_alibi in [False, True]:
            slopes = get_nape_slopes(H, H // 2).to(dev) if use_alibi else None
            q0 = torch.randn(B, H, N, D, device=dev) * 2; k0 = torch.randn(B, H, N, D, device=dev) * 2; v0 = torch.randn(B, H, N, D, device=dev)
            do = torch.randn(B, H, N, D, device=dev)
            def run(fn):
                q = q0.detach().clone().to(dtype).requires_grad_(True); k = k0.detach().clone().to(dtype).requires_grad_(True); v = v0.detach().clone().to(dtype).requires_grad_(True)
                o = fn(q, k, v); o.backward(do.to(dtype)); return o, q.grad, k.grad, v.grad
            oe = run(lambda q, k, v: eager(q, k, v, slopes, 4.0))
            ot = run(lambda q, k, v: stieltjes_attention(q, k, v, causal=True, sm_scale=1 / D ** 0.5, stieltjes_q=4.0,
                                                          num_iter=30, normalize=True, ift_grad=True, alibi_slopes=slopes))
            errs = [rel(a, b) for a, b in zip(ot, oe)]
            tol = 1e-4 if dtype == torch.float32 else 3e-2
            good = all(e < tol for e in errs) and not any(torch.isnan(t).any() for t in ot)
            ok &= good
            print(f"{str(dtype):15s} B={B} H={H:2d} N={N:4d} D={D:2d} alibi={use_alibi!s:5s} o={errs[0]:.1e} dq={errs[1]:.1e} dk={errs[2]:.1e} dv={errs[3]:.1e} {'OK' if good else 'FAIL'}")
print("ALL OK" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
