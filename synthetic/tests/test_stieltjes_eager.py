"""Eager Stieltjes (dense + windowed): forward vs 200-step bisection reference, λ
convergence, masked-entry leak, exact zeros off-support, sparsity level, and the
hand-written backward vs autograd through a fully unrolled fp64 solver.
Run from synthetic/:  .venv/bin/python tests/test_stieltjes_eager.py   (CPU ok)
"""
import torch, sys, math
sys.path.insert(0, ".")
from src.attention.stieltjes_eager import stieltjes_normalize, stieltjes_reference, _solve_lambda, _window_c

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ok = True
CASES = [(q, d) for q in [1.0, 2.0, 4.0, 8.0, 16.0] for d in [None]] + [(4.0, 1.0), (4.0, 2.0), (4.0, 4.0), (16.0, 1.5)]

# 1) forward
for q, d in CASES:
    c = _window_c(q, d)
    for N in [64, 512, 4096, 16384]:
        s = torch.randn(2, 3, 8, N, device=dev) * 3
        mask = torch.rand_like(s) < 0.3
        s = s.masked_fill(mask, torch.finfo(s.dtype).min)
        p = stieltjes_normalize(s, q=q, window=d)
        ref = stieltjes_reference(s, q=q, window=d)
        err = (p - ref).abs().max().item()
        rowsum = (p.sum(-1) - 1).abs().max().item()
        leak = p[mask].abs().max().item()
        x = s.float() - s.float().max(-1, keepdim=True).values
        lam = _solve_lambda(x, q, c)
        w = ((lam - x).clamp(min=1e-6).pow(-q) - c).clamp(min=0) * (~mask)
        conv = (w.sum(-1) - 1).abs().max().item()
        nnz = (p > 0).float().sum(-1).mean().item()
        exact_zero = (p == 0).float().mean().item()
        good = err < 1e-5 and rowsum < 1e-5 and leak == 0 and conv < 1e-4 and (d is None or nnz < N)
        ok &= good
        print(f"fwd q={q:<4} d={str(d):<4} N={N:<6} max|p-ref|={err:.1e} rowsum={rowsum:.1e} leak={leak:.0e} |Σw-1|={conv:.1e} "
              f"nnz/row={nnz:7.1f} zeros={exact_zero:.2f} {'OK' if good else 'FAIL'}")

# 2) backward vs autograd through a fully unrolled fp64 solver (the ground truth)
for q, d in [(2.0, None), (4.0, None), (16.0, None), (4.0, 1.0), (4.0, 2.0), (16.0, 1.5)]:
    c = _window_c(q, d)
    s = (torch.randn(2, 4, 6, 40, device=dev, dtype=torch.float64) * 2).requires_grad_(True)
    mask = torch.rand(2, 4, 6, 40, device=dev) < 0.2
    g = torch.randn(2, 4, 6, 40, device=dev, dtype=torch.float64)
    p = stieltjes_normalize(s.masked_fill(mask, -1e30), q=q, window=d); (p * g).sum().backward()
    grad_ours = s.grad.clone(); s.grad = None
    p2 = stieltjes_reference(s.masked_fill(mask, -1e30), q=q, window=d); (p2 * g).sum().backward()
    grad_ref = s.grad.clone(); s.grad = None

    def L_unrolled(t):
        x = t - t.max(-1, keepdim=True).values
        valid = ~mask
        K = valid.sum(-1, keepdim=True).to(t.dtype)
        lo = torch.full_like(K, (1 + c) ** (-1 / q)); hi = K.pow(1.0 / q) + 1e-6
        if c > 0: hi = hi.clamp(max=c ** (-1 / q) + 1e-6)
        lam = 0.5 * (lo + hi)
        for it in range(80):
            inv = (lam - x).clamp(min=1e-6).reciprocal(); w = inv.pow(q) - c
            on = valid & (w > 0)
            f = torch.where(on, w, torch.zeros_like(w)).sum(-1, keepdim=True) - 1
            fp = -q * torch.where(on, (w + c) * inv, torch.zeros_like(w)).sum(-1, keepdim=True)
            lo = torch.where(f > 0, lam, lo); hi = torch.where(f <= 0, lam, hi)
            if it < 12: lam = 0.5 * (lo + hi); continue
            n = lam - f / fp; lam = torch.where((n > lo) & (n < hi), n, 0.5 * (lo + hi))
        w = (lam - x).pow(-q) - c
        w = torch.where(valid & (w > 0), w, torch.zeros_like(w))
        return ((w / w.sum(-1, keepdim=True)) * g).sum()
    L_unrolled(s.masked_fill(mask, -1e30)).backward()
    grad_unrolled = s.grad.clone(); s.grad = None
    rel = ((grad_ours - grad_ref).norm() / grad_ref.norm()).item()
    rel_u = ((grad_ours - grad_unrolled).norm() / grad_unrolled.norm()).item()
    good = rel < 1e-5 and rel_u < 1e-6
    ok &= good
    print(f"bwd q={q:<4} d={str(d):<4} rel_err_vs_ref={rel:.1e} rel_err_vs_unrolled={rel_u:.1e} {'OK' if good else 'FAIL'}")

# 3) dense == windowed with an infinite window (bit-for-bit), and the on/off boundary
s = torch.randn(4, 8, 300, device=dev) * 3
same = torch.equal(stieltjes_normalize(s, q=4.0), stieltjes_normalize(s, q=4.0, window=float("inf")))
ok &= same; print("dense == window=inf bit-for-bit:", same)

# 4) dispersion sanity: 1 target vs N-1 distractors at margin 4
def pt(N, m, q, d=None):
    s = torch.zeros(1, N, device=dev); s[0, 0] = m
    return stieltjes_normalize(s, q=q, window=d)[0, 0].item()
Ns = [64, 256, 1024, 4096, 16384, 65536]
for q, d in [(4.0, None), (16.0, None), (4.0, 2.0)]:
    print(f"p_target q={q} d={d} margin 4:", [round(pt(N, 4.0, q, d), 3) for N in Ns])
print("ALL OK" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
