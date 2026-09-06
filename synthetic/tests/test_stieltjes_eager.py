import torch, sys, math
sys.path.insert(0, ".")
from src.attention.stieltjes_eager import stieltjes_normalize, stieltjes_reference, _solve_lambda

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ok = True

# 1) forward: sums to 1, matches bisection reference, handles masked entries, at many N
for q in [1.0, 2.0, 4.0, 8.0]:
    for N in [64, 512, 4096, 16384]:
        s = torch.randn(2, 3, 8, N, device=dev) * 3
        mask = torch.rand_like(s) < 0.3
        s = s.masked_fill(mask, torch.finfo(s.dtype).min)
        p = stieltjes_normalize(s, q=q)
        ref = stieltjes_reference(s, q=q)
        err = (p - ref).abs().max().item()
        rowsum = (p.sum(-1) - 1).abs().max().item()
        masked_leak = p[mask].abs().max().item() if mask.any() else 0.0
        # also verify λ converged: Σ w = 1 before normalisation
        x = s.float() - s.float().max(-1, keepdim=True).values
        lam = _solve_lambda(x, q)
        S = ((lam - x).clamp(min=1e-6).pow(-q) * (~mask)).sum(-1)
        conv = (S - 1).abs().max().item()
        good = err < 1e-5 and rowsum < 1e-5 and masked_leak == 0 and conv < 1e-4
        ok &= good
        print(f"fwd q={q:<4} N={N:<6} max|p-ref|={err:.2e} rowsum_err={rowsum:.1e} leak={masked_leak:.1e} |Σw-1|={conv:.1e} {'OK' if good else 'FAIL'}")

# 2) backward: compare to autograd through reference (double precision) and gradcheck
for q in [2.0, 4.0]:
    s = (torch.randn(2, 4, 6, 40, device=dev, dtype=torch.float64) * 2).requires_grad_(True)
    mask = torch.rand(2, 4, 6, 40, device=dev) < 0.2
    s_in = s.masked_fill(mask, -1e30)
    g = torch.randn(2, 4, 6, 40, device=dev, dtype=torch.float64)
    p = stieltjes_normalize(s.masked_fill(mask, -1e30), q=q); (p * g).sum().backward()
    grad_ours = s.grad.clone(); s.grad = None
    p2 = stieltjes_reference(s.masked_fill(mask, -1e30), q=q); (p2 * g).sum().backward()
    grad_ref = s.grad.clone(); s.grad = None
    rel = ((grad_ours - grad_ref).norm() / grad_ref.norm()).item()
    # Exact check: autograd through the fully unrolled solver (differentiable λ).
    def L_unrolled(t):
        x = t - t.max(-1, keepdim=True).values
        valid = ~mask
        K = valid.sum(-1, keepdim=True).to(t.dtype)
        lo = torch.ones_like(K); hi = K.pow(1.0 / q) + 1e-6; lam = 0.5 * (lo + hi)
        for it in range(60):
            diff = (lam - x).clamp(min=1e-6); w = torch.where(valid, diff.pow(-q), torch.zeros_like(x))
            f = w.sum(-1, keepdim=True) - 1; fp = -q * torch.where(valid, w / diff, torch.zeros_like(x)).sum(-1, keepdim=True)
            lo = torch.where(f > 0, lam, lo); hi = torch.where(f <= 0, lam, hi)
            if it < 12: lam = 0.5 * (lo + hi); continue
            n = lam - f / fp; lam = torch.where((n > lo) & (n < hi), n, 0.5 * (lo + hi))
        w = torch.where(valid, (lam - x).pow(-q), torch.zeros_like(x))
        return ((w / w.sum(-1, keepdim=True)) * g).sum()
    L_unrolled(s.masked_fill(mask, -1e30)).backward()
    grad_unrolled = s.grad.clone(); s.grad = None
    rel_u = ((grad_ours - grad_unrolled).norm() / grad_unrolled.norm()).item()
    gc = rel_u < 1e-6
    good = rel < 1e-5 and gc
    ok &= good
    print(f"bwd q={q} rel_err_vs_ref={rel:.2e} rel_err_vs_unrolled={rel_u:.2e} gradcheck={gc} {'OK' if good else 'FAIL'}")

# 3) dispersion sanity: 1 target vs distractors, calibrated at 64
def pt(N, m, q):
    s = torch.zeros(1, N, device=dev); s[0, 0] = m
    return stieltjes_normalize(s, q=q)[0, 0].item()
print("p_target (q=4, margin 7.907):", [round(pt(N, 7.907, 4.0), 3) for N in [64, 256, 1024, 4096, 16384]])
print("ALL OK" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
