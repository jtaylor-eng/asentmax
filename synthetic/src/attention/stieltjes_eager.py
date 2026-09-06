"""
Eager (dense) Stieltjes attention normalization.

    p_j = (λ - s_j)^{-q} / S,   S = Σ_j (λ - s_j)^{-q},   λ chosen so S = 1

Reference semantics match ``kernels/adasplash/triton_stieltjes.py`` with
``normalize=True`` and a fully converged solver (the Triton kernel with
constant-1.1 Newton init does NOT converge at long N for q <= 2; here λ is
solved to machine precision by bracketed Newton, so no length-dependent
solver artifact can leak into the OOD numbers).

Gradient: exact implicit-function gradient of the normalized mapping
(same formula as the kernel's IFT_NORM mode):

    r_j   = (λ - s_j)^{-q-1}
    D     = Σ_j r_j
    δ     = Σ_j dP_j r_j / D
    dL/ds_j = (q / S) r_j (dP_j - δ)

Masked logits (very negative) get exactly zero weight and zero gradient.
Intended for the eager attention path (q_len x k_len materialised), i.e.
training at short context and autoregressive evaluation; the O(N^2) memory
makes it unsuitable for prefill beyond ~8k tokens.
"""
import torch


def _solve_lambda(x: torch.Tensor, q: float, num_iter: int = 30, eps: float = 1e-6) -> torch.Tensor:
    """Solve Σ_j (λ - x_j)^{-q} = 1 per row for centred logits (row max == 0).

    Bracketed Newton: root lies in [1, K^{1/q}] where K = number of finite
    entries in the row (f(1) >= 0 from the max entry alone; f(K^{1/q}) <= 0).
    Newton steps that leave the bracket fall back to bisection, so the solve
    is monotone and converges for any q > 0 and any N.
    Returns λ with shape x.shape[:-1] + (1,), dtype float32.
    """
    finite = torch.isfinite(x) & (x > -1e30)
    K = finite.sum(dim=-1, keepdim=True).clamp(min=1).to(x.dtype)
    lo = torch.ones_like(K)
    hi = K.pow(1.0 / q) + eps
    # Hybrid: a fixed number of bisection steps first (the residual can be
    # ~eps^{-q} near λ=1 when the top two logits nearly tie, which stalls
    # Newton), then safeguarded Newton, which is quadratic once close.
    lam = 0.5 * (lo + hi)
    n_bisect = min(12, num_iter // 2)

    for it in range(num_iter):
        diff = (lam - x).clamp(min=eps)
        inv = diff.reciprocal()
        w = inv.pow(q)
        f = w.sum(dim=-1, keepdim=True) - 1.0
        fp = -q * (w * inv).sum(dim=-1, keepdim=True)
        # update bracket by sign of f (f decreasing in λ)
        lo = torch.where(f > 0, lam, lo)
        hi = torch.where(f <= 0, lam, hi)
        if it < n_bisect:
            lam = 0.5 * (lo + hi)
            continue
        newton = lam - f / fp
        inside = (newton > lo) & (newton < hi)
        lam = torch.where(inside, newton, 0.5 * (lo + hi))
    return lam


# torch.compile fuses the ~8 tiny kernels per Newton iteration into a few; ~7x faster at
# training shapes (B=128,H=8,N=64) and bit-identical. Falls back to eager if compile is
# unavailable (e.g. no Triton / CPU).
_solve_lambda_compiled = None

def _solve(x, q, num_iter, eps):
    global _solve_lambda_compiled
    if x.is_cuda:
        if _solve_lambda_compiled is None:
            try:
                _solve_lambda_compiled = torch.compile(_solve_lambda, dynamic=True)
            except Exception:
                _solve_lambda_compiled = _solve_lambda
        try:
            return _solve_lambda_compiled(x, q, num_iter, eps)
        except Exception:
            _solve_lambda_compiled = _solve_lambda
    return _solve_lambda(x, q, num_iter, eps)


class _StieltjesNormalize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, q: float, num_iter: int, eps: float):
        s = scores if scores.dtype == torch.float64 else scores.to(torch.float32)
        s_max = s.max(dim=-1, keepdim=True).values
        x = s - s_max
        with torch.no_grad():
            lam = _solve(x, q, num_iter, eps)
        diff = (lam - x).clamp(min=eps)
        inv = diff.reciprocal()
        w = inv.pow(q)
        # zero out masked entries explicitly (they underflow to 0 anyway)
        valid = torch.isfinite(s) & (s > -1e30)
        w = torch.where(valid, w, torch.zeros_like(w))
        S = w.sum(dim=-1, keepdim=True).clamp(min=eps)
        p = w / S
        r = torch.where(valid, w * inv, torch.zeros_like(w))
        ctx.save_for_backward(r, S)
        ctx.q = q
        return p

    @staticmethod
    def backward(ctx, dP: torch.Tensor):
        r, S = ctx.saved_tensors
        q = ctx.q
        dP = dP.to(r.dtype)
        D = r.sum(dim=-1, keepdim=True).clamp(min=1e-30)
        delta = (dP * r).sum(dim=-1, keepdim=True) / D
        dS = (q / S) * r * (dP - delta)
        return dS, None, None, None


def stieltjes_normalize(scores: torch.Tensor, q: float = 4.0, num_iter: int = 30, eps: float = 1e-6) -> torch.Tensor:
    """Row-wise normalized Stieltjes mapping over the last dim (drop-in for softmax)."""
    return _StieltjesNormalize.apply(scores, float(q), int(num_iter), float(eps))


def stieltjes_reference(scores: torch.Tensor, q: float = 4.0) -> torch.Tensor:
    """Slow autograd-through-bisection reference (for tests only)."""
    s = scores.to(torch.float64)
    x = s - s.max(dim=-1, keepdim=True).values.detach()
    valid = torch.isfinite(s) & (s > -1e30)
    K = valid.sum(-1, keepdim=True).to(torch.float64)
    lo = torch.ones_like(K); hi = K.pow(1.0 / q) + 1e-6
    with torch.no_grad():
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f = torch.where(valid, (mid - x).clamp(min=1e-12).pow(-q), torch.zeros_like(x)).sum(-1, keepdim=True) - 1
            lo = torch.where(f > 0, mid, lo); hi = torch.where(f <= 0, mid, hi)
        lam = 0.5 * (lo + hi)
    # differentiate through the implicit λ using the exact fixed point: p = w/S with λ(s)
    # Implement λ(s) gradient via a one-step Newton correction, which is exact at the root.
    lam = lam.detach()
    diff = (lam - x).clamp(min=1e-12)
    w = torch.where(valid, diff.pow(-q), torch.zeros_like(x))
    f = w.sum(-1, keepdim=True) - 1
    fp = -q * torch.where(valid, diff.pow(-q - 1), torch.zeros_like(x)).sum(-1, keepdim=True)
    lam = lam - f / fp  # differentiable Newton step (exact gradient at the root)
    diff = (lam - x).clamp(min=1e-12)
    w = torch.where(valid, diff.pow(-q), torch.zeros_like(x))
    return (w / w.sum(-1, keepdim=True)).to(scores.dtype)
