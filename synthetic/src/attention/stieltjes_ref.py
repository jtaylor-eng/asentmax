"""
Original PyTorch reference of the Stieltjes simplex mapping (verbatim from
`stieltjes_torch_ref.py`, ex jtaylor-eng/probability-simplex-mappings), kept
for the eager-vs-ref-vs-triton comparison.

Semantics (deliberately NOT changed):
  * forward: 15-step bisection for λ over [eps, 4096^{1/q}], returns the
    UNNORMALISED weights (λ - x)^{-q} (rows sum to 1 ± ~5e-4; wrong for
    N > 4096 diffuse rows because the upper bracket is capped);
  * backward: autograd only sees the max-centering, so λ is treated as a
    constant and the whole normalisation correction lands on the argmax
    column (== the Triton kernel's block_lambda_grad mode).
"""
import torch


class ProbabilitySimplexMapping(torch.nn.Module):
    """Minimal stand-in for the upstream base class."""


class StieltjesTransform(ProbabilitySimplexMapping):
    """Stieltjes transform as introduced, using binary search."""
    def __init__(
        self,
        q: float = 20,
        num_iter: int = 15,
        eps: float = 1e-9
    ):
        super().__init__()
        self._q = q
        self._num_iter = num_iter
        self._eps = eps

    def _line_search_bs(self, shifted_logits, dim, lb, ub):
        for _ in range(self._num_iter):
            mid = (lb + ub) / 2.0
            
            prob_sum = torch.sum(
                torch.pow((mid - shifted_logits).clamp(min=self._eps), -self._q),
                dim=dim,
                keepdim=True
            ) - 1
            
            lb = torch.where(prob_sum > 0, mid, lb)
            ub = torch.where(prob_sum <= 0, mid, ub)

        return lb, ub

    def translate_logits(
        self,
        logits,
        dim,
        **kwargs,
    ) -> torch.Tensor:
        """Calculates 1 / (lambda_q - x_i)^q"""
        
        # logits = torch.clamp(logits, min=-50.0, max=50.0)
        
        x_max = torch.max(logits, dim=dim, keepdim=True).values
        x_i = logits - x_max

        lb = torch.full_like(x_max, self._eps)
        # ub = torch.full_like(x_max, logits.shape[dim] ** (1.0/ self._q))
        static_max_len = 4096.0 
        ub = torch.full_like(x_max, static_max_len ** (1.0/ self._q))

        lb, ub = self._line_search_bs(
            shifted_logits=x_i,
            dim=dim,
            lb=lb,
            ub=ub
        )
        lambda_1 = (lb + ub) / 2.0
        
        # 1 / (lambda_q - x_i)^q
        return torch.pow((lambda_1 - x_i).clamp(min=self._eps), -self._q)


def stieltjes_ref_normalize(scores: torch.Tensor, q: float = 4.0, num_iter: int = 15) -> torch.Tensor:
    """Drop-in for `stieltjes_normalize` with the reference's semantics (see module doc)."""
    return StieltjesTransform(q=q, num_iter=num_iter).translate_logits(scores, dim=-1)

    # def translate_logits(self, logits, dim, **kwargs) -> torch.Tensor:
    #     x_max = torch.max(logits, dim=dim, keepdim=True).values
    #     x_i = logits - x_max
        
    #     # Initial guess: We know lambda must be at least 1.0 (since x_max = 0)
    #     # Using 1.1 is a safe, close starting point
    #     lambd = torch.full_like(x_max, 1.1)
        
    #     # 3 iterations of Newton-Raphson is usually enough for float16/bfloat16 precision
    #     for _ in range(3):
    #         diff = (lambd - x_i).clamp(min=self._eps)
            
    #         # f(lambda)
    #         f_val = torch.sum(torch.pow(diff, -self._q), dim=dim, keepdim=True) - 1.0
    #         # f'(lambda)
    #         f_deriv = -self._q * torch.sum(torch.pow(diff, -self._q - 1.0), dim=dim, keepdim=True)
            
    #         # Update
    #         lambd = lambd - (f_val / f_deriv)

    #     return torch.pow((lambd - x_i).clamp(min=self._eps), -self._q)
