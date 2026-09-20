"""Model-level check for stieltjes_impl in {eager, triton} across the Stieltjes
variants (dense q=4, dense q=16, windowed q=4 d=2, AS-windowed): a full
SparseGemma2Attention layer on (a) a right-padded training batch and (b) a
left-padded generation batch, verifying triton==eager (fwd+bwd) and that the
triton path actually dispatches to the kernel only on (a).
Run from synthetic/:  .venv/bin/python tests/test_stieltjes_impl_layer.py
"""
import sys, torch
sys.path.insert(0, ".")
from transformers.models.gemma2.modeling_gemma2 import Gemma2Config
from src.models.architectures import sparse_gemma as sg

torch.manual_seed(0); dev = "cuda"

def make(impl, variant, H=16, D=16):
    cfg = Gemma2Config(hidden_size=H * D, num_attention_heads=H, num_key_value_heads=H, head_dim=D,
                       intermediate_size=4 * H * D, num_hidden_layers=1, max_position_embeddings=4096,
                       attention_dropout=0.0, attn_logit_softcapping=None, query_pre_attn_scalar=D)
    for k, v in {**dict(attn_type="stieltjes", stieltjes_num_iter=30, stieltjes_impl=impl, entmax_alpha=1.0,
                        use_fast_attn=False, apply_rotary=False, apply_nape=True, attn_scale_type=None,
                        _attn_implementation="eager"), **variant}.items():
        setattr(cfg, k, v)
    return cfg

VARIANTS = {
    "q4":        dict(stieltjes_q=4.0),
    "q16":       dict(stieltjes_q=16.0),
    "q4_d2":     dict(stieltjes_q=4.0, stieltjes_window=2.0),
    "as_q4_d2":  dict(stieltjes_q=4.0, stieltjes_window=2.0, attn_scale_type="adapt-softplus-tanh", attn_scale_proj_bias=True),
}

calls = {"n": 0}
orig = sg.triton_stieltjes_attention
def spy(*a, **k):
    calls["n"] += 1; return orig(*a, **k)
sg.triton_stieltjes_attention = spy

def run(layer, x, mask2d, dtype):
    layer = layer.to(dtype); layer.zero_grad(set_to_none=True)
    x = x.to(dtype).detach().requires_grad_(True)
    cache_position = torch.arange(x.shape[1], device=dev)
    m4 = layer._update_causal_mask(mask2d, x, cache_position, None)
    out, _, _ = layer(x, attention_mask=m4, position_ids=cache_position[None], cache_position=cache_position)
    loss = ((out.float() ** 2) * mask2d[..., None].float()).mean(); loss.backward()
    return out.detach().float(), x.grad.float(), {n: p.grad.float().clone() for n, p in layer.named_parameters()}

def rel(a, b): return ((a - b).norm() / b.norm().clamp(min=1e-30)).item()

ok = True
B, T, Hd = 4, 48, 256
for vname, variant in VARIANTS.items():
    layer_e = sg.SparseGemma2Attention(make("eager", variant), 0).to(dev)
    layer_t = sg.SparseGemma2Attention(make("triton", variant), 0).to(dev); layer_t.load_state_dict(layer_e.state_dict())
    for dtype in [torch.float32, torch.bfloat16]:
        for pad in ["right", "left"]:
            x = torch.randn(B, T, Hd, device=dev)
            lengths = torch.tensor([48, 40, 33, 20], device=dev)
            ar = torch.arange(T, device=dev)[None]
            mask = (ar < lengths[:, None]) if pad == "right" else (ar >= (T - lengths)[:, None])
            calls["n"] = 0; oe, ge, pe = run(layer_e, x, mask, dtype); ne = calls["n"]
            calls["n"] = 0; ot, gt, pt = run(layer_t, x, mask, dtype); nt = calls["n"]
            valid = mask[..., None].float()
            e_o = rel(ot * valid, oe * valid); e_g = rel(gt * valid, ge * valid)
            e_p = max(rel(pt[n], pe[n]) for n in pe)
            tol = 1e-4 if dtype == torch.float32 else 3e-2
            expect_kernel = (pad == "right")
            good = e_o < tol and e_g < tol and e_p < tol and ((nt > 0) == expect_kernel) and ne == 0
            if not good and dtype == torch.bfloat16 and e_o < tol:
                # bf16 grads differ by more than tol: decide by distance to fp32 truth (same weights,
                # fp32 eager). Pass if triton-bf16 is at least as close to it as eager-bf16 is.
                of, gf, pf = run(layer_e, x, mask, torch.float32); layer_e.to(torch.bfloat16)
                de = max(rel(pe[n], pf[n]) for n in pe); dt = max(rel(pt[n], pf[n]) for n in pt)
                good = dt <= de * 1.1 and ((nt > 0) == expect_kernel) and ne == 0
                print(f"{vname:9s} bf16 dparams vs fp32 truth: eager={de:.1e} triton={dt:.1e}")
            ok &= good
            print(f"{vname:9s} {str(dtype)[6:]:8s} pad={pad:5s} triton-vs-eager: out={e_o:.1e} dx={e_g:.1e} dparams={e_p:.1e} "
                  f"kernel_calls={nt} (expect {'>0' if expect_kernel else '0'})  {'OK' if good else 'FAIL'}")
print("ALL OK" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
