"""Model-level check for stieltjes_impl in {eager, triton}: a full
SparseGemma2Attention layer on (a) a right-padded training batch and (b) a
left-padded generation batch, verifying triton==eager (fwd+bwd) and that the
triton path actually dispatches to the kernel only on (a).
Run from synthetic/:  .venv/bin/python tests/test_stieltjes_impl_layer.py
"""
import sys, torch, copy
sys.path.insert(0, ".")
from transformers.models.gemma2.modeling_gemma2 import Gemma2Config
from src.models.architectures import sparse_gemma as sg

torch.manual_seed(0); dev = "cuda"

def make(impl, H=16, D=16):
    cfg = Gemma2Config(hidden_size=H * D, num_attention_heads=H, num_key_value_heads=H, head_dim=D,
                       intermediate_size=4 * H * D, num_hidden_layers=1, max_position_embeddings=4096,
                       attention_dropout=0.0, attn_logit_softcapping=None, query_pre_attn_scalar=D)
    for k, v in dict(attn_type="stieltjes", stieltjes_q=4.0, stieltjes_num_iter=30, stieltjes_impl=impl,
                     entmax_alpha=1.0, use_fast_attn=False, apply_rotary=False, apply_nape=True,
                     attn_scale_type=None, _attn_implementation="eager").items():
        setattr(cfg, k, v)
    return cfg

cfg_e = make("eager"); layer_e = sg.SparseGemma2Attention(cfg_e, 0).to(dev)
layers = {"eager": layer_e}
for impl in ["triton"]:
    l = sg.SparseGemma2Attention(make(impl), 0).to(dev); l.load_state_dict(layer_e.state_dict()); layers[impl] = l

calls = {"n": 0}
orig = sg.triton_stieltjes_attention
def spy(*a, **k):
    calls["n"] += 1; return orig(*a, **k)
sg.triton_stieltjes_attention = spy

def run(layer, x, mask2d, dtype):
    layer = layer.to(dtype); layer.zero_grad(set_to_none=True)
    x = x.to(dtype).detach().requires_grad_(True)
    cache_position = torch.arange(x.shape[1], device=dev)
    # model-level mask prep, as SparseGemma2Model does for the eager path
    m4 = layer._update_causal_mask(mask2d, x, cache_position, None)
    out, _, _ = layer(x, attention_mask=m4, position_ids=cache_position[None], cache_position=cache_position)
    # loss on real positions only (as in training: padded positions are masked out of the loss)
    loss = ((out.float() ** 2) * mask2d[..., None].float()).mean(); loss.backward()
    return out.detach().float(), x.grad.float(), {n: p.grad.float().clone() for n, p in layer.named_parameters()}

def rel(a, b): return ((a - b).norm() / b.norm().clamp(min=1e-30)).item()

ok = True
B, T, Hd = 4, 48, 256
for dtype in [torch.float32, torch.bfloat16]:
    for pad in ["right", "left"]:
        x = torch.randn(B, T, Hd, device=dev)
        lengths = torch.tensor([48, 40, 33, 20], device=dev)
        ar = torch.arange(T, device=dev)[None]
        mask = (ar < lengths[:, None]) if pad == "right" else (ar >= (T - lengths)[:, None])
        res = {}
        for impl, layer in layers.items():
            calls["n"] = 0
            res[impl] = run(layer, x, mask, dtype) + (calls["n"],)
        oe, ge, pe, _ = res["eager"]; ot, gt, pt, nt = res["triton"]
        # compare only on real (unmasked) positions
        valid = mask[..., None].float()
        e_o = rel(ot * valid, oe * valid); e_g = rel(gt * valid, ge * valid)
        e_p = max(rel(pt[n], pe[n]) for n in pe)
        tol = 1e-4 if dtype == torch.float32 else 3e-2
        expect_kernel = (pad == "right")
        good = e_o < tol and e_g < tol and e_p < tol and ((nt > 0) == expect_kernel)
        ok &= good
        print(f"{str(dtype):15s} pad={pad:5s} triton-vs-eager: out={e_o:.1e} dx={e_g:.1e} dparams={e_p:.1e} kernel_calls={nt} (expect {'>0' if expect_kernel else '0'})  {'OK' if good else 'FAIL'}")
print("ALL OK" if ok else "SOME FAILED"); sys.exit(0 if ok else 1)
