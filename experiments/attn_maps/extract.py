#!/usr/bin/env python3
"""Extract attention maps from a trained checkpoint (teacher-forced, no generation).

Every SparseGemma2Attention.forward is replaced by an eager, query-chunked forward that is the
same math as the trained path (qk scale, NAPE/ALiBi bias, AS scaler, causal mask, the row's
attention map in fp32) but materialises the attention probabilities p row-chunk by row-chunk,
so ~4k-token sequences fit for every method (softmax, entmax-1.5, Stieltjes dense/windowed).
A sanity check compares the model's logits under the patched forward against the original
(trained) forward on the first sample and reports teacher-forced target-token accuracy.

Output, one .npz per (task, method, test stem), samples run at batch 1 (no padding):
  meta_json                     task/method/ckpt/stem/heads/layers/alibi heads, sanity numbers
  s<i>_input_ids, s<i>_target_mask   per sample
  s<i>_full    [Lyr,H,L,L] fp16 p (only if L <= --full-max)
  s<i>_pool_mean / s<i>_pool_max [Lyr,H,B,B] fp16: p binned to BxB (key: summed mass /
               max weight in the bin; query: averaged / max over the rows in the bin)
  s<i>_rows    [Lyr,H,K,L] fp16 full-resolution rows for K target-region query positions
  s<i>_row_q   [K] the query positions of those rows
  s<i>_nnz / _m90 / _pmax / _ent / _argmax   [Lyr,H,L] per-row stats: support (#p>0),
               #keys holding 90% of the mass, max weight, entropy (nats), argmax key index
  s<i>_scaler  [Lyr,Hs,L] adaptive scale per (scaled head, query) for AS methods

Usage (from synthetic/, venv active, PROJECT_ROOT/DATA_PATH set):
  python3 ../experiments/attn_maps/extract.py --ckpt <ckpt> --task mqmtar --method asentmax_zi_w20k \
      --stems test_0_64 test_5_4096 --out-dir <dir> [--samples 4] [--map-samples 2]
"""
import argparse, json, math, os, subprocess, sys, time
import numpy as np
import torch
sys.path.insert(0, ".")

DATA = {
    "sort": "sort/40M-tr_32-64_vt_64-4096_vocab_32/",
    "reverse": "reverse/30M_tr-32-64_val_64-256_t-64-512/",
    "copy": "copy/20M_V-32_tr-32-64_val_64-512_t-64-4096/",
    "mqmtar": "mqmtar/50M_abc-256_vocab-10K_kv-len-2_num_kv-80_num_q-4/",
}
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def method_overrides(method):
    """Hydra overrides for `method`, extracted from run_one.sh's method block (single source of truth)."""
    run_one = os.path.join(REPO, "experiments", "osc", "run_one.sh")
    block = subprocess.check_output(["sed", "-n", "/^# ---------------- method overrides/,/^OV+=(/p", run_one], text=True)
    script = f'log() {{ echo "$*" >&2; }}; method={method}\n{block}\nprintf "%s\\n" "${{OV[@]}}"'
    out = subprocess.check_output(["bash", "-c", script], text=True)
    return [l for l in out.splitlines() if l.strip()]


def scaler_of(layer, hidden_states, q_len, k_len):
    """The adaptive (or learned / nakanishi) per-query scale the layer multiplies its NoPE queries by."""
    bsz = hidden_states.shape[0]
    lp = layer.log_position[:, :, k_len - q_len:k_len, :]
    t = layer.attn_scale_type
    if t == "nakanishi":
        return (lp * layer.attn_scale_beta).expand(bsz, layer.num_scaled_heads, q_len, 1)
    if t == "learn":
        beta, gamma = layer.attn_scale_beta, layer.attn_scale_gamma
    else:
        beta = torch.nn.functional.softplus(layer.attn_scale_beta_proj(hidden_states)).view(bsz, q_len, -1, 1).transpose(1, 2)
        gamma = layer.attn_scale_gamma_range * torch.tanh(layer.attn_scale_gamma_proj(hidden_states)).view(bsz, q_len, -1, 1).transpose(1, 2)
    return layer.attn_scale_delta + beta * (lp ** gamma)


class Recorder:
    """Per-layer collection for one sample; filled by the patched forward."""
    def __init__(self, n_layers, full, bins, row_q):
        self.n_layers, self.full, self.bins, self.row_q = n_layers, full, bins, row_q
        self.reset()

    def reset(self):
        self.layers = [dict() for _ in range(self.n_layers)]


def make_forward(layer, rec, chunk, enabled):
    orig = layer.forward

    def forward(hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, cache_position=None, **kw):
        if not enabled["on"]:
            return orig(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                        past_key_value=past_key_value, output_attentions=output_attentions,
                        use_cache=use_cache, cache_position=cache_position, **kw)
        assert hidden_states.shape[0] == 1, "batch 1 only (no padding)"
        assert not layer.apply_rotary, "NAPE models only"
        _, L, _ = hidden_states.shape
        H, D = layer.num_heads, layer.head_dim
        q = layer.q_proj(hidden_states).view(1, L, H, D).transpose(1, 2)
        k = layer.k_proj(hidden_states).view(1, L, layer.num_key_value_heads, D).transpose(1, 2)
        v = layer.v_proj(hidden_states).view(1, L, layer.num_key_value_heads, D).transpose(1, 2)
        from transformers.models.gemma2.modeling_gemma2 import repeat_kv
        k = repeat_kv(k, layer.num_key_value_groups); v = repeat_kv(v, layer.num_key_value_groups)
        st = rec.layers[layer.layer_idx]
        if layer.attn_scale_type:
            s = scaler_of(layer, hidden_states, L, L)                     # [1,Hs,L,1]
            q = q.clone(); q[:, -layer.num_scaled_heads:] = q[:, -layer.num_scaled_heads:] * s
            st["scaler"] = s[0, :, :, 0].float().cpu()
        # fp32 score path (q.k, scale, bias, map, p.v), matching the fused kernels (fp32 accumulation);
        # the model's own eager fallback rounds logits to bf16, which visibly perturbs high-q Stieltjes rows.
        q = q.float() * layer.qk_scale; k = k.float(); v = v.float()
        slopes = layer.alibi_slopes if (layer.apply_nape and layer.alibi_slopes is not None) else None
        B = rec.bins
        out = torch.empty(1, H, L, D, device=q.device, dtype=torch.float32)
        stats = {n: torch.empty(H, L) for n in ("nnz", "m90", "pmax", "ent")}
        stats["argmax"] = torch.empty(H, L, dtype=torch.int32)
        full = torch.empty(H, L, L, dtype=torch.float16) if rec.full else None
        if not rec.full:
            kbin = (torch.arange(L, device=q.device) * B // L)
            pool_sum = torch.zeros(H, B, B, device=q.device); pool_max = torch.zeros(H, B, B, device=q.device)
            rows_per_bin = torch.bincount(kbin, minlength=B).float()
        rows = torch.empty(H, len(rec.row_q), L, dtype=torch.float16)
        row_q = torch.as_tensor(rec.row_q, device=q.device)
        for a in range(0, L, chunk):
            b = min(L, a + chunk)
            s = torch.matmul(q[:, :, a:b], k.transpose(2, 3))                 # [1,H,r,L] fp32
            # ALiBi/NAPE: bias = slope * (j - i) for j <= i (same as make_bias_window), fp32 for this row block
            if slopes is not None:
                qi = torch.arange(a, b, device=s.device); kj = torch.arange(L, device=s.device)
                diff = (kj[None, :] - qi[:, None]).clamp(max=0).float()
                s = s + slopes.to(s.device).float().view(1, -1, 1, 1) * diff.view(1, 1, b - a, L)
            causal = torch.arange(L, device=s.device)[None, :] > torch.arange(a, b, device=s.device)[:, None]
            s = s.masked_fill(causal.view(1, 1, b - a, L), torch.finfo(torch.bfloat16).min)
            p = layer.attn_func(s)                                          # fp32 map, as in the eager path
            out[:, :, a:b] = torch.matmul(p.to(v.dtype), v)
            p = p[0]                                                        # [H,r,L]
            stats["nnz"][:, a:b] = (p > 0).sum(-1).float().cpu()
            ps, idx = p.sort(-1, descending=True)
            stats["m90"][:, a:b] = ((ps.cumsum(-1) < 0.9).sum(-1) + 1).float().cpu()
            stats["pmax"][:, a:b] = ps[..., 0].cpu(); stats["argmax"][:, a:b] = idx[..., 0].int().cpu()
            stats["ent"][:, a:b] = -(p * p.clamp_min(1e-30).log()).sum(-1).cpu()
            del ps, idx
            if full is not None:
                full[:, a:b] = p.half().cpu()
            else:
                pk = torch.zeros(H, b - a, B, device=p.device).index_add_(2, kbin, p)          # key-binned mass
                mk = torch.zeros(H, b - a, B, device=p.device).scatter_reduce_(
                    2, kbin.view(1, 1, -1).expand_as(p), p, reduce="amax")
                qb = kbin[a:b]
                pool_sum.index_add_(1, qb, pk)
                pool_max.scatter_reduce_(1, qb.view(1, -1, 1).expand_as(mk), mk, reduce="amax")
            sel = ((row_q >= a) & (row_q < b)).nonzero().flatten()
            for t in sel.tolist():
                rows[:, t] = p[:, row_q[t] - a].half().cpu()
        st.update({n: x for n, x in stats.items()})
        st["rows"] = rows
        if full is not None:
            st["full"] = full
        else:
            st["pool_mean"] = (pool_sum / rows_per_bin.view(1, -1, 1).clamp_min(1)).half().cpu()
            st["pool_max"] = pool_max.half().cpu()
        o = out.to(hidden_states.dtype).transpose(1, 2).reshape(1, L, -1)
        return layer.o_proj(o), None, past_key_value

    return forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--task", required=True, choices=list(DATA))
    ap.add_argument("--method", required=True); ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--tag", default=None)
    ap.add_argument("--samples", type=int, default=4, help="samples per stem for row stats/rows")
    ap.add_argument("--map-samples", type=int, default=2, help="of those, how many keep full/pooled maps")
    ap.add_argument("--full-max", type=int, default=640); ap.add_argument("--bins", type=int, default=512)
    ap.add_argument("--n-rows", type=int, default=8); ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--data-root", default="${oc.env:DATA_PATH}")
    ap.add_argument("--data-path", default=None, help="full dataset dir (overrides --data-root/<task dir>)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    tag = a.tag or a.method
    data_path = a.data_path or f"{a.data_root}/{DATA[a.task]}"

    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    ov = method_overrides(a.method)
    with initialize_config_dir(version_base="1.3", config_dir=os.path.abspath("configs")):
        cfg = compose(config_name="train.yaml", overrides=[
            f"experiment=entmax/{a.task}", "logger=csv", "task_name=attn_maps",
            f"data.data_provider.path={data_path}", *ov])
    with open_dict(cfg):
        cfg.data.batch_config.test.size = 1
        cfg.data.data_loader_kwargs.num_workers = 0
    dm = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)(tokenizer=dm.tokenizer)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=True)
    model = model.cuda().to(torch.bfloat16).eval()       # trainer precision is bf16-true

    from src.models.architectures.sparse_gemma import SparseGemma2Attention
    layers = [m for m in model.modules() if isinstance(m, SparseGemma2Attention)]
    L0 = layers[0]
    enabled = {"on": False}
    rec = Recorder(len(layers), full=True, bins=a.bins, row_q=[])
    for layer in layers:
        layer.forward = make_forward(layer, rec, a.chunk, enabled)

    for stem in a.stems:
        out_path = os.path.join(a.out_dir, f"{a.task}__{tag}__{stem}.npz")
        if os.path.exists(out_path):
            print("exists, skip", out_path); continue
        with open_dict(cfg):
            cfg.data.data_provider.file_subset = {"test": [stem]}
        dm = hydra.utils.instantiate(cfg.data); dm.setup("test")
        loader = dm.test_dataloader()
        if isinstance(loader, (list, tuple)):
            if not loader: raise SystemExit(f"no test split matching stem {stem} under {data_path}")
            loader = loader[0]
        arrays, sanity = {}, {}
        t0 = time.time()
        for i, batch in enumerate(loader):
            if i >= a.samples: break
            ids = batch["input_ids"].cuda(); mask = batch["input_mask"].cuda()
            tmask = batch["target_mask"][0]; tgt = batch["targets"][0]
            L = ids.shape[1]
            tpos = tmask.nonzero().flatten().tolist()
            pick = np.unique(np.linspace(0, len(tpos) - 1, min(a.n_rows, len(tpos))).round().astype(int))
            rec.row_q = [tpos[j] for j in pick]
            rec.full = L <= a.full_max
            with torch.no_grad():
                if i == 0:  # sanity: patched eager forward vs the trained (original) path
                    enabled["on"] = False
                    lo_ref = model.net(ids, mask).logits.float()
                enabled["on"] = True; rec.reset()
                lo = model.net(ids, mask).logits.float()
            pred = lo[0].argmax(-1).cpu()
            acc_tok = (pred[tmask] == tgt[tmask]).float().mean().item()
            seq_ok = bool((pred[tmask] == tgt[tmask]).all())
            sanity.setdefault("tf_token_acc", []).append(acc_tok); sanity.setdefault("tf_seq_exact", []).append(seq_ok)
            if i == 0:
                d = (lo - lo_ref).abs().max().item()
                agree = (lo_ref[0].argmax(-1).cpu()[tmask] == pred[tmask]).float().mean().item()
                sanity.update(seq_len=L, max_abs_logit_diff=d, argmax_agree_vs_trained_path=agree,
                              logit_scale=lo_ref.abs().max().item())
            pre = f"s{i}_"
            arrays[pre + "input_ids"] = ids[0].cpu().numpy().astype(np.int32)
            arrays[pre + "target_mask"] = tmask.numpy()
            arrays[pre + "row_q"] = np.array(rec.row_q, dtype=np.int32)
            keys = ["nnz", "m90", "pmax", "ent", "argmax", "rows"]
            if i < a.map_samples:
                keys += ["full"] if rec.full else ["pool_mean", "pool_max"]
            if L0.attn_scale_type: keys.append("scaler")
            for kname in keys:
                x = torch.stack([st[kname] for st in rec.layers]).numpy()
                arrays[pre + kname] = x.astype(np.float16) if x.dtype == np.float32 and kname not in ("nnz", "m90") else x
            rec.reset()
        meta = dict(task=a.task, method=a.method, tag=tag, ckpt=a.ckpt, stem=stem, n_samples=min(a.samples, i + 1),
                    n_layers=len(layers), n_heads=L0.num_heads,
                    alibi_slopes=L0.alibi_slopes.tolist() if L0.alibi_slopes is not None else None,
                    attn_type=L0.attn_type, entmax_alpha=L0.entmax_alpha, attn_scale_type=L0.attn_scale_type,
                    stieltjes_q=getattr(L0, "stieltjes_q", None), stieltjes_window=getattr(L0, "stieltjes_window", None),
                    num_scaled_heads=getattr(L0, "num_scaled_heads", None), bins=a.bins, full_max=a.full_max,
                    overrides=ov, sanity=sanity, seconds=round(time.time() - t0, 1))
        arrays["meta_json"] = np.array(json.dumps(meta))
        np.savez_compressed(out_path, **arrays)
        s = sanity
        print(f"[{tag} {stem}] L={s['seq_len']} logit|diff|={s['max_abs_logit_diff']:.3g} (scale {s['logit_scale']:.3g}) "
              f"argmax-agree={s['argmax_agree_vs_trained_path']:.3f} tf-tok-acc={np.mean(s['tf_token_acc']):.3f} "
              f"tf-exact={np.mean(s['tf_seq_exact']):.2f} {meta['seconds']}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
