#!/usr/bin/env python3
"""Long-context tail statistics: how attention on the *correct* key evolves with sequence length n.

For each test sample the correct key of every target query is known from the task structure:
  copy:   the query at input position i predicts trg[j] (i = src_len + 1 + j); its source token sits at
          input position 1 + j (verified by token equality).
  mqmtar: the target is [2 v1 v2] per query; v1/v2 are copied from the KV pair whose key bigram
          matches the query's key bigram ("k1 k2 1 v1 v2" in the source); the correct key for the
          query predicting v1 (v2) is the source position of v1 (v2). "2"/EOS predictions have no key.
The model runs its normal (trained, fast-kernel) forward; a hook on every attention layer recomputes
the score rows of the T target queries only (fp32, same scale / NAPE bias / adaptive scale / map as
the eager path; checked against the layer output, `rel_err` below), so n up to 65k is cheap.

Per (layer, head, target query) it stores, with R = correct key position and offsets o in -2..2:
  p_off[5]  attention weight on key R+o          s_off[5]  score on key R+o
  p_win     mass on R-2..R+2                     s_out     max score outside R-2..R+2 (best distractor)
  s_max     max score                            s_minsup  min score with p > 0 (support cutoff)
  pmax, nnz, m90 (#keys for 90% mass), ent, rank (#keys with score > s[R])
plus the adaptive scaler at the target queries (AS rows), teacher-forced correctness per target
query, and per-layer rel_err = |o_proj(p.v) - layer output| / |layer output| on those rows.

Output: <out-dir>/<task>__<tag>.npz with arrays keyed <stem>__<name>, each [S, Lyr, H, T] etc.
Usage (from synthetic/): python3 ../experiments/attn_maps/tail.py --ckpt C --task mqmtar --method M \
    --tag T --stems test_0_64 ... --out-dir D [--samples 8]
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import DATA, method_overrides, scaler_of   # noqa: E402

OFFS = (-2, -1, 0, 1, 2)


def correct_keys(task, ids, tmask, targets, sep_id):
    """-> (query positions Q, correct key positions R) for target queries that have a correct key."""
    ids = ids.tolist(); sep = ids.index(sep_id)
    src = ids[1:sep]                                     # input pos of src[k] = 1 + k
    qpos = [i for i in range(len(ids)) if tmask[i]]
    Q, R = [], []
    if task == "copy":
        for j, i in enumerate(qpos):
            if j < len(src) and targets[i] == src[j]:
                Q.append(i); R.append(1 + j)
    elif task == "mqmtar":
        qb = src[-12:]                                   # num_q=4 queries of "2 k1 k2"
        assert qb[0::3] == [2] * 4, qb
        kv = {}
        for p in range(len(src) - 4 - 12):
            if src[p + 2] == 1:
                kv.setdefault((src[p], src[p + 1]), []).append(p + 3)
        for j, i in enumerate(qpos):
            qi, r = divmod(j, 3)
            if qi >= 4 or r == 0: continue
            key = (qb[3 * qi + 1], qb[3 * qi + 2])
            if key not in kv: continue
            vpos = kv[key][-1] + (r - 1)                 # src index of v1 / v2
            if src[vpos] == targets[i]:
                Q.append(i); R.append(1 + vpos)
    else:
        raise ValueError(task)
    return Q, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--task", required=True, choices=["copy", "mqmtar"])
    ap.add_argument("--method", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--stems", nargs="+", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--samples", type=int, default=8); ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-q", type=int, default=256, help="max target queries per sample (copy)")
    ap.add_argument("--data-path", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, f"{a.task}__{a.tag}.npz")
    data_path = a.data_path or f"${{oc.env:DATA_PATH}}/{DATA[a.task]}"

    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    ov = method_overrides(a.method)
    with initialize_config_dir(version_base="1.3", config_dir=os.path.abspath("configs")):
        cfg = compose(config_name="train.yaml", overrides=[
            f"experiment=entmax/{a.task}", "logger=csv", "task_name=attn_tail",
            f"data.data_provider.path={data_path}", *ov])
    with open_dict(cfg):
        cfg.data.batch_config.test.size = 1
        cfg.data.data_loader_kwargs.num_workers = 0
    dm = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)(tokenizer=dm.tokenizer)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    model = model.cuda().to(torch.bfloat16).eval()
    sep_id = dm.tokenizer.sep_token_id

    from src.models.architectures.sparse_gemma import SparseGemma2Attention
    from transformers.models.gemma2.modeling_gemma2 import repeat_kv
    layers = [m for m in model.modules() if isinstance(m, SparseGemma2Attention)]
    L0 = layers[0]; H = L0.num_heads
    ctx = {}   # Q, R tensors for the current sample; per-layer results

    def hook(layer):
        orig = layer.forward

        def forward(*args, **kw):
            res = orig(*args, **kw)
            hs = kw["hidden_states"] if "hidden_states" in kw else args[0]
            Q, R = ctx["Q"], ctx["R"]; T = len(Q); L = hs.shape[1]; D = layer.head_dim
            q = layer.q_proj(hs).view(1, L, H, D).transpose(1, 2)[:, :, Q]          # [1,H,T,D]
            k = repeat_kv(layer.k_proj(hs).view(1, L, layer.num_key_value_heads, D).transpose(1, 2), layer.num_key_value_groups)
            v = repeat_kv(layer.v_proj(hs).view(1, L, layer.num_key_value_heads, D).transpose(1, 2), layer.num_key_value_groups)
            st = {}
            if layer.attn_scale_type:
                sc = scaler_of(layer, hs, L, L)[:, :, Q]                               # [1,Hs,T,1]
                q = q.clone(); q[:, -layer.num_scaled_heads:] = q[:, -layer.num_scaled_heads:] * sc
                st["scaler"] = sc[0, :, :, 0].float().cpu().numpy()
            q = q.float() * layer.qk_scale; k = k.float()[0]; v = v.float()[0]
            slopes = layer.alibi_slopes.to(q.device).float() if (layer.apply_nape and layer.alibi_slopes is not None) else None
            names = ["p_off", "s_off", "p_win", "s_out", "s_max", "s_minsup", "pmax", "nnz", "m90", "ent", "rank"]
            acc = {n: [] for n in names}; outs = []
            kj = torch.arange(L, device=q.device)
            for a0 in range(0, T, a.chunk):
                qi = torch.as_tensor(Q[a0:a0 + a.chunk], device=q.device); ri = torch.as_tensor(R[a0:a0 + a.chunk], device=q.device)
                s = torch.einsum("htd,hld->htl", q[0, :, a0:a0 + a.chunk], k)            # [H,t,L]
                if slopes is not None:
                    s = s + slopes.view(-1, 1, 1) * (kj[None, :] - qi[:, None]).clamp(max=0).float()[None]
                causal = kj[None, :] > qi[:, None]
                s = s.masked_fill(causal[None], torch.finfo(torch.bfloat16).min)
                p = layer.attn_func(s.unsqueeze(0))[0]                                 # [H,t,L] fp32
                outs.append(torch.einsum("htl,hld->htd", p, v))
                idx = (ri[:, None] + torch.tensor(OFFS, device=q.device)[None]).clamp(0)    # [t,5]
                valid = (idx <= qi[:, None])
                gi = idx[None].expand(H, -1, -1)
                po = p.gather(2, gi) * valid; so = s.gather(2, gi).masked_fill(~valid[None], float("nan"))
                win = (kj[None, :] >= (ri - 2)[:, None]) & (kj[None, :] <= (ri + 2)[:, None])
                acc["p_off"].append(po); acc["s_off"].append(so)
                acc["p_win"].append((p * win[None]).sum(-1))
                acc["s_out"].append(s.masked_fill(win[None] | causal[None], -float("inf")).max(-1).values)
                acc["s_max"].append(s.max(-1).values)
                acc["s_minsup"].append(s.masked_fill(p <= 0, float("inf")).min(-1).values)
                ps = p.sort(-1, descending=True).values
                acc["pmax"].append(ps[..., 0]); acc["nnz"].append((p > 0).sum(-1).float())
                acc["m90"].append(((ps.cumsum(-1) < 0.9).sum(-1) + 1).float())
                acc["ent"].append(-(p * p.clamp_min(1e-30).log()).sum(-1))
                sR = s.gather(2, ri.view(1, -1, 1).expand(H, -1, 1))
                acc["rank"].append((s > sR).sum(-1).float())
                del s, p, ps
            for n in names:
                st[n] = torch.cat(acc[n], 1).cpu().numpy()                             # [H,T(,5)]
            o = torch.cat(outs, 1).transpose(0, 1).reshape(1, T, -1).to(hs.dtype)       # [1,T,H*D]
            mine = layer.o_proj(o).float(); ref = res[0][:, Q].float()
            st["rel_err"] = ((mine - ref).norm() / ref.norm().clamp_min(1e-12)).item()
            ctx["layers"][layer.layer_idx] = st
            return res
        return forward

    for layer in layers:
        layer.forward = hook(layer)

    arrays = {}
    old = dict(np.load(out_path)) if os.path.exists(out_path) else {}
    summary = []
    for stem in a.stems:
        if f"{stem}__p_off" in old:
            print("cached", stem); arrays.update({k: v for k, v in old.items() if k.startswith(stem + "__")}); continue
        with open_dict(cfg):
            cfg.data.data_provider.file_subset = {"test": [stem]}
        dm = hydra.utils.instantiate(cfg.data); dm.setup("test")
        loader = dm.test_dataloader()
        if isinstance(loader, (list, tuple)):
            if not loader: raise SystemExit(f"no test split {stem}")
            loader = loader[0]
        per = []; t0 = time.time()
        for si, batch in enumerate(loader):
            if si >= a.samples: break
            ids = batch["input_ids"][0]; tm = batch["target_mask"][0].numpy(); tg = batch["targets"][0].numpy()
            Q, R = correct_keys(a.task, ids, tm, tg, sep_id)
            if len(Q) > a.max_q:                         # copy: evenly spaced subset of target queries
                sel = np.unique(np.linspace(0, len(Q) - 1, a.max_q).round().astype(int))
                Q = [Q[j] for j in sel]; R = [R[j] for j in sel]
            ctx.update(Q=Q, R=R, layers={})
            with torch.no_grad():
                lo = model.net(ids[None].cuda(), batch["input_mask"].cuda()).logits[0]
            pred = lo.argmax(-1).cpu().numpy()
            rec = {n: np.stack([ctx["layers"][l][n] for l in range(len(layers))]) for n in ctx["layers"][0] if n != "rel_err"}
            rec["rel_err"] = np.array([ctx["layers"][l]["rel_err"] for l in range(len(layers))])
            rec["correct"] = (pred[Q] == tg[Q]); rec["seq_exact"] = np.array(bool((pred[tm] == tg[tm]).all()))
            rec["Q"] = np.array(Q); rec["R"] = np.array(R); rec["L"] = np.array(len(ids))
            per.append(rec)
        T_AX = {"correct": 0, "Q": 0, "R": 0}          # everything else with a query axis has it at 2
        for n in per[0]:
            x = [r[n] for r in per]
            if n in ("rel_err", "seq_exact", "L"):
                arrays[f"{stem}__{n}"] = np.stack(x); continue
            ax = T_AX.get(n, 2); Tm = max(r.shape[ax] for r in x)
            fill = False if x[0].dtype == bool else (-1 if x[0].dtype.kind == "i" else np.nan)
            pad = []
            for r in x:
                w = [(0, 0)] * r.ndim; w[ax] = (0, Tm - r.shape[ax])
                pad.append(np.pad(r, w, constant_values=fill))
            arrays[f"{stem}__{n}"] = np.stack(pad)     # [S, ...], query axis NaN/-1-padded to the longest sample
        Tn = max(len(r["Q"]) for r in per)
        L = int(per[0]["L"]); err = max(r["rel_err"].max() for r in per)
        tok = np.mean([r["correct"].mean() for r in per]); ex = np.mean([r["seq_exact"] for r in per])
        pw = arrays[f"{stem}__p_win"]
        print(f"[{a.tag} {stem}] L={L} T={Tn} n_samp={len(per)} max rel_err={err:.2e} tf-tok={tok:.3f} tf-exact={ex:.2f} "
              f"p_win max-head per layer={np.round(pw.mean(axis=(0, 3)).max(-1), 3).tolist()} {time.time() - t0:.1f}s", flush=True)
        summary.append(dict(stem=stem, L=L, rel_err=err, tf_tok=tok, tf_exact=ex))
        meta = dict(task=a.task, method=a.method, tag=a.tag, ckpt=a.ckpt, offsets=OFFS, n_layers=len(layers), n_heads=H,
                    alibi_slopes=L0.alibi_slopes.tolist() if L0.alibi_slopes is not None else None,
                    num_scaled_heads=getattr(L0, "num_scaled_heads", None), attn_scale_type=L0.attn_scale_type)
        arrays["meta_json"] = np.array(json.dumps(meta))
        np.savez_compressed(out_path, **arrays)         # rewrite after every stem (progress survives timeouts)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
