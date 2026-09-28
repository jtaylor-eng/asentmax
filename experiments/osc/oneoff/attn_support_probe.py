#!/usr/bin/env python3
"""Attention-support diagnostic for the MQMTAR plateau.

For a checkpoint, run a few real training batches through the model with attention forced onto the
eager path and record, per layer/head, the mean number of keys with non-zero weight (support size),
the mean max weight, and (for adaptive-scale models) the per-query scaler distribution. Mechanism
under test (reproduction_0920.md §3.4): ASEntmax on the plateau has near-one-hot rows (no gradient to
the correct key), dense/windowed Stieltjes keep a wider support.

Usage (from synthetic/, venv active, env PROJECT_ROOT/DATA_PATH set):
  python3 ../experiments/osc/oneoff/attn_support_probe.py --ckpt <last.ckpt> --task mqmtar \
      --method asentmax_w20k --data '${oc.env:DATA_PATH}/mqmtar/...' [--batches 4] [--out probe.json]
Method overrides come from run_one.sh's method block (same extraction as the smoke script).
"""
import argparse, json, math, os, subprocess, sys
import torch
sys.path.insert(0, ".")


def method_overrides(method):
    run_one = os.path.join(os.path.dirname(__file__), "..", "run_one.sh")
    block = subprocess.check_output(["sed", "-n", "/^# ---------------- method overrides/,/^OV+=(/p", run_one], text=True)
    script = f'log() {{ :; }}; method={method}\n{block}\nprintf "%s\\n" "${{OV[@]}}"'
    out = subprocess.check_output(["bash", "-c", script], text=True)
    return [l for l in out.splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--task", required=True)
    ap.add_argument("--method", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--batches", type=int, default=4); ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    cfg_dir = os.path.abspath("configs")
    with initialize_config_dir(version_base="1.3", config_dir=cfg_dir):
        cfg = compose(config_name="train.yaml", overrides=[f"experiment=entmax/{a.task}", "logger=csv",
                      f"data.data_provider.path={a.data}", "task_name=probe", *method_overrides(a.method)])
    with open_dict(cfg):
        # eager everywhere so attention weights are materialised by self.attn_func
        cfg.model.net.use_fast_attn = False
        if cfg.model.net.get("attn_type", "regular") == "stieltjes":
            cfg.model.net.stieltjes_impl = "eager"
        # the eager entmax path builds a dense (max_pos x max_pos) ALiBi tensor; the repo's 131k
        # max_position_embeddings would be 137 GB. Training sequences here are <= 2 x max train len.
        cfg.model.net.max_position_embeddings = 4096
    dm = hydra.utils.instantiate(cfg.data); dm.setup("fit")
    model = hydra.utils.instantiate(cfg.model)(tokenizer=dm.tokenizer)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=True); model = model.cuda().to(torch.bfloat16).eval()  # trainer precision bf16-true

    from src.models.architectures.sparse_gemma import SparseGemma2Attention
    layers = [m for m in model.modules() if isinstance(m, SparseGemma2Attention)]
    stats = {i: dict(nnz=[], pmax=[], scaler=[]) for i in range(len(layers))}

    # Wrap attn_func to capture the weights it returns; wrap _apply_length_scaling to capture the scaler.
    for i, layer in enumerate(layers):
        f = layer.attn_func
        def wrapped(x, _f=f, _i=i):
            p = _f(x)
            nz = (p > 0).sum(-1).float()          # support per row
            stats[_i]["nnz"].append(nz.mean().item())
            stats[_i]["pmax"].append(p.max(-1).values.mean().item())
            return p
        layer.attn_func = wrapped
        if layer.attn_scale_type == "adapt-softplus-tanh":
            g = layer._apply_length_scaling
            def wrapped_scale(q, h, ql, kl, _layer=layer, _i=i, _g=g):
                bsz = q.shape[0]
                beta = torch.nn.functional.softplus(_layer.attn_scale_beta_proj(h)).view(bsz, ql, -1, 1).transpose(1, 2)
                gamma = _layer.attn_scale_gamma_range * torch.tanh(_layer.attn_scale_gamma_proj(h)).view(bsz, ql, -1, 1).transpose(1, 2)
                lp = _layer.log_position[:, :, kl - ql:kl, :]
                s = (_layer.attn_scale_delta + beta * (lp ** gamma)).float().flatten()
                stats[_i]["scaler"].append([s.quantile(t).item() for t in (0.05, 0.5, 0.95)] + [s.max().item()])
                return _g(q, h, ql, kl)
            layer._apply_length_scaling = wrapped_scale

    loader = dm.train_dataloader(); n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
            model.model_step("train", batch, n)
            n += 1
            if n >= a.batches: break

    out = {"ckpt": a.ckpt, "method": a.method, "batches": n, "layers": {}}
    for i, s in stats.items():
        row = dict(nnz_mean=sum(s["nnz"]) / max(len(s["nnz"]), 1), pmax_mean=sum(s["pmax"]) / max(len(s["pmax"]), 1))
        if s["scaler"]:
            q = list(zip(*s["scaler"])); row["scaler_p5_p50_p95_max"] = [sum(x) / len(x) for x in q]
        out["layers"][i] = row
        print(f"layer {i}: support {row['nnz_mean']:6.2f} keys/row   max-weight {row['pmax_mean']:.3f}"
              + (f"   scaler p5/p50/p95/max {'/'.join(f'{v:.2f}' for v in row['scaler_p5_p50_p95_max'])}" if "scaler_p5_p50_p95_max" in row else ""))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
