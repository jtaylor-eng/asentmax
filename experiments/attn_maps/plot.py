#!/usr/bin/env python3
"""Plot attention maps from extract.py's .npz files.

Per (task, stem):
  <task>__<stem>__compare.png   rows = mapping family (primary run), cols = layers; for each layer the
                                 "content head" (highest mean max-weight on target-region queries whose
                                 argmax key is > 2 positions back, i.e. not a local/diagonal head).
                                 Log colour scale, floor 1e-4; exact zeros (sparse maps) are white.
                                 L <= 640: the full map; longer: 512x512 max-pooled map (max weight per bin).
  <task>__<stem>__rows.png       same heads, full-resolution weight vs key position for 8 target queries (log y)
Per (task, run, stem), primary runs only:
  <task>__<run>__<stem>__allheads.png   layers x heads grid of maps
Per task:
  <task>__support.png            per layer: #keys holding 90% of the mass and #nonzero keys, mean over
                                 target-region queries, ID vs OOD, one line per family
Usage: python3 plot.py --in-dir <npz dir> --out-dir <png dir> [--manifest manifest.tsv]
"""
import argparse, csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

FAMILIES = ["softmax", "entmax", "asentmax", "stieltjes", "wstieltjes", "aswstieltjes"]
FLOOR = 1e-4
CMAP = plt.get_cmap("magma_r").copy(); CMAP.set_under("white"); CMAP.set_bad("white")


def load(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    return meta, z


def label(meta, fam=None):
    m = meta["method"]; t = meta["tag"]
    return f"{fam + ': ' if fam else ''}{t}"


def map_of(z, s, layer, head):
    """(matrix, extent, pooled?) for sample s."""
    if f"s{s}_full" in z.files:
        return z[f"s{s}_full"][layer, head].astype(np.float32), False
    return z[f"s{s}_pool_max"][layer, head].astype(np.float32), True


def content_head(z, meta, s, layer):
    tm = z[f"s{s}_target_mask"]
    qpos = np.nonzero(tm)[0]
    pmax = z[f"s{s}_pmax"][layer][:, qpos].astype(np.float32)        # [H,T]
    am = z[f"s{s}_argmax"][layer][:, qpos]
    far = ((qpos[None, :] - am) > 2) & (am > 0)       # not local/diagonal, not the BOS sink
    score = (pmax * far).mean(-1)
    return int(score.argmax()), score


def head_kind(meta, h):
    sl = meta.get("alibi_slopes")
    return "NoPE" if (sl is None or sl[h] == 0) else f"ALiBi {sl[h]:.2g}"


def show_map(ax, M, pooled, L, tm=None):
    M = np.where(M > 0, np.maximum(M, FLOOR * 1.0001), 0.0)
    ax.imshow(M, cmap=CMAP, norm=LogNorm(vmin=FLOOR, vmax=1.0), interpolation="nearest",
              extent=(0, L, L, 0), aspect="auto")
    if tm is not None:
        t0 = int(np.argmax(tm))
        ax.axhline(t0, color="tab:blue", lw=0.5, alpha=0.6); ax.axvline(t0, color="tab:blue", lw=0.5, alpha=0.6)
    ax.set_xticks([]); ax.set_yticks([])


def pick_runs(files, manifest):
    """{(task, family): npz-prefix tag} for primary runs, from the manifest."""
    prim = {}
    for r in manifest:
        if r["primary"] == "1":
            prim[(r["task"], r["family"])] = f"{r['method']}_s{r['seed']}_lr{r['lr']}"
    return prim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", default=None); ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--no-allheads", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.in_dir, "*.npz")))
    man = list(csv.DictReader(open(a.manifest or os.path.join(a.in_dir, "manifest.tsv")), delimiter="\t"))
    prim = pick_runs(files, man)
    by = {}   # (task, stem) -> [(family, path)]
    for (task, fam), tag in prim.items():
        for p in glob.glob(os.path.join(a.in_dir, f"{task}__{tag}__*.npz")):
            stem = os.path.basename(p)[:-4].split("__")[-1]
            by.setdefault((task, stem), []).append((fam, p))
    s = a.sample
    support = {}  # task -> fam -> stem -> (m90[layers], nnz[layers])
    for (task, stem), lst in sorted(by.items()):
        lst.sort(key=lambda t: FAMILIES.index(t[0]))
        metas = [load(p) for _, p in lst]
        nl = max(m["n_layers"] for m, _ in metas)
        # ---- compare heatmaps + rows
        fig, axs = plt.subplots(len(lst), nl, figsize=(2.6 * nl, 2.6 * len(lst)), squeeze=False)
        fig2, axs2 = plt.subplots(len(lst), nl, figsize=(3.2 * nl, 2.2 * len(lst)), squeeze=False)
        for i, ((fam, _), (meta, z)) in enumerate(zip(lst, metas)):
            ids = z[f"s{s}_input_ids"]; L = len(ids); tm = z[f"s{s}_target_mask"]
            san = meta["sanity"]
            for l in range(nl):
                ax, ax2 = axs[i, l], axs2[i, l]
                if l >= meta["n_layers"]:
                    ax.axis("off"); ax2.axis("off"); continue
                h, _ = content_head(z, meta, s, l)
                M, pooled = map_of(z, s, l, h)
                show_map(ax, M, pooled, L, tm)
                ax.set_title(f"L{l} h{h} ({head_kind(meta, h)})", fontsize=7)
                rows = z[f"s{s}_rows"][l, h].astype(np.float32); rq = z[f"s{s}_row_q"]
                for r in range(rows.shape[0]):
                    y = rows[r, : rq[r] + 1][::-1]                      # index = distance back from the query
                    dist = np.arange(1, len(y) + 1); nz = y > 0
                    ax2.plot(dist[nz], y[nz], ".", ms=1.5, alpha=0.7, color=f"C{r}")
                ax2.set_xscale("log"); ax2.set_yscale("log")
                ax2.set_ylim(1e-6, 1.5); ax2.set_xlim(1, L); ax2.tick_params(labelsize=6)
                ax2.set_title(f"L{l} h{h}", fontsize=7)
                if l == 0:
                    lab = f"{fam}\n{meta['tag']}\ntf-exact {np.mean(san['tf_seq_exact']):.2f}"
                    ax.set_ylabel(lab, fontsize=7); ax2.set_ylabel(lab, fontsize=7)
            # support stats over all samples
            m90 = []; nnz = []
            for k in range(meta["n_samples"]):
                tmk = z[f"s{k}_target_mask"]
                m90.append(z[f"s{k}_m90"][:, :, tmk].mean(axis=(1, 2))); nnz.append(z[f"s{k}_nnz"][:, :, tmk].mean(axis=(1, 2)))
            support.setdefault(task, {}).setdefault(fam, {})[stem] = (np.mean(m90, 0), np.mean(nnz, 0), L, meta["tag"])
        kind = "full map" if L <= metas[0][0]["full_max"] else "512x512 max-pooled"
        fig.suptitle(f"{task} {stem} (L={L}, sample {s}, {kind}; log colour {FLOOR:g}..1, white = 0; blue = target start)", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, f"{task}__{stem}__compare.png"), dpi=130); plt.close(fig)
        fig2.suptitle(f"{task} {stem}: weight vs distance back from the query (log-log; key 1 = self) for 8 target queries; "
                      f"only nonzero weights drawn", fontsize=9)
        fig2.tight_layout(); fig2.savefig(os.path.join(a.out_dir, f"{task}__{stem}__rows.png"), dpi=130); plt.close(fig2)
        print("wrote", task, stem)
        # ---- all heads per run
        if not a.no_allheads:
            for (fam, p), (meta, z) in zip(lst, metas):
                nl_, nh = meta["n_layers"], meta["n_heads"]; L = len(z[f"s{s}_input_ids"]); tm = z[f"s{s}_target_mask"]
                fig, axs = plt.subplots(nl_, nh, figsize=(1.3 * nh, 1.4 * nl_), squeeze=False)
                for l in range(nl_):
                    for h in range(nh):
                        M, pooled = map_of(z, s, l, h); show_map(axs[l, h], M, pooled, L, tm)
                        axs[l, h].set_title(f"L{l}h{h} {'N' if head_kind(meta, h) == 'NoPE' else 'A'}", fontsize=6)
                fig.suptitle(f"{task} {stem} {fam}: {meta['tag']} (N = NoPE head, A = ALiBi head)", fontsize=9)
                fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, f"{task}__{meta['tag']}__{stem}__allheads.png"), dpi=110); plt.close(fig)
    # ---- support summary
    for task, fams in support.items():
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
        for fam in FAMILIES:
            if fam not in fams: continue
            for j, stem in enumerate(sorted(fams[fam], key=lambda st: fams[fam][st][2])):
                m90, nnz, L, tag = fams[fam][stem]
                c = f"C{FAMILIES.index(fam)}"; ls = "-" if j == 0 else "--"
                axs[0].plot(m90, ls, color=c, marker="o", ms=3, label=f"{fam} L={L}")
                axs[1].plot(nnz, ls, color=c, marker="o", ms=3, label=f"{fam} L={L}")
        for ax, t in zip(axs, ["#keys for 90% of mass", "#nonzero keys (support)"]):
            ax.set_yscale("log"); ax.set_xlabel("layer"); ax.set_title(f"{task}: {t}, mean over heads x target queries", fontsize=9)
        axs[1].legend(fontsize=6, ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, f"{task}__support.png"), dpi=130); plt.close(fig)
        print("wrote", task, "support")


if __name__ == "__main__":
    main()
