#!/usr/bin/env python3
"""Plots for tail.py output: how attention on the correct key evolves with length n (copy, mqmtar).

Head choice: per run, the "retrieval head" = (layer, head) with the largest mean window mass (mass on
keys R-2..R+2 around the correct key R) at the ID length; that same head is tracked across n
(solid). Dashed = best head at each n (max over all heads), an upper envelope that tolerates the
role moving between heads.

<task>__tail_main.png   (a) window mass on the correct key vs n   (b) leakage 1 - window mass, log-log
                        (c) test exact-match ladder (runs_0920.json) and teacher-forced token accuracy
<task>__tail_mech.png   (a) score gap: max score in the window minus the best distractor score outside it
                        (b) fraction of target queries whose correct key is not the top-scoring key
                        (c) support size (#keys with p > 0) of the retrieval head
                        (d) adaptive scaler of the retrieval head at the target queries (AS rows; NaN if it is an ALiBi head)
Usage: python3 plot_tail.py --in-dir <attn_tail dir> --out-dir <png dir> [--runs context/runs_0920.json]
"""
import argparse, csv, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILIES = ["softmax", "entmax", "asentmax", "stieltjes", "wstieltjes", "aswstieltjes"]
LADDER_N = {"ID": 64, "2x": 128, "4x": 256, "8x": 512, "16x": 1024, "32x": 2048, "64x": 4096, "256x": 16384, "1024x": 65536}


def stems_of(z):
    st = sorted({k.split("__")[0] for k in z.files if "__" in k}, key=lambda s: int(s.split("_")[-1]))
    return st, [int(s.split("_")[-1]) for s in st]


def nanmean(x, axis=None):
    with np.errstate(all="ignore"):
        import warnings; warnings.simplefilter("ignore")
        return np.nanmean(x, axis=axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--runs", default=None); ap.add_argument("--manifest", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    man = list(csv.DictReader(open(a.manifest or os.path.join(a.in_dir, "manifest.tsv")), delimiter="\t"))
    runs = json.load(open(a.runs)) if a.runs else []
    for task in sorted({r["task"] for r in man}):
        rows = sorted([r for r in man if r["task"] == task and r["primary"] == "1"], key=lambda r: FAMILIES.index(r["family"]))
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
        fig2, ax2 = plt.subplots(1, 4, figsize=(20, 4.4))
        table = []
        for r in rows:
            tag = f"{r['method']}_s{r['seed']}_lr{r['lr']}"
            p = os.path.join(a.in_dir, f"{task}__{tag}.npz")
            if not os.path.exists(p): print("missing", p); continue
            z = np.load(p); meta = json.loads(str(z["meta_json"]))
            stems, ns = stems_of(z)
            c = f"C{FAMILIES.index(r['family'])}"
            pick = " [best test ladder of val-tied]" if r.get("pick") == "best_ladder" else ""
            lab = f"{r['family']}: {r['method']} s{r['seed']}{pick}"
            pw0 = nanmean(z[f"{stems[0]}__p_win"], axis=(0, 3))            # [Lyr,H] at ID
            l, h = np.unravel_index(np.nanargmax(pw0), pw0.shape)
            hk = "NoPE" if (meta["alibi_slopes"] is None or meta["alibi_slopes"][h] == 0) else "ALiBi"
            lab += f" (L{l}h{h} {hk})"
            win, env, gap, rank, width, nnz, scal, tok = [], [], [], [], [], [], [], []
            for s in stems:
                pw = z[f"{s}__p_win"]                                       # [S,Lyr,H,T]
                win.append(nanmean(pw[:, l, h])); env.append(np.nanmax(nanmean(pw, axis=(0, 3))))
                so = z[f"{s}__s_off"][:, l, h]                              # [S,T,5]
                gap.append(nanmean(np.nanmax(so, -1) - z[f"{s}__s_out"][:, l, h]))
                rank.append(nanmean(z[f"{s}__rank"][:, l, h] > 0))
                width.append(nanmean(z[f"{s}__s_max"][:, l, h] - z[f"{s}__s_minsup"][:, l, h]))
                nnz.append(nanmean(z[f"{s}__nnz"][:, l, h]))
                if f"{s}__scaler" in z.files:                               # [S,Lyr,Hs,T]; scaled heads are the last Hs
                    hs = h - (meta["n_heads"] - meta["num_scaled_heads"])
                    scal.append(np.nanmedian(z[f"{s}__scaler"][:, l, hs]) if hs >= 0 else np.nan)
                cor = z[f"{s}__correct"]; tok.append(cor.sum() / max((z[f"{s}__Q"] >= 0).sum(), 1))
            ns_a = np.array(ns)
            ax[0].plot(ns_a, win, "-o", color=c, ms=3, label=lab); ax[0].plot(ns_a, env, "--", color=c, lw=0.8, alpha=0.7)
            ax[1].plot(ns_a, np.clip(1 - np.array(win), 1e-5, 1), "-o", color=c, ms=3)
            ax[1].plot(ns_a, np.clip(1 - np.array(env), 1e-5, 1), "--", color=c, lw=0.8, alpha=0.7)
            ax[2].plot(ns_a, tok, ":", color=c, lw=1)
            rr = [x for x in runs if x["task"] == task and x["method"] == r["method"] and x["seed"] == int(r["seed"]) and x["lr"] == r["lr"]]
            if rr:
                lad = (rr[0]["ladder_last"] if rr[0]["degenerate"] else rr[0]["ladder"]) or {}
                pts = [(LADDER_N[k], float(v) / 100) for k, v in lad.items() if k in LADDER_N and v not in ("SKIPPED", "ERR")]
                if pts: ax[2].plot(*zip(*pts), "-s", color=c, ms=3)
            ax2[0].plot(ns_a, gap, "-o", color=c, ms=3, label=lab)
            ax2[1].plot(ns_a, rank, "-o", color=c, ms=3)
            ax2[2].plot(ns_a, nnz, "-o", color=c, ms=3)
            if scal: ax2[3].plot(ns_a[: len(scal)], scal, "-o", color=c, ms=3, label=lab)
            table.append((r["family"], r["method"], int(l), int(h), ns, [round(float(x), 3) for x in win], [round(float(x), 2) for x in gap]))
        for x in list(ax) + list(ax2):
            x.set_xscale("log", base=2); x.set_xlabel("n (nominal test length)"); x.grid(alpha=0.3, which="both")
        ax[0].set_ylabel("mass on correct key +-2"); ax[0].set_title("(a) retrieval head (solid), best head at each n (dashed)", fontsize=9)
        ax[1].set_yscale("log"); ax[1].set_ylabel("1 - window mass (leakage)"); ax[1].set_title("(b) leakage, log-log (slope = dispersion exponent)", fontsize=9)
        ax[2].set_ylabel("accuracy"); ax[2].set_title("(c) test exact-match ladder (squares), teacher-forced token acc (dotted)", fontsize=9)
        ax[0].legend(fontsize=6.5, loc="lower left")
        fig.suptitle(f"{task}: attention on the correct key vs length (8 test samples/length, mean over target queries)", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, f"{task}__tail_main.png"), dpi=130); plt.close(fig)
        ax2[0].set_ylabel("score(correct window) - max score elsewhere"); ax2[0].axhline(0, color="k", lw=0.5)
        ax2[0].set_title("(a) logit gap to best distractor (post-scale, incl. NAPE bias)", fontsize=9)
        ax2[1].set_ylabel("fraction of queries"); ax2[1].set_title("(b) correct key is NOT the top-scoring key", fontsize=9)
        ax2[2].set_yscale("log"); ax2[2].set_ylabel("#keys with p > 0"); ax2[2].set_title("(c) support size of the retrieval head", fontsize=9)
        ax2[3].set_ylabel("scaler (median over target queries)"); ax2[3].set_title("(d) adaptive scale of the retrieval head (AS rows, NoPE heads only)", fontsize=9)
        ax2[0].legend(fontsize=6.5)
        fig2.suptitle(f"{task}: mechanism (same retrieval head per run)", fontsize=10)
        fig2.tight_layout(); fig2.savefig(os.path.join(a.out_dir, f"{task}__tail_mech.png"), dpi=130); plt.close(fig2)
        print(f"== {task}")
        for t in table: print("  ", t[0], t[1], f"L{t[2]}h{t[3]}", "n", t[4], "win", t[5], "gap", t[6])


if __name__ == "__main__":
    main()
