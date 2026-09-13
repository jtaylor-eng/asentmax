#!/usr/bin/env python3
"""Paper-protocol checkpoint selection with the documented fallback.

Primary monitor: exact-match acc at 8x (sort: BLEU@4x). If the primary monitor is
degenerate (identically 0 / constant -> ModelCheckpoint keeps the FIRST ckpt on ties),
fall back per the paper: BLEU at 4x, then 2x. Reports, per run, the step the protocol
would pick and whether that is the saved best ckpt, the last ckpt, or neither.
Usage: reselect.py [--results DIR] [--project-root DIR] [--emit-manifest FILE]
"""
import argparse, csv, glob, os, re, sys
PRIMARY = {"sort": "val/bleu_epoch/dataloader_idx_2", "reverse": "val/acc_epoch/dataloader_idx_4",
           "copy": "val/acc_epoch/dataloader_idx_3", "mqmtar": "val/acc_epoch/dataloader_idx_3"}
FALLBACK = {"sort": ["val/bleu_epoch/dataloader_idx_1"],
            "reverse": ["val/bleu_epoch/dataloader_idx_3", "val/bleu_epoch/dataloader_idx_2"],
            "copy": ["val/bleu_epoch/dataloader_idx_2", "val/bleu_epoch/dataloader_idx_1"],
            "mqmtar": ["val/bleu_epoch/dataloader_idx_2", "val/bleu_epoch/dataloader_idx_1"]}

def series(rows, key):
    return [(int(r["step"]), float(r[key])) for r in rows if r.get(key) not in (None, "")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("RESULTS_ROOT"))
    ap.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT"))
    ap.add_argument("--emit-manifest", default=None)
    a = ap.parse_args()
    out = []
    for run in sorted(glob.glob(os.path.join(a.results, "*", "*", "s*_lr*"))):
        task, method, sl = run.split(os.sep)[-3:]
        seed, lr = re.match(r"s(\d+)_lr(.+)", sl).groups()
        csvs = sorted(glob.glob(os.path.join(a.project_root, "logs", f"t1_{task}_{method}_s{seed}_lr{lr}", "runs", "*", "csv", "version_0", "metrics.csv")))
        if not csvs: continue
        rows = list(csv.DictReader(open(csvs[-1])))
        prim = series(rows, PRIMARY[task])
        if not prim: continue
        last_step = prim[-1][0]
        pmax = max(v for _, v in prim)
        degenerate = pmax == 0.0 or all(v == prim[0][1] for _, v in prim)
        used, pick_step, pick_val = PRIMARY[task], None, None
        if not degenerate:
            pick_step, pick_val = max(prim, key=lambda t: (t[1], t[0]))  # ties -> latest
        else:
            for fb in FALLBACK[task]:
                s = series(rows, fb)
                if s and max(v for _, v in s) > 0:
                    used = fb; pick_step, pick_val = max(s, key=lambda t: (t[1], t[0])); break
        best = glob.glob(os.path.join(run, "checkpoints", "epoch=*.ckpt"))
        best_step = int(re.search(r"step=(\d+)", best[0]).group(1)) if best else None
        # ModelCheckpoint saves at step S+1 boundary offsets; treat |diff|<=2 as same
        if pick_step is None: where = "none"
        elif best_step is not None and abs(best_step - pick_step) <= 2: where = "best"
        elif abs(last_step - pick_step) <= 2: where = "last"
        else: where = "neither"
        out.append((task, method, seed, lr, degenerate, used.split("/")[1][:4] + used[-1], pick_step, pick_val, best_step, last_step, where))
    print(f"{'task':<8}{'method':<11}{'s':<2}{'lr':<6}{'degen':<6}{'monitor':<8}{'pick_step':>9}{'val':>7}{'best_ckpt':>10}{'last':>8}  where")
    for r in out:
        print(f"{r[0]:<8}{r[1]:<11}{r[2]:<2}{r[3]:<6}{str(r[4]):<6}{r[5]:<8}{str(r[6]):>9}{(r[7] if r[7] is not None else float('nan')):>7.3f}{str(r[8]):>10}{r[9]:>8}  {r[10]}")
    if a.emit_manifest:
        with open(a.emit_manifest, "w") as f:
            for r in out:
                if r[10] == "last": f.write(f"{r[0]} {r[1]} {r[2]} {r[3]}\n")
        print("manifest (runs to re-ladder from last.ckpt):", a.emit_manifest)

if __name__ == "__main__":
    main()
