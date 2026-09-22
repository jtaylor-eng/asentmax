#!/usr/bin/env python3
"""Dump every Table-1 run (task, method, seed, lr) to JSON for the LR-sweep report.
Per run: selection metric (paper protocol, see aggregate.py), ladder (best ckpt), ladder_last,
ladder_tiebreak, NaN/divergence flag, mqmtar plateau-escape step, train wall time.
Usage: dump_runs.py --out runs.json   (env RESULTS_ROOT / PROJECT_ROOT from env.sh)
"""
import argparse, csv, glob, json, os, re
from aggregate import PAPER, PRIMARY, FALLBACK, read_ladder, series

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("RESULTS_ROOT"))
    ap.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = []
    for run in sorted(glob.glob(os.path.join(a.results, "*", "*", "s*_lr*"))):
        task, method, sl = run.split(os.sep)[-3:]
        if task not in PAPER: continue
        seed, lr = re.match(r"s(\d+)_lr(.+)", sl).groups()
        csvs = sorted(glob.glob(os.path.join(a.project_root, "logs", f"t1_{task}_{method}_s{seed}_lr{lr}", "runs", "*", "csv", "version_0", "metrics.csv")))
        rec = dict(task=task, method=method, seed=int(seed), lr=lr, lr_f=float(lr))
        if csvs:
            rows = list(csv.DictReader(open(csvs[-1])))
            prim = series(rows, PRIMARY[task])
            tl = [(int(r["step"]), float(r["train/loss_step"])) for r in rows if r.get("train/loss_step")]
            rec["diverged"] = any(v != v for _, v in tl[-50:]) if tl else False
            rec["last_step"] = tl[-1][0] if tl else None
            rec["final_loss"] = tl[-1][1] if tl else None
            esc = [s for s, v in tl if v < 0.3]
            rec["escape_step"] = esc[0] if esc else None
            rec["primary_max"] = max(v for _, v in prim) if prim else None
            rec["primary_last"] = prim[-1][1] if prim else None
            degenerate = (not prim) or max(v for _, v in prim) == 0.0
            rec["degenerate"] = degenerate
            sel_key, sel_val = PRIMARY[task], rec["primary_max"]
            if degenerate:
                sel_val = None
                for fb in FALLBACK[task]:
                    s = series(rows, fb)
                    if s and max(v for _, v in s) > 0:
                        sel_key = fb; sel_val = max(v for _, v in s); break
            rec["sel_key"] = sel_key; rec["sel_val"] = sel_val
        for name in ("ladder", "ladder_last", "ladder_tiebreak"):
            lad = read_ladder(os.path.join(run, f"{name}.tsv"))
            rec[name] = lad if lad else None
        log = os.path.join(run, "run.log")
        if os.path.exists(log):
            t = open(log).read()
            m = re.findall(r"\[(\d\d-\d\d \d\d:\d\d:\d\d)\] TRAIN (start|done)", t)
            rec["train_done"] = any(k == "done" for _, k in m)
        out.append(rec)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out, len(out), "runs")

if __name__ == "__main__":
    main()
