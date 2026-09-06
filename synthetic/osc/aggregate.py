#!/usr/bin/env python3
"""Aggregate Table-1 results from $RESULTS_ROOT with the paper's selection protocol.

For each (task, method): among all (seed, lr) runs, pick the run whose best-checkpoint
validation monitor value is highest (ties -> higher mean OOD test acc, then lower lr).
Sort uses BLEU@4x (val idx_2); other tasks use exact-match acc @8x (val idx_3/4).
Emits: markdown table vs paper, per-cell provenance, and a mean±std-over-seeds row at the
selected LR. Usage: aggregate.py [--results DIR] [--status]
"""
import argparse, csv, glob, os, re, statistics, sys
from collections import defaultdict

PAPER = {
  "sort":    (["ID","2x","4x","8x"], {"softmax":[100,0,0,0], "asentmax":[100,100,79.7,0]}),
  "reverse": (["ID","1.5x","2x","4x","8x"], {"softmax":[100,36,0,0,0], "asentmax":[100,100,99.8,96.4,56.7]}),
  "copy":    (["ID","2x","4x","8x","16x","32x","64x"], {"softmax":[100,100,99.9,99.9,99.4,96.1,85.5], "asentmax":[100,100,99.9,99.7,99.4,96.3,86.6]}),
  "mqmtar":  (["ID","2x","4x","16x","64x","256x","1024x"], {"softmax":[100,100,100,99.5,97.8,80.2,3.0], "asentmax":[100,100,100,99.7,99.6,99.0,95.3]}),
}
MONITOR = {"sort":"val/bleu_epoch/dataloader_idx_2", "reverse":"val/acc_epoch/dataloader_idx_4",
           "copy":"val/acc_epoch/dataloader_idx_3", "mqmtar":"val/acc_epoch/dataloader_idx_3"}
METHOD_ORDER = ["softmax","asentmax","stieltjes","asstieltjes"]

def read_ladder(p):
    d = {}
    for line in open(p):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4: d[parts[0]] = parts[3]
    return d

def find_metrics_csv(project_root, task, method, seed, lr):
    pat = os.path.join(project_root, "logs", f"t1_{task}_{method}_s{seed}_lr{lr}", "runs", "*", "csv", "version_0", "metrics.csv")
    fs = sorted(glob.glob(pat)); return fs

def best_val(project_root, task, method, seed, lr):
    key = MONITOR[task]; best = None; step = None
    for f in find_metrics_csv(project_root, task, method, seed, lr):
        for row in csv.DictReader(open(f)):
            v = row.get(key)
            if v not in (None, ""):
                v = float(v)
                if best is None or v > best: best, step = v, int(row.get("step", 0))
    return best, step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("RESULTS_ROOT"))
    ap.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT"))
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = a.results
    runs = defaultdict(list)   # (task, method) -> list of dict
    for ladder in sorted(glob.glob(os.path.join(root, "*", "*", "s*_lr*", "ladder.tsv"))):
        run_dir = os.path.dirname(ladder)
        task, method, sl = run_dir.split(os.sep)[-3:]
        m = re.match(r"s(\d+)_lr(.+)", sl); seed, lr = m.group(1), m.group(2)
        lad = read_ladder(ladder)
        labels = PAPER[task][0]
        complete = all(l in lad for l in labels) or (method.endswith("stieltjes") and task == "mqmtar" and all(l in lad for l in labels[:5]))
        bv, bstep = best_val(a.project_root, task, method, seed, lr)
        vals = [lad.get(l, "-") for l in labels]
        nums = [float(v) for v in vals if v not in ("-", "SKIPPED", "ERR")]
        runs[(task, method)].append(dict(seed=seed, lr=lr, ladder=lad, vals=vals, best_val=bv, best_step=bstep,
                                         mean_ood=(statistics.mean(nums[1:]) if len(nums) > 1 else 0.0), complete=complete, dir=run_dir))

    if a.status:
        tot = 0
        for task in PAPER:
            for method in METHOD_ORDER:
                for r in runs.get((task, method), []):
                    tot += 1
                    print(f"{task:<8}{method:<12}s{r['seed']} lr={r['lr']:<6} val={r['best_val'] if r['best_val'] is not None else '-':<8} "
                          f"{'done' if r['complete'] else 'partial'}  " + " ".join(f"{v:>6}" for v in r["vals"]))
        print(f"{tot} runs with ladders")
        return

    out = []
    def P(s=""): out.append(s)
    P("# Table 1 reproduction (OSC A100, paper protocol)\n")
    P("Selection: per (task, method), the single run (over seeds x LRs) with the highest best-checkpoint")
    P("validation monitor (sort: BLEU@4x; others: exact-match@8x); reported numbers are that run's")
    P("test ladder from its best checkpoint. 100 test samples/length. `skip` = early-stopped after an exact 0.0.\n")
    for task, (labels, paper) in PAPER.items():
        P(f"## {task}\n")
        P("| method | " + " | ".join(labels) + " | selected (seed, lr, val, step) |")
        P("|---|" + "---:|" * len(labels) + "---|")
        for method in METHOD_ORDER:
            rs = [r for r in runs.get((task, method), []) if r["complete"]]
            if method in paper:
                P(f"| {method} (paper) | " + " | ".join(f"{v}" for v in paper[method]) + " | best of 3 seeds x LRs, 1K samples |")
            if not rs:
                if runs.get((task, method)): P(f"| {method} (ours) | " + " | ".join(["…"] * len(labels)) + f" | {len(runs[(task, method)])} runs in progress |")
                continue
            rs.sort(key=lambda r: (-(r["best_val"] if r["best_val"] is not None else -1), -r["mean_ood"], float(r["lr"])))
            b = rs[0]
            cells = ["skip" if v == "SKIPPED" else v for v in b["vals"]]
            P(f"| {method} (ours) | " + " | ".join(cells) + f" | s{b['seed']}, lr={b['lr']}, val={b['best_val']:.3f}@{b['best_step']} |")
            # mean±std over seeds at the selected LR
            same_lr = [r for r in rs if r["lr"] == b["lr"]]
            if len(same_lr) > 1:
                ms = []
                for i, l in enumerate(labels):
                    xs = [float(r["vals"][i]) if r["vals"][i] not in ("-", "SKIPPED", "ERR") else 0.0 for r in same_lr]
                    ms.append(f"{statistics.mean(xs):.1f}±{statistics.pstdev(xs):.1f}")
                P(f"| {method} (ours, mean±std, {len(same_lr)} seeds @ lr={b['lr']}) | " + " | ".join(ms) + " | |")
        P()
        P("<details><summary>all runs</summary>\n")
        P("| method | seed | lr | val monitor | " + " | ".join(labels) + " |")
        P("|---|---|---|---|" + "---:|" * len(labels))
        for method in METHOD_ORDER:
            for r in sorted(runs.get((task, method), []), key=lambda r: (r["lr"], r["seed"])):
                bv = f"{r['best_val']:.3f}" if r["best_val"] is not None else "-"
                P(f"| {method} | {r['seed']} | {r['lr']} | {bv} | " + " | ".join(("skip" if v == "SKIPPED" else v) for v in r["vals"]) + " |")
        P("\n</details>\n")
    text = "\n".join(out)
    if a.out: open(a.out, "w").write(text); print("wrote", a.out)
    else: print(text)

if __name__ == "__main__":
    main()
