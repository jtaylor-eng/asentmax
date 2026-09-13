#!/usr/bin/env python3
"""Aggregate Table-1 results with the paper's selection protocol.

Per run: primary monitor = exact-match acc @8x (sort: BLEU @4x). If the primary monitor
is degenerate (never > 0), fall back per the paper to BLEU @4x then @2x. The checkpoint
evaluated is:
  - the saved best-by-primary ckpt (ladder.tsv) when the primary monitor is informative;
  - last.ckpt (ladder_last.tsv) when the primary was degenerate, since the callback then
    kept the *first* checkpoint on ties (a known artifact) and the fallback pick is not a
    saved checkpoint. This is noted per cell.
Per (task, method): the single run with the highest selection value (primary if informative,
else fallback BLEU) is reported. Also prints all runs and mean±std over seeds at the
selected LR.
"""
import argparse, csv, glob, os, re, statistics, sys
from collections import defaultdict

PAPER = {
  "sort":    (["ID","2x","4x","8x"], {"softmax":[100,0,0,0], "asentmax":[100,100,79.7,0]}),
  "reverse": (["ID","1.5x","2x","4x","8x"], {"softmax":[100,36,0,0,0], "asentmax":[100,100,99.8,96.4,56.7]}),
  "copy":    (["ID","2x","4x","8x","16x","32x","64x"], {"softmax":[100,100,99.9,99.9,99.4,96.1,85.5], "asentmax":[100,100,99.9,99.7,99.4,96.3,86.6]}),
  "mqmtar":  (["ID","2x","4x","16x","64x","256x","1024x"], {"softmax":[100,100,100,99.5,97.8,80.2,3.0], "asentmax":[100,100,100,99.7,99.6,99.0,95.3]}),
}
PRIMARY = {"sort":"val/bleu_epoch/dataloader_idx_2", "reverse":"val/acc_epoch/dataloader_idx_4",
           "copy":"val/acc_epoch/dataloader_idx_3", "mqmtar":"val/acc_epoch/dataloader_idx_3"}
FALLBACK = {"sort":["val/bleu_epoch/dataloader_idx_1"],
            "reverse":["val/bleu_epoch/dataloader_idx_3","val/bleu_epoch/dataloader_idx_2"],
            "copy":["val/bleu_epoch/dataloader_idx_2","val/bleu_epoch/dataloader_idx_1"],
            "mqmtar":["val/bleu_epoch/dataloader_idx_2","val/bleu_epoch/dataloader_idx_1"]}
METHOD_ORDER = ["softmax","asentmax","stieltjes","asstieltjes","stieltjes_eager","asstieltjes_eager"]
LOCAL = {  # previous single-seed 4070 run (from synthetic/TABLE1_REPRODUCED_FIXED.md)
  "sort":    {"softmax":[100,0,"skip","skip"], "asentmax":[100,96,69,0]},
  "reverse": {"softmax":[100,76,0,"skip","skip"], "asentmax":[100,100,100,92,36]},
  "copy":    {"softmax":[100,100,100,64,0,"skip","skip"], "asentmax":[100,100,100,100,100,94,74]},
  "mqmtar":  {"softmax":[100,98,99,83,54,"-","-"], "asentmax":[100,100,100,100,100,"-","-"]},
}

def read_ladder(p):
    d = {}
    if not os.path.exists(p): return d
    for line in open(p):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 4: d[parts[0]] = parts[3]
    return d

def series(rows, key):
    return [(int(r["step"]), float(r[key])) for r in rows if r.get(key) not in (None, "")]

def collect(results, project_root):
    runs = defaultdict(list)
    for run in sorted(glob.glob(os.path.join(results, "*", "*", "s*_lr*"))):
        task, method, sl = run.split(os.sep)[-3:]
        if task not in PAPER: continue
        seed, lr = re.match(r"s(\d+)_lr(.+)", sl).groups()
        csvs = sorted(glob.glob(os.path.join(project_root, "logs", f"t1_{task}_{method}_s{seed}_lr{lr}", "runs", "*", "csv", "version_0", "metrics.csv")))
        if not csvs: continue
        rows = list(csv.DictReader(open(csvs[-1])))
        prim = series(rows, PRIMARY[task])
        tl = [float(r["train/loss_step"]) for r in rows if r.get("train/loss_step")]
        diverged = any(x != x for x in tl[-50:]) if tl else False
        if not prim: continue
        degenerate = max(v for _, v in prim) == 0.0
        sel_key, sel_val, sel_step = PRIMARY[task], None, None
        if not degenerate:
            sel_step, sel_val = max(prim, key=lambda t: (t[1], t[0]))
        else:
            for fb in FALLBACK[task]:
                s = series(rows, fb)
                if s and max(v for _, v in s) > 0:
                    sel_key = fb; sel_step, sel_val = max(s, key=lambda t: (t[1], t[0])); break
        ladder_file = "ladder_last.tsv" if degenerate else "ladder.tsv"
        lad = read_ladder(os.path.join(run, ladder_file))
        if degenerate and not lad:  # re-ladder not done yet; fall back to what exists
            lad = read_ladder(os.path.join(run, "ladder.tsv")); ladder_file = "ladder.tsv (pending last)"
        labels = PAPER[task][0]
        # eager Stieltjes ladders are capped at 64x on mqmtar (O(N^2) prefill); the Triton path runs the full ladder
        n_expected = 5 if (method.endswith("stieltjes_eager") and task == "mqmtar") else len(labels)
        complete = sum(l in lad and lad[l] not in ("ERR",) for l in labels) >= n_expected
        vals = [lad.get(l, "-") for l in labels]
        best = glob.glob(os.path.join(run, "checkpoints", "epoch=*.ckpt"))
        best_step = int(re.search(r"step=(\d+)", best[0]).group(1)) if best else None
        used_step = prim[-1][0] if degenerate else best_step
        runs[(task, method)].append(dict(seed=seed, lr=lr, vals=vals, sel_key=sel_key, sel_val=sel_val, sel_step=sel_step,
            degenerate=degenerate, diverged=diverged, complete=complete, ladder_file=ladder_file, used_step=used_step, dir=run))
    return runs

def fmt(v): return "skip" if v == "SKIPPED" else str(v)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("RESULTS_ROOT"))
    ap.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT"))
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = collect(a.results, a.project_root)
    if a.status:
        for task in PAPER:
            for method in METHOD_ORDER:
                for r in runs.get((task, method), []):
                    sv = f"{r['sel_val']:.3f}" if r["sel_val"] is not None else "-"
                    print(f"{task:<8}{method:<11}s{r['seed']} lr={r['lr']:<6} {r['sel_key'].split('/')[1][:4]}{r['sel_key'][-1]}={sv:<6} "
                          f"{'DEGEN' if r['degenerate'] else '     '} {'NaN' if r['diverged'] else '   '} {r['ladder_file']:<24} "
                          + " ".join(f"{fmt(v):>5}" for v in r["vals"]))
        return
    out = []; P = out.append
    P("# Table 1 reproduction on OSC (A100) — paper protocol, 2 seeds x 3 LRs (MQMTAR: 2 LRs)\n")
    for task, (labels, paper) in PAPER.items():
        P(f"## {task}\n")
        P("| method | " + " | ".join(labels) + " | selected run |")
        P("|---|" + "---:|" * len(labels) + "---|")
        for method in METHOD_ORDER:
            rs = [r for r in runs.get((task, method), []) if r["complete"]]
            if method in paper:
                P(f"| {method} (paper) | " + " | ".join(str(v) for v in paper[method]) + " | best of 3 seeds x LRs, 1K samples |")
            if method in LOCAL[task]:
                P(f"| {method} (local 4070, 1 seed, last.ckpt) | " + " | ".join(str(v) for v in LOCAL[task][method]) + " | README recipe LR |")
            if not rs:
                n = len(runs.get((task, method), []))
                if n: P(f"| {method} (ours) | " + " | ".join(["…"] * len(labels)) + f" | {n} runs, ladders pending |")
                continue
            rs.sort(key=lambda r: (-(r["sel_val"] if r["sel_val"] is not None else -1), float(r["lr"])))
            b = rs[0]
            note = f"s{b['seed']}, lr={b['lr']}, {b['sel_key'].split('/')[1][:4]}@{b['sel_key'][-1]}={b['sel_val']:.3f}"
            note += ", last.ckpt (8x monitor degenerate)" if b["degenerate"] else f", ckpt step {b['used_step']}"
            P(f"| **{method} (ours)** | " + " | ".join(fmt(v) for v in b["vals"]) + f" | {note} |")
            same_lr = [r for r in rs if r["lr"] == b["lr"]]
            if len(same_lr) > 1:
                ms = []
                for i in range(len(labels)):
                    xs = [float(r["vals"][i]) if r["vals"][i] not in ("-", "SKIPPED", "ERR") else 0.0 for r in same_lr]
                    ms.append(f"{statistics.mean(xs):.0f}±{statistics.pstdev(xs):.0f}")
                P(f"| {method} (ours, mean±std over {len(same_lr)} seeds @ lr={b['lr']}) | " + " | ".join(ms) + " | |")
        P("")
        P("<details><summary>all runs</summary>\n")
        P("| method | seed | lr | selection metric | ckpt | " + " | ".join(labels) + " |")
        P("|---|---|---|---|---|" + "---:|" * len(labels))
        for method in METHOD_ORDER:
            for r in sorted(runs.get((task, method), []), key=lambda r: (float(r["lr"]), r["seed"])):
                sv = f"{r['sel_key'].split('/')[1][:4]}@{r['sel_key'][-1]}={r['sel_val']:.3f}" if r["sel_val"] is not None else "-"
                ck = ("last" if r["degenerate"] else f"best@{r['used_step']}") + (" NaN-diverged" if r["diverged"] else "")
                P(f"| {method} | {r['seed']} | {r['lr']} | {sv} | {ck} | " + " | ".join(fmt(v) for v in r["vals"]) + " |")
        P("\n</details>\n")
    text = "\n".join(out)
    if a.out: open(a.out, "w").write(text); print("wrote", a.out)
    else: print(text)

if __name__ == "__main__":
    main()
