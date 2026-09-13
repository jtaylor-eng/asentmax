#!/usr/bin/env bash
# Quick progress view for all Table-1 runs: step / it/s / loss / status. Run on a login node.
source /fs/scratch/PAS2836/$USER/asentmax/experiments/osc/env.sh
python3 - "$RESULTS_ROOT" "$PROJECT_ROOT" <<'PY'
import csv, glob, os, re, sys, datetime
R, P = sys.argv[1], sys.argv[2]
STEPS = {"sort": 312500, "reverse": 234375, "copy": 156250, "mqmtar": 390625}
rows = []
for run in sorted(glob.glob(f"{R}/*/*/s*_lr*")):
    task, method, sl = run.split("/")[-3:]
    seed, lr = re.match(r"s(\d+)_lr(.+)", sl).groups()
    log = os.path.join(run, "run.log")
    if not os.path.exists(log): continue
    txt = open(log).read()
    status = "done" if "=== DONE" in txt else ("FAILED" if ("FAILED" in txt or "ERROR" in txt) else ("eval" if "eval ckpt" in txt else "train"))
    m = re.search(r"\[(\d\d-\d\d \d\d:\d\d:\d\d)\] TRAIN start", txt)
    step, ips, loss = 0, 0.0, float("nan")
    csvs = sorted(glob.glob(f"{P}/logs/t1_{task}_{method}_s{seed}_lr{lr}/runs/*/csv/version_0/metrics.csv"))
    if csvs:
        last = None
        for r in csv.DictReader(open(csvs[-1])):
            if r.get("train/loss_step"): last = r
        if last:
            step = int(last["step"]); loss = float(last["train/loss_step"])
            if m:
                st = datetime.datetime.strptime(m.group(1), "%m-%d %H:%M:%S").replace(year=datetime.datetime.now().year)
                el = (datetime.datetime.now() - st).total_seconds()
                ips = step / el if el > 0 else 0
    tot = STEPS[task]
    eta = (tot - step) / ips / 3600 if ips > 0 and status == "train" else 0
    rows.append((task, method, seed, lr, status, step, tot, ips, loss, eta))
print(f"{'task':<8}{'method':<11}{'s':<2}{'lr':<6}{'status':<7}{'step':>8}/{'total':<7}{'it/s':>6}{'loss':>9}{'eta_h':>7}")
for r in rows:
    print(f"{r[0]:<8}{r[1]:<11}{r[2]:<2}{r[3]:<6}{r[4]:<7}{r[5]:>8}/{r[6]:<7}{r[7]:>6.1f}{r[8]:>9.4f}{r[9]:>7.1f}")
from collections import Counter
print(Counter(r[4] for r in rows))
PY
