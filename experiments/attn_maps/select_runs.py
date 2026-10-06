#!/usr/bin/env python3
"""Pick the checkpoint(s) to extract attention maps from: best run per mapping family per task,
by the validation-only protocol of reproduction_0913.md §1 (never by test score).

Family = method name with the _q<N>/_d<N>/_zi/_w20k suffixes stripped, so e.g. the best
aswstieltjes run may be aswstieltjes_q2_d4_w20k. Rule per (task, family):
  1. max validation selection value (sel_val: acc@8x; sort BLEU@4x; reverse BLEU@4x fallback);
  2. ties -> tie-break validation ladder (longest length first) where it was run;
  3. still tied -> ALL tied runs are extracted (cheap); the first by (lr, method, seed) is marked
     primary=1, the rest primary=0, so plots can show the tie honestly.
Checkpoint: degenerate primary monitor (reverse) -> last.ckpt, else the saved best epoch=*.ckpt
(resolved on OSC by the sbatch script), as in aggregate.py.

Writes a TSV: task family method seed lr ckpt_kind primary sel_val tiebreak
Usage: python3 select_runs.py --runs context/runs_0920.json --out experiments/attn_maps/manifest.tsv
"""
import argparse, json, re

FAMILIES = ["softmax", "entmax", "asentmax", "stieltjes", "wstieltjes", "aswstieltjes"]


def family(m):
    b = re.sub(r"_w20k$", "", m); b = re.sub(r"_zi$", "", b)
    b = re.sub(r"_q\d+", "", b); return re.sub(r"_d[\d.]+", "", b)


def tb_key(r):
    tb = r.get("ladder_tiebreak") or {}
    labs = sorted(tb, key=lambda l: -int(re.sub(r"\D", "", l)))     # longest length first
    return tuple(float(tb[l]) for l in labs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = json.load(open(a.runs))
    rows = []
    for task in ["sort", "copy", "reverse", "mqmtar"]:
        for fam in FAMILIES:
            rs = [r for r in runs if r["task"] == task and family(r["method"]) == fam and r.get("sel_val") is not None]
            if not rs: continue
            best = max(r["sel_val"] for r in rs)
            tied = [r for r in rs if r["sel_val"] == best]
            if len(tied) > 1 and any(r.get("ladder_tiebreak") for r in tied):
                with_tb = [r for r in tied if r.get("ladder_tiebreak")]
                top = max(tb_key(r) for r in with_tb)
                tied = [r for r in with_tb if tb_key(r) == top]
            tied.sort(key=lambda r: (r["lr_f"], r["method"], r["seed"]))
            for i, r in enumerate(tied):
                rows.append((task, fam, r["method"], r["seed"], r["lr"], "last" if r["degenerate"] else "best",
                             int(i == 0), f"{r['sel_val']:.3f}", json.dumps(r.get("ladder_tiebreak") or {}, separators=(",", ":"))))
    with open(a.out, "w") as f:
        f.write("task\tfamily\tmethod\tseed\tlr\tckpt_kind\tprimary\tsel_val\ttiebreak\n")
        for r in rows: f.write("\t".join(map(str, r)) + "\n")
    for r in rows: print("\t".join(map(str, r)))
    print(f"{len(rows)} runs -> {a.out}")


if __name__ == "__main__":
    main()
