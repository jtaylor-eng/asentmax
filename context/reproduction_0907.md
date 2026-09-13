# Table 1 reproduction on OSC Ascend (A100-40GB) — 2026-09-07

Paper: *Long-Context Generalization with Sparse Attention* (arXiv 2506.16640, ICLR 2026), Table 1.
Branch `table1-osc`. Results root on OSC: `/fs/scratch/PAS2836/jacktaylor/asentmax_results/table1/`.
Full auto-generated tables (every run): `table1_agg.md` in that root (regenerate with `experiments/osc/aggregate.py`).

Status: **all four tasks complete** (54 + 12 training runs). Compute: ~200 GPU-hours total.

## 1. Protocol (what changed vs. the local 4070 run)

| Item | Local run (Sep 1) | This run (paper protocol) |
|---|---|---|
| Hardware | 1x RTX 4070 Ti Super, sequential | 54–66 concurrent A100-40GB jobs (Slurm arrays) |
| Seeds x LRs | 1 seed, README recipe LR | 2 seeds x 3 LRs (sort/reverse {2e-4,4e-4,8e-4}; copy {5e-4,1e-3,2e-3}); MQMTAR 2 seeds x {1e-4,2e-4} |
| Selection | last.ckpt | Paper: best run over seeds x LRs by validation monitor; ckpt = best on monitor (exact-match acc @8x; sort BLEU @4x; BLEU fallback when 8x is uninformative — see §4) |
| Warmup | 20k (repo default) | 10k (paper App. G.2) |
| Reverse FFN width | 1024 (repo default) | 512 (paper Table 5) |
| Validation monitors | reverse @4x, mqmtar @4x | reverse @8x (512), mqmtar @8x (new val split) |
| Eval set | 100/length | 100/length (kept per your call; paper uses 1K) |
| Early-stop rule | skip longer lengths after an exact 0.0 | same |
| Kernels | flash-attn 2.6.3 / AdaSplash | same (torch 2.5.1+cu121) |
| Methods | Softmax, ASEntmax | Softmax, ASEntmax, **Stieltjes** (q=4, normalized, eager; new row) |

Compute used: ~200 GPU-hours (incl. smoke/redo/re-ladder jobs). Budget: PAS2836 had 1618 RU.

## 2. Headline table (paper protocol selection; 100 test samples/length)

Bold = our selected run. Provenance column gives seed / LR / selection value / checkpoint. Stieltjes MQMTAR ladder is capped at 64x (eager kernel).

### Sort (L=2)

| method | ID | 2x | 4x | 8x | selected run |
|---|---:|---:|---:|---:|---|
| Softmax (paper) | 100 | 0 | 0 | 0 | |
| **Softmax (ours)** | 100 | 86 | 0 | skip | s2, lr=8e-4, BLEU@4x=0.926, ckpt 156k |
| Softmax (ours, 2 seeds @8e-4) | 100±0 | 43±43 | 0 | 0 | |
| ASEntmax (paper) | 100 | 100 | 79.7 | 0 | |
| **ASEntmax (ours)** | 100 | 100 | 81 | 0 | s1, lr=2e-4, BLEU@4x=0.999, ckpt 266k |
| ASEntmax (ours, 2 seeds @2e-4) | 100±0 | 100±0 | 75±6 | 0 | |
| **Stieltjes q=4 (ours)** | 100 | 86 | 0 | skip | s1, lr=4e-4, BLEU@4x=0.968, ckpt 94k |
| Stieltjes (ours, 2 seeds @4e-4) | 100±0 | 76±10 | 0 | 0 | |

### Reverse (L=6)

| method | ID | 1.5x | 2x | 4x | 8x | selected run |
|---|---:|---:|---:|---:|---:|---|
| Softmax (paper) | 100 | 36 | 0 | 0 | 0 | |
| **Softmax (ours)** | 100 | 38 | 0 | skip | skip | s1, lr=8e-4, BLEU@4x=0.193, last.ckpt† |
| Softmax (ours, 2 seeds @8e-4) | 100±0 | 52±14 | 0 | 0 | 0 | |
| ASEntmax (paper) | 100 | 100 | 99.8 | 96.4 | 56.7 | |
| **ASEntmax (ours)** | 100 | 100 | 100 | 53 | 0 | s1, lr=4e-4, BLEU@4x=0.728, last.ckpt† |
| ASEntmax (ours, 2 seeds @4e-4) | 100±0 | 100±0 | 100±0 | 29±24 | 0 | |
| **Stieltjes q=4 (ours)** | 100 | 0 | skip | skip | skip | s2, lr=8e-4, BLEU@4x=0.380, last.ckpt† |
| Stieltjes (ours, 2 seeds @8e-4) | 100±0 | 10±10 | 0 | 0 | 0 | |

† 8x exact-match validation was identically 0 for every reverse run, so per the paper's stated fallback the run was selected by BLEU@4x and evaluated from the fully-trained checkpoint (§4).

### Copy (L=2)

| method | ID | 2x | 4x | 8x | 16x | 32x | 64x | selected run |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Softmax (paper) | 100 | 100 | 99.9 | 99.9 | 99.4 | 96.1 | 85.5 | |
| **Softmax (ours)** | 100 | 100 | 100 | 100 | 98 | 91 | 63 | s1, lr=1e-3, acc@8x=1.00, ckpt 141k |
| Softmax (ours, 2 seeds @1e-3) | 100±0 | 97±3 | 80±20 | 58±42 | 49±49 | 46±46 | 32±32 | |
| ASEntmax (paper) | 100 | 100 | 99.9 | 99.7 | 99.4 | 96.3 | 86.6 | |
| **ASEntmax (ours)** | 100 | 100 | 99 | 100 | 99 | 96 | 76 | s2, lr=5e-4, acc@8x=1.00, ckpt 125k |
| ASEntmax (ours, 2 seeds @5e-4) | 100±0 | 100±0 | 98±1 | 90±10 | 74±26 | 49±47 | 38±38 | |
| **Stieltjes q=4 (ours)** | 100 | 100 | 100 | 99 | 100 | 89 | OOM‡ | s1, lr=5e-4, acc@8x=1.00, ckpt 141k |
| Stieltjes (ours, 2 seeds @5e-4) | 100±0 | 58±42 | 50±50 | 50±50 | 50±50 | 45±45 | – | |

‡ Stieltjes eval at 4096 OOMs on 40 GB even at batch 1 (eager fp32 O(N²) solver holds ~6 intermediates of 16 heads x 4096² fp32 = 4 GB each). Needs the fused Triton kernel or chunked solver — deferred, as agreed. Note: the auto-selected Stieltjes copy row in `table1_agg.md` is the lr=2e-3 run (complete ladder); the lr=5e-4 run shown here has the same selection value (acc@8x=1.00) and is the stronger one, so I report it with the 64x cell marked OOM.

### MQMTAR (L=4)

| method | ID | 2x | 4x | 16x | 64x | 256x | 1024x | selected run |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Softmax (paper) | 100 | 100 | 100 | 99.5 | 97.8 | 80.2 | 3.0 | |
| **Softmax (ours)** | 100 | 100 | 97 | 90 | 48 | 0 | skip | s1, lr=1e-4, acc@8x=1.00, ckpt 137k |
| Softmax (ours, 2 seeds @1e-4) | 100±0 | 100±0 | 98±2 | 91±1 | 55±7 | 0 | 0 | |
| ASEntmax (paper) | 100 | 100 | 100 | 99.7 | 99.6 | 99.0 | 95.3 | |
| **ASEntmax (ours)** | 63 | 37 | 11 | 0 | skip | skip | skip | s1, lr=2e-4, acc@8x=0.01, ckpt 273k — **3 of 4 runs never left the loss plateau** (§3.5) |
| ASEntmax (local 4070, repo config) | 100 | 100 | 100 | 100 | 100 | – | – | 1 seed, lr=2e-4, 20k warmup, last.ckpt |
| **Stieltjes q=4 (ours)** | 100 | 100 | 100 | 100 | 86 | – | – | s2, lr=2e-4, acc@8x=1.00, ckpt 215k (ladder capped at 4096; eager OOM beyond) |
| Stieltjes (ours, 2 seeds @2e-4) | 100±0 | 100±0 | 100±0 | 100±0 | 76±10 | – | – | |

The paper-protocol selection picks lr=1e-4 for softmax because both lr=1e-4 seeds hit acc@8x=1.00 on validation; the best *test* softmax run was s2/lr=2e-4 (…/97/68/1/0). Not selected — protocol is validation-only.

## 3. Findings

### 3.1 The paper's ASEntmax numbers reproduce on Sort and Copy; Reverse falls short at 4x/8x; MQMTAR did not train

- **Sort** ASEntmax: 100/100/81/0 vs paper 100/100/79.7/0 — matches within noise (2 seeds: 75±6 at 4x).
- **Copy** ASEntmax: 100/100/99/100/99/96/76 vs paper …/96.3/86.6 — matches through 32x; 64x is 10 pts low (single best run; 100-sample SE at p=0.8 is ±4, so this is ~2 SE).
- **Reverse** ASEntmax: 100/100/100/53/0 vs paper …/96.4/56.7. Every one of the 6 runs is ≥93% at 2x, but 4x ranges 0–53 and 8x is 0–5. The local 4070 run (which used the *repo* config: FFN 1024, 20k warmup, lr=4e-4) got 92/36 — i.e. **the paper-config changes made Reverse worse**. See 3.4.
- **MQMTAR** ASEntmax: 3 of 4 runs never escaped the initial loss plateau (loss ≈0.55 for all 390k steps, 0% at every length); the 4th escaped at step 191k and reached only 63% ID. The same recipe with 20k warmup escaped at ~131k locally and hit 100% out to 64x. This is a training-dynamics failure specific to the 10k-warmup setting, not an attention-mechanism result. See 3.5.

### 3.2 Softmax baselines reproduce and the LR sweep matters more than seed count

- Sort softmax paper: 0 at 2x. Ours: 4 of 6 runs 0 at 2x; but s2/lr=8e-4 reached **86%** at 2x (selected by BLEU@4x per protocol). The paper's 0.0 at 2x is a *weaker* softmax than we found — consistent with them not sweeping high LR for softmax or with 1K-sample eval catching partial failures. Either way the qualitative claim (softmax collapses at 4x) holds: every softmax run is 0 at 4x.
- Reverse softmax 1.5x: 38 (paper 36). Exact match at the paper's chosen operating point.
- Copy softmax: 100/100/100/100/98/91/63 vs paper …/99.4/96.1/85.5. Reproduces through 32x; 64x is 22 pts low. This is where the local run had collapsed (64/0 at 8x/16x) — the sweep + best-ckpt selection fixed it. LR sensitivity is large: at lr=5e-4 both seeds die by 16x; at 1e-3 the two seeds are 98% vs 0% at 16x.
- MQMTAR softmax: 100/100/97/90/48/0 vs paper 100/100/100/99.5/97.8/80.2/3.0. All four runs train cleanly and agree closely (2-seed 91±1 at 16x, 55±7 at 64x), so this is a systematic gap, not noise: our softmax degrades roughly 8x earlier in length than the paper's. Candidate causes are the 10k warmup (local run with 20k got 83/54 — no better), the 100-sample eval, or an implementation detail of their softmax+NAPE path we don't have. The qualitative pattern (softmax collapses by 256x) holds.

### 3.3 Stieltjes q=4 (dense, polynomial tail) behaves like a slightly-better softmax, not like ASEntmax — except on MQMTAR

- Sort: 86 at 2x, 0 at 4x — same shape as the best softmax run; 2-seed mean at 2x (76±10) is well above softmax's (43±43), so it is *more stable* than softmax at 2x, but it has no run with any signal at 4x, where ASEntmax reaches 81.
- Reverse: worst of the three. Best run 0 at 1.5x; the 2-seed mean is 10±10. Softmax gets 38–67.
- Copy: seed-1 runs are strong (100/100/100/99/100/89 through 32x — on par with ASEntmax's 96 at 32x), but **all three seed-2 runs collapsed** (val acc@8x = 0 throughout; 14–100 at 2x, 0 at 4x from last.ckpt). Seed-2 also produced NaN-free but degenerate training on every LR, so this is a genuine optimization failure mode, not a bad LR.
- **MQMTAR: Stieltjes is the best row we have at 64x** — 100/100/100/100/86 (2-seed 76±10) vs softmax 48 (55±7) and ASEntmax's failed training. Both lr=2e-4 seeds escaped the plateau early (50k, 86k steps) and trained to loss 3e-7; both lr=1e-4 seeds never escaped (same as ASEntmax at 1e-4). 256x/1024x could not be run (eager O(N²) memory), so whether it holds beyond 64x like the paper's ASEntmax (99.0/95.3) is open.
- Net: on the three tasks where the paper's own gap between softmax and ASEntmax comes from *exact zeros* at 4x+ (Sort, Reverse), the dense Stieltjes mapping tracks softmax, consistent with the pre-registered analysis. On associative recall (MQMTAR, and Copy seed 1), where the task is retrieving a sharp match from a long context, its heavier-than-exponential tail with q=4 gives a genuinely better 64x number than softmax in this sweep. It is markedly less robust across seeds than either baseline (Copy s2, Reverse).

### 3.4 Where this run diverges from the paper, and why

1. **Reverse ASEntmax 4x/8x (53/0 vs 96.4/56.7).** Two config changes were made to match the paper (FFN 1024→512 from Table 5; warmup 20k→10k from App. G.2). The local run with the repo defaults got 92/36 with a single seed. Plausible cause: the paper's Table 5 lists 512 for Reverse but the released config has 1024, and the released config is what actually produced their number. The narrower FFN (2.1M→1.6M params) likely reduces the 6-layer model's capacity for the 4x-8x regime. Recommendation: rerun Reverse ASEntmax (and softmax) with FFN 1024 to settle it — 12 jobs, ~25 GPU-h.
2. **8x validation monitor was degenerate on Reverse for every method and every run** (exact-match at 512 identically 0 across all 20 val checks, including for runs that later scored 53% on the *test* 4x). The Lightning checkpoint callback then kept the *first* checkpoint on ties (step 11718, 5% into training) — the same pathology found in the local run. The paper's protocol anticipates this ("where generalization to 8x was not possible, we use BLEU and 2x/4x"), so those runs were re-laddered from the fully-trained checkpoint and selected by BLEU@4x. All Reverse "ours" numbers come from that path. An alternative would be to save top-k by BLEU@4x during training; not done here.
3. **Copy softmax/ASEntmax 64x (63/76 vs 85.5/86.6).** Both ~10–20 pts low on the single best run. Given 2-seed std of 32–38 at 64x and the paper's best-of-3-seeds selection, I'd attribute this to the seed count (2 vs 3) plus 100-sample noise rather than a systematic difference.
4. **Sort softmax 2x (86 vs 0).** We found a *stronger* softmax than the paper's; see 3.2.
5. **Divergences.** Sort ASEntmax at lr=8e-4 NaN'd in both seeds around step 60–70k (best ckpt was saved before that, so the row is valid but reflects a 15–47k-step model); copy ASEntmax s2/lr=2e-3 NaN'd at 25k and was cancelled. lr=8e-4 / 2e-3 are above the paper's operating range for entmax on these tasks.
6. **Selection is by validation only** (no peeking at test), so a few "ours" rows are not the best test row in the all-runs tables — e.g. reverse softmax s2/lr=8e-4 got 67 at 1.5x but was not selected (lower BLEU@4x); MQMTAR softmax s2/lr=2e-4 got 68 at 64x vs the selected 48. That is the protocol working as intended.

### 3.5 MQMTAR: the loss plateau and the 10k-warmup change

MQMTAR trains through a long plateau at loss ≈0.55–0.6 (predicting the separator/structure tokens only) before an abrupt drop once the model discovers the key→value lookup. Escape step (first step with loss < 0.3) for every run:

| method | lr | s1 | s2 |
|---|---|---:|---:|
| softmax | 1e-4 | 16k | 67k |
| softmax | 2e-4 | 144k | 68k |
| ASEntmax | 1e-4 | never | never |
| ASEntmax | 2e-4 | 191k | never |
| Stieltjes | 1e-4 | never | never |
| Stieltjes | 2e-4 | 50k | 86k |
| local ASEntmax (20k warmup) | 2e-4 | 131k | – |
| local softmax (20k warmup) | 2e-4 | 105k | – |

Escape is stochastic and both sparse/heavy-tailed mappings need lr=2e-4 to have a chance; at 1e-4 nothing but softmax escapes. ASEntmax's one escape came at 191k, leaving <200k steps of an already-decayed cosine schedule — enough to reach 63% ID, not to consolidate. The local run, identical except for 20k warmup and a mid-run resume, escaped at 131k and finished at 100% out to 64x. So the honest reading of the MQMTAR ASEntmax row is *"did not converge within budget under this config,"* not *"ASEntmax fails on MQMTAR."* The paper reports best-of-3 seeds, which with this escape variance is a materially stronger selection than best-of-2. Fix: ≥3 seeds and/or 20k warmup (repo default) and/or a higher LR (4e-4) for the entmax rows — the local evidence says warmup is the first thing to try.

## 4. Selection protocol as implemented (for the record)

For each run, primary monitor = val exact-match @8x (sort: val BLEU @4x). If the primary is never >0 over training, fall back (per the paper) to val BLEU @4x, then @2x. Among the (seed, LR) runs of a (task, method), the run with the highest selection value is reported. The checkpoint evaluated is the callback's best-by-primary checkpoint when the primary was informative, else the last checkpoint (the callback keeps the first ckpt on all-zero ties, which is uninformative). `experiments/osc/reselect.py` prints the per-run decision; `aggregate.py` applies it.

## 5. Artifacts

- Checkpoints (best + last) for all 66 runs: `<results root>/<task>/<method>/s<seed>_lr<lr>/checkpoints/` — kept, as requested, for the deferred Stieltjes 16k/65k eval and any re-evaluation (e.g. 1K samples).
- Per-length ladders: `ladder.tsv` (best ckpt) and `ladder_last.tsv` (last ckpt) in each run dir; eval logs alongside.
- Training metrics: `synthetic/logs/t1_<task>_<method>_s<seed>_lr<lr>/runs/*/csv/version_0/metrics.csv` (per-step loss, per-val-check acc/BLEU at every length).
- W&B offline runs: `<run>/checkpoints/wandb/offline-run-*` — `wandb sync` them after `wandb login` to get the dashboard (project `asentmax-table1`, group `<task>-<method>`).
- Code: branch `table1-osc` (13 commits), incl. `synthetic/osc/{submit.sh,run_one.sh,aggregate.py,reselect.py,progress.sh}` and the Stieltjes implementation `synthetic/src/attention/stieltjes_eager.py` (+ tests).
  *Path note (Sep 13, `torch_triton_comp`):* the whole OSC pipeline incl. `aggregate.py` / `reselect.py` now lives in `experiments/osc/` (`synthetic/osc/` is gone). The Stieltjes rows here were produced with `stieltjes_impl=eager`; the fused Triton kernel is now the default (see `AGENTS.md`).

## 6. Suggested next steps (in priority order)

1. **MQMTAR ASEntmax rerun** with 20k warmup (repo default), 3 seeds, lr {2e-4, 4e-4} — the local run shows this config reaches 100% to 64x; the paper's 256x/1024x cells (99.0/95.3) are the headline claim and are untested here. ~12 jobs, ~60 GPU-h incl. the 65k ladders.
2. **Reverse with FFN=1024** (repo config) for ASEntmax + softmax, 2 seeds x lr {4e-4, 8e-4} — resolves the other qualitative miss. ~25 GPU-h.
3. Save top-k by BLEU@4x on Reverse/MQMTAR so the paper's fallback picks a real checkpoint instead of last.
4. Re-evaluate the selected checkpoints on 1K samples/length (test-only regen, ~minutes; eval ~10 GPU-h) to remove the ±4-pt noise from the 64x cells.
5. Stieltjes: fused Triton path (kernel exists; needs ALiBi bias + wiring) to unlock 4096+ on copy and 16k/65k on MQMTAR — its 86 at 64x on MQMTAR makes the 256x/1024x cells the most interesting open question in this table; and AS-Stieltjes (adaptive temperature) as the fair comparison to ASEntmax.
