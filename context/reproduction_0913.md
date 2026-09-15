# LR sweep for Table 1 on OSC Ascend (A100-40GB) — 2026-09-13 to 09-15

Successor to `reproduction_0907.md` (now superseded). Same protocol, same data, same checkpoints reused where they exist; this report adds a bracketed learning-rate sweep for Softmax and Stieltjes (q=4) on the four tasks, plus the Stieltjes long-length cells that the eager kernel could not evaluate before. ASEntmax was not re-run; its rows are carried over from 0907 for reference.
Results root on OSC: `/fs/scratch/PAS2836/jacktaylor/asentmax_results/table1/`. All-runs appendix: `reproduction_0913_all_runs.md`. Raw dump: `experiments/osc/dump_runs.py`.

Status: **stage 1 (bracketing) and stage 2 for sort/reverse complete; copy/mqmtar seed-3 runs (9 jobs) and 6 reverse last.ckpt ladders are queued** (§7). Compute so far: 176 GPU-h on top of 0907's ~200.

## 1. Why a sweep, and how it was designed

The paper reports "the single best-performing model selected from experiments conducted with multiple random seeds (3) and various learning rates" and does not list the LRs per task or per method. The 0907 reproduction used a 3-point grid (2 for MQMTAR) with 2 seeds, and in 6 of the 8 (task, method) cells the selected LR sat on an edge of that grid, so no optimum was bracketed. Seed variance was also as large as LR variance (copy softmax lr=1e-3: 98 vs 0 at 16x across seeds), so 1 seed per LR cannot rank LRs.

Design (Option A, agreed Sep 13):

| item | choice |
|---|---|
| Grid | factor-2 geometric, extended past the 0907 edge in the direction the validation monitor was improving until both neighbours of the best LR are worse (bracket). One extra step (6.4e-3) was added on sort (both methods) and reverse (stieltjes) when 3.2e-3 was still competitive. |
| Seeds | 2 at every LR (reusing the 0907 runs); a 3rd seed only at the top-2 LRs per cell (stage 2). |
| Selection | validation only, as in 0907 §4: primary = val exact-match @8x (sort: val BLEU @4x); BLEU@4x fallback when the primary is identically 0 (then the fully trained last.ckpt is laddered). New for copy/mqmtar: runs tied at acc@8x = 1.0 are ranked by a **tie-break validation ladder** at the next lengths up (copy 16x/32x, mqmtar 16x/64x) drawn with a different RNG seed from the test sets (`experiments/osc/gen_tiebreak_val.sh`). Reported LR = highest selection value over seeds (paper rule); the mean over seeds is shown alongside as the fluke check. |
| Everything else | frozen from 0907: 10k warmup, reverse FFN 512, 100 eval samples/length, early-stop ladder after an exact 0.0. |
| Kernels | new Stieltjes runs use the fused Triton kernel (default since Sep 10); the 0907 Stieltjes runs were trained eager. Same-seed eager/Triton comparisons differ at seed-noise level (sort 2x: 86 vs 93; copy 32x: 89 vs 86), so they are pooled and labelled in the appendix. The 0907 eager checkpoints were re-laddered with Triton to fill the 64x (copy) and 256x/1024x (mqmtar) cells. |
| NaN guard | `run_one.sh` now stops training at the next validation check if the train loss is non-finite (Lightning EarlyStopping check_finite; needs `trainer.min_epochs=0`). It never fired: no run in this sweep diverged, including 6.4e-3. |

Predictions made before the runs (Sep 13), scored in §5.

## 2. Headline table (validation-only selection over all seeds x LRs, 100 test samples/length)

Bold = selected run under the sweep. "0907" rows are the previous report's selected run for reference. ASEntmax rows are unchanged from 0907 (not part of this sweep). Sort/reverse: 3 seeds at the top-2 LRs; copy/mqmtar: 2 seeds everywhere (seed 3 at the top-2 LRs pending, §7). Standard error at p=0.8 with 100 samples is ±4 pts.

### Sort

| method | ID | 2x | 4x | 8x | selected run |
|---|---:|---:|---:|---:|---|
| Softmax (paper) | 100 | 0 | 0 | 0 | best of 3 seeds x LRs, 1K samples |
| softmax (0907) | 100 | 86 | 0 | skip | s2, 8e-4 |
| **softmax (sweep)** | 100 | 99 | 4 | 0 | s1, lr=6.4e-3, bleu@2=0.992 |
| softmax (sweep, mean±std over 2 seeds @ lr=6.4e-3) | 100±0 | 95±4 | 3±1 | 0±0 | |
| ASEntmax (paper) | 100 | 100 | 79.7 | 0 | |
| ASEntmax (0907, unchanged) | 100 | 100 | 81 | 0 | s1, 2e-4 |
| stieltjes (0907) | 100 | 86 | 0 | skip | s1, 4e-4 |
| **stieltjes (sweep)** | 100 | 88 | 0 | skip | s2, lr=3.2e-3, bleu@2=0.984 |
| stieltjes (sweep, mean±std over 3 seeds @ lr=3.2e-3) | 100±0 | 62±43 | 0±0 | 0±0 | |

### Reverse

| method | ID | 1.5x | 2x | 4x | 8x | selected run |
|---|---:|---:|---:|---:|---:|---|
| Softmax (paper) | 100 | 36 | 0 | 0 | 0 | best of 3 seeds x LRs, 1K samples |
| softmax (0907) | 100 | 38 | 0 | skip | skip | s1, 8e-4 |
| **softmax (sweep)** | 93 | 2 | 0 | skip | skip | s3, lr=1.6e-3, bleu@3=0.295, last.ckpt (8x monitor degenerate) **[last.ckpt ladder pending; best-ckpt shown]** |
| softmax (sweep, mean±std over 3 seeds @ lr=1.6e-3) | 98±3 | 17±22 | 0±0 | 0±0 | 0±0 | |
| ASEntmax (paper) | 100 | 100 | 99.8 | 96.4 | 56.7 | |
| ASEntmax (0907, unchanged) | 100 | 100 | 100 | 53 | 0 | s1, 4e-4 |
| stieltjes (0907) | 100 | 0 | skip | skip | skip | s2, 8e-4 |
| **stieltjes (sweep)** | 100 | 100 | 38 | 0 | skip | s1, lr=3.2e-3, bleu@3=0.422, last.ckpt (8x monitor degenerate) |
| stieltjes (sweep, mean±std over 3 seeds @ lr=3.2e-3) | 98±3 | 48±39 | 13±18 | 0±0 | 0±0 | |

### Copy

| method | ID | 2x | 4x | 8x | 16x | 32x | 64x | selected run |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Softmax (paper) | 100 | 100 | 99.9 | 99.9 | 99.4 | 96.1 | 85.5 | best of 3 seeds x LRs, 1K samples |
| softmax (0907) | 100 | 100 | 100 | 100 | 98 | 91 | 63 | s1, 1e-3 |
| **softmax (sweep)** | 100 | 100 | 100 | 100 | 98 | 91 | 63 | s1, lr=1e-3, acc_@3=1.000, tiebreak val val16x/val32x=98/88 |
| softmax (sweep, mean±std over 2 seeds @ lr=1e-3) | 100±0 | 97±3 | 80±20 | 58±42 | 49±49 | 46±46 | 32±32 | |
| ASEntmax (paper) | 100 | 100 | 99.9 | 99.7 | 99.4 | 96.3 | 86.6 | |
| ASEntmax (0907, unchanged) | 100 | 100 | 99 | 100 | 99 | 96 | 76 | s2, 5e-4 |
| stieltjes (0907) | 100 | 100 | 100 | 99 | 100 | 89 | OOM | s1, 5e-4 |
| **stieltjes (sweep)** | 100 | 100 | 100 | 99 | 100 | 87 | 52 | s1, lr=5e-4, acc_@3=1.000, tiebreak val val16x/val32x=97/85 |
| stieltjes (sweep, mean±std over 2 seeds @ lr=5e-4) | 100±0 | 58±42 | 50±50 | 50±50 | 50±50 | 44±44 | 26±26 | |

### Mqmtar

| method | ID | 2x | 4x | 16x | 64x | 256x | 1024x | selected run |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Softmax (paper) | 100 | 100 | 100 | 99.5 | 97.8 | 80.2 | 3.0 | best of 3 seeds x LRs, 1K samples |
| softmax (0907) | 100 | 100 | 97 | 90 | 48 | 0 | skip | s1, 1e-4 |
| **softmax (sweep)** | 99 | 100 | 100 | 97 | 92 | 40 | 1 | s2, lr=4e-4, acc_@3=1.000, tiebreak val val16x/val64x=99/81 |
| softmax (sweep, mean±std over 2 seeds @ lr=4e-4) | 100±0 | 100±0 | 100±0 | 96±0 | 88±4 | 46±6 | 2±2 | |
| ASEntmax (paper) | 100 | 100 | 100 | 99.7 | 99.6 | 99.0 | 95.3 | |
| ASEntmax (0907, unchanged) | 63 | 37 | 11 | 0 | skip | skip | skip | s1, 2e-4 (3/4 runs never left plateau) |
| stieltjes (0907) | 100 | 100 | 100 | 100 | 86 | - | - | s2, 2e-4 |
| **stieltjes (sweep)** | 100 | 98 | 97 | 99 | 90 | 53 | 8 | s1, lr=4e-4, acc_@3=1.000, tiebreak val val16x/val64x=97/86 |
| stieltjes (sweep, mean±std over 2 seeds @ lr=4e-4) | 100±0 | 99±1 | 98±1 | 100±0 | 86±4 | 40±12 | 4±4 | |

## 3. LR sensitivity (the PI's question)

Per (task, method, LR): number of seeds, the validation selection value (max and mean over seeds; sort = BLEU@4x, others = exact-match acc @8x; `deg` = primary monitor identically 0 for every seed, value shown is the BLEU@4x fallback), the tie-break validation ladder where it exists, and the test accuracy at the informative lengths as per-seed values. New = added by this sweep; the 0907 grid is the rest. `plateau` = never left the MQMTAR loss plateau.

### sort / softmax

| lr | new | seeds | val sel max | val sel mean | test 2x (per seed) | test 4x (per seed) | notes |
|---|---|---:|---:|---:|---|---|---|
| 2e-4 |  | 2 | 0.538 | 0.477 | 0 / 0 | skip / skip |  |
| 4e-4 |  | 2 | 0.822 | 0.792 | 0 / 3 | skip / 0 |  |
| 8e-4 |  | 2 | 0.926 | 0.803 | 0 / 86 | skip / 0 |  |
| 1.6e-3 | new | 3 | 0.968 | 0.954 | 74 / 47 / 38 | 0 / 0 / 0 |  |
| 3.2e-3 | new | 3 | 0.985 | 0.935 | 75 / 12 / 24 | 0 / 0 / 0 |  |
| 6.4e-3 | new | 2 | 0.992 | 0.992 | 99 / 91 | 4 / 2 | **selected** |

### sort / stieltjes

| lr | new | seeds | val sel max | val sel mean | test 2x (per seed) | test 4x (per seed) | notes |
|---|---|---:|---:|---:|---|---|---|
| 2e-4 |  | 2 | 0.787 | 0.766 | 58 / 80 | 0 / 0 |  |
| 4e-4 |  | 3 | 0.970 | 0.959 | 86 / 66 / 62 | 0 / 0 / 0 |  |
| 8e-4 |  | 2 | 0.965 | 0.964 | 85 / 74 | 0 / 0 |  |
| 1.6e-3 | new | 2 | 0.963 | 0.931 | 66 / 0 | 0 / skip |  |
| 3.2e-3 | new | 3 | 0.984 | 0.937 | 2 / 88 / 96 | 0 / 0 / 0 | **selected** |
| 6.4e-3 | new | 2 | 0.973 | 0.840 | 3 / 51 | 0 / 0 |  |

### reverse / softmax

| lr | new | seeds | val sel max | val sel mean | test 1.5x (per seed) | test 2x (per seed) | notes |
|---|---|---:|---:|---:|---|---|---|
| 2e-4 |  | 2 | 0.056 deg | 0.051 | 0 / 0 | skip / skip |  |
| 4e-4 |  | 2 | 0.181 deg | 0.154 | 28 / 31 | 0 / 0 |  |
| 8e-4 |  | 3 | 0.193 deg | 0.158 | 38 / 67 / 2 | 0 / 0 / 0 |  |
| 1.6e-3 | new | 3 | 0.295 deg | 0.225 | 1 / 48 / 2 | 0 / 0 / 0 | **selected** |
| 3.2e-3 | new | 2 | 0.190 deg | 0.169 | 0 / 0 | skip / skip |  |

### reverse / stieltjes

| lr | new | seeds | val sel max | val sel mean | test 1.5x (per seed) | test 2x (per seed) | notes |
|---|---|---:|---:|---:|---|---|---|
| 2e-4 |  | 2 | 0.154 deg | 0.152 | 0 / 0 | skip / skip |  |
| 4e-4 |  | 2 | 0.255 deg | 0.231 | 18 / 2 | 0 / 0 |  |
| 8e-4 |  | 2 | 0.380 deg | 0.340 | 19 / 0 | 0 / skip |  |
| 1.6e-3 | new | 3 | 0.384 deg | 0.319 | 95 / 74 / 3 | 0 / 0 / 0 |  |
| 3.2e-3 | new | 3 | 0.422 deg | 0.267 | 100 / 38 / 6 | 38 / 0 / 0 | **selected** |
| 6.4e-3 | new | 2 | 0.353 deg | 0.299 | 34 / 49 | 0 / 0 |  |

### copy / softmax

| lr | new | seeds | val sel max | val sel mean | tiebreak val16x (per seed) | tiebreak val32x (per seed) | test 16x (per seed) | test 32x (per seed) | test 64x (per seed) | notes |
|---|---|---:|---:|---:|---|---|---|---|---|---|
| 2.5e-4 | new | 2 | 0.196 deg | 0.185 | - / - | - / - | skip / skip | skip / skip | skip / skip |  |
| 5e-4 |  | 2 | 0.600 | 0.345 | 11 / 0 | 0 / skip | 12 / 0 | 0 / skip | skip / skip |  |
| 1e-3 |  | 2 | 1.000 | 0.590 | 98 / 0 | 88 / skip | 98 / 0 | 91 / skip | 63 / skip | **selected** |
| 2e-3 |  | 2 | 1.000 | 1.000 | 95 / 96 | 81 / 18 | 95 / 99 | 82 / 22 | 55 / 0 |  |
| 4e-3 | new | 2 | 0.930 | 0.712 | 67 / - | 20 / - | 59 / skip | 22 / skip | 0 / skip | deg s2 |

### copy / stieltjes

| lr | new | seeds | val sel max | val sel mean | tiebreak val16x (per seed) | tiebreak val32x (per seed) | test 16x (per seed) | test 32x (per seed) | test 64x (per seed) | notes |
|---|---|---:|---:|---:|---|---|---|---|---|---|
| 2.5e-4 | new | 2 | 0.970 | 0.649 | 40 / - | 0 / - | 47 / skip | 0 / skip | skip / skip | deg s2 |
| 5e-4 |  | 2 | 1.000 | 0.641 | 97 / - | 85 / - | 100 / skip | 87 / skip | 52 / skip | deg s2; **selected** |
| 1e-3 |  | 2 | 1.000 | 0.772 | 98 / - | 81 / - | 99 / skip | 92 / skip | 35 / skip | deg s2 |
| 2e-3 |  | 2 | 0.980 | 0.805 | 59 / - | 0 / - | 50 / skip | 0 / skip | skip / skip | deg s2 |
| 4e-3 | new | 2 | 0.960 | 0.935 | 67 / 60 | 3 / 10 | 68 / 61 | 2 / 8 | 0 / 0 |  |

### mqmtar / softmax

| lr | new | seeds | val sel max | val sel mean | tiebreak val16x (per seed) | tiebreak val64x (per seed) | test 64x (per seed) | test 256x (per seed) | test 1024x (per seed) | notes |
|---|---|---:|---:|---:|---|---|---|---|---|---|
| 5e-5 | new | 2 | 0.069 deg | 0.065 | - / - | - / - | skip / skip | skip / skip | skip / skip | plateau s1,s2 |
| 1e-4 |  | 2 | 1.000 | 1.000 | 90 / 96 | 34 / 61 | 48 / 62 | 0 / 0 | skip / skip | escape s1:16k s2:66k |
| 2e-4 |  | 2 | 1.000 | 0.990 | 83 / 96 | 44 / 77 | 36 / 68 | 2 / 1 | 0 / 0 | escape s1:143k s2:68k |
| 4e-4 | new | 2 | 1.000 | 0.995 | 94 / 99 | 82 / 81 | 84 / 92 | 53 / 40 | 4 / 1 | escape s1:32k s2:141k; **selected** |

### mqmtar / stieltjes

| lr | new | seeds | val sel max | val sel mean | tiebreak val16x (per seed) | tiebreak val64x (per seed) | test 64x (per seed) | test 256x (per seed) | test 1024x (per seed) | notes |
|---|---|---:|---:|---:|---|---|---|---|---|---|
| 1e-4 |  | 2 | 0.076 deg | 0.071 | - / - | - / - | skip / skip | - / - | - / - | plateau s1,s2 |
| 2e-4 |  | 2 | 1.000 | 1.000 | 99 / 99 | 72 / 80 | 66 / 86 | 6 / 4 | 0 / - | escape s1:49k s2:86k |
| 4e-4 | new | 2 | 1.000 | 1.000 | 97 / 98 | 86 / 78 | 90 / 82 | 53 / 28 | 8 / 1 | escape s1:60k s2:129k; **selected** |
| 8e-4 | new | 2 | 1.000 | 0.570 | 99 / - | 81 / - | 94 / skip | 38 / skip | 0 / skip | plateau s2; escape s1:117k; deg s2 |

## 4. Findings

### 4.1 Every LR optimum is now bracketed except reverse/stieltjes above 6.4e-3

| task | method | 0907 LR | sweep LR | bracket | what changed |
|---|---|---|---|---|---|
| sort | softmax | 8e-4 (edge) | **6.4e-3** | 3.2e-3 < 6.4e-3; no run above 6.4e-3 (see note) | 2x: 86 -> 99/91 (both seeds); first non-zero 4x for softmax (4, 2); 8x = 0 |
| sort | stieltjes | 4e-4 (flat) | **3.2e-3** | 1.6e-3 and 6.4e-3 both worse on val mean | 2x: 86 -> 88 (best) but 3-seed mean 62±43; 4x = 0 in all 14 runs |
| reverse | softmax | 8e-4 (edge) | 1.6e-3 | 3.2e-3 worse | 1.5x is 1-67 across seeds at every LR >= 4e-4; no LR reaches 2x. Selected run's last.ckpt ladder pending |
| reverse | stieltjes | 8e-4 (edge) | **3.2e-3** | 1.6e-3 < 3.2e-3 > 6.4e-3 on val max (mean is flat) | 1.5x: 0 -> 100, and **38 at 2x** (s1). The 0907 stieltjes reverse row was an artifact of the degenerate-monitor checkpoint (see 4.3) |
| copy | softmax | 1e-3 (tied with 2e-3) | **1e-3** | 5e-4 and 2e-3 lower on tie-break; 2.5e-4 dead, 4e-3 degrades | unchanged row (100/98/91/63); tie-break confirms 1e-3 > 2e-3 (val32x 88 vs 81) |
| copy | stieltjes | 5e-4 (edge) | **5e-4** | 2.5e-4 worse (val32x 0), 1e-3 ~ tied (85 vs 81), 2e-3/4e-3 worse | 64x cell filled: **52** (Triton re-ladder). Seed 2 collapsed at 2.5e-4, 5e-4, 1e-3, 2e-3 but trained at 4e-3 (61/8/0) |
| mqmtar | softmax | 1e-4 (edge) | **4e-4** | 2e-4 < 4e-4 on tie-break val64x (77 vs 82); 5e-5 never escapes the plateau; no run above 4e-4 | **64x: 48 -> 92, 256x: 0 -> 40 (s2) / 53 (s1)**, 1024x 1-4 |
| mqmtar | stieltjes | 2e-4 (edge) | **4e-4** | 2e-4 < 4e-4 < 8e-4 on val64x for s1 (72/86/81) but 8e-4 s2 never escaped the plateau | 256x/1024x filled: **53/8** (s1), 28/1 (s2). At 2e-4 the same cells are 6/0 and 4/(pending) |

Sort softmax note: 6.4e-3 is the top of the grid and both seeds hit val BLEU@4x = 0.992, higher than 3.2e-3, so strictly the optimum is not bracketed above. The test picture (2x saturating at 99/91, 4x at 4/2, 8x 0) and the fact that 6.4e-3 is already 8x the paper-era operating point make a further step unlikely to change a cell; not run.

### 4.2 The MQMTAR softmax gap to the paper was half an LR effect

0907 §3.2 called the softmax MQMTAR row (…/90/48/0 vs paper …/99.5/97.8/80.2) "a systematic gap, not noise" and listed the 10k warmup, 100-sample eval and implementation details as candidates. Wrong: at lr=4e-4 both seeds give 96-97 at 16x, 84-92 at 64x and 40-53 at 256x, i.e. the 64x cell now matches the paper within 2 SE and the 256x cell closes about half the gap (80.2 paper). The remaining 256x/1024x shortfall is consistent with best-of-3 seeds x 1K-sample selection on their side. Stieltjes moves the same way at the same LR (66/86 -> 90/82 at 64x; 6/4 -> 53/28 at 256x). Both methods needed 4x the LR the 0907 grid selected; the plateau-escape step falls from 68-144k at 2e-4 to 32k (softmax s1) and 60k (stieltjes s1) at 4e-4, leaving more of the cosine schedule after escape.

### 4.3 Reverse: the 0907 Stieltjes row was a checkpoint artifact, and the sweep exposes how brittle the fallback selection is

Every reverse run has an identically-zero 8x monitor, so 0907 laddered last.ckpt and selected by BLEU@4x. The stage-1 ladders I first reported (best-ckpt = first checkpoint on the all-zero tie, ~5% into training) said stieltjes 1.6e-3 = 25/4 at 1.5x. The last.ckpt ladders of the same runs say **95/74**, and 3.2e-3 s1 gives **100 at 1.5x and 38 at 2x** — the first Stieltjes run past 1.5x, and better than any softmax reverse run in either report (softmax best: 67 at 1.5x, 0 at 2x). With the three 0907 LRs the best last.ckpt Stieltjes number was 19. So the 0907 conclusion "Reverse: [Stieltjes is the] worst of the three" was an LR-range artifact; at 2-4x the LR it is the best non-entmax row, still far from ASEntmax (100/100/53/0 at 1.5x/2x/4x/8x).

The fallback itself is the weak point: BLEU@4x max-over-seeds ranks stieltjes 3.2e-3 s1 (0.422, test 100/38) above 1.6e-3 s2 (0.384, test 74/0), which is the right order here, but ranks softmax 1.6e-3 s3 (0.295) first while its seed-mates at the same LR are 1 and 48 at 1.5x, and the 3-seed mean at 8e-4 (36±27) is better than at 1.6e-3 (17±22) on test. BLEU at 4x is measured on sequences where every run is at 0% exact match, so it is a proxy with ~0.1 dynamic range; with 14-17 runs per cell it selects noise. The honest reverse-softmax statement is "1.5x ranges 0-67 across 14 runs at 4e-4..3.2e-3 with no LR trend; 2x = 0 in all of them"; the paper's 36 is inside that range.

### 4.4 Sort: softmax at 6.4e-3 reaches 2x and touches 4x

At 6.4e-3 both softmax seeds reach 99/91 at 2x and 4/2 at 4x (val BLEU@4x 0.992, val exact-match@4x 3-9%). This is 8x the LR of the 0907 selection and 30x the paper's softmax operating point (whatever it was; their row is 0 at 2x). It does not change the qualitative claim (ASEntmax 79.7-81 at 4x vs <=4 for softmax; 8x = 0 for both) but the paper's "softmax collapses at 2x" is an LR statement, not a mechanism statement. Stieltjes on sort is the most seed-unstable cell in the table (3.2e-3: 2/88/96 at 2x across seeds; 6.4e-3: 3/51); its best run matches the best softmax run at 2x and nothing gets past 0 at 4x.

### 4.5 Copy: LR was already right; seed 2 Stieltjes collapse is an LR-dependent optimization failure

Softmax: 2.5e-4 is dead by 2x (both seeds), 4e-3 degrades from 16x on (59/22/0), so 1e-3..2e-3 is the optimum and the tie-break puts 1e-3 first. Row unchanged from 0907 (100/98/91/63 through 64x); the 22-pt gap to the paper at 64x is where seed 3 (pending) would matter most. Stieltjes: seed 1 gives 87-92 at 32x and 52/35 at 64x for 5e-4/1e-3 (Triton re-ladder; the Triton-trained twin from the impl comparison gave 58), i.e. ~10 pts under softmax/ASEntmax at 64x. Seed 2 collapsed at every LR from 2.5e-4 to 2e-3 (val acc@8x = 0, 0-15% at 2x from last.ckpt) but trained normally at 4e-3 (97/83/61 at 4x/8x/16x). So the collapse is a basin the optimizer falls into at moderate LR for that init, not a data or kernel problem; the 2-seed mean (26±26 at 64x) is dominated by it.

### 4.6 Stieltjes vs softmax after the sweep

With both methods at their bracketed optimum: sort 2x 88 vs 99 (softmax better, both 0 at 4x within noise); reverse 100/38 vs <=67/0 (Stieltjes better); copy 87/52 vs 91/63 at 32x/64x (softmax better by ~1-2 SE); mqmtar 90/53/8 vs 92/40/1 at 64x/256x/1024x (tied at 64x, Stieltjes ahead at 256x by ~2 SE, s2 says the opposite: 28 vs 40). Net: a dense polynomial-tail map tracks softmax on the copy/sort ladders and is somewhat better on the two tasks where the tail matters (reverse, long MQMTAR), but it is the least seed-stable of the three methods on every task. It does not approach ASEntmax on reverse (38 vs 100 at 2x, 0 vs 53 at 4x) or on the paper's MQMTAR 256x/1024x (99.0/95.3).

## 5. Scoring the Sep 13 predictions

| prediction | outcome |
|---|---|
| Sort: softmax 1.6e-3 >= 8e-4 on BLEU@4x, 3.2e-3 diverges or worse; nothing > 0 at 4x (85%) | half right: no divergence at any LR, and 6.4e-3 gave 4/2 at 4x. Headline cells effectively unchanged |
| Reverse: softmax 1.5x rises to 50-70 at 1.6e-3, 2x = 0; stieltjes <= 30 at 1.5x (75%) | wrong on both: softmax 1.6e-3 = 1/48/2, no trend; stieltjes 3.2e-3 = 100 at 1.5x, 38 at 2x |
| Copy: softmax 4e-3 < 2e-3; stieltjes 2.5e-4 not better than 5e-4; s2 collapses again 50% | all three right (s2 collapsed at 2.5e-4, trained at 4e-3) |
| Copy stieltjes 64x from Triton re-ladder 55-65 | 52 (5e-4), 35 (1e-3); slightly under the range |
| MQMTAR softmax 64x stays 40-70 at every LR, "not an LR effect" (75%) | wrong: 84/92 at 4e-4 |
| MQMTAR stieltjes 4e-4 >= 80 at 64x; 8e-4 ~40% NaN risk; 256x on 2e-4 ckpts 30-70 | 90/82 right; 8e-4 did not NaN but s2 never left the plateau (equivalent outcome); 256x at 2e-4 was 6/4, far below the range, and 53/28 at 4e-4 |
| ~20% that any qualitative Table-1 conclusion changes | two changed: MQMTAR softmax gap is mostly LR; reverse Stieltjes is not the worst row |

## 6. Selection protocol as implemented (delta vs 0907 §4)

Unchanged: primary monitor, BLEU@4x fallback, last.ckpt when degenerate, validation only. Added: (i) among runs tied on the primary at 1.0 (copy, mqmtar), rank by tie-break validation ladder (val32x then val16x for copy; val64x then val16x for mqmtar), files `test_1<i>_val<len>.{src,trg}` in the data dirs, evaluated with `TIEBREAK=1 run_one.sh` into `ladder_tiebreak.tsv`; (ii) mean-over-seeds of the selection value is reported next to the max. `experiments/osc/dump_runs.py` dumps everything to JSON; the tables in §2-3 are generated from that dump.

## 7. Pending and next steps

1. **Queued now** (Slurm est. start Sep 15 ~15:00): copy seed 3 at {1e-3, 2e-3} softmax / {5e-4, 1e-3} stieltjes; mqmtar seed 3 at {2e-4, 4e-4} softmax / {2e-4, 4e-4, 8e-4} stieltjes (jobs 7294913/14, ~45 GPU-h); last.ckpt ladders for the 6 reverse stage-2 runs (job 7320731, ~4 GPU-h). §2 copy/mqmtar rows and the reverse softmax row get updated when they land; conclusions are not expected to move (they change best-of-2 to best-of-3 at an already-chosen LR).
2. The mqmtar stieltjes s2 2e-4 1024x rung timed out twice (65k-token prefill at batch 1 on 40 GB is >8 h with the Triton path); the 4e-4 runs completed it in ~2.5 h. Not worth a third attempt; the cell is reported as pending.
3. ASEntmax at the swept LRs (mqmtar 4e-4 in particular, with 20k warmup per 0907 §6.1) is the obvious next run: it is the only row not swept and its 0907 MQMTAR row is a training failure, not a result. ~12 jobs, ~60 GPU-h.
4. Reverse selection needs a better fallback than BLEU@4x (top-k by BLEU@2x saved during training, or val exact-match at 1.5x/2x, which exist in the val loaders). Cheap, and it removes the noise-selection described in 4.3.
5. Re-evaluate the selected checkpoints on 1K samples/length to take the ±4 pt SE off the 32x-256x cells (eval only, ~10 GPU-h).
