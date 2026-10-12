# LR sweep for Table 1 on OSC Ascend (A100-40GB) — 2026-09-13 to 09-15

Successor to `reproduction_0907.md` (now superseded). Same protocol, same data, same checkpoints reused where they exist; this report adds a bracketed learning-rate sweep for Softmax and Stieltjes (q=4) on the four tasks, plus the Stieltjes long-length cells that the eager kernel could not evaluate before. ASEntmax was not re-run; its rows are carried over from 0907 for reference.
Results root on OSC: `/fs/scratch/PAS2836/jacktaylor/asentmax_results/table1/`. All-runs appendix: `reproduction_0913_all_runs.md`. Raw dump: `experiments/osc/dump_runs.py`.

Status: **complete** (Sep 15). 51 new training runs + 41 eval-only jobs, 210 GPU-h on top of 0907's ~200 (estimate was 185). No run diverged; no job failed. One eval rung (mqmtar stieltjes s2 2e-4 at 1024x) was dropped after two 8-12 h timeouts (§7).

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

Bold = selected run under the sweep. "0907" rows are the previous report's selected run for reference. ASEntmax rows are unchanged from 0907 (not part of this sweep). Every cell has 2 seeds at every LR and 3 seeds at the top-2 LRs. Standard error at p=0.8 with 100 samples is ±4 pts.

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
| **softmax (sweep)** | 100 | 22 | 0 | skip | skip | s3, lr=1.6e-3, bleu@3=0.295, last.ckpt (8x monitor degenerate) |
| softmax (sweep, mean±std over 3 seeds @ lr=1.6e-3) | 100±0 | 24±19 | 0±0 | 0±0 | 0±0 | |
| ASEntmax (paper) | 100 | 100 | 99.8 | 96.4 | 56.7 | |
| ASEntmax (0907, unchanged) | 100 | 100 | 100 | 53 | 0 | s1, 4e-4 |
| stieltjes (0907) | 100 | 0 | skip | skip | skip | s2, 8e-4 |
| **stieltjes (sweep)** | 100 | 100 | 38 | 0 | skip | s1, lr=3.2e-3, bleu@3=0.422, last.ckpt (8x monitor degenerate) |
| stieltjes (sweep, mean±std over 3 seeds @ lr=3.2e-3) | 100±0 | 78±28 | 13±18 | 0±0 | 0±0 | |

### Copy

| method | ID | 2x | 4x | 8x | 16x | 32x | 64x | selected run |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Softmax (paper) | 100 | 100 | 99.9 | 99.9 | 99.4 | 96.1 | 85.5 | best of 3 seeds x LRs, 1K samples |
| softmax (0907) | 100 | 100 | 100 | 100 | 98 | 91 | 63 | s1, 1e-3 |
| **softmax (sweep)** | 100 | 100 | 100 | 100 | 98 | 91 | 63 | s1, lr=1e-3, acc_@3=1.000, tiebreak val 16x/32x=98/88 |
| softmax (sweep, mean±std over 3 seeds @ lr=1e-3) | 100±0 | 98±3 | 87±18 | 72±39 | 65±46 | 59±42 | 40±28 | |
| ASEntmax (paper) | 100 | 100 | 99.9 | 99.7 | 99.4 | 96.3 | 86.6 | |
| ASEntmax (0907, unchanged) | 100 | 100 | 99 | 100 | 99 | 96 | 76 | s2, 5e-4 |
| stieltjes (0907) | 100 | 100 | 100 | 99 | 100 | 89 | OOM | s1, 5e-4 |
| **stieltjes (sweep)** | 100 | 100 | 100 | 99 | 100 | 87 | 52 | s1, lr=5e-4, acc_@3=1.000, tiebreak val 16x/32x=97/85 |
| stieltjes (sweep, mean±std over 3 seeds @ lr=5e-4) | 100±0 | 72±40 | 67±47 | 66±46 | 61±44 | 44±36 | 18±24 | |

### Mqmtar

| method | ID | 2x | 4x | 16x | 64x | 256x | 1024x | selected run |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Softmax (paper) | 100 | 100 | 100 | 99.5 | 97.8 | 80.2 | 3.0 | best of 3 seeds x LRs, 1K samples |
| softmax (0907) | 100 | 100 | 97 | 90 | 48 | 0 | skip | s1, 1e-4 |
| **softmax (sweep)** | 100 | 100 | 100 | 100 | 93 | 42 | 0 | s3, lr=2e-4, acc_@3=1.000, tiebreak val 16x/64x=97/94 |
| softmax (sweep, mean±std over 3 seeds @ lr=2e-4) | 100±0 | 100±0 | 98±3 | 96±4 | 66±23 | 15±19 | 0±0 | |
| ASEntmax (paper) | 100 | 100 | 100 | 99.7 | 99.6 | 99.0 | 95.3 | |
| ASEntmax (0907, unchanged) | 63 | 37 | 11 | 0 | skip | skip | skip | s1, 2e-4 (3/4 runs never left plateau) |
| stieltjes (0907) | 100 | 100 | 100 | 100 | 86 | - | - | s2, 2e-4 |
| **stieltjes (sweep)** | 100 | 100 | 97 | 99 | 94 | 63 | 12 | s3, lr=4e-4, acc_@3=1.000, tiebreak val 16x/64x=98/91 |
| stieltjes (sweep, mean±std over 3 seeds @ lr=4e-4) | 100±0 | 99±1 | 98±1 | 99±0 | 89±5 | 48±15 | 7±5 | |

## 3. LR sensitivity (the PI's question)

Per (task, method, LR): number of seeds, the validation selection value (max and mean over seeds; sort = BLEU@4x, others = exact-match acc @8x; `deg` = primary monitor identically 0 for every seed, value shown is the BLEU@4x fallback), the tie-break validation ladder where it exists, and the test accuracy at the informative lengths as per-seed values (seed order 1 / 2 / 3). New = added by this sweep; the 0907 grid is the rest. `plateau` = never left the MQMTAR loss plateau; `escape` = step at which train loss first drops below 0.3.

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
| 8e-4 |  | 3 | 0.193 deg | 0.158 | 38 / 67 / 31 | 0 / 0 / 0 |  |
| 1.6e-3 | new | 3 | 0.295 deg | 0.225 | 1 / 48 / 22 | 0 / 0 / 0 | **selected** |
| 3.2e-3 | new | 2 | 0.190 deg | 0.169 | 0 / 0 | skip / skip |  |

### reverse / stieltjes

| lr | new | seeds | val sel max | val sel mean | test 1.5x (per seed) | test 2x (per seed) | notes |
|---|---|---:|---:|---:|---|---|---|
| 2e-4 |  | 2 | 0.154 deg | 0.152 | 0 / 0 | skip / skip |  |
| 4e-4 |  | 2 | 0.255 deg | 0.231 | 18 / 2 | 0 / 0 |  |
| 8e-4 |  | 2 | 0.380 deg | 0.340 | 19 / 0 | 0 / skip |  |
| 1.6e-3 | new | 3 | 0.384 deg | 0.319 | 95 / 74 / 77 | 0 / 0 / 0 |  |
| 3.2e-3 | new | 3 | 0.422 deg | 0.267 | 100 / 38 / 95 | 38 / 0 / 0 | **selected** |
| 6.4e-3 | new | 2 | 0.353 deg | 0.299 | 96 / 99 | 39 / 85 |  |

### copy / softmax

| lr | new | seeds | val sel max | val sel mean | tiebreak 16x (per seed) | tiebreak 32x (per seed) | test 16x (per seed) | test 32x (per seed) | test 64x (per seed) | notes |
|---|---|---:|---:|---:|---|---|---|---|---|---|
| 2.5e-4 | new | 2 | 0.196 deg | 0.185 | - / - | - / - | skip / skip | skip / skip | skip / skip |  |
| 5e-4 |  | 2 | 0.600 | 0.345 | 11 / 0 | 0 / skip | 12 / 0 | 0 / skip | skip / skip |  |
| 1e-3 |  | 3 | 1.000 | 0.727 | 98 / 0 / 96 | 88 / skip / 86 | 98 / 0 / 98 | 91 / skip / 87 | 63 / skip / 57 | **selected** |
| 2e-3 |  | 3 | 1.000 | 0.997 | 95 / 96 / 91 | 81 / 18 / 76 | 95 / 99 / 91 | 82 / 22 / 71 | 55 / 0 / 24 |  |
| 4e-3 | new | 2 | 0.930 | 0.712 | 67 / - | 20 / - | 59 / skip | 22 / skip | 0 / skip | collapsed s2 |

### copy / stieltjes

| lr | new | seeds | val sel max | val sel mean | tiebreak 16x (per seed) | tiebreak 32x (per seed) | test 16x (per seed) | test 32x (per seed) | test 64x (per seed) | notes |
|---|---|---:|---:|---:|---|---|---|---|---|---|
| 2.5e-4 | new | 2 | 0.970 | 0.649 | 40 / - | 0 / - | 47 / skip | 0 / skip | skip / skip | collapsed s2 |
| 5e-4 |  | 3 | 1.000 | 0.757 | 97 / - / 88 | 85 / - / 48 | 100 / skip / 84 | 87 / skip / 45 | 52 / skip / 1 | collapsed s2; **selected** |
| 1e-3 |  | 3 | 1.000 | 0.605 | 98 / - / - | 81 / - / - | 99 / skip / 0 | 92 / skip / skip | 35 / skip / skip | collapsed s2 |
| 2e-3 |  | 2 | 0.980 | 0.805 | 59 / - | 0 / - | 50 / skip | 0 / skip | skip / skip | collapsed s2 |
| 4e-3 | new | 2 | 0.960 | 0.935 | 67 / 60 | 3 / 10 | 68 / 61 | 2 / 8 | 0 / 0 |  |

### mqmtar / softmax

| lr | new | seeds | val sel max | val sel mean | tiebreak 16x (per seed) | tiebreak 64x (per seed) | test 64x (per seed) | test 256x (per seed) | test 1024x (per seed) | notes |
|---|---|---:|---:|---:|---|---|---|---|---|---|
| 5e-5 | new | 2 | 0.069 deg | 0.065 | - / - | - / - | skip / skip | skip / skip | skip / skip | plateau s1,s2 |
| 1e-4 |  | 2 | 1.000 | 1.000 | 90 / 96 | 34 / 61 | 48 / 62 | 0 / 0 | skip / skip | escape s1:16k s2:66k |
| 2e-4 |  | 3 | 1.000 | 0.993 | 83 / 96 / 97 | 44 / 77 / 94 | 36 / 68 / 93 | 2 / 1 / 42 | 0 / 0 / 0 | escape s1:143k s2:68k s3:63k; **selected** |
| 4e-4 | new | 3 | 1.000 | 0.997 | 94 / 99 / 100 | 82 / 81 / 88 | 84 / 92 / 90 | 53 / 40 / 32 | 4 / 1 / 0 | escape s1:32k s2:141k s3:72k |

### mqmtar / stieltjes

| lr | new | seeds | val sel max | val sel mean | tiebreak 16x (per seed) | tiebreak 64x (per seed) | test 64x (per seed) | test 256x (per seed) | test 1024x (per seed) | notes |
|---|---|---:|---:|---:|---|---|---|---|---|---|
| 1e-4 |  | 2 | 0.076 deg | 0.071 | - / - | - / - | skip / skip | - / - | - / - | plateau s1,s2 |
| 2e-4 |  | 3 | 1.000 | 1.000 | 99 / 99 / 95 | 72 / 80 / 72 | 66 / 86 / 66 | 6 / 4 / 0 | 0 / - / skip | escape s1:49k s2:86k s3:136k |
| 4e-4 | new | 3 | 1.000 | 1.000 | 97 / 98 / 98 | 86 / 78 / 91 | 90 / 82 / 94 | 53 / 28 / 63 | 8 / 1 / 12 | escape s1:60k s2:129k s3:86k; **selected** |
| 8e-4 | new | 3 | 1.000 | 0.401 | 99 / - / - | 81 / - / - | 94 / skip / skip | 38 / skip / skip | 0 / skip / skip | plateau s2,s3; escape s1:117k; collapsed s2,s3 |
## 4. Findings

### 4.1 Every LR optimum is bracketed except reverse/stieltjes and sort/softmax at the top of the grid

| task | method | 0907 LR | sweep LR | bracket | what changed |
|---|---|---|---|---|---|
| sort | softmax | 8e-4 (edge) | **6.4e-3** | 3.2e-3 < 6.4e-3 = top of grid (see note) | 2x: 86 -> 99/91 (both seeds); first non-zero 4x for softmax (4, 2); 8x = 0 |
| sort | stieltjes | 4e-4 (flat) | **3.2e-3** | 1.6e-3 and 6.4e-3 both worse on val mean | 2x: 86 -> 88 (best) but 3-seed mean 62±43; 4x = 0 in all 14 runs |
| reverse | softmax | 8e-4 (edge) | 1.6e-3 | 3.2e-3 worse | 1.5x is 0-67 across 14 runs at every LR >= 4e-4 with no trend; 2x = 0 in all. Selected run (s3, val BLEU 0.295) is 22 at 1.5x; its seed-mates are 48 and 1 |
| reverse | stieltjes | 8e-4 (edge) | **3.2e-3** | 1.6e-3 < 3.2e-3 on val max, but 6.4e-3 (val 0.353) gives the best *test* rows: 96/39 and **99/85** at 1.5x/2x | 1.5x: 0 -> 100, 2x: 0 -> 38 (selected) / 85 (6.4e-3 s2, not selected). Bracket open above 6.4e-3 |
| copy | softmax | 1e-3 (tied with 2e-3) | **1e-3** | 5e-4 and 2e-3 lower on tie-break; 2.5e-4 dead by 2x; 4e-3 degrades from 16x | row unchanged (…/98/91/63); seed 3 at 1e-3 = 98/87/57, confirming s1 was not a fluke (s2 collapsed) |
| copy | stieltjes | 5e-4 (edge) | **5e-4** | 2.5e-4 worse (val32x 0), 1e-3 ~ tied (85 vs 81), 2e-3/4e-3 worse | 64x filled: **52**. 1 of 3 seeds is good at 5e-4 (s3: 84/45/1), 1 of 3 at 1e-3 (s3 died by 16x) |
| mqmtar | softmax | 1e-4 (edge) | 2e-4 (s3, by tie-break) | 5e-5 never escapes the plateau; 1e-4 < 2e-4 ~ 4e-4 on val64x; top of grid 4e-4 | **64x: 48 -> 93, 256x: 0 -> 42.** 3-seed means at 64x: 1e-4 55, 2e-4 66±23, 4e-4 89±3 |
| mqmtar | stieltjes | 2e-4 (edge) | **4e-4** | 2e-4 < 4e-4 on val64x (72/80/? vs 86/78/94); 8e-4: 2 of 3 seeds never escape the plateau | 256x/1024x filled: **63/12** (s3), 53/8 (s1), 28/1 (s2). At 2e-4: 6/0, 4/-, 0/skip |

Sort softmax note: 6.4e-3 is the top of the grid and both seeds hit val BLEU@4x = 0.992, so strictly the optimum is not bracketed above. Test is saturating at 2x (99/91), 4x is 4/2 and 8x is 0; a further step was judged unlikely to change a cell. Reverse stieltjes above 6.4e-3 is the one open bracket that could still change a headline number (see 4.3).

### 4.2 The MQMTAR softmax gap to the paper was mostly an LR/seed effect

0907 §3.2 called the softmax MQMTAR row (…/90/48/0 vs paper …/99.5/97.8/80.2) "a systematic gap, not noise". Wrong. At 4e-4 all three seeds give 84-92 at 64x (mean 89±3) and 32-53 at 256x; at 2e-4 seed 3 gives 93/42 and the protocol selects it (highest tie-break val64x, 94). The 64x cell now matches the paper within 1-2 SE; 256x closes half the gap to 80.2 and 1024x stays at 0-4 (paper 3.0). What remains is consistent with best-of-3-seeds x 1K-sample selection. Stieltjes moves the same way: 66/86/66 at 2e-4 -> 90/82/94 at 4e-4, and 256x from 0-6 to 28-63.

The LR effect is about plateau escape (0907 §3.5): mean escape step is 92k at 2e-4 vs 82k at 4e-4 for softmax, but the *variance* is what matters (16k-144k at 1e-4/2e-4). The run that escapes early gets most of the cosine schedule at a useful LR. This is also why the max-over-seeds selection picks 2e-4 s3 (escaped at 64k, val64x 94) over the 4e-4 runs whose 3-seed mean is clearly higher (89 vs 66 at 64x on test). If the paper's "best of 3 seeds" rule is applied literally, 2e-4 is the answer; if the question is "which LR should you use", it is 4e-4.

### 4.3 Reverse: the 0907 Stieltjes row was an artifact, and the paper's fallback selection is noise-limited

Every reverse run has an identically-zero 8x monitor, so per 0907 §4 the fully-trained last.ckpt is laddered and runs are ranked by val BLEU@4x. With the 0907 grid the best Stieltjes reverse number was 19 at 1.5x. At 3.2e-3, s1 gives **100 at 1.5x and 38 at 2x**; at 6.4e-3 both seeds are better still on test (96/39 and 99/85) but rank lower on val BLEU@4x (0.353/0.244 vs 0.422), so 3.2e-3 s1 is the reported row. Either way the 0907 finding "Reverse: Stieltjes is the worst of the three" was an LR-range artifact; at 4-8x that LR it is the best non-entmax reverse row by a wide margin (softmax best: 67 at 1.5x, 0 at 2x in all 14 runs), though still far from ASEntmax (100/100/53/0 at 1.5x/2x/4x/8x).

The fallback is the weak point. BLEU@4x is measured on sequences where every run has 0% exact match, so its dynamic range is ~0.1 and it does not track 1.5x/2x accuracy: softmax 1.6e-3 s3 (BLEU 0.295, test 22) is ranked above 8e-4 s2 (0.142, test 67); stieltjes 6.4e-3 s2 (0.244, test 99/85) ranks 10th of 14. With 14-15 runs per cell, max-over-runs of a proxy this weak selects noise. Recommendation in §7.

### 4.4 Sort: softmax at 6.4e-3 reaches 2x and touches 4x

At 6.4e-3 both softmax seeds reach 99/91 at 2x and 4/2 at 4x (val BLEU@4x 0.992, val exact-match@4x 3-9%). That is 8x the LR of the 0907 selection. The qualitative claim survives (ASEntmax 79.7-81 at 4x vs <=4 for softmax; 8x = 0 for both), but the paper's 0 at 2x for softmax is an LR statement, not a mechanism statement. Stieltjes on sort is the most seed-unstable cell in the table (3.2e-3: 2/88/96 at 2x across seeds; 6.4e-3: 3/51); its best run is 10 pts under the best softmax run at 2x and nothing gets past 0 at 4x.

### 4.5 Copy: the LR was already right; Stieltjes seed instability is the finding

Softmax: 2.5e-4 dead by 2x (both seeds), 4e-3 degrades from 16x on (59/22/0), so 1e-3..2e-3 is the optimum, tie-break puts 1e-3 first, and seed 3 at 1e-3 (98/87/57) confirms the selected s1 row (98/91/63) is representative of a trained seed; s2 collapsed. The 22-pt gap to the paper at 64x is 2 SE + best-of-3 vs best-of-2-good-seeds. Stieltjes: at 5e-4 the three seeds are 87/52, collapsed, 45/1 at 32x/64x; at 1e-3 they are 92/35, collapsed, dead by 16x. Seed 2 collapsed at every LR from 2.5e-4 to 2e-3 but trained normally at 4e-3 (61/8/0), so the collapse is an optimization basin at moderate LR for some inits, not a data or kernel problem. Best-run Stieltjes is ~1-2 SE under softmax/ASEntmax at 64x; mean-over-seeds is far under both.

### 4.6 Stieltjes vs softmax after the sweep

At the selected runs: sort 2x 88 vs 99 (softmax; both 0-4 at 4x); reverse 100/38 vs 22/0 at 1.5x/2x (Stieltjes, and 99/85 in the unselected 6.4e-3 run); copy 87/52 vs 91/63 at 32x/64x (softmax by 1-2 SE); mqmtar 94/63/12 vs 93/42/0 at 64x/256x/1024x (tied at 64x, Stieltjes ahead at 256x by ~3 SE, and the 3-seed 4e-4 means agree: 89±5 / 48±15 vs 89±3 / 42±9). Net: a dense polynomial-tail map at q=4 tracks softmax on sort and copy, is clearly better on reverse and modestly better on the MQMTAR long tail, and is the least seed-stable of the three methods everywhere. It does not approach ASEntmax on reverse (38 vs 53 at 4x is not the comparison; ASEntmax is 100 at 2x where Stieltjes is 38-85) or on the paper's MQMTAR 256x/1024x (99.0/95.3).

## 5. Scoring the Sep 13 predictions

| prediction | outcome |
|---|---|
| Sort: softmax 1.6e-3 >= 8e-4 on BLEU@4x, 3.2e-3 diverges or worse; nothing > 0 at 4x (85%) | half right: no divergence at any LR up to 6.4e-3, which gave 4/2 at 4x. Headline cells effectively unchanged |
| Reverse: softmax 1.5x rises to 50-70 at 1.6e-3, 2x = 0; stieltjes <= 30 at 1.5x (75%) | wrong on both: softmax 1.6e-3 = 1/48/22, no trend; stieltjes reaches 100 at 1.5x and 38-85 at 2x |
| Copy: softmax 4e-3 < 2e-3; stieltjes 2.5e-4 not better than 5e-4; s2 collapses again 50% | all right (s2 collapsed at 2.5e-4..2e-3, trained at 4e-3) |
| Copy stieltjes 64x from Triton re-ladder 55-65 | 52 (5e-4), 35 (1e-3); under the range |
| MQMTAR softmax 64x stays 40-70 at every LR, "not an LR effect" (75%) | wrong: 84-93 at 4e-4 and 2e-4 s3 |
| MQMTAR stieltjes 4e-4 >= 80 at 64x; 8e-4 ~40% NaN risk; 256x on 2e-4 ckpts 30-70 | 90/82/94 right; 8e-4 never NaN'd but 2 of 3 seeds never left the plateau (equivalent outcome); 256x at 2e-4 was 0-6, far below the range; 28-63 at 4e-4 |
| ~20% that any qualitative Table-1 conclusion changes | two changed: the MQMTAR softmax gap is mostly LR/seed; reverse Stieltjes is the best non-entmax row, not the worst |

## 6. Selection protocol as implemented (delta vs 0907 §4)

Unchanged: primary monitor, BLEU@4x fallback, last.ckpt when degenerate, validation only. Added: (i) among runs tied on the primary at 1.0 (copy, mqmtar), rank by a tie-break validation ladder (val32x then val16x for copy; val64x then val16x for mqmtar) on splits drawn with RNG seed 4243 (`gen_tiebreak_val.sh`; files `test_1<i>_val<len>.{src,trg}`; evaluated with `TIEBREAK=1 run_one.sh` into `ladder_tiebreak.tsv`); (ii) mean-over-seeds of the selection value is reported next to the max. `experiments/osc/dump_runs.py` dumps every run to JSON; §2-3 and the appendix are generated from that dump.

## 7. Dropped, and next steps

- Dropped: mqmtar stieltjes s2 2e-4 at 1024x. The 65k-token prefill at batch 1 on 40 GB timed out at 8 h and again at 12 h with the Triton path; the same rung took 2.5 h on the 4e-4 checkpoints, so this one checkpoint hits a slow path I did not chase. The cell is shown as `-`; every other cell in the table is populated.
- Reverse selection needs a better fallback than BLEU@4x: val exact-match at 1.5x/2x exists in the val loaders and would rank the 6.4e-3 stieltjes runs (99/85 on test) first instead of 10th. Cheap (a config change + re-selection, no training). Doing this before quoting the reverse rows externally is advisable.
- ASEntmax at the swept LRs is the obvious next spend: it is the only row not swept, its 0907 MQMTAR row is a training failure (3 of 4 runs on the plateau), and the sweep shows the plateau is LR-sensitive. mqmtar {2e-4, 4e-4} x 3 seeds + reverse {8e-4, 1.6e-3} x 2 seeds, ~14 jobs, ~60 GPU-h. A 20k-warmup variant (0907 §6.1) can ride in the same array.
- Reverse stieltjes above 6.4e-3 (1.28e-2, 2 seeds, ~9 GPU-h) to close the one open bracket.
- Re-evaluate the selected checkpoints on 1K samples/length to take the ±4 pt SE off the 32x-256x cells (eval only, ~10 GPU-h).
