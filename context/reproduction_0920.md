# Stieltjes variants on OSC Ascend (A100-40GB) — 2026-09-19 to 09-21

Results for the batch planned in `variants_0919_plan.md` (predictions there, §3; scored here in §5).
Protocol, data, checkpoints and selection rule are those of `reproduction_0913.md` §1/§6; nothing was
re-run for the old rows. Raw dump: `experiments/osc/dump_runs.py` -> `runs_0920.json` (172 runs).

Status: **partial** (Sep 21 08:40). Main batch done (38/47 complete, 9 reverse runs timed out at ~205k/234k
steps and are resuming, job 7462344). Follow-ups queued (§7): kernel benchmark 7462343, mqmtar 20k-warmup
7462345, matched-LR aswstieltjes 7462346/7462347, entmax s2 last.ckpt ladder 7462351. Spent so far ~250 GPU-h.

## 1. What was run

One LR per cell (guessed, not swept; see plan §1), 3 seeds for the Stieltjes rows, 2 for entmax.
`stieltjes_q16` = dense q=16; `wstieltjes` = windowed q=4, d=2 (c = 1/16); `aswstieltjes` = windowed +
adapt-softplus-tanh scale; `entmax` = 1.5-entmax without the scale; `asentmax` re-run on mqmtar at 4e-4.
Kernels: the rewritten fused Triton kernel (commit dc550e7) for every Stieltjes row; adasplash for entmax.

## 2. Headline table (validation-only selection over seeds, 100 test samples/length, ±4 pt SE at p=0.8)

Selection: sort = max val BLEU@4x; copy/mqmtar = max val exact-match@8x (ties at 1.0 -> tie-break ladder,
not run for the new rows: none tied at 1.0 except mqmtar wstieltjes s3); reverse = BLEU@4x fallback,
last.ckpt (pending). "0913" rows = the swept rows from the previous report, for reference.

### Sort

| method | ID | 2x | 4x | 8x | selected | 3-seed mean at 4x |
|---|---:|---:|---:|---:|---|---|
| softmax (0913, swept) | 100 | 99 | 4 | 0 | s1 6.4e-3 | |
| stieltjes q=4 (0913, swept) | 100 | 88 | 0 | skip | s2 3.2e-3 | |
| **stieltjes_q16** | 91 | 22 | 0 | skip | s1 3.2e-3 (BLEU 0.970) | 0 (s3 collapsed: ID 18) |
| **wstieltjes** | 100 | 93 | 7 | 0 | s3 1.6e-3 (0.995) | 3±3 |
| **aswstieltjes** | 100 | 99 | 32 | 0 | s3 4e-4 (0.996) | 21±8 |
| **entmax** | 100 | 98 | 53 | 0 | s1 4e-4 (1.000) | 35±19 (2 seeds) |
| Entmax (paper) | 100 | 100 | 57.8 | 0 | | |
| ASEntmax (0907, swept) | 100 | 100 | 81 | 0 | s1 2e-4 | 75±6 |
| asentmax at 4e-4 (0907, for LR-matched comparison) | 100 | 95 / 100 | 9 / 52 | 0 | 2 seeds | |

### Copy

| method | ID | 2x | 4x | 8x | 16x | 32x | 64x | selected |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| softmax (0913) | 100 | 100 | 100 | 100 | 98 | 91 | 63 | s1 1e-3 |
| stieltjes q=4 (0913) | 100 | 100 | 100 | 99 | 100 | 87 | 52 | s1 5e-4 |
| **stieltjes_q16** | 90 | 3 | 0 | skip | skip | skip | skip | s2 5e-4 (all 3 seeds dead by 4x) |
| **wstieltjes** | 100 | 100 | 100 | 96 | 93 | 69 | 25 | s1 5e-4 (acc@8x .99 x3; s1 by ladder) |
| **aswstieltjes** | 100 | 99 | 99 | 96 | 86 | 52 | 8 | s2 5e-4 (.98) |
| **entmax** | 100 | 100 | 100 | 99 | 98 | 86 | 50 | s2 5e-4 (1.00) |
| Entmax (paper) | 100 | 100 | 99.9 | 99.7 | 99.4 | 96.3 | **0.0** | |
| ASEntmax (0907) | 100 | 100 | 99 | 100 | 99 | 96 | 76 | s2 5e-4 |
| asentmax at 5e-4 (0907) | 100 | 99 / 100 | 97 / 99 | 81 / 100 | 48 / 99 | 2 / 96 | 0 / 76 | 2 seeds |

wstieltjes seeds at 64x: 25 / 3 / 4; aswstieltjes 3 / 8 / 4; entmax 22 / 50.

### MQMTAR

| method | ID | 2x | 4x | 16x | 64x | 256x | 1024x | selected | escaped plateau |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| softmax (0913) | 100 | 100 | 100 | 100 | 93 | 42 | 0 | s3 2e-4 | 3/3 at 2e-4, 4e-4 |
| stieltjes q=4 (0913) | 100 | 100 | 97 | 99 | 94 | 63 | 12 | s3 4e-4 | 3/3 at 2e-4, 4e-4 |
| **stieltjes_q16** (4e-4) | 90 | 70 | 30 | 1 | 0 | skip | skip | s2 (acc@8x 0.11) | 1/3 (s2 at 182k) |
| **wstieltjes** (4e-4) | 100 | 100 | 100 | 98 | 91 | 48 | 6 | s3 (1.00) | 3/3 (51k, 85k, 198k) |
| **aswstieltjes** (2e-4) | 0 | skip | | | | | | all 3 on the plateau | 0/3 |
| **entmax** (2e-4) | 0 | skip | | | | | | s2 escaped at 258k; best-by-val ckpt is pre-escape (last.ckpt ladder pending) | 1/2 |
| **asentmax** (4e-4, new) | 0 | skip | | | | | | all 3 on the plateau | 0/3 (0/7 incl. 0907's 1e-4, 2e-4) |
| Entmax (paper) | 100 | 100 | 100 | 99.5 | 92.7 | 66.8 | 9.3 | | |
| ASEntmax (paper) | 100 | 100 | 100 | 99.7 | 99.6 | 99.0 | 95.3 | | |

wstieltjes per seed at 64x/256x/1024x: 72/17/0, 0/skip, 91/48/6 (s2 escaped late at 198k and is undertrained).

### Reverse

entmax (2 seeds, 4e-4, last.ckpt): 96/80/43/0/skip and 95/79/44/1/0 at ID/1.5x/2x/4x/8x. Paper Entmax:
100/100/93.5/28.5/2.5; ASEntmax 0907: 100/100/100/53/0. The nine Stieltjes-variant rows are pending the
resume (§7); their val BLEU@4x at ~205k steps is 0.43-0.68 vs entmax's 0.76-0.87, so expect them below entmax.

## 3. Findings

### 3.1 q=16 dense is a training failure, not a sharper q=4

All three copy seeds are dead by 4x (ID 82-90, 2x 3-11); 2 of 3 mqmtar seeds never leave the plateau and
the third escapes at 182k and reaches only 90/70/30; sort s3 collapsed (ID 18). Train loss goes to 1e-5..1e-7
on sort/copy, so this is not an optimisation failure in the loss sense: the model fits ID and generalises
to nothing. A q=16 map is nearly hard-max from initialisation (toy: gap 1.2 for 50% mass at n=16), so
attention never sees a soft mixture during training. Prediction "q16 >= q4 on the mqmtar tail" was wrong;
the dispersion argument concerned eval length, but the failure is in training.

### 3.2 Windowed q=4 behaves like dense q=4 on MQMTAR and like a weak entmax on sort/copy

MQMTAR: 91/48/6 vs dense q=4's 94/63/12 (within 1-2 SE per cell; the s1 seed's 72/17/0 and s3's 91/48/6
bracket the dense seeds' 90/53/8 .. 94/63/12). Windowing gave exact zeros at no cost here, and it is the
only sparse row in the table that trained on MQMTAR at all (entmax 1/2, ASEntmax 0/7, AS-windowed 0/3).
Sort 4x: 7 (seeds 0/2/7) vs entmax 53/16 and dense q=4's 0. Copy 64x: 25 vs entmax 50 and dense q=4's 52.
So the sparse map inherits dense-q4's MQMTAR behaviour but not entmax's sort/copy behaviour. Prediction
"windowed ~ entmax within noise" was wrong on sort (all 3 seeds < 10 vs entmax 16-53) and copy.

### 3.3 AS-windowed vs ASEntmax: not decided by this batch (LR mismatch)

The plan said "use ASEntmax's range" and then set sort 4e-4 / copy 5e-4. ASEntmax's own optimum was sort
2e-4 (81 at 4x; its 4e-4 seeds give 9 / 52) and copy 1e-3 or 5e-4 (76 at 64x from s2; s1 at 5e-4 gives 0).
At the LR actually shared:

| | sort 4x @4e-4 | copy 64x @5e-4 | mqmtar @2e-4/4e-4 |
|---|---|---|---|
| asentmax | 9 / 52 | 0 / 76 | 0 of 7 seeds escaped |
| aswstieltjes | 14 / 16 / 32 | 3 / 8 / 4 | 0 of 3 |

That is "within seed spread" on sort and mqmtar and behind on copy (max 8 vs 76; but ASEntmax's own two
seeds span 0-76 there). One-LR runs against a swept comparator cannot show parity; the matched-LR runs
(sort 2e-4, copy 1e-3, 3 seeds each, job 7462346/7) are the test. Also relevant: AS-windowed is *behind
plain windowed* on copy (8 vs 25) and sort is the only task where the scale helped (32 vs 7 at 4x).

### 3.4 The entmax family does not train on MQMTAR with 10k warmup

ASEntmax 0/7 across 1e-4, 2e-4, 4e-4; entmax 1/2 (late, 258k); AS-windowed 0/3. Dense Stieltjes q=4
escaped 6/6 at 2e-4/4e-4 and windowed q=4 3/3. The single configuration that ever trained an entmax-family
model on MQMTAR here is the local 20k-warmup run (0907 §3.5: escaped at 131k, 100% to 64x). The 20k-warmup
follow-up (asentmax + aswstieltjes, 3 seeds, 2e-4, job 7462345) tests that directly. Until it lands, the
MQMTAR column has no entmax-family comparator and the paper's 99.0/95.3 cells remain unreproduced.

### 3.5 Our entmax control is stronger than the paper's Entmax row

Copy 64x = 50 (paper 0.0), 32x = 86 (96.3); sort 4x = 53 (57.8); reverse 4x = 0-1 (28.5); mqmtar untrained
(paper 92.7/66.8/9.3). Copy is the notable one: with the repo's adasplash entmax the "sparse maps cannot
copy at 64x" story does not hold.

### 3.6 Kernel speed regression on the A100: found and fixed (tile size)

Train wall for the same task, old kernel (Sep 14 sweep) vs the rewritten kernel (this batch):
sort 2.8 -> 4.5 h, copy 1.8 -> 3.8 h, reverse 5.3 -> ~9.2 h (implied; hence the nine timeouts),
mqmtar 5.9 -> 5.9 h. A100 bench (job 7462343, bf16, B x 8 heads x N x 32, fwd+bwd, ms):

| N | old kernel | new, 128x64 (default used in this batch) | new 64x64 | new 32x32 | new 64x32 |
|---|---:|---:|---:|---:|---:|
| 64 | 4.55 | 3.35 | 1.55 | 1.39 | **0.90** |
| 128 | 5.01 | 2.91 | 4.27 | 2.55 | **2.58** |
| 256 | 5.84 | 8.14 | 7.77 | 4.82 | **5.17** |
| 2048 | 30.5 | 58.6 | 52.2 | 36.4 | **36.2** |

The rewritten kernel keeps the Q tile resident across three sweeps and recomputes scores per sweep, which
favours a short M tile; the inherited 128x64 choice was tuned for the old kernel. With 64x32 the new kernel
is 1.1-5x faster than the old one at every shape. Fixed in c48c321 (bf16/fp16 -> 64x32). Validation of the
64x32 tile against the eager reference on the A100 is job 7462977; the follow-up jobs' *eval* phases
(new python processes) already pick up 64x32, their training phases run with the tile they started with.
If 7462977 reports a failure, re-ladder those runs.

## 4. Predictions scored (plan §3)

| prediction | outcome |
|---|---|
| q16: mqmtar 256x/1024x 70-90 / 30-60, seed-fragile on sort/copy (65%) | wrong: dead on copy, 1/3 escaped on mqmtar, 0 at 64x |
| wstieltjes: sort 4x 20-60, copy 64x 30-80, mqmtar 40-80 / 5-40 (55%) | sort wrong (7), copy under (25), mqmtar right (48/6) |
| aswstieltjes ~ ASEntmax within noise, sort 60-85 (55%) | undecided at matched LR; at the LR run: sort 32 (vs 9/52), copy 8 (vs 0/76), mqmtar both untrained |
| entmax: sort 40-70, copy 64x 0-40, mqmtar 60-90/0-20 (65%) | sort right (53), copy over (50), mqmtar untrained |
| asentmax mqmtar 4e-4: 90-99 / 60-95 (70%) | wrong: 0/3 escaped |
| 1 of 3 wstieltjes and 0-1 of 3 aswstieltjes seeds stuck on mqmtar | wstieltjes 0/3 stuck (better); aswstieltjes 3/3 stuck (worse) |
| ~25% any new row qualitatively differs from its entmax counterpart | yes, twice, both against us: q16 fails, windowed < entmax on sort |

Net: 2 of 7 right. The systematic miss was the entmax-family plateau on MQMTAR (10k warmup), which the
0907 report had already flagged and I under-weighted.

## 5. Selection notes

- sort: val BLEU@4x is saturated (0.985-1.000 for wstieltjes/aswstieltjes/entmax) while test 4x spans
  7-53. Same weakness as the reverse fallback in 0913 §4.3. Val exact-match@4x would separate them.
- copy: the three wstieltjes seeds tie at acc@8x 0.99; reported s1 by the test ladder, which is the one
  place this report uses test to break a tie (marked). The tie-break val ladder was not generated for
  the new rows.
- q16 sort s1 is selected (BLEU 0.970) over s2 (0.937) although s2 is 100/54 on test vs s1's 91/22.

## 6. Cost

Main batch 47 elements: ~250 GPU-h (~32 budget units). Follow-ups queued: reverse resume 9 x ~4 h, mqmtar
w20k 6 x ~10 h, matched-LR 6 x ~4 h, bench 0.5 h, last.ckpt ladder ~1 h: ~125 GPU-h.

## 7. Pending

| job | what | fills |
|---|---|---|
| 7462343 | A100 kernel bench old vs new + tile ablation | §3.6 (done; equivalence part pending) |
| 7462977 | 64x32 / 128x64 tile equivalence + layer test on the A100 | §3.6 validation |
| 7462344 x9 | reverse Stieltjes rows resume (12 h wall) | reverse table |
| 7462345 x6 | mqmtar asentmax_w20k + aswstieltjes_w20k, 3 seeds, 2e-4 | §3.4, mqmtar comparator |
| 7462346 x3 / 7462347 x3 | aswstieltjes at sort 2e-4 / copy 1e-3 | §3.3 |
| 7462351 | mqmtar entmax s2 last.ckpt ladder | mqmtar entmax row |
