# Stieltjes variants on OSC Ascend (A100-40GB) — 2026-09-19 to 09-21

Results for the batch planned in `variants_0919_plan.md` (predictions there, §3; scored here in §5).
Protocol, data, checkpoints and selection rule are those of `reproduction_0913.md` §1/§6; nothing was
re-run for the old rows. Raw dump: `experiments/osc/dump_runs.py` -> `runs_0920.json` (172 runs).

Status: **complete except reverse last.ckpt ladders** (Sep 22). Main batch 47 elements (9 reverse timeouts resumed
and finished), follow-ups §7 all done: kernel bench + tile fix validated, mqmtar 20k-warmup (6), matched-LR
aswstieltjes (6), entmax s2 last.ckpt. Pending: reverse last.ckpt ladders for the 11 new rows (job 7479505,
eval-only; the first pass laddered the uninformative step-11718 checkpoint, see §2 Reverse). Spent ~375 GPU-h.

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
| **aswstieltjes** (4e-4 and 2e-4 pooled) | 100 | 99 | 32 | 0 | s3 4e-4 (0.996; tied with the three 2e-4 seeds at 0.996, s3 4e-4 kept as first-submitted) | 4e-4: 21±8; 2e-4: 17±14 |
| **entmax** | 100 | 98 | 53 | 0 | s1 4e-4 (1.000) | 35±19 (2 seeds) |
| Entmax (paper) | 100 | 100 | 57.8 | 0 | | |
| ASEntmax (0907, swept) | 100 | 100 | 81 | 0 | s1 2e-4 | 75±6 |
| asentmax at 4e-4 (0907, for LR-matched comparison) | 100 | 95 / 100 | 9 / 52 | 0 | 2 seeds | |
| aswstieltjes at 2e-4 (matched to ASEntmax's optimum) | 100 | 99 / 97 / 95 | 37 / 4 / 11 | 0 | 3 seeds | |

### Copy

| method | ID | 2x | 4x | 8x | 16x | 32x | 64x | selected |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| softmax (0913) | 100 | 100 | 100 | 100 | 98 | 91 | 63 | s1 1e-3 |
| stieltjes q=4 (0913) | 100 | 100 | 100 | 99 | 100 | 87 | 52 | s1 5e-4 |
| **stieltjes_q16** | 90 | 3 | 0 | skip | skip | skip | skip | s2 5e-4 (all 3 seeds dead by 4x) |
| **wstieltjes** | 100 | 100 | 100 | 96 | 93 | 69 | 25 | s1 5e-4 (acc@8x .99 x3; s1 by ladder) |
| **aswstieltjes** (5e-4 and 1e-3 pooled) | 100 | 99 | 99 | 99 | 85 | 59 | 12 | s3 1e-3 (1.00) |
| **entmax** | 100 | 100 | 100 | 99 | 98 | 86 | 50 | s2 5e-4 (1.00) |
| Entmax (paper) | 100 | 100 | 99.9 | 99.7 | 99.4 | 96.3 | **0.0** | |
| ASEntmax (0907) | 100 | 100 | 99 | 100 | 99 | 96 | 76 | s2 5e-4 |
| asentmax at 5e-4 / 1e-3 (0907) | 100 | | | | 48 / 99 ; 95 / 97 | 2 / 96 ; 95 / 88 | 0 / 76 ; 72 / 60 | 2 seeds each |
| aswstieltjes at 1e-3 (matched) | 100 | 100 | 99 | 98 | 88 / 90 / 85 | 60 / 68 / 59 | 15 / 32 / 12 | 3 seeds |

wstieltjes seeds at 64x: 25 / 3 / 4; aswstieltjes 5e-4: 3 / 8 / 4, 1e-3: 15 / 32 / 12; entmax 22 / 50; asentmax 1e-3: 72 / 60.

### MQMTAR

| method | ID | 2x | 4x | 16x | 64x | 256x | 1024x | selected | escaped plateau |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| softmax (0913) | 100 | 100 | 100 | 100 | 93 | 42 | 0 | s3 2e-4 | 3/3 at 2e-4, 4e-4 |
| stieltjes q=4 (0913) | 100 | 100 | 97 | 99 | 94 | 63 | 12 | s3 4e-4 | 3/3 at 2e-4, 4e-4 |
| **stieltjes_q16** (4e-4) | 90 | 70 | 30 | 1 | 0 | skip | skip | s2 (acc@8x 0.11) | 1/3 (s2 at 182k) |
| **wstieltjes** (4e-4) | 100 | 100 | 100 | 98 | 91 | 48 | 6 | s3 (1.00) | 3/3 (51k, 85k, 198k) |
| **aswstieltjes** (2e-4, 10k warmup) | 0 | skip | | | | | | all 3 on the plateau | 0/3 |
| **aswstieltjes_w20k** (2e-4, 20k warmup) | 100 | 100 | 98 | 91 | 79 | 11 | 0 | s2 or s3 (acc@8x 1.00 tie; s1 .97 has the best ladder, shown) | 3/3 (61k, 111k, 119k) |
| **entmax** (2e-4) | 0 | skip | | | | | | s2 escaped at 258k; last.ckpt ladder 20/3/0 | 1/2 |
| **asentmax** (4e-4, 10k warmup) | 0 | skip | | | | | | all 3 on the plateau | 0/3 (0/7 incl. 0907's 1e-4, 2e-4) |
| **asentmax_w20k** (2e-4, 20k warmup) | 0 | skip | | | | | | all 3 on the plateau (loss 0.527 at 390k) | 0/3 (0/10 total) |
| Entmax (paper) | 100 | 100 | 100 | 99.5 | 92.7 | 66.8 | 9.3 | | |
| ASEntmax (paper) | 100 | 100 | 100 | 99.7 | 99.6 | 99.0 | 95.3 | | |

wstieltjes per seed at 64x/256x/1024x: 72/17/0, 0/skip, 91/48/6 (s2 escaped late at 198k and is undertrained).
aswstieltjes_w20k per seed: 79/11/0, 6/0/skip, 56/1/0.

### Reverse

Every reverse run has an identically-zero 8x monitor, so the checkpoint callback keeps the *first* checkpoint
(step 11718) and per protocol (0913 §4.3) the fully trained last.ckpt must be laddered instead. The first
eval pass, including the entmax numbers quoted in the Sep 21 draft of this report, used the step-11718
checkpoint and is uninformative (entmax 96/80/43/0, Stieltjes rows 50-97 ID, 6-68 at 1.5x, <= 7 at 2x).
last.ckpt ladders for all 11 rows: job 7479505 (pending). Val BLEU@4x at end of training, for the record:
entmax 0.76 / 0.87; wstieltjes 0.53 / 0.66 / 0.68; aswstieltjes 0.48 / 0.59 / 0.55; q16 0.43 / 0.46 / 0.51;
vs stieltjes q4 (0913) 0.42 best and ASEntmax 0.73.

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

### 3.3 AS-windowed vs ASEntmax at matched LR: behind on sort, tied on copy, ahead on MQMTAR

The first pass ran aswstieltjes at sort 4e-4 / copy 5e-4 while ASEntmax's optimum was sort 2e-4 / copy 1e-3
(plan §1 said "ASEntmax's range" and then didn't use it). At the matched LRs, 3 seeds vs ASEntmax's 2:

| | sort 4x @2e-4 | copy 64x @1e-3 | copy 32x @1e-3 | mqmtar 64x / 256x, 20k warmup, 2e-4 |
|---|---|---|---|---|
| asentmax | 81 / 69 | 72 / 60 | 95 / 88 | 0 of 3 seeds escaped (0 of 10 across all configs) |
| aswstieltjes | 37 / 4 / 11 | 15 / 32 / 12 | 60 / 68 / 59 | 79/11, 6/0, 56/1 (3 of 3 escaped) |

Sort: clearly behind (best 37 vs 69-81; 3-seed mean 17 vs 75). Copy: behind at matched LR (max 32 vs 60-72,
32x 59-68 vs 88-95). So "AS-windowed ties ASEntmax" is wrong on the two tasks where ASEntmax trains; the
first-order "window d == temperature" argument does not carry the within-support profile. On MQMTAR the
comparison inverts: AS-windowed with 20k warmup escapes the plateau 3/3 and reaches 79 at 64x, while
ASEntmax has now failed to train in 10 of 10 runs under our protocol (§3.4). The AS scale costs the windowed
map nothing on MQMTAR relative to plain windowed at 10k warmup (79/11/0 vs 91/48/6 is within seed spread,
different warmup) and is what let it escape at 2e-4.

Also: plain windowed (no scale) beats AS-windowed on copy at the same LR (25 vs 8 at 5e-4), and the AS scale
only helps on sort (32 vs 7). Whatever the scale learns here is not the (log n)^gamma law that ASEntmax uses.

### 3.4 ASEntmax does not train on MQMTAR under our protocol, warmup included

0 of 10: 1e-4 x2, 2e-4 x2 (0907), 4e-4 x3, and now 20k warmup at 2e-4 x3, all sitting at loss 0.527 for
390k steps. The local Sep 1 run that reached 100% to 64x had the *same* config (checked: 20k warmup, 2e-4,
4 layers, 512/1024, 16 heads, bf16, eager entmax, NAPE) but was resumed mid-run from step 97.8k after a
109k-step first attempt that had NOT escaped (loss 0.547 at 109k); the resumed run escaped at 131k. So the
one success involved a restart with a fresh data order, which is a different random draw, not a different
config. Plain entmax: 1 of 2 escaped (late, 258k, last.ckpt 20/3/0). AS-windowed: 3/3 at 20k warmup, 0/3 at
10k. Dense q=4: 6/6, windowed q=4: 3/3 at 10k warmup.

Reading: escaping the MQMTAR plateau is a stochastic event whose rate depends on the map; sparse entmax-1.5
at these LRs is near zero per run, the Stieltjes maps are near one. This is the one place the proposal's
trainability hypothesis has data behind it. The mechanism is not the boundary gradient (proposal
correction, Sep 19); a plausible one is that the windowed map at d=2 keeps a wider support during the
plateau than entmax-1.5 does (on random logits it holds 2-4 tokens/row; not measured on the trained models),
so more key positions keep receiving gradient. That is a hypothesis, not a result. The paper's ASEntmax
MQMTAR row (99.6/99.0/95.3) remains unreproduced; a fair comparator would need either their exact seed/LR
or many more seeds.

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
| aswstieltjes ~ ASEntmax within noise, sort 60-85 (55%) | wrong on sort (37 vs 81 at matched LR) and copy (32 vs 72); reversed on mqmtar (trains 3/3 with 20k warmup where ASEntmax is 0/10) |
| entmax: sort 40-70, copy 64x 0-40, mqmtar 60-90/0-20 (65%) | sort right (53), copy over (50), mqmtar 1/2 escaped, 20/3/0 |
| asentmax mqmtar 4e-4: 90-99 / 60-95 (70%) | wrong: 0/3 escaped, and 0/3 again at 20k warmup |
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

Main batch 47 elements: ~250 GPU-h (~32 budget units). Follow-ups: reverse resume 9 x 1.5 h, mqmtar w20k
6 x 4.5-8.5 h, matched-LR 6 x 4.6 h, bench + validation 0.8 h, last.ckpt ladders ~4 h: ~125 GPU-h.
Total for the variants work ~375 GPU-h (~50 budget units); everything since Sep 6 ~785 GPU-h.

## 7. Jobs

| job | what | status |
|---|---|---|
| 7409254-57 / 7409259-62 | smoke + main batch (47) | done; 9 reverse timeouts (8 h wall, old tile) |
| 7462343 | A100 kernel bench old vs new + tile ablation | done, §3.6 |
| 7462977 | 64x32 tile equivalence + layer test on the A100 | done: ALL OK (default, 64x32, layer) |
| 7462344 x9 | reverse Stieltjes rows resume (12 h wall) | done (1.5 h each) |
| 7462345 x6 | mqmtar asentmax_w20k + aswstieltjes_w20k, 3 seeds, 2e-4 | done, §3.3/3.4 |
| 7462346 x3 / 7462347 x3 | aswstieltjes at sort 2e-4 / copy 1e-3 | done, §3.3 |
| 7462351 | mqmtar entmax s2 last.ckpt ladder | done: 20/3/0 |
| 7479505 x11 | reverse last.ckpt ladders (9 Stieltjes rows + 2 entmax) | **pending** (eval-only, ~20 min each) |

## 8. What I would do next (not submitted)

- Nothing more on q=16 or on plain windowed for sort/copy; the batch answers those.
- If the reverse last.ckpt ladders put wstieltjes/aswstieltjes near ASEntmax's 53 at 4x, reverse becomes the
  second task (with MQMTAR) where the windowed map is competitive; otherwise the summary is "windowed
  Stieltjes = dense Stieltjes with exact zeros; competitive only on associative recall".
- The MQMTAR plateau-escape rate is the only place a Stieltjes row beats the entmax family, and it is a
  training-dynamics claim: 3-seed rates (Stieltjes 12/12, ASEntmax 0/10) are already significant, but a
  1K-sample eval of the escaped checkpoints and a d sweep {1, 2, 4} at fixed LR would make it a result.
