# LM results Oct 30: PG-19 ladder, q3/d2 + softmax seeds, stage 3 (T=2048) partial

Follows reproduction_lm_1028.md. Jobs: PG-19 eval 8196382 (done, 43 checkpoints), seeds 8196383 (done),
T=2048 array 8196384 (softmax done; ASEntmax and q3/d2 hit the 2.5 h wall guard at steps 7300 / 5500 and are
resuming as 8217752). Code nanoGPT `stieltjes_lm` @ 9b59e89. Data: `data/pg19_long/` (128 x 16385 tokens from
91 PG-19 test books, tokens 2k+ of each, two segments from 37 books).

## 1. PG-19: the long-document gain does NOT transfer to plain prose

PG-19 mean loss over the first L tokens of each segment (512 / 2k / 4k / 8k / 16k), selected LR, 3 seeds:

| row | 512 | 2k | 4k | 8k | 16k (mean, sd over 3 seeds) | rise 2k -> 16k (mean, sd) |
|---|---:|---:|---:|---:|---|---|
| softmax | 4.09 | 4.04 | 4.04 | 4.06 | 4.087 (0.060) | +0.047 (0.031) |
| ASEntmax | 4.07 | 4.02 | 4.02 | 4.05 | 4.079 (0.002) | +0.058 (0.005) |
| windowed q4/d2 | 4.09 | 4.03 | 4.02 | 4.05 | 4.081 (0.022) | +0.051 (0.021) |
| windowed q3/d2 | 4.09 | 4.03 | 4.02 | 4.04 | 4.075 (0.022) | +0.049 (0.014) |
| entmax (1 seed) | 4.08 | 4.04 | 4.06 | 4.11 | 4.184 | +0.147 |
| AS-windowed q2/d4 (1 seed) | 4.08 | 4.05 | 4.09 | 4.15 | 4.207 | +0.157 |
| dense Stieltjes q4 (1 seed) | 4.11 | 4.11 | 4.19 | 4.34 | 4.479 | +0.370 |

- softmax, ASEntmax, windowed q3/q4 d2 are indistinguishable on PG-19 at every length (16k means within 0.012,
  rises +0.05 for all four). Loss by position at 16k is flat for all of them (softmax 4.06 -> 4.06 over 8 buckets).
- Dense Stieltjes, plain entmax and q2/d4 degrade here too, so the ranking of the bad maps is the same; the
  ranking among the good maps is a tie.
- So the ArXiv gap (0.29 nats, now 6 windowed seeds vs 3 softmax: 4.462 sd 0.09 vs 4.748 sd 0.16) is specific
  to ArXiv LaTeX, not a property of "long documents". Prediction 1 (windowed flattest on PG-19, 70%) wrong.
- Why ArXiv and not PG-19: both datasets gain ~0.1 nats per position from 2k vs 512 context, so both carry
  long-range signal. The difference is what happens to softmax's dispersed NoPE heads: on ArXiv (OOD text,
  heavy notation / macro reuse) their dispersion costs 0.3-0.5 at late positions; on prose the local ALiBi
  heads apparently suffice and dispersion is harmless. A plausible reading is that the windowed map protects
  the model when the NoPE heads matter and the text is unlike training data; it does nothing when they don't.
  That is a narrower claim than "better long-context LM" and needs a long-document set the model was NOT
  trained on but that is in-domain prose (e.g. held-out FineWeb long pages) to pin down.
- PG-19 absolute loss ~4.0 (prediction 2 right).

## 2. q3/d2 seeds and the softmax re-seeds (T=512)

| row | val@512 (3 seeds) | ArXiv 16k per seed | mean (sd) | PG-19 16k mean (sd) | FineWeb 16k mean (sd) |
|---|---|---|---|---|---|
| softmax @3e-3 | 3.760 (0.003) | 4.83 / 4.56 / 4.85 | 4.748 (0.160) | 4.087 (0.060) | 3.906 (0.089) |
| q4/d2 @1.5e-3 | 3.763 (0.003) | 4.49 / 4.47 / 4.37 | 4.446 (0.066) | 4.081 (0.022) | 3.872 (0.049) |
| q3/d2 @1.5e-3 | 3.767 (0.003) | 4.38 / 4.62 / 4.44 | 4.479 (0.130) | 4.075 (0.022) | 3.843 (0.022) |

(The "two more seeds of softmax" request re-ran seeds 1 and 2 of softmax @3e-3; those existed from Oct 28 and
the array skipped them in 10 s. The softmax numbers above are the same three seeds as before.)

- q3/d2 seed 1 is an ArXiv outlier (4.62, rise +0.24); its PG-19 and FineWeb numbers are normal. So q3/d2 is
  not better than q4/d2 (mean 4.48 vs 4.45; prediction 6 half right: within 0.05, but not "rise < 0.12 in all
  seeds"). The grid's q3-over-q4 edge was a single-seed effect; call q3 and q4 equivalent.
- With 6 windowed-d2 seeds the ArXiv gap to softmax is 0.29 (pooled sd 0.09 vs softmax 0.16).

## 3a. Stage 3 complete (resume 8217752): at T=2048 the sparse maps keep improving with context; softmax does not

All six runs, 1 seed, 1B tokens, 8 x 2048 x 8 per step. Selection by val@2048 (final.json). Figure `t2048_1030.png`.

| row | lr | val@2048 | FineWeb 2k / 16k | ArXiv 2k / 4k / 8k / 16k | PG-19 2k / 16k |
|---|---|---:|---|---|---|
| softmax | 1.5e-3 | 3.7237 | 3.711 / 3.707 | 4.387 / 4.426 / 4.452 / 4.450 | 4.031 / 4.049 |
| softmax | **3e-3** | **3.7164** | 3.704 / 3.691 | 4.381 / 4.397 / 4.442 / 4.490 | 4.016 / 4.052 |
| ASEntmax | **1.5e-3** | **3.6851** | 3.690 / 3.715 | 4.303 / 4.268 / 4.220 / 4.198 | 4.011 / 3.993 |
| ASEntmax | 3e-3 | 3.7302 | 3.714 / 3.756 | 4.310 / 4.435 / 4.534 / 4.573 | 4.029 / 4.121 |
| windowed q3/d2 | **1.5e-3** | **3.7095** | 3.710 / 3.716 | 4.277 / 4.211 / 4.090 / 3.994 | 4.019 / 3.978 |
| windowed q3/d2 | 3e-3 | 3.7263 | 3.715 / 3.804 | 4.326 / 4.340 / 4.374 / 4.448 | 4.038 / 4.073 |

Selected-LR comparison, with the T=512 3-seed means of the same rows:

| row | val@train length | ArXiv 16k (T=2048 / T=512) | rise 2k->16k | PG-19 16k (T=2048 / T=512) | FineWeb 16k |
|---|---:|---|---:|---|---:|
| softmax | 3.716 | 4.490 / 4.748 | +0.11 | 4.052 / 4.087 | 3.691 |
| ASEntmax | 3.685 | 4.198 / 4.687 | **-0.11** | 3.993 / 4.079 | 3.715 |
| windowed q3/d2 | 3.710 | **3.994** / 4.479 | **-0.28** | **3.978** / 4.075 | 3.716 |

- At the training length (val@2048) ASEntmax 3.685 < windowed 3.710 < softmax 3.716. The ASEntmax-softmax gap
  (0.031) is 10x the T=512 seed sd; this is the first setting where a sparse map beats softmax at the train
  length, and it is the paper's ingredient (entmax + adaptive scale) at the paper's context length. 1 seed,
  2 LRs; the periodic val trajectories (metrics.jsonl) are consistent (ASEntmax 3.664 vs softmax 3.735 at the
  last periodic eval). Prediction 3 (all within 0.01) wrong.
- On ArXiv the T=2048 sparse maps IMPROVE past the training length: windowed 4.28 -> 3.99 from 2k to 16k,
  ASEntmax 4.30 -> 4.20, while softmax rises 4.38 -> 4.49. This is the paper's Table-3 shape ("decreasing
  perplexity to 8x") reproduced for the first time, and it holds for windowed Stieltjes more strongly than for
  ASEntmax. At 16k the windowed row is 0.50 nats below softmax (vs 0.29 at T=512) and 0.20 below ASEntmax.
  Prediction 4 (gap shrinks below 0.15) wrong in the opposite direction: training longer widens it.
- On PG-19 the T=2048 sparse maps also keep improving slightly (windowed 4.02 -> 3.98, ASEntmax 4.01 -> 3.99)
  while softmax turns up after 4k (4.00 -> 4.05); at 16k the gap is 0.07 (windowed) / 0.06 (ASEntmax). Small
  but it is the same sign as ArXiv, so the "ArXiv-only" conclusion of §1 was a T=512 artefact: at T=512 no
  model could use prose context beyond 4k; at T=2048 the sparse ones can.
- FineWeb 16k: all three within 0.025 (3.69-3.72); softmax best. FineWeb windows past ~2k are mostly other
  documents, so this measures tolerance, not use, of long context.
- Mechanism unchanged: NoPE-head support 512 -> 16k is 23 -> 54 for windowed q3/d2, 86 -> 895 for ASEntmax, all
  keys for softmax (entropy 8.17 of 9.57 at 16k).
- LR sensitivity: both sparse rows are much better at 1.5e-3 than 3e-3 (ASEntmax 3e-3 is the worst row at 16k
  ArXiv, 4.57); softmax prefers 3e-3. The selection rule picked the right LR for each, but the grid is 2 points.
- Cost at T=2048: softmax 824 ms/step, ASEntmax 1218 (1.5x), windowed Stieltjes ~1600 (1.9x).

## 3b. Earlier partial note (softmax only)

softmax_xs trained at T=2048 (same 1B tokens, 8 x 2048 x 8 per step), val at the training length:

| lr | val@2048 | FineWeb 512 / 2k / 4k / 8k / 16k | ArXiv 2k / 16k | PG-19 2k / 16k |
|---|---:|---|---|---|
| 1.5e-3 | 3.7237 | 3.808 / 3.711 / 3.698 / 3.700 / 3.707 | 4.387 / 4.450 | 4.031 / 4.049 |
| 3e-3 | 3.7164 | 3.800 / 3.704 / 3.688 / 3.688 / 3.691 | 4.381 / 4.490 | 4.016 / 4.052 |

- Training at 2048 removes most of softmax's extrapolation problem: FineWeb is flat to 16k (8x), ArXiv rises
  +0.06..+0.11 from 2k to 16k (vs +0.31 for the T=512 softmax seeds), PG-19 +0.02..+0.04. Prediction 5
  (T=2048 softmax beats T=512 softmax at 16k ArXiv by > 0.2) right: 4.45 vs 4.75.
- At 16k on ArXiv, T=2048 softmax (4.45 / 4.49) already matches the T=512 windowed Stieltjes seeds (4.37-4.62).
  Unless T=2048 windowed Stieltjes improves on its T=512 self by a similar amount, the ArXiv gap closes at the
  paper's training length. Pending (8217752).
- Step times at T=2048: softmax 824 ms (1.9x its T=512 time), ASEntmax 1218 ms, windowed Stieltjes ~1600 ms
  (resume-loop timing; the 9.3 s/step in the log is the eval-interval average). The 2.5 h wall guard is too
  short for Stieltjes at T=2048; the resume handles it but a 4 h segment would avoid the restart.

## 4. Predictions scored
| # | prediction | result |
|---|---|---|
| 1 | windowed flattest on PG-19, softmax rises > 0.2 | wrong: all good maps tie at +0.05 |
| 2 | PG-19 loss ~4.0 at 2k | right (4.02-4.03) |
| 3 | T=2048 rows within 0.01 at val@2048 | wrong: ASEntmax 0.031 below softmax, windowed 0.007 below |
| 4 | T=2048: windowed still rises least on ArXiv but gap < 0.15 | wrong, other direction: gap grows to 0.50; windowed loss falls 2k -> 16k |
| 5 | T=2048 softmax beats T=512 softmax at 16k ArXiv by > 0.2 | right (0.30) |
| 6 | q3/d2 3-seed mean within 0.05 of q4/d2, rise < 0.12 all seeds | half (0.033 yes; one seed rises +0.24) |

## 5. Updated picture (supersedes the version written before stage 3 finished)
1. T=512: no map beats softmax at the training length (9 seeds across 3 rows within 0.007). T=2048 (1 seed):
   ASEntmax beats softmax by 0.031 and windowed Stieltjes by 0.007 at val@2048.
2. Long documents, T=512 models: windowed Stieltjes has the smallest 2k -> 16k rise on ArXiv (0.29 nats below
   softmax at 16k, 6 vs 3 seeds) and ties softmax / ASEntmax on PG-19.
3. Long documents, T=2048 models: the sparse maps' loss keeps FALLING past the training length (windowed
   4.28 -> 3.99 on ArXiv, ASEntmax 4.30 -> 4.20; both also fall slightly on PG-19) while softmax's rises. At
   16k ArXiv: windowed 3.99, ASEntmax 4.20, softmax 4.49. This is the paper's qualitative claim, reproduced at
   51M with their training length, and windowed Stieltjes does it better than their map.
4. The training length matters more than the map for softmax (T=2048 softmax 4.49 vs T=512 4.75 at 16k ArXiv);
   for the sparse maps it unlocks use of long context rather than just tolerance of it.
5. Dense Stieltjes, q=2, d>=3 and plain entmax degrade on every long-document set at T=512.
6. Still 1 seed at T=2048 and a 2-point LR grid. Seeds for the three T=2048 rows at their selected LRs
   (6 runs, ~16 GPU-h with 4 h segments) are the obvious next step before any of §3 is a claim.
