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

## 3. Stage 3, T=2048: softmax done; ASEntmax and q3/d2 resuming

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
| 3 | T=2048 rows within 0.01 at val@2048 | pending |
| 4 | T=2048: windowed still rises least on ArXiv but gap < 0.15 | pending; softmax's T=2048 rise is already only +0.06..0.11 |
| 5 | T=2048 softmax beats T=512 softmax at 16k ArXiv by > 0.2 | right (0.30) |
| 6 | q3/d2 3-seed mean within 0.05 of q4/d2, rise < 0.12 all seeds | half (0.033 yes; one seed rises +0.24) |

## 5. Updated picture
1. At the training length no map beats softmax (T=512: 9 seeds across 3 rows within 0.007; T=2048 pending).
2. Windowed Stieltjes (q 3-4, d 1.5-2) extrapolates to 32x with the smallest loss rise on FineWeb and ArXiv,
   and is tied with softmax / ASEntmax on PG-19 prose. The ArXiv advantage (0.29 nats, 6 vs 3 seeds) is real but
   dataset-specific.
3. Training softmax at 2048 instead of 512 buys more ArXiv robustness (0.30) than switching the attention map
   does at 512. Whether the map still adds anything on top of T=2048 training is the stage-3 question.
4. Dense Stieltjes, q=2, d>=3 and plain entmax degrade on every long-document set.
