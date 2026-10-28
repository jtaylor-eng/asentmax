# LM stages 0-2 results (Oct 28): diagnostics, seeds, AS disentangle, q x d grid

**Complete as of Oct 28 12:00** (retry array 8156274 finished, 17/17). Sections 1-7 below were written on the partial
results; §8 adds the retry and is the summary. Figure: `nanoGPT/results/lm1012/stages_1028.png`.

Plan: lm_stages_1027_plan.md. Prior results: reproduction_lm_1018.md (§4g: windowed Stieltjes q4/d2 0.34 nats
better than softmax at 16k on ArXiv papers, 1 seed). Code nanoGPT `stieltjes_lm` @ 7ab1523.

Status: diag job 8142593 done (except the q/d gradient at 8192, OOM, and the dense row, a bug). Stage-1 array
8142594: 6 of 15 completed (softmax/ASEntmax seeds and 6e-3); the 9 Stieltjes runs and all 8 stage-2 runs died
at launch on a row-parser bug (`--stieltjes_q=3` fails nanoGPT's configurator type assert; the old parser
emitted `4.0`). Fixed; retry array 8156274_[0-16] + diag 8156275 submitted Oct 28 07:50.

## 1. Seeds: the long-context numbers are noisy, the train-length numbers are not

| row | seed | val@512 (FineWeb) | ArXiv 2k | ArXiv 16k | FineWeb 16k |
|---|---|---:|---:|---:|---:|
| softmax_xs @3e-3 | 1337 | 3.7616 | 4.499 | 4.829 | 3.862 |
| | 1 | 3.7562 | 4.383 | 4.563 | 3.848 |
| | 2 | 3.7620 | 4.436 | 4.851 | 4.008 |
| asentmax_zi_xs @1.5e-3 | 1337 | 3.7616 | 4.475 | 4.881 | 3.906 |
| | 1 | 3.7565 | 4.408 | 4.652 | 3.890 |
| | 2 | 3.7644 | 4.407 | 4.527 | 3.913 |
| wstieltjes_q4_d2_xs @1.5e-3 | 1337 | 3.7632 | 4.417 | 4.494 | 3.842 |
| | 1, 2 | pending | | | |

- val@512 seed spread: 0.006 (softmax), 0.008 (ASEntmax). All between-row gaps at 512 in the earlier reports
  (0.000-0.008) are inside this. Prediction 2 right.
- ArXiv 16k seed spread: 0.29 (softmax), 0.35 (ASEntmax). The earlier 0.34 margin for windowed Stieltjes was a
  single-seed number against a single-seed softmax; softmax's best seed (4.563) is 0.07 from windowed's one
  seed (4.494). The windowed seeds decide whether anything survives (prediction 1, 75%, now looks likely wrong
  as stated; the retry has them).
- Across LRs the dense maps' 16k ArXiv loss also swings widely (softmax 5.43 / 4.73 / 4.83 / 4.52 at 7.5e-4 ..
  6e-3; ASEntmax 4.85 / 4.88 / 4.44 / 4.75) while windowed's three LRs sit at 4.69 / 4.49 / 4.33. If the
  windowed seeds also cluster, the defensible claim becomes "windowed Stieltjes extrapolates with far less
  seed/LR variance", not "0.34 nats better".

## 2. LR 6e-3 bracket
softmax_xs 3.7772, asentmax_zi_xs 3.7818 at 512: both worse than 3e-3 (3.7616 / 3.7675), so the selected LRs
stand. Prediction 5 right for these two (windowed 6e-3 pending). Note softmax @6e-3 has its best ArXiv 16k
(4.517): long-context behaviour is LR-sensitive for softmax in a way val@512 does not see.

## 3. Stage 0(a): inference-time q and d on the trained windowed q4/d2 model (64 papers, same papers for all rows)

| lr | setting | 512 | 2048 | 4096 | 8192 | 16384 |
|---|---|---:|---:|---:|---:|---:|
| 1.5e-3 | trained q4 d2 | 4.663 | 4.377 | 4.414 | 4.439 | 4.459 |
| | d 1.5 | 4.849 | 4.569 | 4.604 | 4.633 | 4.654 |
| | d 3 | 5.112 | 4.904 | 4.978 | 5.022 | 5.061 |
| | dense | 6.110 | 6.349 | 6.627 | 6.789 | 6.872 |
| | q 3 | 4.667 | 4.393 | 4.442 | 4.474 | 4.496 |
| | q 6 | 4.704 | 4.393 | 4.404 | 4.404 | 4.413 |
| | q 8 | 4.803 | 4.488 | 4.497 | 4.472 | 4.453 |
| 3e-3 | trained q4 d2 | 4.675 | 4.342 | 4.351 | 4.328 | 4.296 |
| | d 1.5 | 4.845 | 4.530 | 4.559 | 4.545 | 4.518 |
| | q 6 | 4.706 | 4.349 | 4.358 | 4.340 | 4.309 |

- d: the trained value is a sharp optimum in both directions (d 1.5: +0.19; d 3: +0.5; dense: +1.5 and growing
  with length). The model co-adapted to its window; d is not a free inference-time knob. Prediction 8 right.
- q: raising q to 6 at inference costs 0.04 at 512 and gains up to 0.046 at 16k (lr 1.5e-3), ~0 at 3e-3. A
  lighter tail helps slightly at long range; q=3 (heavier) hurts slightly. Small effects either way.

## 4. Stage 0(b): dL/dq, dL/dd at L=2048 (8 papers; finite differences at +-0.25 d, +-0.5 q)
- wstieltjes_q4_d2 (both LRs): loss rises for d+0.25 AND d-0.25 (4.497 / 4.469 vs 4.445): at a local optimum in
  d; q+-0.5 changes loss by < 0.005. Consistent with §3.
- aswstieltjes_q2_d4_zi (both LRs): dL/dq = -0.08 to -0.09, FD q+0.5 -> -0.013/-0.023, q-0.5 -> +0.055: this
  model wants a LARGER q, i.e. a lighter tail; dL/dd = -0.04, d+0.25 helps slightly. So the AS-windowed row's
  trouble may be q=2 rather than the scale. The retry's q2/d4-without-AS and q4/d2+AS rows separate the two.
- Gradient probe at 8192 OOMs (fp32 per-head maps under autograd); rerun at 4096.

## 5. Stage 0(c): NoPE heads capped to the last 2048 keys (same 32 papers for all rows; cap is a no-op at L=2048)

| row | 2048 (= uncapped baseline, 32 papers) | 4096 capped | 8192 capped | rise 2k -> 8k capped | rise 2k -> 8k uncapped (128 papers) |
|---|---:|---:|---:|---:|---:|
| softmax | 4.314 | 4.521 | 4.713 | +0.40 | +0.25 |
| entmax | 4.319 | 4.511 | 4.633 | +0.31 | +0.28 |
| ASEntmax | 4.318 | 4.474 | 4.560 | +0.24 | +0.31 |
| windowed Stieltjes | 4.285 | 4.366 | 4.407 | +0.12 | +0.06 |

- Capping does not make softmax flat: its loss still rises with position under the cap (per-position buckets
  at 8192: 4.07 -> 4.93), and the cap makes the 2k -> 8k rise worse (+0.40 vs +0.25). So the degradation is not
  simply "too many keys"; the paper's observation that NoPE heads carry an early-token bias (they use the first
  tokens as a sink) is the likely reason a hard cap hurts: it removes those tokens. Capped softmax is 0.31
  behind capped windowed Stieltjes at 8192 on the same papers. Prediction 9 (gain is mostly a cap effect) wrong.
  Caveat: the cap design confounds "fewer keys" with "no access to the document start"; a cap that keeps the
  first 64 tokens would be the clean version.

## 6. Predictions scored so far
| # | prediction | status |
|---|---|---|
| 1 | windowed beats softmax by >= 0.2 at 16k ArXiv in all 3 seeds | pending (windowed seeds); softmax's best seed is already within 0.07 |
| 2 | val@512 seed spread <= 0.01 | right (0.006, 0.008) |
| 3, 4 | AS vs q/d disentangle | pending; gradient probe hints q=2 is the problem, not AS |
| 5 | 6e-3 does not win at 512 | right for softmax, ASEntmax; windowed pending |
| 6, 7 | q/d grid | pending |
| 8 | no inference-time d change improves 16k by > 0.05 | right (all d changes hurt; q6 gains 0.046 at one LR) |
| 9 | capped softmax within 0.1 of windowed at 8k | wrong (0.31 behind) |

## 7. Retry (array 8156274): all 17 completed; gradient probe at 4096 still OOMs (not rerun; the 2048 results stand)

## 8. Seeds complete: the long-context gain is real and the mechanism is variance, not just mean

ArXiv papers, selected LR, 3 seeds each (1337, 1, 2):

| row | val@512 (mean, sd) | ArXiv 2k | ArXiv 16k (per seed) | mean | sd | rise 2k -> 16k per seed |
|---|---|---:|---|---:|---:|---|
| softmax @3e-3 | 3.760 (0.003) | 4.44 | 4.83 / 4.56 / 4.85 | 4.748 | 0.160 | +0.33 / +0.18 / +0.42 |
| ASEntmax @1.5e-3 | 3.761 (0.004) | 4.43 | 4.88 / 4.65 / 4.53 | 4.687 | 0.179 | +0.41 / +0.24 / +0.12 |
| windowed q4/d2 @1.5e-3 | 3.763 (0.003) | 4.37 | 4.49 / 4.47 / 4.37 | **4.446** | **0.066** | +0.08 / +0.11 / +0.04 |

- Windowed Stieltjes beats softmax at 16k by 0.30 nats on the seed means (Welch t = 3.0, df 2.7; ~p 0.05
  one-sided) and ASEntmax by 0.24 (t = 2.2, not significant at n = 3). Every windowed seed is below every
  softmax and ASEntmax seed at 16k. Prediction 1 (>= 0.2 in all 3 seeds vs softmax): the per-seed-pair minimum
  gap is 4.563 - 4.494 = 0.07, so wrong as stated, but the mean gap is 0.30.
- The clearer statement is about the rise from 2k to 16k: windowed +0.075 (sd 0.03) vs softmax +0.31 (sd 0.12)
  and ASEntmax +0.26 (sd 0.14). Windowed Stieltjes' extrapolation is both smaller and 4x less variable.
- FineWeb 16k: windowed 3.87 (sd 0.05) vs softmax 3.91 (0.09) vs ASEntmax 3.90 (0.01): same direction, small.
- val@512: identical across rows (3.760 / 3.761 / 3.763, sd 0.003). No map costs anything at the train length.

## 9. AS disentangle (1 seed, 3 LRs): the collapse was q=2, not the scale

ArXiv 16k (and 2k -> 16k rise) by LR:

| row | 7.5e-4 | 1.5e-3 | 3e-3 | NoPE support 512 -> 16k (1.5e-3) |
|---|---|---|---|---|
| q4 d2 | 4.69 (+0.26) | 4.49 (+0.08) | 4.33 (-0.04) | 55 -> 130 |
| q4 d2 + AS | 4.79 (+0.36) | 4.43 (+0.07) | 4.62 (+0.20) | 39 -> 89 |
| q2 d4 | 5.03 (+0.50) | 5.08 (+0.53) | 5.69 (+1.19) | 90 -> 256 |
| q2 d4 + AS | 5.05 (+0.57) | 4.91 (+0.44) | 5.09 (+0.65) | 91 -> 243 |

- q2/d4 without the scale collapses as badly as with it (worse at 3e-3). The scale is not the cause;
  prediction 3 (AS is the culprit, 60%) wrong, prediction 4 (q4/d2+AS degrades like q2/d4+AS, 55%) wrong:
  q4/d2+AS is fine at 1.5e-3 (4.43, the best single number in this table besides q4/d2 @3e-3) and only
  mediocre at 3e-3.
- The AS scale is roughly neutral for windowed Stieltjes: it helps q2/d4 a little, and for q4/d2 it is
  within LR noise. The gradient probe (§4) had this right: the q2/d4 model "wanted" a larger q.
- Mechanism: the q2 heavy tail admits 2-3x more keys into the window (NoPE support 90 -> 256 vs 55 -> 130)
  and the support grows faster with length. Window size, not the scale, sets long-context behaviour.
- 6e-3 for windowed q4/d2: 3.7818 at 512 and 5.23 at 16k ArXiv, clearly too high. Prediction 5 right for all three.

## 10. q x d grid (lr 1.5e-3, 1 seed): tighter window extrapolates better, costs a little at 512

ArXiv 16k loss (FineWeb val@512 in brackets):

| | d = 1.5 | d = 2 | d = 3 |
|---|---|---|---|
| q = 3 | **4.39** (3.780) | 4.37 (3.766) | 5.06 (3.756) |
| q = 4 | 4.44 (3.782) | 4.49 (3.763) | 5.01 (3.755) |
| q = 6 | 4.57 (3.776) | 4.54 (3.764) | 4.60 (3.758) |

NoPE support 512 -> 16k: d 1.5: 14-19 -> 32-48; d 2: 37-89 -> 87-257; d 3: 93-246 -> 272-1571.

- d is the lever. d=3 at q 3-4 collapses like dense (+0.6 over d=2); d=1.5 and d=2 are close. Prediction 6
  (best at d=1.5, 50%): roughly a tie between d 1.5 and 2, so half right.
- q matters less, and lighter is not better: q=6 is worst at d <= 2 (its support grows 3-10x faster than q=3/4
  at the same d, because a lighter tail puts more keys within the threshold). q=3 is best at every d <= 2.
  Prediction 7 (q6 within 0.02 of q4, 60%) wrong: q6 is 0.05-0.13 worse at 16k.
- The 512 cost of a tight window is small but monotone: d 1.5 is +0.017 over d 3, i.e. a real trade-off
  (3-5 seed-sd). q3/d2 (4.37 at 16k, 3.766 at 512) is the best compromise in this grid; q4/d2 is within noise.
- Best 16k numbers in the whole study are q4/d2 @3e-3 (4.33) and q3/d2 @1.5e-3 (4.37); windowed q3-q4, d 1.5-2
  all land in 4.37-4.49, i.e. the result is robust to the exact (q, d).

## 11. Predictions, final score

| # | prediction | result |
|---|---|---|
| 1 | windowed >= 0.2 better than softmax at 16k in all 3 seeds | wrong as stated (min pairwise 0.07); mean gap 0.30 |
| 2 | val@512 seed spread <= 0.01 | right |
| 3 | AS is the culprit for q2/d4's collapse | wrong (q=2 is) |
| 4 | q4/d2+AS degrades like q2/d4+AS | wrong |
| 5 | 6e-3 does not win at 512 | right (all three rows) |
| 6 | best 16k at d=1.5 | half (d 1.5 ~ d 2) |
| 7 | q6 within 0.02 of q4 | wrong (0.05-0.13 worse) |
| 8 | no inference-time d change helps | right |
| 9 | capped softmax within 0.1 of windowed | wrong (0.31 behind) |

3.5 / 9. The misses share a cause: I over-weighted the adaptive scale (the paper's ingredient) and under-weighted
the window itself. The data say: the window d sets everything; q should be 3-4; the scale is neutral.

## 12. What is now established (51M, 1B tokens, T=512, exp-slope NAPE, FineWeb; ArXiv eval is OOD text)

1. No attention map beats softmax at the train length (val@512 within 0.003 across softmax, ASEntmax, windowed
   Stieltjes, 3 seeds each).
2. On long documents, windowed Stieltjes (q 3-4, d 1.5-2) extrapolates with a 2k -> 16k rise of +0.04..+0.11
   (3 seeds + 4 grid cells) vs +0.18..+0.42 for softmax and +0.12..+0.41 for ASEntmax. Mean gap at 16k: 0.30
   nats vs softmax (p ~ 0.05, n = 3), 0.24 vs ASEntmax.
3. The mechanism is bounded support: NoPE-head support grows ~2.4x from 512 to 16k for d <= 2 vs 32x (all
   keys) for softmax and 6.8x for entmax. Capping softmax to 2048 keys does not reproduce it.
4. Dense Stieltjes and d >= 3 collapse; the AS scale is neutral; q = 2 (heavy tail) is bad, q = 6 (light tail)
   is slightly bad, q 3-4 is right.
5. Cost: 1.9x softmax step time at 15 Newton sweeps; no sparsity exploited in the kernel yet.

Open: scale (stage 4, 125M at T=2048), whether the gain survives training at T=2048 (stage 3), and whether
non-LaTeX long documents (PG-19) show the same.
