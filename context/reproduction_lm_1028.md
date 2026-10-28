# LM stages 0-2 results (Oct 28, partial): diagnostics, seeds, LR bracket; retry of 17 failed runs pending

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

## 7. Pending (array 8156274, diag 8156275)
wstieltjes_q4_d2_xs seeds 1, 2 and 6e-3; aswstieltjes_q4_d2_zi_xs and wstieltjes_q2_d4_xs at 3 LRs; stage-2
q{3,4,6} x d{1.5,2,3}; gradient probe at 4096 incl. dense row.
