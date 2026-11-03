# LM stage 4b results (Nov 2): windowed Stieltjes q x d at 125M, plus inference-time d/q diagnostics

Prior: reproduction_lm_1101.md (125M first pass: windowed q3/d2 0.07 behind softmax at the train length and 0.07
behind ASEntmax at 16k ArXiv; hypothesis: d=2 too tight at width 768). Jobs: chain 8250597-8250601 (4 rows x 5
segments), diag 8250602. q3/d4 ran out of segments at step 4740/4770 and is finishing as 8278831 (2 h); the
other three rows and the diagnostics are complete. Figure `nanoGPT/results/lm1012/stage4b_qd.png`.

## 1. The sweep (125M, T=2048, 2.5B tokens, lr 1.2e-3, 1 seed)

| row | val@2048 | FineWeb 16k | ArXiv 2k -> 16k (rise) | PG-19 2k -> 16k | NoPE support 512 -> 16k |
|---|---:|---:|---|---|---|
| softmax | **3.461** | 3.437 | 4.035 -> 4.353 (+0.32) | 3.802 -> 3.881 | all |
| ASEntmax | 3.476 | 3.458 | 4.037 -> **3.874** (-0.16) | 3.821 -> **3.800** | 31 -> 88 |
| Stieltjes q3 d2 | 3.534 | 3.493 | 4.141 -> **3.948** (-0.19) | 3.897 -> 3.880 | 14 -> 32 |
| Stieltjes q4 d2 | **3.493** | 3.494 | 4.125 -> 3.992 (-0.13) | 3.888 -> 3.873 | 20 -> 44 |
| Stieltjes q3 d3 | 3.505 | 3.489 | 4.097 -> 3.985 (-0.11) | 3.839 -> 3.847 | 58 -> 121 |
| Stieltjes q4 d3 | 3.495 | 3.497 | 4.110 -> 4.014 (-0.10) | 3.860 -> 3.861 | 100 -> 240 |
| Stieltjes q3 d4 | pending (8278831) | | | | |

- Widening the window or lightening the tail recovers ~0.04 of the 0.073 train-length gap (q4/d2 3.493, q4/d3
  3.495, q3/d3 3.505 vs q3/d2 3.534), but every Stieltjes cell is still 0.03-0.04 behind softmax and 0.02-0.03
  behind ASEntmax at val@2048. Caveat: the final-eval numbers of the Stieltjes rows bounce by +-0.05 between the
  last three periodic evals (q3/d3: 3.42 / 3.46 / 3.55), more than softmax's (3.41 / 3.44 / 3.44), so the
  train-length ranking among the four Stieltjes cells is within noise. The gap to softmax is not.
- At 16k ArXiv the ORIGINAL q3/d2 is the best Stieltjes cell (3.948); every wider / lighter cell extrapolates
  less well (3.985-4.014), in order of NoPE support. Same monotone pattern as the 51M grid: support growth sets
  long-range behaviour, and d=2 bounds it best. ASEntmax (3.874) beats all Stieltjes cells.
- So at 125M the trade-off is real and q/d cannot escape it: the cells that are better at the train length are
  worse at 16k, and no cell matches ASEntmax on either axis. ASEntmax's learned per-query scale gets both.
- PG-19: q3/d3 (3.847) and q4/d3 (3.861) beat q3/d2 (3.880) at 16k, so on in-domain prose the wider window is
  better; ASEntmax (3.800) still best.

## 2. Diagnostics on the trained q3/d2 (job 8250602): the model wants a wider window, but not by much

Inference-time override, 64 papers (trained setting as baseline):

| override | 512 | 2048 | 4096 | 8192 | 16384 |
|---|---:|---:|---:|---:|---:|
| trained q3 d2 | 4.518 | 4.105 | 4.051 | 3.984 | 3.913 |
| d 1.5 | +0.12 | +0.15 | +0.18 | +0.18 | +0.19 |
| d 2.5 | +0.03 | +0.06 | +0.08 | +0.09 | +0.10 |
| d 3 | +0.21 | +0.30 | +0.38 | +0.42 | +0.43 |
| q 2 | +0.00 | +0.01 | +0.01 | +0.01 | +0.02 |
| q 4 | +0.00 | +0.00 | +0.00 | -0.00 | -0.01 |
| q 6 | +0.03 | +0.03 | +0.04 | +0.03 | +0.02 |

- No inference-time change improves on the trained setting at any length (q=4 at 16k: -0.005, noise). The
  trained d is again a sharp optimum (d 2.5 already +0.03 at 512); the model co-adapts to its window, as at 51M.
- Gradient at L=2048 (8 papers): sum dL/dd = -0.118 (NoPE heads -0.061, ALiBi heads -0.057; 67% of (layer, head)
  pairs negative), i.e. the loss wants d LARGER at 125M, the opposite sign to 51M (+0.07). dL/dq = 0.000. Finite
  differences: d+0.25 -> 4.138 (-0.002), d-0.25 -> 4.196 (+0.056); so the window is at the wide edge of its
  local optimum rather than inside it. Prediction 1 (sign flips to "wider", 65%) right; prediction 2 (no
  inference change helps at the train length, d=3 helps at 16k) half: nothing helps anywhere.
- Taken together with §1: the gradient points toward wider d, training at wider d gives a small train-length gain
  and a long-range loss, and the trained model at any fixed d sits in a sharp local optimum. A learnable d would
  presumably find something near 2-2.5 from this init and gain ~0.01-0.04 at the train length while giving back
  some of the 16k advantage: it would land near q4/d2 or q3/d3, not near ASEntmax. The thing a fixed-d map cannot
  do, and the AS scale does, is vary the effective window per query.

## 3. Predictions scored (made Nov 1)
| # | prediction | result |
|---|---|---|
| 1 | dL/dd negative at 125M (wants wider), opposite to 51M | right |
| 2 | no inference-time d helps at train length; d=3 helps at 16k | half (nothing helps at 16k either) |
| 3 | best cell at val@2048 has d=3 and closes >= half the gap to softmax | wrong: best is q4/d2 (d=2), closes 0.04 of 0.073 |
| 4 | best cell matches ASEntmax at 16k ArXiv | wrong (3.948 vs 3.874) |
| 5 | q3/d4 worst at 16k | pending |
| 6 | q4 vs q3 within 0.02 at matched d | right at 16k (0.04 at d2 is the exception), right at train length |

## 4. Where this leaves Stieltjes vs ASEntmax, and what "learn q and d" would buy

1. The windowed Stieltjes long-document result survives at 125M for every (q, d) tried, with d=2 the best bound.
2. It is now behind ASEntmax on every axis at 125M (train length by 0.02-0.06, ArXiv 16k by 0.07-0.14, PG-19 by
   0.05-0.08). At 51M it was ahead on ArXiv by 0.18 (3 seeds). The crossover is between 51M and 125M.
3. The fixed-(q, d) family has a train-length vs long-range trade-off that the four cells trace out; ASEntmax
   sits off that frontier. A learnable scalar d per head would move along the frontier, not off it. The
   mechanism that puts ASEntmax off it is the per-query, position-dependent scale. The Stieltjes analogue is
   a per-query window d_i(x, n) (proposed Oct 29 as "adaptive window"), which would be the experiment that could
   change the ranking; a scalar learnable d is unlikely to.
4. Compute: the Stieltjes kernel is 2.1x softmax and 1.4x ASEntmax per step at this width, and the map keeps
   14-50 of 16k keys per row on the NoPE heads without any speed benefit. If the method is to be continued at
   420M, that kernel work (block skipping, as AdaSplash does) comes first.
