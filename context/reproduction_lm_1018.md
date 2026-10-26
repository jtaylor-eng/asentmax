# LM results (Oct 18): Stieltjes variants vs softmax, nanoGPT on FineWeb (plan: lm_plan_1012.md)

Code jtaylor-eng/nanoGPT `stieltjes_lm` @ ee7ab3a (bug-fix fbc2b33 after the first array, 7796844, died on a
cwd path bug). Probe 7809994, array 7809996_[0-11] (Oct 15 03:48-06:10, A100-40GB). Summaries copied to
`submodules/nanoGPT/results/lm1012/` (final.json, lengths.json, metrics.jsonl, run.log); checkpoints on OSC scratch.
Setup: 50.9M params, 8L x 8H x 512, NAPE (heads 0-3 ALiBi, 4-7 NoPE), T=512, 1.0B tokens, 7630 steps, 1 seed.
Stieltjes Newton sweeps = 15 (probe: worst max|S-1| 5.9e-5 < 1e-4 in 6/6 cells; val loss = 30-sweep to 4 dp).

## 1. Val loss at T=512 (final.json, 13M val tokens; selection metric)

| row | 7.5e-4 | 1.5e-3 | 3e-3 | best | vs softmax | train h | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| softmax | 3.7924 | 3.7587 | **3.7531** | 3e-3 | - | 0.96 | 303k |
| stieltjes_q4 (dense) | 3.8109 | 3.7767 | **3.7721** | 3e-3 | +0.019 | 1.84 | 159k |
| wstieltjes_q4_d2 | 3.8077 | **3.7776** | 3.7837 | 1.5e-3 | +0.025 | 1.83 | 160k |
| aswstieltjes_q2_d4_zi | 3.7949 | 3.7612 | **3.7542** | 3e-3 | +0.001 | 1.88 | 155k |

No divergence or loss spikes at any LR (max post-warmup train loss is identical across rows per LR).
**Caveat: 3 of 4 rows select the top of the grid (3e-3)**, so their optimum may be higher; only wstieltjes is bracketed.

## 2. Length extrapolation (selected LR; same 2M val tokens per length; reported, not selected on)

| row | 512 | 1024 | 2048 | 4096 | 4096 - 512 | (4096-512 at 7.5e-4 / 1.5e-3 / 3e-3) |
|---|---:|---:|---:|---:|---:|---|
| softmax | 3.756 | 3.711 | 3.724 | **3.820** | **+0.064** | +0.096 / +0.081 / +0.064 |
| stieltjes_q4 | 3.774 | 3.737 | 3.891 | 4.308 | +0.533 | +0.537 / +0.359 / +0.533 |
| wstieltjes_q4_d2 | 3.780 | 3.743 | 3.816 | 4.032 | +0.252 | +0.246 / +0.252 / +0.311 |
| aswstieltjes_q2_d4_zi | 3.750 | 3.710 | 3.763 | 3.969 | +0.219 | +0.211 / +0.175 / +0.219 |

Ranking is the same at every LR. Loss by position at L=4096 (16 buckets of 256): all rows match through
position ~1024; past 2x the train length softmax rises 3.70 -> 3.99, AS-windowed 3.74 -> 4.35, windowed
3.79 -> 4.38, dense 3.81 -> 4.91.

## 3. Attention statistics (last-quarter query rows, mean over layers; 4 val sequences)

| row | head type | support @512 | support @4096 | entropy @512 | entropy @4096 |
|---|---|---:|---:|---:|---:|
| softmax | ALiBi | 256 | 256 | 1.44 | 1.47 |
| softmax | NoPE | 449 (all) | 3584 (all) | 3.97 | 6.45 |
| stieltjes_q4 | ALiBi | all | all | 0.98 | 0.98 |
| stieltjes_q4 | NoPE | all | all | 4.32 | 7.54 (uniform = 8.18) |
| wstieltjes_q4_d2 | ALiBi | 2.7 | 2.8 | 0.49 | 0.53 |
| wstieltjes_q4_d2 | NoPE | 39 | 74 | 2.64 | 3.63 |
| aswstieltjes_q2_d4_zi | ALiBi | 5.9 | 6.2 | 1.06 | 1.16 |
| aswstieltjes_q2_d4_zi | NoPE | 70 | 120 | 3.05 | 4.00 |

(softmax ALiBi "support 256" is fp32 underflow of exp(-slope * distance), not true sparsity.)

## 3a. Findings

1. At the training length, Stieltjes costs 0.02-0.025 nats (dense, windowed), and AS-windowed q2/d4 ties softmax
   (+0.001, within single-seed noise). The scale-vs-map confound is unresolved: softmax_as_zi was not run.
2. Length extrapolation: softmax with NAPE extrapolates best, the reverse of the synthetic ranking. Dense
   Stieltjes is worst (+0.53), as the dispersion argument predicts: its NoPE heads go near-uniform at 4096
   (entropy 7.54 of 8.18 max). Windowing halves the damage (+0.25); the AS scale helps a little more (+0.22).
3. Windowed maps are sparse (39-120 keys on NoPE heads) but support still roughly doubles from 512 to 4096,
   because a fixed threshold relative to the top score admits more keys as n grows. Exact zeros do not make
   the head's output length-invariant; they make it a sum over a support that grows with n.
4. Hypothesis, not tested: softmax NoPE heads at 512 are already high-entropy averaging heads, and a dispersed
   average drifts slowly with n, while sparse heads that learned to pick ~40 keys see a different selected set
   at 4096. Per-head output-norm drift vs n would test it.

## 4. Predictions scored (lm_plan_1012.md)

| # | prediction | outcome |
|---|---|---|
| 1 | Stieltjes within 0.05 of softmax at 512; softmax best or tied (60%) | right (+0.001..+0.025; AS-w ties) |
| 2 | dense q4 best Stieltjes row at 512 (55%) | wrong: AS-windowed best |
| 3 | windowed >= 0.02 worse than dense at 512 (55%) | wrong: +0.006 |
| 4 | smallest 4096 rise: aswstieltjes_q2_d4_zi (55%) | wrong: softmax (+0.064); AS-w best of the Stieltjes rows |
| 5 | largest 4096 rise: dense q4 (60%) | right (+0.53) |
| 6 | Stieltjes step 3-6x softmax (30%) | wrong as run: 1.9x at 15 sweeps (3.0x at 30) |
| 7 | >= 1 Stieltjes row diverges at 3e-3 (40%) | wrong: none |
| 8 | probe picks 15 sweeps (85%) | right |

3 of 8. The systematic miss: carrying the synthetic-task ranking (sparse > softmax on length) over to LM with NAPE.

## 4a. Control: softmax + AS scale (zero-init 0.05), job 7915703 (Oct 18, ee7ab3a)

| row | 7.5e-4 | 1.5e-3 | 3e-3 | best | 4096 - 512 (sel.) | 4096 - 512 per LR |
|---|---:|---:|---:|---:|---:|---|
| softmax | 3.7924 | 3.7587 | 3.7531 | 3.7531 | +0.064 | +0.096 / +0.081 / +0.064 |
| softmax_as_zi | 3.7939 | 3.7570 | 3.7537 | 3.7537 | +0.156 | +0.108 / +0.150 / +0.156 |
| aswstieltjes_q2_d4_zi | 3.7949 | 3.7612 | 3.7542 | 3.7542 | +0.219 | +0.211 / +0.175 / +0.219 |

- At 512 the three rows are within 0.0011 at every LR (single seed): the AS scale does not help softmax and the
  Stieltjes map does not hurt once the scale is there. The tie is not evidence for the map.
- At 4096 the AS scale itself costs softmax +0.09 (+0.064 -> +0.156), roughly 60% of the AS-windowed row's
  excess over softmax; the remaining +0.06 is attributable to the windowed q2/d4 map. NoPE-head entropy at 4096:
  softmax 6.45, softmax_as 5.85, aswstieltjes 4.00, i.e. sharper heads extrapolate worse in this setup.
- Ranking of 4096 rise (all LRs consistent): softmax < softmax_as_zi < aswstieltjes < wstieltjes < dense stieltjes.
- Verdict for Stieltjes attention in LM: no gain at the train length, worse length extrapolation, 1.9x step time.
  Prediction (55% that the scale accounts for all of the 512 tie): right.

## 4b. Entmax / ASEntmax under the same protocol (array 7972134, Oct 21, commit 6be8074; first attempt 7965545 lost to a GPFS stall)

AdaSplash 1.5-entmax kernel (vendored from this repo, NAPE slopes), 10 Halley iterations; ASEntmax = same + zero-init AS scale.

| row | 7.5e-4 | 1.5e-3 | 3e-3 | best | vs softmax | 4096 - 512 (sel.) | per LR | ms/step |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| softmax | 3.7924 | 3.7587 | 3.7531 | 3.7531 | - | +0.064 | +0.096 / +0.081 / +0.064 | 432 |
| softmax_as_zi | 3.7939 | 3.7570 | 3.7537 | 3.7537 | +0.001 | +0.156 | +0.108 / +0.150 / +0.156 | 461 |
| entmax | 3.7984 | 3.7620 | 3.7615 | 3.7615 | +0.008 | +0.115 | +0.194 / +0.221 / +0.115 | 698 |
| asentmax_zi | 3.8018 | 3.7688 | 3.7652 | 3.7652 | +0.012 | +0.137 | +0.147 / +0.128 / +0.137 | 724 |
| aswstieltjes_q2_d4_zi | 3.7949 | 3.7612 | 3.7542 | 3.7542 | +0.001 | +0.219 | +0.211 / +0.175 / +0.219 | 845 |
| wstieltjes_q4_d2 | 3.8077 | 3.7776 | 3.7837 | 3.7776 | +0.025 | +0.252 | +0.246 / +0.252 / +0.311 | 821 |
| stieltjes_q4 | 3.8109 | 3.7767 | 3.7721 | 3.7721 | +0.019 | +0.533 | +0.537 / +0.359 / +0.533 | 824 |

- At 512: entmax +0.008, ASEntmax +0.012 over softmax (1 seed; likely real but small). The AS scale costs entmax
  a little (+0.004) where it was neutral for softmax.
- At 4096: softmax (+0.064) still extrapolates best. Entmax +0.115 at 3e-3 but +0.19..+0.22 at the lower LRs
  (its ranking vs softmax_as depends on LR); ASEntmax +0.13..+0.15 at every LR, i.e. the scale makes entmax's
  extrapolation more LR-stable but not better than softmax's. Last-256-position loss at 4096: softmax 3.99,
  entmax 4.12, ASEntmax 4.16, softmax_as 4.22.
- Entmax's NoPE-head support is sparse (76 keys at 512) but still doubles to 145 at 4096, like windowed Stieltjes
  (39 -> 74). A fixed threshold relative to the top score does not give an n-independent support in LM.
- Ranking of the 4096 rise, best to worst: softmax < entmax ~ ASEntmax ~ softmax_as < AS-windowed Stieltjes <
  windowed Stieltjes < dense Stieltjes. Every map sharper than softmax extrapolates worse here.
- Predictions scored: entmax within 0.01 of softmax at 512 (65%) right; entmax rises more than softmax at 4096
  (65%) right; ASEntmax rises more than entmax (60%) right at the selected LR, wrong at the two lower LRs;
  ASEntmax within 0.005 of best at 512 (55%) wrong (+0.012).
- Relation to arXiv 2506.16640 §5.2 / Table 3 (correction of an earlier wrong statement that the paper has no LM
  result): the paper DOES report LM gains. 420M LLaMA-3-style models (24L x 12H x 768, ~300M non-embedding),
  DCLM-Edu, 7B tokens, T=2048, LR 3e-4, NAPE with the original exponential ALiBi slopes; ArXiv ppl at
  2K/4K/8K/16K: softmax 17.10/13.87/12.46/12.71, entmax 16.85/13.36/11.04/10.07, ASEntmax 16.86/13.31/10.89/10.01.
  Our runs do NOT reproduce that ordering. Setup differences, most important first:
  1. ALiBi slopes. We used the synthetic-task slopes 1, 1/2, 1/3, 1/4 (heads locked to ~10 tokens; all long-range
     work on the 4 NoPE heads). The paper's LM runs use exponential ALiBi slopes (2^-8i/H), under which the
     ALiBi heads can themselves attend hundreds of tokens back. Our choice penalises any map that sharpens.
     -> tested in §4c.
  2. Scale: 51M/25M-non-emb on 1B tokens vs 420M/~300M on 7B; T=512 vs 2048.
  3. Eval text: FineWeb web pages (short, concatenated) vs 16K-token ArXiv/PubMed documents with real
     long-range structure.
  4. LR: ours selects 3e-3; theirs is a single 3e-4.
  Caveats on our numbers: single seed, small model, LR grid capped at 3e-3, 8x extrapolation only.

## 4c. Exponential ALiBi slopes (the paper's LM setting), array 8000083, Oct 23, commit cf1f03b

`_xs` rows: NAPE ALiBi heads use 2^(-8h/4) = 0.25, 0.0625, 0.0156, 0.0039 instead of 1, 1/2, 1/3, 1/4. Nothing else changed.

| row | 7.5e-4 | 1.5e-3 | 3e-3 | best@512 | val@4096 (sel.) | 4096 - 512 | per LR | last-256 @4096 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| softmax (1/h) | 3.7924 | 3.7587 | 3.7531 | 3.7531 | 3.820 | +0.064 | +0.096 / +0.081 / +0.064 | 3.99 |
| softmax_xs | 3.8136 | 3.7729 | 3.7616 | 3.7616 | 3.740 | -0.024 | +0.087 / -0.014 / -0.024 | 3.81 |
| entmax (1/h) | 3.7984 | 3.7620 | 3.7615 | 3.7615 | 3.879 | +0.115 | +0.194 / +0.221 / +0.115 | 4.12 |
| entmax_xs | 3.7947 | 3.7558 | 3.7653 | 3.7558 | 3.751 | -0.008 | -0.034 / -0.008 / -0.034 | 3.84 |
| asentmax_zi (1/h) | 3.8018 | 3.7688 | 3.7652 | 3.7652 | 3.898 | +0.137 | +0.147 / +0.128 / +0.137 | 4.16 |
| asentmax_zi_xs | 3.7948 | 3.7616 | 3.7675 | 3.7616 | 3.738 | -0.020 | -0.014 / -0.020 / -0.046 | 3.82 |

- The slope choice dominated the earlier length result. With the paper's slopes all three maps extrapolate to 8x
  with NO loss increase (4096 within -0.05..+0.09 of 512 at every LR except softmax_xs at 7.5e-4); the last-256
  positions at 4096 sit at 3.81-3.84 vs 3.99-4.16 before. The ALiBi heads now attend 17-2300 keys back (entropy
  1.8-3.3) instead of 4-256 (0.75-1.5), so the long-range work no longer falls on the NoPE heads alone.
- At 512 the ordering flips: entmax_xs 3.7558 and asentmax_xs 3.7616 now BEAT softmax_xs 3.7616/3.7729 at the
  matched LR (entmax -0.017 at 1.5e-3, -0.004 at 7.5e-4; +0.004 at 3e-3). Softmax lost 0.009 from the slope
  change; entmax/ASEntmax gained 0.004-0.006. Single seed, so the 0.00-0.02 margins are suggestive, not firm.
- At 4096 (selected LRs): asentmax_xs 3.738 < softmax_xs 3.740 < entmax_xs 3.751, all within 0.013. At 3e-3,
  where all three are directly comparable, asentmax_xs 3.76 / entmax_xs 3.79 / softmax_xs 3.81 in the last-256
  bucket, i.e. the paper's ordering (ASEntmax < entmax < softmax) appears at long range, by 0.02-0.05.
- The AS scale now helps entmax at 4096 (-0.013 vs entmax_xs at selected LRs, -0.03 in the tail at 3e-3) where
  with 1/h slopes it hurt. Consistent with the paper's claim that the scale matters at long range.
- Reading: our earlier "sharper maps extrapolate worse" was an artifact of the 1/h slopes. Under the paper's
  slopes we see the paper's qualitative ordering at 8x, with much smaller margins than their Table 3
  (their 16K ArXiv ppl: softmax 12.7 vs entmax 10.1, ~0.23 nats) and no clear winner at the train length.
  Remaining differences from the paper: 51M vs 420M, 1B vs 7B tokens, T=512 vs 2048, FineWeb vs ArXiv/PubMed.
- Predictions scored: all three rows improve at 4096 (75%) right; entmax-softmax gap shrinks/reverses (55%)
  right; entmax beats softmax at 4096 as in the paper (35%) wrong at selected LRs (+0.011), right in the
  long-position tail at 3e-3; ordering at 512 unchanged with softmax ahead (65%) wrong, entmax ahead.
- Stieltjes rows were not rerun with exp slopes. Given the slope effect, the §1-2 Stieltjes extrapolation numbers
  are also slope-confounded; the §1 train-length deficit (+0.02 dense/windowed) is probably not, since the softmax
  control moved only 0.009.

## 4d. Stieltjes rows with exponential slopes (array 8098758, Oct 26, commit cf1f03b, 15 Newton sweeps)

All six exp-slope rows, best LR by val@512; the Stieltjes rows' 1/h numbers from §1-2 for reference.

| row (exp slopes) | 7.5e-4 | 1.5e-3 | 3e-3 | best@512 | vs softmax_xs | val@4096 | 4096 - 512 | per LR | last-256 @4096 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| softmax_xs | 3.8136 | 3.7729 | 3.7616 | 3.7616 | - | 3.740 | -0.024 | +0.087 / -0.014 / -0.024 | 3.81 |
| entmax_xs | 3.7947 | 3.7558 | 3.7653 | 3.7558 | -0.006 | 3.751 | -0.008 | -0.034 / -0.008 / -0.034 | 3.84 |
| asentmax_zi_xs | 3.7948 | 3.7616 | 3.7675 | 3.7616 | 0.000 | 3.738 | -0.020 | -0.014 / -0.020 / -0.046 | 3.82 |
| stieltjes_q4_xs | 3.8104 | 3.7715 | 3.7615 | 3.7615 | 0.000 | 3.871 | +0.106 | +0.097 / +0.045 / +0.106 | 4.09 |
| wstieltjes_q4_d2_xs | 3.7931 | 3.7632 | 3.7671 | 3.7632 | +0.002 | **3.720** | **-0.047** | -0.017 / -0.047 / -0.044 | **3.76** |
| aswstieltjes_q2_d4_zi_xs | 3.8033 | 3.7657 | 3.7553 | **3.7553** | -0.006 | 3.791 | +0.039 | +0.035 / +0.026 / +0.039 | 3.92 |

1/h -> exp, per Stieltjes row (best@512; 4096-512): dense 3.7721 -> 3.7615 (+0.53 -> +0.11); windowed q4/d2
3.7776 -> 3.7632 (+0.25 -> -0.05); AS-windowed q2/d4 3.7542 -> 3.7553 (+0.22 -> +0.04). Softmax: 3.7531 -> 3.7616.

- Train length: all six rows within 0.008 of each other (3.7553..3.7632). The §1 Stieltjes deficit is gone:
  dense and windowed q4 gained 0.011-0.014 from the slope change while softmax lost 0.009. Single seed; these
  margins are below what one seed can resolve.
- 4096: windowed Stieltjes q4/d2 is the best row of all six (3.720; last-256 3.76), ahead of ASEntmax 3.738,
  softmax 3.740, entmax 3.751. Same ordering at the shared LRs 1.5e-3 and 3e-3 (3.720/3.725 vs 3.738/3.718 for
  ASEntmax). Its NoPE-head support is the smallest and grows least: 48 -> 67 keys (entmax 88 -> 316).
- Dense Stieltjes q4 is still the worst at 4096 (+0.106; NoPE entropy 7.68 of 8.18 max at 4096, i.e. near
  uniform): the dispersion failure is a property of the dense map and survives the slope fix, though the slopes
  cut the damage 5x (+0.53 -> +0.11).
- AS-windowed q2/d4 is best at 512 (3.7553) but +0.039 at 4096, behind the un-scaled windowed q4/d2 row and the
  three non-Stieltjes rows. Its NoPE support grows 94 -> 195. The adaptive scale that helps entmax (asentmax_xs
  -0.013 vs entmax_xs) hurts windowed Stieltjes here; the q2 vs q4 difference is confounded with it (not run: q4/d2
  + AS, q2/d4 without AS).
- Predictions (§4c follow-up) scored: all Stieltjes rows within +0.05 at 4096 (60%) wrong, dense +0.106; dense
  worst at 4096 (70%) right; dense/windowed train-length deficit >= 0.015 persists (65%) wrong, gone; AS-windowed
  within 0.005 of ASEntmax at 512 (55%) right (-0.006); windowed NoPE support ~2x from 512 to 4096 (80%) partly
  wrong: 1.4x for q4/d2, 2.1x for AS q2/d4.

## 4f. Ladder to 16k (32x) for the exp-slope rows (job 8118074, Oct 27; `lengths_16k.json`, plot `lengths_16k.png`)

Same 2M val tokens per length. Selected LR per row (by val@512). Attention stats on 2 sequences.

| row (exp slopes) | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 16k - 512 | last-1k @16k | NoPE support 512 / 4k / 16k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| softmax | 3.764 | 3.715 | 3.706 | 3.740 | 3.799 | 3.862 | +0.098 | 3.97 | all / all / all (ent 7.92 of 9.57) |
| entmax | 3.759 | 3.711 | 3.707 | 3.751 | 3.837 | 3.956 | +0.197 | 4.16 | 91 / 319 / 615 |
| ASEntmax | 3.758 | 3.709 | 3.700 | 3.738 | 3.812 | 3.906 | +0.148 | 4.06 | 92 / 318 / 634 |
| Stieltjes q4 dense | 3.765 | 3.720 | 3.744 | 3.871 | 4.043 | 4.198 | +0.433 | 4.41 | all / all / all (ent 9.41, near uniform) |
| Stieltjes q4 d=2 windowed | 3.767 | 3.718 | 3.704 | **3.720** | **3.766** | **3.842** | **+0.076** | 3.98 | 49 / 69 / 84 |
| Stieltjes q2 d=4 + AS | 3.752 | 3.705 | 3.716 | 3.791 | 3.897 | 4.014 | +0.262 | 4.20 | 96 / 195 / 267 |

16384 at every LR (7.5e-4 / 1.5e-3 / 3e-3): softmax 4.053/3.843/3.862; entmax 3.917/3.956/3.856; ASEntmax
3.963/3.906/3.819; dense 4.181/4.089/4.198; windowed q4/d2 3.968/3.842/3.833; AS-windowed 4.117/3.987/4.014.

- Everything degrades past 8x; nothing holds flat to 16k. The best row (windowed q4/d2) is +0.076 at 32x, softmax
  +0.098. Expected on FineWeb: pages are a few hundred to a few k tokens, so most of a 16k window is unrelated
  documents and there is little long-range signal to use. This ladder measures robustness, not gain.
- Windowed Stieltjes q4/d2 stays the best row at 4k, 8k and 16k at its selected LR, by 0.02 over softmax; but
  at 3e-3 ASEntmax (3.819) edges it (3.833), and at 7.5e-4 entmax is best. Margins of 0.01-0.02 on one seed:
  the ranking among {windowed Stieltjes, softmax, ASEntmax} is not resolved. Entmax without the scale is clearly
  behind softmax at 16k (+0.10 at selected LRs); the AS scale recovers half of that (-0.05), consistent with
  the paper's claim that the scale matters at long range, though here neither beats softmax.
- Mechanism matches the earlier picture. Windowed q4/d2's NoPE support grows 49 -> 84 keys over 32x (1.7x);
  entmax's grows 91 -> 615 (6.8x); the AS-windowed row's 96 -> 267 (2.8x). Dense rows attend to everything and
  go near-uniform (softmax entropy 7.92 of 9.57 max, dense Stieltjes 9.41). The ALiBi heads are flat in entropy
  from 512 to 16k for every row (e.g. windowed 1.74 -> 1.79), so the degradation is all in the NoPE heads.
- Dense Stieltjes q4 is the worst row at every length past 2k (+0.43 at 16k), as before.
- Compared with the paper's Table 3 (ArXiv ppl decreasing out to 16K for entmax/ASEntmax): not reproduced here
  for any map, softmax included. The paper's eval documents are >=16K tokens of a single paper; ours are not.
  Our result is the non-contradicting version: on text without long-range structure, no map gains past 2-4x,
  and the sparse maps with bounded support degrade least (windowed Stieltjes) or about like softmax (ASEntmax).

## 4e. Where this leaves the LM comparison (all exp-slope, 1 seed, 51M, 1B tokens, T=512)

| question | answer |
|---|---|
| any map better than softmax at the train length? | no row beats softmax by more than 0.006; all within 0.008 |
| best extrapolation to 8x? | windowed Stieltjes q4/d2 (3.720), then ASEntmax 3.738, softmax 3.740, entmax 3.751 |
| to 32x (16k)? | windowed Stieltjes 3.842, softmax 3.862, ASEntmax 3.906, entmax 3.956, AS-windowed 4.014, dense 4.198 (§4f); all degrade past 8x |
| worst? | dense Stieltjes q4 (3.871): dense heavy tail disperses, as the proposal's own theory predicts |
| does the AS scale help? | entmax: yes at 4096 (-0.013). windowed Stieltjes: no (q2/d4+AS +0.07 behind q4/d2) |
| cost | Stieltjes 1.9x softmax step time (15 sweeps); entmax 1.6x |

The one Stieltjes result worth following up is windowed q4/d2 at 4096 (-0.02 vs softmax, -0.03 vs entmax,
single seed). Minimum to make it a claim: 2 more seeds of softmax_xs, entmax_xs, asentmax_zi_xs,
wstieltjes_q4_d2_xs at their selected LRs (8 runs, ~13 GPU-h), plus wstieltjes q4/d2 + AS to separate the
scale from q (3 runs).

## 5. Next options

a. LR bracket: 6e-3 for softmax, stieltjes_q4, aswstieltjes (3 runs, ~5 GPU-h), since 3e-3 is the grid edge.
b. softmax_as_zi at 1.5e-3/3e-3/6e-3: separates the AS scale from the map in the tie at 512 and the +0.22.
c. Seeds 2-3 at the selected LRs (8 runs): is the 0.001 tie and the 0.02 gap real.
d. Diagnostics on the existing ckpts (no training): per-head output drift vs n; loss at 4096 with NoPE heads
   capped to a 512 window (is the damage in NoPE heads?).
