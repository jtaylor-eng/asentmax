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

## 5. Next options

a. LR bracket: 6e-3 for softmax, stieltjes_q4, aswstieltjes (3 runs, ~5 GPU-h), since 3e-3 is the grid edge.
b. softmax_as_zi at 1.5e-3/3e-3/6e-3: separates the AS scale from the map in the tie at 512 and the +0.22.
c. Seeds 2-3 at the selected LRs (8 runs): is the 0.001 tie and the 0.02 gap real.
d. Diagnostics on the existing ckpts (no training): per-head output drift vs n; loss at 4096 with NoPE heads
   capped to a 512 window (is the damage in NoPE heads?).
