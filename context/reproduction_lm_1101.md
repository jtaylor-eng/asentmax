# LM stage 4 results (Nov 1): 125M at T=2048, 2.5B tokens, first pass (1 seed, 2 LRs)

Plan and predictions: chat of Oct 30 (reproduced in §4). Prior: reproduction_lm_1030.md (51M, T=2048, 3 seeds).
Jobs: chain 8233113-8233116 (4 x 4 h segments, afterany; the first chain 8226513-5 lost its dependents to
aftercorr treating the wall-guard exit 4 as failure). Data: FineWeb shards 11-35 (`fineweb25B`), val shared.
Model: 12L x 12H x 768, 123.6M params (85M non-embedding), NAPE exp slopes 2^(-8h/6) on heads 1-6, NoPE on 7-12.
0.5M tokens/step, 4,770 steps, warmup 250, cosine to lr/10. Sparse rows at micro-batch 8 x 32 accum (memory).
Figure: `nanoGPT/results/lm1012/stage4_125m.png`.

## 1. Results (both LRs; selection by val@2048)

| row | lr | val@2048 | FineWeb 16k | ArXiv 2k / 4k / 8k / 16k | PG-19 2k / 4k / 8k / 16k |
|---|---|---:|---:|---|---|
| softmax | 6e-4 | 3.5119 | 3.503 | 4.202 / 4.329 / 4.439 / 4.481 | 3.873 / 3.872 / 3.926 / 3.961 |
| softmax | **1.2e-3** | **3.4614** | 3.437 | 4.035 / 4.104 / 4.267 / 4.353 | 3.802 / 3.796 / 3.847 / 3.881 |
| ASEntmax | 6e-4 | 3.5183 | 3.520 | 4.075 / 3.997 / 3.879 / 3.821 | 3.859 / 3.828 / 3.831 / 3.842 |
| ASEntmax | **1.2e-3** | **3.4755** | 3.458 | 4.037 / 4.006 / 3.933 / 3.874 | 3.821 / 3.793 / 3.796 / 3.800 |
| windowed q3/d2 | 6e-4 | 3.5412 | 3.555 | 4.150 / 4.108 / 4.046 / 4.019 | 3.940 / 3.913 / 3.907 / 3.915 |
| windowed q3/d2 | **1.2e-3** | **3.5340** | 3.493 | 4.141 / 4.105 / 4.021 / 3.948 | 3.897 / 3.872 / 3.873 / 3.880 |

Selected LR, with the 51M T=2048 3-seed means:

| row | val@train len (125M / 51M) | ArXiv 16k (125M / 51M) | rise 2k -> 16k | PG-19 16k (125M / 51M) | rise |
|---|---|---|---:|---|---:|
| softmax | 3.461 / 3.704 | 4.353 / 4.544 | +0.32 | 3.881 / 4.070 | +0.08 |
| ASEntmax | 3.476 / 3.687 | **3.874** / 4.156 | **-0.16** | **3.800** / 4.007 | -0.02 |
| windowed q3/d2 | 3.534 / 3.704 | 3.948 / 3.976 | -0.19 | 3.880 / 3.986 | -0.02 |

## 2. Reading

1. **The long-context result scales.** At 125M the sparse maps' loss still falls from 2k to 16k on ArXiv (ASEntmax
   -0.16, windowed -0.19) while softmax's rises (+0.32, worse than at 51M: +0.19). Gap to softmax at 16k:
   0.48 (ASEntmax), 0.41 (windowed). On PG-19 the same sign, gaps 0.08 / 0.00. Loss by position at 16k: the
   sparse maps sit at 3.8-3.9 from position 4k onward; softmax at 4.3-4.6. The mechanism is the same (NoPE
   support 512 -> 16k: softmax all keys, ent 7.74; ASEntmax 31 -> 88; windowed 14 -> 32).
2. **Windowed Stieltjes lost its lead to ASEntmax, and lost 0.07 at the training length.** At 51M windowed was
   0.18 ahead of ASEntmax at 16k ArXiv and tied softmax at the train length. At 125M: val@2048 softmax 3.461 <
   ASEntmax 3.476 < windowed 3.534; ArXiv 16k ASEntmax 3.874 < windowed 3.948. The periodic val trajectory shows
   windowed 0.03-0.05 behind from step 2250 onward, not a final-eval artefact; the gap is also present at 6e-4
   (3.541 vs 3.512). Windowed's NoPE support at 125M is 14 keys at 512 (vs 23 at 51M, 55 for q4/d2 at 51M):
   with 64-dim heads at d=768 the score gaps are larger and the d=2 window admits fewer keys. The likely
   reading is that d=2 is too tight at this width (q3/d2 was tuned at 51M), i.e. the window needs re-tuning
   with scale, which ASEntmax's learned per-query scale does for free. Not resolved by this pass.
3. **Softmax's train-length lead (0.015 over ASEntmax, 0.073 over windowed) at 1 seed.** The 51M seed sd was
   0.003-0.015, so 0.015 is not interpretable and 0.073 is.
4. LR: all three rows selected the grid top (1.2e-3), and the sparse rows gain more from the higher LR at long
   context. One more LR point (2.4e-3) is needed before the train-length comparison means anything; the stage-4
   plan allowed for that.
5. Cost at 125M (A100-40GB): softmax 6.4 s/step (MFU 25%), ASEntmax 10.0 (16%), windowed Stieltjes 13.7 (11%).
   Per run: 8.5 / 13 / 18 h train. The sparse kernels are the bottleneck at this width; the Stieltjes kernel
   does not exploit the sparsity it produces.

## 3. Predictions scored (made Oct 30, before the runs)
| # | prediction | result |
|---|---|---|
| 1 | val@2048 within 0.015 across rows (55%) | wrong: windowed 0.073 behind |
| 2 | sparse maps' ArXiv loss still falls 2k -> 16k, softmax's rises (70%) | right |
| 3 | windowed ahead of ASEntmax at 16k ArXiv (60%) | wrong: 0.07 behind |
| 4 | gap to softmax at 16k ArXiv >= 0.3 (60%) | right (0.41 / 0.48) |
| 5 | softmax closer on PG-19 at 125M, gap < 0.05 (55%) | right for windowed (0.00), wrong for ASEntmax (0.08) |
| 6 | RULER | not run |
3 / 5.

## 4. Decision
The stage-4 rule was: stage 5 (420M, 7B tokens) is warranted if predictions 2 and 4 hold. They do, for both
sparse maps. But the row to carry forward is no longer clear: ASEntmax is better than windowed Stieltjes at 125M
on every number, and softmax is 0.07 better than windowed at the training length. Before 420M, the cheap
questions are (a) one more LR (2.4e-3) for all three rows, 3 runs, ~40 GPU-h; (b) whether a wider window
(d = 3, or q4/d2) recovers windowed Stieltjes at 125M, 2 runs, ~36 GPU-h; (c) a second seed of the three
selected rows, 3 runs, ~40 GPU-h. Of these, (b) decides whether Stieltjes stays in the story at scale; (a) and
(c) decide whether the ASEntmax-vs-softmax train-length gap is real.
