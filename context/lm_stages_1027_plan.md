# LM stages 0-2 plan (Oct 27): where does windowed Stieltjes' long-document gain come from, and is it real?

Follows reproduction_lm_1018.md §4g (windowed q4/d2 0.34 nats better than softmax at 16k on ArXiv papers, 1 seed).
Code: nanoGPT `stieltjes_lm` @ 2ad78b5. Results under `$SCRATCH/nanogpt_results/lm1012/`. Jobs: diag 8142593,
stage-1 array 8142594_[0-14], stage-2 array 8142595_[0-7]. All exp-slope NAPE; training protocol unchanged (51M,
1B FineWeb tokens, T=512); every new run gets the FineWeb 4k + 16k ladders and the ArXiv ladder.

## Stage 0: diagnostics on existing checkpoints (no training, ~2 GPU-h)
(a) inference-time q/d sweep on the trained wstieltjes_q4_d2_xs (lr 1.5e-3, 3e-3): d in {1.25, 1.5, 3, 4, dense},
    q in {3, 6, 8}; ArXiv 512..16k on 64 papers -> `arxiv_d<d>.json`, `arxiv_q<q>.json`.
(b) dL/dq, dL/dd per layer/head on the three Stieltjes rows at L=2048, 8192 (8 papers), with finite-difference
    checks at d +- 0.25, q +- 0.5 -> `qd_grad_L<L>.json`. (Differentiable map `stieltjes_normalize_qd`
    verified against fp64 finite differences to 5 decimals.)
(c) NoPE heads capped to the last 2048 keys at inference for softmax / entmax / ASEntmax / windowed q4/d2 ->
    `arxiv_cap2048.json`. If the capped softmax matches windowed Stieltjes, the gain is "a learned soft cap".

Local smoke (wstieltjes_q4_d2_xs lr1.5e-3, 4 papers, L=2048/4096): d=1.5 hurts (+0.20), q=6 helps a little
(-0.01), cap-2048 is neutral at 4096 (+0.01). Gradients at L=2048: sum dL/dd = +0.061 (want smaller d, almost
all from NoPE heads), sum dL/dq = -0.008 (want larger q, weakly). The FD check shows loss rises for BOTH
d +- 0.25 (4.561 / 4.541 vs 4.505), i.e. d=2 is at a local optimum for this model and the gradient is only
locally informative; the inference sweep in (a) is the real answer.

## Stage 1: AS disentangle + LR bracket + seeds (15 runs, ~25 GPU-h)
- aswstieltjes_q4_d2_zi_xs and wstieltjes_q2_d4_xs at 7.5e-4 / 1.5e-3 / 3e-3 (6): separates the scale from
  (q, d) in the AS-windowed row's ArXiv collapse.
- 6e-3 for softmax_xs, asentmax_zi_xs, wstieltjes_q4_d2_xs (3): brackets the grid edge.
- seeds 1, 2 for softmax_xs @3e-3, asentmax_zi_xs @1.5e-3, wstieltjes_q4_d2_xs @1.5e-3 (6): dirs `lr<lr>_s<seed>`.

## Stage 2: q/d grid (8 runs, ~15 GPU-h)
wstieltjes_q{3,4,6}_d{1.5,2,3}_xs at 1.5e-3, minus the existing q4/d2.

## Predictions (score in the results report)
| # | prediction | conf |
|---|---|---|
| 1 | seeds: windowed q4/d2 beats softmax at 16k ArXiv by >= 0.2 in all 3 seeds | 75% |
| 2 | seeds: FineWeb @512 spread within a row <= 0.01 (so the 0.006-0.01 row gaps at 512 are noise) | 70% |
| 3 | AS is the culprit: wstieltjes_q2_d4_xs (no scale) stays flat on ArXiv past 2k (16k-2k < +0.15) | 60% |
| 4 | aswstieltjes_q4_d2_zi_xs degrades on ArXiv like the q2/d4 AS row (16k-2k > +0.4) | 55% |
| 5 | 6e-3 does not beat 3e-3 at 512 for any of the 3 rows | 60% |
| 6 | stage 2: best 16k ArXiv loss at d=1.5 (tighter window), not d=3 | 50% |
| 7 | stage 2: q=6 within 0.02 of q=4 at matched d at 512 and 16k (q matters less than d) | 60% |
| 8 | stage 0(a): no inference-time d change improves 16k ArXiv by > 0.05 over the trained d=2 | 60% |
| 9 | stage 0(c): capped softmax at 8k within 0.1 of windowed Stieltjes (the gain is mostly a cap effect) | 50% |
