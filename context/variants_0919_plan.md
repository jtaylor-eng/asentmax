# Stieltjes variants batch — plan and pre-registered predictions (2026-09-19)

Successor experiment to `reproduction_0913.md`. Status: **submitted** (see §5 for job IDs). Results will go
in `reproduction_09xx.md` (dated successor), not here.

## 1. What is being run

| row | attention | seeds | LR (sort / reverse / copy / mqmtar) | why |
|---|---|---|---|---|
| `stieltjes_q16` | dense Stieltjes, q=16 | 1 2 3 | 3.2e-3 / 3.2e-3 / 5e-4 / 4e-4 | tests the dispersion theory: q=16 needs gap ~(2n)^{1/16} (2.1 at 65k) vs q=4's 17.8 |
| `wstieltjes` | windowed, q=4, d=2 (c = 1/16) | 1 2 3 | 1.6e-3 / 1.6e-3 / 5e-4 / 4e-4 | the sparse variant (`stieltjes_proposal.tex` §2); exact zeros beyond gap ~d |
| `aswstieltjes` | windowed q=4 d=2 + adapt-softplus-tanh scale | 1 2 3 | 4e-4 / 4e-4 / 5e-4 / 2e-4 | the fair comparison to ASEntmax |
| `entmax` | 1.5-entmax, no scale | 1 2 | 4e-4 / 4e-4 / 5e-4 / 2e-4 | control for `wstieltjes` (paper's "Entmax" row; never run by us) |
| `asentmax` | (mqmtar only) | 1 2 3 | 4e-4 | 0907 row is a training failure (3/4 on the plateau at 1e-4/2e-4); the sweep showed 4e-4 escapes reliably |

LRs are guesses, not swept: the 0913 q=4 optimum for the Stieltjes rows (halved on sort/reverse for the
windowed one since sparse maps sat lower), ASEntmax's 0907 optimum for the entmax-family rows (NaN'd at
>= 8e-4 on sort/copy). One-LR results support existence claims ("X reaches Y at L") only; any surprising
win gets a bracketed sweep before it is quoted. Everything else (data, steps, warmup 10k, reverse FFN
512, 100-sample evals, validation-only selection, tie-break ladders) is frozen from 0913.

Cost estimate: 14 x 3 + 4 x 2 + 3 = 53 training runs, ~300 GPU-h (~39 of the 1132 remaining budget units
at the measured 0.13/GPU-h). Smoke first (1500 steps, seed 1, every cell, own results root), full batch
chained on `afterok`.

## 2. Implementation (this commit)

- `synthetic/src/attention/stieltjes_eager.py` (145 -> 130 lines) and
  `synthetic/src/kernels/adasplash/triton_stieltjes.py` (1458 -> 430 lines) rewritten as one windowed map
  `w = [(λ-s)^{-q} - c]_+`, c = d^{-q}; `c = 0` is the dense map bit-for-bit. Solver bracket
  `[(1+c)^{-1/q}, min(d, K^{1/q})]`; implicit-function gradient = the dense formula with `r` zeroed off
  the support. The Triton file lost the four gradient modes, Halley, per-row init, argmax tracking and
  the in-file tests; the single remaining mode is the one the repo has always trained with.
- Config: `++model.net.stieltjes_window=<d>` (0/absent = dense). Method names in `run_one.sh`:
  `stieltjes_q16`, `wstieltjes`, `aswstieltjes`, `entmax`.
- Verified locally (RTX 4070 Ti SUPER; `synthetic/tests/test_stieltjes_*.py`):
  - eager vs 200-step bisection: |p - ref| <= 5e-7 at N in {64..16k}, q in {1,2,4,8,16}, d in {1,2,4,inf};
    hand-written backward vs autograd through an 80-step unrolled fp64 solver: rel err <= 8e-10.
  - triton vs eager (fwd + dq, dk, dv), 106 configs incl. ALiBi, q in {4,16}, d in {1,1.5,2,inf}:
    fp32 <= 2.4e-6, bf16 <= 6e-3; unnormalised row-sum residual |S-1| <= 1.6e-5 up to N = 16k.
  - full attention layer, right/left-padded batches, all four variants incl. AS: triton == eager to
    fp32 3e-6 on outputs, dx and every parameter grad; kernel dispatches only on right-padded input.
  - 1500-step train + eval through `src/train.py`/`src/eval.py` for all five rows (sort, lr 3.2e-3).

## 3. Predictions (scored in the results report)

Headline cells, best-of-seeds under the validation protocol, ±4 pt SE:

| row | sort 4x | reverse 2x / 4x | copy 64x | mqmtar 256x / 1024x | conf |
|---|---|---|---|---|---|
| stieltjes_q16 | 0 | 40-90 / 0-20 | 40-70 | **70-90 / 30-60** (vs q=4: 63/12) | 65% |
| wstieltjes | 20-60 (first non-zero non-entmax cell) | 60-100 / 10-50 | 30-80 | 40-80 / 5-40 | 55% |
| aswstieltjes | 60-85 (~ASEntmax 80) | 100 / 40-90 | 70-90 | 80-99 / 40-95 | 55% |
| entmax | 40-70 (paper 57.8) | 90-100 / 10-40 (paper 28.5 at 4x) | 0-40 (paper 0.0) | 60-90 / 0-20 (paper 66.8/9.3) | 65% |
| asentmax mqmtar 4e-4 | | | | 90-99 / 60-95 (paper 99.0/95.3) | 70% |

Mechanism predictions:
- q=16 holds the MQMTAR tail better than q=4 (dispersion is the q=4 failure mode) but is the most
  seed-fragile row on sort/copy (sharper map; 1500-step local loss 0.35 vs 0.10 for q=4).
- Windowed matches entmax-1.5 within noise on every cell (same n-independent threshold; the (q,d) profile
  difference is second order). If it beats entmax anywhere by > 2 SE on all 3 seeds, that is the finding.
- AS-windowed ties ASEntmax within noise (proposal §2, last paragraph). No headroom on mqmtar 64x/sort 8x.
- Trainability: with the C^1 argument gone (proposal correction in this commit), windowed inherits
  entmax's plateau risk on mqmtar: expect 1 of 3 wstieltjes seeds and 0-1 of 3 aswstieltjes seeds to
  never escape at the chosen LR (the q=4 dense row escaped 3/3 at 4e-4).
- ~25% that any of the three new rows is qualitatively different from its entmax-family counterpart.

## 4. What would change the picture

- wstieltjes >> entmax on reverse/sort at equal LR: (q,d) profile matters; sweep d.
- q=16 >= AS rows on mqmtar 1024x: dense heavy tail + no scale beats sparse + scale on associative
  recall; that would be worth a 1K-sample re-eval and a q sweep.
- Any row with all 3 seeds on the mqmtar plateau: LR guess was wrong, not the method; add the 2x LR.

## 5. Jobs

(filled in at submission)
