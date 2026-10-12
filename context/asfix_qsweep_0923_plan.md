# ASEntmax-on-MQMTAR fix + q sweep — plan and predictions (2026-09-23)

Successor batch to `reproduction_0920.md`. Results go in a dated successor report. Submitted from f2d4339
(probe from 14341ce).

## 1. ASEntmax fix (job 7576491 x9, mqmtar, lr 2e-4, 3 seeds)

Diagnosis (0920 §3.4 + this commit): the adaptive scaler delta + beta (log n)^gamma is not ~1 at
initialisation. Under the HF init (Linear N(0, 0.02), bias 0) on RMSNorm'd inputs, beta = softplus(.)
has median 0.69 and gamma = 2 tanh(.) has sd 0.76, so at n = 64 the per-query scaler has median 1.7,
p95 5, max 20; at n = 4096 p95 11, max 64-74. Roughly 5% of (query, head) rows are near-hard-max from
step 0, entmax-1.5 gives those rows a one-token support, and a key at zero mass gets zero gradient; on
MQMTAR the first useful head has to find a content match among ~100 keys, so it never does. Plain entmax
(scaler = 1) escaped 1/2; ASEntmax 0/10 across 1e-4 .. 4e-4 and 10k/20k warmup.

Fix: `++model.net.attn_scale_zero_init=0.05` (method suffix `_zi`): projection weights 0, gamma bias 0,
beta bias softplus^-1(0.05), so scaler = 1.05 for every query at step 0; the scale is learned from there
(verified: both projections receive gradient). Same architecture at convergence; only the starting point
differs.

| arm | runs | what it tests |
|---|---|---|
| asentmax_zi_w20k | 3 | fix + the warmup that ever worked |
| asentmax_zi | 3 | fix alone at the paper's 10k warmup |
| entmax_w20k | 3 | plain-entmax control at the same LR/warmup (was 1/2 at 10k) |

Predictions (escape = train loss < 0.3 before 390k):
- asentmax_zi_w20k escapes 2-3 of 3 (70%); if it escapes, reaches >= 90 at 64x and >= 60 at 256x (65%).
- asentmax_zi at 10k escapes 1-2 of 3 (55%).
- entmax_w20k escapes 1-2 of 3 (60%).
- If zi escapes 0/3: the init is not the mechanism and the probe (below) decides what is.

Mechanism probe (job 7576554, done in 6 min): attention support size (keys/row with p > 0), max weight
and the AS scaler quantiles on 4 real MQMTAR batches for 9 existing last.ckpts.

| checkpoint | outcome | support/row, layers 0-3 | scaler p50 / p95 / max (worst layer) |
|---|---|---|---|
| asentmax_w20k s1 | stuck | 5.4 / 2.2 / 7.9 / 13.7 | 1.00 / 84 / 344 |
| asentmax 4e-4 s2 | stuck | 4.2 / 3.5 / 7.2 / 4.4 | 1.08 / 18 / 287 |
| aswstieltjes 10k s1 | stuck | 5.2 / 7.1 / 3.9 / 5.2 | 1.02 / 7 / 77 |
| aswstieltjes_w20k s1 | escaped | 4.6 / 6.2 / 3.0 / 2.3 | 1.03 / 2.5 / 76 |
| wstieltjes s3 | escaped | 4.0 / 2.1 / 2.5 / 1.2 | - |
| entmax s1 | stuck | 7.4 / 3.9 / 7.0 / 15.7 | - |
| entmax s2 | escaped late | 7.6 / 5.5 / 8.1 / 7.6 | - |
| stieltjes q4 s3 | escaped | 43 (dense) | - |
| stieltjes q16 s1 | stuck | 43 (dense) | - |

Scored: the prediction "stuck ASEntmax has support <= 2, escaped Stieltjes >= 5" is WRONG. Support does
not separate stuck from escaped for any map (2-8 keys/row throughout; the escaped windowed run is the
sparsest; dense q4 and q16 have identical full support with opposite outcomes). What does separate
ASEntmax is the scaler: on the stuck ASEntmax checkpoints a subset of queries has scale p95 18-84, max
287-344 (layer 1), i.e. some heads are hard-max and self-locked. The stuck and escaped AS-windowed runs
share a similar tail (max 76-77), so end-of-training scaler explosion alone is not proof of causation;
the asfix arms are the causal test. The trainability difference between the sparse maps (windowed q4
3/3 at 4e-4 vs entmax 1/2 at 2e-4) is not explained by support size and remains open.

## 2. q sweep (jobs 7576492-5, 9 elements each after cancelling the aswstieltjes_q4 duplicate; 36 runs)

q in {2, 4, 8} for stieltjes, wstieltjes (d = 2), asstieltjes (new row: dense + adaptive scale),
aswstieltjes; 1 seed, 1 LR per (task, row): dense/windowed at the swept q=4 optimum, AS rows at the
ASEntmax-matched values (sort 2e-4, reverse 4e-4, copy 1e-3, mqmtar 2e-4 + 20k warmup). q=4 cells for
stieltjes / wstieltjes / aswstieltjes come from 0913 / 0920; asstieltjes_q4 is new.

Single-seed, single-LR: rows will be reported as ranges over q, not as claims. Seed-1 values from the
3-seed rows show why: wstieltjes sort 2x is 85/89/93 across seeds (fine) but copy 64x is 25/3/4.

Predictions (test %, +-4 SE, and the 1-seed noise on top):
- q = 2 (interior alpha = 0.5, disperses as n^{1/2}): dense q2 below q4 on mqmtar 256x+ (75%) and on
  reverse 2x (60%); windowed q2 is the widest sparse map (support ~2-3x q4's) and the best windowed row on
  sort 4x, 10-30 (55%), still below entmax's 53.
- q = 8 dense: collapses on copy like q16 did, 0 by 8x (60%); mqmtar escapes but 256x <= q4 (60%); sort 2x
  <= 50 (65%).
- q = 8 windowed: sharpest sparse row, worst on sort/copy (65%), ~q4 on mqmtar 64x (55%).
- asstieltjes (new, dense + AS): the AS scale on a dense map sharpens it; within noise of dense q4 on
  mqmtar, above dense on sort 4x (10-30 vs 0) (55%).
- Ordering by q on sort 4x for the windowed rows: q2 > q4 > q8 (60%). On mqmtar 256x: q4 >= q8 > q2 (55%).

## 3. Cost

asfix 9 x ~10 h = ~90 GPU-h; qsweep 36 x ~4.5 h = ~165 GPU-h; probe ~0.5 h. Total ~255 GPU-h
(~33 budget units; ~1030 remain). Wall ~2-3 days.

## 4. Implementation (f2d4339)

- `sparse_gemma.py`: `attn_scale_zero_init` (config, default 0 = HF init); `_zero_init_scale()` runs after
  `post_init()` in `SparseGemma2ForCausalLM`.
- `run_one.sh`: method name suffixes compose: `<row>[_q<N>][_zi][_w20k]`; q parsed from the name for the
  four Stieltjes rows (default 4).
- `submit_variants.sh asfix | qsweep`.
- `smoke_stieltjes_local.sh` now evals run_one.sh's method block directly (no hand-copied overrides).
- `tests/test_stieltjes_triton_alibi.py`: q in {2, 8} dense and windowed added (190/190 OK).
- `oneoff/attn_support_probe.py` + `attn_probe.sbatch`.
