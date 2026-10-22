# LM plan (Oct 12): Stieltjes variants vs softmax on FineWeb, nanoGPT

Code: github jtaylor-eng/nanoGPT branch `stieltjes_lm` (submodule `submodules/nanoGPT`), commit 85452bc.
Kernel + eager reference vendored verbatim from this repo @ c7093f0 (`nanoGPT/stieltjes/`).
OSC: checkout `/fs/scratch/PAS2836/jacktaylor/nanoGPT`, results `$SCRATCH/nanogpt_results/lm1012/<row>/lr<lr>/`
(`metrics.jsonl`, `final.json`, `lengths.json`, `run.log`, `ckpt.pt`). Jobs: probe 7796843, array 7796844_[0-11]
(afterany:probe).

## Question
Synthetic tasks: only windowed AS Stieltjes (q2, d2/4) competes with ASEntmax (MQMTAR); dense q4 ~ softmax.
In LM: (1) do Stieltjes rows cost loss at the train length? (2) does the windowed map degrade less beyond it?

## Protocol (decisions a-d recommended; e = 15 Newton sweeps iff probe residual < 1e-4)
- Model 8L x 8H x 512 (head dim 64), no bias, no dropout, 50.9M params incl. tied embedding.
  NAPE positions for all rows (heads 0-3 ALiBi slopes 1/1..1/4, heads 4-7 NoPE; no wpe) as in the synthetic runs.
- Data: FineWeb-10B GPT-2 tokens (HF kjj0/fineweb10B-gpt2), train = shards 1-10 (1.0B tok), val = val shard (100M).
- T = 512; 32 x 512 x 8 accum = 131k tok/step; 7630 steps = 1.0B tokens (~20 tok/param).
- AdamW (0.9, 0.95), wd 0.1, clip 1.0, warmup 400, cosine to lr/10, bf16 autocast, torch.compile.
- Rows: softmax | stieltjes_q4 (dense) | wstieltjes_q4_d2 | aswstieltjes_q2_d4_zi (AS zero-init eps 0.05).
- LR grid {7.5e-4, 1.5e-3, 3e-3}, seed 1337. 12 runs.
- Selection: final val loss at T=512 (`final.json`, 800 batches = 13M tokens). Last checkpoint only.
- Reported, never selected on: `eval_lengths.py` val loss at 512/1024/2048/4096 on the same 2M val tokens
  (non-overlapping windows from the start of val.bin), loss by 16 position buckets, and per-layer/head attention
  entropy / exact support / max prob (all rows and last-quarter rows) at 512 and 4096.

## Probe (7796843, before the array)
Per row: ms/step at the real batch (num_iter 30 and 15). Per Stieltjes row: 800 training steps, then
max |S-1| (solver residual before normalisation) at num_iter 10/15/30 at T=512, 4096 on real val text.
`osc/probe_decide.py` writes `lm1012/NUM_ITER` (15 iff max residual @15 < 1e-4 in all 6 row x length cells).
Local pre-check (300-step checkpoint, stieltjes_q4, 4070 Ti): max |S-1| = 3.0e-5 at 15 and at 30, i.e. the Newton
solve converged before 15; val loss identical to 5 decimals at 10/15/30.

## Local verification (done)
- `tests/test_attn_paths.py`: training path (SDPA+NAPE mask / fused Triton) vs materialised eager path, all 5 rows
  (incl. softmax_as_zi), T in {64, 300}: fp32 rel err out 2e-7..9e-7, grads <= 4e-6; bf16 <= 4e-2. ALL OK.
- 300-step runs (bs 8) with compile: softmax / stieltjes_q4 / aswstieltjes_q2_d4_zi all learn (10.9 -> 6.5-6.6).
  Step time 29 ms vs 44-45 ms (Stieltjes 1.5x softmax at T=512, num_iter 30).
- Memory: micro-batch 32 needs 8.2 GB softmax, Stieltjes ~14 GB (64 OOMs a 16 GB card) -> 32 x 8 accum.

## Predictions (before compute; score in reproduction_LM_<MMDD>.md)
| # | prediction | conf |
|---|---|---|
| 1 | val@512: every Stieltjes row within 0.05 nats of softmax; softmax best or tied | 60% |
| 2 | dense q4 best Stieltjes row at 512 | 55% |
| 3 | wstieltjes_q4_d2 >= 0.02 worse than dense at 512 | 55% |
| 4 | smallest rise val@4096 - val@512: aswstieltjes_q2_d4_zi | 55% |
| 5 | largest rise at 4096: dense q4 | 60% |
| 6 | A100 Stieltjes step time 3-6x softmax (local 4070 Ti measured 1.5x, so now ~30%) | 30% |
| 7 | at least one Stieltjes row diverges / NaNs at 3e-3 | 40% |
| 8 | probe picks num_iter 15 | 85% |

## Not run (options for later)
softmax_as_zi (separates scale from map; row is implemented and tested), asentmax (needs adasplash),
seeds at the selected LR, longer T / more tokens.
