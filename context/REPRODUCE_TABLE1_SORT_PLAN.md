# Plan: Reproduce Table 1 — Sort column (all 5 methods)

Paper: "Long-Context Generalization with Sparse Attention" (arXiv:2506.16640)
Target: the **Sort (L=2)** block of Table 1 — exact-match accuracy (%) at ID / 2x / 4x / 8x.

## SCOPE DECISION (user, this session)
**Full-fidelity on a 2-method subset: Softmax vs ASEntmax**, each at the full/near-full
312,500-step schedule. ~5-7h total for the two → fits one overnight window with margin.
The other three methods (Top-K, SSMax, Entmax) are DEFERRED. The two chosen methods are the
paper's headline contrast (dense softmax collapses OOD; adaptive-sparse ASEntmax generalizes).

Expected target cells:
| Method    | ID (64) | 2x (128) | 4x (256) | 8x (512) |
|-----------|---------|----------|----------|----------|
| Softmax   | 100.0   | 0.0      | 0.0      | 0.0      |
| ASEntmax  | 100.0   | 100.0    | 79.7     | 0.0      |

## Paper's reported numbers (what we are trying to reproduce)

| Method        | ID (64) | 2x (128) | 4x (256) | 8x (512) |
|---------------|---------|----------|----------|----------|
| Softmax       | 100.0   | 0.0      | 0.0      | 0.0      |
| Top-K (K=32)  | 100.0   | 92.5     | 0.0      | 0.0      |
| SSMax         | 100.0   | 0.0      | 0.0      | 0.0      |
| Entmax        | 100.0   | 99.3     | 57.8     | 0.0      |
| ASEntmax      | 100.0   | 100.0    | 79.7     | 0.0      |

Note: OOD dies quickly here (everything is 0 by 8x except a partial for Entmax/ASEntmax at 4x),
which is exactly why Sort was chosen for an overnight/low-VRAM reproduction.

## Task / data facts (verified from code)

- Generator: `scripts/generate_data.py --task_type sort` → `Sort.sample_func`:
  src = random ints in [0, vocab_size), trg = sorted(src). vocab_size=32.
- Train length: seq_len=48, vary_len=16 → lengths sampled uniform in [32, 64]. "ID" = n=64 (max train len).
- Eval lengths (MDPS, default): 64 128 256 512 1024 2048 4096 with vary_len 0 (fixed length each).
  → multipliers 64=ID, 128=2x, 256=4x, 512=8x, 1024=16x, 2048=32x, 4096=64x.
  Table 1 only reports up to 8x (512) for Sort.
- Data dir name expected by config: `${DATA_PATH}/sort/40M-tr_32-64_vt_64-4096_vocab_32/`
- Tokenizer: SimpleTokenizer(max_vocab_id=31) → 32 data tokens + 5 specials (pad/bos/eos/mask/sep) = 37 vocab.
- Model (sort.yaml overrides): SparseGemma2, 2 layers, d=256, 8 heads, head_dim=32, intermediate=1024.
  Positional encoding: NAPE (apply_nape=True, apply_rotary=False) per README reproduction commands.
- Checkpoint selection metric: `val/bleu/dataloader_idx_2` (= val length 256 = 4x), mode=max.
- Loss on target tokens only; eval = autoregressive generation + exact-match on decoded string.

## The 5 method configs (Hydra overrides on top of `experiment=entmax/sort`)

All use NAPE. Base experiment already sets architecture. Differences:

1. Softmax (baseline):
   `model.net.entmax_alpha=1.0 ++model.net.attn_scale_type=null`
   (paper impl uses flash_attention_2; see "eager vs flash" below)

2. Top-K (K=32):
   `++model.net.attn_type=topk ++model.net.topk_size=32 model.net.entmax_alpha=1.0`
   (topk path uses softmax over top-32 logits + NAPE alibi bias; eager AttentionNoCache)

3. SSMax (scalable softmax, Nakanishi 2025):
   `model.net.entmax_alpha=1.0 ++model.net.attn_scale_type="nakanishi"`

4. Entmax (sparse, no length scaling):
   `model.net.entmax_alpha=1.5 ++model.net.attn_scale_type=null`

5. ASEntmax (this paper — sparse + adaptive length scaling):
   `model.net.entmax_alpha=1.5 ++model.net.attn_scale_type="adapt-softplus-tanh" ++model.net.attn_scale_proj_bias=True`

Common flags: `++model.net.apply_rotary=False ++model.net.apply_nape=True`

## Batch sizes (RTX 4070 Ti Super, 16 GB) — model is tiny (~few M params)

- Train batch: keep 128 (data/sort.yaml). Memory is trivial for a 2-layer d=256 model at len<=64.
- Val batch: 32 (default). Test batch: 8 (default).
- These are safe at 16 GB. Autoregressive gen at 8x (512) with batch 8 is fine (small KV cache).
- If any OOM appears at longer OOD during exploration, drop test batch to 4. Not expected <=8x.

## STATUS: READY TO RUN (validated end-to-end this session)

Environment built + verified:
- uv venv at synthetic/.venv, torch 2.5.1+cu121, RTX 4070 Ti Super (cap 8.9), 15.6 GB free.
- flash-attn 2.6.3 via PREBUILT wheel (torch2.4+cu123+cxx11abiFALSE+cp312, ABI-compatible) —
  forward pass verified. No source build / no nvcc needed. (venv-local nvcc was unnecessary.)
- adasplash (Triton) verified running for ASEntmax; entmax/lightning/transformers 4.46.3 all import.
- HF login already done by user; gated google/gemma-2-2b config pulls fine (config only, weights random-init).

Pipeline validated:
- Both models train → checkpoint → eval, EXIT 0 (smoke + probe + full-orchestrator dry runs).
- Per-length eval via file_subset isolates one length; parser reads test/acc_epoch correctly.
- Early-stop ladder, table assembly, resume-skip logic all exercised.

Measured throughput: ~37 steps/s. Full 312,500-step schedule ≈ 2.35h/model → ~4.7h for both + eval.
Full paper fidelity retained (no step reduction needed). Data index (.idx) cached → no rebuild.

Code fixes applied this session (faithful, minimal):
- src/train.py:96 — wrapped `cfg.trainer.pop('matmul_precision')` in `open_dict` (struct-mode bug).
- configs/eval.yaml — allow `experiment=` composition; eval uses `+seed=4` (no top-level seed key).
- Installed hydra-colorlog (config referenced it but it was missing from requirements).

Runner: synthetic/run_overnight_table1_sort.sh  (MAX_STEPS and RESULTS_DIR env-overridable).
Results land in synthetic/results_table1_sort/{softmax,asentmax}/ with run.log + ladder.tsv,
and a final side-by-side vs-paper table printed to run.log.

## Requested deviations from paper (per user) — IMPLEMENTED

1. Cap eval sets to **100 samples** (default dev/test = 1000): `--dev_size 100 --test_size 100`.
2. **Early-stop OOD ladder**: once a model hits 0.0 exact-match at a length, do NOT eval the next
   longer length for that model. The framework tests ALL length-dataloaders in one `trainer.test()`
   pass, so this needs a small orchestration wrapper: evaluate length-by-length (ascending) via the
   data provider's `file_subset`, stop the ladder at the first 0.0. (Table shows Sort only to 8x, so
   the ladder is 64→128→256→512 with a hard stop at 512 regardless.)
3. Keep code + data generation otherwise faithful to the paper.

## Known blockers / decisions (RESOLVED this session)

B1. **Gated Gemma config**: RESOLVED → user doing HF login + accepting google/gemma-2-2b license.
    Only config.json is used (weights random-init). Token persists to ~/.cache/huggingface/token.

B2. **flash-attn 2.6.3**: RESOLVED → real flash_attention_2 for Softmax/SSMax. CUDA via Option A:
    venv-local `nvidia-cuda-nvcc-cu12==12.1.105`, CUDA_HOME pointed at it, MAX_JOBS=4 to bound RAM.
    Check for a prebuilt wheel (torch2.5.1/cu12/py3.12/cxx11abi) first; fall back to source build.

B3. **entmax/asentmax kernels**: fast adasplash (Triton, runtime-compiled, no nvcc) for ASEntmax.
    Fall back to eager entmax_bisect if kernels misbehave.

B4. **W&B**: use CSV logger (`logger=csv`) for the unattended run; parse metrics.csv.

B5. **Env**: build uv venv from synthetic/requirements.txt. evaluate.load("exact_match") needs net
    on first use.

## The big one — TRAINING TIME BUDGET (needs your call)

- Full paper run per model = 40M-sample epoch = **312,500 steps** (warmup 20k), batch 128.
- 5 models. Even at an optimistic 25–35 steps/s for this tiny model, that is ~2.5–3.5 h *per model*
  → ~13–18 h total for the 5, before eval. That likely overruns a single overnight window and leaves
  no margin. Data generation of 40M samples adds ~1 h and ~10 GB.
- Options to fit overnight (ranked):
    (i)  Reduce to a fixed step budget per model (e.g. 40k–80k steps) with proportionally smaller
         train set + rescaled warmup/cosine. Sort converges to 100% ID fast; OOD ranking usually
         emerges well before 312k steps. Deviates from paper's exact step count.
    (ii) Full fidelity on a subset of methods (e.g. Softmax vs ASEntmax as the headline contrast),
         each at full/near-full steps.
    (iii) Full 312.5k on all 5 across ~2 nights.
- I recommend (i) with ~60k steps/model as the overnight target, clearly labeled as a
  reduced-budget reproduction, then compare the ranking + OOD-collapse pattern to the paper.

## Execution outline (once decisions are made)

1. `uv venv` + install requirements; export PROJECT_ROOT, DATA_PATH; set logger=csv.
2. Patch B1 (local Gemma2Config) and choose eager vs fast (B2/B3).
3. Generate sort data: train (size per budget), dev/test 100 samples each at lengths 64..512(..4096).
4. Train 5 models sequentially (background, notify_on_complete), checkpoint on val 4x bleu.
5. Eval ladder per model: length-by-length ascending, stop at first 0.0 exact-match (cap 8x).
6. Assemble results into the Table 1 Sort block; compare to paper; write a short report.

## Open questions for the user
- Q1 (blocker): approve local-config patch for Gemma (B1b) vs HF login (B1a)?
- Q2 (blocker): run everything eager to skip flash-attn (B2), yes?
- Q3 (budget): pick a training budget — reduced-step all-5 (recommend ~60k), full-fidelity subset,
  or multi-night full run?
- Q4: fast adasplash kernels for entmax, or pure-PyTorch eager entmax for maximum robustness?
