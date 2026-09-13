# AGENTS.md — entrypoint for AI agents working in this repo

Fork of [deep-spin/asentmax](https://github.com/deep-spin/asentmax) (paper: *Long-Context
Generalization with Sparse Attention*, arXiv 2506.16640). Two goals:

1. Reproduce the paper's Table 1 (Sort / Reverse / Copy / MQMTAR length-generalization ladders)
   for Softmax and ASEntmax under the paper's protocol on OSC A100s.
2. Add and evaluate a new attention row, **Stieltjes** (q=4): a heavy-tailed simplex map
   `p_i = 1/(lambda_q - x_i)^q`, with `lambda_q` found by a 1-D root solve. Motivation and the two
   regularizer families are in `context/stieltjes_proposal.tex` (one page; Overleaf boilerplate stripped).

Branch of record: `torch_triton_comp` (remote `fork` = github `jtaylor-eng`; `origin` = upstream deep-spin).
Fork owner: Jack Taylor. Work style: analysis/report first, then explicit go-ahead; predictions and a
confidence estimate *before* OSC compute is spent; results reported as tables with paper numbers alongside.

## Read these first (in order)

| file | status |
|---|---|
| `context/reproduction_0907.md` | **CURRENT** results, findings (§3), selection protocol (§4), next steps (§6) |
| `context/reproduction_0907_all_runs.md` | appendix to the above: every (task, method, seed, lr) run |
| `context/TABLE1_REPRODUCED_FIXED.md` | **SUPERSEDED** — local single-seed last.ckpt pass; kept for provenance only |
| `context/*.pdf` | reference papers (gitignored; present locally) |

Data-generation recipes: root `README.md` (upstream) and `experiments/osc/datagen.sbatch` (exact args used for our four datasets, incl. the 100-sample val/test cap and the trimmed validation splits).

Results reports are dated snapshots: **do not append new runs to `reproduction_0907.md`**. Write a
dated successor (`context/reproduction_<MMDD>.md`, same section layout), mark the old one superseded
in this table, and keep the all-runs appendix pattern.

## Layout

```
attention.py                  upstream torchtitan attention module — untouched, do not edit
README.md                     upstream README (data-generation recipes) — do not edit
scripts/generate_data.py      dataset generator (used by experiments/*/gen_* and osc/datagen.sbatch)
synthetic/                    Lightning + Hydra training tree (the thing that actually runs)
  src/train.py, src/eval.py   entrypoints; configs/experiment/entmax/<task>.yaml per task
  src/models/architectures/sparse_gemma.py   attention dispatch: attn_type in {regular, topk, stieltjes,...}
  src/attention/stieltjes_eager.py           reference dense Stieltjes normaliser (O(N^2) fp32)
  src/kernels/adasplash/triton_stieltjes.py  fused Triton Stieltjes attention (ALiBi/NAPE bias, smem-aware tiles)
  tests/test_stieltjes_*.py                  eager/triton/layer equivalence tests (need a GPU for triton)
  configs/, requirements.txt, README.md      upstream (fork edits: eval.yaml, experiment/*, sparse_gemma.yaml, +logger/csv_wandb_offline.yaml, src/data/file.py lock-safe .idx build)
  data/ logs/ results_*/ .venv/              gitignored local artifacts (data is ~34 GB)
experiments/                  fork-added, non-library
  osc/                        the OSC pipeline: env.sh (all paths) · setup_env.sh (one-time venv) · submit.sh -> array_worker.sbatch -> run_one.sh ·
                              datagen.sbatch · progress.sh · aggregate.py / reselect.py (result tables; per-run selection decision) · submit_impl_cmp.sh
  osc/oneoff/                 single-use smoke / diagnostic sbatch scripts, kept for reference
  tests/smoke_stieltjes_local.sh   ~1 min local end-to-end (train 60 steps + eval) on the sort data
  hf/gemma-2-2b/config.json   vendored HF config so no network/token is needed (model weights are random-init)
context/                      papers, proposal, reports (see table above)
```

## Stieltjes implementation facts

- Enabled via `++model.net.attn_type=stieltjes ++model.net.stieltjes_q=4.0 ++model.net.stieltjes_num_iter=30
  ++model.net.stieltjes_impl={triton|eager}` with `attn_implementation=eager use_fast_attn=False`.
- Default `stieltjes_impl` is **triton** (`sparse_gemma.py`). Triton is used for training and unpadded prefill;
  decode and left-padded batches fall back to eager. `stieltjes_eager` as a *method name* in `run_one.sh`
  forces eager everywhere (OOMs at ~4k prefill on 40 GB; kept only for the impl comparison).
- fp32 D=64 uses 64x64 tiles on A100-class smem (128x64 backward exceeds 166,912 B).
- Results before Sep 10 (everything in `reproduction_0907.md`) were produced with the eager path, which is
  why the Stieltjes MQMTAR ladder there stops at 64x and Copy 64x is OOM. Those cells are the open question.

## Method overrides (Hydra, all NAPE; canonical copy lives in `experiments/osc/run_one.sh`)

```
common:    ++model.net.apply_rotary=False ++model.net.apply_nape=True
softmax:   model.net.entmax_alpha=1.0 ++model.net.attn_implementation=flash_attention_2 ++model.net.use_fast_attn=True ++model.net.attn_scale_type=null
asentmax:  model.net.entmax_alpha=1.5 ++model.net.attn_implementation=eager ++model.net.use_fast_attn=True ++model.net.attn_scale_type=adapt-softplus-tanh ++model.net.attn_scale_proj_bias=True
stieltjes: model.net.entmax_alpha=1.0 ++model.net.attn_type=stieltjes ++model.net.stieltjes_q=4.0 ++model.net.stieltjes_num_iter=30 ++model.net.stieltjes_impl=triton ++model.net.attn_implementation=eager ++model.net.use_fast_attn=False ++model.net.attn_scale_type=null
(not run yet: topk  ++model.net.attn_type=topk ++model.net.topk_size=32 alpha=1.0;  ssmax alpha=1.0 attn_scale_type=nakanishi;  entmax alpha=1.5 attn_scale_type=null)
```

Per-task data paths, step budgets, ladder lengths and LR grids: `run_one.sh` (case block) and `submit.sh`.

## OSC (Ohio Supercomputer Center, cluster Ascend, project PAS2836)

- Repo copy: `/fs/scratch/PAS2836/jacktaylor/asentmax`. Synced by **git bundle**, not push:
  `git bundle create` locally -> scp to `$SCRATCH/asentmax.bundle` -> on OSC `git fetch origin && git reset --hard origin/<branch>`.
  (https fetch fails on OSC; SSH remotes only.)
- `experiments/osc/env.sh` defines every path (`REPO`, `VENV`, `DATA_ROOT`, `RESULTS_ROOT`, caches on scratch). Source it.
- `$SCRATCH` is not exported by default: `export SCRATCH=/fs/scratch/PAS2836/$USER` in every shell, including inside sbatch.
- `submit.sh` refuses a dirty tree — commit before submitting. Manifests + slurm logs go under `$RESULTS_ROOT/{manifests,slurm}`.
- 1-GPU jobs land on A100-PCIE-40GB (80 GB only on 4-GPU `quad` nodes). Run dirs: `$RESULTS_ROOT/<task>/<method>/s<seed>_lr<lr>/` with `ladder.tsv`, `ladder_last.tsv`, `run.log`, `checkpoints/`.
- Datasets (~35 GB) live in `DATA_ROOT` and are staged to `$TMPDIR` by `run_one.sh` (random seeks on GPFS are ~20x slower).
- `experiments/osc/progress.sh` on a login node = step / it/s / loss / status for all runs.

## Invariants — do not break

- Selection is validation-only (`reproduction_0907.md` §4): primary = val exact-match @8x (sort: BLEU @4x); BLEU fallback when the primary is uninformative; never pick runs by test score.
- Eval sets are 100 samples/length (paper: 1K) -> ±4 pt standard error at p=0.8. Say so when comparing to paper cells.
- Early-stop ladder: after an exact 0.0 at a length, longer lengths are `skip`, not 0.
- Keep the upstream files (`README.md`, `attention.py`, `synthetic/src/**` other than the Stieltjes files and the few listed edits) as close to `origin/main` as possible so upstream merges stay clean.
- Never commit under `synthetic/data`, `synthetic/logs`, `synthetic/results_*`, `*.ckpt`, `context/*.pdf` (all gitignored).

## Local dev

- `synthetic/.venv` (uv; torch 2.5.1+cu121, flash-attn 2.6.3 prebuilt, adasplash). `source synthetic/.venv/bin/activate`.
- Env expected by the synthetic tree: `PROJECT_ROOT=<repo>/synthetic`, `DATA_PATH=<repo>/synthetic/data`, `HF_HUB_OFFLINE=1`.
- Sanity: `bash experiments/tests/smoke_stieltjes_local.sh`; kernel tests are plain scripts (no pytest installed): `cd synthetic && python3 tests/test_stieltjes_eager.py` (CPU ok), `test_stieltjes_triton_alibi.py` / `test_stieltjes_impl_layer.py` (CUDA).
- Use `python3` (no `python` on this host); PEP 668 -> venv/uv only.
