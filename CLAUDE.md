@AGENTS.md

# Additions (Nov 2026)
- Project brief: `../../CLAUDE.md` (Stieltjes). This repo now lives at `~/Desktop/research/Stieltjes/submodules/asentmax`; `~/Desktop/projects/...` paths in older reports are the same files.
- `context/` here is the archive of every dated report, including the LM ones (`reproduction_lm_*.md`) whose code is in `../nanoGPT`.
- research-loop directions use `git push fork` + `git fetch` on the OSC login node, not the bundle method above; per-direction checkouts live under `/fs/scratch/PAS2836/jacktaylor/wt/asentmax/<slug>`.
- Current local branch: `lr_sweep` (registry base branch for directions). `torch_triton_comp` is the branch of record named in AGENTS.md; check `git log` before assuming which is ahead.
