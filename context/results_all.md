# Combined best results — all simplex-map variants (updated from reproduction_*.md files)

**Maintenance note**: whenever a new `reproduction_<MMDD>.md` report is completed, update this file
by reading all reproduction_*.md files and re-extracting the best result per variant row. The AGENTS.md
table-of-contents block and the source column below must reflect the new report. Never hand-edit cells
without also checking whether the source report has superseded them.

Standard error: ±4 pts at p=0.8 with 100 test samples/length. "best" = the validation-selected run
(best val score over seeds x LRs for that row), per the protocol in `reproduction_0913.md §6`.
LR column = the LR of the selected run. "skip" = early-stopped (exact 0.0 at preceding length).
"-" = not evaluated (OOM or not run). Source = the reproduction_*.md that supplies this row.

---

## Sort (train L=2; informative lengths: 2x, 4x)

| variant | q | d | scale | best LR | ID | 2x | 4x | 8x | source |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| softmax | — | — | none | 6.4e-3 | 100 | 99 | 4 | 0 | 0913 |
| entmax-1.5 | — | — | none | 4e-4 | 100 | 98 | 53 | 0 | 0920 |
| asentmax (ASEntmax) | — | — | adapt | 2e-4 | 100 | 100 | 81 | 0 | 0907 |
| stieltjes (dense) | q=4 | — | none | 3.2e-3 | 100 | 88 | 0 | skip | 0913 |
| stieltjes (dense) | q=8 | — | none | 3.2e-3 | 41 | 1 | 0 | — | 0929 §2 |
| stieltjes (dense) | q=16 | — | none | 3.2e-3 | 91 | 22 | 0 | skip | 0920 |
| wstieltjes (windowed) | q=4 | d=2 | none | 1.6e-3 | 100 | 93 | 7 | 0 | 0920 |
| wstieltjes (windowed) | q=2 | d=2 | none | 1.6e-3 | 97 | 34 | 0 | — | 0929 §2 |
| wstieltjes (windowed) | q=8 | d=2 | none | 1.6e-3 | 92 | 27 | 1 | 0 | 0929 §2 |
| asstieltjes (dense+AS) | q=4 | — | adapt | 2e-4 | 31 | 0 | — | — | 0929 §2 |
| asstieltjes (dense+AS) | q=2 | — | adapt | 2e-4 | 49 | 0 | — | — | 0929 §2 |
| asstieltjes (dense+AS) | q=8 | — | adapt | 2e-4 | 100 | 98 | 2 | 0 | 0929 §2 |
| aswstieltjes (windowed+AS) | q=4 | d=2 | adapt | 4e-4 | 100 | 99 | 32 | 0 | 0920 |
| aswstieltjes (windowed+AS) | q=2 | d=2 | adapt | 2e-4 | 100 | 100 | 62* | 0 | 0929 §8.1 |
| aswstieltjes (windowed+AS) | q=8 | d=2 | adapt | 4e-4 | 100 | 96 | 1 | 0 | 0929 §2 |

*aswstieltjes q=2 sort 4x = 62 is a single-seed outlier; 3-seed mean = 21 ± 29. Do not cite as the row value.

**Paper reference (1K samples, best of 3 seeds x LRs):** Softmax 100/0/0; Entmax 100/100/57.8/0;
ASEntmax 100/100/79.7/0.

---

## Reverse (train L=6; informative lengths: 1.5x, 2x, 4x)

| variant | q | d | scale | best LR | ID | 1.5x | 2x | 4x | 8x | source |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| softmax | — | — | none | 1.6e-3 | 100 | 22 | 0 | skip | skip | 0913 |
| entmax-1.5 | — | — | none | 4e-4 | 100 | 100 | 99 | 61 | 0 | 0920 |
| asentmax (ASEntmax) | — | — | adapt | 4e-4 | 100 | 100 | 100 | 53 | 0 | 0907 |
| stieltjes (dense) | q=4 | — | none | 3.2e-3 | 100 | 100 | 38 | 0 | skip | 0913 |
| stieltjes (dense) | q=8 | — | none | 3.2e-3 | 100 | 100 | 96 | 0 | — | 0929 §2 |
| stieltjes (dense) | q=16 | — | none | 3.2e-3 | 100 | 94 | 8 | 0 | skip | 0920 |
| wstieltjes (windowed) | q=4 | d=2 | none | 1.6e-3 | 100 | 100 | 100 | 49 | 0 | 0920 |
| wstieltjes (windowed) | q=2 | d=2 | none | 1.6e-3 | 100 | 100 | 99 | 3 | 0 | 0929 §2 |
| wstieltjes (windowed) | q=8 | d=2 | none | 1.6e-3 | 100 | 100 | 53 | 0 | — | 0929 §2 |
| asstieltjes (dense+AS) | q=4 | — | adapt | 4e-4 | 100 | 88 | 0 | — | — | 0929 §2 |
| asstieltjes (dense+AS) | q=2 | — | adapt | 4e-4 | 100 | 17 | 0 | — | — | 0929 §2 |
| asstieltjes (dense+AS) | q=8 | — | adapt | 4e-4 | 100 | 97 | 52 | 0 | — | 0929 §2 |
| aswstieltjes (windowed+AS) | q=4 | d=2 | adapt | 4e-4 | 100 | 100 | 100 | 0 | skip | 0920 |
| aswstieltjes (windowed+AS) | q=2 | d=2 | adapt | 4e-4 | 100 | 100 | 100 | 14 | 0 | 0929 §8.1 |
| aswstieltjes (windowed+AS) | q=8 | d=2 | adapt | 4e-4 | 100 | 97 | 46 | 0 | — | 0929 §2 |

Note: all reverse rows selected by val BLEU@4x (8x monitor is identically 0); last.ckpt used.
The val BLEU proxy is noisy and may mis-rank seeds within a row (see 0913 §4.3). q=8 dense reaches
96 at 2x — the only task where sharper interior helps the dense map.

**Paper reference:** Softmax 100/36/0/0/0; Entmax 100/100/93.5/28.5/2.5; ASEntmax 100/100/99.8/96.4/56.7.

---

## Copy (train L=2; informative lengths: 16x, 32x, 64x)

| variant | q | d | scale | best LR | ID | 8x | 16x | 32x | 64x | source |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| softmax | — | — | none | 1e-3 | 100 | 100 | 98 | 91 | 63 | 0913 |
| entmax-1.5 | — | — | none | 5e-4 | 100 | 99 | 86 | 50 | 0920 |
| asentmax (ASEntmax) | — | — | adapt | 5e-4 | 100 | 100 | 96 | 76 | 0907 |
| stieltjes (dense) | q=4 | — | none | 5e-4 | 100 | 99 | 87 | 52 | 0913 |
| stieltjes (dense) | q=2 | — | none | 5e-4 | 0 | — | — | — | — | 0929 §2 |
| stieltjes (dense) | q=8 | — | none | 5e-4 | 75 | 31 | 1 | 0 | — | 0929 §2 |
| stieltjes (dense) | q=16 | — | none | 5e-4 | 90 | skip | skip | skip | skip | 0920 |
| wstieltjes (windowed) | q=4 | d=2 | none | 5e-4 | 100 | 96 | 69 | 25 | 0920 |
| wstieltjes (windowed) | q=2 | d=2 | none | 5e-4 | 97 | 85 | 60 | 10 | — | 0929 §2 |
| wstieltjes (windowed) | q=8 | d=2 | none | 5e-4 | 85 | 39 | 0 | — | — | 0929 §2 |
| asstieltjes (dense+AS) | q=4 | — | adapt | 1e-3 | 100 | 90 | 14 | — | — | 0929 §2 |
| asstieltjes (dense+AS) | q=2 | — | adapt | 1e-3 | 0 | — | — | — | — | 0929 §2 |
| asstieltjes (dense+AS) | q=8 | — | adapt | 1e-3 | 90 | 51 | 6 | 0 | — | 0929 §2 |
| aswstieltjes (windowed+AS) | q=4 | d=2 | adapt | 1e-3 | 100 | 99 | 59 | 12 | 0920 |
| aswstieltjes (windowed+AS) | q=2 | d=2 | adapt | 1e-3 | 96 | 89 | 49 | 4 | — | 0929 §2 |
| aswstieltjes (windowed+AS) | q=8 | d=2 | adapt | 1e-3 | 79 | 32 | 1 | 0 | — | 0929 §2 |

**Paper reference (1K samples):** Softmax 99.9/99.4/96.1/85.5; Entmax 99.9/99.7/99.4/96.3/0.0;
ASEntmax 99.9/99.7/99.4/96.3/86.6.

---

## MQMTAR (train L=4; informative lengths: 64x, 256x, 1024x)

Escape = whether the run escaped the MQMTAR training plateau (loss < 0.3).
20k = 20k-step warmup (paper uses 10k; required for some rows). All non-_w20k rows use 10k warmup.

| variant | q | d | scale | warmup | best LR | 16x | 64x | 256x | 1024x | escape | source |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| softmax | — | — | none | 10k | 2e-4 | 100 | 93 | 42 | 0 | 3/3 | 0913 |
| entmax-1.5 | — | — | none | 10k | 2e-4 | — | — | — | — | 1/2 | 0920 |
| asentmax (ASEntmax) | — | — | adapt | 10k | 2e-4 | 0 | skip | skip | skip | 0/10 | 0907/0920 |
| asentmax_zi_w20k | — | — | adapt+zi | 20k | 2e-4 | 100 | 96 | 77 | 38 | 3/3 | 0929 §1 |
| stieltjes (dense) | q=4 | — | none | 10k | 4e-4 | 99 | 94 | 63 | 12 | 3/3 | 0913 |
| stieltjes (dense) | q=2 | — | none | 10k | 4e-4 | 1 | 0 | skip | skip | partial | 0929 §2 |
| stieltjes (dense) | q=8 | — | none | 10k | 4e-4 | 38 | 3 | 0 | skip | partial | 0929 §2 |
| stieltjes (dense) | q=16 | — | none | 10k | 4e-4 | 1 | 0 | skip | skip | 1/3 | 0920 |
| wstieltjes (windowed) | q=4 | d=2 | none | 10k | 4e-4 | 98 | 91 | 48 | 6 | 3/3 | 0920 |
| wstieltjes (windowed) | q=2 | d=2 | none | 10k | 4e-4 | 94 | 77 | 28 | 0 | partial | 0929 §2 |
| wstieltjes (windowed) | q=8 | d=2 | none | 10k | 4e-4 | 43 | 2 | 0 | skip | partial | 0929 §2 |
| asstieltjes (dense+AS) | q=4 | — | adapt | 20k | 2e-4 | 98 | 93 | 36 | 0 | 1/1 | 0929 §2 |
| asstieltjes (dense+AS) | q=2 | — | adapt | 20k | 2e-4 | 0 | skip | skip | skip | 0/1 | 0929 §2 |
| asstieltjes (dense+AS) | q=8 | — | adapt | 20k | 2e-4 | — | — | — | — | never | 0929 §2 |
| aswstieltjes (windowed+AS) | q=4 | d=2 | adapt | 20k | 2e-4 | 91 | 79 | 11 | 0 | 3/3 | 0929 §2 |
| aswstieltjes (windowed+AS) | q=2 | d=2 | adapt | 20k | 2e-4 | 100 | 96 | 88 | 45 | 5/5 | 0929 §8 |
| aswstieltjes (windowed+AS) | q=2 | d=4 | adapt | 20k | 2e-4 | 77 | 100 | 91 | 49 | 2/2 | 0929 §8.2 |
| aswstieltjes (windowed+AS) | q=2 | d=1 | adapt | 20k | 2e-4 | — | — | — | — | 0.5/2 | 0929 §8.2 |
| aswstieltjes (windowed+AS) | q=8 | d=2 | adapt | 20k | 2e-4 | — | — | — | — | never | 0929 §2 |

Notes:
- asentmax (ASEntmax) escape rate is 0/10 across all LRs/configs without zero-init. Zero-init (_zi)
  + 20k warmup + 2e-4 is the only working config (3/3 escape); 10k warmup or 4e-4 with zero-init
  still fail.
- aswstieltjes q=2 d=2 best-seed result is 100/96/88/45 (s1); 5-seed mean at 256x is 57 ± 30.
- aswstieltjes q=2 d=4 best-seed result is 100/91/49 (s2); d=4 is the best single MQMTAR ladder
  in the project and is within 8 pts of the paper's ASEntmax at 256x.
- entmax-1.5 escaped 1/2 and the escaped run's last.ckpt gives 20/3/0 — effectively untrained.
- q=8 and q=2 (dense) are the wrong direction: dispersion (dense q=2) or training collapse
  (dense q=8) on every task except reverse.

**Paper reference (1K samples, best of 3 seeds x LRs):** Softmax 99.5/97.8/80.2/3.0;
Entmax 99.5/92.7/66.8/9.3; ASEntmax 99.7/99.6/99.0/95.3.

---

## Summary: best cells per task (our reproduced numbers, not paper numbers)

| variant | Sort 4x | Reverse 4x | Copy 64x | MQMTAR 256x |
|---|---:|---:|---:|---:|
| softmax | 4 | 0 | 63 | 42 |
| entmax-1.5 | 53 | 61 | 50 | — |
| asentmax (orig, HF init) | 81 | 53 | 76 | 0 |
| asentmax_zi_w20k | — | — | — | 77 |
| stieltjes q=4 (dense) | 0 | 38 | 52 | 63 |
| wstieltjes q=4, d=2 | 7 | 49 | 25 | 48 |
| aswstieltjes q=2, d=2 | 21† | 100 | 49 | 88 |
| aswstieltjes q=2, d=4 | — | — | — | 91 |

†3-seed mean; single-seed outlier was 62.

The MQMTAR 1024x column (paper target 95.3) ranges 0–49 for every row we have.
The best 1024x result is aswstieltjes q=2, d=4, s2: **49**.
