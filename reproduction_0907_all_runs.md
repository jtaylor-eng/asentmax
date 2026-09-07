# Table 1 reproduction on OSC (A100) — paper protocol, 2 seeds x 3 LRs (MQMTAR: 2 LRs)

## sort

| method | ID | 2x | 4x | 8x | selected run |
|---|---:|---:|---:|---:|---|
| softmax (paper) | 100 | 0 | 0 | 0 | best of 3 seeds x LRs, 1K samples |
| softmax (local 4070, 1 seed, last.ckpt) | 100 | 0 | skip | skip | README recipe LR |
| **softmax (ours)** | 100.0 | 86.0 | 0.0 | skip | s2, lr=8e-4, bleu@2=0.926, ckpt step 156250 |
| softmax (ours, mean±std over 2 seeds @ lr=8e-4) | 100±0 | 43±43 | 0±0 | 0±0 | |
| asentmax (paper) | 100 | 100 | 79.7 | 0 | best of 3 seeds x LRs, 1K samples |
| asentmax (local 4070, 1 seed, last.ckpt) | 100 | 96 | 69 | 0 | README recipe LR |
| **asentmax (ours)** | 100.0 | 100.0 | 81.0 | 0.0 | s1, lr=2e-4, bleu@2=0.999, ckpt step 265625 |
| asentmax (ours, mean±std over 2 seeds @ lr=2e-4) | 100±0 | 100±0 | 75±6 | 0±0 | |
| **stieltjes (ours)** | 100.0 | 86.0 | 0.0 | skip | s1, lr=4e-4, bleu@2=0.968, ckpt step 93750 |
| stieltjes (ours, mean±std over 2 seeds @ lr=4e-4) | 100±0 | 76±10 | 0±0 | 0±0 | |

<details><summary>all runs</summary>

| method | seed | lr | selection metric | ckpt | ID | 2x | 4x | 8x |
|---|---|---|---|---|---:|---:|---:|---:|
| softmax | 1 | 2e-4 | bleu@2=0.415 | best@312500 | 100.0 | 0.0 | skip | skip |
| softmax | 2 | 2e-4 | bleu@2=0.538 | best@265625 | 100.0 | 0.0 | skip | skip |
| softmax | 1 | 4e-4 | bleu@2=0.761 | best@265625 | 100.0 | 0.0 | skip | skip |
| softmax | 2 | 4e-4 | bleu@2=0.822 | best@93750 | 100.0 | 3.0 | 0.0 | skip |
| softmax | 1 | 8e-4 | bleu@2=0.680 | best@31250 | 100.0 | 0.0 | skip | skip |
| softmax | 2 | 8e-4 | bleu@2=0.926 | best@156250 | 100.0 | 86.0 | 0.0 | skip |
| asentmax | 1 | 2e-4 | bleu@2=0.999 | best@265625 | 100.0 | 100.0 | 81.0 | 0.0 |
| asentmax | 2 | 2e-4 | bleu@2=0.999 | best@250000 | 100.0 | 100.0 | 69.0 | 0.0 |
| asentmax | 1 | 4e-4 | bleu@2=0.992 | best@31250 | 100.0 | 95.0 | 9.0 | 0.0 |
| asentmax | 2 | 4e-4 | bleu@2=0.998 | best@250000 | 100.0 | 100.0 | 52.0 | 0.0 |
| asentmax | 1 | 8e-4 | bleu@2=0.988 | best@15625 NaN-diverged | 100.0 | 97.0 | 2.0 | 0.0 |
| asentmax | 2 | 8e-4 | bleu@2=0.988 | best@46875 NaN-diverged | 100.0 | 90.0 | 3.0 | 0.0 |
| stieltjes | 1 | 2e-4 | bleu@2=0.746 | best@140625 | 100.0 | 58.0 | 0.0 | skip |
| stieltjes | 2 | 2e-4 | bleu@2=0.787 | best@265625 | 100.0 | 80.0 | 0.0 | skip |
| stieltjes | 1 | 4e-4 | bleu@2=0.968 | best@93750 | 100.0 | 86.0 | 0.0 | skip |
| stieltjes | 2 | 4e-4 | bleu@2=0.938 | best@125000 | 100.0 | 66.0 | 0.0 | skip |
| stieltjes | 1 | 8e-4 | bleu@2=0.965 | best@31250 | 100.0 | 85.0 | 0.0 | skip |
| stieltjes | 2 | 8e-4 | bleu@2=0.963 | best@31250 | 98.0 | 74.0 | 0.0 | skip |

</details>

## reverse

| method | ID | 1.5x | 2x | 4x | 8x | selected run |
|---|---:|---:|---:|---:|---:|---|
| softmax (paper) | 100 | 36 | 0 | 0 | 0 | best of 3 seeds x LRs, 1K samples |
| softmax (local 4070, 1 seed, last.ckpt) | 100 | 76 | 0 | skip | skip | README recipe LR |
| **softmax (ours)** | 100.0 | 38.0 | 0.0 | skip | skip | s1, lr=8e-4, bleu@3=0.193, last.ckpt (8x monitor degenerate) |
| softmax (ours, mean±std over 2 seeds @ lr=8e-4) | 100±0 | 52±14 | 0±0 | 0±0 | 0±0 | |
| asentmax (paper) | 100 | 100 | 99.8 | 96.4 | 56.7 | best of 3 seeds x LRs, 1K samples |
| asentmax (local 4070, 1 seed, last.ckpt) | 100 | 100 | 100 | 92 | 36 | README recipe LR |
| **asentmax (ours)** | 100.0 | 100.0 | 100.0 | 53.0 | 0.0 | s1, lr=4e-4, bleu@3=0.728, last.ckpt (8x monitor degenerate) |
| asentmax (ours, mean±std over 2 seeds @ lr=4e-4) | 100±0 | 100±0 | 100±0 | 29±24 | 0±0 | |
| **stieltjes (ours)** | 100.0 | 0.0 | skip | skip | skip | s2, lr=8e-4, bleu@3=0.380, last.ckpt (8x monitor degenerate) |
| stieltjes (ours, mean±std over 2 seeds @ lr=8e-4) | 100±0 | 10±10 | 0±0 | 0±0 | 0±0 | |

<details><summary>all runs</summary>

| method | seed | lr | selection metric | ckpt | ID | 1.5x | 2x | 4x | 8x |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| softmax | 1 | 2e-4 | bleu@3=0.045 | last | 100.0 | 0.0 | skip | skip | skip |
| softmax | 2 | 2e-4 | bleu@3=0.056 | last | 100.0 | 0.0 | skip | skip | skip |
| softmax | 1 | 4e-4 | bleu@3=0.181 | last | 100.0 | 28.0 | 0.0 | skip | skip |
| softmax | 2 | 4e-4 | bleu@3=0.126 | last | 100.0 | 31.0 | 0.0 | skip | skip |
| softmax | 1 | 8e-4 | bleu@3=0.193 | last | 100.0 | 38.0 | 0.0 | skip | skip |
| softmax | 2 | 8e-4 | bleu@3=0.142 | last | 100.0 | 67.0 | 0.0 | skip | skip |
| asentmax | 1 | 2e-4 | bleu@3=0.544 | last | 100.0 | 100.0 | 99.0 | 1.0 | - |
| asentmax | 2 | 2e-4 | bleu@3=0.561 | last | 100.0 | 100.0 | 98.0 | 0.0 | skip |
| asentmax | 1 | 4e-4 | bleu@3=0.728 | last | 100.0 | 100.0 | 100.0 | 53.0 | 0.0 |
| asentmax | 2 | 4e-4 | bleu@3=0.632 | last | 100.0 | 100.0 | 99.0 | 5.0 | 0.0 |
| asentmax | 1 | 8e-4 | bleu@3=0.563 | last | 100.0 | 100.0 | 93.0 | 0.0 | skip |
| asentmax | 2 | 8e-4 | bleu@3=0.667 | last | 100.0 | 100.0 | 100.0 | 2.0 | 0.0 |
| stieltjes | 1 | 2e-4 | bleu@3=0.150 | last | 100.0 | 0.0 | skip | skip | skip |
| stieltjes | 2 | 2e-4 | bleu@3=0.154 | last | 100.0 | 0.0 | skip | skip | skip |
| stieltjes | 1 | 4e-4 | bleu@3=0.206 | last | 100.0 | 18.0 | 0.0 | skip | skip |
| stieltjes | 2 | 4e-4 | bleu@3=0.255 | last | 100.0 | 2.0 | 0.0 | skip | skip |
| stieltjes | 1 | 8e-4 | bleu@3=0.299 | last | 100.0 | 19.0 | 0.0 | skip | skip |
| stieltjes | 2 | 8e-4 | bleu@3=0.380 | last | 100.0 | 0.0 | skip | skip | skip |

</details>

## copy

| method | ID | 2x | 4x | 8x | 16x | 32x | 64x | selected run |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| softmax (paper) | 100 | 100 | 99.9 | 99.9 | 99.4 | 96.1 | 85.5 | best of 3 seeds x LRs, 1K samples |
| softmax (local 4070, 1 seed, last.ckpt) | 100 | 100 | 100 | 64 | 0 | skip | skip | README recipe LR |
| **softmax (ours)** | 100.0 | 100.0 | 100.0 | 100.0 | 98.0 | 91.0 | 63.0 | s1, lr=1e-3, acc_@3=1.000, ckpt step 140625 |
| softmax (ours, mean±std over 2 seeds @ lr=1e-3) | 100±0 | 97±3 | 80±20 | 58±42 | 49±49 | 46±46 | 32±32 | |
| asentmax (paper) | 100 | 100 | 99.9 | 99.7 | 99.4 | 96.3 | 86.6 | best of 3 seeds x LRs, 1K samples |
| asentmax (local 4070, 1 seed, last.ckpt) | 100 | 100 | 100 | 100 | 100 | 94 | 74 | README recipe LR |
| **asentmax (ours)** | 100.0 | 100.0 | 99.0 | 100.0 | 99.0 | 96.0 | 76.0 | s2, lr=5e-4, acc_@3=1.000, ckpt step 125000 |
| asentmax (ours, mean±std over 2 seeds @ lr=5e-4) | 100±0 | 100±0 | 98±1 | 90±10 | 74±26 | 49±47 | 38±38 | |
| **stieltjes (ours)** | 100.0 | 100.0 | 100.0 | 97.0 | 50.0 | 0.0 | skip | s1, lr=2e-3, acc_@3=0.980, ckpt step 140625 |
| stieltjes (ours, mean±std over 2 seeds @ lr=2e-3) | 100±0 | 100±0 | 50±50 | 48±48 | 25±25 | 0±0 | 0±0 | |

<details><summary>all runs</summary>

| method | seed | lr | selection metric | ckpt | ID | 2x | 4x | 8x | 16x | 32x | 64x |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| softmax | 1 | 5e-4 | acc_@3=0.600 | best@140625 | 100.0 | 100.0 | 86.0 | 67.0 | 12.0 | 0.0 | skip |
| softmax | 2 | 5e-4 | acc_@3=0.090 | best@140625 | 100.0 | 100.0 | 54.0 | 4.0 | 0.0 | skip | skip |
| softmax | 1 | 1e-3 | acc_@3=1.000 | best@140625 | 100.0 | 100.0 | 100.0 | 100.0 | 98.0 | 91.0 | 63.0 |
| softmax | 2 | 1e-3 | acc_@3=0.180 | best@46875 | 100.0 | 94.0 | 61.0 | 17.0 | 0.0 | skip | skip |
| softmax | 1 | 2e-3 | acc_@3=1.000 | best@125000 | 100.0 | 100.0 | 100.0 | 98.0 | 95.0 | 82.0 | 55.0 |
| softmax | 2 | 2e-3 | acc_@3=1.000 | best@140625 | 100.0 | 100.0 | 100.0 | 99.0 | 99.0 | 22.0 | 0.0 |
| asentmax | 1 | 5e-4 | acc_@3=0.850 | best@31250 | 100.0 | 99.0 | 97.0 | 81.0 | 48.0 | 2.0 | 0.0 |
| asentmax | 2 | 5e-4 | acc_@3=1.000 | best@125000 | 100.0 | 100.0 | 99.0 | 100.0 | 99.0 | 96.0 | 76.0 |
| asentmax | 1 | 1e-3 | acc_@3=1.000 | best@125000 | 100.0 | 100.0 | 100.0 | 98.0 | 95.0 | 95.0 | 72.0 |
| asentmax | 2 | 1e-3 | acc_@3=0.990 | best@125000 | 100.0 | 100.0 | 99.0 | 97.0 | 97.0 | 88.0 | 60.0 |
| asentmax | 1 | 2e-3 | acc_@3=0.940 | best@156250 | 100.0 | 100.0 | 98.0 | 95.0 | 84.0 | 35.0 | 5.0 |
| asentmax | 2 | 2e-3 | acc_@3=0.220 | best@15625 NaN-diverged | - | - | - | - | - | - | - |
| stieltjes | 1 | 5e-4 | acc_@3=1.000 | best@140625 | 100.0 | 100.0 | 100.0 | 99.0 | 100.0 | 89.0 | ERR |
| stieltjes | 2 | 5e-4 | bleu@2=0.282 | last | 100.0 | 15.0 | 0.0 | skip | skip | skip | skip |
| stieltjes | 1 | 1e-3 | acc_@3=1.000 | best@125000 | 100.0 | 100.0 | 100.0 | 99.0 | 99.0 | 90.0 | ERR |
| stieltjes | 2 | 1e-3 | bleu@2=0.544 | last | 100.0 | 14.0 | 0.0 | skip | skip | skip | skip |
| stieltjes | 1 | 2e-3 | acc_@3=0.980 | best@140625 | 100.0 | 100.0 | 100.0 | 97.0 | 50.0 | 0.0 | skip |
| stieltjes | 2 | 2e-3 | bleu@2=0.630 | last | 100.0 | 100.0 | 0.0 | skip | skip | skip | skip |

</details>

## mqmtar

| method | ID | 2x | 4x | 16x | 64x | 256x | 1024x | selected run |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| softmax (paper) | 100 | 100 | 100 | 99.5 | 97.8 | 80.2 | 3.0 | best of 3 seeds x LRs, 1K samples |
| softmax (local 4070, 1 seed, last.ckpt) | 100 | 98 | 99 | 83 | 54 | - | - | README recipe LR |
| softmax (ours) | … | … | … | … | … | … | … | 4 runs, ladders pending |
| asentmax (paper) | 100 | 100 | 100 | 99.7 | 99.6 | 99.0 | 95.3 | best of 3 seeds x LRs, 1K samples |
| asentmax (local 4070, 1 seed, last.ckpt) | 100 | 100 | 100 | 100 | 100 | - | - | README recipe LR |
| asentmax (ours) | … | … | … | … | … | … | … | 4 runs, ladders pending |
| stieltjes (ours) | … | … | … | … | … | … | … | 4 runs, ladders pending |

<details><summary>all runs</summary>

| method | seed | lr | selection metric | ckpt | ID | 2x | 4x | 16x | 64x | 256x | 1024x |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| softmax | 1 | 1e-4 | acc_@3=0.940 | best@78124 | - | - | - | - | - | - | - |
| softmax | 2 | 1e-4 | acc_@3=0.800 | best@78124 | - | - | - | - | - | - | - |
| softmax | 1 | 2e-4 | bleu@2=0.049 | last | - | - | - | - | - | - | - |
| softmax | 2 | 2e-4 | acc_@3=0.460 | best@78124 | - | - | - | - | - | - | - |
| asentmax | 1 | 1e-4 | bleu@2=0.058 | last | - | - | - | - | - | - | - |
| asentmax | 2 | 1e-4 | bleu@2=0.053 | last | - | - | - | - | - | - | - |
| asentmax | 1 | 2e-4 | bleu@2=0.051 | last | - | - | - | - | - | - | - |
| asentmax | 2 | 2e-4 | bleu@2=0.045 | last | - | - | - | - | - | - | - |
| stieltjes | 1 | 1e-4 | bleu@2=0.049 | last | - | - | - | - | - | - | - |
| stieltjes | 2 | 1e-4 | bleu@2=0.060 | last | - | - | - | - | - | - | - |
| stieltjes | 1 | 2e-4 | bleu@2=0.052 | last | - | - | - | - | - | - | - |
| stieltjes | 2 | 2e-4 | bleu@2=0.051 | last | - | - | - | - | - | - | - |

</details>
