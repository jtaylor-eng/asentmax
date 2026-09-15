# LR sweep — every Softmax / Stieltjes run (appendix to reproduction_0913.md)

Columns: `sel` = validation selection value (sort: BLEU@4x; else exact-match acc@8x; `deg` = primary identically 0, BLEU@4x fallback shown). `ckpt` = which checkpoint the reported ladder comes from. `impl` = Stieltjes kernel used for training (0907 runs eager; sweep runs triton). `tiebreak` = validation ladder on the seed-4243 long splits (copy 16x/32x, mqmtar 16x/64x). `esc` = MQMTAR plateau-escape step (first train loss < 0.3). Ladders are test exact-match %, 100 samples/length; `skip` = not evaluated after an exact 0.0 at a shorter length.

## sort

| method | seed | lr | new | impl | sel | ckpt | ID | 2x | 4x | 8x |
|---|---|---|---|---|---|---|---:|---:|---:|---:|
| softmax | 1 | 2e-4 |  |  | 0.415 | best | 100 | 0 | skip | skip |
| softmax | 2 | 2e-4 |  |  | 0.538 | best | 100 | 0 | skip | skip |
| softmax | 1 | 4e-4 |  |  | 0.761 | best | 100 | 0 | skip | skip |
| softmax | 2 | 4e-4 |  |  | 0.822 | best | 100 | 3 | 0 | skip |
| softmax | 1 | 8e-4 |  |  | 0.680 | best | 100 | 0 | skip | skip |
| softmax | 2 | 8e-4 |  |  | 0.926 | best | 100 | 86 | 0 | skip |
| softmax | 1 | 1.6e-3 | new |  | 0.968 | best | 98 | 74 | 0 | skip |
| softmax | 2 | 1.6e-3 | new |  | 0.950 | best | 100 | 47 | 0 | skip |
| softmax | 3 | 1.6e-3 | new |  | 0.945 | best | 100 | 38 | 0 | skip |
| softmax | 1 | 3.2e-3 | new |  | 0.985 | best | 100 | 75 | 0 | skip |
| softmax | 2 | 3.2e-3 | new |  | 0.915 | best | 57 | 12 | 0 | skip |
| softmax | 3 | 3.2e-3 | new |  | 0.904 | best | 100 | 24 | 0 | skip |
| softmax | 1 | 6.4e-3 | new |  | 0.992 | best | 100 | 99 | 4 | 0 |
| softmax | 2 | 6.4e-3 | new |  | 0.992 | best | 100 | 91 | 2 | 0 |
| stieltjes | 1 | 2e-4 |  | eager | 0.746 | best | 100 | 58 | 0 | skip |
| stieltjes | 2 | 2e-4 |  | eager | 0.787 | best | 100 | 80 | 0 | skip |
| stieltjes | 1 | 4e-4 |  | eager | 0.968 | best | 100 | 86 | 0 | skip |
| stieltjes | 2 | 4e-4 |  | eager | 0.938 | best | 100 | 66 | 0 | skip |
| stieltjes | 3 | 4e-4 |  | triton | 0.970 | best | 100 | 62 | 0 | skip |
| stieltjes | 1 | 8e-4 |  | eager | 0.965 | best | 100 | 85 | 0 | skip |
| stieltjes | 2 | 8e-4 |  | eager | 0.963 | best | 98 | 74 | 0 | skip |
| stieltjes | 1 | 1.6e-3 | new | triton | 0.963 | best | 100 | 66 | 0 | skip |
| stieltjes | 2 | 1.6e-3 | new | triton | 0.900 | best | 97 | 0 | skip | skip |
| stieltjes | 1 | 3.2e-3 | new | triton | 0.856 | best | 100 | 2 | 0 | skip |
| stieltjes | 2 | 3.2e-3 | new | triton | 0.984 | best | 100 | 88 | 0 | skip |
| stieltjes | 3 | 3.2e-3 | new | triton | 0.971 | best | 100 | 96 | 0 | skip |
| stieltjes | 1 | 6.4e-3 | new | triton | 0.707 | best | 99 | 3 | 0 | skip |
| stieltjes | 2 | 6.4e-3 | new | triton | 0.973 | best | 100 | 51 | 0 | skip |

## reverse

| method | seed | lr | new | impl | sel | ckpt | ID | 1.5x | 2x | 4x | 8x |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| softmax | 1 | 2e-4 |  |  | 0.045 deg | last | 100 | 0 | skip | skip | skip |
| softmax | 2 | 2e-4 |  |  | 0.056 deg | last | 100 | 0 | skip | skip | skip |
| softmax | 1 | 4e-4 |  |  | 0.181 deg | last | 100 | 28 | 0 | skip | skip |
| softmax | 2 | 4e-4 |  |  | 0.126 deg | last | 100 | 31 | 0 | skip | skip |
| softmax | 1 | 8e-4 |  |  | 0.193 deg | last | 100 | 38 | 0 | skip | skip |
| softmax | 2 | 8e-4 |  |  | 0.142 deg | last | 100 | 67 | 0 | skip | skip |
| softmax | 3 | 8e-4 |  |  | 0.138 deg | best (last pending) | 92 | 2 | 0 | skip | skip |
| softmax | 1 | 1.6e-3 | new |  | 0.222 deg | last | 100 | 1 | 0 | skip | skip |
| softmax | 2 | 1.6e-3 | new |  | 0.157 deg | last | 100 | 48 | 0 | skip | skip |
| softmax | 3 | 1.6e-3 | new |  | 0.295 deg | best (last pending) | 93 | 2 | 0 | skip | skip |
| softmax | 1 | 3.2e-3 | new |  | 0.148 deg | last | 100 | 0 | skip | skip | skip |
| softmax | 2 | 3.2e-3 | new |  | 0.190 deg | last | 100 | 0 | skip | skip | skip |
| stieltjes | 1 | 2e-4 |  | eager | 0.150 deg | last | 100 | 0 | skip | skip | skip |
| stieltjes | 2 | 2e-4 |  | eager | 0.154 deg | last | 100 | 0 | skip | skip | skip |
| stieltjes | 1 | 4e-4 |  | eager | 0.206 deg | last | 100 | 18 | 0 | skip | skip |
| stieltjes | 2 | 4e-4 |  | eager | 0.255 deg | last | 100 | 2 | 0 | skip | skip |
| stieltjes | 1 | 8e-4 |  | eager | 0.299 deg | last | 100 | 19 | 0 | skip | skip |
| stieltjes | 2 | 8e-4 |  | eager | 0.380 deg | last | 100 | 0 | skip | skip | skip |
| stieltjes | 1 | 1.6e-3 | new | triton | 0.263 deg | last | 100 | 95 | 0 | skip | skip |
| stieltjes | 2 | 1.6e-3 | new | triton | 0.384 deg | last | 100 | 74 | 0 | skip | skip |
| stieltjes | 3 | 1.6e-3 | new | triton | 0.309 deg | best (last pending) | 91 | 3 | 0 | skip | skip |
| stieltjes | 1 | 3.2e-3 | new | triton | 0.422 deg | last | 100 | 100 | 38 | 0 | skip |
| stieltjes | 2 | 3.2e-3 | new | triton | 0.118 deg | last | 100 | 38 | 0 | skip | skip |
| stieltjes | 3 | 3.2e-3 | new | triton | 0.260 deg | best (last pending) | 94 | 6 | 0 | skip | skip |
| stieltjes | 1 | 6.4e-3 | new | triton | 0.353 deg | best (last pending) | 96 | 34 | 0 | skip | skip |
| stieltjes | 2 | 6.4e-3 | new | triton | 0.244 deg | best (last pending) | 95 | 49 | 0 | skip | skip |

## copy

| method | seed | lr | new | impl | sel | ckpt | tb 16x | tb 32x | ID | 2x | 4x | 8x | 16x | 32x | 64x |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| softmax | 1 | 2.5e-4 | new |  | 0.173 deg | last | - | - | 100 | 42 | 0 | skip | skip | skip | skip |
| softmax | 2 | 2.5e-4 | new |  | 0.196 deg | last | - | - | 100 | 0 | skip | skip | skip | skip | skip |
| softmax | 1 | 5e-4 |  |  | 0.600 | best | 11 | 0 | 100 | 100 | 86 | 67 | 12 | 0 | skip |
| softmax | 2 | 5e-4 |  |  | 0.090 | best | 0 | skip | 100 | 100 | 54 | 4 | 0 | skip | skip |
| softmax | 1 | 1e-3 |  |  | 1.000 | best | 98 | 88 | 100 | 100 | 100 | 100 | 98 | 91 | 63 |
| softmax | 2 | 1e-3 |  |  | 0.180 | best | 0 | skip | 100 | 94 | 61 | 17 | 0 | skip | skip |
| softmax | 1 | 2e-3 |  |  | 1.000 | best | 95 | 81 | 100 | 100 | 100 | 98 | 95 | 82 | 55 |
| softmax | 2 | 2e-3 |  |  | 1.000 | best | 96 | 18 | 100 | 100 | 100 | 99 | 99 | 22 | 0 |
| softmax | 1 | 4e-3 | new |  | 0.930 | best | 67 | 20 | 100 | 100 | 96 | 94 | 59 | 22 | 0 |
| softmax | 2 | 4e-3 | new |  | 0.494 deg | last | - | - | 100 | 100 | 31 | 0 | skip | skip | skip |
| stieltjes | 1 | 2.5e-4 | new | triton | 0.970 | best | 40 | 0 | 100 | 100 | 100 | 98 | 47 | 0 | skip |
| stieltjes | 2 | 2.5e-4 | new | triton | 0.329 deg | last | - | - | 100 | 99 | 10 | 0 | skip | skip | skip |
| stieltjes | 1 | 5e-4 |  | eager | 1.000 | best | 97 | 85 | 100 | 100 | 100 | 99 | 100 | 87 | 52 |
| stieltjes | 2 | 5e-4 |  | eager | 0.282 deg | last | - | - | 100 | 15 | 0 | skip | skip | skip | skip |
| stieltjes | 1 | 1e-3 |  | eager | 1.000 | best | 98 | 81 | 100 | 100 | 100 | 99 | 99 | 92 | 35 |
| stieltjes | 2 | 1e-3 |  | eager | 0.544 deg | last | - | - | 100 | 14 | 0 | skip | skip | skip | skip |
| stieltjes | 1 | 2e-3 |  | eager | 0.980 | best | 59 | 0 | 100 | 100 | 100 | 97 | 50 | 0 | skip |
| stieltjes | 2 | 2e-3 |  | eager | 0.630 deg | last | - | - | 100 | 100 | 0 | skip | skip | skip | skip |
| stieltjes | 1 | 4e-3 | new | triton | 0.960 | best | 67 | 3 | 99 | 100 | 99 | 93 | 68 | 2 | 0 |
| stieltjes | 2 | 4e-3 | new | triton | 0.910 | best | 60 | 10 | 100 | 99 | 97 | 83 | 61 | 8 | 0 |

## mqmtar

| method | seed | lr | new | impl | sel | ckpt | tb 16x | tb 64x | ID | 2x | 4x | 16x | 64x | 256x | 1024x | esc |
|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| softmax | 1 | 5e-5 | new |  | 0.062 deg | last | - | - | 0 | skip | skip | skip | skip | skip | skip | never |
| softmax | 2 | 5e-5 | new |  | 0.069 deg | last | - | - | 0 | skip | skip | skip | skip | skip | skip | never |
| softmax | 1 | 1e-4 |  |  | 1.000 | best | 90 | 34 | 100 | 100 | 97 | 90 | 48 | 0 | skip | 16k |
| softmax | 2 | 1e-4 |  |  | 1.000 | best | 96 | 61 | 100 | 100 | 100 | 92 | 62 | 0 | skip | 66k |
| softmax | 1 | 2e-4 |  |  | 0.980 | best | 83 | 44 | 100 | 100 | 94 | 90 | 36 | 2 | 0 | 143k |
| softmax | 2 | 2e-4 |  |  | 1.000 | best | 96 | 77 | 100 | 100 | 99 | 97 | 68 | 1 | 0 | 68k |
| softmax | 1 | 4e-4 | new |  | 0.990 | best | 94 | 82 | 100 | 99 | 100 | 96 | 84 | 53 | 4 | 32k |
| softmax | 2 | 4e-4 | new |  | 1.000 | best | 99 | 81 | 99 | 100 | 100 | 97 | 92 | 40 | 1 | 141k |
| stieltjes | 1 | 1e-4 |  | eager | 0.076 deg | best (last pending) | - | - | 0 | skip | skip | skip | skip | - | - | never |
| stieltjes | 2 | 1e-4 |  | eager | 0.067 deg | best (last pending) | - | - | 0 | skip | skip | skip | skip | - | - | never |
| stieltjes | 1 | 2e-4 |  | eager | 1.000 | best | 99 | 72 | 100 | 100 | 100 | 99 | 66 | 6 | 0 | 49k |
| stieltjes | 2 | 2e-4 |  | eager | 1.000 | best | 99 | 80 | 100 | 100 | 100 | 99 | 86 | 4 | - | 86k |
| stieltjes | 1 | 4e-4 | new | triton | 1.000 | best | 97 | 86 | 100 | 98 | 97 | 99 | 90 | 53 | 8 | 60k |
| stieltjes | 2 | 4e-4 | new | triton | 1.000 | best | 98 | 78 | 99 | 100 | 99 | 100 | 82 | 28 | 1 | 129k |
| stieltjes | 1 | 8e-4 | new | triton | 1.000 | best | 99 | 81 | 99 | 100 | 100 | 96 | 94 | 38 | 0 | 117k |
| stieltjes | 2 | 8e-4 | new | triton | 0.140 deg | last | - | - | 2 | 0 | skip | skip | skip | skip | skip | never |
