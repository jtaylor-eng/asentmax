# ASEntmax MQMTAR fix + q sweep — results (2026-09-29)

Batch planned in `asfix_qsweep_0923_plan.md` (predictions there, scored in §4; follow-up predictions §7). Protocol as in
`reproduction_0913.md`; 100 test samples/length (±4 pt SE at p = 0.8). All 45 runs completed, no
failures; ~255 GPU-h. Reverse rows laddered from last.ckpt (job 7583832; the best-by-monitor pass
picked the step-11718 ckpt again, now fixed in `run_one.sh`).

## 1. ASEntmax on MQMTAR: the zero-init fix works with 20k warmup (3 of 3), not with 10k (0 of 3)

lr 2e-4 throughout. "escape" = first step with train loss < 0.3 (plateau is 0.527).

| run | escape step | ID | 2x | 4x | 16x | 64x | 256x | 1024x |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| asentmax_zi_w20k s1 | 147k | 100 | 100 | 100 | 100 | 96 | 77 | 38 |
| asentmax_zi_w20k s2 | 193k | 98 | 97 | 93 | 59 | 11 | 0 | skip |
| asentmax_zi_w20k s3 | 78k | 99 | 98 | 85 | 50 | 0 | skip | skip |
| asentmax_zi (10k) s1-3 | never x3 | 0 | | | | | | |
| entmax_w20k s1 | 171k | 100 | 100 | 100 | 99 | 91 | 58 | 7 |
| entmax_w20k s2 | never | 0 | | | | | | |
| entmax_w20k s3 | 206k | 89 | 70 | 39 | 3 | 0 | skip | skip |
| asentmax, HF init, all prior configs (10 runs) | never x10 | 0 | | | | | | |
| ASEntmax (paper) | | 100 | 100 | 100 | 99.7 | 99.6 | 99.0 | 95.3 |
| stieltjes q4 (0913, best of 3) | 60-130k | 100 | 100 | 97 | 99 | 94 | 63 | 12 |

Selected by val acc@8x: s1 (the only one with a full ladder). Best ASEntmax MQMTAR row we have:
100/100/100/100/96/77/38 vs paper 99.6/99.0/95.3 at 64x/256x/1024x. Escape rate for ASEntmax at 20k
warmup went from 0/3 (HF init, job 7462345) to 3/3 (zero init) at the same LR, seeds and schedule; at
10k warmup it is still 0/3. So: init is causal, but only in combination with the longer warmup, and
the two seeds that escaped late (147k, 193k) had too little schedule left to consolidate (11 and 0 at
64x). This is the same "escape step decides the ladder" pattern as every other method on MQMTAR.

Plain entmax at the same config: 2/3 escaped (171k, 206k), one full ladder (91/58/7). So the family
trains on MQMTAR with 20k warmup; the HF-init scale was what took ASEntmax from ~2/3 to 0/10.

What the paper's 99.0/95.3 needs beyond this: an escape before ~80k steps (s3 escaped at 78k and still
only reached 50 at 16x, so probably also more than 390k steps or a higher LR after escape), or best of
many seeds. Not attempted.

## 2. q sweep (1 seed, 1 LR per task/row; q=4 cells from the 3-seed rows where they exist, seed 1)

Read as ranges; single-seed noise on the 3-seed rows is up to 40 pts at the informative length.

### Sort (ID / 2x / 4x / 8x)

| row | q=2 | q=4 | q=8 |
|---|---|---|---|
| stieltjes (3.2e-3) | 100/0 | 100/88/0 (s2) | 41/1/0 |
| wstieltjes (1.6e-3) | 97/34/0 | 95/85/0 (s1) | 92/27/1/0 |
| asstieltjes (2e-4) | 49/0 | 31/0 | 100/98/2/0 |
| aswstieltjes (2e-4) | 100/100/**62**/0 | 100/99/37/0 (s1) | 100/96/1/0 |
| ref | entmax 53, ASEntmax 81/69 at 4x | | |

### Copy (8x / 16x / 32x / 64x)

| row | q=2 | q=4 | q=8 |
|---|---|---|---|
| stieltjes (5e-4) | 0 by 4x | 99/100/87/52 (s1) | 75/31/1/0 |
| wstieltjes (5e-4) | 97/85/60/10 | 96/93/69/25 (s1) | 85/39/0 |
| asstieltjes (1e-3) | 0 by 8x | 100/98/90/14 | 90/51/6/0 |
| aswstieltjes (1e-3) | 96/89/49/4 | 97/88/60/15 (s1) | 79/32/1/0 |
| ref | entmax 86/50, ASEntmax 95/72 at 32x/64x | | |

### MQMTAR (16x / 64x / 256x / 1024x; escape step)

| row | q=2 | q=4 | q=8 |
|---|---|---|---|
| stieltjes (4e-4) | 1/0, esc 69k | 99/94/63/12 (s3), esc 87k | 38/3/0, esc 64k |
| wstieltjes (4e-4) | 94/77/28/0, esc 100k | 98/91/48/6 (s3), esc 51k | 43/2/0, esc 95k |
| asstieltjes (2e-4 w20k) | 0 at 16x, esc 150k | 98/93/36/0, esc 69k | never |
| aswstieltjes (2e-4 w20k) | 100/**96/88/45**, esc 141k | 91/79/11/0 (s1), esc 61k | never |
| ref | ASEntmax_zi 96/77/38; paper 99.6/99.0/95.3 | | |

### Reverse (ID / 1.5x / 2x / 4x / 8x; last.ckpt, job 7583832)

| row | q=2 | q=4 | q=8 |
|---|---|---|---|
| stieltjes (3.2e-3) | 100/46/0 | 100/100/38/0 (s1) | 100/100/96/0 |
| wstieltjes (1.6e-3) | 100/100/99/3/0 | 100/100/100/49/0 (s3; s1/s2 0 at 4x) | 100/100/53/0 |
| asstieltjes (4e-4) | 100/17/0 | 100/88/0 | 100/97/52/0 |
| aswstieltjes (4e-4) | 100/100/100/14/0 | 100/100/100/0 (s2; s3 1) | 100/97/46/0 |
| ref | entmax 99/61, ASEntmax 100/53 at 2x/4x | | |

Reverse is the one task where q=8 is not worst: dense q8 gets 96 at 2x (dense q4 38, q2 0), i.e. the
sharper interior helps the dense map here, the opposite of every other task. Windowed q2 is 99 at 2x
and 3-14 at 4x, on par with windowed q4's 2x and below its best seed at 4x (49, itself a 1-of-3 seed).

## 3. Findings

3.1 q = 8 is the wrong direction on three of four tasks. Dense q8 collapses on copy (0 at 64x, 31 at 16x vs
q4's 100) and sort (ID 41), and its mqmtar tail is gone (3 at 64x). Windowed q8 loses to q4 on sort, copy
and mqmtar. The AS rows at q8 never escape on mqmtar. Reverse is the exception: dense q8 reaches 96 at 2x
where dense q4 gets 38 and q2 gets 0, so the sharper interior helps the dense map exactly where its
dispersion was the failure. With q16 already dead, the sharp-interior direction is closed except on reverse.

3.2 q = 2 splits by whether the map is windowed. Dense q2 (interior alpha = 0.5, dispersion ~ n^{1/2}) is
the worst dense row on every task (sort 0 at 2x, copy 0 at 4x, mqmtar 1 at 16x): the dispersion argument
from the theory note is confirmed at the wide end. Windowed q2 is a different story: the window removes
the dispersion and leaves the widest interior, and it is the best or near-best Stieltjes row on three
tasks:
  aswstieltjes_q2: sort 4x 62 (vs ASEntmax 81/69, entmax 53; any q4 seed <= 37), mqmtar 96/88/45 at
  64x/256x/1024x (the best mqmtar ladder in the whole project; ASEntmax_zi 96/77/38, paper 99.6/99.0/95.3),
  copy 49/4 at 32x/64x (below q4's 60/15, within 1-seed noise).
  wstieltjes_q2: copy 60/10 vs q4's 69/25, mqmtar 77/28 vs q4's 91/48.
One seed each. The sort and mqmtar cells are 2-3 SE above anything from the q4 rows, so they are worth
seeds; the copy cells are not distinguishable from q4.

3.3 The AS scale on the dense map (asstieltjes, new row) does nothing useful: sort 31-49 ID at q2/q4
(the scale plus a dense heavy tail overshoots), copy 14 at 64x vs dense 52, mqmtar 93/36/0 at q4 vs
dense 94/63/12. Only q8 trains on sort and it generalises to 2. Drop it.

3.4 Reading across the table: the two knobs do separate. q controls the interior (q8 too sharp to
train on copy, q2 too flat to hold a tail when dense); d controls dispersion (the window is what
makes q2 usable). The one region entmax does not cover, a wide interior with a hard cutoff, is where
the single best cells came from. The next test is that region, with seeds: aswstieltjes_q2 on sort and
mqmtar x 3 seeds, plus d in {1, 4} at q2 to see whether the window width matters as much as q did.

## 4. Predictions scored (plan §1-2)

| prediction | outcome |
|---|---|
| asentmax_zi_w20k escapes 2-3/3 (70%), >= 90 at 64x, >= 60 at 256x if so (65%) | right: 3/3; s1 96/77, the other two escaped too late |
| asentmax_zi at 10k escapes 1-2/3 (55%) | wrong: 0/3; warmup still binds |
| entmax_w20k escapes 1-2/3 (60%) | right: 2/3 |
| probe: stuck ASEntmax support <= 2, escaped Stieltjes >= 5 (70%) | wrong (scored in the plan file) |
| dense q2 below q4 on mqmtar 256x+ (75%) and reverse 2x (60%) | right on both (0 vs 63; 0 vs 38) |
| windowed q2 best windowed row on sort 4x, 10-30, below entmax's 53 (55%) | half: wstieltjes q2 gets 0; AS-windowed q2 gets 62, above entmax |
| dense q8 collapses on copy (60%); mqmtar 256x <= q4 (60%); sort 2x <= 50 (65%) | right x3 |
| windowed q8 worst on sort/copy (65%), ~q4 on mqmtar 64x (55%) | right / wrong (2 vs 91) |
| asstieltjes ~ dense q4 on mqmtar, above dense on sort 4x (55%) | wrong on both (36 vs 63; 0 vs 0) |
| sort 4x windowed order q2 > q4 > q8 (60%); mqmtar 256x q4 >= q8 > q2 (55%) | AS rows: right / wrong (q2 88 > q4 11 > q8 dead) |

6 of 10 right. The systematic miss: I had q2 as "too flat" from the dense dispersion argument and did
not separate it from the windowed case, where the window removes exactly the failure mode.

## 5. Selection notes

- mqmtar asentmax_zi_w20k: val acc@8x picks s1 (only full ladder); no tie.
- q-sweep rows are single runs; nothing to select. Reverse rows are ranked by val BLEU@4x per protocol.
- The sort 4x = 62 and mqmtar 88/45 cells are one seed at one LR and must not be quoted as row values.

## 6. Jobs

| job | what | status |
|---|---|---|
| 7576491 x9 | asfix (asentmax_zi x2 warmups, entmax_w20k), mqmtar | done |
| 7576492-5 x9 | q sweep, 4 tasks | done |
| 7576554 | attention-support probe | done (plan file) |
| 7583832 x9 | reverse last.ckpt ladders for the q-sweep rows | done (§2 Reverse) |
| 7583955 x6 | sort: aswstieltjes_q2 seeds 2-3; aswstieltjes_q2_d{1,4} seeds 1-2 | pending |
| 7583956 x9 | mqmtar: aswstieltjes_q2_w20k seeds 2-3; _d{1,4}_w20k seeds 1-2; asentmax_zi_w20k at 4e-4 x3 | pending |

## 7. Follow-up predictions (Sep 29, before 7583955/6 landed)

- aswstieltjes_q2 sort 4x, 3-seed mean: 30-55 (65%); at least one seed < 20 (60%). Beats entmax's
  best single seed (53) on the mean: 25%.
- aswstieltjes_q2_w20k mqmtar, 3 seeds: 3/3 escape (65%); 3-seed mean at 256x 50-80 (55%); the seed-1
  88/45 is the top seed, not the median (70%).
- d at q2 on sort 4x: d=4 >= d=2 > d=1 (55%); d=1 is the sharpest window and lands at 0-20.
- d at q2 on mqmtar: d=1 escapes but with the worst tail (256x < 30, 60%); d=4 ~ d=2 (55%).
- asentmax_zi_w20k at 4e-4: escapes 2-3/3 (60%) and earlier than at 2e-4 (median escape < 120k, 55%);
  one seed NaNs or collapses (35%). If escape is early, 256x >= 85 (55%).
