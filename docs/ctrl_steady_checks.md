# CTRL-48 steady gap: score-only + offline checks of H-teacher vs H-student (09/11/2026)

Controller agent. Nothing was trained, no repo code changed, nothing committed. Scratch: /tmp/steady_checks/ (score.py, agg.py, probe.py, ana.py, ana2.py, res_*.json, pr_s*.txt). The probe wheel was built from a /tmp copy of the workspace and installed with `--target /tmp/steady_checks/site`; it was not installed into the project venv.

**Genomes = the published HEADLINE, not `_winner.yaml.gz`.** s1 = NEURONS#2 (stage1 ckpt pop[2]), s2 = MEMORY#1, s3 = MEMORY#2. They were scored on the GPU path and reproduce the published MULTI-SEED rows exactly: 1.49/0.98/0.367, 1.66/1.11/0.364, 1.46/1.00/0.372, 0/3500 fails.
- Thresholds = the train-seed fit under the as-flown ec (xy 0.5). Only the scoring ec changes, so the address function is identical in every cell.
- "xy 0" = `--xy-offset 1e-6`, because `tilt` requires xy > 0. The position loop stays armed and every episode starts at the origin. The teacher doc's 0.927/0.313 used xy exactly 0 (horizontal channel off), so its numbers differ from the xy0_level row here (0.809/0.185).
- **Side finding (bug, not fixed):** the CPU scorer (`WNN_CONTROLLER_GPU_EVAL=0`) does NOT match GPU on this recipe (ctbr+rf+dres). On s1, seed 99990101: CPU err 1.322 / steady 0.924 / pos 0.373 vs GPU 1.315 / 0.973 / 0.357; GPU = the published number. A parity sweep for this feature set is missing from `cargo test`.

## 1. Score-only, 35 CF2.1 units × 100 eps (seeds 99990101-135), block-2rf regime. stable / err° / steady° / pos m; fails = 0/3500 in every cell
| cell | dres s1 / s2 / s3 steady | dres mean | MPCOF>rate(h10) | plain MPCOF | gap steady (err) vs >rate |
|---|---|---|---|---|---|
| xy 0.5, tilt (as scored) | .984 / 1.107 / .997 | 100 / 1.535 / **1.029** / .368 | 100 / 1.352 / **0.684** / .341 | 100 / 1.808 / 1.063 / .354 | **+0.345** (+0.18) |
| xy 0.5, level | .754 / .912 / .624 | 100 / 1.349 / 0.764 / .368 | 100 / 1.323 / 0.429 / .341 | 100 / 1.573 / 0.658 / .354 | +0.335 (+0.03) |
| xy 0, tilt | .800 / .877 / .927 | 100 / 1.345 / 0.868 / .146 | 100 / 0.981 / 0.450 / .166 | 100 / 1.434 / 1.041 / .175 | +0.418 (+0.36) |
| xy 0, level | .670 / .756 / .575 | 100 / 1.132 / 0.667 / .146 | 100 / 0.809 / 0.185 / .166 | 100 / 1.123 / 0.558 / .175 | **+0.482** (+0.32) |

- **H-teacher's prediction fails.** At xy 0 the teacher's steady falls 0.68 → 0.19 as the Mahony bias shrinks, but the student's gap does not shrink toward +0.1°. It grows, +0.35 → +0.48, and does so on all 3 seeds (+0.49 / +0.57 / +0.39 at xy0_level).
- The student carries a roughly constant ~0.35-0.5° hold-precision excess that does not depend on the bias. That is H2 in the teacher doc's terms and H-student's territory.
- Context: the student beats plain MPCOF on steady in both tilt cells, and on pos in every cell at xy 0.

## 2. Offline label consistency (probe: one DAgger collection round, 200 student-flown episodes at 5°, xy 0.5, own cells, seed 99990101; 8000 hold decisions t ≥ 1.6 s per genome)
- Est labels come from the SAME MPCOF state at each decision, fed the student's own Mahony q through a side-effect-free `decide()` call. Sanity check: the true-q probe reproduces the trained label exactly (max |Δ| = 0).
- **The bias label term is smaller than estimated.** sd(true − est) = 0.032-0.051 rad/s, against a hold label sd of 0.09-0.11. corr(true, est) = 0.82-0.96, sign agreement 0.85-0.91. (The teacher doc's 0.065 was measured on the est-flown teacher's own states.)

| genome / bank | within-address label sd, true / est | cross-episode within-address sd (per-episode means, groups seen in ≥3 eps), true / est | cross-episode sign-conflict minority, true / est |
|---|---|---|---|
| s1 p / q | .036/.035, .036/.033 | .058/.056, .055/.050 | .22/.23, .17/.16 |
| s2 p / q | .039/.035, .036/.032 | .061/.053, .057/.050 | .22/.20, .21/.20 |
| s3 p / q | .039/.034, .038/.032 | .060/.053, .060/.051 | .17/.15, .17/.18 |

- **H-teacher is NOT supported offline either.** Est labels are only 4-13 % tighter within an address, and the sign conflict is the same.
- The conflict is address-resolution limited, not label-source limited. Within-address cross-episode sd (0.05-0.06) is about 60 % of the global between-episode sd (0.08-0.11): the addresses separate only about 60 % of the hold-command variance.

## 3. Last-write outcome: stored cell vs the hold majority at the same (neuron, address)
- Groups were restricted to ≥5 hold visits from ≥3 episodes.
- **Majority OFF (≥90 %):** n ≈ 19-32 k per bank; stored ON 3-5 %, so these cells are consistent.
- **Majority ON (≥90 %):** n = 234-724 per bank; stored ON only **0.47-0.62**, about a coin flip, on all 6 bank × seed cases.
- Over all active groups, stored == majority 0.53-0.57, against about 0.49 if the stored bit were independent of the hold majority.
- **Direct evidence for H-student's write rule.** In BINARY a FALSE write deletes the cell, so a TRUE survives only if the last visitor of that address wanted TRUE. Near-unanimous hold TRUEs are lost about half the time, so the sign of the small hold command is erased.
- Caveat: the overwriting visits are not observed. They may be non-hold states (transients, 8-30° curriculum) sharing the address. A tally would also lose a TRUE if those visits outnumber the hold visits, so arm A1 tests the write rule and arm A3 tests the competing traffic.

## 4. Verdict and recommended A/B arms (paired vs `rf_c200_dres_full_s1..s3`, 35 units, one run at a time)
**H-student survives; H-teacher is refuted as the main cause.**
- Score test: the gap rises to +0.48 at xy 0 instead of falling to +0.1.
- Offline test: est labels are no more consistent per address.
- Last-write test: majority-TRUE cells survive at chance.
1. **A1 `--memory-mode QUAD_WEIGHTED --output-decode antagonist`** (same topology).
   - Expect stored-ON for majority-ON groups to rise from ~0.5 to > 0.8 and tail p/q r to rise from ~0.1 to > 0.4. Steady should move from 1.03 toward ~0.75-0.8 (gap halved).
   - Gate offline first: rerun probe.py + ana2.py on the A1 smoke winner.
2. **A2 `--target-levels 32`.** It averages the hold sign over 3-4 independent addresses per threshold. Expect a smaller effect than A1, and check the collective floor first, as the student doc warns.
3. **A3 `--rg-full-tilt-deg 15`.** Less large-tilt traffic overwriting hold addresses. This is the cheapest test of the caveat in §3; expect a small steady gain.
4. **Deprioritised: est-fed DAgger labels (H-teacher's arm).** The label term is 0.03-0.05 rad/s and est labels are only ≤13 % tighter, so expect ≤0.05° on steady. Worth flying only after A1, if a gap remains at xy 0.5 but not at xy 0.
5. Not an arm: before A1 lands, the gain-2 label/decode mismatch (student doc §5.2) needs the per-bank gain fix, or a learned hold command will run at about 2×.
