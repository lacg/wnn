---
name: ctrl-seed-table
description: Per-seed controller comparison table (TEST, 35 CF2.1 units) from runs.log — the B | F pair table with Δ columns and the M | Y | D box table with the MPCOF eq-info reference row. Use for every controller A/B readout, the hourly tick's controller tables, and whenever Luiz asks "show the table", "pair table", "B vs F", "how is F doing per seed", or "M / Y / D table".
---

# Controller per-seed table

ONE script renders every per-seed controller table: `scripts/ctrl_seed_table.py`. Run it and paste its output
VERBATIM in a fenced block — never retype numbers, never compute a cell by hand.

## Rules the script enforces (Luiz)
- Rows = every seed of the block (s1 → last), finished or not: `flying` for a started run, `–` for not yet.
- Cells are `A | B (| C)` raw values; the reference row (MPCOF eq-info = THE comparator) sits on top.
- **stable % is FLOORED from fails** (19/11): 1 fail in 3500 prints 99.9, never 100.0; means floor to 0.01.
- Mean row is PAIRED: only seeds where every arm with results finished; fails are pooled with the denominator shown.
- While arms have different seed counts, an `all done (n=…)` row follows: each arm over ALL its own finished seeds,
  fails as count/denominator per arm (Luiz 19/11: the paired mean hid fails on seeds other arms had not reached).
- Columns: stable / err / steady / pos / fails. Never add alt, pos_hold or dz_hold (Luiz 18/11).
- Pair style adds Δ (last arm − first arm) for err / steady / pos; `--delta X:Y` adds any pair instead.
- Name arms by recipe (D, D+F, Y, Y+F) — never a generic "B".

## Standard invocations

**FF comparison D | D+F | Y | Y+F (the tick's FF table since 19/11)** — arms named by recipe, never a bare "B"
(Luiz 19/11: "B" meant D in one table and Y in the other). Δ columns: each F against ITS OWN base:
```bash
python3 scripts/ctrl_seed_table.py --style pair --seeds 1-10 \
  --arm 'D=_dmD_full_s{s}$|_n10D_full_s{s}$' --arm 'D+F=_a59_.*_ffD_F_full_s{s}$' \
  --arm 'Y=_dmyY_full_s{s}$|_n10Y_full_s{s}$' --arm 'Y+F=_a59_.*_ffF_full_s{s}$' \
  --delta 'D+F:D' --delta 'Y+F:Y' --ref 'MPCOF eq-info=100.00,0.691,0.165,0.357,0'
```
The mean row pairs over the seeds where every arm WITH RESULTS has finished, so while Y+F is flying its n drops to
Y+F's finished seeds (a fair paired mean); an arm with no results yet shows "–".

Single FF block (pair style, Δ = last − first):
```bash
python3 scripts/ctrl_seed_table.py --style pair --seeds 1-10 \
  --arm 'D=_dmD_full_s{s}$|_n10D_full_s{s}$' --arm 'D+F=_a59_.*_ffD_F_full_s{s}$' \
  --ref 'MPCOF eq-info=100.00,0.691,0.165,0.357,0'
```
FF block on base Y alone (Y = base rows, Y+F = Y + `--yaw-trim-ff`):
```bash
python3 scripts/ctrl_seed_table.py --style pair --seeds 1-10 \
  --arm 'Y=_dmyY_full_s{s}$|_n10Y_full_s{s}$' --arm 'Y+F=_a59_.*_ffF_full_s{s}$' \
  --ref 'MPCOF eq-info=100.00,0.691,0.165,0.357,0'
```
M / Y / D delta-gain block, box style:
```bash
python3 scripts/ctrl_seed_table.py --style box --seeds 1-10 \
  --arm 'M=_dmM_full_s{s}$|_n10M_full_s{s}$' --arm 'Y=_dmyY_full_s{s}$|_n10Y_full_s{s}$' \
  --arm 'D=_dmD_full_s{s}$|_n10D_full_s{s}$' --ref 'MPCOF eq-info=100.00,0.691,0.165,0.357,0'
```

## New blocks
An arm is `LABEL=regex on the run tag`, `{s}` = seed number; alternatives with `|` let one arm span blocks
(e.g. s1-s5 from one ABI, s6-s10 from another). Only pair across ABIs that passed the determinism gate.
Print a one-line legend above the table naming each arm's recipe (e.g. `B = D · F = D + --yaw-trim-ff`).

## After the table
At most two sentences: how many seeds win, and the prereg read (direction until the pre-registered n). Times in
Eastern. Name the arm's hypothesis, not its tag.
