---
name: ids-tables
description: The two standard IDS cohort tables (TEST, GA best_f1 · margin_val_cal) — the unswr width table (C | 80 | E | B per seed) and the unswt per-arm table (F1 / FPR / acc per seed + mean + paired Δ vs ctrl, with RF/XGB rows). Use for every IDS readout, the hourly tick's IDS tables, and whenever Luiz asks "show the ids table", "unswt table", "unswr table", "width table", or "how is E / BC / V02 doing".
---

# IDS cohort tables

TWO scripts render every IDS cohort table. Run them and paste their output VERBATIM in a fenced block,
header lines included. Never retype numbers, never build an ad-hoc table, and never drop a column to save
width. If a table is too wide, change the script's layout and keep every metric, then update this skill.

```bash
python3 logs/ids/ids_width_tick_table.py    # unswr (random_3way): C | 80 | E | B, one row per seed
python3 logs/ids/ids_unswt_tick_table.py    # unswt (temporal_3way): one row per arm
```

## Rules both scripts enforce (Luiz)
- Cell = GA phase (`ga_neurons`), genome `best_f1`, `test_point final`, threshold mode `margin_val_cal`, TEST
  partition (`test_summaries.threshold_metadata`). Never a VAL number, never the max over threshold modes.
- **Metrics are ALWAYS F1 %, FPR %, acc %** (Luiz 22/11: "where are the acc and the mean?"). Plus size
  (cells/neurons) and hours. Never drop one.
- **Every arm gets a mean** over its finished seeds. Δ columns are PAIRED vs the same-seed control
  (`(better/paired)` counts; F1/acc better = higher, FPR better = lower).
- Classical rows (RF/XGB, untuned, VAL-calibrated threshold, mean±SD over seeds r20419-23, same split's TEST)
  sit on top, from the measured baseline files in `logs/ids/unsw_headroom_1611/` (Plane IDS-10). Never
  type classical numbers from memory.
- Status words for unfinished flows (`queued`/`q`, `flying`/`fly`, `paused`), never a blank or 0.
- One-line legend above the table naming each arm's recipe and its Plane story. Name the hypothesis, not a
  flow prefix.

## unswt table (`ids_unswt_tick_table.py`)
Rows = arms, columns = `F1 % s1 | s2 | s3 | mean`, `ΔF1`, `FPR % … | mean`, `ΔFPR`, `acc % … | mean`,
`Δacc`, `cells`, `hours`. Arms live in `ARMS` / `ARM_NAME` / `PLANE`; the control flows in `CONTROLS`.
The script prints TWO unswt tables: the finished IDS-73/74 block paired vs IDS-56 (`CONTROLS`), and the
**lever screens** (since 22/11: C, D24, D16, MDL, E64, flows 7618-7632) paired vs the ADOPTED E + dedupe-on
flows 7600-7602 (`LEVER_CONTROLS`, `LEVER_ARMS`, `LEVER_NAME`). New unswt arms go in the lever table.

## unswr width table (`ids_width_tick_table.py`)
Rows = seeds (+ paired mean row), columns = F1 / FPR / acc / cells-neurons / bits / hours, each cell
`C | 80 | E | B`. Add a new arm by extending `ARMS` + `ARM_NAME` + `HEADER` in the script.

## After the tables
At most two sentences per cohort: seeds won per metric, and the adoption read (direction until n=3; Luiz
decides). Times in Eastern. Post each END readout as a comment on its Plane story.
