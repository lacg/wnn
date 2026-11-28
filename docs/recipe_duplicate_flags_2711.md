# Duplicated flags in the controller recipes (27/11/2026)

Luiz 27/11: "let's clean up by leaving the last one". argparse keeps the LAST value of a repeated
flag, so every flown recipe since the ladder era ran on the last value. This note records the
values that were actually in force and which statements were wrong.

## Effective values (base l2b `_a63_..._gt_l2b_full_sN`, its G+T control `_a62_..._n0g3gz_full_sN`, CTRL-80 arms)

| flag | first (never applied) | effective (last) | origin of the override |
|---|---|---|---|
| `--skip-stages` | `neurons,bits` | `connections,bits` (grid -> NEURONS -> MEMORY) | `_pipeN` arm 15/09, promoted 22/09 (`_pipeN30` = anchor + recipe) |
| `--max-cells` | `180000` | `1000000000` (= the default, off) | same `_pipeN` recipe ("--max-cells off") |
| `--max-output-neurons` | `256` | `512` | same `_pipeN` recipe |
| `--neurons-gens` | `5` | `15` | CTRL-8 (01/11) |
| `--neurons-patience` | `3` | `3` (same value) | CTRL-8 (01/11) |
| `--fit-weight-err-sq / -stable / -steady / -pos` | `0.28125 / 0.225 / 0.39375 / 0.1` (S16noJM x0.9 + w_pos 0.10) | `0.2557 / 0.2045 / 0.3580 / 0.1818` (`wposx2`) | CTRL-30 (07/11), the `C-S2-d_wposx2` lineage |

Every override was deliberate when it was added (the ladder chains append arm flags after the
ladder's own, documented as last-wins in `scripts/seed_arm_chain.sh`; `docs/multi_axis_programme_spec.md`
§1 already says "cells uncapped"). The fault was carrying the dead first copy forward:
`rebuild_argv.py fly` copies a flown `[argv]` verbatim, so the duplicates rode into every CTRL-48+
run. 615 of 619 argv-bearing `.out` files (27/10 to 27/11) carry them. The first-copy values
were the effective ones only on the pre-`_pipeN` ladder runs (`_hd`, `_hd29`, ...), where they
stood alone.

`--max-cells-strict` is inert under the effective value: it only clamps a bits-grow so that the
post-grow count fits `max_cells`, and with `max_cells` at 1e9 (and the BITS stage skipped) nothing
is ever clamped. The 180k cap would have bound if it had been in force: with NEURONS running, neuron
grows stop at the budget, and the current winners carry ~0.6-1.6 M cells.

## Statements that were wrong

- `.claude/agents/controller.md`: "180k current study cap". That cap belongs to the dfa1l / L4-teacher
  era. The current lineage is uncapped. (Not edited here: agent definition.)
- Memory note `reference_controller_tick_format.md`: "`--skip-stages neurons,bits` means NEURONS never
  emits a gen line". That held only on the pre-22/09 ladder. The current base runs GRID / NEURONS / MEMORY,
  and its headline genome can be `NEURONS#k`.
- No adopted lever, prereg or results doc claims a 180k budget or a neurons-skipping plan for any run
  after 22/09. The `docs/multi_axis_programme_spec.md` row "Anchor `_hd29` (... 180k-cell cap)" is correct
  for `_hd29`.

## Paired comparisons: none invalidated

Across all 57 driver pairs (runs.log `START ... control X`, CTRL-48 onward), arm and control carry
identical sequences of every duplicated flag (scan of every `.out` `[argv]` line). The drivers enforce
byte-identical argv minus the lever flags. The ladder-era pairs that did differ in these flags
(`_pipeN` / `_full30` vs `_hd` / `_hd29`) were the stage-pipeline A/B itself, so the difference was
the lever under test, and the control's 180k was inert there because CONNECTIONS never grows structure.

## Going forward

- `logs/controller/ctrl48/rebuild_argv.py` dedupes, keeping each flag's last occurrence at its own
  position, so argparse resolves the same args. It reports each dropped copy on stderr. The new
  `argv` mode prints the deduped flown argv. The output is refused if it still carries a flag twice
  or carries both `--x` and `--no-x`.
- `logs/controller/ctrl80/ab_ylead80.sh` compares the arm against the DEDUPED control argv. It aborts
  unless the raw and deduped control argv give the same full resolver line, digest included.
- Proof: phased_ga's parser + `_resolve_ctbr` + `resolve_block10_args` + calibration token, full
  `vars(args)` digest. Raw vs deduped are identical for l2b s1-s3, G+T s1-s3, l2bylead s1-s3 and
  l2byleadngz s1-s3. Old-rebuild (= the flown arm argv) vs new-rebuild arm argv are identical for
  l2bylead, l2byleadngz and ylead s1-s3. `Y80_DRY=1 Y80_ARMS=ylead` passes.
