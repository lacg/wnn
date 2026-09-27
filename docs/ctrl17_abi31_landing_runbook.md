# ABI-31 landing runbook — CTRL-17 stage-2 trainer fixes (option B)

**When:** `queue_2009c` logs `PARKED — waiting for the ABI-31 stage-2 trainer landing` in
`/private/tmp/queue_2009.log`. That happens after `_full30` x4, the CTRL-16 verdict, and the CTRL-21 probe.
It is the ONLY landing slot, because no controller run is alive and no chain is armed then.
**IDS is not touched:** this is the controller wheel only. There is no worker swap, and IDS flows are
never paused or stopped for this landing.

**What lands:** branch `stage2-trainer-fixes` @ f16ff63b: G1-G4 + G11 (altitude side included), G13 (the
`--expert-drives` raw-pwm fix), sorted (largest-first) sub-batch packing, and the `--rg-easy-tilt-deg` /
`--rg-full-tilt-deg` flags (defaults 8/30, so no run changes until an arm sets them). Audit:
`docs/ctrl17_stage2_trainer_audit.md`). Wheel staged at
`/Volumes/20260401-WDBlack-SN850X-2TB/cargo-target-stage2/wheels/ram_controller-2026.212.37-cp311-abi3-macosx_11_0_arm64.whl`
sha256 `5407561d291e1e29ad5b6b5ad1894c82c41d1042eea182bd97ef6800763af5f8` (this supersedes a65ae29a).
The merge-tree dry run was clean against b03be615.

## Steps (python + wheel in ONE step)
0. Confirm the box is idle: `pgrep -f "MacOS/Python -u -m wnn.control.phased_g[a]"` must print nothing, and
   `pgrep -f "scripts/seed_arm_chai[n].sh"` must print nothing. `queue_2009c` itself is parked; leave it alone.
1. `cd /Users/lacg/wnn && git merge origin/stage2-trainer-fixes`, then push.
2. `shasum -a 256 <wheel>` must match the sha above. Then
   `/Volumes/20260401-WDBlack-SN850X-2TB/wnn/venv/bin/python -m pip install --force-reinstall --no-deps <wheel>`
3. Check: `PYTHONPATH=src/wnn .../venv/bin/python -c "import wnn.control._accel as ra; print(ra.EXPECTED_ABI, ra.ABI_VERSION, hasattr(ra,'sample_calibration_features'))"`
   must print `31 31 True`.
4. Rust suite: `PYO3_PYTHON=.../venv/bin/python cargo test -p ram_controller --lib --no-default-features`
   (221 pass, 2 ignored). Also run `tests/test_controller_eval_ffd_parity.py` (2) and
   `tests/test_controller_eval_batch_packing.py` (24).
5. Smoke ONE tiny run with the recipe's grid flags plus:
   `--airframe cf21_brushless --translation --xy-offset 0.5 --obs-collective-cmd --obs-alt-err --obs-vz --obs-pos-err-xy --obs-vel-xy --fit-weight-pos 0.10 --fit-aggregation zscore --pop 6 --neurons-gens 1 --memory-gens 1 --skip-stages bits,connections --steps 2000 --tilt 5.0 --rg-rounds 2 --rg-episodes-per-round 4 --rg-eval-episodes 4 --num-eval-folds 5 --runs 1 --seed 31337`
   PASS iff all of these hold:
   - `[GATE-λ]` shows λ_alt=0 and λ_pos≈0.1097.
   - No guard fires.
   - MEMORY records a universe.
   - The gen line carries the position metric.
   - The fitter's xy threshold spans are non-zero.
   - The header shows `[RG-TILT] DAgger trainer tilt curriculum 8°→30° (checkpoint eval at 30°); scorer --tilt 5°`
     (defaults unchanged).
   - The packing log reads `packing FFD` if the population splits. A tiny pop fits in one sub-batch and prints
     nothing, which is fine. The first real check is the first growing BITS stage.
   Then smoke the ANCHOR recipe itself: `--obs-pwm` + PIPE_FLAGS, tiny budget, rc 0.
6. `touch experiments/ABI31_LANDED`. queue_2009c checks ABI==31 and flies `_op31` x5, then `_Lop31`.
7. Commit the merge note, comment on CTRL-17, and update the tick STATE.

## Lineage
- Attitude-only runs are byte-identical, pinned in `ctrl17_pins.rs`.
- Translation runs change: thresholds, the MEMORY universe, checkpoint selection, and the RNG stream after
  each checkpoint eval. The scorer's reward is unchanged.
- NEVER pair across the ABI-30/31 trainer boundary. `_op31` s2-s5 vs `_op30` is only the trainer read.

## Open (not in this landing)
- Still open from the audit: G5-G10 and G12.
- Tilt coherence: the flags land here; the `_tc31` arm (5°/5°, paired vs `_op31`, with a 15°/30° stress re-score) is
  approved but its queue slot is pending Luiz.
- The recorder runs without weather.
