# CTRL-45: axis-grouped receptive fields for the cascade WNN controller (DESIGN)

Status: BUILT 08/11/2026 (controller ABI 44, increments 0-3), not deployed, nothing queued.
Luiz's answers to section 9: B1/B2 fixed; `controller_conn_policy` wired (flow adapter);
family = the OUTPUT BANK, `--rf-cross` default 0.2, `--receptive-fields {off,axis}` default
off. Implementation notes that differ from the text below:
- B1 generalises: the output layer is `output_quantum` equal segments (bank, or E/I half
  under the antagonist decode), so the role map is `arch_ops::output_segment_map` and needs
  no decode knowledge. Crossover (both variants) and Lamarckian inheritance also pair output
  neurons by role when level counts differ.
- Found + fixed alongside: `spec_from_arch` dropped `feature_balance_ratio`, `conn_policy`,
  `conn_policy_min`, `conn_mutation_scope` (stages after the first GA stage ran on defaults);
  `default_controller_arch_config` passed the STATE window k to the one-frame output space
  (conn-scope `window` cut it into 44-bit pieces); the flow adapter silently dropped 23
  registered `controller_*` keys (now refused) and 11 keys it does read were unregistered.
- The operator lives in `controller/receptive_fields.rs` (`arch_project_receptive_fields`),
  not inline in arch_ops.rs.

## 1. Problem, hypothesis, constraints

**What we saw (CTRL-37/38, posx2, 35 CF2.1 units).** Each disturbance-information
ingredient helps a little on its own (leak999: +2.5 pp stable, n=3, CI [+0.1, +4.9]).
Combinations do not stack: leak+rateI and leak+rateI+dhat are flat or negative on 2 of 3
seeds, and the arms with extra inputs are unstable across seeds (`logs/controller/ctrl37_38/runs.log`).

**Hypothesis ("input dilution").** Every output neuron has a fixed 24-connection budget
(`--grid-bits 24`, sn=0, so all 24 are sampled sensor bits). `spread` draws them
uniformly from all `nf x bpf` input bits, so adding a feature takes connections away from
every other feature, including the ones that neuron's bank needs:

| layout | nf | bits per feature, spread | bits per own-axis feature, axis-RF (cross 0.2) |
|---|---|---|---|
| posx2 base | 22 | 24/22 = 1.09 | roll/pitch 20/7 = 2.9; yaw, T 20/4 = 5.0 |
| + rateI | 25 | 0.96 | roll/pitch 20/8 = 2.5 |
| + rateI + dhat | 28 | 0.86 | roll/pitch 20/9 = 2.2 |

The 14/08 forensics (`scripts/analyze_winner_connectivity.py`) already found about 33% of
(neuron, feature) pairs at only ONE threshold under spread, which is a half-plane and
cannot express "in this band". Dilution makes that worse with every feature added.

**Why this fits the cascade.** In CTBR mode each output bank is one control axis
(`[T, p_sp, q_sp, r_sp]`, decoded per bank as `motor = n / levels` in
`controller.rs::train_output_step` and `diag_raw_decode`). The firmware rate PID sits under
the WNN and roughly decouples the axes, so a p-bank neuron's label depends mostly on
roll-axis signals. That is the biology-style receptive field Luiz described, with the
family fixed by the output bank.

**Constraints.** IDS outranks this, so it never pauses IDS. Only the controller wheel
changes (no worker swap). Do not install mid-chain while the CTRL-37/38 replicate is still
running (one wheel per chain). The thermometer layout, decode, Metal kernels and MCU
export must not change.

## 2. Survey: what exists and what was learned

### 2.1 Connectivity machinery (read, not assumed)

| piece | where | what it governs |
|---|---|---|
| `--conn-policy {spread,min1,min2,min3,framed1}` | `phased_ga.py` ~L2935 (CLI), ~L3950 (resolve to `_conn_policy`/`_conn_policy_min`) | creation law for **fresh output maps only** (init + output neurogenesis) |
| `_fresh_output_suffix` / `RecurrentArchGenome.random` | `control/recurrent_genome.py` L295, L537 | dispatches to Rust `arch_ops::sample_min_per_cluster`, `sample_framed1`, `framed1_slot_schedule`, `sample_distinct` |
| `--conn-mutation-scope {free,window,feature}` | `recurrent_genome._resample_in_place` -> `arch_ops::resample_suffix_scoped` | where a CONNECTIONS rewire may land (output maps only) |
| `--feature-balance-ratio` | `arch_ops::rebalance_features`, applied at one point: `RecurrentArchGenome._apply_feature_balance` (end of `mutate()`) and in `random()` | global per-feature cap (a re-projection after every operator) |
| `--bits-per-feature` (8) | input geometry: feature `f` owns bits `[f*bpf, (f+1)*bpf)`; `arch_ops::feature_of = (idx % frame_bits) / bpf` | thermometer runs are contiguous, so a family is a union of contiguous runs |
| `--suffix-coverage` | `controller_grid_search._compute_valid_pairs` | per-layer suffix width as a fraction of the input span |
| crossover | `RecurrentArchGenome.crossover` (whole-neuron, same index, `_mix_blocks`), `crossover_average` (same index + `_resize_suffix` free pad) | |
| BITS grow | `set_output_suffix` -> `_resize_suffix` -> free `_sample_distinct` | |
| RecurrentArchConfig builders | **two**: `arch_strategy.default_controller_arch_config(spec)` and an inline one in `controller_grid_search._random_genome` (L275) | see bug B2 |

Feature order is defined once, in Rust `WnnController::compute_features`
(`controller.rs` ~L5797), and mirrored by the Metal twin. Python only duplicates the
**count** (`ControllerSpec.num_features()`, `evaluator.py` ~L375). Two scripts each keep
their own partial **name** list (`scripts/analyze_winner_connectivity.py::feature_names`,
`scripts/analyze_winner_features.py::feature_names`); neither covers the cascade blocks.

Runtime consumers of a connection map (CPU `step`, the Metal rollout and score kernels,
DAgger and BPTT trainers, `scripts/export_controller_c.py` via `genome.to_connections()`)
only see flat bit **indices**. A receptive-field constraint changes which indices get
chosen. It changes no consumer.

### 2.2 What earlier programmes found (so we don't repeat a refuted idea)

- **Specialist round 1 (14-16/08, `experiments/specialist1_markers/`, n=1, PWM motor
  mode, b=15, stale-altitude void era):** spread 87.6% / 3.70 deg; **min2 47.8% / 8.10 deg; min3
  0.0% / 46.7 deg**. Concentrating a neuron on FEWER, RANDOMLY chosen features was
  catastrophic. Axis-RF is not that idea:
  (a) the kept features are the bank's own axis, not a random subset;
  (b) the experiment runs in CTBR, where a bank IS an axis. In PWM mode every motor
  needs every axis, so a RANDOM drop takes away needed signal by construction;
  (c) a cross budget keeps mixed selectivity.
  It is still the main risk (section 8, R1). The decision rule includes a kill criterion.
- **Arm D / D40 (full-window output):** spreading the budget over more inputs (k frames)
  hurt (steady 1.34 -> 10.17 deg). This is dilution in time, the same direction as the
  hypothesis here.
- **The L4 "committee" (08-09/08)** was a committee of WHOLE controllers distilled from
  different teachers (mean-PWM vote; `docs/l4_teacher_screen_results.md`). It moved the
  hold floor through diversity. It is a different mechanism, not a within-network
  specialist split, so nothing here repeats it.
- **Offline connectivity probe (CLOSED 15/09):** per-neuron predictability proxies do not
  rank closed-loop layouts (rho 0.3). So **do not** pick families from an offline
  information measure. The family map is declared from physics (section 3) and judged
  only in closed loop.
- **Feature-balance cap (26/06):** addressed one feature over-wiring (obs_yaw_err 2.14x).
  It is a global cap and conflicts with RF by design, so the combination is refused
  (section 4.4).

## 3. Decision 1: the feature-family map

**One source of truth, declared next to the feature definitions, as data, not as a
hardcoded index list.** New module `src/wnn/control/feature_layout.py`:

```python
class AxisFamily(IntFlag):          # enum lives with its code
	NONE = 0; ROLL = 1; PITCH = 2; YAW = 4; THRUST = 8

@dataclass(frozen=True)
class FeatureDesc:                   # tightly-coupled helper (allowed in the same file)
	name: str
	family: AxisFamily               # bitmask: a feature may belong to >1 family

BANK_FAMILIES: tuple[AxisFamily, ...] = (THRUST, ROLL, PITCH, YAW)   # CTBR [T,p,q,r] == decouple [T,tr,tp,ty]

def feature_layout(spec: ControllerSpec) -> tuple[FeatureDesc, ...]:
	"""compute_features' canonical order, each block gated by the SAME spec flag."""
```

- `ControllerSpec.num_features()` becomes `len(feature_layout(self))`. This removes the
  duplicated count expression.
- Both analysis scripts import `feature_layout` and delete their local `feature_names`.
  The duplicates are folded into this one module.
- The Rust `compute_features` stays the ORDER authority (hot path; Metal twin). The guard
  is a parity test, `tests/controller_feature_layout.py`. It (i) sweeps every obs-flag
  combination and asserts `WnnController.num_features() == len(feature_layout(spec))`.
  (ii) It pins ORDER for the injectable blocks: set sentinel values with
  `set_vertical_obs` / `set_horizontal_obs`, step once, and assert that
  `get_last_feature_vector()` has them at the indices the layout names (`coll_cmd`,
  `alt_err`, `vz`, `e_x`, `e_y`, `v_x`, `v_y`).

**Family table (recipe features first; blocks gated as in compute_features):**

| feature(s) | family | why |
|---|---|---|
| gyro_x / gyro_y / gyro_z | ROLL / PITCH / YAW | body rates p, q, r |
| acc_x / acc_y | PITCH / ROLL | specific force (ABI 32) + rotor drag (0.78): body-x/y force is the translational signal of the pitch/roll axis |
| acc_z | THRUST | |
| tgt_r / tgt_p / tgt_y | ROLL / PITCH / YAW | per-axis reference |
| tilt, tilt_i (off in recipe) | ROLL or PITCH | coupled magnitude |
| roll_err, roll_i / pitch_err, pitch_i | ROLL / PITCH | |
| yaw_pa, yaw_i_pa (off in recipe) | YAW | |
| pwm_0..3 (stripped by the chain's sed; off) | all four | motor space mixes every axis |
| yaw_err, yaw_err_i | YAW | |
| L1 dhat_r/p/y (off) | ROLL / PITCH / YAW | |
| coll_cmd, alt_err, vz | THRUST | |
| e_x, v_x / e_y, v_y | PITCH / ROLL | world frame, but the recipe's initial yaw is bounded by `--tilt 5` (`phased_ga.py` L3750: max_initial_yaw = radians(tilt)), so world-x is approximately body pitch (cos 5 deg = 0.996) |
| rate_i x/y/z (CTRL-38 a1) | ROLL / PITCH / YAW | inner integrator per axis |
| cascade dhat x/y/z (a2) | ROLL / PITCH / YAW | |

Recipe family sizes (22 features): ROLL 7, PITCH 7, YAW 4, THRUST 4. Each of rateI and
dhat adds one feature to each of ROLL, PITCH and YAW.

**Cross-axis signals** (position -> attitude, yaw coupling, the thrust tilt-loss) are not
modelled as extra family edges. They go through the **cross budget** (section 4). That
keeps the map a one-line-per-feature table, and lets selection decide what crosses.

**Review hook:** the acc_x/acc_y and e/v assignments should get a 10-minute
`flight-dynamics` check before build (sign does not matter, only the axis does).

## 4. Decisions 2 and 3: assignment and GA interaction

### 4.1 Assignment: family fixed by the OUTPUT BANK (recommended)

A neuron's family = `BANK_FAMILIES[n // levels_per_motor]`. The T bank gets THRUST.

Why bank-keyed and not "first connection picks the family" (drawn per neuron):
- In CTBR the bank already says which axis the neuron serves. A drawn family would make
  about 3/4 of each bank's neurons specialists for the WRONG axis. That would be worse
  than spread for the bank's own label, which is the min2/min3 failure again.
- It needs no storage: the family can be derived from (index, n_out, mode). So the genome
  and checkpoint formats are unchanged (section 5).
- Mixed selectivity survives through the cross budget.

`--rf-cross c` (default 0.2) means each output neuron may hold **at most**
`floor(c x width)` connections outside its family (24 bits -> 4). It is a **count** budget,
not reserved slot positions. That keeps it independent of position under crossover,
BITS resize and neurogenesis, and keeps the operator idempotent.

A drawn family (Luiz's literal version) is the right shape for **PWM mode**, where a motor
neuron legitimately needs every axis. It needs a stored per-neuron tag, so it is deferred
as a follow-up (`--receptive-fields drawn`). It is not part of CTRL-45.

### 4.2 Mechanism: one projection operator, applied at the single enforcement points

This copies the `_apply_feature_balance` pattern (one re-projection after any operator)
instead of teaching each operator about families:

```
project_receptive_fields(output suffixes, bank families, feature families, budget):
  for each output neuron n (family F = bank_families[n // L]):
    out = connections whose feature's mask & F == 0
    if |out| > budget: keep `budget` of them (counter-RNG pick), redraw the rest
                       uniformly from F's unused bits (partial Fisher-Yates)
  return the indices of changed neurons   (-> their cells are dropped, as elsewhere)
```

- Idempotent. A no-op on a valid genome. When the mode is off: **no RNG draw at all**, so
  every banked stream reproduces bit for bit (pin with a test).
- If F's pool is smaller than `width - budget` (only possible after a large BITS grow),
  the deficit is allowed out-of-family. That is a deterministic, documented fallback
  decided in Rust, the same convention as `sample_min_per_cluster`.

**Where it is applied** (every path that can create or move an output connection):

| site | file | covers |
|---|---|---|
| end of `RecurrentArchGenome.random` | recurrent_genome.py | grid init, `_random_arch_genome` (GA/TS init) |
| end of `mutate()` (next to `_apply_feature_balance`) | recurrent_genome.py | NEURONS (neurogenesis appends), BITS (free `_resize_suffix` grow), CONNECTIONS (any scope) |
| end of `crossover` and `crossover_average` (new `config` param) | recurrent_genome.py + `arch_strategy.py` L250/L543/L559 pass `self._arch_config` | different-L parents mix banks at the same index; the projection repairs, and Lamarckian `_filter_inherited_cells` already drops cells of any neuron that differs from parent1 |
| exit of stats-guided genesis | `arch_adaptation.py` L254/L293 | `set_output_suffix` / `rewire_suffix` |

At init, the projection turns a uniform draw into exactly "`budget` random cross
connections + the rest uniform in-family". That is the intended creation law, so a
separate sampler is unnecessary.

### 4.3 Placement and interfaces

**Rust (crate `ram_controller`, `controller/arch_ops.rs`, next to `rebalance_features`):**

```rust
/// Axis receptive fields (CTRL-45). Returns the changed neuron indices.
#[allow(clippy::too_many_arguments)]
pub fn project_receptive_fields(
	sampled: &mut [i64], offsets: &[usize],     // flat output suffixes (as rebalance_features)
	bank_family: &[u8], levels: usize,          // neuron n -> bank_family[n / levels]
	feature_family: &[u8],                      // per feature, len = frame_bits / bpf
	frame_bits: usize, bpf: usize,
	max_cross: usize,
	seed: u64, generation: u64, genome: u64, layer: u64,
) -> Vec<u32>
```

Sub-draw coordinates must be disjoint from the existing operators in the module (for
example `0x4000 + n`). PyO3 wrapper `arch_project_receptive_fields` goes in
`controller/lib.rs` (registered with the other `arch_*` functions at ~L1294). Python
reaches it only through `wnn.control._accel`.

**Python config (typed, no kwargs).** `RecurrentArchConfig` stays domain-agnostic. It
holds group masks, not axis names:

```python
receptive_fields: str = "off"                 # "off" | "axis"
rf_cross: float = 0.2
rf_bank_masks: tuple[int, ...] = ()           # len = num banks; from BANK_FAMILIES
rf_feature_masks: tuple[int, ...] = ()        # len = nf; from feature_layout(spec)
```

`ControllerSpec` gains `receptive_fields: str = "off"` and `rf_cross: float = 0.2`.
These are serialized with the spec, and old checkpoints load as off.
`default_controller_arch_config(spec)` fills the masks from `feature_layout(spec)` and
`BANK_FAMILIES`.

### 4.4 Composition with existing knobs: an orthogonal flag, NOT a new conn-policy value

`conn_policy` is a **creation law** (init + neurogenesis only). By construction it cannot
govern mutation, crossover or BITS, which is exactly the gap that would let the GA undo
the grouping. RF is an **invariant**, enforced after every operator. As a conn-policy value
it would also rule out "min1 within family" later.

| combination | v1 behaviour |
|---|---|
| `--conn-policy spread` | supported (the CTRL-45 arms) |
| `--conn-policy min1/min2/min3/framed1` | **refused** at CLI resolve (SystemExit). The projection would break min-m structure; add a family-aware min-m only if v1 earns it |
| `--feature-balance-ratio > 1` | **refused** (a global per-feature cap contradicts per-bank families) |
| `--conn-mutation-scope free/window/feature` | supported. `feature` keeps a cross connection in its own feature, `free` is repaired by the projection |
| `--output-mode pwm` without `--decouple-outputs` | **refused** (banks are motors, not axes). The drawn variant is the future answer |
| state layer (sn > 0) | untouched (same convention as conn_policy/scope). The recipe is sn = 0 |

CLI (`phased_ga.py`, next to `--conn-policy`): `--receptive-fields {off,axis}` (default
off) and `--rf-cross FLOAT` (default 0.2, validated in [0, 1]). Startup prints one
`[rf] axis: bank families T/R/P/Y = 4/7/7/4 features, cross budget 4/24` line, so a
marker records what was flown.

## 5. Decision 4: CPU/GPU parity and checkpoints

- **Parity:** no kernel, trainer, decode or exporter consumes families. They consume
  indices. The CPU paths and the Metal paths receive the same connection vector, so
  parity holds by construction. The 14 existing CPU/GPU sweeps (`cargo test -p
  ram_controller --lib --no-default-features`) need no change and must stay green. The
  new operator is a host-side GA operator: like every `arch_ops` function, it has no GPU
  twin, and there is no "GPU path" to port. New cargo tests: invariant holds after
  projection; idempotent; budget 0 = pure family; budget = width = no-op; deterministic
  under the counter RNG; off-mode never called.
- **Checkpoints:** the grouping is **derivable** from (spec obs flags -> layout, mode,
  rf_cross, neuron index, levels). Nothing goes into the genome and the
  `serialize()`/`deserialize()` format is unchanged. A resumed RF run re-derives the
  masks from its spec. The projection is idempotent, so a valid resumed population is
  untouched (note: resume already re-seeds, per memory).
- **MCU exporter:** `scripts/export_controller_c.py` reads `genome.to_connections()` and
  the feature flags. It is unaffected (confirmed L149, L110).
- **Deploy:** ram_controller **ABI bump** (new symbol + the cells op from section 7, inc
  0). Rebuild the controller wheel only (`maturin develop --release -m
  controller/Cargo.toml`). No worker swap, ram_core unchanged. Install at a chain
  boundary, never under a running CTRL-37/38 run.
- **Full-stack path:** CLI `--receptive-fields/--rf-cross` -> `args` -> `ControllerSpec`
  (`phased_ga` spec builder ~L474/L525 and `controller_grid_search._spec` L189) ->
  `default_controller_arch_config` -> `RecurrentArchConfig` -> `_accel` ->
  `arch_project_receptive_fields`. Flows: register `controller_receptive_fields` and
  `controller_rf_cross` in `wnn/ram/experiments/params.py` `KNOWN_PARAMS`, and add them
  to `flow_adapter.py`'s cascade-only refusal tuple (L83, same as
  `controller_obs_rate_i`). The adapter refuses CTBR, so no Svelte change.

## 6. Decision 5: experiment plan (pre-registered)

**Recipe.** Exactly the CTRL-37/38 posx2 command line: `MA` from
`experiments/ctrl22/ARGS_ma_anchor_lag020.txt` with the chain's sed (which strips
`--obs-pwm`, the save flags and the test seeds), plus `--motor-lag-s 0.20 --output-mode
ctbr --action-repeat 10 --ctbr-delta-law --delta-leak 0.7 --delta-max 0.6 --rg-gate off`,
posx2 weights `.2557/.2045/.3580/.1818`, `--test-seeds 99990101..99990135` (35 units),
`--num-eval-folds 5`, pop 50. All arms run on ONE wheel (including the inc-0 fix).
Seeds **s1..s3 = `--base-seed 31337001..31337003`**. s2/s3 share base seeds with the
banked CTRL-37/38 rows, but the wheel differs, so CTRL-45 rows are **never** paired with
banked rows.

**Arms (6 x 3 = 18 runs, about 1 h each, about 18 h serial):**

| arm | receptive fields | ingredients |
|---|---|---|
| a | spread | none (baseline) |
| b | axis, cross 0.2 | none |
| c | axis | leak999 |
| d | axis | leak999 + rateI |
| e | spread | leak999 |
| f | spread | leak999 + rateI |

Luiz's four arms are a-d. **e and f are added** because "ingredients now stack under RF"
is an **interaction** claim. Without same-wheel, same-seed spread twins, the leak and
rateI increments under RF have nothing to be compared against: the banked spread rows are
a different wheel and the s1 seed is new. dhat is **deferred**. It was the weakest
ingredient alone (stable 83.0 / 83.4 vs baseline 86.5 / 89.6 on s3/s4) and becomes a
one-arm follow-up (`d + dhat`, plus its spread twin) only if d stacks.
Fallback if 18 h is too long: a-d plus e (15 runs), dropping the f interaction.

**Order:** s1 for a, b first (go/no-go, below). Then the other arms at s1, then s2 and s3.

**Decision unit and metrics.** For each seed, the paired delta of an arm vs its reference
on the 35-unit headline row (stable %, err deg, steady deg; pos only tie-breaks; alt not
reported). n = 3 seeds. Report mean +/- t-CI (t_{0.975,2} = 4.30). Per-unit (35 x 3)
deltas are supporting only, because units share a genome within a seed. The attitude
block is judged as a whole (Luiz 05/11).

- **H1 (RF works), b vs a.** WIN if Delta-stable mean > 0 with 3/3 seeds >= 0 (or CI
  excluding 0), AND non-inferior on the other two: Delta-err mean <= +0.10 deg and
  Delta-steady mean <= +0.10 deg. LOSS if Delta-stable mean <= -1.0 pp with 3/3 seeds < 0, or
  err or steady worse by > 0.10 deg on 3/3. Otherwise TIE.
- **H2 (stacking under RF).** c vs b (leak adds under RF) and d vs c (rateI adds on top),
  each judged by the H1 rule. "Stacks" = both WIN, or one WIN and one TIE with a positive
  mean.
- **H3 (interaction, exploratory).** (d - c) - (f - e) on stable, and the same on steady.
  Claim "RF changes how ingredients combine" only if its CI excludes 0. Expect it not to
  at n = 3, so report direction only.
- **Adoption:** RF becomes the cascade default only if H1 WINs. Adopting d's stack needs
  H1 WIN and H2 "stacks".
- **Go/no-go after s1 (a, b):** if b collapses (stable < 70% where a >= 80%, the
  min3-style failure), stop and report; do not spend the remaining 16 runs. If b is within
  +/-2 pp, continue.
- **Kill / dose rule:** if H1 is LOSS, close RF at cross 0.2. If H1 is TIE, one dose arm
  (cross 0.4, 3 seeds, vs a) before closing. No further dose search beyond that.

## 7. Build order and minimal first increment

**Increment 0, prerequisite bug fixes (section 9, B1 + B2). Needed before any RF code,
and it changes the baseline, so all CTRL-45 arms fly on it.**
- B1 fix: output neurogenesis becomes bank-aware. `_resize_output_neurons` and
  `set_output_neurons` (recurrent_genome.py L924/L1012) grow and shrink **at the tail of
  each bank's half**: antagonist halves E|I under BINARY (quantum 2 x num_motors = +1 per
  half per bank), whole bank under cumulative. In-half rank is preserved, so a neuron's
  threshold i/L shifts only to i/L'. A new Rust cell op `GenomeCells.reindex_output_neurons(old_to_new:
  Vec<i64>)` (-1 = drop) goes in `controller/cell_remap.rs` + `genome_cells.rs`, so
  Lamarckian cells follow their neuron. Tests: a 256 -> 264 -> 256 round trip keeps every
  surviving neuron's (bank, half, rank) and cells.
- B2 fix: delete the inline `RecurrentArchConfig(...)` in
  `controller_grid_search._random_genome` and route it through the ONE builder
  (`default_controller_arch_config(spec)`), so grid init gets `input_window_k`,
  `conn_mutation_scope` and the new RF masks from the same place as the GA stages.
- Size: about 2.5 agent-hours.

**Increment 1: feature layout as one source of truth.** `feature_layout.py` +
`num_features()` via the layout + scripts deduplicated + parity test. About 1.5
agent-hours. Ships alone (no behaviour change, no ABI).

**Increment 2: Rust operator.** `arch_ops::project_receptive_fields` + PyO3 + cargo
tests + ABI bump. About 1.5 agent-hours.

**Increment 3: wiring.** Config fields, the four enforcement sites, CLI + refusals,
ControllerSpec, KNOWN_PARAMS + adapter refusal, startup line, and a Python test that every
genome after random / mutate (all dims) / crossover / crossover_average satisfies the
invariant, and that off-mode is bit-identical to today (same RNG stream). About 2
agent-hours.

**Increment 4: verification before the cohort.** Smoke at pop 6 (plumbing only, per
Luiz 31/10). Run `analyze_winner_connectivity.py` on the smoke winner: per-bank in-family
share must be >= (width - budget)/width. One pre-deploy wheel review. Write the chain
script `logs/controller/ctrl45/chain.sh` modelled on `ctrl37_38/chain.sh`. About 1
agent-hour.

**Total about 8.5 agent-hours of build, plus about 18 h of flying (15 h fallback).**
The minimal first increment that answers the main question is inc 0 + 1 + 2 + 3, then
arms a and b at s1 only.

## 8. Risks

- **R1, concentration can hurt (min2/min3 precedent).** Mitigated by axis-aligned, not
  random, families; CTBR only; the cross budget; and the go/no-go after s1.
- **R2, too little partial connectivity in small families.** The THRUST and YAW families
  are 4 features = 32 bits. With 20 in-family connections, every T/yaw neuron sees 62% of
  its family, so neurons in those banks overlap heavily, and generalization leans on
  thermometer quantization (at most 9^4 = 6561 patterns) rather than on partial
  connectivity. Watch per-bank errors (yaw err, alt). If T/yaw regress while roll/pitch
  improve, the next knob is a family-coverage cap. Do not pre-build it.
- **R3, cross-axis needs bigger than the budget** (yaw coupling into roll/pitch at
  larger yaw, position under yaw). The recipe bounds initial yaw to 5 deg. The cross
  budget is the escape valve, and the dose arm tests it.
- **R4, connectivity is not searched in this recipe** (`--skip-stages connections,bits`).
  RF acts at grid init and through neurogenesis only, so the experiment tests the
  CREATION LAW plus selection over it. That is the same scope as specialist round 1, so
  the comparison is fair.
- **R5, baseline moves (inc-0 fix).** Handled: a fresh same-wheel baseline (arm a), and
  no pairing with banked rows.
- **R6, RNG drift for unrelated runs.** The off-mode no-draw test guards it.

## 9. Flags for Luiz (decide before build)

1. **B1, suspected pre-existing bug (high confidence from reading; verify with a probe
   before fixing).** Output neurogenesis appends and truncates at the **tail**
   (`recurrent_genome.py` `_resize_output_neurons` L1012, `set_output_neurons` L924;
   `drop_output_neurons_ge` on shrink). The controller assigns banks **by index**
   (`motor = n / levels`, `controller.rs` `train_output_step`; antagonist E/I half =
   `level_idx < L/2`). A 256 -> 264 grow therefore moves nearly every neuron outside bank 0
   to a different bank, level or half. Under `--lamarckian` its inherited cells now drive
   the wrong bank or E/I side until DAgger overwrites the visited addresses. The posx2
   recipe runs this path: 15 NEURONS generations, `--lamarckian`,
   `--max-output-neurons 512`, and NEURONS#k often holds the headline. Under spread the
   CONNECTION maps are bank-agnostic, so only the cells suffer. Under any per-bank
   structure (framed1 per-motor quotas, and RF) the structure itself drifts.
   Probe: grow a cell-carrying 256-neuron genome by one quantum and check that neuron 64's
   cells now decode in bank 0. Fixing it changes the baseline recipe.
2. **B2, pre-existing bug (high confidence).** `controller_grid_search._random_genome`
   (L275-279) builds its own `RecurrentArchConfig` **without `input_window_k`**. So
   `--conn-policy framed1` at GRID drew with k=1 = min1 over the whole K-window, not one
   frame. The SL_C window k-ladder framed1 arms (`scripts/sweep_ladder_chain.sh`) were
   therefore not framed at init: only neurogenesis appends were. Root cause: two config
   builders (a duplicate). Fix by keeping one. Whether to re-fly the ladder is Luiz's call.
3. **Minor:** `KNOWN_PARAMS` has `controller_conn_policy`, but nothing in
   `flow_adapter.py` reads it (dead registry entry; its comment lists only
   spread|min2|min3). Either read and refuse it like the cascade keys, or remove it.
4. **Bank-keyed vs drawn family:** bank-keyed recommended for CTBR (section 4.1); drawn
   deferred to PWM mode. Confirm.
5. **Arms e/f added** (interaction twins). Confirm 18 runs, or the 15-run fallback.
6. **Architecture integrity:** this is NOT a new RAM-neuron type, committee object or
   memory format. Memory, cells, decode and the RAM lookup are untouched. RF is a
   constraint on the existing output layer's connectivity map, implemented as one GA
   connectivity operator beside `rebalance_features`. That is a deliberate extension of
   connectivity, which CLAUDE.md names as the generalization mechanism. No CLAUDE.md rule
   is violated. The one judgement call is that the feature ORDER authority stays in Rust
   `compute_features` while the NAME/FAMILY table lives in Python, guarded by a parity
   test (section 3). The alternative (a Rust-owned table) would need a typed Rust obs-flag
   struct that does not exist today (`new_core` takes 20+ bare bools).

## 10. Rejected alternatives

- **New conn-policy value `axis`:** a creation law cannot constrain mutation, crossover
  or BITS, so the GA would undo the grouping (section 4.4).
- **Family-aware versions of each operator** (resample, resize, crossover, append):
  four or more code paths to keep in sync. One idempotent projection covers them all.
- **Reserved cross SLOT positions** (first k slots free): breaks under BITS resize and
  shrink. A count budget does not.
- **Drawn per-neuron family in CTBR:** most neurons would serve the wrong axis, and it
  needs a stored tag (section 4.1).
- **Graded "closely related" feature distances:** more parameters, no evidence. Bitmask
  families plus a cross budget express the same thing.
- **Families chosen by an offline information proxy:** refuted mechanism (rho 0.3, probe
  closed 15/09).
- **A per-bank sub-network or committee of specialist controllers:** a new structural
  object (needs discussion under Architecture Integrity) for a question a connectivity
  constraint can answer first.
- **Storing families in the genome:** they can be derived; storing them changes the
  checkpoint format for no gain.
