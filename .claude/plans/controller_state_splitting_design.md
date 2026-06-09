# Design: Conflict-Driven State-Splitting Trainer for the WNN Attitude Controller

Status: **DESIGN COMPLETE, NOT IMPLEMENTED** (09/06/2026). Rule 1 — designed first.
Supersedes: option A (the integral hack, `WNN_STATE_INTEGRAL_TARGET`) and the fitness-pressure idea.
Resume memory: `project_controller_state_splitting`. Builds on `project_controller_stability_diagnosis`, `project_lamarckian_redundancy_finding`.

---

## 1. The problem (data-backed)

The WNN attitude controller settles to a flat **steady-state offset** (5° stability ≈83% vs PID 98%; 0% divergence; soft-fails hold ~5.6° flat, tail-std≈0). Root cause: the recurrent **state layer is effectively untrained** → the controller is a memoryless proportional controller → no integral action → cannot drive residual error to 0.

The deeper reason the state never learns: the current state credit signal `d_out` (controller.rs:1102, "what state bit would make the output match PID") is **trivially satisfiable by a proportional controller** → state carries no useful information → the GA drops it (collapses to ~1 cell, memoryless). Option A "fixed" this by force-feeding a hand-designed integral thermometer (`npa` = 3 axes × sn/3 levels), which works at small `sn` (sn=6: 34→46%) but **over-constrains at sn≥9** (degenerate ~36%) because the encoding is feature-specific and imposed, not earned.

## 2. The principle

**Conflict-driven state-splitting** = constructive Mealy/DFA induction from trajectories. The state **emerges from the data's need to disambiguate** — minimal, nothing hand-designed, nothing pressured:

> Process the trajectory. Train `output(input, state) → target_pwm`. When two steps need **different** outputs from the **same** `(input, state)`, the state can't represent it → **split**: add a distinction that separates the two histories. Continue.

Family placement: this is the **U-Tree / utile-distinction** cousin (McCallum, RL/POMDP hidden-state) — *not* the grammatical-inference cousins (RPNI/EDSM/L\*). The split **trigger is a utility** (same `(frame,state)` forced to two different *PWMs*), not a language-membership label. That is the right family for a control task. The grammatical-inference side only informs the *backward-consistency mechanics*.

## 3. Substrate mapping — the controller already IS a Mealy machine

From `controller.rs:965-1193 bptt_train_window`:
- **Output layer** = `OUT[frame, state] → pwm` — the Mealy output function (commit loop 1170-1189, direct PID-PWM target).
- **State layer** = `G[sensor_window, state] → state'` — the transition function (commit loop 1139-1167).
- **State** = `sn` neuron MSBs (1-bit/neuron since the 08/06 change; `state_bits_in = self.state_neurons`). State ∈ {0,1}^sn, realized sparsely via RAM addresses.
- Inputs: 9 features × 8 thermometer bits × 4-frame window = 288-bit sensor input (gyro 3 + accel 3 + target_rpy 3). Attitude is **implicit** (from accel/gyro), never given directly.
- Cells are **QSR** (4-state): FALSE=0=0.0, WEAK_FALSE=1=0.25, WEAK_TRUE=2=0.75, TRUE=3=1.0. Emitted side = MSB `(v>>1)&1` → boundary between WEAK_FALSE (MSB 0) and WEAK_TRUE (MSB 1), i.e. the 0.5 line.
- Training is **full** (ratio 1.0 — every state & output neuron every step). The IDS `neuron_sample_rate=0.25` does **not** apply here (verified: no sampling in controller.rs). Correct — the net is tiny (sn∈3..12, output = num_motors×levels = 4×16 = 64).

## 4. Three departures from textbook U-Tree (the substrate-specific design)

### Departure 1 — the unit of a split is a CELL, not a node (nor a whole neuron)
A state neuron reads `cell = read_cell(n, addr)`, `addr = compute_address(sensor_window, prev_state)`. One neuron carries 2^bits distinct states across its address space. A split writes **one cell** at the divergence address on an existing neuron — not a fresh neuron. `sn` neurons × 2^bits addresses = enormous, sparsely-used state capacity. Budget is effectively non-binding via cells.

### Departure 2 — a split is a SEGMENT operation `[t*, t]`, not a point
The conflict surfaces at `t`; the disambiguating divergence is back at `t*`. A point-write at `t*` evaporates one step later: `state_{t*+1} = G[sensor_window_{t*}, state_{t*}]` hits a fresh address → EMPTY (WEAK_FALSE, MSB 0) → bit drops. So a split must also write **carry-cells** on G across `t*+1 … t-1` to *hold* the bit forward to `t`. **This forward-carry is the real cost** ("backward propagation of splits"), not the backward search.

### Departure 3 — feasibility is gated by CONNECTIVITY (trainer ↔ GA coupling)
The set-cell at `t*` can only separate two histories if the chosen neuron's connections **include a bit that differs** between their sensor windows. No discriminating bit observed → split physically impossible. Memory writes (trainer) can only realize distinctions that connectivity (GA) makes observable — the two learning components meet exactly here. Unmet distinctions become **GA pressure** (see §8).

## 5. Carry mechanics — the QSR latch

A carry = neuron `c` implementing **"if I was on, stay on"**: it observes `prev_state[c]`, and the cell where `prev_state[c]=1` emits TRUE. Set-cell at `t*` + hold-cells downstream = a **set/hold/reset latch** — the canonical 1-bit memory. Neuron `c` observes `{prev_state[c], one sensor trigger bit}`; three cells: **set** (trigger active → nudge up), **hold** (trigger idle, prev=1 → nudge up / no-op), **reset** (opposite trigger → nudge down).

**QSR makes it a debounced / Schmitt latch (not an instantaneous flip-flop).** Nudges move **one step** (`nudge_toward_value`, controller_training.rs:685). The emitted side flips at the WEAK_FALSE↔WEAK_TRUE step. So:
- A **strongly-held** bit (TRUE=3) survives one contradictory nudge (→WEAK_TRUE=2, still MSB 1); needs **two consecutive** opposite nudges to flip. Hold-strength = hysteresis depth = anti-chatter for free.
- A freshly-set (weak) bit flips easily → reinforce (saturate) to lock it.
- "Evaporation" only at **genuinely-untouched** addresses (WEAK_FALSE, MSB 0). A held bit does not drop from a single missed reinforcement.

**Carry footprint = distinct addresses `c` visits while holding** = number of distinct patterns of its *non-self* observed bits over the segment.
- **Minimal carry** (observes only `prev_state[c]` + a stable trigger): every hold step → same address → whole chain is **one cell**, holds for any length, generalizes to held-out perfectly. **Ideal.**
- **Noisy carry** (observes varying sensor bits): hold path spreads across many cells; untouched ones = holes the bit falls through on held-out episodes. ⇒ **robust carry ⇒ minimal, segment-invariant context** (set/reset on sensor, hold on self). Same sparse-connectivity-generalizes thesis, applied to state persistence.

**Collision rules (these set the independence the consistency schedule rides):**
- **Different neurons** (`c1≠c2`): separate tables → **cannot collide** → safe to commit simultaneously (batch). (May still couple *dynamically* if c2 observes prev_state[c1] — that's information flow, not a write conflict.)
- **Same neuron, same address, opposite targets**: over-constraint — one cell can't be both → **allocate another neuron** (the substrate's "this state can't hold both meanings"). v1: report as **saturation pressure** to the GA (§8); do not grow neurons inside the trainer.
- **Same neuron, different addresses**: fine — one neuron carrying multiple latched bits across its address space (the Departure-1 economy).

## 6. The backward walk — finding `t*` and the distinction

### Naive walk fails
"First step back where input differs in any observed bit" → `t−1` almost always (thermometer low-order bits of noisy sensors differ every step) → reinvents the memoryless short-window controller. **The walk must test UTILITY, not difference.**

### Discriminative walk (utile criterion, backward)
Collect the conflict's instances, labeled by which PWM they needed (high/low). Walk back; at each lag, for each **observable** sensor bit, score how cleanly "bit set vs clear at this lag" partitions high from low (decision-stump / information-gain). `t*` = lag of the best separator. This refuses to fire on noise that doesn't align with the conflict.

### The Type-1 / Type-2 fork (reframes v1)
- **Type 1 — event distinction:** a clean `(bit, t*)` separator exists (e.g. roll-rate > threshold 3 steps ago). Discrete event → **latch at `t*`**. Shallow walk. This is the greedy-1-bit arm.
- **Type 2 — accumulative distinction:** *no* single `(bit, lag)` separates the groups, but the **signed sum** of an error feature over the window does. No single `t*`. → **install/increment the thermometer counter** (integral) on the dominant signed-error feature.

**The diagnosed deficit (steady-state offset) is Type 2.** An integral is *by definition* the case where no single past step explains the present need. So the discriminative walk will *correctly fail to find a `t*`* on exactly the conflicts we most need — and that failure is the **signal** "this is accumulative." ⇒ **v1 must be counter-first**, with event-latches as the secondary arm. Both arms fall out of one scan: strong best-stump → Type 1; weak stump but strong cumulative-correlation → Type 2.

### Enablers
- **Thermometer → threshold:** the 8 ordered bits/feature mean "which bit" = "which threshold" → monotone 1D search per feature, interpretable ("remember roll-rate exceeded threshold k, L steps ago").
- **K-fold-accumulate = built-in significance test:** controllers accumulate across 5 folds (CLAUDE.md). Require a separator to hold **across folds** before committing → a spurious (noise) separator won't; a real one will. Free cross-validation of every candidate distinction.
- **Rising weight `w(t)=t/T`:** weight conflict *instances* (and `τ`) so the chosen separator preferentially explains **settled** conflicts, not the startup transient — directly targets the steady-state deficit. (Distinct from `k(e)`, §7 — `w(t)` is per-timestep "which conflicts matter," `k(e)` is per-epoch "how many repairs per re-roll.")
- **Connectivity gating → GA wish-list:** candidates restricted to observed bits/lags. Best-but-unobservable separator → take best *observable* now + emit **connectivity pressure** ("route a neuron to feature f, threshold k, lag L") for the GA. The walk's richest output is a *wish-list of distinctions connectivity can't yet express* — far more informative than scalar fitness.

## 7. Consistency — the epoch schedule (greedy ↔ batch)

Every split ripples downstream (Departure 2), so state assignments must be kept consistent across the trajectory. Bootstrap from the memoryless baseline (pass 1: state≈constant → conflicts bucket by frame-projection alone); each round of splits creates real distinctions → re-roll → re-bucket by `(frame, state)` → residual/new conflicts. Converges when no bucket's weighted PWM spread exceeds `τ`, or connectivity/budget exhausts.

**One knob, scheduled:** `k(e)` = how many scanned conflicts to commit before re-rolling.
- `k=1` → strict greedy (one split/re-roll, always consistent, O(#splits) re-rolls).
- `k=∞` → batch (all splits/one re-roll, cheap, risks intra-round interference).
- `k(e)` grows with epoch → **anneal greedy → batch**.

**Direction (inverts the usual coarse-to-fine reflex):** what makes batching safe is *repair independence* = whether splits land on the same `(neuron, address)`. Early training builds the *first* accumulator → many conflicts are the same integral at different levels → pile on the same neuron → collide → **serialize (greedy)**. Late training adds sparse, unrelated latches → fresh neurons → separate tables → no collision → **batch**. Independence rises with epoch ⇒ `k(e)` rises with epoch.
- v1 form: `k(e) = ceil(K · (e/E)^γ)`, start greedy-heavy (γ ≥ 1).
- **Better (measure, don't guess):** drive `k(e)` directly from the *observed* fraction of a round's splits sharing a `(neuron, address)` — anneal the actual coupling.
- **Companion (same clock, conventional direction):** anneal `τ` coarse→fine (loose early = gross conflicts first; tight late = fine PWM). Reinforces `k(e)`.

## 8. GA handshake (the two-component seam)

The trainer (memory writes, fixed `sn`) and the GA (connectivity + structure) meet at two pressures the trainer *emits*, the GA *consumes*:
1. **Connectivity pressure:** "I wanted to split on feature f / threshold k / lag L but no neuron observes it." → GA mutates a neuron's connections toward that bit.
2. **Saturation pressure (neurogenesis):** "over-constraint with no free capacity here." → GA mutates `sn` up and allocates the new neuron's connectivity. **v1 keeps `sn` fixed**; neurogenesis is a GA-side mutation, NOT intra-training growth (keeps the trainer/GA seam clean, keeps genome bookkeeping sane). This is how minimal `sn` is *discovered* rather than imposed (fixing option A's sn≥9 over-constraint).

## 9. Decrement (anti-windup) — leaky vs. cascade

The integral is a thermometer counter. **Increment** is regular/easy ("if level k−1 on and error>0, set level k"). **Decrement** (unwind) has two designs:
- **Clean top-down cascade:** clear exactly the topmost lit bit, one per step. Exact integral, but each neuron must detect "I'm the top" (observe `prev_state[k+1]`) + delicate one-release-per-step. Fragile.
- **Leaky integrator (v1):** held bits slowly decay toward FALSE when not reinforced — `∫` with a forgetting term. Approximate (bounded residual offset) but a **standard, legitimate anti-windup technique**, and nearly free on QSR (the leak = "weak hold": don't strongly reinforce, let unused levels drift down the ladder). No top-detection, no extra connectivity.

**v1 = leaky.** Cascade is a v2 refinement if the residual offset proves to matter.

---

## 10. Implementation plan

All **controller-only** (ram.rs / ids_cache.rs untouched). Gated behind `WNN_STATE_SPLIT=1`, **default OFF** (like option A). Implement uncommitted → `maturin develop --release` → test → commit only if it works, else revert source + rebuild. Validation harness exists: `build_controller(controller_genome_from_arch(...))` + `run_episode`; held-out via `--report-seed`; multi-seed via the Step-0 harness.

Phased by **risk** (de-risk the most fundamental unknowns first):

### Phase 1 — Latch substrate validation (de-risk carry persistence)
The whole design dies if a latch can't hold. **Hand-install** a set+hold latch on one state neuron (minimal self-loop + trigger connectivity), run the forward roll, assert the bit **holds across N steps** and survives one contradictory nudge (QSR hysteresis). No walk, no scan — pure substrate check.
- Touch: a debug/test entry in controller.rs + a small Python test using `build_controller`.
- Pass criterion: `state_universe` for that neuron stays set across the segment; flips only on sustained reset.

### Phase 2 — Conflict scan + discriminative walk (Type-1 arm only)
New control flow: forward-roll recording `(sensor_window_t, frame_t, state_id_t, pwm_target_t, fold_id)`; **collect across folds**; scan buckets by output address; flag weighted-spread > `τ`; for each conflict run the discriminative `(threshold, lag)` search (connectivity-gated, fold-validated, `w(t)`-weighted); **Type-1 latch** at `t*` (set + carry, QSR nudges).
- Touch: new `state_split_train(...)` in controller.rs (or a `controller_split.rs` module); reuse `compute_address_sparse`, `nudge_toward_value`, `read_cell`/`write_cell`.
- Note: the **walk is search/analysis, not RAM eval** — implement in **Rust** (no Python reimplementation of accelerator logic per CLAUDE.md), but add a debug **trajectory export** so the discriminative logic can be inspected/validated offline.
- Pass criterion: on a task with a known discrete event, the walk finds the right `(threshold, lag)` and the latch resolves the conflict (re-roll shows the bucket's spread drop below `τ`).

### Phase 3 — Type-2 counter arm (the integral — the actual deficit)
Detect "no clean stump but strong signed-sum correlation" → install/increment the **thermometer counter** on the dominant error feature; **leaky** decrement. This is the arm that closes the steady-state gap.
- Touch: counter install/increment + leaky-decay nudge logic in the split module; the Type-1/Type-2 branch in the scan.
- Pass criterion: on the 5° spec, steady-state offset shrinks (soft-fails settle below ~5.6°); `state_universe ≫ 1`; 5° stability climbs from ~83% toward PID's 98%.

### Phase 4 — Consistency loop + `k(e)` schedule
Wrap Phases 2–3 in the round/re-roll loop with `k(e)` (start greedy-heavy `γ≥1`, or measured-coupling-driven) + companion `τ` anneal. Bootstrap from memoryless.
- Touch: the outer loop in the split module; expose `K, γ`, `τ0`, `w(t)` as params (env or genome).
- Pass criterion: converges (rounds terminate); no oscillation; metrics monotone-ish across rounds.

### Phase 5 — GA handshake (connectivity + saturation pressure)
Emit the two pressures from the trainer; wire `phased_ga` (Python) to consume them: connectivity-pressure biases connection mutation toward requested (feature, threshold, lag); saturation-pressure biases `sn` mutation up. v1 `sn` fixed within a genome; growth happens across GA generations.
- Touch: Rust → return pressure records from `state_split_train`; Python `phased_ga.py` / `reward_gated.py` mutation operators read them; full-stack param forwarding (Rule 6).
- Pass criterion: GA discovers a smaller/right `sn` than option A's imposed grid; connectivity converges toward the requested discriminators.

### Phase 6 — Full eval + multi-seed + baselines
Run on the 5° spec, multi-seed (`--report-seed` held-out + Step-0 harness), compare vs: memoryless baseline, option A (sn-sweet-spot), and PID (3.40°/98% reference). Report held-out, NOT during-search gen-line (`project_controller_eval_variance`).
- Pass criterion: closes a meaningful fraction of the 83%→98% stability gap *on held-out*, without option A's sn≥9 collapse.

### Sequencing notes
- Phases 1→2→3 are the critical path (substrate → walk → integral). 4–5 are the optimization wrapper. 6 is the verdict.
- Keep the current `bptt_train_window` + option A paths **intact** for A/B throughout.
- Rebuild: `cd src/wnn/ram/strategies/accelerator && unset CONDA_PREFIX && source wnn/bin/activate && maturin develop --release`. Controller-only; safe for IDS.
- Machine hygiene: RAYON=3 for controller keeps off the IDS worker.

## 11. Open questions / risks (honest)
- **✅ CONFLICT FORMATION (was the "#1 blocker" in 5b; RESOLVED 09/06 with a different root cause than first thought):** 5b found exact-full-frame bucketing yields 0 conflicts on real trajectories. First hypothesis = bucket by output-observed projection — but MEASURED the output-connection union covers 90-100% of out_input → projection ≈ exact, no help. Real root cause (5e probe): **conflict formation is gated by TRAJECTORY STRUCTURE, not bucket granularity.** A fresh/tumbling controller produces chaotic never-repeating trajectories → no recurring situations → ~0 conflicts even at coarse k. A bptt-SEEDED controller (even a poor one, 0% stable) produces structured partially-repetitive motion → situations recur → planted=4, saturation=7, wishes=15 even at EXACT bucketing. **The WARM START (seed-then-split) is the enabler.** Shipped anyway as secondary knobs (commit 0ecba1ff): adaptive coarse-signature bucketing (`scan_conflicts_coarse`, largest-k-that-hits-target) + lag≥1 state-walk (lag-0 separators are the output's job). Coarse bucketing is a density/specificity knob; the load-bearing decision is **Phase 6 = bptt-seed then split**, with a seed strong enough to reach hover (so the steady-state INTEGRAL conflicts specifically appear — the current weak seed surfaces mostly transient conflicts).
- **Type-2 detection threshold:** when is a best-stump "too weak" → declare accumulative? Needs empirical tuning (start: stump-gain below X *and* cumulative-correlation above Y).
- **Leaky-decay rate:** too fast = no integral; too slow = windup. Tune on the 5° task.
- **`k(e)` shape / γ:** measure coupling rather than guess (recommended).
- **Walk cost:** scan is O(conflicts × window × observable-bits × folds) — watch perf; the connectivity gating shrinks the bit set, the thermometer makes it a 1D threshold search.
- **Held-out generalization of carries:** the evaporation hazard (untouched hold-addresses). Mitigated by minimal-context carries; *must* be checked on held-out, not just training folds.
- **`t*` may not exist (Type-2):** by design — that's the integral signal, not a failure. Don't force a Type-1 latch when the scan says accumulative.

### 11b. Phase-6 tuning ledger (the ONLY genuinely deferred work — capability is built)
Every knob below already has a working default (the trainer runs end-to-end on synthetic tasks); Phase 6 *tunes* them against the real 5° task. This list IS the plan for the deferred tuning — none of it is missing capability, only calibration.

| Knob | Where | Current default | Tune against |
|---|---|---|---|
| `tau` (conflict threshold) | `split_train*` arg | 0.1 | PWM-noise floor of the real task; too low = chase noise, too high = miss real conflicts |
| `clean_gain` (Type-1 vs Type-2 split) | `split_train*` arg | 0.999 | when a stump is "clean enough" to latch vs fall through to integral; lower if real separators are imperfect |
| `accum_corr` (Type-2 + bidir detection) | `split_train*` arg | 0.9 | min net-count correlation to install a counter; balances false integrals vs missed ones |
| **bidir-vs-increment preference** | `split_resolve_conflict` / `split_train` | "try bidir first if `sbpn≥5`" | whether to prefer the unwinding integral; may want a corr-margin gate (only go bidir if it beats increment by δ) |
| `k(e)` schedule (greedy→batch) | `split_train_loop` | `k = k_start + round` (linear) | measure same-(neuron,addr) coupling per round and drive `k` from it, rather than linear |
| Type-2 detection margin | `detect_accumulator*` | corr-only | add "stump weak AND cumulative strong" two-sided test (§6) if single-feature false-positives appear |
| pure leaky-decay rate | (not built; bidir cascade used instead) | n/a | only if a *time-forgetting* integral (not just reversal-unwind) proves needed |
| latch nudge strength at plant | `split_plant_latch` | direct write TRUE/WEAK_FALSE | Phase 4 was meant to replace direct-write with incremental evidence-nudging; revisit if confidence dynamics matter |

## 12. Key code anchors
- `controller.rs:965-1193` `bptt_train_window` — current path; the split trainer is a sibling, gated by `WNN_STATE_SPLIT`.
  - forward roll + recording: 1017-1066 (extend recording here).
  - output-credit `d_out`: 1080-1104 (the degenerate signal being replaced).
  - state commit: 1139-1167; output commit: 1170-1189.
- `controller_training.rs:685` `nudge_toward_value` (one-step QSR nudge — the latch primitive); `:677` `nudge_toward_pub`.
- `controller.rs:38` `QSR_WEIGHTS = [0.0, 0.25, 0.75, 1.0]`.
- `compute_address_sparse`, `read_cell`/`write_cell`, `neuron_entries` — address/cell primitives.
- Python: `evaluator.py` (prefix_factor, spec), `phased_ga.py` (grid/stages/mutation), `reward_gated.py:311` (bptt call site), `dagger_train.rs:534` (Rust path).
</content>
</invoke>
