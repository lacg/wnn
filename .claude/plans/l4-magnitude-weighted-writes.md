# L4 — magnitude-priority output writes (CORRECTED design)

**Status:** DESIGN COMPLETE, NOT IMPLEMENTED. Build + arm when L3 drains (4/4 markers).
**Author:** Andrew Martin, 07/08/2026.

## The spec was aimed at dead code — read this first

`docs/hold_floor_levers_spec.md` §L4 says "magnitude-weighted DAgger **conflict** writes",
pointing at the conflict machinery (`controller_split.rs:scan_conflicts`, `Conflict.spread`,
`split_train_loop`). **That code never executes in this programme.** `dagger_train.rs:1266`:

```rust
let use_split = std::env::var("WNN_STATE_SPLIT").map(|s| s == "1").unwrap_or(false)
	&& controller.gpu_dims().2 > 0;          // .2 == state_neurons (controller.rs:991-995)
```

Both conditions required. L1/L1b/L2/L3 all pass `--grid-state-neurons 0 --max-state-neurons 0`
⇒ `state_neurons == 0` ⇒ `use_split == false` regardless of the env var, which no L3 script
sets anyway. Implementing the spec as written would have produced a wheel with **zero
behavioural change**, 4 runs at ~2.5h each identical to control, and a false "L4 refuted" on
the programme's last candidate lever. Same family as the L2 GPU plant omission.

## Where training actually happens

`train_on_trajectory_rs` (`dagger_train.rs:952`) → `bptt_train_window` (`controller.rs:2257`)
→ **section (d)**, `controller.rs:2621-2641`:

```rust
// (d) Commit OUTPUT layer toward PID's PWM at the recorded out address.
for n in 0..num_out {
	...
	let addr = compute_address_sparse(&rec_out_input[d], &self.output_connections[cs..ce], obpn);
	let cur = self.output_memory.read_cell(n, addr);
	if protect_learned && cur != EMPTY_U8 && cell_fire_bit(cur, self.memory_mode) != target_true {
		continue;
	}
	let nv = nudge_cell(cur, target_true, self.memory_mode);
	if nv != cur { self.output_memory.write_cell(n, addr, nv, true); o_writes += 1; }
}
```

With `sn=0` the comment at `:2396` confirms the walk "degenerat[es] to the classic supervised
RAMLayer direct write (visited address → teacher answer)."

## Why "weight the write" cannot mean what the spec meant

`--memory-mode BINARY`: two cell states, so `nudge_cell` goes straight to the target.
**Last writer wins; there is no vote tally.** A weight has nothing to accumulate into.
(QUAD would give a real tally — FALSE→WEAK_FALSE→WEAK_TRUE→TRUE — but switching substrate
breaks comparability with L1-L3 AND carries the dfa1l QUAD cost, ~106h/run. Not this lever.)

The backward walk is `for d in (0..n_rec).rev()` (`:2391`), so the **earliest record in the
window writes last** and owns every contested cell. That ordering is arbitrary w.r.t. error
magnitude — and it IS the hypothesised failure: near-hover samples overwrite what the rare
large corrections taught.

## The lever: two variants, both default-off

**A — `--write-priority-err` (ORDER).** Within a window, order the section-(d) commits by
ascending |attitude error| so the HIGHEST-error record writes last and owns contested cells.
Pure reordering: no new memory, no cell-format change, no ABI break. Exactly inverts the
current arbitrary bias.

**B — `--write-err-floor <deg>` (SKIP).** Skip the commit when the record's |err| is below a
floor, so near-hover samples cannot overwrite corrections at all. Coarser, but it directly
tests "the mass of near-hover samples is the problem" rather than "the ordering is."

A and B are a proper double-dissociation, the same shape L3 used: A says ordering is the
mechanism, B says sample mass is, both say magnitude is, neither rules the family out.

## Parity gate (mandatory before any run)

Both flags default OFF; with them off the write order and set must be **bit-identical** to
today. Test: `cargo test -p ram_controller --lib --no-default-features` (94 tests incl. the
14 CPU/GPU parity sweeps) plus a fresh control run whose NEURONS/MEMORY multi-seed blocks
match a banked L3 marker. A wheel that changes existing recipes is a bug, not a lever.

## Implementation checklist

1. `RewardGatedConfigPacked` — add `write_priority_err: bool`, `write_err_floor: f32`.
2. Thread cfg → `train_on_trajectory_rs` → `bptt_train_window` (new params; existing callers
   pass the off values).
3. Per-record |err|: derive from `targets[t]` vs the recorded attitude in the forward roll —
   already available where `rec_out_input` is built. **Confirm the frame** before using it
   (the frame-misalignment bug precedent).
4. Section (d): apply order (A) / floor (B).
5. PyO3 surface in `controller/lib.rs`; Python flags in `phased_ga.py`; recipe flags in the
   L4 chain script.
6. `maturin develop --release -m controller/Cargo.toml` (swap-free — worker never imports
   ram_controller). Build only when the box is idle.
7. Parity gate above, THEN arm the chain with the `L4_WAIT_PID` gate pattern L3 used.

## Run matrix

2 arms × 2 seeds (31337002, 31337003), same recipe/budget as L3, ~2.5h each ≈ 10h.
Control = the L3 control: `s31337002 1.21/100.0/0.64` · `s31337003 1.58/100.0/0.95`.
SUCCESS (pre-registered): steady < ~0.35° on BOTH seeds for at least one arm.
REFUTATION: steady stays in the 0.57-0.87 band on both seeds for both arms.
n=1 seed ranks nothing.
