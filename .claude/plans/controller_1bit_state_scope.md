# Scope: recurrent STATE = 1 bit/neuron (MSB), not 2-bit QSR (08/06/2026)

## Why
Each state neuron currently feeds back its **2-bit QSR value** (MSB=side/output, LSB=training-confidence). Feeding the LSB (confidence) into the recurrent address is semantically wrong — it makes 1 neuron occupy 2 state dimensions and injects training-noise into the state. Correct: state output = **MSB only** (fired/not), `state_bits_in = sn` not `2·sn`. Wins: semantically right + **halves the forced prefix** (smaller address → fewer cells → faster) + compounds the small-sn plan.

## Output neurons: NO CHANGE (verified)
The output is the final actuator command, decoded **analog** via `strategy_5_qsr_weighted` (4 QSR levels → weights [0,0.25,0.75,1.0] → smooth PWM). It's never fed back as address bits → no confidence-leak bug. The 4-level decode is correct for a continuous output. Leave it.

## The cell stays QSR; only the OUTPUT (feedback) is 1-bit
The state memory cell remains a 2-bit QSR value (graduated nudging keeps training stable). We change only what the neuron EMITS into the recurrence/address: the MSB (`(v>>1)&1`). Training targets the cell's side (TRUE=3 / FALSE=0).

## Touch points

### Rust — controller.rs
- `state_bits_in = 2 * self.state_neurons` → `= self.state_neurons`  (lines ~863, ~994, ~1243).
- Forward/roll state ENCODING (2-bit `[base+2n]=MSB,[base+2n+1]=LSB` → 1-bit `[base+n]=MSB`):
  - step(): 1259-1262 (state-layer input), 1293-1295 (output-layer input).
  - bptt forward roll: 1044-1046 (state input), 1059-1061 (output input).
- Desired-state COMMIT target_val (reconstruct from 1 bit, not 2):
  - edra_train_step: 926-927  `target_val = desired_state_bits[n] ? 3 : 0`.
  - bptt backward: 1147-1148 `target_val = d_s[n] ? 3 : 0`.
  - bptt transition target_sides 1112: `dn[2*n]` → `dn[n]`; d_s slice 1120/1125 over `state_bits_in`(=sn).
  - (option-A integral commit already targets 3/0 per neuron — unchanged, just indexes n.)
- The solve vote loops (`for i in 0..state_bits_in`) + `sol[frame_bits+i]` adapt automatically (state_bits_in now sn).

### Python (already parameterized by prefix_factor!)
- evaluator.py:304 `prefix_factor=2` → `1`.   (forced_prefix = prefix_factor*sn = sn)
- evaluator.py:242 assert `>= 2*state_neurons` → `>= state_neurons` (or prefix_factor*sn).
- evaluator.py:532 length calc `+ 2 * self.spec.state_neurons` → `+ self.spec.state_neurons`.
- phased_ga.py grid `suffix = b - 2*sn` → `b - sn` (both _make_spec callers / valid-pair filter).

## Risk / rollout
- **Controller-only** — IDS uses ram.rs/ids_cache.rs, untouched. Rebuild (maturin) is safe for IDS.
- **BREAKS existing controller genomes/checkpoints** (they assume 2·sn). Acceptable — they're for the old (semantically-wrong) arch; new runs start fresh.
- Implement UNCOMMITTED → maturin rebuild → test (controller trains + runs + state_bits halves + stabilizes ≳ memoryless baseline) → commit only if it works; else revert source + rebuild.
- Gate: none needed (this IS the corrected default); old behaviour recoverable via git.

## Validation
fresh small-sn genome: forced prefix = sn (was 2·sn); state-layer input shrinks; trains without panic; closed-loop stable-rate in the ballpark of the 2-bit baseline (≥ ~70% at sn≈6) → then the small-sn A experiment runs on the corrected arch.
