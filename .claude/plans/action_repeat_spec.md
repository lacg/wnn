# Action-repeat (arm R) — full implementation spec

**Date:** 02/07/2026. Status: SPEC ONLY — deliberately not implemented at session tail.
Sajus frame-skip (=5) adapted to the WNN controller: decide every Nth physical step,
HOLD the PWM in between. Temporal abstraction for a memoryless controller: the 4-frame
window then spans 4N steps of history; jerk drops; each decision's consequence is larger.

## Semantics (locked)
- `action_repeat: int = 1` (spec-level; 1 = today's behavior, bit-identical).
- Decision steps: t % N == 0 (counter zeroed at reset(); episodes align at t=0).
- HOLD steps: feature ACCUMULATORS still tick (integ[], yaw_heading — physical-time
  quantities; `compute_features` stays "called exactly once per timestep"), but NO
  frame-encode, NO ring push, NO state/output forward, NO decode; return held last_pwm;
  prev_state unchanged; last_output_cells / last_*_layer_input caches unchanged (stale
  = the decision step's — intentional).
- Window semantics: ring receives frames ONLY at decision steps → window = last 4
  DECISION frames (spans 4N physical steps). THIS IS THE LOAD-BEARING DEFINITION.
- Delta mode: leak applies per DECODE (i.e. every N steps) — inherent semantics change,
  fine (current recipe is --no-delta-control anyway).
- Jerk metric: held steps contribute 0 deltas (real smoothing, keep).

## ⚠️ THE TRAP FOUND 02/07 — trainer re-forward ring divergence
Trainers RE-FORWARD recorded sensor streams: CPU `split_record` (controller.rs~1489),
`split_retrain_output` (~2457), `bptt_train_window` (~965/1242); GPU `controller_record`
+ `controller_train` kernels (TrainBatch carries recorded gyros/accels/targets/pid_pwms).
If the re-forward pushes the ring EVERY recorded step while deploy pushes only at
decision steps, the re-forwarded addresses differ from deploy's EVEN AT DECISION STEPS
→ trainer writes cells deploy never reads (the +55pp mode-mismatch bug class).
⇒ a `decision-step mask` (or action_repeat param + t%N logic) must thread through EVERY
re-forward path, CPU and GPU. ALSO VERIFY FIRST: do recorded trajectories replay step()'s
caches (then hold steps record STALE decision inputs — coherent, different fix: skip or
dedupe hold-step records) or re-forward raw streams (then the mask is mandatory)?
Check dagger_train.rs rollout_and_label_rs (rolls via controller.step:378 — hold-aware
for free once step() is) vs what it RECORDS per step.

## Surfaces (the full parity set)
1. controller.rs `WnnController`: fields `action_repeat: usize`, `step_counter: usize`,
   `last_pwm: Vec<f32>` (init = hover, same expr as pwm/pwm_prev incl. decouple).
   pyo3 ctor signature: append `action_repeat = 1` (defaults keep all 8 Python sites
   compiling). reset(): counter=0, last_pwm=hover. step(): compute_features FIRST
   (every step), then if hold → return last_pwm.clone(); decision path sets last_pwm
   (guard the clone behind action_repeat>1 to keep N=1 hot path unchanged).
2. Trainer re-forwards (CPU): split_record / split_retrain_output / bptt_train_window
   — decision-mask per the trap above (after the replay-vs-reforward verification).
3. controller_rollout.metal: extract section (1) of `forward_state` (H2 feature derive,
   lines ~173-216) into `derive_features(sensors, F, integ, yaw_heading, pwm_acc)`;
   forward_state calls it (single source). `struct Params`: APPEND `uint action_repeat;`
   (end — layout must match Rust RolloutParams exactly, same for TrainParams if train
   kernels gain it). Rollout kernel: `float last_pwm[4]` (hover init like pwm_acc);
   hold steps call derive_features only + pwm=last_pwm; decision branch updates
   prev_state + last_pwm; jerk/mono/steady/sim-step tail is COMMON (mono_last keeps
   the last DECISION step's count = CPU semantics).
4. controller_record + controller_train kernels: same decision-mask as (2).
5. metal_controller.rs: RolloutParams + TrainParams structs + every unpack/init site.
6. evaluator.py: `ControllerSpec.action_repeat: int = 1`; spec_from_arch propagates
   (like threshold_gamma — DON'T let it silently revert after grid); ctor pass-through
   at the 7 spec-driven WnnController sites (arch_adaptation:61, dagger:157,
   ga_memory:70+164, evaluator:360+915, reward_gated:369). NOT evaluator:214 (feat_ctl
   threshold calibration — must sample features every physical step; leave default 1
   + comment).
7. phased_ga: `--action-repeat` int default 1 → _make_spec param → both call sites
   (probe + grid) → ControllerSpec.

## Verification (the discipline)
- cargo check --workspace (0 warnings) → maturin build (NOT develop — low-edge cells
  must keep the current wheel; heterogeneous mid-sweep wheels forbidden).
- Isolated venv (pattern: /tmp/h2venv) with the BUILT wheel:
  a. N=1 parity: existing tests/test_controller_gpu_parity.py + test_controller_h2_smoke
     must pass BIT-IDENTICAL (action_repeat defaulted everywhere).
  b. N=5 cpu≡gpu: score the same trained controller CPU vs GPU at action_repeat=5 —
     stable/err must match to float tolerance; add as a new targeted test.
  c. Trainer-consistency probe: train a small genome at N=5, verify the deploy-read
     addresses appear in the trained cell set (the trap's direct check).
- Install into wnn-venv ONLY when no controller run is live (E2 driver wakes → it can
  install; or between E2 cells). Then append arm R cells to the E2 driver:
  `"REP|--immigrants 0.15 --action-repeat 5"` (resume-skip keeps done cells).

## Effort
~half a focused day (0.5-1h verify replay-vs-reforward; 1-2h CPU+Python; 1-2h Metal;
1h parity runs). Do it FRESH — this is the +55pp-bug class; no tail-of-session work.

## VERIFIED 02/07 — trainers RE-FORWARD raw streams; the mask is MANDATORY
Read of dagger_train.rs + controller.rs settles the trap question:
- `rollout_and_label_rs` (dagger_train.rs:483) records the RAW per-physical-step
  sensor stream: `traj.gyros/accels/targets/pid_pwms` are pushed EVERY sim step
  (lines 557-570), regardless of what step() cached. It does NOT record step()'s
  `last_*_layer_input` caches. So recorded trajectories are raw sensor streams.
- ALL CPU trainers then RE-FORWARD those raw streams from scratch:
  * `bptt_train_window` (controller.rs:1242) — forward roll calls
    `compute_features` + thresholds + `input_history.push_back` for EVERY t in
    the window (lines 1300-1345), then reads state/output addresses per step.
  * `split_record` (controller.rs:1502) — same per-step re-forward (1528-1572).
  * `split_retrain_output` (controller.rs:2475) — same (2503-2567).
  * GPU `controller_train` / `controller_record` kernels — same re-forward from
    the TrainBatch raw arrays (controller_rollout.metal 624-656 / 736-779).
- Therefore: if step() pushes the ring only at decision steps but the trainers
  keep pushing EVERY recorded step, trainer window contents (hence addresses)
  diverge from deploy EVEN AT DECISION STEPS → the +55pp mode-mismatch class.
  ⇒ implemented the decision-step mask in every re-forward path (branch 2 of
  the trap): hold steps tick `compute_features` (accumulators are physical-time)
  but do NOT frame-encode/ring-push/forward/record/commit.
- Alignment detail: bptt is called in W-sized chunks (train_on_trajectory_rs);
  chunk-local t is NOT episode-aligned when W % N != 0, so the mask uses the
  controller's persistent `step_counter` (reset_state=true → reset() zeroes it;
  reset_state=false carries it across chunks) — identical to deploy alignment.
- Record-space detail: split_record's `step_of`/`ep_lengths` (and the GPU record
  layout + split_train_loop_gpu's ep_start/step_of) move to DECISION-index space
  (records only exist at decision steps); at N=1 this is bit-identical to the
  old physical indexing. Separator lags are therefore in decision steps —
  consistent with the "window = last 4 decision frames" load-bearing definition.
