# Plan: Controller into dashboard+worker (budget scheduler, Option A)

Status: PROPOSED (06/06/2026). Decisions locked: Option A (wrap `run_phased_ga`),
budget-based scheduling (not lanes), type-balanced admission.

## Goal
Run the drone CONTROLLER (`tests/run_phased_ga.py` phased-GA) as dashboard flows
alongside IDS, sharing CPU via a budget, so we stop the ad-hoc detached-process
pattern (which caused tonight's incidents) and monitor the controller in the dashboard.

## Confirmed facts (grounding)
- The FLOW loop already carries the FULL population across experiments
  (`flow.py:1368` `current_population=result.final_population` → `:1403` threads it in).
  So IDS phases are NOT independent; the carry is correct in the flow path. Tonight's
  winner-only bug was ONLY in standalone `run_phased_ga` (now fixed).
- Controller→flow bridge mostly exists: `flow_adapter.py`, worker `architecture_type=="controller"`
  dispatch (`worker.py:553/570`), per-phase controller strategies, DB Iteration schema already
  has `mean_attitude_error_deg`.
- BLOCKER for concurrency: worker claims a flow only if NO flow is `running`
  (`worker.py:374`), and runs ONE flow IN-PROCESS. RAYON is process-global.
  → budget concurrency REQUIRES one SUBPROCESS per flow.

## Design

### Component 1 — Worker becomes a budget-aware, type-balanced SCHEDULER
- Config: `--cpu-budget` (default 13 = 16 − 3 reserved for macOS), override via env.
- Per-flow core need: `wnn_num_threads` param; default by architecture_type (ids=10,
  controller=3) when unset.
- Scheduler loop each poll:
  1. Reap finished flow subprocesses → reclaim their cores.
  2. `remaining = B − sum(running cores)`.
  3. ADMIT loop while something fits:
     - candidates = types with ≥1 queued flow whose cores ≤ remaining.
     - pick `t* = argmin running_count[type]` over candidates (UNDER-REPRESENTED type first
       → keeps types balanced ±1; auto-fills with the present type when the other is absent).
     - admit highest-id queued flow of `t*` (preserves today's within-type ordering).
     - spawn subprocess with `RAYON_NUM_THREADS = flow.cores`; update running set + remaining.
- One SUBPROCESS per flow (each owns its RAYON pool + its own Rust cancel flag →
  also fixes the process-global cancel footgun + gives crash isolation).

### Component 2 — Extract flow execution into a `flow_runner` entrypoint
- Move the worker's current in-process flow-run code into `python -m wnn.ram.experiments.flow_runner <flow_id>`.
- It loads the flow, runs it (IDS via existing `Flow.run`; controller via Component 3),
  reports to the DB exactly as today (tracker/API). Worker no longer runs flows in-process —
  it only schedules + spawns flow_runners. IDS path unchanged except it's now a subprocess.

### Component 3 — Controller flow = wrap `run_phased_ga` (Option A)
- flow_runner, on `architecture_type=="controller"`, calls `run_phased_ga`'s `_run_one`
  IN-PROCESS (the flow_runner IS the RAYON=3 subprocess) with params mapped from flow config:
  tilt, body_rate, yaw_rate, steps, pop, elitism, check_interval, {neurons,bits,conns,memory}_{gens,patience},
  eval_episodes, universe_episodes, num_eval_folds, fit_weights, base_seed, report_seed,
  lamarckian, save_stage_checkpoints. → REUSES the fixed orchestrator (population carry +
  held-out) entirely; zero re-implementation.
- DB reporting: the controller flow defines 5 experiment records (grid, neurons, bits,
  connections, memory) for dashboard visibility. Thread the tracker + per-stage experiment_id
  into run_phased_ga's strategy construction so each stage reports per-gen
  `mean_attitude_error_deg` (Iteration schema ready) + writes its held-out to extra_metrics.
  Controller GA strategies already have `set_tracker` hooks — wire them.

### Component 4 — Queue tooling
- `scripts/queue_controller_flow.py` (mirror `queue_cross_dataset.py`): POST a controller flow
  via the API with the 5 experiments + `wnn_num_threads=3` + the recipe params.

### Component 5 — Migration
- Let the current detached lamarckian (PID 4801) finish OR re-queue it as a flow.
- Retire the ad-hoc detach once Components 1–4 land + verified.

## Risks / sequencing
- Biggest risk: the worker→subprocess change touches ALL flows incl. the paper-critical IDS
  cohort. Mitigation: build + test the scheduler + flow_runner with IDS-as-subprocess FIRST
  (parity vs in-process), deploy ONLY when IDS queue is idle ([[feedback_restart_cancels_running_flow]]).
- Per-stage tracker wiring in run_phased_ga is the only genuinely new controller surface.

## BUILD PROGRESS (06/06)
Architect blueprint produced (code-grounded, file:line anchors). Key adopted recommendation:
**move run_phased_ga's callables → `src/wnn/control/phased_ga.py`** (kills the tests/ import
problem); keep `tests/run_phased_ga.py` as a thin CLI wrapper. Build order (each testable):
1. ✅ DONE — admission policy `src/wnn/ram/experiments/scheduler.py` (pure: budget+type-balance+
   oldest-FIFO+dynamic budget) + `tests/test_scheduler_admission.py` (8/8 pass). Zero infra risk.
2. ✅ DONE — `src/wnn/ram/experiments/flow_runner.py` entrypoint
   (`python -m wnn.ram.experiments.flow_runner <id>`): reuses `FlowWorker._execute_flow` verbatim
   (parity by construction), guards status=='queued' only, exit 0/1/2. heartbeat+should_stop ride
   along inside _execute_flow. Tested: compile+import; running flow→refuse exit2; bogus→exit2.
   LIVE IDS in-proc-vs-subprocess CE/F1 parity test DEFERRED to deploy (step 8, IDS idle).
3. ✅ DONE — worker.py scheduler loop: run() = _reap_finished → _recover_stale_flows → scheduler.admit
   → _spawn_flow(Popen `python -m flow_runner <id>`, RAYON env, start_new_session). budget=detect_budget()
   (13). _handle_signal forwards SIGTERM to children. Tested 8/8 + live spawn+reap smoke. Deploy=restart
   when IDS idle. LIVE admit→spawn of a real queued flow still to verify at deploy.
4. ✅ DONE — (4a) `git mv tests/run_phased_ga.py → src/wnn/control/phased_ga.py` + thin CLI wrapper +
   run_memory_refinement import fix. (4b) `tracker`+`stage_experiment_ids`(indexed [grid,N,B,C,M])+
   `stage_holdouts` out-dict on `_run_one`; `tracker`+`experiment_id`→`set_tracker` on both phase
   runners; `_maybe_holdout` RETURNS the metric. All backward-compat (None defaults; 4-tuple return
   unchanged). Live per-gen DB recording verified at deploy.
5. ✅ DONE — flow_runner: run_one_flow branches controller→`_run_controller_flow`; `_build_phased_ga_args`
   (CLI defaults + param overrides, runs=1); `_controller_stage_experiment_ids` (by-sequence + fallback).
   _run_controller_flow mirrors _execute_flow scaffolding, wraps phased_ga._run_one with tracker+
   stage_experiment_ids+stage_holdouts, marks 5 experiments running→completed + writes during-search +
   held-out extra_metrics. Tested compile/import/arg-mapping. Live E2E at deploy.
6. ✅ DONE + SCOPE — flows/new: added `wnn_num_threads` (CPU Threads) + `check_interval` (Patience Check
   Interval) inputs beside the existing Patience NUMBER field → flat params. Backend: worker reads
   check_interval→ExperimentConfig (IDS early-stopper already honored it, just unplumbed); wnn_num_threads
   already read by worker+scheduler; controller consumes check_interval via _build_phased_ga_args arg-match.
   svelte-check 0/0. Frontend hot-reloads; worker change applies on restart.
7. TODO — `scripts/queue_controller_flow.py`.
8. DEPLOY only when IDS idle.

## SCOPE ADD (06/06) — patience config in the UI
- Patience-CHECK interval (how often the early-stop check runs): default IDS=every 10 gens,
  controller=every 5 gens. Controller already has `--check-interval` (=5 in our runs); IDS needs
  the equivalent param surfaced (find the IDS patience-check cadence; thread a param).
- Dashboard UI: add fields to configure BOTH (a) the patience-check interval and (b) the patience
  NUMBER (patience count) per experiment, if not already present. Defaults as above.
- Wire these flow params → worker → run (controller: --check-interval + --*-patience already exist;
  IDS: thread the new check-interval param + the existing patience).
LANDMINES (from blueprint): verify SQLite WAL in data_layer (concurrent subprocess writes);
heartbeat-on-reap close-out; record_seed_set CWD; emergency_state module-global OK (1 flow/subprocess).

## Test plan
- Unit: scheduler admission — budget bound + type-balance (mock queued flows of both types,
  assert 1 IDS+1 ctrl when both queued; 4 ctrl when only ctrl; IDS-first when under-represented).
- Integration: IDS flow as subprocess (parity with current in-process run).
- E2E: tiny controller flow (pop6, 1 gen/stage) runs via worker → 5 experiments update in DB,
  mean_attitude_error_deg + held-out visible in dashboard.

## DECISIONS (locked 06/06)
- Budget = DYNAMIC from the CPU: `budget = detected_cpu_cores − reserve` (reserve≈3 for macOS),
  read at runtime (not hardcoded 13). 16-core → 13.
- Per-flow RAYON is a FLOW-UI parameter (`wnn_num_threads`), editable in the dashboard, default
  ids=10 / controller=3. So it's changeable per flow anytime.
- Within-type ordering: switch to OLDEST-id-first (FIFO) — drop the legacy highest-id-wins.
- BUILD NOW; DEPLOY (restart worker) ONLY when the IDS queue is idle.
