"""
Flow Runner - Executes a SINGLE flow by id in this process, then exits.

Usage:
    python -m wnn.ram.experiments.flow_runner <flow_id> [--url https://localhost:3000]
                                              [--no-ssl-verify] [--context N]

This is the subprocess entrypoint for the budget-aware scheduler (Step 5 of the
controller->dashboard integration plan). It REUSES FlowWorker's execution
machinery (`_execute_flow` + every `_create_*_evaluator` helper) verbatim, so a
flow run via this entrypoint is the same code path as the legacy in-process
worker run -- parity is guaranteed by construction, not by re-implementation.

The only behavioural difference from the worker is scope: this runs ONE flow and
exits. The scheduler (worker.py, Step 3) owns the poll loop, the CPU budget, and
concurrency -- it admits a queued flow then spawns one of these per flow with
`RAYON_NUM_THREADS` set to that flow's core allotment.

Signals: FlowWorker installs SIGTERM/SIGINT handlers in __init__ that set
`_stop_current_flow` -> `should_stop()` -> graceful stop, AND propagate to the
Rust accelerator's cancel flag. In subprocess mode that's exactly what we want:
the scheduler's `kill <pid>` (reap / user-cancel) becomes a graceful end-of-
generation stop, and each subprocess owns its OWN Rust cancel flag -- which kills
the process-global cancel footgun the integration plan flagged.

Exit codes (for the scheduler's reap to log; the authoritative outcome is always
the flow's DB status, which the scheduler re-reads):
    0  flow reached a terminal-but-OK state (completed / paused / re-queued /
       cancelled-by-user / deleted)
    1  flow failed with an actual error
    2  usage / fetch error (flow id not found, not runnable, network)
"""

import argparse
import sys
from pathlib import Path

from wnn.ram.experiments.worker import FlowWorker


# Flow statuses that are NOT an error outcome for the runner process. A flow that
# was paused, re-queued for resume, cancelled by the user, or deleted is a clean
# exit -- the scheduler just reclaims the cores. Only an actual error is exit 1.
_OK_TERMINAL_STATUSES = {"completed", "paused", "queued", "cancelled"}


def run_one_flow(
    flow_id: int,
    dashboard_url: str = "https://localhost:3000",
    context_size: int = 4,
    verify_ssl: bool | str = False,
) -> int:
    """Fetch a single flow by id and execute it via FlowWorker._execute_flow.

    Returns a process exit code (see module docstring). Kept as a function (not
    inlined into main) so the scheduler's parity test can drive it directly.
    """
    # Build a FlowWorker purely as the execution host: we reuse its client,
    # tracker, heartbeat, signal handlers and evaluator factories. We never call
    # worker.run() -- the scheduler owns the loop. poll_interval is irrelevant
    # here (no loop), so the constructor default is fine.
    worker = FlowWorker(
        dashboard_url=dashboard_url,
        context_size=context_size,
        verify_ssl=verify_ssl,
    )

    # Fetch the one flow we were told to run. get_flow returns the full record
    # with an already-parsed `config` dict (verified shape: id / name / config /
    # config.params) -- the exact shape _execute_flow consumes.
    try:
        flow_data = worker.client.get_flow(flow_id)
    except Exception as e:
        worker._log(f"flow_runner: could not fetch flow {flow_id}: {e}")
        return 2

    if not flow_data:
        worker._log(f"flow_runner: flow {flow_id} not found")
        return 2

    status = flow_data.get("status", "")
    # The scheduler only ever admits QUEUED flows, then spawns us; we flip the
    # status to running via flow_started inside _execute_flow. So 'queued' is the
    # only valid entry state. Refuse anything else -- in particular 'running'
    # (another process owns it -> never double-run) and any terminal state
    # (completed/failed/cancelled/paused -> work is done).
    if status != "queued":
        worker._log(
            f"flow_runner: flow {flow_id} is '{status}', not 'queued' -- refusing to run"
        )
        return 2

    # Controller flows take the Option-A path (wrap phased_ga._run_one) instead
    # of the generic _execute_flow/Flow.run machinery, so they reuse the fixed
    # phased-GA orchestrator (full-population cross-stage carry + per-stage
    # held-out + lamarckian + emergency-dump resume).
    params = (flow_data.get("config") or {}).get("params") or {}
    if params.get("architecture_type") == "controller":
        _run_controller_flow(worker, flow_data)
    else:
        # IDS / LM / tiered: _execute_flow owns the full lifecycle (flow_started,
        # PID registration, heartbeat start/stop, evaluator construction,
        # flow.run, and the terminal status write) in its own try/except/finally.
        # It does NOT re-raise, so we read the outcome back from the DB below.
        worker._execute_flow(flow_data)

    final = worker.client.get_flow(flow_id)
    final_status = final.get("status", "") if final else "deleted"

    if final_status == "failed":
        worker._log(f"flow_runner: flow {flow_id} finished with status 'failed'")
        return 1
    if final_status == "deleted" or final_status in _OK_TERMINAL_STATUSES:
        worker._log(f"flow_runner: flow {flow_id} finished with status '{final_status}'")
        return 0
    # Unexpected lingering status (e.g. still 'running' after _execute_flow
    # returned) -- surface it but don't claim failure.
    worker._log(
        f"flow_runner: flow {flow_id} ended in unexpected status '{final_status}'"
    )
    return 0


# ---------------------------------------------------------------------------
# Controller flow path (Option A: wrap phased_ga._run_one)
# ---------------------------------------------------------------------------

# Stage index -> name, for the 5 controller experiment records / held-out keys.
# Indexed [0]=grid, [1]=neurons, [2]=bits, [3]=connections, [4]=memory (matches
# phased_ga._run_one's stage_experiment_ids contract + stage_results ordering).
_CONTROLLER_STAGES = ["Grid", "Neurons", "Bits", "Connections", "Memory"]


def _build_phased_ga_args(params: dict):
    """Map a controller flow's params onto a phased_ga args namespace.

    Starts from the CLI parser's DEFAULTS (the single source of truth) and
    overrides any arg whose name matches a flow-param key (use the CLI flag name
    with dashes->underscores, e.g. neurons_gens, bits_patience, tilt, base_seed,
    report_seed, fit_weight_err_sq, lamarckian, grid_state_neurons). Unknown
    params (architecture_type, wnn_num_threads, ...) are ignored. runs is pinned
    to 1 (a flow is one phased run).
    """
    from wnn.control import phased_ga

    args = phased_ga.build_arg_parser().parse_args([])
    for key, val in params.items():
        if val is not None and hasattr(args, key):
            setattr(args, key, val)
    args.runs = 1
    return args


def _controller_stage_experiment_ids(worker, flow_id: int) -> list:
    """Resolve the 5 stage experiment ids (indexed grid..memory) for a flow.

    Prefers the tracker's by-sequence lookup (matches how Flow.run maps stages);
    falls back to the flow's experiments listed by the API in id order. Missing
    slots are None (the tracker calls then no-op for that stage)."""
    ids = [None] * 5
    for idx in range(5):
        try:
            exp = worker.tracker.get_experiment_by_flow_sequence(flow_id, idx)
            if exp:
                ids[idx] = exp["id"]
        except Exception:
            pass
    if all(e is None for e in ids):
        # Fallback: take the flow's experiments in API order.
        try:
            exps = worker.client.list_flow_experiments(flow_id)
            for idx, exp in enumerate(exps[:5]):
                ids[idx] = exp.get("id")
        except Exception as e:
            worker._log(f"controller flow {flow_id}: could not resolve experiment ids: {e}")
    return ids


def _run_controller_flow(worker, flow_data: dict):
    """Execute a controller flow by wrapping phased_ga._run_one (Option A).

    Mirrors _execute_flow's scaffolding (flow_started, PID, heartbeat, log) but
    runs the 5-stage phased-GA orchestrator instead of Flow.run, threading the
    tracker + per-stage experiment ids so the dashboard sees per-generation
    mean_attitude_error_deg and per-stage held-out. Lazy controller imports keep
    the drone-sim deps out of IDS flow_runner subprocesses.
    """
    import math

    flow_id = flow_data["id"]
    flow_name = flow_data.get("name", f"Flow {flow_id}")
    params = (flow_data.get("config") or {}).get("params") or {}
    worker.current_flow_id = flow_id

    # Reset the Rust cancel flag (a stale flag would short-circuit the GA).
    try:
        import ram_accelerator
        ram_accelerator.reset_cancel_flag()
    except Exception as e:
        worker._log(f"WARNING: could not reset Rust cancel flag at controller flow start: {e}")

    log_file = worker._open_log_file(flow_name)
    worker._log(f"Logging to: {log_file}")
    worker.client.watch_log(str(log_file.absolute()))
    worker._log("=" * 60)
    worker._log(f"Starting CONTROLLER flow: {flow_name} (ID: {flow_id})")
    worker._log("=" * 60)

    try:
        worker.client.flow_started(flow_id)
        worker.client.register_flow_pid(flow_id, os.getpid())
        worker._start_heartbeat(flow_id)

        # Lazy controller imports (drone-sim deps).
        from wnn.control import phased_ga
        from wnn.control.training import EpisodeConfig
        from wnn.seeds import resolve_seed_set, log_seed_set, record_seed_set

        # Build args from flow params (CLI defaults + overrides).
        args = _build_phased_ga_args(params)
        # Install the phased-GA signal handlers so a scheduler SIGTERM produces an
        # emergency dump + graceful abort identical to the standalone CLI (this
        # subprocess runs only this one flow, so taking over signals is correct).
        phased_ga._install_signal_handlers()

        ec = EpisodeConfig(
            dt=0.001, steps_per_episode=args.steps,
            max_initial_tilt_rad=math.radians(args.tilt),
            max_initial_yaw_rad=math.radians(args.tilt),
            max_initial_body_rate=args.body_rate, max_initial_yaw_rate=args.yaw_rate,
        )

        base = args.base_seed if args.base_seed is not None else args.seed
        seeds = resolve_seed_set(base=base, run_index=0,
                                 train=args.train_seed, test=args.test_seed, val=args.val_seed)
        log_seed_set(seeds)
        try:
            record_seed_set(seeds, script="run_phased_ga_flow", extra={"flow_id": flow_id})
        except Exception as e:
            worker._log(f"controller flow {flow_id}: record_seed_set failed (non-fatal): {e}")

        stage_experiment_ids = _controller_stage_experiment_ids(worker, flow_id)
        worker._log(f"Controller stage experiment ids: {stage_experiment_ids}")
        # Mark all resolved experiments running.
        for eid in stage_experiment_ids:
            if eid is not None:
                try:
                    worker.tracker.update_experiment_status(eid, "running")
                except Exception:
                    pass

        # Run the 5-stage phased GA. tracker drives per-gen iteration recording;
        # stage_holdouts collects each stage's held-out winner metric.
        stage_holdouts: dict = {}
        stage_results, best_final, final_population, pid_m = phased_ga._run_one(
            args, ec, seeds,
            tracker=worker.tracker,
            stage_experiment_ids=stage_experiment_ids,
            stage_holdouts=stage_holdouts,
        )

        # Complete each stage's experiment with its during-search metric + held-out.
        for idx in range(5):
            eid = stage_experiment_ids[idx]
            if eid is None:
                continue
            metric = stage_results[idx][2] if idx < len(stage_results) else None
            extra: dict = {}
            if metric is not None:
                extra["attitude_error_deg"] = float(metric.mean_attitude_error_deg)
                extra["stable_rate"] = float(metric.acc)
                extra["reward"] = float(metric.fitness)
            ho = stage_holdouts.get(_CONTROLLER_STAGES[idx].upper())
            if ho is not None:
                extra["holdout_attitude_error_deg"] = float(ho.mean_attitude_error_deg)
                extra["holdout_stable_rate"] = float(ho.acc)
                extra["holdout_reward"] = float(ho.fitness)
            try:
                if extra:
                    worker.tracker.update_experiment_extra_metrics(eid, extra)
                worker.tracker.update_experiment_status(eid, "completed")
            except Exception as e:
                worker._log(f"controller flow {flow_id}: experiment {eid} completion write failed: {e}")

        worker.client.flow_completed(flow_id)
        final_m = stage_results[-1][2] if stage_results else None
        if final_m is not None:
            worker._log(f"Controller flow completed: err={final_m.mean_attitude_error_deg:.2f}° "
                        f"stable={final_m.acc*100:.0f}% reward={final_m.fitness:.2f}")
        else:
            worker._log("Controller flow completed (no final metric).")

    except Exception as e:
        error_msg = str(e).lower()
        is_shutdown = worker._stop_current_flow or "shutdown" in error_msg or "stopped" in error_msg
        flow_exists = worker.client.get_flow(flow_id) is not None
        if not flow_exists:
            worker._log(f"Controller flow {flow_id} was deleted, cleaning up")
        elif is_shutdown:
            cur = worker.client.get_flow(flow_id)
            if cur and cur.get("status") == "cancelled":
                worker._log("Controller flow was cancelled by user, keeping cancelled status")
            else:
                worker._log("Controller flow stopped due to shutdown, re-queuing for resume")
                try:
                    worker.client.requeue_flow(flow_id)
                except Exception:
                    worker._log(f"Warning: could not re-queue controller flow {flow_id}")
        else:
            worker._log(f"Controller flow failed: {e}")
            import traceback
            traceback.print_exc()
            try:
                worker.client.flow_failed(flow_id, str(e))
            except Exception:
                pass
    finally:
        worker._stop_heartbeat()
        worker.current_flow_id = None
        worker._stop_current_flow = False
        worker._close_log_file()


def main():
    parser = argparse.ArgumentParser(
        description="Flow runner - executes a single flow by id, then exits"
    )
    parser.add_argument("flow_id", type=int, help="Id of the flow to run")
    parser.add_argument("--url", default="https://localhost:3000", help="Dashboard URL")
    parser.add_argument("--context", type=int, default=4, help="Default context size")

    ssl_group = parser.add_mutually_exclusive_group()
    ssl_group.add_argument(
        "--ssl-cert",
        type=Path,
        help="Path to CA certificate for SSL verification (for self-signed certs)",
    )
    ssl_group.add_argument(
        "--no-ssl-verify",
        action="store_true",
        help="Disable SSL certificate verification (development only)",
    )

    args = parser.parse_args()

    if args.no_ssl_verify:
        verify_ssl: bool | str = False
    elif args.ssl_cert:
        verify_ssl = str(args.ssl_cert)
    else:
        verify_ssl = False  # Default: skip SSL verify (self-signed cert)

    exit_code = run_one_flow(
        flow_id=args.flow_id,
        dashboard_url=args.url,
        context_size=args.context,
        verify_ssl=verify_ssl,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
