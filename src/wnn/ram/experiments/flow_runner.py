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

    # Execute. _execute_flow owns the full lifecycle: flow_started, PID
    # registration, heartbeat start/stop, evaluator construction, flow.run, and
    # the terminal status write (completed / failed / paused / re-queued) in its
    # own try/except/finally. It does NOT re-raise, so we read the outcome back
    # from the DB to choose our exit code.
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
