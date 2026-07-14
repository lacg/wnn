"""
Flow Worker - Polls dashboard for queued flows and executes them.

Usage:
    python -m wnn.ram.experiments.worker [--url http://localhost:3000] [--poll-interval 10]

The worker:
1. Polls the dashboard API for flows with status "queued"
2. Picks up the oldest queued flow
3. Runs it using the Flow class
4. Updates status to "completed" or "failed"
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from wnn.ram.fitness import FitnessCalculatorType
from wnn.ram.experiments.dashboard_client import DashboardClient, DashboardClientConfig
from wnn.ram.experiments.flow import Flow, FlowConfig
from wnn.ram.experiments.experiment import ExperimentConfig, ExperimentType, GridSource
from wnn.ram.experiments.tracker import create_tracker, ExperimentTracker
from wnn.ram.experiments import scheduler

# How often to send heartbeats (seconds)
HEARTBEAT_INTERVAL = 30

# Consider a flow stale if no heartbeat for this many seconds
STALE_THRESHOLD = 90


class FlowWorker:
    """
    Worker that polls for queued flows and executes them.
    """

    def __init__(
        self,
        dashboard_url: str = "https://localhost:3000",
        poll_interval: int = 10,
        checkpoint_base_dir: Path = Path("checkpoints"),
        context_size: int = 4,
        db_path: Optional[str] = None,
        verify_ssl: bool | str = False,
        cpu_budget: Optional[int] = None,
    ):
        self.dashboard_url = dashboard_url
        self.poll_interval = poll_interval
        self.checkpoint_base_dir = checkpoint_base_dir
        self.context_size = context_size
        self.verify_ssl = verify_ssl
        self.running = True
        self._stop_current_flow = False  # Flag to stop current flow but keep worker running
        # Pause flag: separate from _stop_current_flow so we can distinguish
        # 'pause and resume later' from 'cancelled / shutdown'. Set when the
        # dashboard's POST /api/flows/<id>/pause flips pause_requested=1.
        self._pause_current_flow = False
        self.current_flow_id: Optional[int] = None

        # Budget-aware scheduler state. The worker no longer runs flows
        # in-process; it spawns one `flow_runner` SUBPROCESS per admitted flow
        # (each owns its own RAYON pool + Rust cancel flag) and schedules them
        # under a CPU budget so e.g. a 3-core controller can co-run with a
        # 10-core IDS flow without stealing IDS cores. Dynamic budget = cores − 3.
        self.cpu_budget = cpu_budget if cpu_budget is not None else scheduler.detect_budget()
        # flow_id -> {"proc": Popen, "type": str, "cores": int, "name": str}
        self._running: dict[int, dict] = {}

        # Heartbeat thread for detecting stale flows
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop_event = threading.Event()

        # Pre-cached data (loaded once at startup)
        self._tokenizer = None
        self._train_tokens: Optional[list[int]] = None
        self._test_tokens: Optional[list[int]] = None
        self._validation_tokens: Optional[list[int]] = None
        self._vocab_size: Optional[int] = None
        self._cluster_order: Optional[list[int]] = None  # Tokens sorted by frequency

        # Log file for current flow (path only, not kept open)
        self._log_file: Optional[Path] = None
        self._log_dir = Path("logs")

        # Setup client with SSL configuration
        config = DashboardClientConfig(base_url=dashboard_url, verify_ssl=verify_ssl)
        self.client = DashboardClient(config, logger=self._log)

        # V2 Tracker for direct database writes (same db as dashboard)
        # Standard location: db/wnn.db relative to project root
        if db_path is None:
            db_path = str(Path("db/wnn.db").absolute())
        self.tracker: Optional[ExperimentTracker] = create_tracker(db_path=db_path, logger=self._log)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _log(self, message: str):
        """Log with timestamp to stdout and log file.

        Uses file locking with open/close pattern to properly coordinate
        with Rust writes. Both Python and Rust open fresh, lock, write, close.
        """
        import fcntl

        # Use HH:MM:SS | format for dashboard parser compatibility
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"{timestamp} | {message}\n"
        print(line, end='', flush=True)

        # Write to log file with proper locking (open fresh each time)
        if self._log_file:
            with open(self._log_file, 'a') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(line)
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _open_log_file(self, flow_name: str) -> Path:
        """Create a log file for a flow and return its path.

        Note: We don't keep the file handle open. Each write opens fresh
        with locking to coordinate with Rust writes.
        """
        now = datetime.now()
        date_dir = self._log_dir / now.strftime("%Y/%m/%d")
        date_dir.mkdir(parents=True, exist_ok=True)

        safe_name = flow_name.lower().replace(" ", "_").replace("/", "_")
        timestamp = now.strftime("%H%M%S")
        log_file = date_dir / f"{safe_name}_{timestamp}.log"

        self._log_file = log_file
        # Create the file (touch) but don't keep it open
        log_file.touch()
        return log_file

    def _close_log_file(self):
        """Mark log file as closed (just clears the path reference)."""
        self._log_file = None

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals.

        SIGTERM: If running a flow, stop it gracefully. If idle, stop worker.
        SIGINT (Ctrl+C): Stop worker entirely

        31/05/2026: in addition to setting the local stop flags, propagate to
        the Rust accelerator's process-wide cancel flag so any in-flight Rust
        evaluator call (search_offspring batch, etc.) returns early with
        partial results instead of running its full ~10-second-to-minutes
        batch. Without this, SIGTERM still waits for Rust to finish naturally.
        """
        # Propagate to Rust ASAP — cheap (atomic store), cooperative (Rust polls
        # between batches). With the scheduler the worker process runs NO Rust
        # training itself (children do), so this mainly matters for the legacy
        # in-process path; harmless either way. Each child also sets its OWN flag
        # when we forward SIGTERM below.
        try:
            import ram_accelerator
            ram_accelerator.set_cancel_flag()
        except Exception as e:
            self._log(f"Could not propagate cancel to Rust accelerator: {e}")

        # Forward the shutdown to every live flow subprocess so each stops
        # gracefully at the end of its current generation and re-queues itself
        # (the flow_runner's _execute_flow handles the graceful unwind).
        self._forward_signal_to_children(signal.SIGTERM)

        if signum == signal.SIGINT:
            # Ctrl+C - stop everything
            self._log(f"Received SIGINT, stopping scheduler...")
            self.running = False
            self._stop_current_flow = True
        else:
            # SIGTERM - stop scheduling new work; the run() loop's shutdown wait
            # reaps the children we just signalled. Keep the legacy in-proc guard
            # for the (now unused) in-process path.
            if self.current_flow_id:
                self._log(f"Received SIGTERM, stopping current in-process flow...")
                self._stop_current_flow = True
                self._log(f"Flow {self.current_flow_id} will stop after current generation/iteration")
            else:
                self._log(f"Received SIGTERM, stopping scheduler "
                          f"({len(self._running)} flow subprocess(es) signalled)...")
                self.running = False

    def _forward_signal_to_children(self, sig: int):
        """Send `sig` to every live flow subprocess (best-effort)."""
        for fid, meta in list(self._running.items()):
            proc = meta.get("proc")
            if proc is None or proc.poll() is not None:
                continue
            try:
                proc.send_signal(sig)
                self._log(f"Forwarded signal {sig} to flow {fid} subprocess (pid={proc.pid})")
            except Exception as e:
                self._log(f"Could not signal flow {fid} subprocess: {e}")

    def should_stop(self) -> bool:
        """Check if shutdown OR pause has been requested. Used by flow/experiment/strategy.

        Checks both local flags and dashboard flow status for cancellation/deletion/pause.
        This provides a fallback if SIGTERM delivery fails (e.g., missing PID).

        Pause and stop both unwind the GA loop the same way (raise StopIteration in
        architecture_strategies._on_generation_start, propagates to flow.run as
        FlowStoppedError). The two are distinguished by the `_pause_current_flow`
        flag, inspected in the worker's exception handler to choose `status='paused'`
        vs `status='cancelled'`.
        """
        # Check local flags first (fast path)
        if self._stop_current_flow or self._pause_current_flow or not self.running:
            return True

        # Periodically check dashboard flow status as fallback
        # This catches cancellation/pause requests when PID wasn't registered
        if self.current_flow_id:
            try:
                flow = self.client.get_flow(self.current_flow_id)
                if flow is None:
                    # Flow was deleted - stop working on it
                    self._log(f"Flow {self.current_flow_id} was deleted, stopping...")
                    self._stop_current_flow = True
                    return True
                if flow.get("status") in ("cancelled", "failed"):
                    self._log(f"Flow {self.current_flow_id} was cancelled via dashboard, stopping...")
                    self._stop_current_flow = True
                    return True
                # Pause requested via POST /api/flows/<id>/pause
                if flow.get("pause_requested"):
                    self._log(f"Flow {self.current_flow_id} pause requested via dashboard, pausing at end of generation...")
                    self._pause_current_flow = True
                    return True
            except Exception:
                pass  # Network error, continue running

        return False

    def _start_heartbeat(self, flow_id: int):
        """Start the heartbeat thread for a flow."""
        self._heartbeat_stop_event.clear()

        def heartbeat_loop():
            while not self._heartbeat_stop_event.wait(HEARTBEAT_INTERVAL):
                try:
                    self.client.send_heartbeat(flow_id)
                except Exception:
                    pass  # Ignore errors, keep trying

        self._heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _stop_heartbeat(self):
        """Stop the heartbeat thread."""
        if self._heartbeat_thread:
            self._heartbeat_stop_event.set()
            self._heartbeat_thread.join(timeout=2)
            self._heartbeat_thread = None

    def _recover_stale_flows(self):
        """Re-queue stale running flows (P3, 12/06/2026 — was: mark FAILED).

        A flow is stale if it's marked as 'running' but hasn't received a
        heartbeat recently (worker crashed or was killed). Re-queuing preserves
        all data and lets the next pickup resume from the per-gen checkpoint;
        the old fail-marking was the root cause of "restart kills running
        flows" (a failed flow loses auto-resume). The dashboard now runs its
        own stale-requeue task (180s) — this worker-side pass is a fallback
        for dashboardless setups and uses the same requeue semantics so the
        two never fight (both target status='running' only).
        """
        try:
            flows = self.client.list_flows(status="running", limit=10)
            now = datetime.utcnow()

            for flow in flows:
                # Skip flows owned by THIS worker. With the scheduler, a flow we
                # spawned runs in its own subprocess and heartbeats itself, so
                # any id in self._running (or the legacy in-proc current_flow_id)
                # is alive-by-us — never mark it stale.
                if flow["id"] == self.current_flow_id or flow["id"] in self._running:
                    continue

                last_heartbeat = flow.get("last_heartbeat")
                is_stale = False

                if last_heartbeat is None:
                    started_at = flow.get("started_at")
                    if started_at:
                        try:
                            started = datetime.fromisoformat(started_at.replace("Z", "+00:00")).replace(tzinfo=None)
                            is_stale = (now - started).total_seconds() > STALE_THRESHOLD
                        except Exception:
                            pass
                else:
                    try:
                        hb_time = datetime.fromisoformat(last_heartbeat.replace("Z", "+00:00")).replace(tzinfo=None)
                        is_stale = (now - hb_time).total_seconds() > STALE_THRESHOLD
                    except Exception:
                        pass

                if is_stale:
                    flow_id = flow["id"]
                    flow_name = flow.get("name", f"Flow {flow_id}")
                    self._log(f"Stale flow detected: {flow_name} (ID: {flow_id}) — re-queuing for resume (data preserved)")
                    try:
                        self.client.requeue_flow(flow_id)
                    except Exception as e:
                        self._log(f"  Failed to re-queue stale flow {flow_id}: {e}")
        except Exception:
            pass

    def _ensure_lm_cache(self):
        """Lazily load GPT-2 tokenizer + WikiText-2 cache on first LM-flow demand.

        Idempotent: returns immediately if already loaded. IDS-only workflows
        never call this and avoid the ~2 GB resident cost.
        """
        if self._tokenizer is not None:
            return

        from collections import Counter

        from datasets import load_dataset
        from transformers import AutoTokenizer

        self._log("Lazy-loading GPT-2 tokenizer + WikiText-2 cache (first LM flow)...")

        # Load tokenizer (will download if not cached)
        self._log("  Loading GPT2 tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self._vocab_size = self._tokenizer.vocab_size
        self._log(f"  Tokenizer loaded (vocab size: {self._vocab_size:,})")

        # Load dataset (will download if not cached)
        self._log("  Loading WikiText-2 dataset...")
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        except Exception:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)

        # Tokenize all three splits: train, test, validation
        # - train: used for training RAM neurons during GA/TS (~2.4M tokens)
        # - test: used for fitness scoring during GA/TS (~284K tokens)
        # - validation: held out for full evaluation only (~248K tokens)
        self._log("  Tokenizing dataset...")
        train_text = "\n".join(dataset["train"]["text"])
        test_text = "\n".join(dataset["test"]["text"])
        validation_text = "\n".join(dataset["validation"]["text"])

        self._train_tokens = self._tokenizer.encode(train_text, add_special_tokens=False)
        self._test_tokens = self._tokenizer.encode(test_text, add_special_tokens=False)
        self._validation_tokens = self._tokenizer.encode(validation_text, add_special_tokens=False)

        self._log(f"  Cached {len(self._train_tokens):,} train, "
                  f"{len(self._test_tokens):,} test, "
                  f"{len(self._validation_tokens):,} validation tokens")

        # Compute cluster order (tokens sorted by frequency, most frequent first)
        self._log("  Computing token frequencies...")
        freq_counter = Counter(self._train_tokens)
        self._cluster_order = sorted(range(self._vocab_size), key=lambda i: -freq_counter.get(i, 0))
        tokens_with_freq = sum(1 for i in range(self._vocab_size) if freq_counter.get(i, 0) > 0)
        self._log(f"  Tokens with freq > 0: {tokens_with_freq:,}")

        self._log("LM cache ready!")

    def run(self):
        """Main worker loop: budget-aware, type-balanced flow SCHEDULER.

        Each poll: reap finished flow subprocesses (reclaim their cores),
        recover stale flows owned by no live worker, then ADMIT queued flows up
        to the CPU budget (`scheduler.admit`) and spawn one `flow_runner`
        subprocess per admission. Gating analysis runs in-process only when fully
        idle (no flow subprocesses), so it never blocks scheduling.
        """
        self._log(f"Scheduler started, polling {self.dashboard_url} every {self.poll_interval}s")
        self._log(f"CPU budget: {self.cpu_budget} cores (detected {os.cpu_count()} cores)")
        self._log(f"Checkpoints will be saved to {self.checkpoint_base_dir}")

        while self.running:
            try:
                # 1. Reap finished flow subprocesses -> reclaim their cores.
                self._reap_finished()

                # 2. Recover stale flows (crashed workers); skips our own children.
                self._recover_stale_flows()

                # 3. Admit queued flows up to the budget, balanced by type.
                # NOTE: the API returns flows ORDER BY id DESC, so a small limit
                # would hand admit() only the NEWEST queued flows — defeating the
                # global-FIFO (oldest-id-first) intent (e.g. low-id cicids tail
                # starved behind high-id ciciot). 300 ≥ any realistic queue depth,
                # so admit() sees the WHOLE queue and picks the true min id.
                try:
                    queued = self.client.list_flows(status="queued", limit=300)
                except Exception as e:
                    self._log(f"Failed to fetch queued flows: {e}")
                    queued = []
                running_meta = [
                    {"id": fid, "type": m["type"], "cores": m["cores"]}
                    for fid, m in self._running.items()
                ]
                admitted = scheduler.admit(queued, running_meta, self.cpu_budget)
                for flow in admitted:
                    self._spawn_flow(flow)

                # 4. If fully idle, drain a gating run in-process; else just wait.
                if not self._running and not admitted:
                    gating_run = self._get_next_gating_run()
                    if gating_run:
                        self._execute_gating_job(gating_run)
                        continue  # poll again immediately, don't sleep
                time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self._log(f"Error in scheduler loop: {e}")
                time.sleep(self.poll_interval)

        # Graceful shutdown: give any live flow subprocesses a moment to stop
        # cleanly (SIGTERM was already forwarded in _handle_signal -> they
        # re-queue themselves), then reap whatever finished.
        if self._running:
            self._log(f"Waiting up to 30s for {len(self._running)} flow subprocess(es) to stop...")
            deadline = time.monotonic() + 30
            while self._running and time.monotonic() < deadline:
                self._reap_finished()
                time.sleep(1)
            if self._running:
                self._log(f"{len(self._running)} flow subprocess(es) still running at shutdown; "
                          f"they own their own re-queue/heartbeat and will be recovered if stale.")
        self._log("Scheduler stopped")

    def _reap_finished(self):
        """Poll spawned flow subprocesses; drop any that exited and reclaim cores."""
        done: list[tuple[int, int]] = []
        for fid, meta in self._running.items():
            rc = meta["proc"].poll()
            if rc is not None:
                done.append((fid, rc))
        for fid, rc in done:
            meta = self._running.pop(fid)
            self._log(
                f"Flow {fid} ('{meta.get('name', fid)}') subprocess exited rc={rc}; "
                f"reclaimed {meta['cores']} cores "
                f"({self.cpu_budget - sum(m['cores'] for m in self._running.values())} now free)"
            )

    def _spawn_flow(self, flow: dict):
        """Spawn one `flow_runner` subprocess for an admitted flow.

        The child runs in its OWN session (start_new_session) so a terminal
        SIGINT doesn't reach it directly — the worker forwards signals
        deliberately in _handle_signal. RAYON_NUM_THREADS pins the child's Rust
        thread pool to its core allotment so the budget is actually enforced.
        """
        fid = flow["id"]
        cores = scheduler.flow_cores(flow)
        ftype = scheduler.flow_type(flow)
        name = flow.get("name", f"Flow {fid}")

        env = os.environ.copy()
        env["RAYON_NUM_THREADS"] = str(cores)

        cmd = [
            sys.executable, "-m", "wnn.ram.experiments.flow_runner", str(fid),
            "--url", self.dashboard_url,
            "--context", str(self.context_size),
        ]
        if self.verify_ssl is False:
            cmd.append("--no-ssl-verify")
        elif isinstance(self.verify_ssl, str):
            cmd += ["--ssl-cert", self.verify_ssl]

        try:
            proc = subprocess.Popen(cmd, env=env, start_new_session=True)
        except Exception as e:
            self._log(f"Failed to spawn flow_runner for flow {fid}: {e}")
            return

        self._running[fid] = {"proc": proc, "type": ftype, "cores": cores, "name": name}
        free = self.cpu_budget - sum(m["cores"] for m in self._running.values())
        self._log(
            f"Spawned flow_runner for flow {fid} ('{name}'): {ftype}, {cores} cores, "
            f"pid={proc.pid} (budget {self.cpu_budget}, {free} free)"
        )

    def _get_next_queued_flow(self) -> Optional[dict]:
        """Get the next queued flow from the dashboard.

        Only returns a flow if no other flow is currently running,
        ensuring at most one flow executes at a time.
        """
        try:
            # Check if any flow is already running (from this or another worker)
            running = self.client.list_flows(status="running", limit=1)
            if running:
                return None  # Wait for the running flow to finish

            flows = self.client.list_flows(status="queued", limit=1)
            if flows:
                return flows[0]
        except Exception as e:
            self._log(f"Failed to fetch queued flows: {e}")
        return None

    def _execute_flow(self, flow_data: dict):
        """Execute a single flow."""
        flow_id = flow_data["id"]
        flow_name = flow_data.get("name", f"Flow {flow_id}")
        self.current_flow_id = flow_id

        self._reset_rust_cancel_flag()

        # Open log file for this flow
        log_file = self._open_log_file(flow_name)
        self._log(f"Logging to: {log_file}")

        # Tell dashboard to watch this log file
        self.client.watch_log(str(log_file.absolute()))

        self._log(f"=" * 60)
        self._log(f"Starting flow: {flow_name} (ID: {flow_id})")
        self._log(f"=" * 60)

        try:
            # Mark as running and register PID for stop/restart
            self.client.flow_started(flow_id)
            self.client.register_flow_pid(flow_id, os.getpid())

            # Start heartbeat thread (allows detection of stale flows if worker crashes)
            self._start_heartbeat(flow_id)

            # Parse flow configuration
            config = flow_data.get("config", {})
            params = config.get("params", {})

            # Ingestion validation: warn loudly on unknown/typo'd keys and log
            # the run's effective config (docs/ARCHITECTURE_REVIEW_2026-06.md §3.2).
            from wnn.ram.experiments.params import validate_flow_params
            validate_flow_params(params, log=self._log, flow_id=flow_id)

            self._apply_env_overrides(params, flow_id)

            flow = self._build_flow(flow_data, flow_id, flow_name, params)

            # Check if we should start from a specific experiment (skip earlier ones)
            start_from_experiment = params.get("start_from_experiment")
            if start_from_experiment is not None:
                self._log(f"Starting from experiment {start_from_experiment} (skipping {start_from_experiment} earlier experiments)")

            # Run the flow (seed checkpoint is already loaded via flow_config.seed_checkpoint_path)
            result = flow.run(resume_from=start_from_experiment)

            # Mark as completed
            self.client.flow_completed(flow_id)
            self._log(f"Flow completed: CE={result.final_fitness:.4f}")

        except Exception as e:
            self._handle_flow_exception(flow_id, e)

        finally:
            self._cleanup_after_flow()

    def _reset_rust_cancel_flag(self):
        """Reset the Rust cancel flag from any prior flow (31/05/2026).

        The flag is process-wide; without this a SIGTERM that arrived between
        flows would short-circuit the next one immediately — silently producing
        untrained, trivial (F1=0.49) genomes for the whole flow. Log on failure
        instead of swallowing: a silently-failed reset here is exactly how a
        stuck flag would go unnoticed (01/06/2026 investigation).
        """
        try:
            import ram_accelerator
            ram_accelerator.reset_cancel_flag()
            if ram_accelerator.is_cancelled():
                self._log("WARNING: cancel flag STILL set after reset_cancel_flag() "
                          "at flow start — training would be short-circuited. Investigate.")
        except Exception as e:
            self._log(f"WARNING: could not reset Rust cancel flag at flow start: {e} "
                      "— a stale flag could short-circuit this flow's training (F1=0.49 risk).")

    def _apply_env_overrides(self, params: dict, flow_id: int):
        """Apply per-flow env overrides read by the Rust accelerator.

        Previous values are saved on self._prev_*_env and restored by
        _cleanup_after_flow() so subsequent flows see the pre-flow state.
        """
        self._apply_oi_override(params, flow_id)
        self._apply_hsr_override(params, flow_id)

        # Per-flow grid-search parallelism cap. Needed for very-large
        # datasets (e.g. CIC-IoT-2023 46M neto_full) where the default
        # 4-wide grid pool OOMs the 64 GB unified memory. Set
        # `wnn_grid_search_parallel: 1` in the flow params to serialize.
        self._prev_grid_parallel_env = self._apply_int_override(
            params, "wnn_grid_search_parallel", "WNN_GRID_SEARCH_PARALLEL", "GRID", flow_id,
        )

        # Per-flow GA hybrid batch-size cap. The auto-estimator in
        # calculate_pool_size() is calibrated for ~100K-row training data
        # (~3K sparse entries/neuron); at 46M rows the actual sparse-cell
        # cost is ~400× higher so the estimator under-budgets and OOMs.
        # Set `wnn_batch_size: 1` for 46M flows to force serial GA eval.
        self._prev_batch_size_env = self._apply_int_override(
            params, "wnn_batch_size", "WNN_BATCH_SIZE", "BATCH", flow_id,
        )

        self._apply_conservative_override(params, flow_id)

        # Cap the rayon CPU pool below the core count (pairs with wnn_no_metal).
        self._prev_num_threads_env = self._apply_int_override(
            params, "wnn_num_threads", "RAYON_NUM_THREADS", "CONSERVATIVE", flow_id,
        )

    def _apply_oi_override(self, params: dict, flow_id: int):
        """Per-flow OI gating: read `wnn_order_independent_train` from flow
        params and set the env var the Rust accelerator reads. Restored
        in _cleanup_after_flow() so subsequent non-OI flows aren't affected.
        Accepts truthy values (True / "1" / "true" / 1).
        """
        oi_param = params.get("wnn_order_independent_train")
        oi_truthy = (
            oi_param is True
            or oi_param == 1
            or (isinstance(oi_param, str) and oi_param.lower() in ("1", "true"))
        )
        self._prev_oi_env = os.environ.get("WNN_ORDER_INDEPENDENT_TRAIN")
        if oi_truthy:
            os.environ["WNN_ORDER_INDEPENDENT_TRAIN"] = "1"
            self._log(f"[OI] WNN_ORDER_INDEPENDENT_TRAIN=1 enabled for flow {flow_id}")

    def _apply_hsr_override(self, params: dict, flow_id: int):
        """Per-flow hybrid-speed-ratio: either an explicit override from
        flow params (numeric or string), or computed from the workload
        function predict_hsr_from_params(). Either path overrides the
        env var the Rust accelerator reads for the B12 CPU+GPU split
        cap; restored in _cleanup_after_flow().
        """
        self._prev_hsr_env = os.environ.get("WNN_HYBRID_SPEED_RATIO")
        hsr_param = params.get("wnn_hybrid_speed_ratio")
        if hsr_param is not None:
            # Explicit override (e.g., HSR sweep experiments).
            hsr_str = str(hsr_param).strip()
            try:
                float(hsr_str)
            except ValueError:
                self._log(f"[HSR] WARN: wnn_hybrid_speed_ratio={hsr_str!r} is not numeric, ignoring")
            else:
                os.environ["WNN_HYBRID_SPEED_RATIO"] = hsr_str
                self._log(f"[HSR] WNN_HYBRID_SPEED_RATIO={hsr_str} set for flow {flow_id} (explicit)")
        else:
            # No explicit param — predict from workload (max_neurons,
            # max_bits, thermo_width, n_train_samples per fold).
            try:
                from .hsr_function import predict_hsr_from_params
                hsr_pred = predict_hsr_from_params(params)
                os.environ["WNN_HYBRID_SPEED_RATIO"] = str(hsr_pred)
                self._log(f"[HSR] WNN_HYBRID_SPEED_RATIO={hsr_pred} set for flow {flow_id} (predicted from workload)")
            except Exception as e:
                self._log(f"[HSR] WARN: predict_hsr_from_params failed ({e}); falling back to env default {self._prev_hsr_env}")

    def _apply_int_override(
        self,
        params: dict,
        param_key: str,
        env_var: str,
        log_tag: str,
        flow_id: int,
    ) -> Optional[str]:
        """Apply one integer-valued per-flow env override; returns the previous env value."""
        prev = os.environ.get(env_var)
        param = params.get(param_key)
        if param is not None:
            try:
                n = max(1, int(param))
            except (TypeError, ValueError):
                self._log(f"[{log_tag}] WARN: {param_key}={param!r} not int, ignoring")
            else:
                os.environ[env_var] = str(n)
                self._log(f"[{log_tag}] {env_var}={n} set for flow {flow_id}")
        return prev

    def _apply_conservative_override(self, params: dict, flow_id: int):
        """Per-flow CPU-only / conservative mode. Set `wnn_no_metal: true`
        to disable Metal GPU and run CPU-only (the accelerator reads
        WNN_NO_METAL). This trades throughput for a much smaller memory
        footprint + leaves the GPU + unified-memory headroom free so the
        rest of the machine stays usable (no beachball). Pair with
        `wnn_num_threads` to cap the rayon CPU pool below the core count.
        """
        self._prev_no_metal_env = os.environ.get("WNN_NO_METAL")
        no_metal_param = params.get("wnn_no_metal")
        no_metal_truthy = (
            no_metal_param is True
            or no_metal_param == 1
            or (isinstance(no_metal_param, str) and no_metal_param.lower() in ("1", "true"))
        )
        if no_metal_truthy:
            os.environ["WNN_NO_METAL"] = "1"
            self._log(f"[CONSERVATIVE] WNN_NO_METAL=1 (CPU-only) set for flow {flow_id}")

    def _create_flow_evaluators(self, architecture_type: str, context_size: int, params: dict):
        """Load data and create the optimizer evaluator (+ IDS test/S1 evaluators).

        Returns (evaluator, ids_test_evaluator, ids_s1_evaluator, ids_s1_test_evaluator);
        the IDS entries are None for non-IDS architectures.
        """
        ids_test_evaluator = None
        ids_s1_evaluator = None
        ids_s1_test_evaluator = None
        if architecture_type == "ids":
            ids_result = self._create_ids_evaluators(params)
            if len(ids_result) == 4:
                # Hierarchical: (s0_opt, s0_test, s1_opt, s1_test)
                evaluator, ids_test_evaluator, ids_s1_evaluator, ids_s1_test_evaluator = ids_result
            else:
                evaluator, ids_test_evaluator = ids_result
        elif architecture_type == "multi_stage":
            evaluator = self._create_multistage_evaluator(context_size, params)
        elif architecture_type == "bitwise":
            evaluator = self._create_bitwise_evaluator(context_size, params)
        elif architecture_type == "controller":
            evaluator = self._create_controller_evaluator(params)
        else:
            evaluator = self._create_evaluator(context_size, params.get("seed"), params.get("neuron_sample_rate", 0.25))
        return evaluator, ids_test_evaluator, ids_s1_evaluator, ids_s1_test_evaluator

    def _build_base_flow_config(
        self,
        flow_data: dict,
        flow_name: str,
        params: dict,
        exp_configs: list,
        context_size: int,
        architecture_type: str,
    ) -> FlowConfig:
        """Create the FlowConfig with shared (architecture-independent) settings."""
        fitness_calculator_type = self._parse_fitness_calculator(params.get("fitness_calculator"))
        return FlowConfig(
            name=flow_name,
            experiments=exp_configs,
            description=flow_data.get("description"),
            tier_config=self._parse_tier_config(params.get("tier_config")),
            optimize_tier0_only=params.get("optimize_tier0_only", params.get("tier0_only", False)),
            context_size=context_size,
            patience=params.get("patience", 3),
            fitness_percentile=params.get("fitness_percentile"),
            fitness_calculator_type=fitness_calculator_type,
            fitness_weight_ce=params.get("fitness_weight_ce", 1.0),
            fitness_weight_acc=params.get("fitness_weight_acc", 1.0),
            fitness_weight_f1=params.get("fitness_weight_f1", params.get("ids_fitness_weight_f1", 0.0)),
            fitness_weight_fpr=params.get("fitness_weight_fpr", params.get("ids_fitness_weight_fpr", 0.0)),
            seed=params.get("seed"),
            architecture_type=architecture_type,
        )

    def _apply_bitwise_config(self, flow_config: FlowConfig, params: dict):
        """Add bitwise-specific config."""
        flow_config.num_clusters = params.get("num_clusters", 16)
        flow_config.memory_mode = params.get("memory_mode", "QUAD_WEIGHTED")
        flow_config.neuron_sample_rate = params.get("neuron_sample_rate", 0.25)
        flow_config.min_bits = params.get("min_bits", 10)
        flow_config.max_bits = params.get("max_bits", 24)
        flow_config.min_neurons = params.get("min_neurons", 10)
        flow_config.max_neurons = params.get("max_neurons", 300)
        flow_config.sparse_threshold = params.get("sparse_threshold")

    def _parse_stage_mode_param(self, raw_mode):
        """Parse the multi-stage `stage_mode` param (string or list) into StageMode list."""
        from wnn.ram.experiments.experiment import StageMode
        mode_map = {
            "input_concat": StageMode.INPUT_CONCAT,
            "selector": StageMode.SELECTOR,
        }
        if isinstance(raw_mode, str):
            return [mode_map.get(raw_mode, StageMode.INPUT_CONCAT)]
        if isinstance(raw_mode, list):
            return [
                mode_map.get(m, StageMode.INPUT_CONCAT) if isinstance(m, str) else m
                for m in raw_mode
            ]
        return None

    def _apply_multistage_config(self, flow_config: FlowConfig, params: dict):
        """Add multi-stage config."""
        flow_config.num_stages = params.get("num_stages", 2)
        flow_config.stage_k = params.get("stage_k")
        flow_config.stage_cluster_type = params.get("stage_cluster_type")
        stage_mode = self._parse_stage_mode_param(params.get("stage_mode"))
        if stage_mode is not None:
            flow_config.stage_mode = stage_mode
        flow_config.memory_mode = params.get("memory_mode", "QUAD_WEIGHTED")
        flow_config.neuron_sample_rate = params.get("neuron_sample_rate", 0.25)
        flow_config.min_bits = params.get("min_bits", 4)
        flow_config.max_bits = params.get("max_bits", 24)
        flow_config.min_neurons = params.get("min_neurons", 5)
        flow_config.max_neurons = params.get("max_neurons", 300)
        flow_config.invalid_mode = params.get("invalid_mode", False)
        flow_config.top_m = params.get("top_m", 5)
        flow_config.label_smoothing = params.get("label_smoothing", 0.0)
        flow_config.unigram_lambda = params.get("unigram_lambda", 0.0)
        flow_config.bigram_lambda = params.get("bigram_lambda", 0.0)
        # Per-stage bounds from dashboard
        flow_config.stage_min_bits_list = params.get("stage_min_bits")
        flow_config.stage_max_bits_list = params.get("stage_max_bits")
        flow_config.stage_min_neurons_list = params.get("stage_min_neurons")
        flow_config.stage_max_neurons_list = params.get("stage_max_neurons")

    def _apply_ids_config(self, flow_config: FlowConfig, params: dict):
        """Add IDS-specific config."""
        flow_config.ids_classification = params.get("ids_classification", "binary")
        flow_config.ids_arch_type = params.get("ids_arch_type", "tiered")
        flow_config.ids_n_bits = params.get("ids_n_bits", 8)
        flow_config.ids_val_fraction = params.get("ids_val_fraction", 0.25)
        flow_config.ids_num_parts = params.get("ids_num_parts", 5)
        flow_config.ids_k_folds = params.get("ids_k_folds", 5)
        flow_config.ids_fitness_weight_f1 = params.get("ids_fitness_weight_f1", 0.0)
        flow_config.ids_fitness_weight_fpr = params.get("ids_fitness_weight_fpr", 0.0)
        flow_config.ids_split = params.get("ids_split", "standard")
        flow_config.ids_feature_selection = params.get("ids_feature_selection", "all")
        # Global and per-stage bounds (shared with bitwise/multi_stage)
        flow_config.min_bits = params.get("min_bits", 4)
        flow_config.max_bits = params.get("max_bits", 24)
        flow_config.min_neurons = params.get("min_neurons", 5)
        flow_config.max_neurons = params.get("max_neurons", 300)
        flow_config.stage_min_bits_list = params.get("stage_min_bits")
        flow_config.stage_max_bits_list = params.get("stage_max_bits")
        flow_config.stage_min_neurons_list = params.get("stage_min_neurons")
        flow_config.stage_max_neurons_list = params.get("stage_max_neurons")
        flow_config.max_bit_delta = params.get("max_bit_delta", 0)

        # Build dataset_key for validation cache scoping (prevents cross-dataset cache poisoning).
        # Includes _raw, _inv-<mode>, and _oi<0|1> suffixes so canonical/raw flows and
        # different training algorithms don't share cache when other params match.
        ds = params.get("ids_dataset", "unsw-nb15")
        nb = params.get("ids_n_bits", 8)
        sp = params.get("ids_split", "standard")
        raw_suffix = "_raw" if params.get("ids_raw", False) else ""
        inv_mode = params.get("ids_invalid_encoding")
        inv_suffix = f"_inv-{inv_mode}" if inv_mode and inv_mode != "none" else ""
        oi_suffix = "_oi1" if params.get("wnn_order_independent_train") else "_oi0"
        flow_config.dataset_key = f"{ds}_{nb}b_{sp}{raw_suffix}{inv_suffix}{oi_suffix}"

    def _apply_seed_config(self, flow_config: FlowConfig, flow_data: dict, params: dict, is_ids: bool):
        """Handle leaderboard seeding + seed checkpoint."""
        # Handle leaderboard seeding (explicit param or grid_source=leaderboard)
        use_leaderboard = params.get("seed_from_leaderboard") or params.get("grid_source") == "leaderboard"
        if use_leaderboard:
            flow_config.seed_from_leaderboard = True
            flow_config.seed_leaderboard_task_type = params.get("seed_leaderboard_task_type", "ids" if is_ids else "lm")
            flow_config.seed_leaderboard_stage = params.get("seed_leaderboard_stage", "stage_0")
            flow_config.seed_leaderboard_metric = params.get("seed_leaderboard_metric", "f1_macro" if is_ids else "ce")
            flow_config.seed_leaderboard_count = params.get("seed_leaderboard_count", params.get("population_size", 150))
            self._log(f"Seeding from leaderboard: top {flow_config.seed_leaderboard_count} by {flow_config.seed_leaderboard_metric}")

        # Handle seed checkpoint
        seed_checkpoint_id = flow_data.get("seed_checkpoint_id")
        if seed_checkpoint_id:
            try:
                ckpt = self.client.get_checkpoint(seed_checkpoint_id)
                if ckpt and ckpt.get("file_path"):
                    flow_config.seed_checkpoint_path = ckpt["file_path"]
                    self._log(f"Seeding from checkpoint: {ckpt.get('name')}")
            except Exception as e:
                self._log(f"Warning: Could not fetch seed checkpoint: {e}")

    def _build_flow(self, flow_data: dict, flow_id: int, flow_name: str, params: dict) -> Flow:
        """Assemble the Flow: evaluators + experiment configs + FlowConfig."""
        # Fetch experiments from API (normalized design: experiments stored in DB table)
        experiments = self.client.list_flow_experiments(flow_id)
        self._log(f"Fetched {len(experiments)} experiments from API")

        # Get context size from params or use default
        context_size = params.get("context_size", self.context_size)

        # Setup checkpoint directory
        safe_name = flow_name.lower().replace(" ", "_").replace("/", "_")
        checkpoint_dir = self.checkpoint_base_dir / safe_name

        # Detect architecture type
        architecture_type = params.get("architecture_type", "tiered")
        is_bitwise = architecture_type == "bitwise"
        is_multi_stage = architecture_type == "multi_stage"
        is_ids = architecture_type == "ids"

        # Load data and create appropriate evaluator
        evaluator, ids_test_evaluator, ids_s1_evaluator, ids_s1_test_evaluator = (
            self._create_flow_evaluators(architecture_type, context_size, params)
        )

        # Build experiment configs
        exp_configs = self._build_experiment_configs(experiments, params, evaluator.vocab_size)

        # Create flow config
        flow_config = self._build_base_flow_config(
            flow_data, flow_name, params, exp_configs, context_size, architecture_type,
        )
        if is_bitwise:
            self._apply_bitwise_config(flow_config, params)
        if is_multi_stage:
            self._apply_multistage_config(flow_config, params)
        if is_ids:
            self._apply_ids_config(flow_config, params)
        self._apply_seed_config(flow_config, flow_data, params, is_ids)

        # Create full evaluator for validation (trains on ALL train data, evals on validation set)
        full_evaluator = None
        if is_ids:
            full_evaluator = ids_test_evaluator
        elif is_bitwise and self._validation_tokens:
            full_evaluator = self._create_full_evaluator(context_size, params)

        # Create flow (pass existing flow_id to avoid duplication)
        return Flow(
            config=flow_config,
            evaluator=evaluator,
            full_evaluator=full_evaluator,
            logger=self._log,
            checkpoint_dir=checkpoint_dir,
            dashboard_client=self.client,
            flow_id=flow_id,
            tracker=self.tracker,
            shutdown_check=self.should_stop,  # Pass shutdown check for graceful stop
            s1_evaluator=ids_s1_evaluator,
            s1_full_evaluator=ids_s1_test_evaluator,
        )

    def _handle_flow_exception(self, flow_id: int, e: Exception):
        """Classify a flow exception: deleted / paused / shutdown / real failure."""
        error_msg = str(e).lower()
        is_pause = self._pause_current_flow
        is_shutdown = self._stop_current_flow or "shutdown" in error_msg or "stopped" in error_msg

        # Check if flow still exists (might have been deleted). GUARDED
        # (dashboard-review 2.7): the dashboard being down is often the very
        # fault that killed the flow — an unguarded raise HERE skipped the
        # whole requeue/pause/fail classification and wedged the flow in
        # 'running'. With the dashboard down there is nothing useful we can
        # tell it anyway; leave the flow for stale-heartbeat recovery, which
        # RE-QUEUES it with data preserved.
        try:
            flow_exists = self.client.get_flow(flow_id) is not None
        except Exception as exc:
            self._log(
                f"Dashboard unreachable while handling flow {flow_id} exception ({exc}); "
                f"leaving flow for stale-requeue recovery (original error: {e})"
            )
            return

        if not flow_exists:
            # Flow was deleted - just clean up and move on
            self._log(f"Flow {flow_id} was deleted, cleaning up")
        elif is_pause:
            # Pause requested via dashboard — mark flow paused (NOT cancelled),
            # leaving the per-gen checkpoint on disk so resume can pick up.
            # Worker proceeds to the next queued flow per spec.
            self._log(f"Flow {flow_id} paused gracefully at end of generation")
            try:
                self.client.update_flow(flow_id, status="paused")
            except Exception as exc:
                self._log(f"Warning: Could not mark flow {flow_id} as paused: {exc}")
        elif is_shutdown:
            # Check if user already cancelled this flow via dashboard
            current_flow = self.client.get_flow(flow_id)
            flow_status = current_flow.get("status", "") if current_flow else ""
            if flow_status == "cancelled":
                self._log(f"Flow was cancelled by user, keeping cancelled status")
            else:
                # Graceful shutdown (e.g., worker restart) - re-queue for resume
                self._log(f"Flow stopped due to shutdown, re-queuing for resume")
                try:
                    self.client.requeue_flow(flow_id)
                except Exception:
                    self._log(f"Warning: Could not re-queue flow {flow_id}")
        else:
            # Actual error - mark as failed
            self._log(f"Flow failed: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.client.flow_failed(flow_id, str(e))
            except Exception:
                pass

    def _restore_env_var(self, env_var: str, prev_attr: str):
        """Restore one env var to its pre-flow state (None means it wasn't set)."""
        prev = getattr(self, prev_attr, None)
        if prev is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = prev

    def _cleanup_after_flow(self):
        """Stop heartbeat, reset per-flow flags, restore env overrides, close log."""
        self._stop_heartbeat()
        self.current_flow_id = None
        self._stop_current_flow = False  # Reset for next flow
        self._pause_current_flow = False  # Reset pause flag for next flow
        # Restore env vars to their prior state (None means it wasn't set
        # before this flow). Ensures next flow sees the pre-flow state.
        self._restore_env_var("WNN_ORDER_INDEPENDENT_TRAIN", "_prev_oi_env")
        self._restore_env_var("WNN_HYBRID_SPEED_RATIO", "_prev_hsr_env")
        self._restore_env_var("WNN_GRID_SEARCH_PARALLEL", "_prev_grid_parallel_env")
        self._restore_env_var("WNN_BATCH_SIZE", "_prev_batch_size_env")
        self._restore_env_var("WNN_NO_METAL", "_prev_no_metal_env")
        self._restore_env_var("RAYON_NUM_THREADS", "_prev_num_threads_env")
        self._close_log_file()

    def _create_evaluator(self, context_size: int, seed: Optional[int] = None, neuron_sample_rate: float = 0.25):
        """Create the cached evaluator using pre-cached data."""
        self._ensure_lm_cache()
        self._log(f"Creating evaluator (context_size={context_size}, sample_rate={neuron_sample_rate})...")

        from wnn.ram.architecture.tiered_evaluator import TieredEvaluator

        evaluator = TieredEvaluator(
            train_tokens=self._train_tokens,
            eval_tokens=self._test_tokens,
            vocab_size=self._vocab_size,
            context_size=context_size,
            cluster_order=self._cluster_order,
            num_parts=3,
            num_negatives=5,
            empty_value=0.0,
            seed=seed or int(time.time() * 1000) % (2**32),
            use_hybrid=True,
            log_path=str(self._log_file) if self._log_file else None,
            neuron_sample_rate=neuron_sample_rate,
        )

        self._log(f"  Vocab: {evaluator.vocab_size:,}, Context: {context_size}")
        return evaluator

    def _create_controller_evaluator(self, params: dict):
        """Create a ControllerEvaluator for an architecture_type='controller' flow.

        Lazy import (control → ram only at runtime, never at worker startup) so a
        controller flow doesn't pull the drone-sim deps into IDS flows. The flow's
        controller_* params map to the ControllerSpec + episode plan via flow_adapter.
        """
        from wnn.control.flow_adapter import build_controller_evaluator
        self._log("Creating ControllerEvaluator (architecture_type='controller')...")
        return build_controller_evaluator(params)

    def _create_bitwise_evaluator(self, context_size: int, params: dict):
        """Create a BitwiseEvaluator using pre-cached data.

        Uses train split for training, test split for fitness scoring (GA/TS).
        Validation split is reserved for full_evaluator (final validation).
        """
        self._ensure_lm_cache()
        self._log(f"Creating BitwiseEvaluator (context_size={context_size})...")

        from wnn.ram.architecture.bitwise_evaluator import BitwiseEvaluator

        # Map memory mode string to integer
        memory_mode_map = {
            "TERNARY": 0,
            "QUAD_BINARY": 1,
            "QUAD_WEIGHTED": 2,
            "BINARY": 3,   # classical WiSARD 1-bit (granularity-ablation arm)
        }
        memory_mode = memory_mode_map.get(params.get("memory_mode", "QUAD_WEIGHTED"), 2)
        neuron_sample_rate = params.get("neuron_sample_rate", 0.25)
        train_parts = params.get("train_parts", 4)
        eval_parts = params.get("eval_parts", 1)

        evaluator = BitwiseEvaluator(
            train_tokens=self._train_tokens,
            eval_tokens=self._test_tokens,
            vocab_size=self._vocab_size,
            context_size=context_size,
            num_parts=train_parts,
            num_eval_parts=eval_parts,
            memory_mode=memory_mode,
            neuron_sample_rate=neuron_sample_rate,
        )

        self._log(f"  BitwiseEvaluator: vocab={evaluator.vocab_size:,}, "
                   f"context={context_size}, mode={params.get('memory_mode', 'QUAD_WEIGHTED')}, "
                   f"sample_rate={neuron_sample_rate}, "
                   f"train_parts={train_parts}, eval_parts={eval_parts}")
        return evaluator

    def _create_multistage_evaluator(self, context_size: int, params: dict):
        """Create a MultiStageEvaluator using pre-cached data.

        Uses train split for training, test split for fitness scoring (GA/TS).
        """
        self._ensure_lm_cache()
        # Per-stage context sizes (e.g., [2, 4] for S0=2, S1=4)
        stage_context_size = params.get("stage_context_size")
        effective_ctx = stage_context_size if stage_context_size else context_size
        self._log(f"Creating MultiStageEvaluator (context_size={effective_ctx})...")

        from wnn.ram.architecture.multistage_evaluator import MultiStageEvaluator

        memory_mode_map = {"TERNARY": 0, "QUAD_BINARY": 1, "QUAD_WEIGHTED": 2, "BINARY": 3}
        memory_mode = memory_mode_map.get(params.get("memory_mode", "QUAD_WEIGHTED"), 2)
        neuron_sample_rate = params.get("neuron_sample_rate", 0.25)
        train_parts = params.get("train_parts", 4)
        eval_parts = params.get("eval_parts", 1)

        num_stages = params.get("num_stages", 2)
        stage_k = params.get("stage_k")
        stage_cluster_type = params.get("stage_cluster_type")

        # Handle custom clustering strategies (semantic, semantic_bitwise, etc.)
        custom_cluster_of = None
        strategy_types = {"semantic", "semantic_bitwise"}
        if stage_cluster_type and any(t in strategy_types for t in stage_cluster_type):
            from wnn.ram.architecture.token_clustering import get_clustering_strategy
            from wnn.ram.architecture.multistage_evaluator import compute_default_k

            # Compute K values first (strategies need K)
            if stage_k is None:
                stage_k = compute_default_k(num_stages, stage_cluster_type, self._vocab_size)

            cache_dir = str(Path("cache"))
            rust_types = list(stage_cluster_type)

            for i, stype in enumerate(stage_cluster_type):
                if stype not in strategy_types:
                    continue
                strategy = get_clustering_strategy(stype)
                k = stage_k[i]
                self._log(f"  Computing {stype} clustering for stage {i} (k={k})...")
                clustering = strategy.create(self._vocab_size, k, cache_dir)
                custom_cluster_of = clustering.cluster_of
                rust_types[i] = strategy.rust_stage_type
                self._log(f"  {stype} clustering: k={k}, max_group={clustering.max_cluster_size}, "
                           f"min_group={min(len(g) for g in clustering.cluster_tokens)}")

            stage_cluster_type = rust_types

        # Parse stage_mode from params (could be string or list of ints)
        raw_stage_mode = params.get("stage_mode")
        if raw_stage_mode is not None:
            from wnn.ram.experiments.experiment import StageMode
            if isinstance(raw_stage_mode, str):
                mode_map = {"input_concat": StageMode.INPUT_CONCAT, "selector": StageMode.SELECTOR}
                stage_mode = [mode_map.get(raw_stage_mode, StageMode.INPUT_CONCAT)]
            elif isinstance(raw_stage_mode, list):
                stage_mode = raw_stage_mode
            else:
                stage_mode = None
        else:
            stage_mode = None

        reweight_rounds = params.get("reweight_rounds", 0)
        reweight_max_boost = params.get("reweight_max_boost", 4)

        evaluator = MultiStageEvaluator(
            train_tokens=self._train_tokens,
            eval_tokens=self._test_tokens,
            vocab_size=self._vocab_size,
            context_size=effective_ctx,
            num_stages=num_stages,
            stage_k=stage_k,
            stage_cluster_type=stage_cluster_type,
            stage_mode=stage_mode,
            target_stage=0,  # Flow.run() switches this at stage boundaries
            num_parts=train_parts,
            num_eval_parts=eval_parts,
            memory_mode=memory_mode,
            neuron_sample_rate=neuron_sample_rate,
            log_path=str(self._log_file) if self._log_file else None,
            custom_cluster_of=custom_cluster_of,
            reweight_rounds=reweight_rounds,
            reweight_max_boost=reweight_max_boost,
        )

        self._log(f"  MultiStageEvaluator: stages={num_stages}, k={stage_k}, "
                   f"types={stage_cluster_type}, mode={params.get('memory_mode', 'QUAD_WEIGHTED')}, "
                   f"sample_rate={neuron_sample_rate}")

        return evaluator

    def _create_full_evaluator(self, context_size: int, params: dict):
        """Create a full BitwiseEvaluator for validation summaries.

        Used ONLY for the 1-3 genome summary at end of each phase.
        Trains on train set, evaluates on validation set (held-out).
        GA/TS never touches this evaluator.
        """
        self._ensure_lm_cache()
        from wnn.ram.architecture.bitwise_evaluator import BitwiseEvaluator

        memory_mode_map = {"TERNARY": 0, "QUAD_BINARY": 1, "QUAD_WEIGHTED": 2, "BINARY": 3}
        memory_mode = memory_mode_map.get(params.get("memory_mode", "QUAD_WEIGHTED"), 2)
        neuron_sample_rate = params.get("neuron_sample_rate", 0.25)
        full_train_parts = params.get("full_train_parts", 1)
        full_eval_parts = params.get("full_eval_parts", 1)

        full_eval = BitwiseEvaluator(
            train_tokens=self._train_tokens,
            eval_tokens=self._validation_tokens,
            vocab_size=self._vocab_size,
            context_size=context_size,
            num_parts=full_train_parts,
            num_eval_parts=full_eval_parts,
            memory_mode=memory_mode,
            neuron_sample_rate=neuron_sample_rate,
        )

        self._log(f"  Full evaluator: train={len(self._train_tokens):,} tokens ({full_train_parts} parts), "
                   f"validation={len(self._validation_tokens):,} tokens ({full_eval_parts} parts)")
        return full_eval

    def _create_ids_evaluators(self, params: dict):
        """Create IDS evaluators: optimizer (131K/44K) + test (175K/82K).

        Returns (optimizer_evaluator, test_evaluator) for flat classification,
        or (s0_opt, s0_test, s1_opt, s1_test) for hierarchical classification.
        """
        import numpy as np
        from wnn.ids import load_unsw_nb15, split_train_validation
        from wnn.ids.dataset import IDSDataset
        from wnn.ram.architecture.ids_evaluator import IDSEvaluator

        classification = params.get("ids_classification", "binary")
        n_bits = params.get("ids_n_bits", 8)
        val_fraction = params.get("ids_val_fraction", 0.25)
        num_parts = params.get("ids_num_parts", 5)
        k_folds = params.get("ids_k_folds", 5)
        split = params.get("ids_split", "standard")
        seed = params.get("seed", 42)
        neuron_sample_rate = params.get("neuron_sample_rate", 0.25)
        balance_classes = params.get("balance_classes", False)
        single_cluster = params.get("ids_single_cluster", False)
        undersample_majority = params.get("undersample_majority", False)
        flip_labels = params.get("flip_labels", False)
        class_weight_multiplier = params.get("class_weight_multiplier", 1.0)
        # Memory-mode string → int. Threaded into IDSEvaluator → IDSCacheWrapper
        # .set_memory_mode() → EvalSettings.memory_mode. Before 14/07/2026 this
        # was NEVER threaded for IDS (the cache defaulted to QUAD), so TERNARY/
        # BINARY/QSR/PLN flows silently trained+scored as QUAD — only empty_value
        # differed, which the QUAD branch ignores. QSR (stochastic QUAD) + PLN
        # (stochastic TERNARY) added 14/07/2026.
        memory_mode_map = {"TERNARY": 0, "QUAD_BINARY": 1, "QUAD_WEIGHTED": 2, "BINARY": 3, "QSR": 4, "PLN": 5}
        mode_str = params.get("memory_mode", "QUAD_WEIGHTED")
        memory_mode = memory_mode_map.get(mode_str, 2)
        # TERNARY / PLN use the 3-state cells; their u-state (EMPTY) decodes to
        # 0.5 (TERNARY deterministic; PLN = the fair coin's expected value) —
        # otherwise EMPTY collapses to FALSE and it degenerates to a WiSARD.
        # QUAD-family ignores empty_value (WEAK_FALSE=0.25 baseline), BINARY is
        # 1-bit; so gate 0.5 on TERNARY/PLN only, every other mode stays 0.0.
        empty_value = 0.5 if mode_str in ("TERNARY", "PLN") else 0.0
        # QSR/PLN stochastic-coin seed: the flow's seed, so an n-run cohort varies
        # run-to-run while each run stays reproducible (parity). Det. modes ignore it.
        run_seed = int(seed)
        feature_selection = params.get("ids_feature_selection", "all")
        rest_bits = params.get("ids_rest_bits", None)
        auto_max_bits = params.get("ids_auto_max_bits", 32)
        kfold_per_gen = params.get("ids_kfold_per_gen", 1)
        ids_raw = params.get("ids_raw", False)
        dataset_name = params.get("ids_dataset", "unsw-nb15")
        # Phase 4: opt-in disk-backed encoded matrix via np.memmap. Default
        # "memory" (InMemoryEncoded). Set "memmap" via the ids_encoded_storage
        # flow param for 96b × all-46 × 46M and similar tight-RAM configs.
        encoded_storage = params.get("ids_encoded_storage", "memory")
        encoded_storage_dir = params.get("ids_encoded_storage_dir", None)
        # Phase F: opt-in HuggingFace streaming load. Default False (full HF
        # download → in-RAM/memmap). Set "ids_streaming": true on the flow
        # to bypass full materialization — encoder fits via t-digest over
        # streamed chunks, X_train/X_test are StreamingEncoded that re-fetch
        # on each iter_chunks() call. Only ciciot2023* datasets support
        # streaming in v1; other datasets raise NotImplementedError. K-fold
        # CV is not supported in streaming v1 (k_folds must be 1).
        ids_streaming = params.get("ids_streaming", False)
        streaming_chunk_size = params.get("ids_streaming_chunk_size", 1_000_000)
        # Phase F10: warm the OS page cache for MemmapEncoded X_train/X_test
        # at IDSEvaluator construction. "touch" (default) reads every page
        # once (~1-3s for typical IDS-sized memmaps) to amortize cold-cache
        # faults across all subsequent K-fold reads. "willneed" is a
        # best-effort async hint. "none" disables.
        memmap_prefetch = params.get("ids_memmap_prefetch", "touch")
        # Default to "single_bit" when data is raw — i.e., either ids_raw=True or
        # the dataset is one of the raw-by-construction variants. Otherwise default
        # to "none" for back-compat with pre-Phase-C flows (incl. r98).
        raw_by_dataset = dataset_name in ("ciciot2023_canonical", "ciciot2023_neto_full", "ciciot2023_neto_subsample")
        is_raw_data = ids_raw or raw_by_dataset
        ids_invalid_encoding = params.get("ids_invalid_encoding",
                                          "single_bit" if is_raw_data else "none")

        raw_label = " RAW" if is_raw_data else ""
        streaming_label = " (streaming)" if ids_streaming else ""
        self._log(f"Loading {dataset_name}{raw_label}{streaming_label} dataset (classification={classification}, split={split}, feature_selection={feature_selection}, invalid_encoding={ids_invalid_encoding}, encoded_storage={encoded_storage})...")

        # Phase F: streaming mode constraints (v1)
        # F7 lifted the K-fold restriction: IDSEvaluator now auto-detects
        # whether to materialize the streaming source into a memmap (small
        # datasets) or run streaming K-fold via re-stream per fold (large
        # datasets). Threshold default 8 GB packed; configurable per flow.
        if ids_streaming:
            if not dataset_name.startswith("ciciot2023"):
                raise NotImplementedError(
                    f"ids_streaming=True only supports ciciot2023* datasets in v1 "
                    f"(got dataset_name={dataset_name!r}). cicids2017 and unsw-nb15 "
                    f"streaming variants are tracked as follow-up work."
                )
            if encoded_storage != "memory":
                raise NotImplementedError(
                    f"ids_streaming=True is incompatible with encoded_storage={encoded_storage!r}; "
                    f"streaming and memmap are alternative ways to bound memory."
                )
        # Unified entry point (template-method loaders). dataset_name is a registry
        # key: unsw-nb15 / cicids2017 / ciciot2023_neto_full / ciciot2023_neto_subsample.
        # The deprecated bencorn variants (ciciot2023 / _full / _canonical) are no longer
        # routed — load_ids_dataset raises a clear "unknown dataset" for them.
        from wnn.ids import load_ids_dataset
        full_dataset = load_ids_dataset(
            dataset_name, n_bits=n_bits, split=split, feature_selection=feature_selection,
            rest_bits=rest_bits, raw=ids_raw, invalid_encoding=ids_invalid_encoding,
            auto_max_bits=auto_max_bits, encoded_storage=encoded_storage,
            storage_dir=encoded_storage_dir, streaming=ids_streaming,
            streaming_chunk_size=streaming_chunk_size,
        )

        # Phase F10: attach the prefetch mode to the dataset so IDSEvaluator
        # picks it up. Using attribute-injection (rather than a dataclass
        # field) keeps the IDSDataset shape stable for downstream consumers
        # that don't know about prefetch.
        try:
            full_dataset.memmap_prefetch_mode = memmap_prefetch
        except AttributeError:
            pass  # frozen dataclass — fall back to default in IDSEvaluator

        if classification == "hierarchical":
            return self._create_hierarchical_ids_evaluators(
                full_dataset, params, val_fraction, num_parts, k_folds, seed,
            )

        y_train = full_dataset.y_train_binary if classification == "binary" else full_dataset.y_train_multi
        y_test = full_dataset.y_test_binary if classification == "binary" else full_dataset.y_test_multi
        num_classes = 2 if classification == "binary" else len(full_dataset.category_names)

        self._log(f"  Train: {len(y_train):,} examples, Test: {len(y_test):,} examples, "
                   f"Classes: {num_classes}, Features: {full_dataset.X_train.shape[1]}")

        # 80/20 split: K-fold on full 80% train, 20% val untouched until reporting
        # No holdout carving — K-fold's rotating eval prevents overfitting
        self._log("Using 80/20 split: K-fold on 80% train, 20% val for reporting only")

        # Optimizer: K-fold within 80% train (20% val not used during GA search)
        # X_test is set to the val set for structure but K-fold ignores it
        optimizer_dataset = full_dataset

        # Test evaluator: train on full 80%, eval on 20% val (final reported metrics)
        test_dataset = full_dataset

        # Optimizer evaluator: train (K-fold or subset rotation), test for fitness
        kfold_label = f", k_folds={k_folds}" if k_folds > 1 else ""
        kfpg_label = f", kfold_per_gen={kfold_per_gen}" if kfold_per_gen > 1 else ""
        self._log(f"Creating optimizer IDSEvaluator ({len(optimizer_dataset.X_train):,} train / "
                   f"{len(optimizer_dataset.X_test):,} eval, {num_parts} parts{kfold_label}{kfpg_label})...")
        balance_label = ", balanced" if balance_classes else ""
        sc_label = ", single-cluster" if single_cluster else ""
        self._log(f"  neuron_sample_rate={neuron_sample_rate}, balance_classes={balance_classes}{sc_label}")
        us_label = ", undersample" if undersample_majority else ""
        self._log(f"  undersample_majority={undersample_majority}{us_label}")
        optimizer_eval = IDSEvaluator(
            dataset=optimizer_dataset,
            classification=classification,
            num_parts=num_parts,
            k_folds=k_folds,
            kfold_per_gen=kfold_per_gen,
            empty_value=empty_value,
            memory_mode=memory_mode,
            run_seed=run_seed,
            neuron_sample_rate=neuron_sample_rate,
            balance_classes=balance_classes,
            single_cluster=single_cluster,
            undersample_majority=undersample_majority,
            flip_labels=flip_labels,
            class_weight_multiplier=class_weight_multiplier,
        )

        # Test evaluator: full train → validation (for final reporting).
        # M2: reuse the optimizer's Rust cache instead of building a second full
        # copy (~19 GB). optimizer_dataset and test_dataset are the SAME
        # full_dataset (same 80% train / 20% val), so full_train/full_eval are
        # identical; the test evaluator only reads those. `_cache` is None in
        # streaming mode → reuse_cache=None → test_eval builds its own (correct).
        shared_cache = getattr(optimizer_eval, "_cache", None)
        share_label = ", shared cache" if shared_cache is not None else ""
        self._log(f"Creating test IDSEvaluator ({len(test_dataset.X_train):,} train / "
                   f"{len(test_dataset.X_test):,} eval, 1 part{balance_label}{share_label})...")
        test_eval = IDSEvaluator(
            dataset=test_dataset,
            classification=classification,
            num_parts=1,
            empty_value=empty_value,
            memory_mode=memory_mode,
            run_seed=run_seed,
            neuron_sample_rate=neuron_sample_rate,
            balance_classes=balance_classes,
            single_cluster=single_cluster,
            undersample_majority=undersample_majority,
            flip_labels=flip_labels,
            class_weight_multiplier=class_weight_multiplier,
            reuse_cache=shared_cache,
        )

        # Set fitness weights for threshold optimization
        # When set, threshold sweep maximizes fitness instead of F1
        fw_ce = params.get("fitness_weight_ce", 0)
        fw_f1 = params.get("fitness_weight_f1", 0)
        fw_fpr = params.get("fitness_weight_fpr", 0)
        fw_acc = params.get("fitness_weight_acc", 0)
        if single_cluster and (fw_f1 > 0 or fw_fpr > 0 or fw_acc > 0):
            self._log(f"  Fitness threshold: CE={fw_ce}, F1={fw_f1}, FPR={fw_fpr}, Acc={fw_acc}")
            optimizer_eval.set_fitness_weights(fw_ce, fw_f1, fw_fpr, fw_acc)
            test_eval.set_fitness_weights(fw_ce, fw_f1, fw_fpr, fw_acc)

        return optimizer_eval, test_eval

    def _create_hierarchical_ids_evaluators(self, full_dataset, params, val_fraction, num_parts, k_folds, seed):
        """Create S0 (binary) + S1 (attack-only 9-class) evaluator pairs for hierarchical IDS.

        S0: Normal vs Attack (2 classes) — trained on all flows
        S1: 9 attack types — trained on attack flows only

        Returns (s0_opt, s0_test, s1_opt, s1_test).
        """
        from wnn.ids import split_train_validation, create_attack_only_dataset
        from wnn.ram.architecture.ids_evaluator import IDSEvaluator

        # ── S0: Binary (all flows) ──
        self._log("── S0: Binary (Normal vs Attack) ──")
        self._log(f"  Train: {len(full_dataset.y_train_binary):,}, Test: {len(full_dataset.y_test_binary):,}, Classes: 2")

        s0_train_val, s0_test_ds = split_train_validation(full_dataset, val_fraction=val_fraction, seed=seed)

        kfold_label = f", k_folds={k_folds}" if k_folds > 1 else ""
        self._log(f"  S0 optimizer: {len(s0_train_val.X_train):,} train / {len(s0_train_val.X_test):,} eval, "
                   f"{num_parts} parts{kfold_label}")
        neuron_sample_rate = params.get("neuron_sample_rate", 0.25)
        balance_classes = params.get("balance_classes", False)
        # Memory-mode threading (see _create_ids_evaluators for the full rationale).
        memory_mode_map = {"TERNARY": 0, "QUAD_BINARY": 1, "QUAD_WEIGHTED": 2, "BINARY": 3, "QSR": 4, "PLN": 5}
        mode_str = params.get("memory_mode", "QUAD_WEIGHTED")
        memory_mode = memory_mode_map.get(mode_str, 2)
        empty_value = 0.5 if mode_str in ("TERNARY", "PLN") else 0.0
        run_seed = int(seed)
        s0_opt = IDSEvaluator(dataset=s0_train_val, classification="binary", num_parts=num_parts, k_folds=k_folds,
                              empty_value=empty_value, memory_mode=memory_mode, run_seed=run_seed,
                              neuron_sample_rate=neuron_sample_rate, balance_classes=balance_classes)
        self._log(f"  S0 test: {len(s0_test_ds.X_train):,} train / {len(s0_test_ds.X_test):,} eval")
        s0_test = IDSEvaluator(dataset=s0_test_ds, classification="binary", num_parts=1,
                               empty_value=empty_value, memory_mode=memory_mode, run_seed=run_seed,
                               neuron_sample_rate=neuron_sample_rate, balance_classes=balance_classes)

        # ── S1: Attack types (attack flows only) ──
        self._log("── S1: Attack Classification (9 types) ──")
        attack_dataset = create_attack_only_dataset(full_dataset)
        num_attack_classes = len(attack_dataset.category_names)

        s1_train_val, s1_test_ds = split_train_validation(attack_dataset, val_fraction=val_fraction, seed=seed)

        self._log(f"  S1 optimizer: {len(s1_train_val.X_train):,} train / {len(s1_train_val.X_test):,} eval, "
                   f"{num_parts} parts{kfold_label}")
        s1_opt = IDSEvaluator(dataset=s1_train_val, classification="multi", num_parts=num_parts, k_folds=k_folds,
                              empty_value=empty_value, memory_mode=memory_mode, run_seed=run_seed,
                              neuron_sample_rate=neuron_sample_rate, balance_classes=balance_classes)
        self._log(f"  S1 test: {len(s1_test_ds.X_train):,} train / {len(s1_test_ds.X_test):,} eval, "
                   f"Classes: {num_attack_classes}")
        s1_test = IDSEvaluator(dataset=s1_test_ds, classification="multi", num_parts=1,
                               empty_value=empty_value, memory_mode=memory_mode, run_seed=run_seed,
                               neuron_sample_rate=neuron_sample_rate, balance_classes=balance_classes)

        return s0_opt, s0_test, s1_opt, s1_test

    def _build_experiment_configs(
        self,
        experiments: list[dict],
        params: dict,
        vocab_size: int,
    ) -> list[ExperimentConfig]:
        """Build experiment configs from API experiment data.

        Handles two formats:
        1. API format: {phase_type: "ga_neurons", ...} - experiments from DB
        2. Config format: {experiment_type: "ga", optimize_neurons: true, ...} - legacy
        """
        # Flow-level defaults from params
        tier_config = self._parse_tier_config(params.get("tier_config"))
        tier0_only = params.get("tier0_only", params.get("optimize_tier0_only", False))
        patience = params.get("patience", 3)
        # How often the early-stop patience check runs (gens between checks).
        # Surfaced from the UI; falls back to the ExperimentConfig default (10).
        check_interval = params.get("check_interval", 10)
        fitness_percentile = params.get("fitness_percentile")
        seed = params.get("seed")
        # Threshold params are in % from the UI (e.g. 0 = 0%, 1 = 1%)
        threshold_start_pct = params.get("threshold_start", 0.0)
        threshold_step_pct = params.get("threshold_step", 1.0)
        # Convert to fractions for internal use
        per_phase_delta = threshold_step_pct / 100.0  # 1% → 0.01
        min_accuracy_floor = params.get("min_accuracy_floor", 0.0)

        # Fitness calculator defaults from flow params
        default_fitness_type = self._parse_fitness_calculator(params.get("fitness_calculator"))
        default_weight_ce = params.get("fitness_weight_ce", 1.0)
        default_weight_acc = params.get("fitness_weight_acc", 1.0)

        exp_configs = []
        for exp_data in experiments:
            # Merge per-experiment params into exp_data for uniform access
            if exp_data.get("params") and isinstance(exp_data["params"], dict):
                for k, v in exp_data["params"].items():
                    if k not in exp_data:  # Don't override top-level fields
                        exp_data[k] = v

            # Parse phase_type (API format) or use direct fields (config format)
            phase_type = exp_data.get("phase_type", "")
            if phase_type:
                # API format: phase_type like "ga_neurons", "ts_bits", "ga_connections", "grid_search",
                # "neurogenesis", "synaptogenesis", "axonogenesis"
                # Genesis phase strings → unified LAMARCKIAN + genesis_mode.
                genesis_modes = {"neurogenesis", "synaptogenesis", "axonogenesis"}
                genesis_mode = "neurogenesis"
                if phase_type == "grid_search":
                    experiment_type = ExperimentType.GRID_SEARCH
                    optimize_neurons = False
                    optimize_bits = False
                    optimize_connections = False
                elif phase_type == "lambda_sweep":
                    experiment_type = ExperimentType.LAMBDA_SWEEP
                    optimize_neurons = False
                    optimize_bits = False
                    optimize_connections = False
                elif phase_type in genesis_modes:
                    experiment_type = ExperimentType.LAMARCKIAN
                    genesis_mode = phase_type
                    optimize_neurons = False
                    optimize_bits = False
                    optimize_connections = False
                else:
                    experiment_type = ExperimentType.GA if phase_type.startswith("ga") else ExperimentType.TS
                    optimize_neurons = "neurons" in phase_type
                    optimize_bits = "bits" in phase_type
                    optimize_connections = "connections" in phase_type
            else:
                # Legacy config format
                raw_type = exp_data.get("experiment_type", "ga")
                genesis_modes = {"neurogenesis", "synaptogenesis", "axonogenesis"}
                genesis_mode = raw_type if raw_type in genesis_modes else "neurogenesis"
                type_map = {
                    "ga": ExperimentType.GA,
                    "ts": ExperimentType.TS,
                    "grid_search": ExperimentType.GRID_SEARCH,
                    "lambda_sweep": ExperimentType.LAMBDA_SWEEP,
                }
                if raw_type in genesis_modes:
                    experiment_type = ExperimentType.LAMARCKIAN
                else:
                    experiment_type = type_map.get(raw_type, ExperimentType.GA)
                optimize_neurons = exp_data.get("optimize_neurons", False)
                optimize_bits = exp_data.get("optimize_bits", False)
                optimize_connections = exp_data.get("optimize_connections", False)

            # Get experiment-specific settings or fall back to flow params
            exp_tier_config = self._parse_tier_config(exp_data.get("tier_config")) or tier_config
            exp_fitness_type = (
                self._parse_fitness_calculator(exp_data["fitness_calculator"])
                if exp_data.get("fitness_calculator")
                else default_fitness_type
            )
            exp_weight_ce = exp_data.get("fitness_weight_ce") or default_weight_ce
            exp_weight_acc = exp_data.get("fitness_weight_acc") or default_weight_acc
            exp_weight_f1 = exp_data.get("fitness_weight_f1") or params.get("fitness_weight_f1", params.get("ids_fitness_weight_f1", 0.0))
            exp_weight_fpr = exp_data.get("fitness_weight_fpr") or params.get("fitness_weight_fpr", params.get("ids_fitness_weight_fpr", 0.0))

            # Determine architecture_type and target_stage early (needed for grid selection)
            architecture_type = params.get("architecture_type", "tiered")
            ms_target_stage = 0
            exp_name = exp_data.get("name", "Unnamed")
            if exp_name.startswith("S") and ":" in exp_name[:4]:
                try:
                    ms_target_stage = int(exp_name[1:exp_name.index(":")])
                except (ValueError, IndexError):
                    pass

            # Grid search: select grid based on stage type (tiered vs bitwise vs selector)
            stage_cluster_types = params.get("stage_cluster_type", [])
            is_tiered_stage = (
                architecture_type == "multi_stage"
                and ms_target_stage < len(stage_cluster_types)
                and stage_cluster_types[ms_target_stage] == "tiered"
            )
            # Check if this stage uses SELECTOR mode (previous stage's mode == selector)
            raw_modes = params.get("stage_mode")
            is_selector_stage = False
            if architecture_type == "multi_stage" and ms_target_stage > 0 and raw_modes:
                prev_mode = raw_modes[ms_target_stage - 1] if isinstance(raw_modes, list) else raw_modes
                is_selector_stage = (prev_mode == "selector" or prev_mode == 0)

            # Per-stage bounds from dashboard params
            stage_min_bits_arr = params.get("stage_min_bits")
            stage_max_bits_arr = params.get("stage_max_bits")
            stage_min_neurons_arr = params.get("stage_min_neurons")
            stage_max_neurons_arr = params.get("stage_max_neurons")

            # Check for per-stage grid overrides from dashboard params
            stage_neurons_grid_arr = params.get("stage_neurons_grid")
            stage_bits_grid_arr = params.get("stage_bits_grid")
            has_stage_neurons_grid = (
                stage_neurons_grid_arr
                and ms_target_stage < len(stage_neurons_grid_arr)
                and stage_neurons_grid_arr[ms_target_stage]
            )
            has_stage_bits_grid = (
                stage_bits_grid_arr
                and ms_target_stage < len(stage_bits_grid_arr)
                and stage_bits_grid_arr[ms_target_stage]
            )

            if has_stage_neurons_grid or has_stage_bits_grid:
                # Per-stage grid overrides take precedence
                neurons_grid = stage_neurons_grid_arr[ms_target_stage] if has_stage_neurons_grid else None
                bits_grid = stage_bits_grid_arr[ms_target_stage] if has_stage_bits_grid else None
            elif is_selector_stage:
                # SELECTOR sub-models have ~225× less data — use smaller grid
                neurons_grid = [5, 10, 15]
                bits_grid = [5, 6, 7, 8, 9, 10]
            elif is_tiered_stage:
                neurons_grid = params.get("tiered_neurons_grid") or [20, 30, 40, 50]
                bits_grid = params.get("tiered_bits_grid") or [18, 19, 20, 21, 22, 23]
            else:
                neurons_grid = params.get("neurons_grid")
                bits_grid = params.get("bits_grid")
                # Per-stage bounds override for grid generation
                mn = (stage_min_neurons_arr[ms_target_stage]
                      if stage_min_neurons_arr and ms_target_stage < len(stage_min_neurons_arr)
                      else params.get("min_neurons", 5))
                mx = (stage_max_neurons_arr[ms_target_stage]
                      if stage_max_neurons_arr and ms_target_stage < len(stage_max_neurons_arr)
                      else params.get("max_neurons", 300))
                if not neurons_grid:
                    step = max(10, round((mx - mn) / 6 / 50) * 50) or 50
                    neurons_grid = [mn] + list(range(((mn // step) + 1) * step, mx, step)) + [mx]
                mb = (stage_min_bits_arr[ms_target_stage]
                      if stage_min_bits_arr and ms_target_stage < len(stage_min_bits_arr)
                      else params.get("min_bits", 4))
                xb = (stage_max_bits_arr[ms_target_stage]
                      if stage_max_bits_arr and ms_target_stage < len(stage_max_bits_arr)
                      else params.get("max_bits", 24))
                if not bits_grid:
                    step = max(2, round((xb - mb) / 6 / 2) * 2) or 4
                    bits_grid = [mb] + list(range(((mb // step) + 1) * step, xb, step)) + [xb]
            num_grid_configs = len(neurons_grid) * len(bits_grid)
            grid_top_k = params.get("grid_top_k", 15)  # Default: top-15 configs (balanced seeding)
            # Prefer the LIVE flow config param over the experiment's creation-time
            # snapshot: exp_data.population_size is copied from the flow config when the
            # experiment is created, so a later config PATCH (e.g. pop 50→40 to escape the
            # plateau-escalation OOM) only updates params — the stale exp_data value must
            # NOT shadow it. They are equal except after a PATCH, where params is the intent.
            pop_size = params.get("population_size") or exp_data.get("population_size") or 50

            is_adaptation = experiment_type in (
                ExperimentType.LAMARCKIAN,
                ExperimentType.NEUROGENESIS, ExperimentType.SYNAPTOGENESIS, ExperimentType.AXONOGENESIS,
            )
            if experiment_type == ExperimentType.GRID_SEARCH:
                max_iters = num_grid_configs  # One iteration per (neurons, bits) config
            elif is_adaptation:
                max_iters = exp_data.get("max_iterations") or params.get("adaptation_iterations", 50)
            else:
                max_iters = exp_data.get("max_iterations") or params.get("ga_generations", 250)

            # Determine cluster_type from architecture_type
            from wnn.ram.experiments.experiment import ClusterType
            if architecture_type == "multi_stage":
                cluster_type = ClusterType.MULTI_STAGE
            elif architecture_type == "bitwise":
                cluster_type = ClusterType.BITWISE
            else:
                cluster_type = ClusterType.TIERED

            exp_config = ExperimentConfig(
                name=exp_name,
                experiment_type=experiment_type,
                architecture_type=architecture_type,
                phase_type=phase_type,
                genesis_mode=genesis_mode,
                optimize_bits=optimize_bits,
                optimize_neurons=optimize_neurons,
                optimize_connections=optimize_connections,
                generations=max_iters,
                population_size=pop_size,
                iterations=exp_data.get("max_iterations") or (
                    params.get("adaptation_iterations", 50) if is_adaptation
                    else params.get("ts_iterations", 250)
                ),
                neighbors_per_iter=params.get("neighbors_per_iter", 50),
                patience=patience,
                # Magnitude-aware patience: default ON for worker flows since
                # 11/07/2026 (SP wave-1 restart) — F1↑/FPR↓ proportional
                # recovery via the shared check_magnitude_metrics core.
                magnitude_aware_patience=params.get("magnitude_aware_patience", True),
                # Random-search baseline (Review C): zero selection pressure,
                # same eval protocol/budget as the GA. Default OFF.
                random_search=params.get("random_search", False),
                check_interval=check_interval,
                tier_config=exp_tier_config,
                optimize_tier0_only=tier0_only,
                fitness_percentile=fitness_percentile,
                fitness_calculator_type=exp_fitness_type,
                fitness_weight_ce=exp_weight_ce,
                fitness_weight_acc=exp_weight_acc,
                fitness_weight_f1=exp_weight_f1,
                fitness_weight_fpr=exp_weight_fpr,
                min_accuracy_floor=min_accuracy_floor,
                threshold_start=threshold_start_pct / 100.0,  # Convert % to fraction
                threshold_delta=per_phase_delta,
                threshold_reference=max_iters,  # Full ramp within each phase
                seed=seed,
                cluster_type=cluster_type,
                # Stage-specific bounds: tiered/selector stages use their grid extremes,
                # per-stage arrays take precedence over global params
                bitwise_min_bits=(
                    min(bits_grid) if (is_tiered_stage or is_selector_stage)
                    else (stage_min_bits_arr[ms_target_stage] if stage_min_bits_arr and ms_target_stage < len(stage_min_bits_arr) else params.get("min_bits"))
                ),
                bitwise_max_bits=(
                    max(bits_grid) if (is_tiered_stage or is_selector_stage)
                    else (stage_max_bits_arr[ms_target_stage] if stage_max_bits_arr and ms_target_stage < len(stage_max_bits_arr) else params.get("max_bits"))
                ),
                bitwise_min_neurons=(
                    min(neurons_grid) if (is_tiered_stage or is_selector_stage)
                    else (stage_min_neurons_arr[ms_target_stage] if stage_min_neurons_arr and ms_target_stage < len(stage_min_neurons_arr) else params.get("min_neurons"))
                ),
                bitwise_max_neurons=(
                    max(neurons_grid) if (is_tiered_stage or is_selector_stage)
                    else (stage_max_neurons_arr[ms_target_stage] if stage_max_neurons_arr and ms_target_stage < len(stage_max_neurons_arr) else params.get("max_neurons"))
                ),
                # Grid search params
                neurons_grid=neurons_grid if experiment_type == ExperimentType.GRID_SEARCH else None,
                bits_grid=bits_grid if experiment_type == ExperimentType.GRID_SEARCH else None,
                grid_top_k=grid_top_k,
                grid_source=GridSource.LEADERBOARD if params.get("grid_source") == "leaderboard" or params.get("seed_from_leaderboard") else GridSource.RANDOM,
                # Multi-stage fields
                num_stages=params.get("num_stages", 2) if architecture_type == "multi_stage" else 1,
                stage_k=params.get("stage_k") if architecture_type == "multi_stage" else None,
                stage_cluster_type=params.get("stage_cluster_type") if architecture_type == "multi_stage" else None,
                stage_mode=self._parse_stage_mode(params.get("stage_mode")) if architecture_type == "multi_stage" else None,
                target_stage=ms_target_stage if architecture_type in ("multi_stage", "ids") else 0,
                # Lambda sweep params
                lambda_values=exp_data.get("lambda_values") if experiment_type == ExperimentType.LAMBDA_SWEEP else None,
                bigram_lambda_values=exp_data.get("bigram_lambda_values") if experiment_type == ExperimentType.LAMBDA_SWEEP else None,
                s0_checkpoint_id=exp_data.get("s0_checkpoint_id") if experiment_type == ExperimentType.LAMBDA_SWEEP else None,
                s1_checkpoint_id=exp_data.get("s1_checkpoint_id") if experiment_type == ExperimentType.LAMBDA_SWEEP else None,
                genome_type=exp_data.get("genome_type", "best_ce") if experiment_type == ExperimentType.LAMBDA_SWEEP else "best_ce",
                # Cluster crossover ratio
                cluster_crossover_ratio=float(params.get("cluster_crossover_ratio", 0.0)),
                # Pool-and-shuffle crossover ratio
                pool_shuffle_ratio=float(params.get("pool_shuffle_ratio", 0.0)),
                # Assortative mating ratio (NEAT-style)
                assortative_mating_ratio=float(params.get("assortative_mating_ratio", 0.85)),
            )
            exp_configs.append(exp_config)

        return exp_configs

    # =========================================================================
    # Gating Job Methods
    # =========================================================================

    def _get_next_gating_run(self) -> Optional[dict]:
        """Get the next pending gating run."""
        try:
            runs = self.client.get_pending_gating_runs()
            if runs:
                return runs[0]
        except Exception as e:
            self._log(f"Failed to fetch pending gating runs: {e}")
        return None

    def _execute_gating_job(self, gating_run: dict):
        """Execute gating analysis for a gating run."""
        from datetime import datetime
        import gzip
        import json

        gating_run_id = gating_run["id"]
        experiment_id = gating_run["experiment_id"]

        # Get experiment details
        experiment = self.client.get_experiment(experiment_id)
        if not experiment:
            self._log(f"Experiment {experiment_id} not found for gating run {gating_run_id}")
            return

        experiment_name = experiment.get("name", f"Experiment {experiment_id}")

        self._log(f"=" * 60)
        self._log(f"Starting gating analysis: {experiment_name} (Run ID: {gating_run_id})")
        self._log(f"=" * 60)

        try:
            # Update status to running
            self.client.update_gating_run_status(experiment_id, gating_run_id, "running")
            self._log("  Loading checkpoint...")

            # Find the latest checkpoint for this experiment
            checkpoints = self.client.list_checkpoints(experiment_id=experiment_id, limit=10)
            if not checkpoints:
                raise ValueError(f"No checkpoints found for experiment {experiment_id}")

            # Sort by created_at and get the latest
            checkpoints = sorted(checkpoints, key=lambda c: c.get("created_at", ""), reverse=True)
            checkpoint = checkpoints[0]
            checkpoint_path = checkpoint.get("file_path")

            if not checkpoint_path:
                raise ValueError(f"Checkpoint has no file_path")

            self._log(f"  Using checkpoint: {checkpoint.get('name')} ({checkpoint_path})")

            # Load checkpoint
            if checkpoint_path.endswith(".gz"):
                with gzip.open(checkpoint_path, "rt") as f:
                    ckpt_data = json.load(f)
            else:
                with open(checkpoint_path, "r") as f:
                    ckpt_data = json.load(f)

            # Extract best genomes from checkpoint
            best_genomes = self._extract_best_genomes(ckpt_data)
            self._log(f"  Found {len(best_genomes)} unique genomes to test")

            if not best_genomes:
                raise ValueError("No genomes found in checkpoint")

            # Create evaluator
            context_size = experiment.get("context_size", self.context_size)
            evaluator = self._create_evaluator(context_size)

            # Run gating analysis for each genome
            from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

            gating_results = []

            for genome_type, genome_data in best_genomes.items():
                self._log(f"  Analyzing {genome_type}...")

                # Convert to ClusterGenome
                genome = ClusterGenome(
                    bits_per_neuron=genome_data.get("bits_per_neuron", genome_data.get("bits_per_cluster", [])),
                    neurons_per_cluster=genome_data["neurons_per_cluster"],
                    connections=genome_data.get("connections"),
                )

                # Run gated evaluation
                result = evaluator.evaluate_single_full_gated(
                    genome=genome,
                    train_tokens=self._train_tokens,
                    neurons_per_gate=8,
                    bits_per_neuron=12,
                    threshold=0.5,
                    batch_size=256,
                    logger=self._log,
                )

                gating_results.append({
                    "genome_type": genome_type,
                    "ce": result["ce"],
                    "acc": result["acc"],
                    "gated_ce": result["gated_ce"],
                    "gated_acc": result["gated_acc"],
                    "gating_config": result["gating_config"],
                })

            # Update gating run with results (automatically sets status to completed)
            self.client.update_gating_run_results(
                experiment_id,
                gating_run_id,
                results=gating_results,
                genomes_tested=len(gating_results),
            )
            self._log(f"Gating analysis completed for experiment {experiment_id} (run {gating_run_id})")

        except Exception as e:
            self._log(f"Gating analysis failed: {e}")
            import traceback
            traceback.print_exc()

            # Update with failure
            try:
                self.client.update_gating_run_results(
                    experiment_id,
                    gating_run_id,
                    results=[],
                    genomes_tested=0,
                    error=str(e),
                )
            except Exception:
                pass

    def _extract_best_genomes(self, ckpt_data: dict) -> dict:
        """Extract unique best genomes from checkpoint data.

        Returns dict mapping genome_type to genome data, deduplicated.
        """
        genomes = {}

        # Checkpoint format: genome data is nested under phase_result
        phase_result = ckpt_data.get("phase_result", {})

        if "best_genome" in phase_result:
            genomes["best_fitness"] = phase_result["best_genome"]

        if "best_ce_genome" in phase_result:
            genomes["best_ce"] = phase_result["best_ce_genome"]

        if "best_acc_genome" in phase_result:
            genomes["best_acc"] = phase_result["best_acc_genome"]

        # If only one genome type found, use it for all
        if len(genomes) == 1:
            only_genome = list(genomes.values())[0]
            if "best_fitness" not in genomes:
                genomes["best_fitness"] = only_genome
            if "best_ce" not in genomes:
                genomes["best_ce"] = only_genome
            if "best_acc" not in genomes:
                genomes["best_acc"] = only_genome

        # Deduplicate by config hash
        seen_hashes = {}
        deduped = {}

        for genome_type, genome_data in genomes.items():
            # Compute simple hash from bits/neurons
            config_hash = str(genome_data.get("bits_per_neuron", genome_data.get("bits_per_cluster", []))) + str(genome_data.get("neurons_per_cluster", []))
            if config_hash not in seen_hashes:
                seen_hashes[config_hash] = genome_type
                deduped[genome_type] = genome_data
            else:
                self._log(f"    {genome_type} same as {seen_hashes[config_hash]}, skipping")

        return deduped

    def _parse_stage_mode(self, raw_mode) -> Optional[list[int]]:
        """Parse stage_mode from dashboard params into list of StageMode integers."""
        if raw_mode is None:
            return None
        from wnn.ram.experiments.experiment import StageMode
        mode_map = {
            "input_concat": StageMode.INPUT_CONCAT,
            "selector": StageMode.SELECTOR,
        }
        if isinstance(raw_mode, str):
            return [mode_map.get(raw_mode, StageMode.INPUT_CONCAT)]
        if isinstance(raw_mode, list):
            return [
                mode_map.get(m, StageMode.INPUT_CONCAT) if isinstance(m, str) else int(m)
                for m in raw_mode
            ]
        return [int(raw_mode)]

    def _parse_fitness_calculator(self, fitness_calculator: Optional[str]) -> FitnessCalculatorType:
        """Parse fitness calculator string to enum."""
        if not fitness_calculator:
            return FitnessCalculatorType.HARMONIC_RANK  # Default
        try:
            return FitnessCalculatorType[fitness_calculator.upper()]
        except KeyError:
            self._log(f"Warning: Unknown fitness calculator '{fitness_calculator}', using HARMONIC_RANK")
            return FitnessCalculatorType.HARMONIC_RANK

    def _parse_tier_config(self, tier_config) -> Optional[list[tuple]]:
        """Parse tier config from string or array format.

        Formats supported:
        - "100,15,20;400,10,12;rest,5,8"  (3 parts, optimize=True default)
        - "100,15,20,true;400,10,12,false;rest,5,8,false"  (4 parts)

        Returns list of (count, neurons, bits, optimize) tuples.
        """
        if not tier_config:
            return None

        if isinstance(tier_config, str):
            tiers = []
            for tier_str in tier_config.split(";"):
                parts = tier_str.strip().split(",")
                if len(parts) >= 3:
                    count = None if parts[0].strip().lower() == "rest" else int(parts[0])
                    neurons = int(parts[1])
                    bits = int(parts[2])
                    # 4th part: optimize flag (default True for backward compat)
                    optimize = parts[3].strip().lower() == "true" if len(parts) >= 4 else True
                    tiers.append((count, neurons, bits, optimize))
            return tiers if tiers else None

        elif isinstance(tier_config, list):
            # Already in array format - ensure 4 elements per tuple
            result = []
            for t in tier_config:
                if len(t) >= 4:
                    result.append((t[0], t[1], t[2], t[3]))
                else:
                    result.append((t[0], t[1], t[2], True))  # Default optimize=True
            return result

        return None

def main():
    parser = argparse.ArgumentParser(description="Flow worker - polls and executes queued flows")
    parser.add_argument("--url", default="https://localhost:3000", help="Dashboard URL")
    parser.add_argument("--poll-interval", type=int, default=10, help="Seconds between polls")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Base checkpoint directory")
    parser.add_argument("--context", type=int, default=4, help="Default context size")
    parser.add_argument("--cpu-budget", type=int, default=None,
                        help="Total CPU core budget for concurrent flows "
                             "(default: detected cores - 3)")

    # SSL/TLS options
    ssl_group = parser.add_mutually_exclusive_group()
    ssl_group.add_argument(
        "--ssl-cert",
        type=Path,
        help="Path to CA certificate for SSL verification (for self-signed certs)"
    )
    ssl_group.add_argument(
        "--no-ssl-verify",
        action="store_true",
        help="Disable SSL certificate verification (development only)"
    )

    args = parser.parse_args()

    # Determine SSL verification setting
    if args.no_ssl_verify:
        verify_ssl = False
    elif args.ssl_cert:
        verify_ssl = str(args.ssl_cert)
    else:
        verify_ssl = False  # Default: skip SSL verify (self-signed cert)

    worker = FlowWorker(
        dashboard_url=args.url,
        poll_interval=args.poll_interval,
        checkpoint_base_dir=args.checkpoint_dir,
        context_size=args.context,
        verify_ssl=verify_ssl,
        cpu_budget=args.cpu_budget,
    )

    worker.run()


if __name__ == "__main__":
    main()
