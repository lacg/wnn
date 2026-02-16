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
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from wnn.ram.fitness import FitnessCalculatorType
from wnn.ram.experiments.dashboard_client import DashboardClient, DashboardClientConfig
from wnn.ram.experiments.flow import Flow, FlowConfig
from wnn.ram.experiments.experiment import ExperimentConfig, ExperimentType
from wnn.ram.experiments.tracker import create_tracker, ExperimentTracker

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
    ):
        self.dashboard_url = dashboard_url
        self.poll_interval = poll_interval
        self.checkpoint_base_dir = checkpoint_base_dir
        self.context_size = context_size
        self.running = True
        self._stop_current_flow = False  # Flag to stop current flow but keep worker running
        self.current_flow_id: Optional[int] = None

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
        """
        if signum == signal.SIGINT:
            # Ctrl+C - stop everything
            self._log(f"Received SIGINT, stopping worker...")
            self.running = False
            self._stop_current_flow = True
        else:
            # SIGTERM - behavior depends on whether a flow is running
            if self.current_flow_id:
                # Flow is running - stop it gracefully, keep worker alive
                self._log(f"Received SIGTERM, stopping current flow...")
                self._stop_current_flow = True
                self._log(f"Flow {self.current_flow_id} will stop after current generation/iteration")
            else:
                # Worker is idle - stop entirely
                self._log(f"Received SIGTERM while idle, stopping worker...")
                self.running = False

    def should_stop(self) -> bool:
        """Check if shutdown has been requested. Used by flow/experiment/strategy.

        Checks both local flags and dashboard flow status for cancellation/deletion.
        This provides a fallback if SIGTERM delivery fails (e.g., missing PID).
        """
        # Check local flags first (fast path)
        if self._stop_current_flow or not self.running:
            return True

        # Periodically check dashboard flow status as fallback
        # This catches cancellation requests when PID wasn't registered
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
        """Mark stale running flows as failed.

        A flow is stale if it's marked as 'running' but hasn't received
        a heartbeat recently. This happens when a worker crashes or is killed.
        Does NOT auto-requeue — the user can restart manually from the dashboard.
        """
        try:
            flows = self.client.list_flows(status="running", limit=10)
            now = datetime.utcnow()

            for flow in flows:
                # Skip flows owned by THIS worker (currently running)
                if flow["id"] == self.current_flow_id:
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
                    self._log(f"Stale flow detected: {flow_name} (ID: {flow_id}) — marking as failed")
                    try:
                        self.client.flow_failed(flow_id, "Worker crashed or was killed (stale heartbeat)")
                    except Exception as e:
                        self._log(f"  Failed to mark flow {flow_id} as failed: {e}")
        except Exception:
            pass

    def _precache_data(self):
        """Pre-cache tokenizer and dataset at startup to avoid network issues during flow execution."""
        from collections import Counter

        from datasets import load_dataset
        from transformers import AutoTokenizer

        self._log("Pre-caching tokenizer and dataset...")

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

        self._log("Pre-caching complete!")

    def run(self):
        """Main worker loop."""
        self._log(f"Worker started, polling {self.dashboard_url} every {self.poll_interval}s")
        self._log(f"Checkpoints will be saved to {self.checkpoint_base_dir}")

        # Pre-cache data before entering the polling loop
        self._precache_data()

        while self.running:
            try:
                # First, recover any stale flows (from crashed workers)
                self._recover_stale_flows()

                # Check for queued flows (higher priority)
                flow_data = self._get_next_queued_flow()

                if flow_data:
                    self._execute_flow(flow_data)
                else:
                    # No flows, check for pending gating runs
                    gating_run = self._get_next_gating_run()
                    if gating_run:
                        self._execute_gating_job(gating_run)
                    else:
                        # No work, wait before polling again
                        time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self._log(f"Error in worker loop: {e}")
                time.sleep(self.poll_interval)

        self._log("Worker stopped")

    def _get_next_queued_flow(self) -> Optional[dict]:
        """Get the next queued flow from the dashboard."""
        try:
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
            import os
            self.client.register_flow_pid(flow_id, os.getpid())

            # Start heartbeat thread (allows detection of stale flows if worker crashes)
            self._start_heartbeat(flow_id)

            # Parse flow configuration
            config = flow_data.get("config", {})
            params = config.get("params", {})

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

            # Load data and create appropriate evaluator
            if is_bitwise:
                evaluator = self._create_bitwise_evaluator(context_size, params)
            else:
                evaluator = self._create_evaluator(context_size, params.get("seed"))

            # Build experiment configs
            exp_configs = self._build_experiment_configs(experiments, params, evaluator.vocab_size)

            # Parse tier config
            tier_config = self._parse_tier_config(params.get("tier_config"))

            # Parse fitness calculator settings
            fitness_calculator_type = self._parse_fitness_calculator(params.get("fitness_calculator"))
            fitness_weight_ce = params.get("fitness_weight_ce", 1.0)
            fitness_weight_acc = params.get("fitness_weight_acc", 1.0)

            # Create flow config
            flow_config = FlowConfig(
                name=flow_name,
                experiments=exp_configs,
                description=flow_data.get("description"),
                tier_config=tier_config,
                optimize_tier0_only=params.get("optimize_tier0_only", params.get("tier0_only", False)),
                context_size=context_size,
                patience=params.get("patience", 10),
                fitness_percentile=params.get("fitness_percentile"),
                fitness_calculator_type=fitness_calculator_type,
                fitness_weight_ce=fitness_weight_ce,
                fitness_weight_acc=fitness_weight_acc,
                seed=params.get("seed"),
                architecture_type=architecture_type,
            )

            # Add bitwise-specific config
            if is_bitwise:
                flow_config.num_clusters = params.get("num_clusters", 16)
                flow_config.memory_mode = params.get("memory_mode", "QUAD_WEIGHTED")
                flow_config.neuron_sample_rate = params.get("neuron_sample_rate", 0.25)
                flow_config.min_bits = params.get("min_bits", 10)
                flow_config.max_bits = params.get("max_bits", 24)
                flow_config.min_neurons = params.get("min_neurons", 10)
                flow_config.max_neurons = params.get("max_neurons", 300)
                flow_config.sparse_threshold = params.get("sparse_threshold")

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

            # Create full evaluator for validation (trains on ALL train data, evals on validation set)
            full_evaluator = None
            if is_bitwise and self._validation_tokens:
                full_evaluator = self._create_full_evaluator(context_size, params)

            # Create and run flow (pass existing flow_id to avoid duplication)
            flow = Flow(
                config=flow_config,
                evaluator=evaluator,
                full_evaluator=full_evaluator,
                logger=self._log,
                checkpoint_dir=checkpoint_dir,
                dashboard_client=self.client,
                flow_id=flow_id,
                tracker=self.tracker,
                shutdown_check=self.should_stop,  # Pass shutdown check for graceful stop
            )

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
            error_msg = str(e).lower()
            is_shutdown = self._stop_current_flow or "shutdown" in error_msg or "stopped" in error_msg

            # Check if flow still exists (might have been deleted)
            flow_exists = self.client.get_flow(flow_id) is not None

            if not flow_exists:
                # Flow was deleted - just clean up and move on
                self._log(f"Flow {flow_id} was deleted, cleaning up")
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

        finally:
            # Stop heartbeat thread
            self._stop_heartbeat()
            self.current_flow_id = None
            self._stop_current_flow = False  # Reset for next flow
            self._close_log_file()

    def _create_evaluator(self, context_size: int, seed: Optional[int] = None):
        """Create the cached evaluator using pre-cached data."""
        self._log(f"Creating evaluator (context_size={context_size})...")

        from wnn.ram.architecture.cached_evaluator import CachedEvaluator

        evaluator = CachedEvaluator(
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
            log_path=str(self._log_file) if self._log_file else None,  # Pass log path for Rust-side logging
        )

        self._log(f"  Vocab: {evaluator.vocab_size:,}, Context: {context_size}")
        return evaluator

    def _create_bitwise_evaluator(self, context_size: int, params: dict):
        """Create a BitwiseEvaluator using pre-cached data.

        Uses train split for training, test split for fitness scoring (GA/TS).
        Validation split is reserved for full_evaluator (final validation).
        """
        self._log(f"Creating BitwiseEvaluator (context_size={context_size})...")

        from wnn.ram.architecture.bitwise_evaluator import BitwiseEvaluator

        # Map memory mode string to integer
        memory_mode_map = {
            "TERNARY": 0,
            "QUAD_BINARY": 1,
            "QUAD_WEIGHTED": 2,
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

    def _create_full_evaluator(self, context_size: int, params: dict):
        """Create a full BitwiseEvaluator for validation summaries.

        Used ONLY for the 1-3 genome summary at end of each phase.
        Trains on train set, evaluates on validation set (held-out).
        GA/TS never touches this evaluator.
        """
        from wnn.ram.architecture.bitwise_evaluator import BitwiseEvaluator

        memory_mode_map = {"TERNARY": 0, "QUAD_BINARY": 1, "QUAD_WEIGHTED": 2}
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
        patience = params.get("patience", 10)
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
            # Parse phase_type (API format) or use direct fields (config format)
            phase_type = exp_data.get("phase_type", "")
            if phase_type:
                # API format: phase_type like "ga_neurons", "ts_bits", "ga_connections", "grid_search",
                # "neurogenesis", "synaptogenesis", "axonogenesis"
                adaptation_types = {
                    "neurogenesis": ExperimentType.NEUROGENESIS,
                    "synaptogenesis": ExperimentType.SYNAPTOGENESIS,
                    "axonogenesis": ExperimentType.AXONOGENESIS,
                }
                if phase_type == "grid_search":
                    experiment_type = ExperimentType.GRID_SEARCH
                    optimize_neurons = False
                    optimize_bits = False
                    optimize_connections = False
                elif phase_type in adaptation_types:
                    experiment_type = adaptation_types[phase_type]
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
                experiment_type = ExperimentType.GA if raw_type == "ga" else ExperimentType.TS
                optimize_neurons = exp_data.get("optimize_neurons", False)
                optimize_bits = exp_data.get("optimize_bits", False)
                optimize_connections = exp_data.get("optimize_connections", False)

            # Get experiment-specific settings or fall back to flow params
            exp_tier_config = self._parse_tier_config(exp_data.get("tier_config")) or tier_config
            exp_fitness_type = self._parse_fitness_calculator(exp_data.get("fitness_calculator")) or default_fitness_type
            exp_weight_ce = exp_data.get("fitness_weight_ce") or default_weight_ce
            exp_weight_acc = exp_data.get("fitness_weight_acc") or default_weight_acc

            # Grid search: set grid params and correct max_iterations
            neurons_grid = params.get("neurons_grid", [50, 100, 150, 200])
            bits_grid = params.get("bits_grid", [14, 16, 18, 20])
            num_grid_configs = len(neurons_grid) * len(bits_grid)
            grid_top_k = params.get("grid_top_k", num_grid_configs)  # Default: use all configs
            pop_size = exp_data.get("population_size") or params.get("population_size", 50)

            is_adaptation = experiment_type in (
                ExperimentType.NEUROGENESIS, ExperimentType.SYNAPTOGENESIS, ExperimentType.AXONOGENESIS,
            )
            if experiment_type == ExperimentType.GRID_SEARCH:
                max_iters = 1  # Grid search is a single step
            elif is_adaptation:
                max_iters = exp_data.get("max_iterations") or params.get("adaptation_iterations", 50)
            else:
                max_iters = exp_data.get("max_iterations") or params.get("ga_generations", 250)

            # Determine cluster_type from architecture_type
            architecture_type = params.get("architecture_type", "tiered")
            from wnn.ram.experiments.experiment import ClusterType
            cluster_type = ClusterType.BITWISE if architecture_type == "bitwise" else ClusterType.TIERED

            exp_config = ExperimentConfig(
                name=exp_data.get("name", "Unnamed"),
                experiment_type=experiment_type,
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
                tier_config=exp_tier_config,
                optimize_tier0_only=tier0_only,
                fitness_percentile=fitness_percentile,
                fitness_calculator_type=exp_fitness_type,
                fitness_weight_ce=exp_weight_ce,
                fitness_weight_acc=exp_weight_acc,
                min_accuracy_floor=min_accuracy_floor,
                threshold_start=threshold_start_pct / 100.0,  # Convert % to fraction
                threshold_delta=per_phase_delta,
                threshold_reference=max_iters,  # Full ramp within each phase
                seed=seed,
                cluster_type=cluster_type,
                # Bitwise-specific bounds from flow params
                bitwise_min_bits=params.get("min_bits"),
                bitwise_max_bits=params.get("max_bits"),
                bitwise_min_neurons=params.get("min_neurons"),
                bitwise_max_neurons=params.get("max_neurons"),
                # Grid search params
                neurons_grid=neurons_grid if experiment_type == ExperimentType.GRID_SEARCH else None,
                bits_grid=bits_grid if experiment_type == ExperimentType.GRID_SEARCH else None,
                grid_top_k=grid_top_k,
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

    def _parse_fitness_calculator(self, fitness_calculator: Optional[str]) -> FitnessCalculatorType:
        """Parse fitness calculator string to enum."""
        if not fitness_calculator:
            return FitnessCalculatorType.NORMALIZED  # Default
        try:
            return FitnessCalculatorType[fitness_calculator.upper()]
        except KeyError:
            self._log(f"Warning: Unknown fitness calculator '{fitness_calculator}', using NORMALIZED")
            return FitnessCalculatorType.NORMALIZED

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
    )

    worker.run()


if __name__ == "__main__":
    main()
