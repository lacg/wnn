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
from wnn.ram.experiments.experiment import ExperimentConfig, ExperimentType, GridSource
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
            is_multi_stage = architecture_type == "multi_stage"
            is_ids = architecture_type == "ids"

            # Load data and create appropriate evaluator
            ids_s1_evaluator = None
            ids_s1_test_evaluator = None
            if is_ids:
                ids_result = self._create_ids_evaluators(params)
                if len(ids_result) == 4:
                    # Hierarchical: (s0_opt, s0_test, s1_opt, s1_test)
                    evaluator, ids_test_evaluator, ids_s1_evaluator, ids_s1_test_evaluator = ids_result
                else:
                    evaluator, ids_test_evaluator = ids_result
            elif is_multi_stage:
                evaluator = self._create_multistage_evaluator(context_size, params)
            elif is_bitwise:
                evaluator = self._create_bitwise_evaluator(context_size, params)
            else:
                evaluator = self._create_evaluator(context_size, params.get("seed"), params.get("neuron_sample_rate", 0.25))

            # Build experiment configs
            exp_configs = self._build_experiment_configs(experiments, params, evaluator.vocab_size)

            # Parse tier config
            tier_config = self._parse_tier_config(params.get("tier_config"))

            # Parse fitness calculator settings
            fitness_calculator_type = self._parse_fitness_calculator(params.get("fitness_calculator"))
            fitness_weight_ce = params.get("fitness_weight_ce", 1.0)
            fitness_weight_acc = params.get("fitness_weight_acc", 1.0)
            fitness_weight_f1 = params.get("fitness_weight_f1", params.get("ids_fitness_weight_f1", 0.0))
            fitness_weight_fpr = params.get("fitness_weight_fpr", params.get("ids_fitness_weight_fpr", 0.0))

            # Create flow config
            flow_config = FlowConfig(
                name=flow_name,
                experiments=exp_configs,
                description=flow_data.get("description"),
                tier_config=tier_config,
                optimize_tier0_only=params.get("optimize_tier0_only", params.get("tier0_only", False)),
                context_size=context_size,
                patience=params.get("patience", 3),
                fitness_percentile=params.get("fitness_percentile"),
                fitness_calculator_type=fitness_calculator_type,
                fitness_weight_ce=fitness_weight_ce,
                fitness_weight_acc=fitness_weight_acc,
                fitness_weight_f1=fitness_weight_f1,
                fitness_weight_fpr=fitness_weight_fpr,
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

            # Add multi-stage config
            if is_multi_stage:
                flow_config.num_stages = params.get("num_stages", 2)
                flow_config.stage_k = params.get("stage_k")
                flow_config.stage_cluster_type = params.get("stage_cluster_type")
                # Parse stage_mode
                raw_mode = params.get("stage_mode")
                if isinstance(raw_mode, str):
                    from wnn.ram.experiments.experiment import StageMode
                    mode_map = {
                        "input_concat": StageMode.INPUT_CONCAT,
                        "selector": StageMode.SELECTOR,
                    }
                    flow_config.stage_mode = [mode_map.get(raw_mode, StageMode.INPUT_CONCAT)]
                elif isinstance(raw_mode, list):
                    from wnn.ram.experiments.experiment import StageMode
                    mode_map = {
                        "input_concat": StageMode.INPUT_CONCAT,
                        "selector": StageMode.SELECTOR,
                    }
                    flow_config.stage_mode = [
                        mode_map.get(m, StageMode.INPUT_CONCAT) if isinstance(m, str) else m
                        for m in raw_mode
                    ]
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

            # Add IDS-specific config
            if is_ids:
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

            # Create full evaluator for validation (trains on ALL train data, evals on validation set)
            full_evaluator = None
            if is_ids:
                full_evaluator = ids_test_evaluator
            elif is_bitwise and self._validation_tokens:
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
                s1_evaluator=ids_s1_evaluator,
                s1_full_evaluator=ids_s1_test_evaluator,
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

    def _create_evaluator(self, context_size: int, seed: Optional[int] = None, neuron_sample_rate: float = 0.25):
        """Create the cached evaluator using pre-cached data."""
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

    def _create_multistage_evaluator(self, context_size: int, params: dict):
        """Create a MultiStageEvaluator using pre-cached data.

        Uses train split for training, test split for fitness scoring (GA/TS).
        """
        # Per-stage context sizes (e.g., [2, 4] for S0=2, S1=4)
        stage_context_size = params.get("stage_context_size")
        effective_ctx = stage_context_size if stage_context_size else context_size
        self._log(f"Creating MultiStageEvaluator (context_size={effective_ctx})...")

        from wnn.ram.architecture.multistage_evaluator import MultiStageEvaluator

        memory_mode_map = {"TERNARY": 0, "QUAD_BINARY": 1, "QUAD_WEIGHTED": 2}
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

    def _create_ids_evaluators(self, params: dict):
        """Create IDS evaluators: optimizer (131K/44K) + test (175K/82K).

        Returns (optimizer_evaluator, test_evaluator) for flat classification,
        or (s0_opt, s0_test, s1_opt, s1_test) for hierarchical classification.
        """
        from wnn.ids import load_unsw_nb15, split_train_validation
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
        feature_selection = params.get("ids_feature_selection", "all")
        rest_bits = params.get("ids_rest_bits", None)
        kfold_per_gen = params.get("ids_kfold_per_gen", 1)

        dataset_name = params.get("ids_dataset", "unsw-nb15")
        self._log(f"Loading {dataset_name} dataset (classification={classification}, split={split}, feature_selection={feature_selection})...")
        if dataset_name == "cicids2017":
            from wnn.ids.cicids2017 import load_cicids2017
            full_dataset = load_cicids2017(n_bits=n_bits, split=split, feature_selection=feature_selection)
        else:
            full_dataset = load_unsw_nb15(n_bits=n_bits, split=split, feature_selection=feature_selection, rest_bits=rest_bits)

        if classification == "hierarchical":
            return self._create_hierarchical_ids_evaluators(
                full_dataset, params, val_fraction, num_parts, k_folds, seed,
            )

        y_train = full_dataset.y_train_binary if classification == "binary" else full_dataset.y_train_multi
        y_test = full_dataset.y_test_binary if classification == "binary" else full_dataset.y_test_multi
        num_classes = 2 if classification == "binary" else len(full_dataset.category_names)

        self._log(f"  Train: {len(y_train):,} examples, Test: {len(y_test):,} examples, "
                   f"Classes: {num_classes}, Features: {full_dataset.X_train.shape[1]}")

        # Split training data: 75% optimizer train, 25% optimizer eval (validation)
        train_val_dataset, test_dataset = split_train_validation(
            full_dataset, val_fraction=val_fraction, seed=seed,
        )

        # Optimizer evaluator: train (K-fold or subset rotation), validation for fitness
        kfold_label = f", k_folds={k_folds}" if k_folds > 1 else ""
        kfpg_label = f", kfold_per_gen={kfold_per_gen}" if kfold_per_gen > 1 else ""
        self._log(f"Creating optimizer IDSEvaluator ({len(train_val_dataset.X_train):,} train / "
                   f"{len(train_val_dataset.X_test):,} eval, {num_parts} parts{kfold_label}{kfpg_label})...")
        balance_label = ", balanced" if balance_classes else ""
        sc_label = ", single-cluster" if single_cluster else ""
        self._log(f"  neuron_sample_rate={neuron_sample_rate}, balance_classes={balance_classes}{sc_label}")
        optimizer_eval = IDSEvaluator(
            dataset=train_val_dataset,
            classification=classification,
            num_parts=num_parts,
            k_folds=k_folds,
            kfold_per_gen=kfold_per_gen,
            neuron_sample_rate=neuron_sample_rate,
            balance_classes=balance_classes,
            single_cluster=single_cluster,
        )

        # Test evaluator: full 175K train, 82K test (for overfitting monitoring + final reporting)
        self._log(f"Creating test IDSEvaluator ({len(test_dataset.X_train):,} train / "
                   f"{len(test_dataset.X_test):,} eval, 1 part{balance_label})...")
        test_eval = IDSEvaluator(
            dataset=test_dataset,
            classification=classification,
            num_parts=1,
            neuron_sample_rate=neuron_sample_rate,
            balance_classes=balance_classes,
            single_cluster=single_cluster,
        )

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
        s0_opt = IDSEvaluator(dataset=s0_train_val, classification="binary", num_parts=num_parts, k_folds=k_folds,
                              neuron_sample_rate=neuron_sample_rate, balance_classes=balance_classes)
        self._log(f"  S0 test: {len(s0_test_ds.X_train):,} train / {len(s0_test_ds.X_test):,} eval")
        s0_test = IDSEvaluator(dataset=s0_test_ds, classification="binary", num_parts=1,
                               neuron_sample_rate=neuron_sample_rate, balance_classes=balance_classes)

        # ── S1: Attack types (attack flows only) ──
        self._log("── S1: Attack Classification (9 types) ──")
        attack_dataset = create_attack_only_dataset(full_dataset)
        num_attack_classes = len(attack_dataset.category_names)

        s1_train_val, s1_test_ds = split_train_validation(attack_dataset, val_fraction=val_fraction, seed=seed)

        self._log(f"  S1 optimizer: {len(s1_train_val.X_train):,} train / {len(s1_train_val.X_test):,} eval, "
                   f"{num_parts} parts{kfold_label}")
        s1_opt = IDSEvaluator(dataset=s1_train_val, classification="multi", num_parts=num_parts, k_folds=k_folds,
                              neuron_sample_rate=neuron_sample_rate, balance_classes=balance_classes)
        self._log(f"  S1 test: {len(s1_test_ds.X_train):,} train / {len(s1_test_ds.X_test):,} eval, "
                   f"Classes: {num_attack_classes}")
        s1_test = IDSEvaluator(dataset=s1_test_ds, classification="multi", num_parts=1,
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
                elif phase_type == "lambda_sweep":
                    experiment_type = ExperimentType.LAMBDA_SWEEP
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
                type_map = {
                    "ga": ExperimentType.GA,
                    "ts": ExperimentType.TS,
                    "grid_search": ExperimentType.GRID_SEARCH,
                    "lambda_sweep": ExperimentType.LAMBDA_SWEEP,
                    "neurogenesis": ExperimentType.NEUROGENESIS,
                    "synaptogenesis": ExperimentType.SYNAPTOGENESIS,
                    "axonogenesis": ExperimentType.AXONOGENESIS,
                }
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
            pop_size = exp_data.get("population_size") or params.get("population_size", 50)

            is_adaptation = experiment_type in (
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
