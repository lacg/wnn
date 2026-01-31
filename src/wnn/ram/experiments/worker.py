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
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from wnn.ram.experiments.dashboard_client import DashboardClient, DashboardClientConfig
from wnn.ram.experiments.flow import Flow, FlowConfig
from wnn.ram.experiments.experiment import ExperimentConfig
from wnn.ram.experiments.tracker import create_tracker, ExperimentTracker


class FlowWorker:
    """
    Worker that polls for queued flows and executes them.
    """

    def __init__(
        self,
        dashboard_url: str = "http://localhost:3000",
        poll_interval: int = 10,
        checkpoint_base_dir: Path = Path("checkpoints"),
        context_size: int = 4,
        db_path: Optional[str] = None,
    ):
        self.dashboard_url = dashboard_url
        self.poll_interval = poll_interval
        self.checkpoint_base_dir = checkpoint_base_dir
        self.context_size = context_size
        self.running = True
        self.current_flow_id: Optional[int] = None

        # Pre-cached data (loaded once at startup)
        self._tokenizer = None
        self._train_tokens: Optional[list[int]] = None
        self._eval_tokens: Optional[list[int]] = None
        self._vocab_size: Optional[int] = None
        self._cluster_order: Optional[list[int]] = None  # Tokens sorted by frequency

        # Log file for current flow (path only, not kept open)
        self._log_file: Optional[Path] = None
        self._log_dir = Path("logs")

        # Setup client
        config = DashboardClientConfig(base_url=dashboard_url)
        self.client = DashboardClient(config, logger=self._log)

        # V2 Tracker for direct database writes (same db as dashboard)
        # Default to dashboard/dashboard.db relative to working directory
        if db_path is None:
            db_path = str(Path("dashboard/dashboard.db").absolute())
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
        """Handle shutdown signals."""
        self._log(f"Received signal {signum}, requesting graceful shutdown...")
        self.running = False
        if self.current_flow_id:
            self._log(f"Flow {self.current_flow_id} will stop after current generation/iteration")

    def should_stop(self) -> bool:
        """Check if shutdown has been requested. Used by flow/experiment/strategy."""
        return not self.running

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

        # Tokenize once and cache
        self._log("  Tokenizing dataset...")
        train_text = "\n".join(dataset["train"]["text"])
        eval_text = "\n".join(dataset["validation"]["text"])

        self._train_tokens = self._tokenizer.encode(train_text, add_special_tokens=False)[:200_000]
        self._eval_tokens = self._tokenizer.encode(eval_text, add_special_tokens=False)[:50_000]

        self._log(f"  Cached {len(self._train_tokens):,} train tokens, {len(self._eval_tokens):,} eval tokens")

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
                # Check for queued flows
                flow_data = self._get_next_queued_flow()

                if flow_data:
                    self._execute_flow(flow_data)
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

            # Parse flow configuration
            config = flow_data.get("config", {})
            params = config.get("params", {})
            experiments = config.get("experiments", [])

            # Get context size from params or use default
            context_size = params.get("context_size", self.context_size)

            # Setup checkpoint directory
            safe_name = flow_name.lower().replace(" ", "_").replace("/", "_")
            checkpoint_dir = self.checkpoint_base_dir / safe_name

            # Load data and create evaluator
            evaluator = self._create_evaluator(context_size, params.get("seed"))

            # Build experiment configs
            exp_configs = self._build_experiment_configs(experiments, params, evaluator.vocab_size)

            # Parse tier config
            tier_config = self._parse_tier_config(params.get("tier_config"))

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
                seed=params.get("seed"),
            )

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

            # Create and run flow (pass existing flow_id to avoid duplication)
            flow = Flow(
                config=flow_config,
                evaluator=evaluator,
                logger=self._log,
                checkpoint_dir=checkpoint_dir,
                dashboard_client=self.client,
                flow_id=flow_id,
                tracker=self.tracker,
                shutdown_check=self.should_stop,  # Pass shutdown check for graceful stop
            )

            result = flow.run()

            # Mark as completed
            self.client.flow_completed(flow_id)
            self._log(f"Flow completed: CE={result.final_fitness:.4f}")

        except Exception as e:
            self._log(f"Flow failed: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.client.flow_failed(flow_id, str(e))
            except Exception:
                pass

        finally:
            self.current_flow_id = None
            self._close_log_file()

    def _create_evaluator(self, context_size: int, seed: Optional[int] = None):
        """Create the cached evaluator using pre-cached data."""
        self._log(f"Creating evaluator (context_size={context_size})...")

        from wnn.ram.architecture.cached_evaluator import CachedEvaluator

        evaluator = CachedEvaluator(
            train_tokens=self._train_tokens,
            eval_tokens=self._eval_tokens,
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

    def _build_experiment_configs(
        self,
        experiments: list[dict],
        params: dict,
        vocab_size: int,
    ) -> list[ExperimentConfig]:
        """Build experiment configs from flow data."""
        tier_config = self._parse_tier_config(params.get("tier_config"))
        tier0_only = params.get("optimize_tier0_only", params.get("tier0_only", False))
        patience = params.get("patience", 10)
        fitness_percentile = params.get("fitness_percentile")
        seed = params.get("seed")

        exp_configs = []
        for exp_data in experiments:
            exp_params = exp_data.get("params", {})
            exp_config = ExperimentConfig(
                name=exp_data.get("name", "Unnamed"),
                experiment_type=exp_data.get("experiment_type", "ga"),
                optimize_bits=exp_data.get("optimize_bits", False),
                optimize_neurons=exp_data.get("optimize_neurons", False),
                optimize_connections=exp_data.get("optimize_connections", False),
                generations=exp_params.get("generations", params.get("ga_generations", 250)),
                population_size=exp_params.get("population_size", params.get("population_size", 50)),
                iterations=exp_params.get("iterations", params.get("ts_iterations", 250)),
                neighbors_per_iter=exp_params.get("neighbors_per_iter", params.get("neighbors_per_iter", 50)),
                patience=exp_params.get("patience", patience),
                tier_config=tier_config,
                optimize_tier0_only=tier0_only,
                fitness_percentile=fitness_percentile,
                seed=seed,
            )
            exp_configs.append(exp_config)

        return exp_configs

    def _parse_tier_config(self, tier_config) -> Optional[list[tuple]]:
        """Parse tier config from string or array format."""
        if not tier_config:
            return None

        if isinstance(tier_config, str):
            # Parse string format: "100,15,20;400,10,12;rest,5,8"
            tiers = []
            for tier_str in tier_config.split(";"):
                parts = tier_str.strip().split(",")
                if len(parts) == 3:
                    count = None if parts[0].strip().lower() == "rest" else int(parts[0])
                    neurons = int(parts[1])
                    bits = int(parts[2])
                    tiers.append((count, neurons, bits))
            return tiers if tiers else None

        elif isinstance(tier_config, list):
            # Already in array format
            return [(t[0], t[1], t[2]) for t in tier_config]

        return None


def main():
    parser = argparse.ArgumentParser(description="Flow worker - polls and executes queued flows")
    parser.add_argument("--url", default="http://localhost:3000", help="Dashboard URL")
    parser.add_argument("--poll-interval", type=int, default=10, help="Seconds between polls")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Base checkpoint directory")
    parser.add_argument("--context", type=int, default=4, help="Default context size")

    args = parser.parse_args()

    worker = FlowWorker(
        dashboard_url=args.url,
        poll_interval=args.poll_interval,
        checkpoint_base_dir=args.checkpoint_dir,
        context_size=args.context,
    )

    worker.run()


if __name__ == "__main__":
    main()
