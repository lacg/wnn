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
    ):
        self.dashboard_url = dashboard_url
        self.poll_interval = poll_interval
        self.checkpoint_base_dir = checkpoint_base_dir
        self.context_size = context_size
        self.running = True
        self.current_flow_id: Optional[int] = None

        # Setup client
        config = DashboardClientConfig(base_url=dashboard_url)
        self.client = DashboardClient(config, logger=self._log)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _log(self, message: str):
        """Log with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        self._log(f"Received signal {signum}, shutting down...")
        self.running = False
        if self.current_flow_id:
            self._log(f"Flow {self.current_flow_id} will continue in background")

    def run(self):
        """Main worker loop."""
        self._log(f"Worker started, polling {self.dashboard_url} every {self.poll_interval}s")
        self._log(f"Checkpoints will be saved to {self.checkpoint_base_dir}")

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

        self._log(f"=" * 60)
        self._log(f"Starting flow: {flow_name} (ID: {flow_id})")
        self._log(f"=" * 60)

        try:
            # Mark as running
            self.client.flow_started(flow_id)

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

            # Create and run flow
            flow = Flow(
                config=flow_config,
                evaluator=evaluator,
                logger=self._log,
                checkpoint_dir=checkpoint_dir,
                dashboard_client=self.client,
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

    def _create_evaluator(self, context_size: int, seed: Optional[int] = None):
        """Create the cached evaluator with data."""
        self._log(f"Loading WikiText-2 dataset...")

        from datasets import load_dataset
        from transformers import GPT2TokenizerFast

        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        except Exception:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        def tokenize_text(text: str) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=False)

        train_text = "\n".join(dataset["train"]["text"])
        eval_text = "\n".join(dataset["validation"]["text"])

        train_tokens = tokenize_text(train_text)[:200_000]
        eval_tokens = tokenize_text(eval_text)[:50_000]

        self._log(f"  Train: {len(train_tokens):,} tokens, Eval: {len(eval_tokens):,} tokens")

        from wnn.ram.architecture.cached_evaluator import CachedEvaluator

        evaluator = CachedEvaluator(
            train_tokens=train_tokens,
            eval_tokens=eval_tokens,
            vocab_size=tokenizer.vocab_size,
            context_size=context_size,
            token_parts=3,
            rotation_mode="per_iteration",
            seed=seed or int(time.time() * 1000) % (2**32),
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
