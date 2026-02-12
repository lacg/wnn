#!/usr/bin/env python3
"""
RAMLM Benchmark - Test RAM Language Model on WikiText-2.

This script tests the proper RAM WNN architecture (RAMLM + RAMClusterLayer)
with connectivity optimization on real language data.

Usage:
    python ramlm_benchmark.py                    # Quick test (10k tokens)
    python ramlm_benchmark.py --full             # Full dataset
    python ramlm_benchmark.py --optimize         # Include connectivity optimization
    python ramlm_benchmark.py --context 6        # 6-gram context

Logs are saved to: wnn/logs/YYYY/MM/DD/ramlm_HHMMSS.log
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from time import time
from typing import Optional


# =============================================================================
# Logging Setup
# =============================================================================

class Logger:
    """Simple logger that writes to both console and file."""

    def __init__(self, log_dir: Optional[Path] = None):
        self.start_time = time()
        self.log_file = None

        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%H%M%S")
            self.log_file = log_dir / f"ramlm_{timestamp}.log"
            # Write header
            with open(self.log_file, "w") as f:
                f.write(f"RAMLM Benchmark - {datetime.now().isoformat()}\n")
                f.write("=" * 60 + "\n\n")

    def log(self, msg: str = "", end: str = "\n"):
        """Log message to console and file."""
        elapsed = time() - self.start_time
        timestamp = f"[{elapsed:7.1f}s]"
        full_msg = f"{timestamp} {msg}"

        print(full_msg, end=end, flush=True)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(full_msg + end)

    def section(self, title: str):
        """Log a section header."""
        self.log()
        self.log("=" * 60)
        self.log(title)
        self.log("=" * 60)


# Global logger
logger: Logger = None


def log(msg: str = "", end: str = "\n"):
    """Convenience function for logging."""
    if logger:
        logger.log(msg, end)
    else:
        print(msg, end=end, flush=True)


# =============================================================================
# Data Loading
# =============================================================================

def load_wikitext2(full_data: bool = False):
    """Load and tokenize WikiText-2."""
    log("Loading WikiText-2 dataset...")

    start = time()
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    log(f"  Dataset loaded in {time() - start:.1f}s")

    # Get raw text
    train_text = "\n".join([t for t in dataset["train"]["text"] if t.strip()])
    val_text = "\n".join([t for t in dataset["validation"]["text"] if t.strip()])
    test_text = "\n".join([t for t in dataset["test"]["text"] if t.strip()])

    log(f"  Train: {len(train_text):,} chars")
    log(f"  Val: {len(val_text):,} chars")
    log(f"  Test: {len(test_text):,} chars")

    # Tokenize
    log("Tokenizing with GPT-2...")
    start = time()
    from wnn.tokenizers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer()

    train_tokens = tokenizer.encode(train_text)
    val_tokens = tokenizer.encode(val_text)
    test_tokens = tokenizer.encode(test_text)

    log(f"  Tokenized in {time() - start:.1f}s")
    log(f"  Train: {len(train_tokens):,} tokens")
    log(f"  Val: {len(val_tokens):,} tokens")
    log(f"  Test: {len(test_tokens):,} tokens")

    if not full_data:
        # Use smaller samples for quick testing
        train_tokens = train_tokens[:50_000]
        val_tokens = val_tokens[:10_000]
        test_tokens = test_tokens[:10_000]
        log(f"  Using sample: {len(train_tokens):,} train, {len(val_tokens):,} val, {len(test_tokens):,} test")

    return train_tokens, val_tokens, test_tokens


# =============================================================================
# Model Creation
# =============================================================================

def create_model(args):
    """Create RAMLM model."""
    from wnn.ram.core import RAMLM

    model = RAMLM(
        vocab_size=50257,
        context_size=args.context,
        neurons_per_cluster=args.neurons,
        bits_per_neuron=args.bits,
        pad_token_id=50256,
        rng=args.seed,
    )

    log(f"Model created:")
    log(f"  Context size: {args.context} tokens")
    log(f"  Neurons per cluster: {args.neurons}")
    log(f"  Bits per neuron: {args.bits}")
    log(f"  Total neurons: {model.layer.total_neurons:,}")
    log(f"  Total input bits: {model.total_input_bits}")
    log(f"  Memory cells per neuron: {2**args.bits:,}")
    log(f"  Total memory cells: {model.layer.total_neurons * (2**args.bits):,}")

    # Estimate memory
    memory_bytes = model.layer.total_neurons * (2**args.bits) * 2 // 8  # 2 bits per cell
    log(f"  Estimated memory: {memory_bytes / 1e6:.1f} MB")

    return model


# =============================================================================
# Training
# =============================================================================

def train_model(model, train_tokens, args):
    """Train the model (Phase 1: memory writes)."""
    logger.section("Phase 1: Memory Training")

    total_examples = len(train_tokens) - model.context_size
    log(f"Training on {total_examples:,} examples...")
    log(f"Global top-k: {args.top_k}")
    log(f"Using: train_epoch_fast (vectorized encoding)")

    start = time()
    stats = model.train_epoch_fast(
        train_tokens,
        global_top_k=args.top_k,
        batch_size=args.batch_size,
        verbose=True,
    )
    train_time = time() - start

    log(f"Training complete in {train_time:.1f}s")
    log(f"  Modified {stats['modified']:,} memory cells")
    log(f"  Rate: {total_examples / train_time:.0f} examples/s")

    return train_time


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(model, tokens, name: str = "", batch_size: int = 10000):
    """Evaluate the model on a token sequence using fast batch evaluation."""
    total_examples = len(tokens) - model.context_size
    log(f"Evaluating on {total_examples:,} {name} examples...")
    log(f"Using: evaluate_fast (batch_size={batch_size})")

    start = time()
    stats = model.evaluate_fast(tokens, batch_size=batch_size, verbose=True)
    eval_time = time() - start

    log(f"Evaluation complete in {eval_time:.1f}s")
    log(f"  Rate: {total_examples / eval_time:.0f} examples/s")

    return stats, eval_time


# =============================================================================
# Connectivity Optimization
# =============================================================================

def optimize_connectivity(model, train_tokens, val_tokens, args):
    """Optimize connectivity (Phase 2)."""
    logger.section("Phase 2: Connectivity Optimization")

    from wnn.ram.core import OptimizationMethod
    from wnn.ram.strategies.connectivity import ConnectivityOptimizer, OptimizationConfig

    log(f"Method: {args.opt_method}")
    log(f"Iterations: {args.opt_iterations}")
    log(f"Neighbors per iteration: {args.opt_neighbors}")

    config = OptimizationConfig(
        method=OptimizationMethod[args.opt_method],
        ts_iterations=args.opt_iterations,
        ts_neighbors_per_iter=args.opt_neighbors,
        ts_mutation_rate=0.01,
        verbose=True,
    )

    optimizer = ConnectivityOptimizer(config=config)

    start = time()

    # Custom train/eval with progress logging
    def train_fn():
        log("    [opt] Training...")
        model.train_epoch(train_tokens, global_top_k=args.top_k, verbose=False)

    def eval_fn() -> float:
        log("    [opt] Evaluating...")
        stats = model.evaluate(val_tokens, verbose=False)
        log(f"    [opt] CE: {stats['cross_entropy']:.4f}")
        return stats['cross_entropy']

    result = optimizer.optimize(
        model=model,
        train_fn=train_fn,
        eval_fn=eval_fn,
        total_input_bits=model.total_input_bits,
        num_neurons=model.layer.total_neurons,
        bits_per_neuron=model.layer.bits_per_neuron,
    )

    opt_time = time() - start

    log(f"Optimization complete in {opt_time:.1f}s")
    log(f"  Initial CE: {result.initial_cross_entropy:.4f}")
    log(f"  Final CE: {result.final_cross_entropy:.4f}")
    log(f"  Improvement: {result.improvement_percent:.2f}%")

    return result, opt_time


# =============================================================================
# Main
# =============================================================================

def main():
    global logger

    parser = argparse.ArgumentParser(
        description="RAMLM Benchmark on WikiText-2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data options
    parser.add_argument("--full", action="store_true",
                        help="Use full WikiText-2 dataset (2M tokens)")

    # Model options
    parser.add_argument("--context", type=int, default=4,
                        help="Context size (n-gram order)")
    parser.add_argument("--neurons", type=int, default=5,
                        help="Neurons per cluster (odd for majority)")
    parser.add_argument("--bits", type=int, default=8,
                        help="Bits per neuron (memory = 2^bits)")
    parser.add_argument("--top-k", type=int, default=500,
                        help="Global top-k for FALSE training")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Batch size for training/evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Optimization options
    parser.add_argument("--optimize", action="store_true",
                        help="Run connectivity optimization (Phase 2)")
    parser.add_argument("--opt-method", type=str, default="TABU_SEARCH",
                        choices=["TABU_SEARCH", "GENETIC_ALGORITHM", "SIMULATED_ANNEALING"],
                        help="Optimization method")
    parser.add_argument("--opt-iterations", type=int, default=5,
                        help="Optimization iterations")
    parser.add_argument("--opt-neighbors", type=int, default=20,
                        help="Neighbors per iteration (TS)")

    # Output options
    parser.add_argument("--no-log", action="store_true",
                        help="Don't write log file")

    args = parser.parse_args()

    # Setup logging
    if not args.no_log:
        today = datetime.now()
        log_dir = Path(__file__).parent.parent / "logs" / f"{today.year}" / f"{today.month:02d}" / f"{today.day:02d}"
        logger = Logger(log_dir)
        log(f"Log file: {logger.log_file}")
    else:
        logger = Logger(None)

    # Print configuration
    logger.section("Configuration")
    log(f"Data: {'FULL WikiText-2' if args.full else 'Sample (50k train, 10k val)'}")
    log(f"Context: {args.context} tokens")
    log(f"Neurons per cluster: {args.neurons}")
    log(f"Bits per neuron: {args.bits}")
    log(f"Top-k: {args.top_k}")
    log(f"Batch size: {args.batch_size}")
    log(f"Optimization: {'Yes' if args.optimize else 'No'}")
    if args.optimize:
        log(f"  Method: {args.opt_method}")
        log(f"  Iterations: {args.opt_iterations}")

    # Load data
    logger.section("Data Loading")
    train_tokens, val_tokens, test_tokens = load_wikitext2(args.full)

    # Create model
    logger.section("Model Creation")
    model = create_model(args)

    # Train
    train_time = train_model(model, train_tokens, args)

    # Evaluate on validation
    logger.section("Validation Evaluation")
    val_stats, val_time = evaluate_model(model, val_tokens, "validation", 100)  # Small batches are faster

    # Optimize connectivity if requested
    opt_result = None
    opt_time = 0
    if args.optimize:
        opt_result, opt_time = optimize_connectivity(model, train_tokens, val_tokens, args)

        # Re-evaluate after optimization
        logger.section("Post-Optimization Evaluation")
        val_stats_post, _ = evaluate_model(model, val_tokens, "validation", 100)

    # Final evaluation on test set
    logger.section("Test Evaluation")
    test_stats, test_time = evaluate_model(model, test_tokens, "test", 100)  # Small batches are faster

    # Summary
    logger.section("SUMMARY")
    log(f"Model: {args.context}-gram, {args.neurons} neurons/cluster, {args.bits} bits/neuron")
    log(f"Data: {len(train_tokens):,} train, {len(val_tokens):,} val, {len(test_tokens):,} test")
    log()
    log("Results:")
    log(f"  Validation CE: {val_stats['cross_entropy']:.4f}")
    log(f"  Validation PPL: {val_stats['perplexity']:.2f}")
    log(f"  Validation Acc: {val_stats['accuracy']:.2%}")
    log()
    log(f"  Test CE: {test_stats['cross_entropy']:.4f}")
    log(f"  Test PPL: {test_stats['perplexity']:.2f}")
    log(f"  Test Acc: {test_stats['accuracy']:.2%}")

    if opt_result:
        log()
        log("Optimization:")
        log(f"  Initial CE: {opt_result.initial_cross_entropy:.4f}")
        log(f"  Final CE: {opt_result.final_cross_entropy:.4f}")
        log(f"  Improvement: {opt_result.improvement_percent:.2f}%")

    log()
    log("Timings:")
    log(f"  Training: {train_time:.1f}s")
    log(f"  Validation eval: {val_time:.1f}s")
    log(f"  Test eval: {test_time:.1f}s")
    if args.optimize:
        log(f"  Optimization: {opt_time:.1f}s")
    log(f"  Total: {time() - logger.start_time:.1f}s")

    if logger.log_file:
        log()
        log(f"Log saved to: {logger.log_file}")


if __name__ == "__main__":
    main()
