#!/usr/bin/env python3
"""
Analyze sweep results and run a hybrid experiment with best tier configurations.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

WNN_DIR = Path("/Users/lacg/Library/Mobile Documents/com~apple~CloudDocs/Studies/research/wnn")
SWEEP_OUTPUT = WNN_DIR / "nohup_sweep.out"
RESULTS_DIR = WNN_DIR / "experiments"


def wait_for_sweep_completion():
    """Wait for sweep to complete."""
    print("Waiting for sweep to complete...")
    while True:
        if SWEEP_OUTPUT.exists():
            content = SWEEP_OUTPUT.read_text()
            # Check if we see the summary table (indicates completion)
            if "SWEEP RESULTS" in content or "Best PPL:" in content:
                print("\nSweep completed!")
                return True
            # Check progress
            lines = content.strip().split('\n')
            for line in reversed(lines[-20:]):
                if line.strip():
                    print(f"\r{line[:80]}", end="", flush=True)
                    break
        time.sleep(60)


def find_latest_results_file():
    """Find the most recent sweep results JSON."""
    json_files = list(RESULTS_DIR.glob("sweep_standard_*.json"))
    if not json_files:
        json_files = list(RESULTS_DIR.glob("sweep_*.json"))
    if not json_files:
        print("ERROR: No sweep results found in experiments/")
        return None
    return max(json_files, key=lambda f: f.stat().st_mtime)


def parse_tier_config(config_str: str) -> list[dict]:
    """Parse tiered config string like '100,15,16;400,10,10;rest,5,8'."""
    tiers = []
    for i, part in enumerate(config_str.split(';')):
        count, neurons, bits = part.split(',')
        tiers.append({
            'tier': i,
            'count': count,
            'neurons': int(neurons),
            'bits': int(bits),
        })
    return tiers


def analyze_results(results_file: Path) -> dict:
    """Analyze sweep results and find best config per tier."""
    with open(results_file) as f:
        results = json.load(f)

    print(f"\nAnalyzing {len(results)} experiments from {results_file.name}")
    print("=" * 80)

    # Collect per-tier results
    tier_results = {0: [], 1: [], 2: []}

    for exp in results:
        name = exp['name']
        config = exp['config']
        tier_configs = parse_tier_config(config)

        print(f"\n{name}: {config}")
        print(f"  Overall PPL: {exp['overall_ppl']:.1f}, Acc: {exp['overall_accuracy']:.2%}")

        for tr in exp.get('tier_results', []):
            tier_idx = tr['tier']
            tc = tier_configs[tier_idx] if tier_idx < len(tier_configs) else None

            tier_results[tier_idx].append({
                'experiment': name,
                'config': f"{tc['count']},{tc['neurons']},{tc['bits']}" if tc else "unknown",
                'neurons': tc['neurons'] if tc else 0,
                'bits': tc['bits'] if tc else 0,
                'ppl': tr['ppl'],
                'accuracy': tr['accuracy'],
                'data_pct': tr['data_pct'],
            })

            print(f"  Tier {tier_idx}: PPL {tr['ppl']:.1f}, Acc {tr['accuracy']:.2%} "
                  f"({tc['neurons']}n×{tc['bits']}b)" if tc else f"  Tier {tier_idx}: PPL {tr['ppl']:.1f}")

    # Find best config for each tier (lowest PPL)
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION PER TIER (by PPL)")
    print("=" * 80)

    best_per_tier = {}
    for tier_idx in [0, 1, 2]:
        if tier_results[tier_idx]:
            best = min(tier_results[tier_idx], key=lambda x: x['ppl'])
            best_per_tier[tier_idx] = best
            print(f"\nTier {tier_idx}: {best['config']}")
            print(f"  From experiment: {best['experiment']}")
            print(f"  PPL: {best['ppl']:.1f}, Accuracy: {best['accuracy']:.2%}")
            print(f"  Data coverage: {best['data_pct']:.1f}%")

    return best_per_tier


def create_hybrid_config(best_per_tier: dict) -> str:
    """Create a hybrid tiered config from best per-tier results."""
    parts = []
    for tier_idx in [0, 1, 2]:
        if tier_idx in best_per_tier:
            config = best_per_tier[tier_idx]['config']
            # Handle 'rest' for tier 2
            count, neurons, bits = config.split(',')
            if tier_idx == 2:
                parts.append(f"rest,{neurons},{bits}")
            else:
                parts.append(config)

    return ";".join(parts)


def run_hybrid_experiment(config: str):
    """Run the hybrid experiment."""
    print("\n" + "=" * 80)
    print("RUNNING HYBRID EXPERIMENT")
    print("=" * 80)
    print(f"Config: {config}")
    print()

    cmd = [
        sys.executable,
        str(WNN_DIR / "tests" / "ramlm_full_benchmark.py"),
        "--mode", "full",
        "--tiered", config,
        "--context", "4",
        "--per-tier",
        "--full-data",
        "--optimize",
        "--strategy", "GA,TS",
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Run and stream output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end='')

    process.wait()
    return process.returncode


def run_extended_sweep():
    """Run the extended sweep (adds 4 more experiments to standard).

    Uses the existing standard sweep results file to skip already-completed experiments.
    """
    print("\n" + "=" * 80)
    print("STARTING EXTENDED SWEEP")
    print("=" * 80)

    # Find the latest standard sweep results to continue from
    # This ensures we don't re-run experiments that already completed
    existing_results = find_latest_results_file()
    if existing_results:
        output_file = existing_results.name
        print(f"Continuing from existing results: {output_file}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"sweep_extended_{timestamp}.json"
        print(f"Starting fresh: {output_file}")

    cmd = [
        sys.executable,
        str(WNN_DIR / "tests" / "ramlm_full_benchmark.py"),
        "--sweep",
        "--set", "extended",
        "--output", output_file,
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Output: experiments/{output_file}")
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Run and stream output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end='')

    process.wait()
    return process.returncode


def main():
    print("=" * 80)
    print("SWEEP ANALYSIS AND HYBRID EXPERIMENT")
    print("=" * 80)

    # Check if sweep is still running
    if SWEEP_OUTPUT.exists():
        content = SWEEP_OUTPUT.read_text()
        if "Best PPL:" not in content and "SWEEP RESULTS" not in content:
            wait_for_sweep_completion()

    # Find and analyze results
    results_file = find_latest_results_file()
    if not results_file:
        return 1

    best_per_tier = analyze_results(results_file)

    if len(best_per_tier) < 3:
        print("\nERROR: Could not find results for all tiers")
        return 1

    # Create hybrid config
    hybrid_config = create_hybrid_config(best_per_tier)

    print("\n" + "=" * 80)
    print("PROPOSED HYBRID CONFIGURATION")
    print("=" * 80)
    print(f"\nCombining best configs from each tier:")
    for tier_idx, best in best_per_tier.items():
        print(f"  Tier {tier_idx}: {best['config']} (from {best['experiment']}, PPL={best['ppl']:.1f})")
    print(f"\nHybrid config: {hybrid_config}")

    # Run hybrid experiment (auto-yes)
    print("\n" + "=" * 80)
    print("RUNNING HYBRID EXPERIMENT")
    print("=" * 80)
    hybrid_result = run_hybrid_experiment(hybrid_config)

    if hybrid_result != 0:
        print("WARNING: Hybrid experiment may have failed")

    # Run extended sweep
    print("\n" + "=" * 80)
    print("HYBRID COMPLETE - STARTING EXTENDED SWEEP")
    print("=" * 80)
    return run_extended_sweep()


if __name__ == "__main__":
    sys.exit(main())
