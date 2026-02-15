#!/usr/bin/env python3
"""
Migrate existing checkpoint files to the dashboard database.

This script scans checkpoint directories, parses checkpoint files,
and registers them with the dashboard API.

Usage:
    python scripts/migrate_checkpoints.py [--checkpoint-dir DIR] [--dashboard-url URL] [--dry-run]

Examples:
    # Dry run to see what would be migrated
    python scripts/migrate_checkpoints.py --dry-run

    # Actually migrate
    python scripts/migrate_checkpoints.py

    # Custom checkpoint directory
    python scripts/migrate_checkpoints.py --checkpoint-dir /path/to/checkpoints
"""

import argparse
import gzip
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def parse_checkpoint_file(filepath: Path) -> Optional[dict]:
    """
    Parse a checkpoint file and extract metadata.

    Returns dict with:
        - name: Phase name
        - final_fitness: CE loss
        - final_accuracy: Accuracy
        - iterations_run: Number of iterations
        - genome_stats: Optional genome statistics
        - file_size_bytes: File size
    """
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Warning: Could not parse {filepath}: {e}")
        return None

    phase_result = data.get('phase_result', {})
    metadata = data.get('_metadata', {})

    # Extract basic info
    result = {
        'name': phase_result.get('phase_name', filepath.stem),
        'final_fitness': phase_result.get('final_fitness'),
        'final_accuracy': phase_result.get('final_accuracy'),
        'iterations_run': phase_result.get('iterations_run'),
        'file_size_bytes': filepath.stat().st_size,
        'file_path': str(filepath.absolute()),
    }

    # Try to extract genome stats
    best_genome = phase_result.get('best_genome')
    if best_genome:
        try:
            # Parse genome structure to get stats
            bits = best_genome.get('bits_per_neuron', [])
            neurons = best_genome.get('neurons_per_cluster', [])
            connections = best_genome.get('connections_per_cluster', [])

            if bits and neurons:
                total_neurons = sum(neurons)
                total_connections = sum(len(c) for c in connections) if connections else 0

                result['genome_stats'] = {
                    'num_clusters': len(bits),
                    'total_neurons': total_neurons,
                    'total_connections': total_connections,
                    'bits_range': [min(bits), max(bits)] if bits else [0, 0],
                    'neurons_range': [min(neurons), max(neurons)] if neurons else [0, 0],
                }
        except Exception as e:
            print(f"  Warning: Could not extract genome stats: {e}")

    # Determine if this is a final checkpoint (last phase in a pass)
    name_lower = result['name'].lower()
    result['is_final'] = 'phase_3b' in name_lower or 'connections' in name_lower

    return result


def scan_checkpoint_directory(base_dir: Path) -> list[dict]:
    """
    Scan a directory for checkpoint files.

    Returns list of checkpoint metadata dicts, grouped by parent directory.
    """
    checkpoints = []

    # Find all .json.gz files
    for filepath in base_dir.rglob('*.json.gz'):
        result = parse_checkpoint_file(filepath)
        if result:
            # Add parent directory info for grouping
            result['parent_dir'] = filepath.parent.name
            result['created_at'] = datetime.fromtimestamp(
                filepath.stat().st_mtime
            ).isoformat()
            checkpoints.append(result)

    # Sort by creation time
    checkpoints.sort(key=lambda x: x.get('created_at', ''))

    return checkpoints


def migrate_checkpoints(
    checkpoint_dir: Path,
    dashboard_url: str,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Migrate checkpoints to dashboard.

    Returns (success_count, error_count)
    """
    from wnn.ram.experiments.dashboard_client import DashboardClient, DashboardClientConfig

    # Scan for checkpoints
    print(f"\nScanning {checkpoint_dir}...")
    checkpoints = scan_checkpoint_directory(checkpoint_dir)

    if not checkpoints:
        print("No checkpoint files found.")
        return 0, 0

    print(f"Found {len(checkpoints)} checkpoint files.\n")

    if dry_run:
        print("DRY RUN - Would migrate the following checkpoints:\n")
        for ckpt in checkpoints:
            fitness = ckpt.get('final_fitness', 0)
            accuracy = ckpt.get('final_accuracy', 0)
            acc_str = f"{accuracy:.4%}" if accuracy else "N/A"
            print(f"  [{ckpt['parent_dir']}] {ckpt['name']}")
            print(f"    CE: {fitness:.4f}, Acc: {acc_str}")
            print(f"    Path: {ckpt['file_path']}")
            print(f"    Final: {ckpt.get('is_final', False)}")
            print()
        return len(checkpoints), 0

    # Connect to dashboard
    print(f"Connecting to dashboard at {dashboard_url}...")
    try:
        config = DashboardClientConfig(base_url=dashboard_url)
        client = DashboardClient(config, logger=lambda x: None)

        if not client.ping():
            print(f"Error: Could not connect to dashboard at {dashboard_url}")
            return 0, len(checkpoints)
    except Exception as e:
        print(f"Error: {e}")
        return 0, len(checkpoints)

    print("Connected.\n")

    # Group checkpoints by parent directory (experiment)
    groups: dict[str, list[dict]] = {}
    for ckpt in checkpoints:
        parent = ckpt['parent_dir']
        if parent not in groups:
            groups[parent] = []
        groups[parent].append(ckpt)

    success_count = 0
    error_count = 0

    # Check for existing checkpoints to avoid duplicates
    print("Checking for existing checkpoints...")
    existing_paths = set()
    try:
        existing = client.list_checkpoints(limit=1000)
        existing_paths = {c.get('file_path') for c in existing}
        print(f"Found {len(existing_paths)} existing checkpoints.\n")
    except Exception as e:
        print(f"Warning: Could not fetch existing checkpoints: {e}\n")

    # Create experiments and register checkpoints
    for group_name, group_checkpoints in groups.items():
        print(f"Processing {group_name} ({len(group_checkpoints)} checkpoints)...")

        # Create a dummy experiment for this group
        try:
            exp_id = client.create_experiment(
                name=f"Migrated: {group_name}",
                log_path="",
                config={"migrated": True, "original_dir": group_name},
            )
            print(f"  Created experiment {exp_id}")
        except Exception as e:
            print(f"  Error creating experiment: {e}")
            error_count += len(group_checkpoints)
            continue

        # Register each checkpoint
        for ckpt in group_checkpoints:
            # Skip if already exists
            if ckpt['file_path'] in existing_paths:
                print(f"  Skipping (exists): {ckpt['name']}")
                continue

            try:
                ckpt_id = client.checkpoint_created(
                    experiment_id=exp_id,
                    file_path=ckpt['file_path'],
                    name=ckpt['name'],
                    final_fitness=ckpt.get('final_fitness'),
                    final_accuracy=ckpt.get('final_accuracy'),
                    iterations_run=ckpt.get('iterations_run'),
                    genome_stats=ckpt.get('genome_stats'),
                    is_final=ckpt.get('is_final', False),
                )
                print(f"  Registered checkpoint {ckpt_id}: {ckpt['name']}")
                success_count += 1
            except Exception as e:
                print(f"  Error registering {ckpt['name']}: {e}")
                error_count += 1

    return success_count, error_count


def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing checkpoint files to the dashboard database."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Base directory containing checkpoint files (default: checkpoints)",
    )
    parser.add_argument(
        "--dashboard-url",
        type=str,
        default="http://localhost:3000",
        help="Dashboard API URL (default: http://localhost:3000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )

    args = parser.parse_args()

    # Resolve checkpoint directory
    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.is_absolute():
        # Try relative to current directory, then relative to script
        if not checkpoint_dir.exists():
            script_dir = Path(__file__).parent.parent
            checkpoint_dir = script_dir / args.checkpoint_dir

    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {args.checkpoint_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  Checkpoint Migration Tool")
    print("=" * 60)
    print(f"  Directory: {checkpoint_dir}")
    print(f"  Dashboard: {args.dashboard_url}")
    print(f"  Dry run:   {args.dry_run}")

    success, errors = migrate_checkpoints(
        checkpoint_dir=checkpoint_dir,
        dashboard_url=args.dashboard_url,
        dry_run=args.dry_run,
    )

    print("\n" + "=" * 60)
    print(f"  Migration complete: {success} succeeded, {errors} failed")
    print("=" * 60)

    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()
