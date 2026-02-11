#!/usr/bin/env python3
"""
WNN Experiment CLI (wnn-exp)

Command-line interface for managing flows and checkpoints.

Usage:
    wnn-exp flow list [--status STATUS]
    wnn-exp flow create --name NAME [--template TEMPLATE] [--seed-from ID]
    wnn-exp flow show FLOW_ID
    wnn-exp flow run FLOW_ID [--resume-from INDEX]
    wnn-exp checkpoint list [--final-only] [--experiment-id ID]
    wnn-exp checkpoint show CHECKPOINT_ID
    wnn-exp checkpoint delete CHECKPOINT_ID [--force]
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from wnn.ram.experiments.dashboard_client import DashboardClient, DashboardClientConfig

# Create Typer apps
app = typer.Typer(
	name="wnn-exp",
	help="WNN Experiment Manager - Manage flows and checkpoints",
	no_args_is_help=True,
)
flow_app = typer.Typer(help="Flow management commands")
checkpoint_app = typer.Typer(help="Checkpoint management commands")
app.add_typer(flow_app, name="flow")
app.add_typer(checkpoint_app, name="checkpoint")

console = Console()


def get_client(base_url: str = "https://localhost:3000") -> DashboardClient:
	"""Create dashboard client."""
	config = DashboardClientConfig(base_url=base_url)
	return DashboardClient(config, logger=lambda x: None)


def format_datetime(dt_str: Optional[str]) -> str:
	"""Format datetime string for display."""
	if not dt_str:
		return "-"
	try:
		dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
		return dt.strftime("%Y-%m-%d %H:%M")
	except Exception:
		return dt_str


def format_duration(start_str: Optional[str], end_str: Optional[str]) -> str:
	"""Format duration between two datetime strings."""
	if not start_str:
		return "-"
	if not end_str:
		return "Running..."
	try:
		start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
		end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
		delta = end - start
		seconds = int(delta.total_seconds())
		if seconds < 60:
			return f"{seconds}s"
		if seconds < 3600:
			return f"{seconds // 60}m {seconds % 60}s"
		hours = seconds // 3600
		mins = (seconds % 3600) // 60
		return f"{hours}h {mins}m"
	except Exception:
		return "-"


def status_color(status: str) -> str:
	"""Get color for status."""
	colors = {
		"pending": "yellow",
		"running": "blue",
		"completed": "green",
		"failed": "red",
		"cancelled": "dim",
	}
	return colors.get(status.lower(), "white")


# =============================================================================
# Flow commands
# =============================================================================

@flow_app.command("list")
def flow_list(
	status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
	limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
	url: str = typer.Option("https://localhost:3000", "--url", help="Dashboard URL"),
):
	"""List all flows."""
	try:
		client = get_client(url)
		flows = client.list_flows(status=status, limit=limit)

		if not flows:
			rprint("[dim]No flows found.[/dim]")
			return

		table = Table(title="Flows")
		table.add_column("ID", style="cyan", width=6)
		table.add_column("Name", style="white")
		table.add_column("Status", width=10)
		table.add_column("Experiments", justify="right", width=6)
		table.add_column("Created", width=16)
		table.add_column("Duration", width=12)

		for flow in flows:
			config = flow.get("config", {})
			experiments = config.get("experiments", [])
			status_val = flow.get("status", "unknown")

			table.add_row(
				str(flow.get("id", "")),
				flow.get("name", ""),
				f"[{status_color(status_val)}]{status_val}[/{status_color(status_val)}]",
				str(len(experiments)),
				format_datetime(flow.get("created_at")),
				format_duration(flow.get("started_at"), flow.get("completed_at")),
			)

		console.print(table)

	except ConnectionError as e:
		rprint(f"[red]Error: Could not connect to dashboard at {url}[/red]")
		rprint(f"[dim]{e}[/dim]")
		raise typer.Exit(1)
	except Exception as e:
		rprint(f"[red]Error: {e}[/red]")
		raise typer.Exit(1)


@flow_app.command("show")
def flow_show(
	flow_id: int = typer.Argument(..., help="Flow ID"),
	url: str = typer.Option("https://localhost:3000", "--url", help="Dashboard URL"),
):
	"""Show flow details."""
	try:
		client = get_client(url)
		flow = client.get_flow(flow_id)

		if not flow:
			rprint(f"[red]Flow {flow_id} not found[/red]")
			raise typer.Exit(1)

		# Header
		status = flow.get("status", "unknown")
		rprint(f"\n[bold]{flow.get('name')}[/bold] [dim](ID: {flow_id})[/dim]")
		rprint(f"Status: [{status_color(status)}]{status}[/{status_color(status)}]")

		if flow.get("description"):
			rprint(f"[dim]{flow.get('description')}[/dim]")

		# Metadata
		rprint(f"\nCreated: {format_datetime(flow.get('created_at'))}")
		if flow.get("started_at"):
			rprint(f"Started: {format_datetime(flow.get('started_at'))}")
		if flow.get("completed_at"):
			rprint(f"Completed: {format_datetime(flow.get('completed_at'))}")
			rprint(f"Duration: {format_duration(flow.get('started_at'), flow.get('completed_at'))}")

		# Config
		config = flow.get("config", {})
		if config.get("template"):
			rprint(f"Template: {config.get('template')}")

		# Experiments
		experiments = config.get("experiments", [])
		if experiments:
			rprint(f"\n[bold]Experiments ({len(experiments)}):[/bold]")
			for i, exp in enumerate(experiments):
				exp_type = exp.get("experiment_type", "?").upper()
				opts = []
				if exp.get("optimize_bits"):
					opts.append("bits")
				if exp.get("optimize_neurons"):
					opts.append("neurons")
				if exp.get("optimize_connections"):
					opts.append("connections")
				opts_str = ", ".join(opts) if opts else "none"
				rprint(f"  {i+1}. [cyan]{exp.get('name', f'Experiment {i+1}')}[/cyan] ({exp_type}) - {opts_str}")

		# Fetch completed experiments
		try:
			completed = client.list_flow_experiments(flow_id)
			if completed:
				rprint(f"\n[bold]Completed Runs ({len(completed)}):[/bold]")
				for exp in completed:
					exp_status = exp.get("status", "unknown")
					rprint(f"  • {exp.get('name')} [{status_color(exp_status)}]{exp_status}[/{status_color(exp_status)}]")
		except Exception:
			pass

		rprint("")

	except ConnectionError as e:
		rprint(f"[red]Error: Could not connect to dashboard at {url}[/red]")
		raise typer.Exit(1)
	except Exception as e:
		rprint(f"[red]Error: {e}[/red]")
		raise typer.Exit(1)


@flow_app.command("create")
def flow_create(
	name: str = typer.Option(..., "--name", "-n", help="Flow name"),
	template: str = typer.Option("standard-6-phase", "--template", "-t", help="Template name"),
	description: Optional[str] = typer.Option(None, "--description", "-d", help="Description"),
	seed_from: Optional[int] = typer.Option(None, "--seed-from", help="Seed from checkpoint ID"),
	phase_order: str = typer.Option("neurons_first", "--phase-order", help="Phase order: neurons_first or bits_first"),
	ga_generations: int = typer.Option(250, "--ga-gens", help="GA generations"),
	ts_iterations: int = typer.Option(250, "--ts-iters", help="TS iterations"),
	population_size: int = typer.Option(50, "--population", help="Population size"),
	patience: int = typer.Option(10, "--patience", help="Early stopping patience"),
	url: str = typer.Option("https://localhost:3000", "--url", help="Dashboard URL"),
):
	"""Create a new flow."""
	from wnn.ram.experiments.dashboard_client import FlowConfig

	try:
		client = get_client(url)

		# Create flow config
		flow_config = FlowConfig.standard_6_phase(
			name=name,
			phase_order=phase_order,
			ga_generations=ga_generations,
			ts_iterations=ts_iterations,
			population_size=population_size,
			patience=patience,
		)
		flow_config.description = description

		# Create flow
		flow_id = client.create_flow(flow_config, seed_checkpoint_id=seed_from)

		rprint(f"[green]Created flow {flow_id}: {name}[/green]")
		rprint(f"[dim]Template: {template}, Experiments: {len(flow_config.experiments)}[/dim]")

		if seed_from:
			rprint(f"[dim]Seeded from checkpoint: {seed_from}[/dim]")

		rprint(f"\nTo run: [cyan]wnn-exp flow run {flow_id}[/cyan]")

	except ConnectionError as e:
		rprint(f"[red]Error: Could not connect to dashboard at {url}[/red]")
		raise typer.Exit(1)
	except Exception as e:
		rprint(f"[red]Error: {e}[/red]")
		raise typer.Exit(1)


@flow_app.command("run")
def flow_run(
	flow_id: int = typer.Argument(..., help="Flow ID to run"),
	resume_from: Optional[str] = typer.Option(
		None, "--resume-from", "-r",
		help="Resume from phase (1a, 1b, 2a, 2b, 3a, 3b)"
	),
	checkpoint_dir: Optional[Path] = typer.Option(
		None, "--checkpoint-dir", "-c",
		help="Directory for checkpoints (required for resume)"
	),
	context: int = typer.Option(4, "--context", help="Context window size"),
	cluster_type: str = typer.Option("tiered", "--cluster-type", help="Cluster type: tiered or bitwise"),
	bitwise_neurons: int = typer.Option(200, "--bitwise-neurons", help="Neurons per cluster (bitwise mode)"),
	bitwise_bits: int = typer.Option(16, "--bitwise-bits", help="Bits per neuron (bitwise mode)"),
	check_interval: int = typer.Option(10, "--check-interval", help="Patience check every N gens/iters"),
	threshold_delta: float = typer.Option(0.01, "--threshold-delta", help="Progressive threshold delta"),
	threshold_reference: int = typer.Option(100, "--threshold-reference", help="Threshold reference window"),
	url: str = typer.Option("https://localhost:3000", "--url", help="Dashboard URL"),
):
	"""
	Run a flow (execute all experiments in sequence).

	This command fetches the flow configuration from the dashboard and
	executes all experiments. Use --resume-from to continue from a specific phase.

	Examples:
		wnn-exp flow run 5                      # Run flow 5 from start
		wnn-exp flow run 5 --resume-from 3b    # Resume from phase 3b
	"""
	import time
	from wnn.ram.experiments.flow import Flow, FlowConfig, ExperimentConfig
	from wnn.ram.experiments.experiment import ExperimentType, ClusterType
	from wnn.ram.strategies.connectivity.adaptive_cluster import ClusterGenome

	# Phase name to index mapping
	PHASE_MAP = {
		"1a": 0, "1b": 1,
		"2a": 2, "2b": 3,
		"3a": 4, "3b": 5,
	}

	try:
		client = get_client(url)

		# Fetch flow from dashboard
		rprint(f"[dim]Fetching flow {flow_id}...[/dim]")
		flow_data = client.get_flow(flow_id)

		if not flow_data:
			rprint(f"[red]Flow {flow_id} not found[/red]")
			raise typer.Exit(1)

		flow_name = flow_data.get("name", f"Flow {flow_id}")
		config = flow_data.get("config", {})
		experiments = config.get("experiments", [])
		params = config.get("params", {})

		rprint(f"[bold]{flow_name}[/bold]")
		rprint(f"[dim]Experiments: {len(experiments)}[/dim]")

		# Validate resume_from
		resume_idx = None
		if resume_from:
			if resume_from.lower() not in PHASE_MAP:
				rprint(f"[red]Invalid phase: {resume_from}. Valid: 1a, 1b, 2a, 2b, 3a, 3b[/red]")
				raise typer.Exit(1)
			resume_idx = PHASE_MAP[resume_from.lower()]

			if not checkpoint_dir:
				rprint("[red]--checkpoint-dir is required for --resume-from[/red]")
				raise typer.Exit(1)

			if not checkpoint_dir.exists():
				rprint(f"[red]Checkpoint directory not found: {checkpoint_dir}[/red]")
				raise typer.Exit(1)

			rprint(f"[yellow]Resuming from phase {resume_from} (experiment {resume_idx})[/yellow]")

		# Parse tier config from params
		tier_config = params.get("tier_config")
		tier0_only = params.get("optimize_tier0_only", params.get("tier0_only", False))
		patience = params.get("patience", 10)
		fitness_percentile = params.get("fitness_percentile")
		seed = params.get("seed")
		phase_order = params.get("phase_order", "neurons_first")

		# Use context_size from params if CLI default was used
		effective_context = params.get("context_size", context)

		# Load WikiText-2 dataset
		rprint("\n[dim]Loading WikiText-2 dataset...[/dim]")
		from datasets import load_dataset

		try:
			dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
		except Exception:
			dataset = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)

		# Tokenize
		from transformers import GPT2TokenizerFast
		tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

		def tokenize_text(text: str) -> list[int]:
			return tokenizer.encode(text, add_special_tokens=False)

		# Prepare data
		train_text = "\n".join(dataset["train"]["text"])
		eval_text = "\n".join(dataset["validation"]["text"])

		train_tokens = tokenize_text(train_text)[:200_000]
		eval_tokens = tokenize_text(eval_text)[:50_000]

		rprint(f"[dim]  Train: {len(train_tokens):,} tokens, Eval: {len(eval_tokens):,} tokens[/dim]")

		# Parse cluster type
		effective_cluster_type = ClusterType.BITWISE if cluster_type.lower() == "bitwise" else ClusterType.TIERED

		# Create evaluator based on cluster type
		effective_seed = seed or int(time.time() * 1000) % (2**32)

		if effective_cluster_type == ClusterType.BITWISE:
			from wnn.ram.architecture.bitwise_evaluator import BitwiseEvaluator

			evaluator = BitwiseEvaluator(
				train_tokens=train_tokens,
				eval_tokens=eval_tokens,
				vocab_size=tokenizer.vocab_size,
				context_size=effective_context,
				neurons_per_cluster=bitwise_neurons,
				bits_per_neuron=bitwise_bits,
				num_parts=3,
				seed=effective_seed,
			)
			rprint(f"[dim]  Bitwise mode: {evaluator}[/dim]")
		else:
			from wnn.ram.strategies.connectivity.evaluator import CachedEvaluator

			evaluator = CachedEvaluator(
				train_tokens=train_tokens,
				eval_tokens=eval_tokens,
				vocab_size=tokenizer.vocab_size,
				context_size=effective_context,
				token_parts=3,
				rotation_mode="per_iteration",
				seed=effective_seed,
			)

		rprint(f"[dim]  Vocab: {evaluator.vocab_size:,}, Context: {effective_context}[/dim]")

		# Build experiment configs from dashboard data
		exp_configs = []
		for exp_data in experiments:
			raw_type = exp_data.get("experiment_type", "ga")
			exp_type = ExperimentType.GA if raw_type == "ga" else ExperimentType.TS

			exp_config = ExperimentConfig(
				name=exp_data.get("name", "Unnamed"),
				experiment_type=exp_type,
				optimize_bits=exp_data.get("optimize_bits", False),
				optimize_neurons=exp_data.get("optimize_neurons", False),
				optimize_connections=exp_data.get("optimize_connections", False),
				generations=exp_data.get("params", {}).get("generations", 250),
				population_size=exp_data.get("params", {}).get("population_size", 50),
				iterations=exp_data.get("params", {}).get("iterations", 250),
				neighbors_per_iter=exp_data.get("params", {}).get("neighbors_per_iter", 50),
				patience=exp_data.get("params", {}).get("patience", patience),
				tier_config=tier_config,
				optimize_tier0_only=tier0_only,
				fitness_percentile=fitness_percentile,
				seed=seed,
				cluster_type=effective_cluster_type,
				bitwise_neurons_per_cluster=bitwise_neurons,
				bitwise_bits_per_neuron=bitwise_bits,
				check_interval=check_interval,
				threshold_delta=threshold_delta,
				threshold_reference=threshold_reference,
			)
			exp_configs.append(exp_config)

		# Create flow config
		flow_config = FlowConfig(
			name=flow_name,
			experiments=exp_configs,
			description=flow_data.get("description"),
			tier_config=tier_config,
			optimize_tier0_only=tier0_only,
			context_size=effective_context,
			patience=patience,
			fitness_percentile=fitness_percentile,
			seed=seed,
		)

		# Load seed from checkpoint if resuming or if flow has seed_checkpoint_id
		seed_checkpoint_id = flow_data.get("seed_checkpoint_id")
		seed_genome = None
		if seed_checkpoint_id:
			try:
				ckpt = client.get_checkpoint(seed_checkpoint_id)
				if ckpt and ckpt.get("file_path"):
					flow_config.seed_checkpoint_path = ckpt["file_path"]
					rprint(f"[dim]Seed checkpoint: {ckpt.get('name')}[/dim]")
			except Exception as e:
				rprint(f"[yellow]Warning: Could not fetch seed checkpoint: {e}[/yellow]")

		# Create checkpoint directory if needed
		if checkpoint_dir:
			checkpoint_dir.mkdir(parents=True, exist_ok=True)

		# Logger function
		def log_fn(msg: str):
			timestamp = time.strftime("%H:%M:%S")
			rprint(f"[dim]{timestamp}[/dim] | {msg}")

		# Mark flow as started
		try:
			client.flow_started(flow_id)
		except Exception:
			pass

		# Create and run flow
		rprint("\n[bold green]Starting flow execution...[/bold green]\n")

		flow = Flow(
			config=flow_config,
			evaluator=evaluator,
			logger=log_fn,
			checkpoint_dir=checkpoint_dir,
			dashboard_client=client,
		)

		try:
			result = flow.run(resume_from=resume_idx, seed_genome=seed_genome)

			# Success
			rprint(f"\n[bold green]Flow completed![/bold green]")
			rprint(f"  Final CE: {result.final_fitness:.4f}")
			if result.final_accuracy:
				rprint(f"  Final Accuracy: {result.final_accuracy:.2%}")
			rprint(f"  Duration: {result.total_elapsed_seconds:.1f}s")

		except KeyboardInterrupt:
			rprint("\n[yellow]Flow interrupted by user[/yellow]")
			try:
				client.update_flow(flow_id, status="cancelled")
			except Exception:
				pass
			raise typer.Exit(130)

		except Exception as e:
			rprint(f"\n[red]Flow failed: {e}[/red]")
			try:
				client.flow_failed(flow_id, str(e))
			except Exception:
				pass
			raise typer.Exit(1)

	except ConnectionError as e:
		rprint(f"[red]Error: Could not connect to dashboard at {url}[/red]")
		raise typer.Exit(1)
	except typer.Exit:
		raise
	except Exception as e:
		rprint(f"[red]Error: {e}[/red]")
		import traceback
		traceback.print_exc()
		raise typer.Exit(1)


@flow_app.command("delete")
def flow_delete(
	flow_id: int = typer.Argument(..., help="Flow ID"),
	force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
	url: str = typer.Option("https://localhost:3000", "--url", help="Dashboard URL"),
):
	"""Delete a flow."""
	try:
		client = get_client(url)

		if not force:
			flow = client.get_flow(flow_id)
			if not flow:
				rprint(f"[red]Flow {flow_id} not found[/red]")
				raise typer.Exit(1)

			confirm = typer.confirm(f"Delete flow '{flow.get('name')}'?")
			if not confirm:
				rprint("[dim]Cancelled[/dim]")
				raise typer.Exit(0)

		client.delete_flow(flow_id)
		rprint(f"[green]Deleted flow {flow_id}[/green]")

	except ConnectionError as e:
		rprint(f"[red]Error: Could not connect to dashboard at {url}[/red]")
		raise typer.Exit(1)
	except Exception as e:
		rprint(f"[red]Error: {e}[/red]")
		raise typer.Exit(1)


# =============================================================================
# Checkpoint commands
# =============================================================================

@checkpoint_app.command("list")
def checkpoint_list(
	final_only: bool = typer.Option(False, "--final-only", "-f", help="Show only final checkpoints"),
	experiment_id: Optional[int] = typer.Option(None, "--experiment-id", "-e", help="Filter by experiment"),
	limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
	url: str = typer.Option("https://localhost:3000", "--url", help="Dashboard URL"),
):
	"""List checkpoints."""
	try:
		client = get_client(url)
		checkpoints = client.list_checkpoints(
			experiment_id=experiment_id,
			is_final=final_only if final_only else None,
			limit=limit,
		)

		if not checkpoints:
			rprint("[dim]No checkpoints found.[/dim]")
			return

		table = Table(title="Checkpoints")
		table.add_column("ID", style="cyan", width=6)
		table.add_column("Name", style="white")
		table.add_column("Fitness", justify="right", width=10)
		table.add_column("Accuracy", justify="right", width=10)
		table.add_column("Iters", justify="right", width=6)
		table.add_column("Size", justify="right", width=10)
		table.add_column("Final", width=5)
		table.add_column("Refs", justify="right", width=4)

		for ckpt in checkpoints:
			fitness = ckpt.get("final_fitness")
			accuracy = ckpt.get("final_accuracy")
			size = ckpt.get("file_size_bytes")
			is_final = ckpt.get("is_final", False)
			refs = ckpt.get("reference_count", 0)

			table.add_row(
				str(ckpt.get("id", "")),
				ckpt.get("name", ""),
				f"{fitness:.4f}" if fitness else "-",
				f"{accuracy:.2%}" if accuracy else "-",
				str(ckpt.get("iterations_run", "-")),
				format_bytes(size) if size else "-",
				"[green]✓[/green]" if is_final else "",
				str(refs) if refs > 0 else "-",
			)

		console.print(table)

	except ConnectionError as e:
		rprint(f"[red]Error: Could not connect to dashboard at {url}[/red]")
		raise typer.Exit(1)
	except Exception as e:
		rprint(f"[red]Error: {e}[/red]")
		raise typer.Exit(1)


def format_bytes(size: int) -> str:
	"""Format byte size for display."""
	if size < 1024:
		return f"{size} B"
	if size < 1024 * 1024:
		return f"{size / 1024:.1f} KB"
	return f"{size / (1024 * 1024):.1f} MB"


@checkpoint_app.command("show")
def checkpoint_show(
	checkpoint_id: int = typer.Argument(..., help="Checkpoint ID"),
	url: str = typer.Option("https://localhost:3000", "--url", help="Dashboard URL"),
):
	"""Show checkpoint details."""
	try:
		client = get_client(url)
		ckpt = client.get_checkpoint(checkpoint_id)

		if not ckpt:
			rprint(f"[red]Checkpoint {checkpoint_id} not found[/red]")
			raise typer.Exit(1)

		rprint(f"\n[bold]{ckpt.get('name')}[/bold] [dim](ID: {checkpoint_id})[/dim]")

		if ckpt.get("is_final"):
			rprint("[green]Final checkpoint[/green]")

		rprint(f"\nFile: [dim]{ckpt.get('file_path')}[/dim]")
		if ckpt.get("file_size_bytes"):
			rprint(f"Size: {format_bytes(ckpt.get('file_size_bytes'))}")

		rprint(f"\nExperiment ID: {ckpt.get('experiment_id')}")
		rprint(f"Created: {format_datetime(ckpt.get('created_at'))}")

		if ckpt.get("final_fitness"):
			rprint(f"\nFitness (CE): {ckpt.get('final_fitness'):.4f}")
		if ckpt.get("final_accuracy"):
			rprint(f"Accuracy: {ckpt.get('final_accuracy'):.2%}")
		if ckpt.get("iterations_run"):
			rprint(f"Iterations: {ckpt.get('iterations_run')}")

		refs = ckpt.get("reference_count", 0)
		if refs > 0:
			rprint(f"\n[yellow]Referenced by {refs} experiment(s)[/yellow]")

		# Genome stats
		stats = ckpt.get("genome_stats")
		if stats:
			rprint(f"\n[bold]Genome Stats:[/bold]")
			rprint(f"  Clusters: {stats.get('num_clusters', '-'):,}")
			rprint(f"  Neurons: {stats.get('total_neurons', '-'):,}")
			rprint(f"  Connections: {stats.get('total_connections', '-'):,}")
			bits_range = stats.get("bits_range", ["-", "-"])
			neurons_range = stats.get("neurons_range", ["-", "-"])
			rprint(f"  Bits range: {bits_range[0]} - {bits_range[1]}")
			rprint(f"  Neurons range: {neurons_range[0]} - {neurons_range[1]}")

		rprint("")

	except ConnectionError as e:
		rprint(f"[red]Error: Could not connect to dashboard at {url}[/red]")
		raise typer.Exit(1)
	except Exception as e:
		rprint(f"[red]Error: {e}[/red]")
		raise typer.Exit(1)


@checkpoint_app.command("delete")
def checkpoint_delete(
	checkpoint_id: int = typer.Argument(..., help="Checkpoint ID"),
	force: bool = typer.Option(False, "--force", "-f", help="Force delete even if referenced"),
	url: str = typer.Option("https://localhost:3000", "--url", help="Dashboard URL"),
):
	"""Delete a checkpoint."""
	try:
		client = get_client(url)

		# Get checkpoint info first
		ckpt = client.get_checkpoint(checkpoint_id)
		if not ckpt:
			rprint(f"[red]Checkpoint {checkpoint_id} not found[/red]")
			raise typer.Exit(1)

		refs = ckpt.get("reference_count", 0)
		if refs > 0 and not force:
			rprint(f"[yellow]Checkpoint has {refs} reference(s).[/yellow]")
			rprint("Use --force to delete anyway.")
			raise typer.Exit(1)

		confirm = typer.confirm(f"Delete checkpoint '{ckpt.get('name')}'?")
		if not confirm:
			rprint("[dim]Cancelled[/dim]")
			raise typer.Exit(0)

		client.delete_checkpoint(checkpoint_id, force=force)
		rprint(f"[green]Deleted checkpoint {checkpoint_id}[/green]")

	except ConnectionError as e:
		rprint(f"[red]Error: Could not connect to dashboard at {url}[/red]")
		raise typer.Exit(1)
	except Exception as e:
		rprint(f"[red]Error: {e}[/red]")
		raise typer.Exit(1)


@checkpoint_app.command("seed")
def checkpoint_seed(
	checkpoint_id: int = typer.Argument(..., help="Checkpoint ID to seed from"),
	flow_name: str = typer.Option(..., "--flow-name", "-n", help="Name for new flow"),
	template: str = typer.Option("standard-6-phase", "--template", "-t", help="Template name"),
	url: str = typer.Option("https://localhost:3000", "--url", help="Dashboard URL"),
):
	"""Create a new flow seeded from a checkpoint."""
	from wnn.ram.experiments.dashboard_client import FlowConfig

	try:
		client = get_client(url)

		# Verify checkpoint exists
		ckpt = client.get_checkpoint(checkpoint_id)
		if not ckpt:
			rprint(f"[red]Checkpoint {checkpoint_id} not found[/red]")
			raise typer.Exit(1)

		# Create flow config
		flow_config = FlowConfig.standard_6_phase(
			name=flow_name,
		)
		flow_config.description = f"Seeded from checkpoint: {ckpt.get('name')}"

		# Create flow with seed
		flow_id = client.create_flow(flow_config, seed_checkpoint_id=checkpoint_id)

		rprint(f"[green]Created flow {flow_id}: {flow_name}[/green]")
		rprint(f"[dim]Seeded from: {ckpt.get('name')} (CE: {ckpt.get('final_fitness', '-'):.4f})[/dim]")
		rprint(f"\nTo run: [cyan]wnn-exp flow run {flow_id}[/cyan]")

	except ConnectionError as e:
		rprint(f"[red]Error: Could not connect to dashboard at {url}[/red]")
		raise typer.Exit(1)
	except Exception as e:
		rprint(f"[red]Error: {e}[/red]")
		raise typer.Exit(1)


# =============================================================================
# Status command
# =============================================================================

@app.command("status")
def status(
	url: str = typer.Option("https://localhost:3000", "--url", help="Dashboard URL"),
):
	"""Check dashboard connection status."""
	try:
		client = get_client(url)
		if client.ping():
			rprint(f"[green]✓ Dashboard is reachable at {url}[/green]")

			# Show summary
			flows = client.list_flows(limit=100)
			running = sum(1 for f in flows if f.get("status") == "running")
			completed = sum(1 for f in flows if f.get("status") == "completed")
			rprint(f"  Flows: {len(flows)} total ({running} running, {completed} completed)")

			checkpoints = client.list_checkpoints(limit=100)
			final = sum(1 for c in checkpoints if c.get("is_final"))
			rprint(f"  Checkpoints: {len(checkpoints)} total ({final} final)")
		else:
			rprint(f"[red]✗ Dashboard is not reachable at {url}[/red]")
			raise typer.Exit(1)

	except Exception as e:
		rprint(f"[red]✗ Error connecting to dashboard: {e}[/red]")
		raise typer.Exit(1)


def main():
	"""Entry point for the CLI."""
	app()


if __name__ == "__main__":
	main()
