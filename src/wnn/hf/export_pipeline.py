"""
HuggingFace export pipeline for WNN genomes.

Exports a genome from the best_genomes leaderboard to HuggingFace Hub.
Generates a model card with metrics and verification instructions.

Usage:
	wnn-export --best-genome-id 42 --repo lacg/wnn-ids-best-ce
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def export_to_hf(
	genome_data: dict,
	metrics: dict,
	task_type: str,
	repo_id: str,
	context_size: int = 4,
	token: str | None = None,
) -> str:
	"""Export a genome to HuggingFace Hub.

	Args:
		genome_data: Dict with connections, bits_per_neuron, neurons_per_cluster, tiers_json
		metrics: Dict with ce, accuracy, f1_macro, fpr
		task_type: "lm" or "ids"
		repo_id: HuggingFace repo ID (e.g., "lacg/wnn-ids-best-ce")
		context_size: Context window size
		token: HuggingFace API token (or from env/CLI login)

	Returns:
		URL of the uploaded model
	"""
	from wnn.hf import WNNConfig, WNNForCausalLM

	# Parse genome architecture
	connections_str = genome_data.get("connections_json", "")
	connections = [int(c) for c in connections_str.split(",") if c.strip()] if connections_str else []

	tiers_json = genome_data.get("tiers_json", "[]")
	tiers = json.loads(tiers_json) if tiers_json.startswith("[") else []

	bits_per_neuron = []
	neurons_per_cluster = []

	if tiers and isinstance(tiers[0], dict):
		for tier in tiers:
			n = tier.get("neurons", 1)
			b = tier.get("bits", 1)
			c = tier.get("clusters", 1)
			bits_per_neuron.extend([b] * c)
			neurons_per_cluster.extend([n] * c)
	else:
		total_clusters = genome_data.get("total_clusters", 1)
		total_neurons = genome_data.get("total_neurons", total_clusters)
		npg = max(1, total_neurons // total_clusters)
		bits_per_neuron = [16] * total_clusters
		neurons_per_cluster = [npg] * total_clusters

	# Create config
	config = WNNConfig(
		context_size=context_size,
		architecture_type=genome_data.get("architecture_type", "bitwise"),
		bits_per_neuron=bits_per_neuron,
		neurons_per_cluster=neurons_per_cluster,
		num_clusters=len(neurons_per_cluster),
	)

	# Create model
	model = WNNForCausalLM(config)

	# Load connections into model
	if connections:
		neuron_offset = 0
		conn_idx = 0
		for cluster_idx in range(len(neurons_per_cluster)):
			n = neurons_per_cluster[cluster_idx]
			for neuron_idx in range(n):
				global_neuron = neuron_offset + neuron_idx
				b = bits_per_neuron[global_neuron] if global_neuron < len(bits_per_neuron) else 1
				for bit_idx in range(b):
					if conn_idx < len(connections):
						model.connections.data[global_neuron, bit_idx] = connections[conn_idx]
						conn_idx += 1
			neuron_offset += n

	# Generate model card
	model_card = _generate_model_card(repo_id, task_type, metrics, config)

	# Save and push
	with tempfile.TemporaryDirectory() as tmpdir:
		local_dir = Path(tmpdir) / "model"
		model.save_pretrained(str(local_dir))

		# Write model card
		(local_dir / "README.md").write_text(model_card)

		# Push to hub
		model.push_to_hub(repo_id, token=token)

	return f"https://huggingface.co/{repo_id}"


def _generate_model_card(
	repo_id: str,
	task_type: str,
	metrics: dict,
	config,
) -> str:
	"""Generate a HuggingFace model card."""
	task_desc = "Intrusion Detection System (IDS)" if task_type == "ids" else "Language Model"
	dataset = "UNSW-NB15" if task_type == "ids" else "WikiText-2"

	metrics_table = "| Metric | Value |\n|--------|-------|\n"
	metrics_table += f"| Cross-Entropy | {metrics.get('ce', 'N/A'):.4f} |\n"
	metrics_table += f"| Accuracy | {metrics.get('accuracy', 0) * 100:.2f}% |\n"
	if metrics.get("f1_macro") is not None:
		metrics_table += f"| F1-Macro | {metrics['f1_macro'] * 100:.2f}% |\n"
	if metrics.get("fpr") is not None:
		metrics_table += f"| False Positive Rate | {metrics['fpr'] * 100:.2f}% |\n"

	return f"""---
library_name: wnn
tags:
- weightless-neural-network
- ram-neuron
- {task_type}
license: mit
datasets:
- {dataset.lower().replace(' ', '-')}
---

# {repo_id}

A **Weightless Neural Network (WNN)** {task_desc} using RAM-based neurons.

## Architecture

- **Type**: {config.architecture_type}
- **Clusters**: {config.num_clusters}
- **Total Neurons**: {config.total_neurons}
- **Context Size**: {config.context_size}
- **Memory Mode**: {config.memory_mode}

## Results

{metrics_table}

## Verification

Reproduce these results:

```bash
pip install ram-wnn[hf]
wnn-verify {repo_id} --dataset {dataset.lower().replace(' ', '-')}
```

## How It Works

WNN models use RAM-based neurons instead of traditional weighted neurons.
Each neuron has a learned connectivity pattern (which input bits to observe)
and a memory table (mapping addresses to outputs). The architecture is
discovered via genetic algorithm and tabu search optimization.

## Citation

```bibtex
@software{{wnn_ram,
  title = {{Weightless Neural Network RAM Language Model}},
  url = {{https://huggingface.co/{repo_id}}},
}}
```
"""


def main():
	"""CLI entry point for wnn-export."""
	parser = argparse.ArgumentParser(description="Export WNN genome to HuggingFace Hub")
	parser.add_argument("--best-genome-id", type=int, required=True, help="Best genome ID from leaderboard")
	parser.add_argument("--repo", required=True, help="HuggingFace repo ID (e.g., lacg/wnn-ids-best)")
	parser.add_argument("--token", help="HuggingFace API token")
	parser.add_argument("--dashboard-url", default="https://localhost:3000", help="Dashboard URL")
	parser.add_argument("--context-size", type=int, default=4, help="Context window size")
	parser.add_argument("--no-ssl-verify", action="store_true", help="Disable SSL verification")
	args = parser.parse_args()

	# Fetch genome data from dashboard
	from wnn.ram.experiments.dashboard_client import DashboardClient, DashboardClientConfig

	client_config = DashboardClientConfig(
		base_url=args.dashboard_url,
		verify_ssl=not args.no_ssl_verify,
	)
	client = DashboardClient(client_config)

	# Get best genome entry
	entry = client.get_best_genome(args.best_genome_id)
	if not entry:
		print(f"Error: Best genome {args.best_genome_id} not found")
		sys.exit(1)

	# Get full genome data
	data = client.get_best_genome_data(args.best_genome_id)
	if not data:
		print(f"Error: Genome data not available for {args.best_genome_id}")
		sys.exit(1)

	metrics = {
		"ce": entry.get("ce", 0),
		"accuracy": entry.get("accuracy", 0),
		"f1_macro": entry.get("f1_macro"),
		"fpr": entry.get("fpr"),
	}

	print(f"Exporting genome {args.best_genome_id} to {args.repo}...")
	url = export_to_hf(
		genome_data=data,
		metrics=metrics,
		task_type=entry.get("task_type", "lm"),
		repo_id=args.repo,
		context_size=args.context_size,
		token=args.token,
	)
	print(f"Exported to: {url}")

	# Update dashboard with HF info
	try:
		import requests
		resp = requests.post(
			f"{args.dashboard_url}/api/best-genomes/{args.best_genome_id}/export-hf",
			json={"repo_id": args.repo},
			verify=not args.no_ssl_verify,
		)
		if resp.ok:
			print("Dashboard updated with HF repo info")
	except Exception as e:
		print(f"Warning: Failed to update dashboard: {e}")


if __name__ == "__main__":
	main()
