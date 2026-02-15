"""
lm-eval-harness adapter for WNN models.

Makes WNNForCausalLM compatible with the lm-eval-harness evaluation framework.
Usage:
    lm_eval --model hf --model_args pretrained=./exports/wnn-model --tasks wikitext

Since WNNForCausalLM inherits from PreTrainedModel and implements forward()
with proper CausalLMOutput, it works directly with lm-eval-harness's HuggingFace
integration. This module provides additional utilities for standalone evaluation.
"""

from pathlib import Path
from typing import Optional

try:
	import torch
	HAS_TORCH = True
except ImportError:
	HAS_TORCH = False


def evaluate_perplexity(
	model_path: str,
	dataset_name: str = "wikitext",
	dataset_config: str = "wikitext-2-raw-v1",
	split: str = "test",
	context_size: Optional[int] = None,
	batch_size: int = 1,
) -> dict:
	"""
	Evaluate a WNN model's perplexity on a dataset.

	Args:
		model_path: Path to HF model directory or Hub model ID
		dataset_name: HuggingFace dataset name
		dataset_config: Dataset configuration
		split: Dataset split to evaluate on
		context_size: Override context size (uses model config if None)
		batch_size: Batch size for evaluation

	Returns:
		Dict with 'cross_entropy', 'perplexity', 'num_tokens'
	"""
	if not HAS_TORCH:
		raise ImportError("PyTorch required for evaluation. Install with: pip install torch")

	from wnn.hf import WNNForCausalLM

	model = WNNForCausalLM.from_pretrained(model_path)
	return model.evaluate(
		dataset_name=dataset_name,
		dataset_config=dataset_config,
		split=split,
	)


def run_lm_eval(
	model_path: str,
	tasks: str = "wikitext",
	batch_size: int = 1,
	output_path: Optional[str] = None,
) -> dict:
	"""
	Run lm-eval-harness evaluation on a WNN model.

	Requires lm-eval to be installed: pip install lm-eval

	Args:
		model_path: Path to HF model directory or Hub model ID
		tasks: Comma-separated task names
		batch_size: Batch size
		output_path: Optional path to save results JSON

	Returns:
		Evaluation results dict
	"""
	try:
		import lm_eval
	except ImportError:
		raise ImportError(
			"lm-eval-harness required. Install with: pip install lm-eval"
		)

	results = lm_eval.simple_evaluate(
		model="hf",
		model_args=f"pretrained={model_path}",
		tasks=tasks.split(","),
		batch_size=batch_size,
	)

	if output_path:
		import json
		Path(output_path).parent.mkdir(parents=True, exist_ok=True)
		with open(output_path, "w") as f:
			json.dump(results["results"], f, indent=2)

	return results["results"]


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Evaluate WNN models")
	parser.add_argument("model_path", help="Path to HF model directory")
	parser.add_argument("--tasks", default="wikitext", help="Evaluation tasks")
	parser.add_argument("--batch-size", type=int, default=1)
	parser.add_argument("--output", help="Output JSON path")
	parser.add_argument(
		"--mode",
		choices=["perplexity", "lm-eval"],
		default="perplexity",
		help="Evaluation mode",
	)
	args = parser.parse_args()

	if args.mode == "perplexity":
		results = evaluate_perplexity(args.model_path, batch_size=args.batch_size)
		print(f"Cross-Entropy: {results['cross_entropy']:.4f}")
		print(f"Perplexity: {results['perplexity']:.1f}")
		print(f"Tokens evaluated: {results['num_tokens']}")
	else:
		results = run_lm_eval(
			args.model_path,
			tasks=args.tasks,
			batch_size=args.batch_size,
			output_path=args.output,
		)
		for task, metrics in results.items():
			print(f"\n{task}:")
			for k, v in metrics.items():
				if isinstance(v, float):
					print(f"  {k}: {v:.4f}")
				else:
					print(f"  {k}: {v}")
