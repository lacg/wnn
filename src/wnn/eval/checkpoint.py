"""
Checkpoint — Save and load reproducible model checkpoints.

Checkpoint format:
	checkpoint_dir/
	├── config.json          # Architecture + eval_task + metadata (required)
	├── connections.pt       # Connection maps per cluster (required, ~381 KB)
	└── memory.pt            # Trained memory tables (optional, ~58 MB)

The minimal reproducible unit is config.json + connections.pt (~381 KB).
Memory is optional because anyone can retrain from connections + data,
proving the result is reproducible, not just loadable.

Usage:
	# Save
	from wnn.eval import Checkpoint
	Checkpoint.save(
		path="checkpoints/bitwise_best",
		model=model,
		eval_task=task,
		results={"ce": 9.15, "ppl": 9430, "accuracy": 0.066},
		save_memory=False,  # Default: skip memory for small checkpoints
	)

	# Load
	ckpt = Checkpoint.load("checkpoints/bitwise_best")
	model = ckpt.create_model()
	print(ckpt.results)  # {"ce": 9.15, ...}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

from wnn.eval.task import EvalTask


# Bump when checkpoint format changes in incompatible ways
CHECKPOINT_VERSION = "1.0.0"


@dataclass
class Checkpoint:
	"""A loaded checkpoint with all metadata."""

	# Required
	model_type: str
	model_config: dict[str, Any]
	connections: dict[str, torch.Tensor]

	# Evaluation
	eval_task: Optional[EvalTask] = None
	results: dict[str, Any] = field(default_factory=dict)

	# Optional
	memory: Optional[dict[str, torch.Tensor]] = None
	encoder_type: str = "binary"
	metadata: dict[str, Any] = field(default_factory=dict)

	@staticmethod
	def save(
		path: str | Path,
		model: torch.nn.Module,
		eval_task: Optional[EvalTask] = None,
		results: Optional[dict[str, Any]] = None,
		save_memory: bool = False,
		extra_metadata: Optional[dict[str, Any]] = None,
	) -> Path:
		"""Save a model checkpoint.

		Args:
			path: Directory to save checkpoint to.
			model: The model to save (RAMLM or BitwiseRAMLM).
			eval_task: Evaluation task specification.
			results: Evaluation results dict (ce, ppl, accuracy, etc.).
			save_memory: Whether to save memory tables. Default False
				for smaller checkpoints (~381 KB vs ~58 MB).
			extra_metadata: Additional metadata to store.

		Returns:
			Path to the checkpoint directory.
		"""
		path = Path(path)
		path.mkdir(parents=True, exist_ok=True)

		# ── Extract model info ────────────────────────────────────
		model_type = type(model).__name__
		model_config = model.get_config() if hasattr(model, "get_config") else {}

		# Encoder type
		encoder_type = "binary"
		if hasattr(model, "encoder"):
			encoder_type = type(model.encoder).__name__

		# ── Connections ───────────────────────────────────────────
		connections = _extract_connections(model)
		torch.save(connections, path / "connections.pt")

		# ── Memory (optional) ─────────────────────────────────────
		if save_memory:
			memory = _extract_memory(model)
			if memory:
				torch.save(memory, path / "memory.pt")

		# ── Config JSON ───────────────────────────────────────────
		config = {
			"checkpoint_version": CHECKPOINT_VERSION,
			"model_type": model_type,
			"model_config": _make_json_safe(model_config),
			"encoder_type": encoder_type,
			"has_memory": save_memory and bool(_extract_memory(model)),
			"eval_task": eval_task.to_dict() if eval_task else None,
			"results": results or {},
			"metadata": {
				"created_at": datetime.now(timezone.utc).isoformat(),
				"torch_version": torch.__version__,
				**(extra_metadata or {}),
			},
		}
		with open(path / "config.json", "w") as f:
			json.dump(config, f, indent=2)

		return path

	@classmethod
	def load(cls, path: str | Path, device: str = "cpu") -> Checkpoint:
		"""Load a checkpoint from disk.

		Args:
			path: Directory containing the checkpoint.
			device: Device to load tensors onto.

		Returns:
			Checkpoint instance with all loaded data.
		"""
		path = Path(path)

		# ── Config ────────────────────────────────────────────────
		with open(path / "config.json") as f:
			config = json.load(f)

		version = config.get("checkpoint_version", "0.0.0")
		if version != CHECKPOINT_VERSION:
			import warnings
			warnings.warn(
				f"Checkpoint version mismatch: {version} (saved) vs "
				f"{CHECKPOINT_VERSION} (current). Loading anyway.",
				stacklevel=2,
			)

		# ── Connections ───────────────────────────────────────────
		connections = torch.load(
			path / "connections.pt",
			map_location=device,
			weights_only=True,
		)

		# ── Memory (optional) ─────────────────────────────────────
		memory = None
		memory_path = path / "memory.pt"
		if memory_path.exists() and config.get("has_memory", False):
			memory = torch.load(
				memory_path,
				map_location=device,
				weights_only=True,
			)

		# ── Eval task ─────────────────────────────────────────────
		eval_task = None
		if config.get("eval_task"):
			eval_task = EvalTask.from_dict(config["eval_task"])

		return cls(
			model_type=config["model_type"],
			model_config=config.get("model_config", {}),
			connections=connections,
			eval_task=eval_task,
			results=config.get("results", {}),
			memory=memory,
			encoder_type=config.get("encoder_type", "binary"),
			metadata=config.get("metadata", {}),
		)

	def create_model(self, device: str = "cpu") -> torch.nn.Module:
		"""Reconstruct the model from checkpoint data.

		Creates the model architecture from config, loads connections,
		and optionally loads memory. If memory was not saved, the caller
		should retrain using the connections + training data.

		Args:
			device: Device to create model on.

		Returns:
			Reconstructed model (untrained if memory was not saved).
		"""
		model = _create_model_from_config(
			self.model_type,
			self.model_config,
			self.encoder_type,
		)
		_load_connections(model, self.connections)

		if self.memory is not None:
			_load_memory(model, self.memory)

		return model.to(device)

	@property
	def has_memory(self) -> bool:
		"""Whether this checkpoint includes trained memory tables."""
		return self.memory is not None

	@property
	def size_bytes(self) -> dict[str, int]:
		"""Estimated size of each component in bytes."""
		sizes = {}
		conn_bytes = sum(t.nelement() * t.element_size() for t in self.connections.values())
		sizes["connections"] = conn_bytes
		if self.memory is not None:
			mem_bytes = sum(t.nelement() * t.element_size() for t in self.memory.values())
			sizes["memory"] = mem_bytes
		sizes["total"] = sum(sizes.values())
		return sizes

	def summary(self) -> str:
		"""Human-readable summary of the checkpoint."""
		lines = [
			f"Checkpoint: {self.model_type}",
			f"  Encoder: {self.encoder_type}",
			f"  Config: {self.model_config}",
		]
		if self.results:
			ce = self.results.get("ce", "?")
			ppl = self.results.get("ppl", "?")
			acc = self.results.get("accuracy", "?")
			lines.append(f"  Results: CE={ce}, PPL={ppl}, Acc={acc}")
		if self.eval_task:
			lines.append(f"  EvalTask: {self.eval_task.describe()}")
		sizes = self.size_bytes
		lines.append(f"  Connections: {sizes['connections'] / 1024:.1f} KB")
		if self.has_memory:
			lines.append(f"  Memory: {sizes.get('memory', 0) / 1024 / 1024:.1f} MB")
		else:
			lines.append("  Memory: not saved (retrain from connections + data)")
		if self.metadata.get("created_at"):
			lines.append(f"  Created: {self.metadata['created_at']}")
		return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────


def _extract_connections(model: torch.nn.Module) -> dict[str, torch.Tensor]:
	"""Extract connection maps from a model's state dict.

	Connections define which input bits each neuron observes — this IS
	the learned architecture. Keys ending in '.connections' are matched.
	"""
	connections = {}
	state = model.state_dict()
	for key, value in state.items():
		if key.endswith(".connections"):
			connections[key] = value
	if not connections and hasattr(model, "get_connections"):
		connections = model.get_connections()
	return connections


def _extract_memory(model: torch.nn.Module) -> dict[str, torch.Tensor]:
	"""Extract trained memory tables from a model's state dict.

	Only extracts '.memory_words' keys (the bit-packed trained memory).
	Skips connections, caches, and encoder buffers.
	"""
	memory = {}
	state = model.state_dict()
	for key, value in state.items():
		if key.endswith(".memory_words"):
			memory[key] = value
	return memory


def _make_json_safe(obj: Any) -> Any:
	"""Convert non-JSON-serializable values to safe types."""
	if isinstance(obj, dict):
		return {str(k): _make_json_safe(v) for k, v in obj.items()}
	elif isinstance(obj, (list, tuple)):
		return [_make_json_safe(v) for v in obj]
	elif isinstance(obj, torch.Tensor):
		return obj.tolist()
	elif hasattr(obj, "value"):  # Enum
		return obj.value
	elif isinstance(obj, (int, float, str, bool, type(None))):
		return obj
	else:
		return str(obj)


def _create_model_from_config(
	model_type: str,
	model_config: dict,
	encoder_type: str,
) -> torch.nn.Module:
	"""Create model instance from saved config."""
	from wnn.representations import (
		BinaryTokenEncoder,
		GrayCodeTokenEncoder,
		TokenBitEncoderType,
		create_token_bit_encoder,
	)

	# Create encoder
	vocab_size = model_config.get("vocab_size", 50257)
	encoder_map = {
		"binary": TokenBitEncoderType.BINARY,
		"BinaryTokenEncoder": TokenBitEncoderType.BINARY,
		"gray_code": TokenBitEncoderType.GRAY_CODE,
		"GrayCodeTokenEncoder": TokenBitEncoderType.GRAY_CODE,
	}
	enc_type = encoder_map.get(encoder_type, TokenBitEncoderType.BINARY)
	encoder = create_token_bit_encoder(enc_type, vocab_size=vocab_size)

	# Import model classes lazily to avoid circular imports
	if model_type == "BitwiseRAMLM":
		from wnn.ram.core.models.bitwise_ramlm import BitwiseRAMLM
		# Filter config to only valid params
		valid_keys = {
			"vocab_size", "context_size", "neurons_per_cluster",
			"bits_per_neuron", "pad_token_id", "rng", "memory_mode",
			"neuron_sample_rate",
		}
		filtered = {k: v for k, v in model_config.items() if k in valid_keys}
		return BitwiseRAMLM(**filtered, encoder=encoder)

	elif model_type == "RAMLM":
		from wnn.ram.core.models.ramlm import RAMLM
		valid_keys = {
			"vocab_size", "context_size", "neurons_per_cluster",
			"bits_per_neuron", "pad_token_id", "rng", "tiers",
		}
		filtered = {k: v for k, v in model_config.items() if k in valid_keys}
		return RAMLM(**filtered, encoder=encoder)

	else:
		raise ValueError(f"Unknown model type: {model_type}")


def _load_connections(model: torch.nn.Module, connections: dict[str, torch.Tensor]) -> None:
	"""Load connection tensors into a model."""
	state = model.state_dict()
	for key, value in connections.items():
		if key in state:
			state[key] = value
	model.load_state_dict(state, strict=False)


def _load_memory(model: torch.nn.Module, memory: dict[str, torch.Tensor]) -> None:
	"""Load memory tensors into a model."""
	state = model.state_dict()
	for key, value in memory.items():
		if key in state:
			state[key] = value
	model.load_state_dict(state, strict=False)
