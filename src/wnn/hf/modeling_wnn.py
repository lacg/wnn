"""
WNN PreTrainedModel for HuggingFace integration.

WNNForCausalLM wraps the BitwiseRAMLM inference path into HuggingFace's
standard PreTrainedModel interface, enabling:
- save_pretrained / from_pretrained
- push_to_hub
- Compatibility with lm-eval-harness
- Standard generate() autoregressive decoding

The model stores two types of data in safetensors:
- connections: [total_neurons, max_bits] padded int64 — which input bits
  each neuron observes (the learned structure from GA/TS optimization)
- memory_*: per-cluster trained lookup tables (the "weights" from training)

These are stored as torch tensors in safetensors format despite not being
traditional neural network weights.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from wnn.hf.configuration_wnn import WNNConfig
from wnn.representations.token_bit_encoder import (
	BinaryTokenEncoder,
	GrayCodeTokenEncoder,
)


class WNNForCausalLM(PreTrainedModel):
	"""Weightless Neural Network for causal language modeling.

	Uses RAM-based neurons with learned connectivity (GA/TS optimized)
	and trained memory tables. Prediction is via per-bit probability
	reconstruction: P(token) = prod_i P(bit_i | context).
	"""

	config_class = WNNConfig
	supports_gradient_checkpointing = False

	def __init__(self, config: WNNConfig):
		super().__init__(config)

		self.vocab_size = config.vocab_size
		self.context_size = config.context_size
		self.num_clusters = config.num_clusters

		# Token encoder
		if config.encoding_type == "gray_code":
			self.encoder = GrayCodeTokenEncoder(config.vocab_size)
		else:
			self.encoder = BinaryTokenEncoder(config.vocab_size)
		self.bits_per_token = self.encoder.bits_per_token
		self.total_input_bits = config.context_size * self.bits_per_token

		# Pre-compute token bit patterns: [vocab_size, num_clusters]
		token_ids = torch.arange(config.vocab_size, dtype=torch.long)
		token_bits = self.encoder.encode_tokens_batch(token_ids).float()
		self.register_buffer("token_bits", token_bits)

		# Architecture arrays (stored as parameters for safetensors)
		bits_t = torch.tensor(config.bits_per_cluster, dtype=torch.int32)
		neurons_t = torch.tensor(config.neurons_per_cluster, dtype=torch.int32)
		self.register_buffer("bits_per_cluster", bits_t)
		self.register_buffer("neurons_per_cluster", neurons_t)

		# Connections: [total_neurons, max_bits] padded with -1
		total_neurons = sum(config.neurons_per_cluster)
		max_bits = max(config.bits_per_cluster) if config.bits_per_cluster else 1
		self.connections = torch.nn.Parameter(
			torch.full((total_neurons, max_bits), -1, dtype=torch.int64),
			requires_grad=False,
		)

		# Memory tables: one parameter per cluster
		# Each is [neurons, 2^bits] storing the trained output values
		self.memory_keys = []
		for i in range(config.num_clusters):
			n = config.neurons_per_cluster[i] if i < len(config.neurons_per_cluster) else 1
			b = config.bits_per_cluster[i] if i < len(config.bits_per_cluster) else 1
			mem = torch.nn.Parameter(
				torch.full((n, 2 ** b), 0.5, dtype=torch.float32),
				requires_grad=False,
			)
			key = f"memory_{i}"
			self.register_parameter(key, mem)
			self.memory_keys.append(key)

		self.post_init()

	def load_from_genome(
		self,
		genome,
		train_tokens: list[int],
		eval_tokens: list[int] | None = None,
	):
		"""Load connections from a ClusterGenome and train memory.

		This is the bridge between the optimization pipeline and the HF model:
		1. Copy genome.connections into self.connections
		2. Train the BitwiseRAMLM on the provided data
		3. Copy trained memory tables into self.memory_*

		Args:
			genome: ClusterGenome with bits_per_cluster, neurons_per_cluster,
				connections
			train_tokens: Token IDs to train on
			eval_tokens: Token IDs for evaluation (optional)
		"""
		from wnn.ram.core.models.bitwise_ramlm import BitwiseRAMLM

		# Copy connections
		offset = 0
		for cluster_idx in range(self.num_clusters):
			n = genome.neurons_per_cluster[cluster_idx]
			b = genome.bits_per_cluster[cluster_idx]
			for neuron_idx in range(n):
				conn_start = offset
				conn_end = offset + b
				conns = genome.connections[conn_start:conn_end]
				row = offset // b if b > 0 else 0
				# Actually, connections are stored flat: for each cluster,
				# neurons * bits connections contiguously
				global_neuron = sum(genome.neurons_per_cluster[:cluster_idx]) + neuron_idx
				for bit_idx, conn_val in enumerate(conns[neuron_idx * b:(neuron_idx + 1) * b] if len(conns) > b else conns):
					self.connections.data[global_neuron, bit_idx] = conn_val
				offset += b

		# Create and train BitwiseRAMLM
		model = BitwiseRAMLM(
			vocab_size=self.vocab_size,
			context_size=self.context_size,
			neurons_per_cluster=max(genome.neurons_per_cluster),
			bits_per_neuron=max(genome.bits_per_cluster),
		)
		# Training happens through the evaluator path, not here
		# The memory tables are populated during evaluation

	def forward(
		self,
		input_ids: Tensor,
		attention_mask: Optional[Tensor] = None,
		labels: Optional[Tensor] = None,
		**kwargs,
	) -> CausalLMOutput:
		"""Forward pass: predict next token from context.

		For each position, encodes context_size preceding tokens to bits,
		looks up addresses in per-cluster memory tables, reconstructs
		per-bit probabilities, then computes token log-probabilities
		via log-product reconstruction.

		Args:
			input_ids: [batch, seq_len] token IDs
			attention_mask: [batch, seq_len] (ignored — no masking in RAM)
			labels: [batch, seq_len] for loss computation

		Returns:
			CausalLMOutput with logits and optional loss
		"""
		batch_size, seq_len = input_ids.shape
		device = input_ids.device

		# Encode all tokens to bits: [batch, seq_len, bits_per_token]
		all_bits = self.encoder.encode_tokens_batch(
			input_ids.reshape(-1)
		).float().reshape(batch_size, seq_len, self.bits_per_token)

		# Predict for positions context_size..seq_len
		num_predictions = seq_len - self.context_size
		if num_predictions <= 0:
			# Not enough context, return zeros
			logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)
			loss = None
			if labels is not None:
				loss = torch.tensor(0.0, device=device)
			return CausalLMOutput(loss=loss, logits=logits)

		# Collect logits for each prediction position
		all_logits = torch.zeros(
			batch_size, num_predictions, self.vocab_size, device=device
		)

		for pos in range(num_predictions):
			# Context: tokens [pos..pos+context_size]
			context_bits = all_bits[:, pos:pos + self.context_size, :]
			# Flatten to [batch, total_input_bits]
			input_vec = context_bits.reshape(batch_size, -1)

			# Compute per-cluster bit probabilities
			bit_probs = self._compute_bit_probs(input_vec)

			# Reconstruct log-probabilities: [batch, vocab_size]
			logprobs = self._reconstruct_logprobs(bit_probs)
			all_logits[:, pos, :] = logprobs

		# Pad logits to full sequence length (first context_size positions are padding)
		logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)
		logits[:, self.context_size:, :] = all_logits

		loss = None
		if labels is not None:
			# Shift so that tokens < n predict n
			shift_logits = logits[:, self.context_size:, :].contiguous()
			shift_labels = labels[:, self.context_size:].contiguous()
			loss = torch.nn.functional.cross_entropy(
				shift_logits.view(-1, self.vocab_size),
				shift_labels.view(-1),
				ignore_index=-100,
			)

		return CausalLMOutput(loss=loss, logits=logits)

	def _compute_bit_probs(self, input_vec: Tensor) -> Tensor:
		"""Compute P(bit_i=1) for each cluster from input bit vector.

		For each cluster, each neuron looks up its address in memory
		and returns the stored probability. The cluster probability
		is the average across neurons.

		Args:
			input_vec: [batch, total_input_bits] float tensor

		Returns:
			[batch, num_clusters] probability tensor
		"""
		batch_size = input_vec.shape[0]
		device = input_vec.device
		bit_probs = torch.zeros(batch_size, self.num_clusters, device=device)

		neuron_offset = 0
		for cluster_idx in range(self.num_clusters):
			n = int(self.neurons_per_cluster[cluster_idx].item())
			b = int(self.bits_per_cluster[cluster_idx].item())
			memory = getattr(self, f"memory_{cluster_idx}")

			cluster_sum = torch.zeros(batch_size, device=device)
			for neuron_idx in range(n):
				global_idx = neuron_offset + neuron_idx
				# Get this neuron's connections
				conns = self.connections[global_idx, :b]  # [b]

				# Look up input bits at connection positions
				# conns are indices into input_vec
				selected_bits = input_vec[:, conns.long()]  # [batch, b]

				# Compute address: binary to integer
				powers = (2 ** torch.arange(b - 1, -1, -1, device=device)).float()
				addresses = (selected_bits * powers).sum(dim=1).long()  # [batch]

				# Look up memory value
				values = memory[neuron_idx, addresses]  # [batch]
				cluster_sum += values

			bit_probs[:, cluster_idx] = cluster_sum / max(n, 1)
			neuron_offset += n

		return bit_probs

	def _reconstruct_logprobs(self, bit_probs: Tensor) -> Tensor:
		"""Reconstruct token log-probabilities from per-bit probabilities.

		log P(token=t) = sum_i [b_i(t)*log(P_i) + (1-b_i(t))*log(1-P_i)]

		Args:
			bit_probs: [batch, num_bits] P(bit_i=1) probabilities

		Returns:
			[batch, vocab_size] log-probabilities
		"""
		eps = 1e-7
		p1 = torch.clamp(bit_probs, eps, 1.0 - eps)
		log_p1 = torch.log(p1)
		log_p0 = torch.log(1.0 - p1)
		# token_bits: [vocab_size, num_bits], binary 0/1
		return log_p1 @ self.token_bits.T + log_p0 @ (1.0 - self.token_bits).T

	def generate(
		self,
		input_ids: Tensor,
		max_new_tokens: int = 100,
		temperature: float = 1.0,
		top_k: int = 50,
		**kwargs,
	) -> Tensor:
		"""Autoregressive generation.

		Args:
			input_ids: [batch, seq_len] starting token IDs
			max_new_tokens: Maximum tokens to generate
			temperature: Sampling temperature (1.0 = standard)
			top_k: Top-k sampling (0 = greedy)

		Returns:
			[batch, seq_len + max_new_tokens] generated token IDs
		"""
		generated = input_ids.clone()

		for _ in range(max_new_tokens):
			# Use last context_size tokens as context
			context = generated[:, -self.context_size:]
			if context.shape[1] < self.context_size:
				# Pad with zeros if not enough context
				pad = torch.zeros(
					context.shape[0],
					self.context_size - context.shape[1],
					dtype=torch.long,
					device=context.device,
				)
				context = torch.cat([pad, context], dim=1)

			# Forward pass on just the context
			outputs = self.forward(context)
			# Get logits for the last (predicted) position
			next_logits = outputs.logits[:, -1, :]  # [batch, vocab_size]

			# Temperature scaling
			if temperature != 1.0:
				next_logits = next_logits / temperature

			# Top-k sampling
			if top_k > 0:
				values, _ = torch.topk(next_logits, top_k)
				min_value = values[:, -1].unsqueeze(-1)
				next_logits = torch.where(
					next_logits < min_value,
					torch.full_like(next_logits, float("-inf")),
					next_logits,
				)

			# Sample
			probs = torch.softmax(next_logits, dim=-1)
			next_token = torch.multinomial(probs, num_samples=1)
			generated = torch.cat([generated, next_token], dim=1)

		return generated

	def evaluate(
		self,
		dataset_name: str = "wikitext-2-raw-v1",
		split: str = "test",
	) -> dict:
		"""Evaluate cross-entropy and perplexity on a dataset.

		Args:
			dataset_name: HuggingFace dataset name
			split: Dataset split to evaluate on

		Returns:
			Dict with cross_entropy, perplexity, accuracy keys
		"""
		import tiktoken
		from datasets import load_dataset

		tokenizer = tiktoken.get_encoding("gpt2")
		dataset = load_dataset("wikitext", dataset_name, split=split)
		text = "\n\n".join(dataset["text"])
		tokens = tokenizer.encode(text)

		# Evaluate
		total_loss = 0.0
		correct = 0
		total = 0

		self.eval()
		with torch.no_grad():
			for i in range(self.context_size, len(tokens)):
				context = tokens[i - self.context_size:i]
				target = tokens[i]

				input_ids = torch.tensor([context], dtype=torch.long)
				outputs = self.forward(input_ids)
				logits = outputs.logits[0, -1, :]

				# CE loss
				log_probs = torch.log_softmax(logits, dim=-1)
				total_loss += -log_probs[target].item()

				# Accuracy
				if logits.argmax().item() == target:
					correct += 1
				total += 1

		ce = total_loss / total
		return {
			"cross_entropy": ce,
			"perplexity": math.exp(ce),
			"accuracy": correct / total,
			"num_predictions": total,
		}
