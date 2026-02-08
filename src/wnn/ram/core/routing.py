"""
Content-Dependent Routing for RAM WNNs.

Routes inputs to specialized neuron groups based on token identity,
giving input-dependent behavior without full attention.

Architecture:
	RouterRAM: Small RAMClusterLayer that maps input bits -> route index
	RoutedRAMClusterLayer: Multiple expert RAMClusterLayers selected by RouterRAM

This implements a Mixture-of-RAM-Experts pattern where:
1. Router observes the last token's bits (content-dependent)
2. Selects top-k expert groups
3. Each expert processes the full context
4. Expert outputs are averaged

All components use standard RAMClusterLayer (no ad-hoc implementations).

Usage:
	from wnn.ram.core.routing import RouterRAM, RoutedRAMClusterLayer

	layer = RoutedRAMClusterLayer(
		total_input_bits=64,
		num_clusters=50257,
		num_routes=8,
		neurons_per_cluster_per_route=3,
		bits_per_neuron=10,
		top_k_routes=2,
	)

	# Forward: routes input to experts
	probs = layer.forward(input_bits)  # [batch, num_clusters]
"""

from typing import Optional

from torch import Tensor, tensor, long, zeros, float32, stack, arange
from torch import topk as torch_topk

from wnn.ram.core.base import RAMComponent
from wnn.ram.core.RAMClusterLayer import RAMClusterLayer, bits_needed


class RouterRAM(RAMComponent):
	"""Small RAMClusterLayer that maps input bits to a route index.

	The router observes only the last token's bits (content-dependent routing)
	and outputs a probability distribution over routes. This is a standard
	RAMClusterLayer with num_routes clusters.

	Training uses frequency-based heuristics: assign each token pattern
	to the route that best separates its distribution.

	Attributes:
		router_layer: RAMClusterLayer with num_routes clusters.
		num_routes: Number of available routes.
		bits_per_token: Bits used per token (for extracting last token).
	"""

	def __init__(
		self,
		total_input_bits: int,
		num_routes: int = 8,
		router_bits: int = 16,
		rng: Optional[int] = None,
	):
		"""Initialize RouterRAM.

		Args:
			total_input_bits: Total input bits from the full context.
			num_routes: Number of routes (expert groups) to choose from.
			router_bits: Bits each router neuron observes.
			rng: Random seed for reproducible connectivity.
		"""
		super().__init__()
		self.num_routes = num_routes
		self.total_input_bits = total_input_bits

		# Router is a small RAMClusterLayer: maps input bits -> route scores
		# Uses 3 neurons per route for stable voting
		self.router_layer = RAMClusterLayer(
			total_input_bits=total_input_bits,
			num_clusters=num_routes,
			neurons_per_cluster=3,
			bits_per_neuron=min(router_bits, total_input_bits),
			rng=rng,
		)

	def forward(self, input_bits: Tensor) -> Tensor:
		"""Route inputs to expert indices.

		Args:
			input_bits: [batch, total_input_bits] boolean tensor.

		Returns:
			[batch, num_routes] route scores.
		"""
		return self.router_layer.forward(input_bits)

	def route(self, input_bits: Tensor) -> Tensor:
		"""Get route indices (argmax of router scores).

		Args:
			input_bits: [batch, total_input_bits] boolean tensor.

		Returns:
			[batch] route indices (0 to num_routes-1).
		"""
		scores = self.forward(input_bits)
		return scores.argmax(dim=-1)

	def train_routing(
		self,
		input_bits: Tensor,
		targets: Tensor,
		num_routes: Optional[int] = None,
	) -> dict:
		"""Train routing via frequency-based assignment.

		Assigns each unique context pattern to a route based on
		target token frequency distribution, aiming for balanced routes
		with specialized token coverage.

		Strategy: Hash-based round-robin assignment. The last token's
		binary representation is mapped to a route index. Then the
		router memory is trained to reproduce this assignment.

		Args:
			input_bits: [N, total_input_bits] training contexts.
			targets: [N] target token IDs.
			num_routes: Override number of routes (default: self.num_routes).

		Returns:
			Training stats dict.
		"""
		n_routes = num_routes or self.num_routes

		# Simple deterministic assignment: hash last token bits to route
		# Extract last token's bits (last bits_per_token bits from input)
		bits_per_token = self.total_input_bits // max(1, self.total_input_bits // 16)

		# Convert target tokens to route assignments via modulo
		route_assignments = targets % n_routes  # [N]

		# Train router: for each example, TRUE for assigned route, FALSE for others
		total_modified = 0
		n = input_bits.shape[0]

		for route_idx in range(n_routes):
			mask = (route_assignments == route_idx)
			if mask.sum() == 0:
				continue

			route_bits = input_bits[mask]

			# Train TRUE for this route
			true_clusters = tensor([route_idx], dtype=long).expand(route_bits.shape[0])
			# Train FALSE for other routes
			other_routes = [r for r in range(n_routes) if r != route_idx]
			if other_routes:
				false_clusters = tensor([other_routes], dtype=long).expand(route_bits.shape[0], -1)
			else:
				false_clusters = None

			modified = self.router_layer.train_multi_examples(
				route_bits, true_clusters, false_clusters,
			)
			total_modified += modified

		return {"modified": total_modified, "examples": n}


class RoutedRAMClusterLayer(RAMComponent):
	"""Multiple expert RAMClusterLayers selected by RouterRAM.

	Each expert is a standard RAMClusterLayer with fewer neurons per cluster.
	The router selects top-k experts for each input, and their outputs
	are averaged.

	Architecture:
		Router: RAMClusterLayer (num_routes clusters, 3 neurons each)
		Experts: num_routes × RAMClusterLayer (num_clusters clusters each)

	This provides input-dependent processing without full attention:
	different token patterns activate different expert groups.

	Attributes:
		router: RouterRAM for route selection.
		experts: List of RAMClusterLayer expert models.
		num_routes: Number of expert groups.
		top_k_routes: Number of experts to use per input.
	"""

	def __init__(
		self,
		total_input_bits: int,
		num_clusters: int,
		num_routes: int = 8,
		neurons_per_cluster_per_route: int = 3,
		bits_per_neuron: int = 10,
		top_k_routes: int = 2,
		router_bits: int = 16,
		rng: Optional[int] = None,
	):
		"""Initialize RoutedRAMClusterLayer.

		Args:
			total_input_bits: Total input dimension (context_size * bits_per_token).
			num_clusters: Number of output clusters (e.g., vocab_size).
			num_routes: Number of expert groups.
			neurons_per_cluster_per_route: Neurons per cluster per expert.
			bits_per_neuron: Bits each neuron observes.
			top_k_routes: Number of experts to activate per input.
			router_bits: Bits for router neurons.
			rng: Random seed.
		"""
		super().__init__()
		self.total_input_bits = total_input_bits
		self.num_clusters = num_clusters
		self.num_routes = num_routes
		self.top_k_routes = min(top_k_routes, num_routes)
		self.neurons_per_cluster = neurons_per_cluster_per_route
		self.bits_per_neuron = bits_per_neuron

		# Create router
		self.router = RouterRAM(
			total_input_bits=total_input_bits,
			num_routes=num_routes,
			router_bits=router_bits,
			rng=rng,
		)

		# Create expert RAMClusterLayers
		self.experts: list[RAMClusterLayer] = []
		for i in range(num_routes):
			expert = RAMClusterLayer(
				total_input_bits=total_input_bits,
				num_clusters=num_clusters,
				neurons_per_cluster=neurons_per_cluster_per_route,
				bits_per_neuron=bits_per_neuron,
				rng=(rng + i + 1) if rng is not None else None,
			)
			self.experts.append(expert)

		# Cached merged arrays for single-dispatch Metal evaluation
		self._merged_connections = None
		self._merged_memory = None
		self._merged_dirty = True

	@property
	def total_neurons(self) -> int:
		"""Total neurons across all experts + router."""
		expert_neurons = sum(e.total_neurons for e in self.experts)
		router_neurons = self.router.router_layer.total_neurons
		return expert_neurons + router_neurons

	def forward(self, input_bits: Tensor) -> Tensor:
		"""Forward pass with routing.

		1. Router selects top-k routes per input
		2. All experts evaluated in a single Metal GPU dispatch
		3. Selected expert scores are averaged

		Args:
			input_bits: [batch, total_input_bits] boolean tensor.

		Returns:
			[batch, num_clusters] probability scores.
		"""
		batch_size = input_bits.shape[0]

		# Get router scores and select top-k routes
		router_scores = self.router.forward(input_bits)  # [batch, num_routes]
		_, top_routes = torch_topk(router_scores, self.top_k_routes, dim=-1)  # [batch, top_k]

		# Try merged single-dispatch evaluation (all experts at once)
		all_expert_scores = self._forward_merged(input_bits)

		if all_expert_scores is None:
			# Fallback: sequential expert evaluation
			all_expert_scores = stack([
				expert.forward(input_bits) for expert in self.experts
			], dim=1)  # [batch, num_routes, num_clusters]

		# Gather selected expert outputs
		# top_routes: [batch, top_k] -> expand to [batch, top_k, num_clusters]
		top_routes_expanded = top_routes.unsqueeze(-1).expand(-1, -1, self.num_clusters)
		selected_scores = all_expert_scores.gather(1, top_routes_expanded)  # [batch, top_k, num_clusters]

		# Average selected expert scores
		return selected_scores.mean(dim=1)  # [batch, num_clusters]

	def _rebuild_merged_cache(self):
		"""Rebuild cached merged arrays from all experts' data."""
		import numpy as np
		connections_list = [e.memory.connections.flatten().numpy() for e in self.experts]
		memory_list = [e.memory.memory_words.flatten().numpy() for e in self.experts]
		self._merged_connections = np.concatenate(connections_list)
		self._merged_memory = np.concatenate(memory_list)
		self._merged_dirty = False

	def _forward_merged(self, input_bits: Tensor) -> Optional[Tensor]:
		"""Evaluate all experts in a single Metal GPU dispatch.

		Concatenates all experts' connections and memory into one virtual
		layer with num_routes * num_clusters clusters, dispatches once,
		then reshapes to [batch, num_routes, num_clusters].

		Uses cached arrays (rebuilt only after training).
		Returns None if Metal is not available (caller falls back to sequential).
		"""
		try:
			import ram_accelerator
		except ImportError:
			return None

		if not ram_accelerator.ramlm_metal_available():
			return None

		from torch import from_numpy, long as torch_long
		import numpy as np

		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		batch_size = input_bits.shape[0]
		expert0 = self.experts[0]

		# Use cached merged arrays (only rebuild after training)
		if self._merged_dirty:
			self._rebuild_merged_cache()

		merged_num_clusters = self.num_clusters * self.num_routes
		merged_total_neurons = merged_num_clusters * self.neurons_per_cluster

		input_bits_np = input_bits.flatten().to(dtype=torch_long).numpy().astype(np.uint8)

		# Single Metal dispatch for all experts
		scores_flat = ram_accelerator.ramlm_forward_batch_metal_cached(
			input_bits_np,
			self._merged_connections,
			self._merged_memory,
			batch_size,
			expert0.memory.total_input_bits,
			merged_total_neurons,
			expert0.memory.n_bits_per_neuron,
			self.neurons_per_cluster,
			merged_num_clusters,
			expert0.memory.words_per_neuron,
		)

		# Reshape: [batch, routes * clusters] -> [batch, routes, clusters]
		scores = from_numpy(np.array(scores_flat, dtype=np.float32))
		return scores.view(batch_size, self.num_routes, self.num_clusters)

	def train_experts(
		self,
		input_bits: Tensor,
		targets: Tensor,
		false_clusters: Optional[Tensor] = None,
		allow_override: bool = False,
		extra_training: bool = True,
	) -> dict:
		"""Train experts based on router assignments.

		Two-phase training:
		1. Train router via frequency-based heuristics
		2. Train each expert on its assigned subset of data

		If extra_training is True, also trains ALL experts on ALL data
		for robustness (useful early when router isn't accurate yet).
		Disable after router converges to preserve specialization.

		Args:
			input_bits: [N, total_input_bits] training input bits.
			targets: [N] target cluster indices.
			false_clusters: Negative training clusters. Can be:
				- [k] shared negatives (memory-efficient, expanded per-subset)
				- [N, k] per-example negatives (backward-compatible)
			allow_override: Whether to override existing memory cells.
			extra_training: Train all experts on all data for robustness.

		Returns:
			Training stats dict with per-expert breakdown.
		"""
		n = input_bits.shape[0]

		# Normalize false_clusters: if 1D, keep as base for efficient expansion
		shared_negatives = None
		if false_clusters is not None and false_clusters.ndim == 1:
			shared_negatives = false_clusters  # [k] base negatives
			false_clusters = None  # Don't use the 2D path

		# Phase 1: Train router
		router_stats = self.router.train_routing(input_bits, targets)

		# Phase 2: Get route assignments
		route_assignments = self.router.route(input_bits)  # [N]

		# Train each expert on its assigned data
		expert_stats = []
		for route_idx in range(self.num_routes):
			mask = (route_assignments == route_idx)
			count = mask.sum().item()
			if count == 0:
				expert_stats.append({"route": route_idx, "examples": 0, "modified": 0})
				continue

			expert_bits = input_bits[mask]
			expert_targets = targets[mask]

			if shared_negatives is not None:
				# Expand shared negatives only for this subset
				expert_false = shared_negatives.unsqueeze(0).expand(count, -1).contiguous()
				modified = self.experts[route_idx].train_multi_examples(
					expert_bits, expert_targets, expert_false,
					allow_override=allow_override,
				)
			elif false_clusters is not None:
				expert_false = false_clusters[mask]
				modified = self.experts[route_idx].train_multi_examples(
					expert_bits, expert_targets, expert_false,
					allow_override=allow_override,
				)
			else:
				# Train TRUE only via train_batch (one example at a time)
				modified = 0
				for j in range(expert_bits.shape[0]):
					m = self.experts[route_idx].train_batch(
						expert_bits[j:j+1],
						expert_targets[j:j+1],
						false_clusters=None,
						allow_override=allow_override,
					)
					modified += m

			expert_stats.append({
				"route": route_idx,
				"examples": count,
				"modified": modified,
			})

		# Optionally train ALL experts on all data for robustness.
		# Process in batches to avoid materializing [N, k] all at once.
		total_extra = 0
		if extra_training:
			neg = shared_negatives if shared_negatives is not None else false_clusters
			if neg is not None:
				extra_batch_size = 50000
				for route_idx in range(self.num_routes):
					for start in range(0, n, extra_batch_size):
						end = min(start + extra_batch_size, n)
						batch_bits = input_bits[start:end]
						batch_targets = targets[start:end]
						if neg.ndim == 1:
							batch_false = neg.unsqueeze(0).expand(end - start, -1).contiguous()
						else:
							batch_false = neg[start:end]
						modified = self.experts[route_idx].train_multi_examples(
							batch_bits, batch_targets, batch_false,
							allow_override=allow_override,
						)
						total_extra += modified

		# Invalidate merged cache (memory changed)
		self._merged_dirty = True

		return {
			"router": router_stats,
			"experts": expert_stats,
			"total_examples": n,
			"extra_training_modified": total_extra,
		}

	def reset_memory(self) -> None:
		"""Reset all memory cells across router and experts."""
		self.router.router_layer.reset_memory()
		for expert in self.experts:
			expert.reset_memory()
		self._merged_dirty = True

	@property
	def connections(self) -> Tensor:
		"""Get all connections (expert 0 connections for compatibility)."""
		return self.experts[0].connections

	@connections.setter
	def connections(self, value: Tensor) -> None:
		"""Set connections on expert 0 (for compatibility)."""
		self.experts[0].connections = value
