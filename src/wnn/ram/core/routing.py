"""
Content-Dependent Routing for RAM WNNs.

Routes inputs to specialized neuron groups based on input context,
giving input-dependent behavior without full attention.

Architecture:
	RouterRAM: Small RAMClusterLayer that maps input bits -> route index
	RoutedRAMClusterLayer: Multiple expert RAMClusterLayers selected by RouterRAM

This implements a Mixture-of-RAM-Experts pattern where:
1. Router observes the input context bits (content-dependent)
2. Selects top-k expert groups
3. Each expert processes the full context
4. Expert outputs are averaged

Routing strategies (RoutingStrategy enum):
	CONTEXT_HASH: Hash full context bits → route (balanced, context-agnostic)
	LAST_TOKEN: Last token identity → route (content-dependent)
	DISTRIBUTIONAL: K-means on last-token→target co-occurrence → route (semantic)

All components use standard RAMClusterLayer (no ad-hoc implementations).

Usage:
	from wnn.ram.core.routing import RoutedRAMClusterLayer, RoutingStrategy

	layer = RoutedRAMClusterLayer(
		total_input_bits=64,
		num_clusters=50257,
		num_routes=8,
		neurons_per_cluster_per_route=3,
		bits_per_neuron=10,
		top_k_routes=2,
		bits_per_token=16,
		routing_strategy=RoutingStrategy.LAST_TOKEN,
	)

	# Forward: routes input to experts
	probs = layer.forward(input_bits)  # [batch, num_clusters]
"""

from enum import Enum, auto
from typing import Optional
from collections import defaultdict, Counter as PyCounter

from torch import Tensor, tensor, long, zeros, float32, stack, arange
from torch import topk as torch_topk

from wnn.ram.core.base import RAMComponent
from wnn.ram.core.RAMClusterLayer import RAMClusterLayer, bits_needed


class RoutingStrategy(Enum):
	"""Strategy for assigning training examples to routes.

	Routes must be a function of OBSERVABLE input features (not targets),
	so the router can learn to reproduce the assignment at inference time.
	"""
	CONTEXT_HASH = auto()    # Hash full context bits → route (balanced)
	LAST_TOKEN = auto()      # Last token identity → route (content-dependent)
	DISTRIBUTIONAL = auto()  # K-means on co-occurrence → route (semantic)


class RouterRAM(RAMComponent):
	"""Small RAMClusterLayer that maps input bits to a route index.

	The router observes the input context bits and outputs a probability
	distribution over routes. This is a standard RAMClusterLayer with
	num_routes clusters.

	Training assigns routes using the selected RoutingStrategy, then
	trains the router to reproduce that assignment from input bits.

	Attributes:
		router_layer: RAMClusterLayer with num_routes clusters.
		num_routes: Number of available routes.
		bits_per_token: Bits used per token (for extracting last token).
		strategy: Routing strategy for assignment.
	"""

	def __init__(
		self,
		total_input_bits: int,
		num_routes: int = 8,
		router_bits: int = 16,
		bits_per_token: int = 16,
		routing_strategy: RoutingStrategy = RoutingStrategy.LAST_TOKEN,
		rng: Optional[int] = None,
	):
		"""Initialize RouterRAM.

		Args:
			total_input_bits: Total input bits from the full context.
			num_routes: Number of routes (expert groups) to choose from.
			router_bits: Bits each router neuron observes.
			bits_per_token: Bits per token (for extracting last token).
			routing_strategy: How to assign examples to routes.
			rng: Random seed for reproducible connectivity.
		"""
		super().__init__()
		self.num_routes = num_routes
		self.total_input_bits = total_input_bits
		self.bits_per_token = bits_per_token
		self.strategy = routing_strategy
		# Token-to-route mapping, populated during training
		self._token_route_map: Optional[dict[int, int]] = None
		self._token_route_table: Optional[Tensor] = None

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

	def deterministic_route(self, input_bits: Tensor) -> Tensor:
		"""Compute route deterministically (no learned router).

		Uses the same assignment function used during training, computed
		directly from input bits. Always correct — no degeneracy.

		Args:
			input_bits: [batch, total_input_bits] boolean tensor.

		Returns:
			[batch] route indices (0 to num_routes-1).
		"""
		if self.strategy == RoutingStrategy.CONTEXT_HASH:
			return input_bits.sum(dim=1).long() % self.num_routes

		# LAST_TOKEN and DISTRIBUTIONAL both use the cached table
		last_ids = self._extract_last_token_ids(input_bits)
		if self._token_route_map is not None:
			if self._token_route_table is not None:
				return self._token_route_table[
					last_ids.clamp(max=len(self._token_route_table) - 1)
				]
			import numpy as np
			route_arr = np.array(
				[self._token_route_map.get(int(t), 0) for t in last_ids.numpy()],
				dtype=np.int64,
			)
			return tensor(route_arr, dtype=long)

		# Fallback: modulo assignment (before training)
		return last_ids % self.num_routes

	def _extract_last_token_ids(self, input_bits: Tensor) -> Tensor:
		"""Extract last token IDs from input bits (LSB-first encoding).

		Args:
			input_bits: [N, total_input_bits] boolean tensor.

		Returns:
			[N] tensor of last token IDs.
		"""
		last_bits = input_bits[:, -self.bits_per_token:]  # [N, bits_per_token]
		powers = (2 ** arange(self.bits_per_token)).to(last_bits.device)
		return (last_bits.long() * powers).sum(dim=1)

	def _assign_context_hash(self, input_bits: Tensor, n_routes: int) -> Tensor:
		"""Assign routes by hashing full context bits."""
		return input_bits.sum(dim=1).long() % n_routes

	def _assign_last_token(self, input_bits: Tensor, n_routes: int) -> Tensor:
		"""Assign routes by last token identity (balanced by frequency)."""
		last_ids = self._extract_last_token_ids(input_bits)

		if self._token_route_map is None:
			# Count frequency per unique last-token
			unique_ids, inverse, counts = last_ids.unique(
				return_inverse=True, return_counts=True,
			)

			# Sort tokens by frequency (descending) and round-robin to routes
			sorted_idx = counts.argsort(descending=True)
			table = zeros(unique_ids.max().item() + 1, dtype=long)
			for rank, idx in enumerate(sorted_idx):
				table[unique_ids[idx]] = rank % n_routes

			# Cache for deterministic routing at inference
			self._token_route_map = {
				int(unique_ids[idx]): int(table[unique_ids[idx]])
				for idx in range(len(unique_ids))
			}
			self._token_route_table = table
		else:
			table = self._token_route_table

		return table[last_ids.clamp(max=len(table) - 1)]

	def _assign_distributional(
		self, input_bits: Tensor, targets: Tensor, n_routes: int,
	) -> Tensor:
		"""Assign routes by clustering tokens on co-occurrence distributions.

		Builds a target distribution per last-token, then uses k-means
		to group similar tokens into routes. Caches the mapping after
		the first computation for subsequent batches.
		"""
		import numpy as np

		last_ids = self._extract_last_token_ids(input_bits)
		last_np = last_ids.numpy()

		# Use cached map if available (computed on first batch)
		if self._token_route_map is not None:
			route_arr = np.array(
				[self._token_route_map.get(int(t), 0) for t in last_np],
				dtype=np.int64,
			)
			return tensor(route_arr, dtype=long)

		tgt_np = targets.numpy()

		# Build co-occurrence: last_token → target frequency distribution
		cooccurrence: dict[int, PyCounter] = defaultdict(PyCounter)
		for i in range(len(tgt_np)):
			cooccurrence[int(last_np[i])][int(tgt_np[i])] += 1

		unique_tokens = sorted(cooccurrence.keys())
		if len(unique_tokens) <= n_routes:
			# Fewer unique tokens than routes → assign 1:1
			token_to_route = {t: i % n_routes for i, t in enumerate(unique_tokens)}
		else:
			# Build sparse distribution vectors using top-100 global targets
			global_counts: PyCounter = PyCounter()
			for c in cooccurrence.values():
				global_counts.update(c)
			top_targets = [t for t, _ in global_counts.most_common(100)]
			target_to_idx = {t: i for i, t in enumerate(top_targets)}
			n_features = len(top_targets)

			# Build feature matrix [n_tokens, n_features]
			vectors = np.zeros((len(unique_tokens), n_features), dtype=np.float32)
			for i, tok in enumerate(unique_tokens):
				for tgt, cnt in cooccurrence[tok].items():
					if tgt in target_to_idx:
						vectors[i, target_to_idx[tgt]] = cnt
			# L1-normalize rows
			row_sums = vectors.sum(axis=1, keepdims=True)
			row_sums[row_sums == 0] = 1
			vectors /= row_sums

			# Simple k-means (10 iterations)
			rng = np.random.default_rng(42)
			centroid_idx = rng.choice(len(unique_tokens), size=n_routes, replace=False)
			centroids = vectors[centroid_idx].copy()

			for _ in range(10):
				# Assign to nearest centroid (L2 distance)
				dists = np.linalg.norm(
					vectors[:, None, :] - centroids[None, :, :], axis=2,
				)  # [n_tokens, n_routes]
				labels = dists.argmin(axis=1)

				# Update centroids
				for r in range(n_routes):
					mask = labels == r
					if mask.any():
						centroids[r] = vectors[mask].mean(axis=0)

			token_to_route = {
				tok: int(labels[i]) for i, tok in enumerate(unique_tokens)
			}

		# Cache for future batches
		self._token_route_map = token_to_route

		# Map each example to its route
		route_arr = np.array(
			[token_to_route.get(int(t), 0) for t in last_np], dtype=np.int64,
		)
		return tensor(route_arr, dtype=long)

	def train_routing(
		self,
		input_bits: Tensor,
		targets: Tensor,
		num_routes: Optional[int] = None,
	) -> dict:
		"""Train routing via the selected strategy.

		Assigns routes based on observable input features (not targets),
		then trains the router RAM to reproduce the assignment.

		Args:
			input_bits: [N, total_input_bits] training contexts.
			targets: [N] target token IDs.
			num_routes: Override number of routes (default: self.num_routes).

		Returns:
			Training stats dict.
		"""
		n_routes = num_routes or self.num_routes

		# Compute route assignments using the selected strategy
		if self.strategy == RoutingStrategy.CONTEXT_HASH:
			route_assignments = self._assign_context_hash(input_bits, n_routes)
		elif self.strategy == RoutingStrategy.LAST_TOKEN:
			route_assignments = self._assign_last_token(input_bits, n_routes)
		elif self.strategy == RoutingStrategy.DISTRIBUTIONAL:
			route_assignments = self._assign_distributional(
				input_bits, targets, n_routes,
			)
		else:
			raise ValueError(f"Unknown strategy: {self.strategy}")

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

		# Report route distribution
		route_dist = {}
		for r in range(n_routes):
			count = (route_assignments == r).sum().item()
			if count > 0:
				route_dist[r] = count

		return {
			"modified": total_modified,
			"examples": n,
			"strategy": self.strategy.name,
			"route_distribution": route_dist,
		}


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
		bits_per_token: int = 16,
		routing_strategy: RoutingStrategy = RoutingStrategy.LAST_TOKEN,
		use_deterministic_routing: bool = False,
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
			bits_per_token: Bits per token (for extracting last token).
			routing_strategy: How to assign examples to routes.
			use_deterministic_routing: If True, compute route directly at
				inference instead of using the learned router RAM.
			rng: Random seed.
		"""
		super().__init__()
		self.total_input_bits = total_input_bits
		self.num_clusters = num_clusters
		self.num_routes = num_routes
		self.top_k_routes = min(top_k_routes, num_routes)
		self.neurons_per_cluster = neurons_per_cluster_per_route
		self.bits_per_neuron = bits_per_neuron
		self.routing_strategy = routing_strategy
		self.use_deterministic_routing = use_deterministic_routing

		# Create router
		self.router = RouterRAM(
			total_input_bits=total_input_bits,
			num_routes=num_routes,
			router_bits=router_bits,
			bits_per_token=bits_per_token,
			routing_strategy=routing_strategy,
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
		"""Forward pass with selective routing.

		If use_deterministic_routing is True, computes routes directly
		from input bits (always correct, no degeneracy). Otherwise uses
		the learned router RAM.

		Only evaluates the top-k selected experts for each input (not all).
		With top_k=2 out of 8 routes, this evaluates ~25% of expert work.

		Args:
			input_bits: [batch, total_input_bits] boolean tensor.

		Returns:
			[batch, num_clusters] probability scores.
		"""
		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		batch_size = input_bits.shape[0]

		if self.use_deterministic_routing:
			# Deterministic: compute route directly (top-1 only)
			routes = self.router.deterministic_route(input_bits)  # [batch]
			top_routes = routes.unsqueeze(1)  # [batch, 1]
		else:
			# Learned: use router RAM scores (top-k)
			router_scores = self.router.forward(input_bits)  # [batch, num_routes]
			_, top_routes = torch_topk(
				router_scores, self.top_k_routes, dim=-1,
			)  # [batch, top_k]

		# Selective evaluation: only run experts that are actually needed
		output = zeros(batch_size, self.num_clusters, dtype=float32)
		counts = zeros(batch_size, dtype=float32)

		for route_idx in range(self.num_routes):
			# Find inputs that selected this expert (in any top-k slot)
			mask = (top_routes == route_idx).any(dim=1)  # [batch]
			if not mask.any():
				continue

			expert_scores = self._forward_single_expert(route_idx, input_bits[mask])
			output[mask] += expert_scores
			counts[mask] += 1

		# Average over selected experts
		return output / counts.unsqueeze(1).clamp(min=1)

	def _rebuild_expert_caches(self):
		"""Cache per-expert numpy arrays for accelerated forward."""
		import numpy as np
		self._expert_caches = []
		for e in self.experts:
			self._expert_caches.append({
				'connections': e.memory.connections.flatten().numpy().astype(np.int64),
				'memory_words': e.memory.memory_words.flatten().numpy().astype(np.int64),
				'total_input_bits': e.memory.total_input_bits,
				'num_neurons': e.memory.num_neurons,
				'n_bits_per_neuron': e.memory.n_bits_per_neuron,
				'neurons_per_cluster': e.neurons_per_cluster,
				'num_clusters': e.num_clusters,
				'words_per_neuron': e.memory.words_per_neuron,
			})
		self._merged_dirty = False

	def _forward_single_expert(self, route_idx: int, input_bits: Tensor) -> Tensor:
		"""Evaluate a single expert on a subset of inputs via Metal/Rust.

		Args:
			route_idx: Expert index.
			input_bits: [subset, total_input_bits] inputs that need this expert.

		Returns:
			[subset, num_clusters] expert scores.
		"""
		try:
			import ram_accelerator
			import numpy as np
			from torch import from_numpy, long as torch_long
		except ImportError:
			return self.experts[route_idx].forward(input_bits)

		if input_bits.ndim == 1:
			input_bits = input_bits.unsqueeze(0)

		batch_size = input_bits.shape[0]
		input_np = input_bits.flatten().to(dtype=torch_long).numpy().astype(np.uint8)

		# Rebuild caches after training
		if self._merged_dirty:
			self._rebuild_expert_caches()

		cache = self._expert_caches[route_idx]

		# Pick fastest available backend
		use_metal = ram_accelerator.ramlm_metal_available()
		forward_fn = (ram_accelerator.ramlm_forward_batch_metal_cached
					  if use_metal else ram_accelerator.ramlm_forward_batch_numpy)

		scores_flat = forward_fn(
			input_np, cache['connections'], cache['memory_words'],
			batch_size, cache['total_input_bits'], cache['num_neurons'],
			cache['n_bits_per_neuron'], cache['neurons_per_cluster'],
			cache['num_clusters'], cache['words_per_neuron'],
		)
		scores_t = from_numpy(np.array(scores_flat, dtype=np.float32))
		return scores_t.view(batch_size, self.num_clusters)

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
		# Use Rust numpy path to avoid OOM from Python vectorized addresses tensor
		# (Python path materializes [batch, unique_clusters * neurons] = tens of GB)
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
				expert_false = shared_negatives.unsqueeze(0).expand(count, -1).contiguous()
				modified = self.experts[route_idx].train_multi_examples_rust_numpy(
					expert_bits, expert_targets, expert_false,
					allow_override=allow_override,
				)
			elif false_clusters is not None:
				expert_false = false_clusters[mask]
				modified = self.experts[route_idx].train_multi_examples_rust_numpy(
					expert_bits, expert_targets, expert_false,
					allow_override=allow_override,
				)
			else:
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
						modified = self.experts[route_idx].train_multi_examples_rust_numpy(
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
