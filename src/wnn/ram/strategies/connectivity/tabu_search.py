"""
Tabu Search strategy for connectivity optimization.

From Garcia (2003) thesis, Tabu Search achieved the best results:
- 17.27% error reduction on T2 topology
- Only 5 iterations needed for convergence
- 30 neighbors per iteration
- Tabu list size of 5 to avoid cycling
"""

from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from torch import Tensor

from wnn.ram.strategies.connectivity.base import (
	OptimizerResult,
	OptimizerStrategyBase,
	OverfittingControl,
	OverfittingCallback,
)
from wnn.ram.strategies.perplexity import PerplexityCalculator


@dataclass
class TabuSearchConfig:
	"""
	Configuration for Tabu Search optimization.

	Parameters tuned from Garcia (2003) experiments:
	- iterations: 5 was sufficient for convergence
	- neighbors_per_iter: 30 neighbors tested per iteration
	- tabu_size: 5 recent moves stored to avoid cycling
	- mutation_rate: 0.001 per-connection probability
	- early_stop_patience: iterations without improvement before stopping
	- early_stop_threshold: minimum PPL % improvement required to reset patience
	"""
	iterations: int = 5
	neighbors_per_iter: int = 30
	tabu_size: int = 5
	mutation_rate: float = 0.001
	early_stop_patience: int = 1  # Stop if no improvement for 1 check (5 iterations)
	early_stop_threshold_pct: float = 0.02  # Minimum PPL % improvement required (0.02 = 0.02%)


class TabuSearchStrategy(OptimizerStrategyBase):
	"""
	Tabu Search optimization for RAM neuron connectivity patterns.

	Key features from Garcia (2003):
	- Generate multiple neighbors per iteration (unlike SA which uses 1)
	- Keep tabu list to avoid cycling back to recent solutions
	- Always move to best neighbor (even if worse - characteristic of TS)
	- Fewer iterations needed than SA due to parallel exploration

	This was the best-performing method in the thesis, achieving:
	- 17.27% error reduction (24.95% -> 20.64% on Buffalo T2)
	- Consistent results with low variance
	- Fast convergence in just 5 iterations
	"""

	def __init__(
		self,
		config: Optional[TabuSearchConfig] = None,
		seed: Optional[int] = None,
		verbose: bool = False,
		logger: Optional[Callable[[str], None]] = None,
	):
		super().__init__(seed=seed, verbose=verbose, logger=logger)
		self._config = config or TabuSearchConfig()

	@property
	def config(self) -> TabuSearchConfig:
		return self._config

	@property
	def name(self) -> str:
		return "TabuSearch"

	def optimize(
		self,
		connections: Tensor,
		evaluate_fn: Callable[[Tensor], float],
		total_input_bits: int,
		num_neurons: int,
		n_bits_per_neuron: int,
		batch_evaluate_fn: Optional[Callable[[list], list[float]]] = None,
		overfitting_callback: Optional[OverfittingCallback] = None,
		initial_error_hint: Optional[float] = None,
	) -> OptimizerResult:
		"""
		Run Tabu Search optimization.

		The algorithm:
		1. Start with initial connections
		2. Generate N neighbors by mutation
		3. Select best non-tabu neighbor
		4. Add move to tabu list (avoid reverting)
		5. Repeat for configured iterations

		Args:
			batch_evaluate_fn: Optional function to evaluate multiple patterns at once.
				If provided, uses batch evaluation for massive speedup with Rust/joblib.
			overfitting_callback: Optional callback for overfitting detection.
				Called every 5 iterations with (best_connectivity, train_fitness).
				Returns OverfittingControl to signal diversity mode or early stop.
			initial_error_hint: Optional initial error from previous optimizer (e.g., GA).
				If provided, uses this as baseline instead of re-evaluating.
				This ensures consistent baselines when chaining GA → TS.
		"""
		self._ensure_rng()
		cfg = self._config

		# Diversity mode tracking - store original values for restoration
		in_diversity_mode = False
		in_severe_mode = False
		original_neighbors = cfg.neighbors_per_iter
		original_mutation_rate = cfg.mutation_rate
		original_tabu_size = cfg.tabu_size
		early_stopped_overfitting = False

		current = connections.clone()

		# Use hint if provided (for GA → TS chaining), otherwise evaluate
		if initial_error_hint is not None:
			current_error = initial_error_hint
			self._log(f"[TS] Using inherited baseline from previous optimizer: {current_error:.4f}")
		else:
			current_error = evaluate_fn(current) if batch_evaluate_fn is None else batch_evaluate_fn([current], total_pop=1)[0]

		best = current.clone()
		best_error = current_error
		initial_error = current_error

		# Tabu list stores recent moves to avoid cycling
		# Each entry is (neuron_idx, old_connection, new_connection)
		tabu_list: deque = deque(maxlen=cfg.tabu_size)

		history = [(0, current_error)]

		# Early stopping tracking
		patience_counter = 0
		prev_best_for_patience = best_error

		self._log(f"[TS] Initial error: {current_error:.4f}")

		for iteration in range(cfg.iterations):
			# Generate all neighbors first (for batch evaluation)
			neighbor_candidates = []
			for _ in range(cfg.neighbors_per_iter):
				neighbor, move = self._generate_neighbor(
					current, cfg.mutation_rate,
					total_input_bits, num_neurons, n_bits_per_neuron
				)

				# Check if move is tabu (reverting a recent change)
				is_tabu = any(
					m[0] == move[0] and m[2] == move[1]
					for m in tabu_list
				)

				if not is_tabu:
					neighbor_candidates.append((neighbor, move))

			if not neighbor_candidates:
				continue

			# Batch evaluate all non-tabu neighbors at once
			if batch_evaluate_fn is not None:
				to_eval = [n for n, _ in neighbor_candidates]
				errors = batch_evaluate_fn(to_eval, total_pop=len(to_eval))
				neighbors = [(n, e, m) for (n, m), e in zip(neighbor_candidates, errors)]
			else:
				neighbors = [(n, evaluate_fn(n), m) for n, m in neighbor_candidates]

			# Select best non-tabu neighbor
			neighbors.sort(key=lambda x: x[1])
			best_neighbor, best_neighbor_error, best_move = neighbors[0]

			# Update current (always move to best neighbor - TS characteristic)
			current = best_neighbor
			current_error = best_neighbor_error

			# Add move to tabu list
			tabu_list.append(best_move)

			# Update global best
			if current_error < best_error:
				best = current.clone()
				best_error = current_error

			history.append((iteration + 1, best_error))

			# Log every iteration for progress visibility
			if self._verbose:
				avg_neighbor_error = sum(e for _, e, _ in neighbors) / len(neighbors)
				self._log(f"[TS] Iter {iteration + 1}/{cfg.iterations}: current={current_error:.4f}, best={best_error:.4f}, avg={avg_neighbor_error:.4f}")

			# Early stopping check every 5 iterations
			# Uses PPL-based improvement: since PPL = exp(CE), small CE changes = large PPL changes
			if (iteration + 1) % 5 == 0:
				# Calculate PPL improvement percentage
				ppl_improvement = PerplexityCalculator.ppl_improvement_pct(prev_best_for_patience, best_error)

				if ppl_improvement >= cfg.early_stop_threshold_pct:
					patience_counter = 0
					prev_best_for_patience = best_error
				else:
					patience_counter += 1

				if self._verbose:
					self._log(f"[TS] Early stop check: PPL Δ={ppl_improvement:.2f}%, patience={cfg.early_stop_patience - patience_counter}/{cfg.early_stop_patience}")

				if patience_counter > cfg.early_stop_patience:
					self._log(f"[TS] Early stop at iter {iteration + 1}: no PPL improvement >= {cfg.early_stop_threshold_pct}% for {patience_counter * 5} iterations")
					break

				# Overfitting callback check
				if overfitting_callback is not None:
					control = overfitting_callback(best, best_error)

					if control.early_stop:
						self._log(f"[TS] Overfitting early stop at iter {iteration + 1}")
						early_stopped_overfitting = True
						break

					if control.diversity_mode and not in_diversity_mode:
						# Activate diversity mode: ↑neighbors, ↑mutation, ↑tabu_size
						in_diversity_mode = True
						in_severe_mode = control.severe_diversity_mode
						if control.severe_diversity_mode:
							# SEVERE: 3x neighbors, 3x mutation, 3x tabu
							cfg.neighbors_per_iter = int(original_neighbors * 3)
							cfg.mutation_rate = original_mutation_rate * 3
							cfg.tabu_size = max(original_tabu_size * 3, 15)
							self._log(f"[TS] SEVERE Diversity ON: neighbors={cfg.neighbors_per_iter}, mut={cfg.mutation_rate:.4f}, tabu={cfg.tabu_size}")
						else:
							# MILD: 2x neighbors, 2x mutation, 2x tabu
							cfg.neighbors_per_iter = int(original_neighbors * 2)
							cfg.mutation_rate = original_mutation_rate * 2
							cfg.tabu_size = max(original_tabu_size * 2, 10)
							self._log(f"[TS] Diversity mode ON: neighbors={cfg.neighbors_per_iter}, mut={cfg.mutation_rate:.4f}, tabu={cfg.tabu_size}")
						# Resize tabu list to new size
						new_tabu: deque = deque(tabu_list, maxlen=cfg.tabu_size)
						tabu_list = new_tabu

					elif control.diversity_mode and in_diversity_mode:
						# Check if severity level changed
						if control.severe_diversity_mode and not in_severe_mode:
							# Escalate to severe
							in_severe_mode = True
							cfg.neighbors_per_iter = int(original_neighbors * 3)
							cfg.mutation_rate = original_mutation_rate * 3
							cfg.tabu_size = max(original_tabu_size * 3, 15)
							new_tabu = deque(tabu_list, maxlen=cfg.tabu_size)
							tabu_list = new_tabu
							self._log(f"[TS] Escalating to SEVERE: neighbors={cfg.neighbors_per_iter}, mut={cfg.mutation_rate:.4f}, tabu={cfg.tabu_size}")
						elif not control.severe_diversity_mode and in_severe_mode:
							# De-escalate from severe to mild
							in_severe_mode = False
							cfg.neighbors_per_iter = int(original_neighbors * 2)
							cfg.mutation_rate = original_mutation_rate * 2
							cfg.tabu_size = max(original_tabu_size * 2, 10)
							new_tabu = deque(list(tabu_list)[-cfg.tabu_size:], maxlen=cfg.tabu_size)
							tabu_list = new_tabu
							self._log(f"[TS] De-escalating to mild: neighbors={cfg.neighbors_per_iter}, mut={cfg.mutation_rate:.4f}, tabu={cfg.tabu_size}")

					elif not control.diversity_mode and in_diversity_mode:
						# Back to normal: restore original hyperparameters
						in_diversity_mode = False
						in_severe_mode = False
						cfg.neighbors_per_iter = original_neighbors
						cfg.mutation_rate = original_mutation_rate
						cfg.tabu_size = original_tabu_size
						# Shrink tabu list back
						new_tabu = deque(list(tabu_list)[-cfg.tabu_size:], maxlen=cfg.tabu_size)
						tabu_list = new_tabu
						self._log(f"[TS] Diversity mode OFF: restored neighbors={cfg.neighbors_per_iter}, mut={cfg.mutation_rate:.4f}, tabu={cfg.tabu_size}")

		improvement_pct = ((initial_error - best_error) / initial_error * 100) if best_error < initial_error else 0.0

		return OptimizerResult(
			initial_connections=connections,
			optimized_connections=best,
			initial_error=initial_error,
			final_error=best_error,
			improvement_percent=improvement_pct,
			iterations_run=iteration + 1,
			method_name=self.name,
			history=history,
			early_stopped_overfitting=early_stopped_overfitting,
		)

	def __repr__(self) -> str:
		return f"TabuSearchStrategy(config={self._config}, seed={self._seed}, verbose={self._verbose})"
