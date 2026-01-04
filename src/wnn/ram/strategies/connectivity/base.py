"""
Base classes for connectivity optimization strategies.

Based on Garcia (2003) thesis on global optimization methods for
choosing connectivity patterns of weightless neural networks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

from torch import Tensor, manual_seed


@dataclass
class OverfittingControl:
	"""
	Control signal from overfitting callback to optimizer.

	Uses BASELINE-RELATIVE gap detection: monitors how much the val/train ratio
	INCREASES from the initial baseline, not absolute gap percentage.

	This is crucial when train and validation come from different distributions
	(e.g., different Wikipedia articles), where absolute gaps can be 1000%+.

	Thresholds are based on gap ratio INCREASE from baseline:
	- increase < 5%:  Healthy → normal operation
	- increase > 10%: Warning → activate diversity_mode
	- increase > 20%: Critical → early_stop

	When diversity_mode is True, optimizers should:
	- GA: ↑population, ↓elitism, ↑mutation_rate
	- TS: ↑neighbors, ↑mutation_rate, ↑tabu_size
	"""
	early_stop: bool = False
	diversity_mode: bool = False


# Type alias for overfitting callback
# Args: (current_best_connectivity, train_fitness) → control signal
OverfittingCallback = Callable[[Tensor, float], OverfittingControl]


class OverfittingMonitor:
	"""
	Monitors train/validation gap using BASELINE-RELATIVE detection.

	Instead of absolute gap thresholds (which fail when train/val distributions
	differ significantly, e.g., 1000%+ gaps), this monitors how much the
	val/train RATIO increases from the initial baseline.

	Thresholds are based on ratio INCREASE from baseline:
	- increase < healthy_threshold (5%):  Normal operation, exit diversity mode
	- increase > warning_threshold (10%): Activate diversity mode
	- increase > critical_threshold (20%): Early stop (after grace period)

	Example:
		- Baseline: train=180, val=2160 → ratio=12.0x
		- Later: train=170, val=2200 → ratio=12.9x → increase=+7.5% → healthy
		- Later: train=160, val=2400 → ratio=15.0x → increase=+25% → critical!

	Usage:
		monitor = OverfittingMonitor(
			validation_fn=lambda conn: val_batch_fn([conn])[0],
			logger=log,
		)
		# Pass monitor as callback to optimizer
		result = ga.optimize(..., overfitting_callback=monitor)
	"""

	def __init__(
		self,
		validation_fn: Callable[[Tensor], float],
		healthy_threshold: float = 5.0,
		warning_threshold: float = 10.0,
		critical_threshold: float = 20.0,
		grace_checks: int = 1,
		logger: Optional[Callable[[str], None]] = None,
		global_baseline_ratio: Optional[float] = None,
	):
		"""
		Args:
			validation_fn: Function that evaluates connectivity on validation set.
				Takes Tensor, returns validation fitness (e.g., perplexity).
			healthy_threshold: Ratio increase % below which to exit diversity mode (default: 5%)
			warning_threshold: Ratio increase % above which to enter diversity mode (default: 10%)
			critical_threshold: Ratio increase % above which to early stop (default: 20%)
			grace_checks: Number of checks before allowing early stop (default: 1).
				Each check happens every 5 generations/iterations.
			logger: Optional logging function for status messages.
			global_baseline_ratio: Optional pre-computed baseline ratio (val/train).
				If set, this persists across reset() calls.
		"""
		self._validation_fn = validation_fn
		self._healthy_threshold = healthy_threshold
		self._warning_threshold = warning_threshold
		self._critical_threshold = critical_threshold
		self._grace_checks = grace_checks
		self._logger = logger
		self._check_count = 0
		self._in_diversity_mode = False
		self._global_baseline_ratio: Optional[float] = global_baseline_ratio  # Set once, persists across reset()
		self._local_baseline_ratio: Optional[float] = None   # Set per RAM, reset with reset()

	def set_global_baseline(self, train_ppl: float, val_ppl: float) -> None:
		"""
		Set the global baseline ratio from pre-optimization evaluation.
		This baseline persists across reset() calls.

		Args:
			train_ppl: Training perplexity before optimization
			val_ppl: Validation perplexity before optimization
		"""
		self._global_baseline_ratio = val_ppl / train_ppl if train_ppl > 0 else 1.0
		self._log(f"  Global baseline set: train={train_ppl:.1f}, val={val_ppl:.1f}, ratio={self._global_baseline_ratio:.2f}x")
		self._log(f"  Thresholds: healthy <{self._healthy_threshold}%, warning >{self._warning_threshold}%, critical >{self._critical_threshold}%")

	def reset(self) -> None:
		"""Reset state for a new optimization run (per-RAM). Global baseline is preserved."""
		self._check_count = 0
		self._in_diversity_mode = False
		self._local_baseline_ratio = None  # Reset local baseline, keep global

	@property
	def baseline_ratio(self) -> Optional[float]:
		"""Get the active baseline ratio (global if set, otherwise local)."""
		return self._global_baseline_ratio if self._global_baseline_ratio is not None else self._local_baseline_ratio

	def _log(self, msg: str) -> None:
		if self._logger:
			self._logger(msg)

	def __call__(self, connectivity: Tensor, train_fitness: float) -> OverfittingControl:
		"""
		Check for overfitting using baseline-relative gap detection.

		Args:
			connectivity: Current best connectivity pattern
			train_fitness: Current training fitness (e.g., perplexity)

		Returns:
			OverfittingControl with early_stop and diversity_mode flags
		"""
		self._check_count += 1
		val_fitness = self._validation_fn(connectivity)

		# Calculate current ratio (val/train)
		current_ratio = val_fitness / train_fitness if train_fitness > 0 else 1.0

		# Use global baseline if set, otherwise establish local baseline on first check
		baseline = self.baseline_ratio
		if baseline is None:
			self._local_baseline_ratio = current_ratio
			baseline = current_ratio
			self._log(f"    [VAL] val={val_fitness:.1f}, train={train_fitness:.1f}, ratio={current_ratio:.2f}x (local baseline)")
			self._log(f"         thresholds: <{self._healthy_threshold}% healthy, >{self._warning_threshold}% warn, >{self._critical_threshold}% stop")
			# First check always healthy (establishing baseline)
			return OverfittingControl(early_stop=False, diversity_mode=False)

		# Calculate ratio INCREASE from baseline (as percentage)
		ratio_increase_pct = ((current_ratio - baseline) / baseline * 100) if baseline > 0 else 0

		in_grace_period = self._check_count <= self._grace_checks + 1  # +1 because first check sets baseline

		# Determine status and symbol
		if ratio_increase_pct > self._critical_threshold:
			status = "🔴 CRITICAL"
		elif ratio_increase_pct > self._warning_threshold:
			status = "🟡 WARNING"
		elif ratio_increase_pct > self._healthy_threshold:
			status = "🟠 CAUTION"
		else:
			status = "🟢 HEALTHY"

		# Compact log with status indicator
		baseline_type = "global" if self._global_baseline_ratio is not None else "local"
		self._log(f"    [VAL] val={val_fitness:.1f}, train={train_fitness:.1f}, ratio={current_ratio:.2f}x (Δ={ratio_increase_pct:+.1f}% vs {baseline_type} {baseline:.2f}x) {status}")

		if ratio_increase_pct > self._critical_threshold:
			if in_grace_period:
				self._log(f"         → DIVERSITY MODE (grace {self._check_count-1}/{self._grace_checks}, need >{self._critical_threshold}% for stop)")
				self._in_diversity_mode = True
				return OverfittingControl(early_stop=False, diversity_mode=True)
			else:
				self._log(f"         → EARLY STOP (ratio increase >{self._critical_threshold}%)")
				return OverfittingControl(early_stop=True, diversity_mode=False)

		elif ratio_increase_pct > self._warning_threshold:
			self._log(f"         → DIVERSITY MODE (>{self._warning_threshold}%)")
			self._in_diversity_mode = True
			return OverfittingControl(early_stop=False, diversity_mode=True)

		elif ratio_increase_pct > self._healthy_threshold:
			# Between healthy and warning: maintain current state
			if self._in_diversity_mode:
				self._log(f"         → staying in DIVERSITY MODE ({self._healthy_threshold}-{self._warning_threshold}%)")
			return OverfittingControl(early_stop=False, diversity_mode=self._in_diversity_mode)

		else:
			# Healthy: ratio increase below threshold
			if self._in_diversity_mode:
				self._log(f"         → exiting DIVERSITY MODE (<{self._healthy_threshold}%)")
				self._in_diversity_mode = False
			return OverfittingControl(early_stop=False, diversity_mode=False)

	@property
	def check_count(self) -> int:
		"""Number of times this monitor has been called."""
		return self._check_count

@dataclass
class OptimizerResult:
	"""Result of connectivity optimization."""

	initial_connections: Tensor
	optimized_connections: Tensor
	initial_error: float
	final_error: float
	improvement_percent: float
	iterations_run: int
	method_name: str
	history: list = field(default_factory=list)
	early_stopped_overfitting: bool = False  # True if stopped due to overfitting

	def __repr__(self) -> str:
		return (
			f"OptimizerResult("
			f"method={self.method_name}, "
			f"initial_error={self.initial_error:.4f}, "
			f"final_error={self.final_error:.4f}, "
			f"improvement={self.improvement_percent:.2f}%)"
		)


class OptimizerStrategyBase(ABC):
	"""
	Abstract base class for connectivity optimization strategies.

	Implements the Strategy pattern for global optimization of RAM neuron
	connectivity patterns. Based on Garcia (2003) thesis.

	Subclasses must implement:
	- optimize(): The main optimization loop
	- name property

	Usage:
		strategy = TabuSearchStrategy(config)
		result = strategy.optimize(
			connections=initial_connections,
			evaluate_fn=lambda conn: compute_error(conn),
			total_input_bits=16,
			num_neurons=8,
			n_bits_per_neuron=4,
		)
	"""

	def __init__(
		self,
		seed: Optional[int] = None,
		verbose: bool = False,
		logger: Optional[Callable[[str], None]] = None,
	):
		self._seed = seed
		self._verbose = verbose
		self._logger = logger or print
		self._rng_initialized = False

	def _log(self, msg: str) -> None:
		"""Log a message using the configured logger."""
		if self._verbose:
			self._logger(msg)

	@property
	def seed(self) -> Optional[int]:
		return self._seed

	@property
	def verbose(self) -> bool:
		return self._verbose

	@property
	@abstractmethod
	def name(self) -> str:
		"""Return the strategy name."""
		...

	def _ensure_rng(self) -> None:
		"""Initialize RNG if needed."""
		if not self._rng_initialized and self._seed is not None:
			manual_seed(self._seed)
			import random
			random.seed(self._seed)
			self._rng_initialized = True

	@abstractmethod
	def optimize(
		self,
		connections: Tensor,
		evaluate_fn: Callable[[Tensor], float],
		total_input_bits: int,
		num_neurons: int,
		n_bits_per_neuron: int,
	) -> OptimizerResult:
		"""
		Optimize connectivity pattern.

		Args:
			connections: Initial connectivity [num_neurons, n_bits_per_neuron]
			evaluate_fn: Function that evaluates a connectivity pattern,
				returns error rate (lower is better)
			total_input_bits: Size of input vector
			num_neurons: Number of RAM neurons
			n_bits_per_neuron: Fan-in per neuron

		Returns:
			OptimizerResult with optimized connections and statistics
		"""
		...

	def _generate_neighbor(
		self,
		connections: Tensor,
		mutation_rate: float,
		total_input_bits: int,
		num_neurons: int,
		n_bits_per_neuron: int,
	) -> tuple[Tensor, tuple]:
		"""
		Generate neighbor by mutating connections.

		From Garcia (2003) thesis:
		- For each connection, with probability mutation_rate, modify it
		- Generate random value 0..total_input_bits-1
		- Allows changing which input bit a connection observes

		Returns:
			(new_connections, move) where move = (neuron_idx, old_value, new_value)
		"""
		import random

		neighbor = connections.clone()
		last_move = (0, 0, 0)

		for neuron_idx in range(num_neurons):
			for conn_idx in range(n_bits_per_neuron):
				if random.random() < mutation_rate:
					old_value = int(neighbor[neuron_idx, conn_idx].item())
					new_value = random.randint(0, total_input_bits - 1)
					neighbor[neuron_idx, conn_idx] = new_value
					last_move = (neuron_idx, old_value, new_value)

		return neighbor, last_move

	def compute_orthogonality(
		self,
		connections: Tensor,
		sample_inputs: Tensor,
		n_bits_per_neuron: int,
	) -> float:
		"""
		Compute orthogonality score for a connectivity pattern.

		Higher orthogonality means neurons observe different aspects of input,
		leading to better discrimination. From Garcia (2003), SA improved
		orthogonality by 50%, leading to 10% classification improvement.

		Args:
			connections: [num_neurons, n_bits_per_neuron]
			sample_inputs: [batch_size, total_input_bits]
			n_bits_per_neuron: Fan-in per neuron

		Returns:
			Orthogonality score (higher is better, range 0-2)
		"""
		from torch import arange, zeros, int64

		num_neurons = connections.shape[0]
		binary_bases = 2 ** arange(n_bits_per_neuron - 1, -1, -1, dtype=int64)

		# For each neuron, compute its address distribution over samples
		address_distributions = []
		for neuron_idx in range(num_neurons):
			neuron_conn = connections[neuron_idx]
			gathered = sample_inputs[:, neuron_conn].to(int64)
			addresses = (gathered * binary_bases).sum(dim=1)

			# Compute histogram distribution
			hist = zeros(2 ** n_bits_per_neuron)
			for addr in addresses:
				hist[addr] += 1
			hist = hist / hist.sum()
			address_distributions.append(hist)

		# Compute pairwise L1 distances (total variation) between distributions
		total_distance = 0.0
		count = 0
		for i in range(num_neurons):
			for j in range(i + 1, num_neurons):
				distance = (address_distributions[i] - address_distributions[j]).abs().sum().item()
				total_distance += distance
				count += 1

		return total_distance / count if count > 0 else 0.0

	def __repr__(self) -> str:
		return f"{self.__class__.__name__}(seed={self._seed}, verbose={self._verbose})"
