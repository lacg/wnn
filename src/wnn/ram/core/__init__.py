"""
RAM Core Components

Fundamental building blocks for RAM neural networks.
"""

from enum import IntEnum, auto


# =============================================================================
# Core Enums (self-contained)
# =============================================================================

class MemoryVal(IntEnum):
	"""
	Ternary memory values stored as uint8:
	- FALSE = 0
	- TRUE  = 1
	- EMPTY = 2   (safe to overwrite; means "untrained")
	"""
	FALSE	= False
	TRUE	= True
	EMPTY	= 2


class ContextMode(IntEnum):
	"""
	How much context each bit sees in BitLevelMapper.

	Different context modes enable different generalization patterns:
	- CUMULATIVE: Best for carry-chain operations (increment, add)
	- FULL: Best when all bits matter (arbitrary transforms)
	- LOCAL: Best when nearby bits matter most (local patterns)
	- BIDIRECTIONAL: Best when both before/after matter (symmetric ops)
	- CAUSAL: Best for autoregressive tasks (only sees past)
	- SHIFTED: Best for position-based transforms (shift, rotate)
	"""
	CUMULATIVE = 0    # bit i sees bits 0..i-1 (only LOWER bits for flip)
	FULL = 1          # each bit sees all bits
	LOCAL = 2         # each bit sees nearby bits (sliding window)
	BIDIRECTIONAL = 3 # bit i sees bits before AND after (symmetric window)
	CAUSAL = 4        # bit i sees bits 0..i (autoregressive, includes self)
	SHIFTED = 5       # bit i sees bit (i+offset) mod n (position-based transforms)


class BitMapperMode(IntEnum):
	"""
	What to learn in BitLevelMapper.

	Note: Previously named 'OutputMode' in RAMGeneralization.py,
	renamed to avoid conflict with decoder's OutputMode.
	"""
	OUTPUT = 0  # Learn the output bit value directly
	FLIP = 1    # Learn whether to flip (XOR) the input bit


class MapperStrategy(IntEnum):
	"""
	Generalization strategy for MapperFactory.

	Strategies differ in how they reduce the pattern space:
	- DIRECT: No reduction (2^n patterns for n-bit input)
	- BIT_LEVEL: Per-bit learning reduces patterns exponentially
	- COMPOSITIONAL: Group-based reduces to k * 2^(n/k) patterns
	- HYBRID: Combines compositional + bit-level
	- HASH: Hash input to smaller lookup (loses precision but generalizes)
	- RESIDUAL: Learn corrections to identity (good for small changes)
	- SHIFTED: Position-based transforms (shift, rotate) - 100% generalization
	- PARITY: Recurrent XOR for parity computation - 100% generalization
	"""
	DIRECT = 0        # Standard RAMLayer (no generalization)
	BIT_LEVEL = 1     # Use BitLevelMapper
	COMPOSITIONAL = 2 # Use CompositionalMapper
	HYBRID = 3        # Combine compositional + bit-level
	HASH = 4          # Hash input to reduce lookup space
	RESIDUAL = 5      # Identity + learned correction
	SHIFTED = 6       # BitLevelMapper with SHIFTED context (for shift/rotate)
	PARITY = 7        # RecurrentParityMapper (for parity computation)


class LayerType(IntEnum):
	"""Types of layers in the RAM Transformer."""
	EMBEDDING = 0
	INPUT_PROJ = 1
	ATTENTION = 2
	FFN = 3
	OUTPUT_PROJ = 4
	TOKEN_MAPPER = 5


class TrainingMode(IntEnum):
	"""
	Training strategies for EDRA.

	Different modes trade off between training speed and accuracy:
	- GREEDY: Train all layers in one pass (fast, may miss dependencies)
	- ITERATIVE: Multiple passes until stable (slower, better accuracy)
	- LAYERWISE: Train one layer at a time, freeze others (most controlled)
	- OUTPUT_FIRST: Train output layers first, then propagate backward
	"""
	GREEDY = 0        # Train all layers in single backward pass
	ITERATIVE = 1     # Multiple passes until convergence
	LAYERWISE = 2     # Train one layer at a time
	OUTPUT_FIRST = 3  # Prioritize output layers


class TrainingPhase(IntEnum):
	"""
	Training phases for curriculum learning.

	Supports progressive training from simple to complex:
	- WARMUP: Train on easiest examples
	- MAIN: Train on full dataset
	- REFINEMENT: Focus on hard examples
	"""
	WARMUP = 0
	MAIN = 1
	REFINEMENT = 2


class MixingStrategy(IntEnum):
	"""How to mix examples from multiple tasks."""
	ROUND_ROBIN = 0   # Alternate between tasks
	PROPORTIONAL = 1  # Sample proportional to dataset size
	WEIGHTED = 2      # Sample proportional to task weight
	INTERLEAVED = 3   # Mix all examples, shuffle


class StateMode(IntEnum):
	"""
	State transition modes for recurrent networks.

	Controls how the state layer computes new_state from (input, prev_state).
	"""
	LEARNED = 0      # State transition learned via EDRA (default)
	XOR = 1          # State = prev_state XOR input (for parity)
	IDENTITY = 2     # State = input (no memory)
	OR = 3           # State = prev_state OR input (for detection)


class OptimizationMethod(IntEnum):
	"""
	Optimization method for connectivity patterns.

	Based on Garcia (2003) thesis comparing global optimization methods
	for choosing connectivity patterns of weightless neural networks.
	"""
	TABU_SEARCH = auto()          # Best results: 17.27% error reduction, only 5 iterations
	SIMULATED_ANNEALING = auto()  # Good for escaping local minima, 600 iterations
	GENETIC_ALGORITHM = auto()    # Good for large search spaces, can reduce memory by 89%


class BenchmarkMode(IntEnum):
	"""Benchmark execution modes with different parameter scales.

	Controls the intensity of optimization algorithms:
	- FAST: Quick development tests with minimal parameters
	- FULL: Production-level optimization
	- OVERNIGHT: Extended thorough optimization for overnight runs
	"""
	FAST = 0      # Quick test: minimal params for development
	FULL = 1      # Standard: production-level optimization
	OVERNIGHT = 2 # Extended: thorough overnight optimization


class MemoryMode(IntEnum):
	"""Memory mode for BitwiseRAMLM.

	Controls training semantics and forward pass interpretation:
	- TERNARY: 3-state (FALSE/TRUE/EMPTY), majority vote training
	- QUAD_BINARY: 4-state nudging, binary threshold forward (cell >= 2 → true)
	- QUAD_WEIGHTED: 4-state nudging, weighted confidence forward
	"""
	TERNARY = 0       # 3-state, majority vote (default)
	QUAD_BINARY = 1   # 4-state, nudging, binary threshold
	QUAD_WEIGHTED = 2  # 4-state, nudging, weighted confidence


class AccelerationMode(IntEnum):
	"""Hardware acceleration modes for RAM evaluation.

	Controls which compute resources are used:
	- AUTO: Auto-select best backend based on batch size (recommended)
	- PYTORCH: Pure PyTorch (best for small batches, no Rust dependency)
	- CPU: Rust + rayon parallelism (16 cores on M4 Max)
	- METAL: Metal GPU compute shaders (40 cores on M4 Max)
	- HYBRID: Both CPU + GPU in parallel (56 cores total, best for large batches)

	Usage:
		model.forward(bits, backend=AccelerationMode.AUTO)    # Auto-select
		model.forward(bits, backend=AccelerationMode.HYBRID)  # Force hybrid
	"""
	AUTO = 0     # Auto-select best backend based on batch size
	PYTORCH = 1  # Pure PyTorch (no Rust dependency)
	CPU = 2      # Rust + rayon CPU parallelism
	METAL = 3    # Metal GPU compute shaders
	HYBRID = 4   # Both CPU + GPU in parallel


# =============================================================================
# Acceleration Helpers
# =============================================================================

# Default core counts (can be overridden by actual hardware detection)
_DEFAULT_CPU_CORES = 16   # M4 Max CPU cores
_DEFAULT_GPU_CORES = 40   # M4 Max Metal GPU cores

# Runtime-detected values (set by ram_accelerator if available)
_detected_cpu_cores: int | None = None
_detected_gpu_cores: int | None = None
_metal_available: bool | None = None


def set_detected_cores(cpu_cores: int, gpu_cores: int, metal_available: bool) -> None:
	"""
	Set runtime-detected core counts (called by ram_accelerator on import).

	Args:
		cpu_cores: Number of CPU cores (from rayon)
		gpu_cores: Number of GPU cores (from Metal)
		metal_available: Whether Metal GPU is available
	"""
	global _detected_cpu_cores, _detected_gpu_cores, _metal_available
	_detected_cpu_cores = cpu_cores
	_detected_gpu_cores = gpu_cores
	_metal_available = metal_available


def get_effective_cores(
	mode: AccelerationMode,
	cpu_cores: int | None = None,
	gpu_cores: int | None = None,
) -> int:
	"""
	Get effective parallel worker count for acceleration mode.

	Uses runtime-detected values if available, otherwise defaults.
	Can override with explicit core counts.

	Args:
		mode: AccelerationMode (CPU, METAL, or HYBRID)
		cpu_cores: Override CPU core count (default: detected or 16)
		gpu_cores: Override GPU core count (default: detected or 40)

	Returns:
		Number of effective parallel workers

	Examples:
		>>> get_effective_cores(AccelerationMode.CPU)
		16
		>>> get_effective_cores(AccelerationMode.METAL)
		40
		>>> get_effective_cores(AccelerationMode.HYBRID)
		56
	"""
	# Use provided, detected, or default values
	cpu = cpu_cores or _detected_cpu_cores or _DEFAULT_CPU_CORES
	gpu = gpu_cores or _detected_gpu_cores or _DEFAULT_GPU_CORES

	match mode:
		case AccelerationMode.AUTO:
			# Auto mode uses hybrid if available, else CPU
			if _metal_available is False:
				return cpu
			return cpu + gpu
		case AccelerationMode.PYTORCH:
			return 1  # PyTorch uses its own threading
		case AccelerationMode.CPU:
			return cpu
		case AccelerationMode.METAL:
			# Fall back to CPU if Metal not available
			if _metal_available is False:
				return cpu
			return gpu
		case AccelerationMode.HYBRID:
			# Fall back to CPU if Metal not available
			if _metal_available is False:
				return cpu
			return cpu + gpu
		case _:
			return cpu


def get_batch_size_for_mode(
	mode: AccelerationMode,
	benchmark_mode: BenchmarkMode,
	cpu_cores: int | None = None,
	gpu_cores: int | None = None,
) -> tuple[int, int]:
	"""
	Get recommended population/batch sizes for optimization.

	Scales batch sizes based on available cores and benchmark intensity.

	Args:
		mode: AccelerationMode (CPU, METAL, HYBRID)
		benchmark_mode: BenchmarkMode (FAST, FULL, OVERNIGHT)
		cpu_cores: Override CPU core count
		gpu_cores: Override GPU core count

	Returns:
		Tuple of (population_size, iterations/generations)

	Examples:
		>>> get_batch_size_for_mode(AccelerationMode.CPU, BenchmarkMode.FAST)
		(16, 10)
		>>> get_batch_size_for_mode(AccelerationMode.HYBRID, BenchmarkMode.FAST)
		(56, 10)
		>>> get_batch_size_for_mode(AccelerationMode.HYBRID, BenchmarkMode.FULL)
		(112, 100)
	"""
	cores = get_effective_cores(mode, cpu_cores, gpu_cores)

	match benchmark_mode:
		case BenchmarkMode.FAST:
			return (cores, 10)
		case BenchmarkMode.FULL:
			return (cores * 2, 100)
		case BenchmarkMode.OVERNIGHT:
			return (cores * 3, 1000)
		case _:
			return (cores, 10)


# =============================================================================
# Component Imports
# =============================================================================

# Base classes for RAM components
from wnn.ram.core.base import RAMComponent, RAMClusterBase, RAMSequenceModel, RAMTrainable

# Fundamental storage and layer
from wnn.ram.core.Memory import Memory
from wnn.ram.core.SparseMemory import SparseMemory
from wnn.ram.core.RAMLayer import RAMLayer
from wnn.ram.core.RAMClusterLayer import RAMClusterLayer, bits_needed, MemoryBackend
from wnn.ram.core.TieredRAMClusterLayer import TieredRAMClusterLayer, TierConfig
from wnn.ram.core.AdaptiveClusteredRAM import AdaptiveClusteredRAM, ConfigGroup

# Gating mechanisms (Engram-inspired content-based filtering)
from wnn.ram.core.gating import (
	GatingModel,
	RAMGating,
	SoftRAMGating,
	RustRAMGating,
	create_gating,
	compute_beneficial_gates,
	gating_metal_available,
)

# Generalization components
from wnn.ram.core.RAMGeneralization import (
	BitLevelMapper,
	CompositionalMapper,
	GeneralizingProjection,
	HashMapper,
	ResidualMapper,
	RecurrentParityMapper,
)

# Aggregation
from wnn.ram.core.RAMAggregator import RAMAggregator

# Recurrent networks
from wnn.ram.core.recurrent_network import RAMRecurrentNetwork
# StateMode now defined locally above
from wnn.ram.core.sequence import RAMSequence
from wnn.ram.core.multihead_sequence import RAMMultiHeadSequence
from wnn.ram.core.multihead_kv import RAMMultiHeadKV
from wnn.ram.core.multihead_shared import RAMMultiHeadShared
from wnn.ram.core.kv_transformer import RAMKVMemory
from wnn.ram.core.automaton import RAMAutomaton
from wnn.ram.core.trainer import (
	RAMTrainer,
	TrainingStats,
	LayerState,
	EpochStats,
	TrainingCallback,
	Checkpoint,
	# Enhanced curriculum learning
	CurriculumTrainer,
	CurriculumSchedule,
	length_difficulty,
	bit_count_difficulty,
	hamming_difficulty,
	combined_difficulty,
	# Multi-task learning
	MultiTaskTrainer,
	Task,
	# MixingStrategy now defined locally above
	# Contrastive learning
	ContrastiveTrainer,
	Triplet,
	hamming_distance,
	jaccard_similarity,
	normalized_hamming_similarity,
)
# LayerType, TrainingMode, TrainingPhase, MixingStrategy now defined locally above

# Serialization (use model.save() / Model.load() instead)
from wnn.ram.core.serialization import SERIALIZATION_VERSION

# Batch processing utilities
from wnn.ram.core.batch import (
	BatchProcessor,
	BatchResult,
	pad_sequences,
	collate_sequences,
	uncollate_batch,
)

# Generation result types and streaming functions
from wnn.ram.core.generation import (
	GenerationResult,
	BeamCandidate,
	StreamToken,
	stream_greedy_decode,
	stream_sample_decode,
	stream_top_k_decode,
	collect_stream,
)

# Sequence generator wrapper
from wnn.ram.core.sequence_generator import SequenceGenerator

# Factories (self-contained in core)
from wnn.ram.core.mapper_factory import MapperFactory

# Transformer components (submodule)
from wnn.ram.core import models

# Re-export transformer components at core level for convenience
from wnn.ram.core.models import (
	# Computed operations
	ComputedArithmeticFFN,
	ComputedCopyFFN,
	bits_to_int,
	int_to_bits,
	# Attention base classes
	AttentionBase,
	LearnableAttention,
	ComputedAttention,
	# Attention mechanisms
	SoftRAMAttention,
	ComputedSortingAttention,
	SortingAttention,  # Alias
	ComputedMinMaxAttention,
	MinMaxAttention,  # Alias
	RAMAttention,
	RAMCrossAttention,  # Alias
	CrossAttentionMode,
	PositionOnlyAttention,
	PositionPattern,
	# FFN
	TwoLayerFFN,
	RAMFeedForward,
	# Embeddings
	RAMEmbedding,
	PositionEncoding,
	# Transformer blocks
	RAMTransformerBlock,
	RAMTransformer,
	# Seq2Seq
	RAMSeq2Seq,
	RAMEncoderDecoder,
	# Language Models
	RAMLM,
)

# Reporting utilities
from wnn.ram.core.reporting import TierResultsTable, TierResultRow

# Content-dependent routing
from wnn.ram.core.routing import RouterRAM, RoutedRAMClusterLayer, RoutingStrategy


__all__ = [
	# ==== Core Enums (self-contained) ====
	'MemoryVal',
	'ContextMode',
	'BitMapperMode',
	'MapperStrategy',
	'LayerType',
	'TrainingMode',
	'TrainingPhase',
	'MixingStrategy',
	'StateMode',
	'OptimizationMethod',
	'BenchmarkMode',
	'MemoryMode',
	'AccelerationMode',
	# ==== Components ====
	# Base classes
	'RAMComponent',
	'RAMClusterBase',
	'RAMSequenceModel',
	'RAMTrainable',
	# Fundamental
	'Memory',
	'SparseMemory',
	'RAMLayer',
	'RAMClusterLayer',
	'TieredRAMClusterLayer',
	'TierConfig',
	'AdaptiveClusteredRAM',
	'ConfigGroup',
	'bits_needed',
	# Gating mechanisms (Engram-inspired)
	'GatingModel',
	'RAMGating',
	'SoftRAMGating',
	'RustRAMGating',
	'create_gating',
	'compute_beneficial_gates',
	# Generalization
	'BitLevelMapper',
	'CompositionalMapper',
	'GeneralizingProjection',
	'HashMapper',
	'ResidualMapper',
	# Factories
	'MapperFactory',
	# Aggregation
	'RAMAggregator',
	# Recurrent networks
	'RAMRecurrentNetwork',
	'RAMSequence',
	'RAMMultiHeadSequence',
	'RAMMultiHeadKV',
	'RAMMultiHeadShared',
	'RAMKVMemory',
	'RAMAutomaton',
	'RAMTrainer',
	'TrainingStats',
	'LayerState',
	'EpochStats',
	'TrainingCallback',
	'Checkpoint',
	# LayerType, TrainingMode, TrainingPhase now in Core Enums section above
	# Enhanced curriculum learning
	'CurriculumTrainer',
	'CurriculumSchedule',
	'length_difficulty',
	'bit_count_difficulty',
	'hamming_difficulty',
	'combined_difficulty',
	# Multi-task learning
	'MultiTaskTrainer',
	'Task',
	'MixingStrategy',
	# Contrastive learning
	'ContrastiveTrainer',
	'Triplet',
	'hamming_distance',
	'jaccard_similarity',
	'normalized_hamming_similarity',
	# Serialization (use model.save() / Model.load())
	'SERIALIZATION_VERSION',
	# Batch processing utilities
	'BatchProcessor',
	'BatchResult',
	'pad_sequences',
	'collate_sequences',
	'uncollate_batch',
	# Generation types and wrapper
	'GenerationResult',
	'BeamCandidate',
	'SequenceGenerator',
	# Submodules
	'models',
	# Transformer components (re-exported)
	'ComputedArithmeticFFN',
	'ComputedCopyFFN',
	'bits_to_int',
	'int_to_bits',
	'AttentionBase',
	'LearnableAttention',
	'ComputedAttention',
	'SoftRAMAttention',
	'ComputedSortingAttention',
	'SortingAttention',
	'ComputedMinMaxAttention',
	'MinMaxAttention',
	'RAMAttention',
	'RAMCrossAttention',
	'CrossAttentionMode',
	'PositionOnlyAttention',
	'PositionPattern',
	'TwoLayerFFN',
	'RAMFeedForward',
	'RAMEmbedding',
	'PositionEncoding',
	'RAMTransformerBlock',
	'RAMTransformer',
	'RAMSeq2Seq',
	'RAMEncoderDecoder',
	# Language Models
	'RAMLM',
	# Reporting utilities
	'TierResultsTable',
	'TierResultRow',
	# Content-dependent routing
	'RouterRAM',
	'RoutedRAMClusterLayer',
	'RoutingStrategy',
]
