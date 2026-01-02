"""
Strategy Configuration Dataclasses

Typed configuration objects for training and forward strategies.
Using dataclasses instead of **kwargs provides:
- Type safety and IDE autocomplete
- Clear documentation of available options
- Validation at construction time
- Easy serialization for reproducibility
"""

from dataclasses import dataclass, field
from typing import Callable, Any

from torch import Tensor


# =============================================================================
# FORWARD STRATEGY CONFIGS
# =============================================================================

@dataclass(frozen=True)
class ForwardConfig:
    """
    Base configuration for forward strategies.

    Attributes:
        max_length: Maximum output sequence length (None = input length)
        return_intermediates: Whether to return intermediate layer outputs
    """
    max_length: int | None = None
    return_intermediates: bool = False


@dataclass(frozen=True)
class AutoregressiveConfig(ForwardConfig):
    """
    Configuration for autoregressive (token-by-token) generation.

    Attributes:
        use_cache: Cache attention keys/values for efficiency
        stop_token: Token value that signals end of generation (None = no stopping)
    """
    use_cache: bool = True
    stop_token: Tensor | None = None


@dataclass(frozen=True)
class SamplingConfig(ForwardConfig):
    """
    Configuration for sampling-based generation.

    Attributes:
        temperature: Sampling temperature (1.0 = neutral, <1 = sharper, >1 = flatter)
        top_k: Only sample from top-k most likely tokens (None = all)
        top_p: Nucleus sampling threshold (None = disabled)
        seed: Random seed for reproducibility (None = random)
    """
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None


@dataclass(frozen=True)
class BeamSearchConfig(ForwardConfig):
    """
    Configuration for beam search generation.

    Attributes:
        beam_width: Number of beams to maintain
        length_penalty: Penalty for longer sequences (>1 = prefer shorter)
        early_stopping: Stop when all beams have finished
    """
    beam_width: int = 4
    length_penalty: float = 1.0
    early_stopping: bool = True


# =============================================================================
# TRAIN STRATEGY CONFIGS
# =============================================================================

@dataclass
class TrainConfig:
    """
    Base configuration for training strategies.

    Attributes:
        epochs: Maximum number of training epochs
        early_stop: Stop if model converges (zero errors)
        shuffle: Shuffle dataset each epoch
        verbose: Print training progress
    """
    epochs: int = 10
    early_stop: bool = True
    shuffle: bool = True
    verbose: bool = True


@dataclass
class GreedyTrainConfig(TrainConfig):
    """
    Configuration for greedy (single-pass) training.

    Greedy training updates all layers in one backward pass.
    Fast but may miss layer dependencies.
    """
    pass  # Uses base config


@dataclass
class IterativeTrainConfig(TrainConfig):
    """
    Configuration for iterative training.

    Iterative training runs multiple passes until layers stabilize.

    Attributes:
        max_iterations: Maximum iterations per example
        convergence_threshold: Stop when changes fall below this
    """
    max_iterations: int = 5
    convergence_threshold: float = 0.01


@dataclass
class CurriculumTrainConfig(TrainConfig):
    """
    Configuration for curriculum learning.

    Curriculum learning starts with easy examples and progresses to harder ones.

    Attributes:
        num_stages: Number of difficulty stages
        epochs_per_stage: Training epochs per stage
        overlap: Fraction of examples to overlap between stages (0-1)
        patience_per_stage: Early stop patience within each stage
        difficulty_metric: Function to compute example difficulty
    """
    num_stages: int = 5
    epochs_per_stage: int = 3
    overlap: float = 0.2
    patience_per_stage: int = 2
    difficulty_metric: Callable[[list[Tensor], list[Tensor]], float] | None = None


@dataclass
class ContrastiveTrainConfig(TrainConfig):
    """
    Configuration for contrastive learning.

    Contrastive learning trains on triplets (anchor, positive, negative)
    to improve class discrimination.

    Attributes:
        triplets_per_epoch: Maximum triplets to train on per epoch
        hard_negative_ratio: Fraction of hard negatives to sample (0-1)
        margin: Minimum similarity margin between positive and negative
        similarity_fn: Function to compute tensor similarity
    """
    triplets_per_epoch: int | None = None
    hard_negative_ratio: float = 0.5
    margin: float = 0.3
    similarity_fn: Callable[[Tensor, Tensor], float] | None = None


@dataclass
class MultiTaskTrainConfig(TrainConfig):
    """
    Configuration for multi-task learning.

    Multi-task learning trains on multiple tasks simultaneously.

    Attributes:
        mixing: How to mix examples from different tasks
        task_weights: Per-task sampling weights (None = equal)
    """
    mixing: str = "interleaved"  # "round_robin", "proportional", "weighted", "interleaved"
    task_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class ScheduledSamplingConfig(TrainConfig):
    """
    Configuration for scheduled sampling training.

    Scheduled sampling gradually transitions from teacher forcing
    (using ground truth) to using model predictions.

    Attributes:
        schedule: Decay schedule ("linear", "inverse_sigmoid", "exponential")
        start_prob: Initial probability of using ground truth
        end_prob: Final probability of using ground truth
        decay_rate: Rate of decay (interpretation depends on schedule)
    """
    schedule: str = "linear"
    start_prob: float = 1.0
    end_prob: float = 0.0
    decay_rate: float = 0.1


# =============================================================================
# PATIENCE & CALLBACKS
# =============================================================================

@dataclass
class PatienceConfig:
    """
    Configuration for early stopping with patience.

    Attributes:
        patience: Epochs without improvement before stopping
        min_delta: Minimum change to qualify as improvement
        monitor: Metric to monitor ("loss", "accuracy", "error")
        mode: Whether to minimize or maximize the metric
    """
    patience: int = 5
    min_delta: float = 0.0
    monitor: str = "error"
    mode: str = "min"  # "min" or "max"


@dataclass
class CheckpointConfig:
    """
    Configuration for model checkpointing.

    Attributes:
        save_best: Save best model according to monitored metric
        save_every: Save checkpoint every N epochs (None = disabled)
        path: Directory to save checkpoints
    """
    save_best: bool = True
    save_every: int | None = None
    path: str = "./checkpoints"
