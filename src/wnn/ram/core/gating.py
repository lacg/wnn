"""
RAM-based Gating Mechanisms for Cluster Layers

Implements content-based gating for RAM WNN cluster layers, inspired by
DeepSeek's Engram architecture. The key insight is that gating allows
selective filtering of cluster predictions based on input context.

Architecture:
    Input bits [B, input_bits]
           │
           ├──────────────────────────────────┐
           ▼                                  ▼
    ┌─────────────────┐          ┌─────────────────┐
    │  Base RAM WNN   │          │   RAMGating     │
    │  (existing)     │          │   (NEW)         │
    │  - Tiered arch  │          │  - 8 neurons/   │
    │  - Per-cluster  │          │    cluster      │
    │    scoring      │          │  - Majority     │
    │                 │          │    vote → gate  │
    └────────┬────────┘          └────────┬────────┘
             │ scores [B, C]              │ gates [B, C]
             │                            │
             └──────────┬─────────────────┘
                        ▼
                  scores × gates
                        │
                        ▼
                Gated predictions

Key Properties:
- Fully weightless (no nn.Embedding, no gradients)
- Binary gates (0 or 1) via majority voting
- Compatible with existing Rust accelerator (gating is post-eval)
- Staged training for stability (train RAM → freeze → train gating)

Usage:
    # Create gating model
    gating = RAMGating(
        total_input_bits=64,
        num_clusters=50257,
        neurons_per_gate=8,
        bits_per_neuron=12,
    )

    # Forward pass
    gates = gating.forward(input_bits)  # [B, num_clusters] binary

    # Training
    gating.train_step(input_bits, target_gates)
"""

from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor, zeros, ones, float32, long, bool as torch_bool
from torch import arange


class GatingModel(ABC):
    """
    Base class for gating mechanisms.

    Gating models learn which clusters should be active for each input context.
    This provides content-based filtering similar to attention mechanisms but
    using O(1) lookup rather than O(n²) attention computation.
    """

    @abstractmethod
    def forward(self, input_bits: Tensor) -> Tensor:
        """
        Compute gate values for each cluster.

        Args:
            input_bits: [B, total_input_bits] boolean tensor

        Returns:
            [B, num_clusters] gate values (0.0 or 1.0 for binary gates)
        """
        pass

    @abstractmethod
    def train_step(
        self,
        input_bits: Tensor,
        target_gates: Tensor,
        allow_override: bool = False,
    ) -> int:
        """
        Train the gating model on a batch of examples.

        Args:
            input_bits: [B, total_input_bits] input patterns
            target_gates: [B, num_clusters] desired gate values (0 or 1)
            allow_override: Whether to override existing non-EMPTY cells

        Returns:
            Number of memory cells modified
        """
        pass

    @property
    @abstractmethod
    def num_clusters(self) -> int:
        """Number of clusters this gating model covers."""
        pass

    def reset(self) -> None:
        """Reset all learned gates to initial state (all open)."""
        pass


class RAMGating(GatingModel):
    """
    Pure RAM-based gating (binary 0/1, fully weightless).

    Uses dedicated RAM neurons to learn which clusters should be active
    for each input context. The gate output is binary (0 or 1) determined
    by majority voting among the gate neurons.

    Architecture per cluster:
        - neurons_per_gate RAM neurons observe the input
        - Each neuron has bits_per_neuron connections (partial connectivity)
        - Gate = 1 if majority of neurons output TRUE, else 0

    This is analogous to Engram's multi-head structure but using:
        - RAM lookups instead of learned embeddings
        - Majority voting instead of sigmoid(q·k)
        - Binary output instead of continuous gates

    Example:
        gating = RAMGating(
            total_input_bits=64,      # 4 tokens × 16 bits
            num_clusters=50257,        # GPT-2 vocab
            neurons_per_gate=8,        # 8 neurons vote per cluster
            bits_per_neuron=12,        # Each neuron sees 12 bits
        )

        # Forward: returns binary gates
        gates = gating.forward(input_bits)  # [B, 50257], values in {0.0, 1.0}

        # Apply to cluster scores
        gated_scores = scores * gates
    """

    def __init__(
        self,
        total_input_bits: int,
        num_clusters: int,
        neurons_per_gate: int = 8,
        bits_per_neuron: int = 12,
        threshold: float = 0.5,
        rng: Optional[int] = None,
    ):
        """
        Initialize RAM-based gating.

        Args:
            total_input_bits: Number of input bits (context window × bits_per_token)
            num_clusters: Number of clusters to gate (vocabulary size)
            neurons_per_gate: Number of RAM neurons voting on each cluster's gate
                - More neurons = more robust voting but slower
                - Recommended: 8 (good balance of robustness and speed)
            bits_per_neuron: Address bits per gate neuron (partial connectivity)
                - More bits = finer discrimination but sparser training
                - Recommended: 12 (dense backend, good generalization)
            threshold: Fraction of neurons that must fire for gate=1 (default 0.5)
            rng: Random seed for connectivity initialization
        """
        from wnn.ram.core.RAMLayer import RAMLayer

        self._num_clusters = num_clusters
        self._neurons_per_gate = neurons_per_gate
        self._bits_per_neuron = bits_per_neuron
        self._threshold = threshold
        self._total_input_bits = total_input_bits

        # Total neurons = clusters × neurons_per_gate
        # Each cluster has its own set of gate neurons
        total_neurons = num_clusters * neurons_per_gate

        # Create the underlying RAM layer for gating
        # Uses same architecture as cluster layer neurons
        self._gate_layer = RAMLayer(
            total_input_bits=total_input_bits,
            num_neurons=total_neurons,
            n_bits_per_neuron=min(bits_per_neuron, total_input_bits),
            rng=rng,
        )

        # Threshold for majority voting (number of neurons that must fire)
        self._vote_threshold = int(neurons_per_gate * threshold)
        if self._vote_threshold == 0:
            self._vote_threshold = 1  # At least 1 neuron must fire

    @property
    def num_clusters(self) -> int:
        """Number of clusters this gating model covers."""
        return self._num_clusters

    @property
    def neurons_per_gate(self) -> int:
        """Number of neurons voting on each cluster's gate."""
        return self._neurons_per_gate

    @property
    def bits_per_neuron(self) -> int:
        """Address bits per gate neuron."""
        return self._bits_per_neuron

    @property
    def total_neurons(self) -> int:
        """Total number of gate neurons (clusters × neurons_per_gate)."""
        return self._num_clusters * self._neurons_per_gate

    def forward(self, input_bits: Tensor) -> Tensor:
        """
        Compute binary gates for each cluster via majority voting.

        Args:
            input_bits: [B, total_input_bits] boolean tensor

        Returns:
            [B, num_clusters] float tensor with values in {0.0, 1.0}
            - 1.0 = gate open (cluster prediction used)
            - 0.0 = gate closed (cluster prediction suppressed)
        """
        # Ensure 2D input
        if input_bits.ndim == 1:
            input_bits = input_bits.unsqueeze(0)

        batch_size = input_bits.shape[0]
        device = input_bits.device

        # Get raw neuron outputs [B, num_clusters * neurons_per_gate]
        # RAMLayer.forward returns boolean: TRUE cells → True, else False
        raw_outputs = self._gate_layer.forward(input_bits)  # [B, total_neurons]

        # Reshape to [B, num_clusters, neurons_per_gate]
        reshaped = raw_outputs.view(batch_size, self._num_clusters, self._neurons_per_gate)

        # Count TRUE outputs per cluster: [B, num_clusters]
        vote_counts = reshaped.sum(dim=-1).float()

        # Majority vote: gate=1 if count >= threshold
        gates = (vote_counts >= self._vote_threshold).float()

        return gates

    def forward_soft(self, input_bits: Tensor) -> Tensor:
        """
        Compute soft gates (continuous [0,1]) instead of binary.

        Useful for analysis/visualization or as a training signal.
        The soft gate value is the fraction of neurons that fired.

        Args:
            input_bits: [B, total_input_bits] boolean tensor

        Returns:
            [B, num_clusters] float tensor with values in [0.0, 1.0]
        """
        if input_bits.ndim == 1:
            input_bits = input_bits.unsqueeze(0)

        batch_size = input_bits.shape[0]

        raw_outputs = self._gate_layer.forward(input_bits)
        reshaped = raw_outputs.view(batch_size, self._num_clusters, self._neurons_per_gate)

        # Soft gate = fraction of neurons that fired
        soft_gates = reshaped.float().mean(dim=-1)

        return soft_gates

    def train_step(
        self,
        input_bits: Tensor,
        target_gates: Tensor,
        allow_override: bool = False,
    ) -> int:
        """
        Train gate neurons to produce desired gate patterns.

        Uses direct memory writes (EDRA-style): for each cluster, train
        its gate neurons to output TRUE if gate should be open, FALSE if closed.

        Args:
            input_bits: [B, total_input_bits] input patterns
            target_gates: [B, num_clusters] desired gate values (0 or 1, or bool)
            allow_override: Whether to override existing non-EMPTY cells

        Returns:
            Number of memory cells modified
        """
        if input_bits.ndim == 1:
            input_bits = input_bits.unsqueeze(0)
        if target_gates.ndim == 1:
            target_gates = target_gates.unsqueeze(0)

        batch_size = input_bits.shape[0]
        device = input_bits.device
        modified = 0

        # Convert target gates to boolean if needed
        if target_gates.dtype != torch_bool:
            target_gates = target_gates > 0.5

        # Get addresses for all gate neurons
        addresses = self._gate_layer.memory.get_addresses(input_bits)  # [B, total_neurons]

        # For each batch item
        for b in range(batch_size):
            # For each cluster
            for c in range(self._num_clusters):
                target = target_gates[b, c].item()

                # Get neuron indices for this cluster
                start_neuron = c * self._neurons_per_gate
                end_neuron = start_neuron + self._neurons_per_gate

                neuron_indices = arange(start_neuron, end_neuron, dtype=long, device=device)
                cluster_addresses = addresses[b, start_neuron:end_neuron]

                # Target bits: all neurons should output the target gate value
                target_bits = ones(self._neurons_per_gate, dtype=torch_bool, device=device) if target else zeros(self._neurons_per_gate, dtype=torch_bool, device=device)

                # Write to gate layer memory
                if self._gate_layer.memory.explore_batch(
                    neuron_indices, cluster_addresses, target_bits, allow_override
                ):
                    modified += self._neurons_per_gate

        return modified

    def train_batch_vectorized(
        self,
        input_bits: Tensor,
        target_gates: Tensor,
        allow_override: bool = False,
    ) -> int:
        """
        Fully vectorized batch training (faster than train_step for large batches).

        Args:
            input_bits: [B, total_input_bits] input patterns
            target_gates: [B, num_clusters] desired gate values
            allow_override: Whether to override existing non-EMPTY cells

        Returns:
            Number of memory cells modified
        """
        from torch import repeat_interleave

        if input_bits.ndim == 1:
            input_bits = input_bits.unsqueeze(0)
        if target_gates.ndim == 1:
            target_gates = target_gates.unsqueeze(0)

        batch_size = input_bits.shape[0]
        device = input_bits.device

        # Convert target gates to boolean
        if target_gates.dtype != torch_bool:
            target_gates = target_gates > 0.5

        # Get addresses for all neurons: [B, total_neurons]
        addresses = self._gate_layer.memory.get_addresses(input_bits)

        # Expand target gates to per-neuron: [B, num_clusters, neurons_per_gate]
        target_expanded = target_gates.unsqueeze(-1).expand(-1, -1, self._neurons_per_gate)

        # Flatten everything for batch write
        # neuron_indices: repeat [0,1,2,...,total_neurons-1] for each batch item
        neuron_base = arange(self.total_neurons, dtype=long, device=device)
        neuron_indices = neuron_base.unsqueeze(0).expand(batch_size, -1).flatten()

        # addresses: flatten [B, total_neurons] → [B * total_neurons]
        addresses_flat = addresses.flatten()

        # target bits: flatten [B, num_clusters, neurons_per_gate] → [B * total_neurons]
        target_bits_flat = target_expanded.flatten()

        # Single batch write
        modified = 0
        if self._gate_layer.memory.explore_batch(
            neuron_indices, addresses_flat, target_bits_flat, allow_override
        ):
            modified = batch_size * self.total_neurons

        return modified

    def reset(self) -> None:
        """Reset all gate memories to EMPTY (all gates initially open)."""
        self._gate_layer.reset_memory()

    def get_config(self) -> dict:
        """Get configuration dict for serialization."""
        return {
            'total_input_bits': self._total_input_bits,
            'num_clusters': self._num_clusters,
            'neurons_per_gate': self._neurons_per_gate,
            'bits_per_neuron': self._bits_per_neuron,
            'threshold': self._threshold,
        }

    @classmethod
    def from_config(cls, config: dict, rng: Optional[int] = None) -> 'RAMGating':
        """Create RAMGating from configuration dict."""
        return cls(
            total_input_bits=config['total_input_bits'],
            num_clusters=config['num_clusters'],
            neurons_per_gate=config['neurons_per_gate'],
            bits_per_neuron=config['bits_per_neuron'],
            threshold=config.get('threshold', 0.5),
            rng=rng,
        )

    def __repr__(self) -> str:
        return (
            f"RAMGating(clusters={self._num_clusters}, "
            f"neurons_per_gate={self._neurons_per_gate}, "
            f"bits_per_neuron={self._bits_per_neuron}, "
            f"threshold={self._threshold})"
        )


class SoftRAMGating(GatingModel):
    """
    Soft gating variant using continuous gate values [0, 1].

    Instead of binary majority voting, this uses the fraction of neurons
    that fired as a continuous gate signal. This can provide smoother
    gradients for optimization analysis but loses the pure binary property.

    Note: This is primarily for research/comparison purposes. For production
    use, prefer RAMGating with binary gates for true O(1) weightless behavior.
    """

    def __init__(
        self,
        total_input_bits: int,
        num_clusters: int,
        neurons_per_gate: int = 8,
        bits_per_neuron: int = 12,
        rng: Optional[int] = None,
    ):
        """Initialize soft RAM gating (same params as RAMGating)."""
        from wnn.ram.core.RAMLayer import RAMLayer

        self._num_clusters = num_clusters
        self._neurons_per_gate = neurons_per_gate
        self._bits_per_neuron = bits_per_neuron
        self._total_input_bits = total_input_bits

        total_neurons = num_clusters * neurons_per_gate

        self._gate_layer = RAMLayer(
            total_input_bits=total_input_bits,
            num_neurons=total_neurons,
            n_bits_per_neuron=min(bits_per_neuron, total_input_bits),
            rng=rng,
        )

    @property
    def num_clusters(self) -> int:
        return self._num_clusters

    def forward(self, input_bits: Tensor) -> Tensor:
        """
        Compute soft gates as fraction of neurons that fired.

        Args:
            input_bits: [B, total_input_bits] boolean tensor

        Returns:
            [B, num_clusters] float tensor with values in [0.0, 1.0]
        """
        if input_bits.ndim == 1:
            input_bits = input_bits.unsqueeze(0)

        batch_size = input_bits.shape[0]

        raw_outputs = self._gate_layer.forward(input_bits)
        reshaped = raw_outputs.view(batch_size, self._num_clusters, self._neurons_per_gate)

        # Soft gate = fraction of neurons that fired
        soft_gates = reshaped.float().mean(dim=-1)

        return soft_gates

    def train_step(
        self,
        input_bits: Tensor,
        target_gates: Tensor,
        allow_override: bool = False,
    ) -> int:
        """
        Train with soft targets (same implementation as RAMGating).

        For soft targets (values between 0 and 1), we threshold at 0.5
        to determine the target neuron output.
        """
        if input_bits.ndim == 1:
            input_bits = input_bits.unsqueeze(0)
        if target_gates.ndim == 1:
            target_gates = target_gates.unsqueeze(0)

        # Threshold soft targets
        binary_targets = target_gates > 0.5

        batch_size = input_bits.shape[0]
        device = input_bits.device
        modified = 0

        addresses = self._gate_layer.memory.get_addresses(input_bits)

        for b in range(batch_size):
            for c in range(self._num_clusters):
                target = binary_targets[b, c].item()

                start_neuron = c * self._neurons_per_gate
                end_neuron = start_neuron + self._neurons_per_gate

                neuron_indices = arange(start_neuron, end_neuron, dtype=long, device=device)
                cluster_addresses = addresses[b, start_neuron:end_neuron]

                target_bits = ones(self._neurons_per_gate, dtype=torch_bool, device=device) if target else zeros(self._neurons_per_gate, dtype=torch_bool, device=device)

                if self._gate_layer.memory.explore_batch(
                    neuron_indices, cluster_addresses, target_bits, allow_override
                ):
                    modified += self._neurons_per_gate

        return modified

    def reset(self) -> None:
        self._gate_layer.reset_memory()

    def __repr__(self) -> str:
        return (
            f"SoftRAMGating(clusters={self._num_clusters}, "
            f"neurons_per_gate={self._neurons_per_gate}, "
            f"bits_per_neuron={self._bits_per_neuron})"
        )


def compute_beneficial_gates(
    ungated_scores: Tensor,
    targets: Tensor,
    top_k: int = 10,
) -> Tensor:
    """
    Compute target gates that would improve predictions.

    Gate should be 1 (open) for clusters that help the prediction:
    - The correct target cluster
    - Clusters with high scores that are close to target

    Gate should be 0 (closed) for clusters that hurt:
    - High-scoring incorrect clusters (false positives)

    This is used during gating training to provide supervision.

    Args:
        ungated_scores: [B, num_clusters] raw cluster scores
        targets: [B] correct cluster indices
        top_k: Number of top clusters to consider for gating

    Returns:
        [B, num_clusters] binary target gates
    """
    batch_size = ungated_scores.shape[0]
    num_clusters = ungated_scores.shape[1]
    device = ungated_scores.device

    # Initialize all gates to 0 (closed)
    target_gates = zeros(batch_size, num_clusters, dtype=torch_bool, device=device)

    # Always open gate for correct cluster
    for b in range(batch_size):
        target_gates[b, targets[b]] = True

    # Optionally: open gates for top-k predictions
    # (helps model learn to keep good predictions)
    _, top_indices = ungated_scores.topk(top_k, dim=-1)
    for b in range(batch_size):
        for idx in top_indices[b]:
            # Only keep if it's the target or close to it
            # For now, simple heuristic: only keep target
            if idx == targets[b]:
                target_gates[b, idx] = True

    return target_gates.float()


class RustRAMGating(GatingModel):
    """
    Rust-accelerated RAM-based gating (faster than Python RAMGating).

    Uses the Rust ram_accelerator.RAMGatingWrapper for high-performance
    gating operations. Provides the same interface as RAMGating but runs
    significantly faster, especially for large batch sizes.

    This is the recommended gating implementation for production use.

    Example:
        # Falls back to Python RAMGating if Rust not available
        try:
            gating = RustRAMGating(
                total_input_bits=64,
                num_clusters=50257,
                neurons_per_gate=8,
                bits_per_neuron=12,
            )
        except ImportError:
            gating = RAMGating(...)  # Fallback
    """

    def __init__(
        self,
        total_input_bits: int,
        num_clusters: int,
        neurons_per_gate: int = 8,
        bits_per_neuron: int = 12,
        threshold: float = 0.5,
        rng: Optional[int] = None,
    ):
        """
        Initialize Rust-backed RAM gating.

        Args:
            total_input_bits: Number of input bits
            num_clusters: Number of clusters to gate
            neurons_per_gate: Neurons per gate (default 8)
            bits_per_neuron: Address bits per neuron (default 12)
            threshold: Majority voting threshold (default 0.5)
            rng: Random seed for connectivity
        """
        try:
            import ram_accelerator
        except ImportError:
            raise ImportError(
                "ram_accelerator not available. Build with: "
                "cd src/wnn/ram/strategies/accelerator && maturin develop --release"
            )

        self._num_clusters = num_clusters
        self._neurons_per_gate = neurons_per_gate
        self._bits_per_neuron = bits_per_neuron
        self._threshold = threshold
        self._total_input_bits = total_input_bits

        # Create Rust wrapper
        self._gating = ram_accelerator.RAMGatingWrapper(
            num_clusters=num_clusters,
            neurons_per_gate=neurons_per_gate,
            bits_per_neuron=bits_per_neuron,
            total_input_bits=total_input_bits,
            threshold=threshold,
            seed=rng,
        )

        # Store accelerator module reference
        self._accelerator = ram_accelerator

    @property
    def num_clusters(self) -> int:
        """Number of clusters this gating model covers."""
        return self._num_clusters

    @property
    def neurons_per_gate(self) -> int:
        """Number of neurons voting on each cluster's gate."""
        return self._neurons_per_gate

    @property
    def bits_per_neuron(self) -> int:
        """Address bits per gate neuron."""
        return self._bits_per_neuron

    @property
    def total_neurons(self) -> int:
        """Total number of gate neurons."""
        return self._gating.total_neurons()

    @staticmethod
    def metal_available() -> bool:
        """Check if Metal GPU acceleration is available for gating."""
        try:
            import ram_accelerator
            return ram_accelerator.gating_metal_available()
        except (ImportError, AttributeError):
            return False

    def forward(self, input_bits: Tensor) -> Tensor:
        """
        Compute binary gates via Rust accelerator (CPU with rayon parallelism).

        Args:
            input_bits: [B, total_input_bits] boolean tensor

        Returns:
            [B, num_clusters] float tensor with values in {0.0, 1.0}
        """
        import torch

        if input_bits.ndim == 1:
            input_bits = input_bits.unsqueeze(0)

        batch_size = input_bits.shape[0]
        device = input_bits.device

        # Convert to flat list for Rust
        input_flat = input_bits.flatten().tolist()

        # Call Rust forward
        gates_flat = self._gating.forward_batch(input_flat, batch_size)

        # Convert back to tensor
        return torch.tensor(gates_flat, device=device).view(batch_size, self._num_clusters)

    def forward_metal(self, input_bits: Tensor) -> Tensor:
        """
        Compute binary gates on Metal GPU (40 cores on M4 Max).

        Uses Metal compute shaders for parallel gate evaluation.
        Falls back to CPU if Metal is not available.

        Args:
            input_bits: [B, total_input_bits] boolean tensor

        Returns:
            [B, num_clusters] float tensor with values in {0.0, 1.0}
        """
        import torch

        if not self.metal_available():
            return self.forward(input_bits)

        if input_bits.ndim == 1:
            input_bits = input_bits.unsqueeze(0)

        batch_size = input_bits.shape[0]
        device = input_bits.device

        # Convert to flat list for Rust
        input_flat = input_bits.flatten().tolist()

        # Call Rust Metal forward
        gates_flat = self._gating.forward_batch_metal(input_flat, batch_size)

        # Convert back to tensor
        return torch.tensor(gates_flat, device=device).view(batch_size, self._num_clusters)

    def forward_hybrid(self, input_bits: Tensor, cpu_fraction: float = 0.3) -> Tensor:
        """
        Compute binary gates with hybrid CPU+GPU (56 cores total on M4 Max).

        Splits the batch between CPU (rayon, 16 cores) and GPU (Metal, 40 cores)
        for maximum throughput. The cpu_fraction parameter controls the split.

        Args:
            input_bits: [B, total_input_bits] boolean tensor
            cpu_fraction: Fraction of batch to process on CPU (default 0.3)
                         - 0.3 is optimal for M4 Max (16 CPU / 56 total)
                         - Set to 0.0 for GPU-only, 1.0 for CPU-only

        Returns:
            [B, num_clusters] float tensor with values in {0.0, 1.0}
        """
        import torch

        if not self.metal_available():
            return self.forward(input_bits)

        if input_bits.ndim == 1:
            input_bits = input_bits.unsqueeze(0)

        batch_size = input_bits.shape[0]
        device = input_bits.device

        # Convert to flat list for Rust
        input_flat = input_bits.flatten().tolist()

        # Call Rust hybrid forward
        gates_flat = self._gating.forward_batch_hybrid(input_flat, batch_size, cpu_fraction)

        # Convert back to tensor
        return torch.tensor(gates_flat, device=device).view(batch_size, self._num_clusters)

    def train_step(
        self,
        input_bits: Tensor,
        target_gates: Tensor,
        allow_override: bool = False,
    ) -> int:
        """
        Train gating via Rust accelerator.

        Args:
            input_bits: [B, total_input_bits] input patterns
            target_gates: [B, num_clusters] desired gate values
            allow_override: Whether to override existing cells

        Returns:
            Number of memory cells modified
        """
        if input_bits.ndim == 1:
            input_bits = input_bits.unsqueeze(0)
        if target_gates.ndim == 1:
            target_gates = target_gates.unsqueeze(0)

        batch_size = input_bits.shape[0]

        # Convert to flat lists
        input_flat = input_bits.flatten().tolist()
        target_flat = (target_gates > 0.5).flatten().tolist()

        # Call Rust train
        return self._gating.train_batch(input_flat, target_flat, batch_size, allow_override)

    def train_from_targets(
        self,
        input_bits: Tensor,
        targets: Tensor,
    ) -> int:
        """
        Train gating directly from target cluster indices.

        This is a convenience method that computes target gates automatically
        (gate=1 only for the target cluster).

        Args:
            input_bits: [B, total_input_bits] input patterns
            targets: [B] target cluster indices

        Returns:
            Number of memory cells modified
        """
        if input_bits.ndim == 1:
            input_bits = input_bits.unsqueeze(0)
        if targets.ndim == 0:
            targets = targets.unsqueeze(0)

        batch_size = input_bits.shape[0]

        # Convert to flat lists
        input_flat = input_bits.flatten().tolist()
        target_list = targets.tolist()

        # Compute target gates using Rust utility
        target_gates = self._accelerator.compute_target_gates(target_list, self._num_clusters)

        # Train
        return self._gating.train_batch(input_flat, target_gates, batch_size, False)

    def reset(self) -> None:
        """Reset all gate memories to EMPTY."""
        self._gating.reset()

    def memory_stats(self) -> tuple[int, int, int]:
        """Get memory statistics (empty, false, true counts)."""
        return self._gating.memory_stats()

    def export_memory(self) -> bytes:
        """Export memory state for serialization."""
        return bytes(self._gating.export_memory())

    def import_memory(self, data: bytes) -> None:
        """Import memory state from bytes."""
        self._gating.import_memory(list(data))

    def get_config(self) -> dict:
        """Get configuration dict for serialization."""
        return {
            'total_input_bits': self._total_input_bits,
            'num_clusters': self._num_clusters,
            'neurons_per_gate': self._neurons_per_gate,
            'bits_per_neuron': self._bits_per_neuron,
            'threshold': self._threshold,
        }

    @classmethod
    def from_config(cls, config: dict, rng: Optional[int] = None) -> 'RustRAMGating':
        """Create RustRAMGating from configuration dict."""
        return cls(
            total_input_bits=config['total_input_bits'],
            num_clusters=config['num_clusters'],
            neurons_per_gate=config['neurons_per_gate'],
            bits_per_neuron=config['bits_per_neuron'],
            threshold=config.get('threshold', 0.5),
            rng=rng,
        )

    def __repr__(self) -> str:
        return (
            f"RustRAMGating(clusters={self._num_clusters}, "
            f"neurons_per_gate={self._neurons_per_gate}, "
            f"bits_per_neuron={self._bits_per_neuron}, "
            f"threshold={self._threshold})"
        )


def gating_metal_available() -> bool:
    """
    Check if Metal GPU acceleration is available for gating.

    Returns:
        True if Metal gating is available, False otherwise
    """
    try:
        import ram_accelerator
        return ram_accelerator.gating_metal_available()
    except (ImportError, AttributeError):
        return False


def create_gating(
    total_input_bits: int,
    num_clusters: int,
    neurons_per_gate: int = 8,
    bits_per_neuron: int = 12,
    threshold: float = 0.5,
    rng: Optional[int] = None,
    prefer_rust: bool = True,
    accel: str = 'cpu',
) -> GatingModel:
    """
    Factory function to create the best available gating model.

    Attempts to use RustRAMGating (faster) if available, otherwise
    falls back to Python RAMGating.

    Args:
        total_input_bits: Number of input bits
        num_clusters: Number of clusters to gate
        neurons_per_gate: Neurons per gate (default 8)
        bits_per_neuron: Address bits per neuron (default 12)
        threshold: Majority voting threshold (default 0.5)
        rng: Random seed
        prefer_rust: If True, try Rust first (default True)
        accel: Acceleration mode ('cpu', 'metal', or 'hybrid')
               - 'cpu': Use CPU (rayon parallelism, 16 cores on M4 Max)
               - 'metal': Use Metal GPU (40 cores on M4 Max)
               - 'hybrid': Use both CPU+GPU (56 cores total on M4 Max)
               Note: accel only affects forward() behavior in RustRAMGating.
               For Python RAMGating, accel is ignored.

    Returns:
        GatingModel instance (RustRAMGating or RAMGating)
    """
    if prefer_rust:
        try:
            gating = RustRAMGating(
                total_input_bits=total_input_bits,
                num_clusters=num_clusters,
                neurons_per_gate=neurons_per_gate,
                bits_per_neuron=bits_per_neuron,
                threshold=threshold,
                rng=rng,
            )
            # Store acceleration mode for convenience
            gating._accel = accel
            return gating
        except ImportError:
            pass

    return RAMGating(
        total_input_bits=total_input_bits,
        num_clusters=num_clusters,
        neurons_per_gate=neurons_per_gate,
        bits_per_neuron=bits_per_neuron,
        threshold=threshold,
        rng=rng,
    )
