"""
Batch Processing Utilities for RAM Networks

Provides utilities for processing multiple sequences in parallel:
- BatchProcessor: Wrapper that processes batches of sequences
- pad_sequences: Pad sequences to same length
- collate_sequences: Collate list of sequences into batch tensor

Usage:
    # Single sequence processing (current API)
    tokens = [tensor([0,1,0]), tensor([1,0,1]), tensor([0,0,1])]
    output = model(tokens)

    # Batch processing
    batch = [
        [tensor([0,1,0]), tensor([1,0,1])],  # seq 1
        [tensor([0,0,0]), tensor([1,1,1])],  # seq 2
    ]
    outputs = BatchProcessor.process(model, batch)
"""

from typing import TypeVar, Callable
from torch import Tensor, zeros, uint8, stack, cat
from torch.nn import Module
from dataclasses import dataclass


T = TypeVar('T', bound=Module)


@dataclass
class BatchResult:
    """Result of batch processing."""
    outputs: list[list[Tensor]]  # [batch_size][seq_len] tensors
    seq_lengths: list[int]       # Original sequence lengths


def pad_sequences(
    sequences: list[list[Tensor]],
    pad_value: int = 0,
) -> tuple[list[list[Tensor]], list[int]]:
    """
    Pad sequences to the same length.

    Args:
        sequences: List of sequences, each sequence is list of tokens
        pad_value: Value to use for padding

    Returns:
        padded: List of padded sequences
        lengths: Original lengths
    """
    if not sequences:
        return [], []

    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)

    if max_len == 0:
        return sequences, lengths

    # Get token shape from first non-empty sequence
    token_bits = None
    for seq in sequences:
        if seq:
            token_bits = len(seq[0])
            break

    if token_bits is None:
        return sequences, lengths

    # Pad
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padding = [zeros(token_bits, dtype=uint8).fill_(pad_value)
                      for _ in range(max_len - len(seq))]
            padded.append(seq + padding)
        else:
            padded.append(seq)

    return padded, lengths


def collate_sequences(
    sequences: list[list[Tensor]],
) -> Tensor:
    """
    Collate list of sequences into a batch tensor.

    Args:
        sequences: List of sequences (each same length), each token is [token_bits]

    Returns:
        Tensor of shape [batch_size, seq_len, token_bits]
    """
    if not sequences:
        return zeros(0, 0, 0, dtype=uint8)

    batch_size = len(sequences)
    seq_len = len(sequences[0]) if sequences else 0
    token_bits = len(sequences[0][0]) if sequences and sequences[0] else 0

    if seq_len == 0 or token_bits == 0:
        return zeros(batch_size, seq_len, token_bits, dtype=uint8)

    # Stack into [batch, seq_len, token_bits]
    batch = []
    for seq in sequences:
        # [seq_len, token_bits]
        seq_tensor = stack(seq)
        batch.append(seq_tensor)

    return stack(batch)


def uncollate_batch(
    batch: Tensor,
    lengths: list[int] | None = None,
) -> list[list[Tensor]]:
    """
    Convert batch tensor back to list of sequences.

    Args:
        batch: Tensor [batch_size, seq_len, token_bits]
        lengths: Original sequence lengths (to remove padding)

    Returns:
        List of sequences
    """
    batch_size = batch.shape[0]
    seq_len = batch.shape[1]

    sequences = []
    for i in range(batch_size):
        length = lengths[i] if lengths else seq_len
        seq = [batch[i, j] for j in range(length)]
        sequences.append(seq)

    return sequences


class BatchProcessor:
    """
    Wrapper for processing batches of sequences through RAM models.

    Most RAM models process one sequence at a time (list[Tensor]).
    This wrapper enables processing multiple sequences efficiently.
    """

    @staticmethod
    def process(
        model: Callable[[list[Tensor]], list[Tensor]],
        batch: list[list[Tensor]],
        parallel: bool = False,
    ) -> BatchResult:
        """
        Process a batch of sequences through a model.

        Args:
            model: Model that takes list[Tensor] and returns list[Tensor]
            batch: List of sequences to process
            parallel: If True, use parallel processing (requires multiprocessing)

        Returns:
            BatchResult with outputs and original lengths
        """
        lengths = [len(seq) for seq in batch]

        if parallel:
            # For RAM networks, parallel processing isn't beneficial
            # since they're already vectorized internally
            # But we could add this for CPU-bound models
            pass

        outputs = []
        for seq in batch:
            if seq:  # Skip empty sequences
                out = model(seq)
                outputs.append(out)
            else:
                outputs.append([])

        return BatchResult(outputs=outputs, seq_lengths=lengths)

    @staticmethod
    def process_padded(
        model: Callable[[list[Tensor]], list[Tensor]],
        batch: list[list[Tensor]],
        pad_value: int = 0,
    ) -> BatchResult:
        """
        Pad sequences to same length and process.

        Useful when model behavior depends on sequence length.

        Args:
            model: Model that takes list[Tensor] and returns list[Tensor]
            batch: List of variable-length sequences
            pad_value: Padding value

        Returns:
            BatchResult with outputs (padding removed)
        """
        padded, lengths = pad_sequences(batch, pad_value)

        outputs = []
        for seq, length in zip(padded, lengths):
            if length > 0:
                out = model(seq)
                # Remove padding from output
                outputs.append(out[:length])
            else:
                outputs.append([])

        return BatchResult(outputs=outputs, seq_lengths=lengths)


def batch_forward(
    layer,  # RAMLayer
    inputs: Tensor,
) -> Tensor:
    """
    Batch forward for RAMLayer.

    Args:
        layer: RAMLayer instance
        inputs: [batch_size, input_bits] tensor

    Returns:
        [batch_size, num_neurons] boolean outputs
    """
    return layer(inputs)


def batch_commit(
    layer,  # RAMLayer
    inputs: Tensor,
    targets: Tensor,
) -> int:
    """
    Batch commit for RAMLayer (sequential for correctness).

    Note: Commits are done sequentially because they modify shared memory.
    Parallel commits could cause conflicts.

    Args:
        layer: RAMLayer instance
        inputs: [batch_size, input_bits] tensor
        targets: [batch_size, num_neurons] tensor

    Returns:
        Number of successful commits
    """
    batch_size = inputs.shape[0]
    success_count = 0

    for i in range(batch_size):
        if layer.commit(inputs[i], targets[i]):
            success_count += 1

    return success_count
