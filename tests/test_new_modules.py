"""
Tests for New RAM Core Modules

Tests for:
- Serialization (save/load)
- Generation (greedy, beam search, streaming)
- Batch processing
- Attention masking
- Sparse memory
- Sequence generator

Run with: python tests/test_new_modules.py
"""

import tempfile
import os
import sys
from pathlib import Path

from torch import zeros, ones, tensor, uint8, randint, manual_seed


def run_test(test_fn, name):
    """Run a test function and report result."""
    try:
        test_fn()
        print(f"  [PASS] {name}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Serialization Tests
# =============================================================================

def test_memory_save_load():
    """Test Memory save and load."""
    from wnn.ram.core import Memory

    manual_seed(42)
    mem = Memory(
        total_input_bits=8,
        num_neurons=4,
        n_bits_per_neuron=4,
        rng=42,
    )

    # Train some patterns
    input_bits = tensor([1, 0, 1, 0, 1, 1, 0, 0], dtype=uint8).unsqueeze(0)
    target_bits = tensor([1, 0, 1, 1], dtype=uint8).unsqueeze(0)
    mem.commit(input_bits, target_bits)

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "memory.pt")
        mem.save(path)
        assert os.path.exists(path), "File not created"

        # Load
        loaded = Memory.load(path)
        assert loaded.total_input_bits == mem.total_input_bits
        assert loaded.num_neurons == mem.num_neurons

        # Verify same output
        orig_out = mem.forward(input_bits)
        loaded_out = loaded.forward(input_bits)
        assert (orig_out == loaded_out).all(), "Outputs differ after load"


def test_ramlayer_save_load():
    """Test RAMLayer save and load."""
    from wnn.ram.core import RAMLayer

    layer = RAMLayer(
        total_input_bits=6,
        num_neurons=3,
        n_bits_per_neuron=3,
        rng=42,
    )

    input_bits = tensor([1, 0, 1, 0, 1, 1], dtype=uint8).unsqueeze(0)
    target_bits = tensor([1, 0, 1], dtype=uint8).unsqueeze(0)
    layer.commit(input_bits, target_bits)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "layer.pt")
        layer.save(path)

        loaded = RAMLayer.load(path)
        orig_out = layer.forward(input_bits)
        loaded_out = loaded.forward(input_bits)
        assert (orig_out == loaded_out).all(), "Outputs differ after load"


# =============================================================================
# Generation Tests
# =============================================================================

def test_greedy_decode():
    """Test greedy decoding."""
    from wnn.ram.core.generation import greedy_decode

    # Simple model that just returns input
    def model(tokens, context=None):
        return [t.clone() for t in tokens]

    start = tensor([1, 0, 1, 0], dtype=uint8)
    result = greedy_decode(
        model=model,
        encoder_output=None,
        start_token=start,
        max_len=3,
    )

    assert len(result.tokens) == 3, f"Expected 3 tokens, got {len(result.tokens)}"
    assert result.tokens[0].shape == start.shape


def test_beam_search():
    """Test beam search decoding."""
    from wnn.ram.core.generation import beam_search

    def model(tokens, context=None):
        return [t.clone() for t in tokens]

    start = tensor([1, 0, 0, 0], dtype=uint8)
    result = beam_search(
        model=model,
        encoder_output=None,
        start_token=start,
        beam_width=2,
        max_len=3,
    )

    assert len(result.tokens) == 3
    assert result.candidates is not None
    assert len(result.candidates) <= 2


def test_stream_greedy_decode():
    """Test streaming greedy decoding."""
    from wnn.ram.core.generation import stream_greedy_decode, StreamToken

    def model(tokens, context=None):
        return [t.clone() for t in tokens]

    start = tensor([1, 0, 1, 0], dtype=uint8)
    tokens = list(stream_greedy_decode(
        model=model,
        encoder_output=None,
        start_token=start,
        max_len=5,
    ))

    assert len(tokens) == 5
    assert all(isinstance(t, StreamToken) for t in tokens)
    assert tokens[0].step == 0
    assert tokens[-1].step == 4


def test_stream_early_termination():
    """Test that streaming allows early termination."""
    from wnn.ram.core.generation import stream_greedy_decode

    def model(tokens, context=None):
        return [t.clone() for t in tokens]

    start = tensor([1, 0, 1, 0], dtype=uint8)
    stream = stream_greedy_decode(
        model=model,
        encoder_output=None,
        start_token=start,
        max_len=100,
    )

    # Only consume 3 tokens
    collected = []
    for i, token in enumerate(stream):
        collected.append(token)
        if i >= 2:
            break

    assert len(collected) == 3


def test_collect_stream():
    """Test collecting stream into GenerationResult."""
    from wnn.ram.core.generation import stream_greedy_decode, collect_stream

    def model(tokens, context=None):
        return [t.clone() for t in tokens]

    start = tensor([0, 1, 0, 1], dtype=uint8)
    stream = stream_greedy_decode(
        model=model,
        encoder_output=None,
        start_token=start,
        max_len=4,
    )

    result = collect_stream(stream)
    assert len(result.tokens) == 4


# =============================================================================
# Batch Processing Tests
# =============================================================================

def test_pad_sequences():
    """Test sequence padding."""
    from wnn.ram.core.batch import pad_sequences

    sequences = [
        [tensor([1, 0], dtype=uint8), tensor([0, 1], dtype=uint8)],
        [tensor([1, 1], dtype=uint8)],
        [tensor([0, 0], dtype=uint8), tensor([1, 0], dtype=uint8), tensor([0, 1], dtype=uint8)],
    ]

    padded, lengths = pad_sequences(sequences, pad_value=0)

    assert len(padded) == 3
    assert lengths == [2, 1, 3]
    assert len(padded[0]) == 3  # Max length
    assert len(padded[1]) == 3
    assert len(padded[2]) == 3


def test_collate_sequences():
    """Test collating sequences into batch tensor."""
    from wnn.ram.core.batch import collate_sequences

    sequences = [
        [tensor([1, 0, 0], dtype=uint8), tensor([0, 1, 0], dtype=uint8)],
        [tensor([1, 1, 1], dtype=uint8), tensor([0, 0, 0], dtype=uint8)],
    ]

    # Collate same-length sequences
    batch = collate_sequences(sequences)

    assert batch.shape == (2, 2, 3)  # (batch, seq_len, token_bits)


def test_uncollate_batch():
    """Test uncollating batch back to sequences."""
    from wnn.ram.core.batch import collate_sequences, uncollate_batch

    # Use same-length sequences for collate
    sequences = [
        [tensor([1, 0], dtype=uint8), tensor([0, 1], dtype=uint8)],
        [tensor([1, 1], dtype=uint8), tensor([0, 0], dtype=uint8)],
    ]

    batch = collate_sequences(sequences)
    lengths = [2, 2]  # Same length sequences
    recovered = uncollate_batch(batch, lengths)

    assert len(recovered) == 2
    assert len(recovered[0]) == 2
    assert len(recovered[1]) == 2


# =============================================================================
# Attention Masking Tests
# =============================================================================

def test_causal_mask():
    """Test causal (autoregressive) mask."""
    from wnn.ram.core.transformers.attention_mask import AttentionMask, can_attend

    mask = AttentionMask.causal(seq_len=4)

    # Position 0 can only attend to itself
    assert can_attend(mask, 0, 0)
    assert not can_attend(mask, 0, 1)

    # Position 3 can attend to all positions
    assert can_attend(mask, 3, 0)
    assert can_attend(mask, 3, 1)
    assert can_attend(mask, 3, 2)
    assert can_attend(mask, 3, 3)


def test_bidirectional_mask():
    """Test bidirectional (full attention) mask."""
    from wnn.ram.core.transformers.attention_mask import AttentionMask, can_attend

    mask = AttentionMask.bidirectional(query_len=4)

    # All positions can attend to all other positions
    for i in range(4):
        for j in range(4):
            assert can_attend(mask, i, j)


def test_sliding_window_mask():
    """Test sliding window mask."""
    from wnn.ram.core.transformers.attention_mask import AttentionMask, can_attend

    mask = AttentionMask.sliding_window(seq_len=6, window_size=1)

    # Position 3 can attend to 2, 3, 4 (window of 1)
    assert not can_attend(mask, 3, 1)
    assert can_attend(mask, 3, 2)
    assert can_attend(mask, 3, 3)
    assert can_attend(mask, 3, 4)
    assert not can_attend(mask, 3, 5)


def test_combine_masks():
    """Test combining multiple masks."""
    from wnn.ram.core.transformers.attention_mask import AttentionMask, can_attend

    causal = AttentionMask.causal(seq_len=4)
    window = AttentionMask.sliding_window(seq_len=4, window_size=1)

    combined = AttentionMask.combine([causal, window], mode="and")

    # Position 3 can only attend to 2, 3 (intersection of causal and window)
    assert not can_attend(combined, 3, 0)  # Outside window
    assert not can_attend(combined, 3, 1)  # Outside window
    assert can_attend(combined, 3, 2)
    assert can_attend(combined, 3, 3)


# =============================================================================
# Sparse Memory Tests
# =============================================================================

def test_sparse_basic_read_write():
    """Test basic sparse memory operations."""
    from wnn.ram.core import SparseMemory

    mem = SparseMemory(
        total_input_bits=16,
        num_neurons=4,
        n_bits_per_neuron=8,
        rng=42,
    )

    # Initially empty
    assert mem.stored_count() == 0
    assert mem.density() == 0.0

    # Write some patterns
    input_bits = tensor([1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0], dtype=uint8).unsqueeze(0)
    target_bits = tensor([1, 0, 1, 1], dtype=uint8).unsqueeze(0)
    mem.commit(input_bits, target_bits)

    # Should have stored something
    assert mem.stored_count() > 0

    # Verify read
    output = mem.forward(input_bits)
    expected = target_bits.squeeze().to(bool)
    assert (output.squeeze() == expected).all()


def test_sparse_vs_dense_compatibility():
    """Test that sparse and dense memory produce same results."""
    from wnn.ram.core import Memory, SparseMemory

    manual_seed(42)

    # Create both with same config
    dense = Memory(
        total_input_bits=8,
        num_neurons=4,
        n_bits_per_neuron=4,
        rng=42,
    )

    sparse = SparseMemory(
        total_input_bits=8,
        num_neurons=4,
        n_bits_per_neuron=4,
        rng=42,
    )

    # Copy connections
    sparse.connections.copy_(dense.connections)

    # Train same patterns
    patterns = [
        (tensor([1, 0, 1, 0, 1, 1, 0, 0], dtype=uint8), tensor([1, 0, 1, 1], dtype=uint8)),
        (tensor([0, 1, 0, 1, 0, 0, 1, 1], dtype=uint8), tensor([0, 1, 0, 0], dtype=uint8)),
    ]

    for inp, tgt in patterns:
        dense.commit(inp.unsqueeze(0), tgt.unsqueeze(0))
        sparse.commit(inp.unsqueeze(0), tgt.unsqueeze(0))

    # Verify same outputs
    for inp, _ in patterns:
        dense_out = dense.forward(inp.unsqueeze(0))
        sparse_out = sparse.forward(inp.unsqueeze(0))
        assert (dense_out == sparse_out).all(), f"Mismatch for input {inp}"


def test_sparse_large_address_space():
    """Test sparse memory with large address space."""
    from wnn.ram.core import SparseMemory

    # 20 bits = 1M addresses per neuron
    mem = SparseMemory(
        total_input_bits=20,
        num_neurons=8,
        n_bits_per_neuron=20,
        rng=42,
    )

    # Write a few patterns
    for i in range(10):
        manual_seed(i)
        inp = randint(0, 2, (1, 20), dtype=uint8)
        tgt = randint(0, 2, (1, 8), dtype=uint8)
        mem.commit(inp, tgt)

    # Should have very low density
    assert mem.density() < 0.001  # Less than 0.1%
    assert mem.stored_count() <= 80  # At most 10 patterns * 8 neurons


# =============================================================================
# Sequence Generator Tests
# =============================================================================

def test_sequence_generator_wrapper():
    """Test SequenceGenerator wrapper."""
    from wnn.ram.core import SequenceGenerator

    def model_fn(tokens, context=None):
        return [t.clone() for t in tokens]

    generator = SequenceGenerator(model=model_fn)

    start = tensor([1, 0, 1, 0], dtype=uint8)
    result = generator.decode(start, max_len=3)

    assert len(result.tokens) == 3


def test_streaming_generation():
    """Test streaming generation through SequenceGenerator."""
    from wnn.ram.core import SequenceGenerator

    def model_fn(tokens, context=None):
        return [t.clone() for t in tokens]

    generator = SequenceGenerator(model=model_fn)

    start = tensor([0, 1, 0, 1], dtype=uint8)
    tokens = list(generator.stream_decode(start, max_len=5))

    assert len(tokens) == 5
    assert all(t.step == i for i, t in enumerate(tokens))


def test_different_strategies():
    """Test different generation strategies."""
    from wnn.ram.core import SequenceGenerator

    def model_fn(tokens, context=None):
        return [t.clone() for t in tokens]

    generator = SequenceGenerator(model=model_fn)
    start = tensor([1, 1, 0, 0], dtype=uint8)

    # Greedy
    greedy_result = generator.decode(start, max_len=3)
    assert len(greedy_result.tokens) == 3

    # Beam search
    beam_result = generator.search(start, beam_width=2, max_len=3)
    assert len(beam_result.tokens) == 3
    assert beam_result.candidates is not None

    # Sampling
    sample_result = generator.sample(start, max_len=3, temperature=1.0)
    assert len(sample_result.tokens) == 3

    # Top-k
    topk_result = generator.top_k(start, max_len=3, k=2)
    assert len(topk_result.tokens) == 3


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Tests for New RAM Core Modules")
    print("=" * 60)

    all_tests = [
        # Serialization
        ("Serialization", [
            (test_memory_save_load, "Memory save/load"),
            (test_ramlayer_save_load, "RAMLayer save/load"),
        ]),
        # Generation
        ("Generation", [
            (test_greedy_decode, "Greedy decode"),
            (test_beam_search, "Beam search"),
            (test_stream_greedy_decode, "Stream greedy decode"),
            (test_stream_early_termination, "Stream early termination"),
            (test_collect_stream, "Collect stream"),
        ]),
        # Batch
        ("Batch Processing", [
            (test_pad_sequences, "Pad sequences"),
            (test_collate_sequences, "Collate sequences"),
            (test_uncollate_batch, "Uncollate batch"),
        ]),
        # Masking
        ("Attention Masking", [
            (test_causal_mask, "Causal mask"),
            (test_bidirectional_mask, "Bidirectional mask"),
            (test_sliding_window_mask, "Sliding window mask"),
            (test_combine_masks, "Combine masks"),
        ]),
        # Sparse Memory
        ("Sparse Memory", [
            (test_sparse_basic_read_write, "Basic read/write"),
            (test_sparse_vs_dense_compatibility, "Sparse vs dense compatibility"),
            (test_sparse_large_address_space, "Large address space"),
        ]),
        # Sequence Generator
        ("Sequence Generator", [
            (test_sequence_generator_wrapper, "Generator wrapper"),
            (test_streaming_generation, "Streaming generation"),
            (test_different_strategies, "Different strategies"),
        ]),
    ]

    total_passed = 0
    total_failed = 0

    for category, tests in all_tests:
        print(f"\n{category}:")
        for test_fn, name in tests:
            if run_test(test_fn, name):
                total_passed += 1
            else:
                total_failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {total_passed} passed, {total_failed} failed")
    print("=" * 60)

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
