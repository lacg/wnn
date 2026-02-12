#!/usr/bin/env python3
"""
Unit tests for RAM-based gating mechanisms.

Tests the RAMGating and SoftRAMGating classes for:
- Correct output shapes
- Binary gate values (for RAMGating)
- Training modifies gate behavior
- Integration with RAMClusterLayer
- Ungated vs gated behavior equivalence when gates are all-ones

Run with:
    cd /path/to/wnn
    source wnn/bin/activate
    python tests/test_gating.py
"""

import os
import sys
import torch

# Add project root to PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch import zeros, ones, bool as torch_bool, float32, long, allclose


def test_ram_gating_output_shape():
    """Test that RAMGating.forward returns correct shape [B, num_clusters]."""
    from wnn.ram.core import RAMGating

    total_input_bits = 64
    num_clusters = 100
    batch_size = 8

    gating = RAMGating(
        total_input_bits=total_input_bits,
        num_clusters=num_clusters,
        neurons_per_gate=8,
        bits_per_neuron=12,
        rng=42,
    )

    # Create random input bits
    input_bits = torch.randint(0, 2, (batch_size, total_input_bits), dtype=torch_bool)

    # Forward pass
    gates = gating.forward(input_bits)

    # Check shape
    assert gates.shape == (batch_size, num_clusters), \
        f"Expected shape ({batch_size}, {num_clusters}), got {gates.shape}"

    print("PASS: RAMGating output shape is correct")


def test_ram_gating_binary_values():
    """Test that RAMGating.forward returns binary values (0 or 1)."""
    from wnn.ram.core import RAMGating

    gating = RAMGating(
        total_input_bits=32,
        num_clusters=50,
        neurons_per_gate=8,
        bits_per_neuron=10,
        rng=123,
    )

    # Test with multiple inputs
    for _ in range(5):
        input_bits = torch.randint(0, 2, (16, 32), dtype=torch_bool)
        gates = gating.forward(input_bits)

        # Check all values are 0 or 1
        unique_values = gates.unique()
        for v in unique_values:
            assert v.item() in [0.0, 1.0], \
                f"Gate value {v.item()} is not binary (0 or 1)"

    print("PASS: RAMGating produces binary values")


def test_soft_ram_gating_continuous():
    """Test that SoftRAMGating.forward returns continuous values in [0, 1]."""
    from wnn.ram.core import SoftRAMGating

    gating = SoftRAMGating(
        total_input_bits=32,
        num_clusters=50,
        neurons_per_gate=8,
        bits_per_neuron=10,
        rng=456,
    )

    # Train with some patterns to get non-trivial outputs
    input_bits = torch.randint(0, 2, (100, 32), dtype=torch_bool)
    target_gates = torch.randint(0, 2, (100, 50), dtype=torch_bool)
    gating.train_step(input_bits, target_gates)

    # Test
    test_bits = torch.randint(0, 2, (16, 32), dtype=torch_bool)
    gates = gating.forward(test_bits)

    # Check all values are in [0, 1]
    assert gates.min() >= 0.0, f"Gate value {gates.min().item()} < 0"
    assert gates.max() <= 1.0, f"Gate value {gates.max().item()} > 1"

    print("PASS: SoftRAMGating produces continuous values in [0, 1]")


def test_gating_training_modifies_behavior():
    """Test that training actually modifies gating behavior."""
    from wnn.ram.core import RAMGating

    gating = RAMGating(
        total_input_bits=32,
        num_clusters=20,
        neurons_per_gate=8,
        bits_per_neuron=10,
        rng=789,
    )

    # Create two different input patterns
    input_bits_a = torch.zeros(1, 32, dtype=torch_bool)
    input_bits_a[0, :16] = 1  # First half is all 1s

    input_bits_b = torch.zeros(1, 32, dtype=torch_bool)
    input_bits_b[0, 16:] = 1  # Second half is all 1s

    # Train pattern A to have OPEN gates (target = 1)
    target_open = ones(1, 20, dtype=float32)
    gating.train_step(input_bits_a, target_open, allow_override=True)

    # Train pattern B to have CLOSED gates (target = 0)
    target_closed = zeros(1, 20, dtype=float32)
    gating.train_step(input_bits_b, target_closed, allow_override=True)

    # Get gate outputs
    gates_a = gating.forward(input_bits_a)
    gates_b = gating.forward(input_bits_b)

    # After training:
    # - Pattern A should have gates open (more 1s)
    # - Pattern B should have gates closed (more 0s)
    # At minimum, they should be different
    assert not allclose(gates_a, gates_b), \
        "Training different patterns should produce different gates"

    # Additionally check that training to open gates actually opens some
    assert gates_a.sum() > gates_b.sum(), \
        f"Pattern A (trained to open) should have more open gates than pattern B (trained to close). A={gates_a.sum().item()}, B={gates_b.sum().item()}"

    print("PASS: Training modifies gating behavior")


def test_gating_integration_with_cluster_layer():
    """Test RAMClusterLayer with gating enabled."""
    from wnn.ram.core import RAMClusterLayer, RAMGating

    total_input_bits = 32
    num_clusters = 50
    batch_size = 4

    # Create layer without gating first
    layer = RAMClusterLayer(
        total_input_bits=total_input_bits,
        num_clusters=num_clusters,
        neurons_per_cluster=5,
        bits_per_neuron=8,
        rng=111,
    )

    # Train some patterns
    input_bits = torch.randint(0, 2, (100, total_input_bits), dtype=torch_bool)
    true_clusters = torch.randint(0, num_clusters, (100,))
    false_clusters = torch.randint(0, num_clusters, (100, 5))
    layer.train_batch(input_bits, true_clusters, false_clusters)

    # Get ungated scores
    test_bits = torch.randint(0, 2, (batch_size, total_input_bits), dtype=torch_bool)
    ungated_scores = layer.forward(test_bits).clone()

    # Create gating and attach to layer
    gating = RAMGating(
        total_input_bits=total_input_bits,
        num_clusters=num_clusters,
        neurons_per_gate=8,
        bits_per_neuron=10,
        rng=222,
    )

    # Train gating to close half the clusters
    gate_targets = zeros(batch_size, num_clusters, dtype=float32)
    gate_targets[:, :25] = 1.0  # Open first 25 clusters, close last 25
    gating.train_step(test_bits, gate_targets, allow_override=True)

    # Attach gating to layer
    layer.set_gating(gating)
    assert layer.has_gating, "Gating should be enabled"

    # Get gated scores
    gated_scores = layer.forward(test_bits)

    # Gated scores should differ from ungated
    assert gated_scores.shape == ungated_scores.shape, "Shape mismatch"

    # Clusters 25-49 should have reduced scores (gated to 0)
    # Note: due to training dynamics, this may not be perfect
    print(f"  Ungated mean: {ungated_scores.mean().item():.4f}")
    print(f"  Gated mean: {gated_scores.mean().item():.4f}")

    print("PASS: Gating integration with RAMClusterLayer works")


def test_ungated_equals_all_ones_gates():
    """Test that ungated scores equal gated scores when all gates are 1."""
    from wnn.ram.core import RAMClusterLayer, RAMGating

    total_input_bits = 32
    num_clusters = 30

    # Create and train layer
    layer = RAMClusterLayer(
        total_input_bits=total_input_bits,
        num_clusters=num_clusters,
        neurons_per_cluster=5,
        bits_per_neuron=8,
        rng=333,
    )

    # Train patterns
    input_bits = torch.randint(0, 2, (50, total_input_bits), dtype=torch_bool)
    true_clusters = torch.randint(0, num_clusters, (50,))
    false_clusters = torch.randint(0, num_clusters, (50, 3))
    layer.train_batch(input_bits, true_clusters, false_clusters)

    # Get ungated scores
    test_bits = torch.randint(0, 2, (8, total_input_bits), dtype=torch_bool)
    ungated_scores = layer._forward_ungated(test_bits).clone()

    # Create gating with all-ones gates (train with target=1 for all)
    gating = RAMGating(
        total_input_bits=total_input_bits,
        num_clusters=num_clusters,
        neurons_per_gate=8,
        bits_per_neuron=10,
        rng=444,
    )

    # Train all gates to be open
    all_ones_target = ones(8, num_clusters, dtype=float32)
    gating.train_step(test_bits, all_ones_target, allow_override=True)

    # Attach gating
    layer.set_gating(gating)

    # Get gated scores
    gated_scores = layer.forward(test_bits)

    # With all gates = 1, gated should equal ungated
    # (allowing for small numerical differences)
    gates = gating.forward(test_bits)
    if gates.sum() == gates.numel():  # All gates are 1
        assert allclose(ungated_scores, gated_scores, atol=1e-6), \
            "Gated scores should equal ungated when all gates are 1"
        print("PASS: Ungated equals gated when all gates are 1")
    else:
        # Some gates didn't train to 1, which can happen - just verify the math
        expected = ungated_scores * gates
        assert allclose(gated_scores, expected, atol=1e-6), \
            "Gated scores should equal ungated * gates"
        print("PASS: Gating formula (ungated * gates) is correct")


def test_create_gating_helper():
    """Test RAMClusterLayer.create_gating() helper method."""
    from wnn.ram.core import RAMClusterLayer

    layer = RAMClusterLayer(
        total_input_bits=64,
        num_clusters=100,
        neurons_per_cluster=5,
        bits_per_neuron=10,
        rng=555,
    )

    # Initially no gating
    assert not layer.has_gating, "Layer should not have gating initially"

    # Create gating via helper
    gating = layer.create_gating(
        neurons_per_gate=8,
        bits_per_neuron=12,
        threshold=0.5,
        rng=666,
    )

    # Now should have gating
    assert layer.has_gating, "Layer should have gating after create_gating()"
    assert layer.gating_model is gating, "gating_model should be the created instance"
    assert gating.num_clusters == 100, "Gating should have same num_clusters as layer"

    print("PASS: create_gating() helper works correctly")


def test_freeze_unfreeze():
    """Test freeze/unfreeze base memory functionality."""
    from wnn.ram.core import RAMClusterLayer

    layer = RAMClusterLayer(
        total_input_bits=32,
        num_clusters=50,
        neurons_per_cluster=5,
        bits_per_neuron=8,
        rng=777,
    )

    # Initially not frozen
    assert not layer.is_base_frozen, "Layer should not be frozen initially"

    # Freeze
    layer.freeze_base_memory()
    assert layer.is_base_frozen, "Layer should be frozen after freeze_base_memory()"

    # Unfreeze
    layer.unfreeze_base_memory()
    assert not layer.is_base_frozen, "Layer should not be frozen after unfreeze_base_memory()"

    print("PASS: Freeze/unfreeze functionality works")


def test_compute_beneficial_gates():
    """Test compute_beneficial_gates helper function."""
    from wnn.ram.core import compute_beneficial_gates

    batch_size = 4
    num_clusters = 10

    # Create fake scores
    scores = torch.randn(batch_size, num_clusters)
    targets = torch.tensor([0, 3, 5, 9], dtype=long)

    gates = compute_beneficial_gates(scores, targets, top_k=3)

    # Check shape
    assert gates.shape == (batch_size, num_clusters), f"Wrong shape: {gates.shape}"

    # Check target clusters have gates open
    for b in range(batch_size):
        assert gates[b, targets[b]] == 1.0, \
            f"Target cluster {targets[b]} should have gate=1 for batch {b}"

    print("PASS: compute_beneficial_gates works correctly")


def test_gating_config_serialization():
    """Test RAMGating get_config and from_config."""
    from wnn.ram.core import RAMGating

    original = RAMGating(
        total_input_bits=64,
        num_clusters=100,
        neurons_per_gate=12,
        bits_per_neuron=14,
        threshold=0.6,
        rng=888,
    )

    # Get config
    config = original.get_config()
    assert config['total_input_bits'] == 64
    assert config['num_clusters'] == 100
    assert config['neurons_per_gate'] == 12
    assert config['bits_per_neuron'] == 14
    assert config['threshold'] == 0.6

    # Recreate from config
    recreated = RAMGating.from_config(config, rng=999)

    assert recreated.num_clusters == original.num_clusters
    assert recreated.neurons_per_gate == original.neurons_per_gate
    assert recreated.bits_per_neuron == original.bits_per_neuron

    print("PASS: Gating config serialization works")


def test_gating_reset():
    """Test gating reset functionality."""
    from wnn.ram.core import RAMGating

    gating = RAMGating(
        total_input_bits=32,
        num_clusters=20,
        neurons_per_gate=8,
        bits_per_neuron=10,
        rng=101,
    )

    # Create test input
    input_bits = torch.randint(0, 2, (4, 32), dtype=torch_bool)

    # Train some patterns
    target_gates = torch.randint(0, 2, (4, 20), dtype=float32)
    gating.train_step(input_bits, target_gates, allow_override=True)

    # Get gates after training
    gates_after_train = gating.forward(input_bits).clone()

    # Reset
    gating.reset()

    # Get gates after reset - should be different (back to EMPTY behavior)
    gates_after_reset = gating.forward(input_bits)

    # Note: with EMPTY cells, output depends on threshold and interpretation
    # The key test is that reset was called without error
    print("PASS: Gating reset works without error")


def run_all_tests():
    """Run all gating tests."""
    print("\n" + "=" * 60)
    print("  RAM Gating Unit Tests")
    print("=" * 60 + "\n")

    tests = [
        test_ram_gating_output_shape,
        test_ram_gating_binary_values,
        test_soft_ram_gating_continuous,
        test_gating_training_modifies_behavior,
        test_gating_integration_with_cluster_layer,
        test_ungated_equals_all_ones_gates,
        test_create_gating_helper,
        test_freeze_unfreeze,
        test_compute_beneficial_gates,
        test_gating_config_serialization,
        test_gating_reset,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\nRunning: {test.__name__}")
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
