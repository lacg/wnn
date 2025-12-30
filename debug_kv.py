#!/usr/bin/env python3

from wnn.ram.decoders import OutputMode
from random import Random
from wnn.ram import RAMTransformer
from wnn.ram.architecture import KVSpec
from torch import cat

# Match the real test parameters
spec = KVSpec(k_bits=3, v_bits=2, query_value=0)
rng = Random(123)

model = RAMTransformer(
    spec=spec,
    neurons_per_head=8,
    n_bits_per_state_neuron=8,
    n_bits_per_output_neuron=4,
    use_hashing=False,  # Test without hashing
    hash_size=4096,
    rng=42,
    max_iters=100,
    # output_mode defaults to RAW
)

print(f"Model: {model.num_heads} heads, {model.neurons_per_head} neurons/head")
print(f"Total state neurons: {model.state_layer.num_neurons}")

# Train for 300 epochs and SAVE the episodes
print(f"\nTraining for 300 epochs (1 write per episode)...")
training_episodes = []
for epoch in range(300):
    windows, target_bits, raw = spec.generate_episode(n_writes=1, rng=rng)  # Only 1 write!
    training_episodes.append((windows, target_bits, raw))
    model.train(windows, target_bits)
    if epoch % 50 == 49:
        print(f"  Epoch {epoch+1}/300 done")

# Test on the SAME training episodes (test memorization)
print(f"\nTesting on SAME 300 training episodes...")
correct = 0
for windows, target_bits, raw in training_episodes:
    out_bits = model.forward(cat([w[0] for w in windows], dim=0).reshape(1, -1))
    pred = spec.decode_value_bits(out_bits)
    expected = spec.oracle_last_write_value(raw)

    if pred == expected:
        correct += 1

print(f"Memorization accuracy: {correct}/300 = {correct/300:.1%}")

# Test on NEW episodes (test generalization)
print(f"\nTesting on 200 NEW episodes...")
correct = 0
for ep in range(200):
    windows, target_bits, raw = spec.generate_episode(n_writes=6, rng=rng)

    out_bits = model.forward(cat([w[0] for w in windows], dim=0).reshape(1, -1))
    pred = spec.decode_value_bits(out_bits)
    expected = spec.oracle_last_write_value(raw)

    if pred == expected:
        correct += 1

print(f"Generalization accuracy: {correct}/200 = {correct/200:.1%}")
