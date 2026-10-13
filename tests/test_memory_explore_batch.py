"""Memory.explore_batch: every masked write lands (regression for the same-word clobber).

Each cell write used to rebuild its 31-cell word from the PRE-batch word, so two
cells of one word written in one batch kept only the last one.
"""
import torch

from wnn.ram.core import MemoryVal
from wnn.ram.core.Memory import Memory


def _memory(neurons: int = 2, bits: int = 6) -> Memory:
	return Memory(total_input_bits=8, num_neurons=neurons, n_bits_per_neuron=bits, rng=1)


def test_two_cells_of_one_word_both_land():
	m = _memory()
	m.explore_batch(torch.tensor([0, 0]), torch.tensor([0, 1]), torch.tensor([True, False]))
	assert (m.get_memory(0, 0), m.get_memory(0, 1)) == (MemoryVal.TRUE, MemoryVal.FALSE)


def test_batch_equals_sequential_explore_on_distinct_cells():
	gen = torch.Generator().manual_seed(3)
	cells = torch.randperm(2 * 64, generator=gen)[:90]  # distinct (neuron, address) cells
	neurons, addresses = cells // 64, cells % 64
	bits = torch.randint(0, 2, cells.shape, generator=gen).bool()
	batch, seq = _memory(), _memory()
	batch.explore_batch(neurons, addresses, bits)
	for n, a, b in zip(neurons.tolist(), addresses.tolist(), bits.tolist()):
		seq.explore(n, a, b)
	assert torch.equal(batch.memory_words, seq.memory_words)


def test_repeated_cell_keeps_last_write_and_other_words_untouched():
	m = _memory()
	m.explore_batch(torch.tensor([1, 1, 1]), torch.tensor([40, 40, 2]), torch.tensor([True, False, True]), allow_override=True)
	assert m.get_memory(1, 40) == MemoryVal.FALSE
	assert m.get_memory(1, 2) == MemoryVal.TRUE
	assert all(m.get_memory(0, a) == MemoryVal.EMPTY for a in range(64))
