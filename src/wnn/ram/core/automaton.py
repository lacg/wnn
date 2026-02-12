from torch.nn import Module
from torch import bool as tbool, cat, stack, Tensor, zeros

from wnn.ram.core import RAMLayer

# -------------------------
# RAMAutomaton fully vectorized
# -------------------------
class RAMAutomaton(Module):
	"""
	Fully vectorized RAM automaton:
		- input_layer: input_bits + prev_state_bits
		- state_layer: input_layer_out + prev_state_bits
		- outputs are torch.bool tensors
	"""
	def __init__(self, input_size_bits: int, state_size_bits: int, n_bits_input_neuron: int, n_bits_state_neuron: int, n_input_neurons: int, n_state_neurons: int, use_hashing: bool = False, hash_size: int = 1024, rng: int = None):
		super().__init__()
		self.input_size_bits = input_size_bits
		self.state_size_bits = state_size_bits

		self.n_input_neurons = n_input_neurons
		self.n_state_neurons = n_state_neurons

		self.input_layer = RAMLayer(total_input_bits=input_size_bits + state_size_bits, num_neurons=n_input_neurons, n_bits_per_neuron=n_bits_input_neuron, use_hashing=use_hashing, hash_size=hash_size, rng=rng)
		self.state_layer = RAMLayer(total_input_bits=n_input_neurons + state_size_bits, num_neurons=n_state_neurons, n_bits_per_neuron=n_bits_state_neuron, use_hashing=use_hashing, hash_size=hash_size, rng=None if rng is None else rng+1)

	def forward_step(self, input_bits: Tensor, prev_state_bits: Tensor):
		"""
		input_bits: [batch, input_size_bits] bool
		prev_state_bits: [batch, state_size_bits] bool
		returns: input_out [batch, n_input_neurons], next_state [batch, n_state_neurons]
		"""
		concat_input = cat([input_bits, prev_state_bits], dim=1)
		input_out = self.input_layer(concat_input)

		concat_state = cat([input_out, prev_state_bits], dim=1)
		next_state = self.state_layer(concat_state)
		return input_out, next_state

	def forward_sequence(self, input_seq_bits: Tensor, init_state: Tensor = None):
		"""
		input_seq_bits: [seq_len, batch, input_size_bits] bool
		init_state: [batch, state_size_bits] bool
		returns: input_out_seq: [seq_len, batch, n_input_neurons], final_state: [batch, n_state_neurons]
		"""
		seq_len, batch, _ = input_seq_bits.shape
		if init_state is None:
			prev_state = zeros(batch, self.state_size_bits, dtype=tbool, device=input_seq_bits.device)
		else:
			prev_state = init_state

		outs = []
		for t in range(seq_len):
			inp = input_seq_bits[t]
			input_out, next_state = self.forward_step(inp, prev_state)
			outs.append(input_out)
			prev_state = next_state
		return stack(outs, dim=0), prev_state

	def train_write_step(self, input_bits, prev_state_bits, target_input_bits, target_state_bits):
		concat_input = cat([input_bits, prev_state_bits], dim=1)
		self.input_layer.commit(concat_input, target_input_bits)

		concat_state = cat([target_input_bits, prev_state_bits], dim=1)
		self.state_layer.commit(concat_state, target_state_bits)
