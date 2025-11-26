from typing import Optional

from torch.nn import Module, ModuleList
from torch import bool as tbool, long, LongTensor, manual_seed, randint, stack, int64

from wnn.ram.RAMNeuron import RAMNeuron

# -------------------------
# RAMLayer: many RAMNeurons + connectivity
# -------------------------
class RAMLayerOriginal(Module):

	"""
	Layer of RAM neurons with external connectivity matrix.
	- total_input_bits: length of input vector this layer receives (could be input+state concat)
	- num_neurons: number of RAM neurons in this layer (each outputs 1 bit)
	- n_bits_per_neuron: how many input bits each neuron addresses (k)
	- connections: tensor shape (num_neurons, n_bits_per_neuron), each entry in [0, total_input_bits)
			If None, random fixed connections will be created at initialization.
	- use_hashing/hash_size: passed to each RAMNeuron
	"""

	def __init__(self, total_input_bits: int, num_neurons: int, n_bits_per_neuron: int, connections: Optional[LongTensor] = None, use_hashing: bool = False, hash_size: int = 1024, rng: Optional[int] = None):
		super().__init__()
		assert total_input_bits >= 1
		assert num_neurons >= 1
		assert n_bits_per_neuron >= 1
		self.total_input_bits = int(total_input_bits)
		self.num_neurons = int(num_neurons)
		self.n_bits_per_neuron = int(n_bits_per_neuron)

		# create independent neurons
		self.neurons = ModuleList([RAMNeuron(n_bits_per_neuron, use_hashing=use_hashing, hash_size=hash_size) for _ in range(self.num_neurons)])

		# connections: if not provided, create random fixed ones
		if connections is None:
			# create shape (num_neurons, n_bits_per_neuron)
			# allow repeated indices (common in WNN designs)
			# store as buffer (non-parameter)
			self.register_buffer("connections", self._randomize_connections(rng))
		else:
			# validate provided connections
			assert isinstance(connections, LongTensor) or isinstance(connections, Tensor)
			conn = connections.clone().long()
			assert conn.shape == (self.num_neurons, self.n_bits_per_neuron)
			assert conn.min().item() >= 0 and conn.max().item() < self.total_input_bits
			self.register_buffer("connections", conn)

	# --- connection management helpers ---
	def _randomize_connections(self, rng: Optional[int] = None) -> Tensor:
		"""Replace connections with new random mapping (in-place)."""
		if rng is not None:
			manual_seed(rng)
		return randint(0, self.total_input_bits, (self.num_neurons, self.n_bits_per_neuron), dtype=long)

	# --- connection management helpers ---
	def randomize_connections(self, rng: Optional[int] = None):
		"""Replace connections with new random mapping (in-place)."""
		self.connections = self._randomize_connections(rng)
		# re-register buffer (PyTorch requires register_buffer only once; assigning is fine)

	def set_connections(self, connections: LongTensor):
		"""Set a custom connections tensor (validate shape & range)."""
		assert connections.shape == (self.num_neurons, self.n_bits_per_neuron)
		assert connections.min().item() >= 0 and connections.max().item() < self.total_input_bits
		self.connections = connections.clone().long()

	# --- forward / write ---
	def forward(self, input_bits: Tensor) -> Tensor:
		"""
		input_bits: Tensor[batch, total_input_bits] values 0/1 or torch.bool
		returns: Tensor[batch, num_neurons] each 0/1 (uint8)
		"""
		# gather inputs for each neuron: shape -> (batch, num_neurons, n_bits_per_neuron)
		# Use advanced indexing: input_bits[:, connections] works because connections is (num_neurons, k)
		# batch_sizeut PyTorch will broadcast; to be explicit:
		if input_bits.dtype != tbool:
			input_bits = input_bits.to(tbool)
		# Gather inputs for each neuron: [batch, num_neurons, n_bits_per_neuron]
		gathered = input_bits[:, self.connections]

		# Prepare bit weights: [1, num_neurons, n_bits_per_neuron]
		bit_weights = stack([neuron._bit_weights for neuron in self.neurons], dim=0)  # [num_neurons, n_bits]
		bit_weights = bit_weights.unsqueeze(0)  # [1, num_neurons, n_bits]

		# Compute addresses for all neurons in batch: [batch, num_neurons]
		addresses = (gathered.to(int64) * bit_weights).sum(dim=-1)
		if self.neurons[0].use_hashing:
				addresses = (addresses % self.neurons[0].memory_size).long()
		else:
				addresses = addresses.long()

		# Get memory values for each neuron in batch: [batch, num_neurons]
		mems = stack([neuron._memory[addr] for neuron, addr in zip(self.neurons, addresses.T)], dim=1)

		# Return bool tensor
		return (mems & 1).to(torch.bool)

	def train_write(self, input_bits: torch.Tensor, target_bits: torch.Tensor):
		"""
		Write target bits to each neuron's memory according to connections.
		- input_bits: [batch, total_input_bits]
		- target_bits: [batch, num_neurons] each 0/1
		"""
		assert input_bits.shape[0] == target_bits.shape[0]
		assert target_bits.shape[1] == self.num_neurons
		gathered = input_bits[:, self.connections]  # [batch, num_neurons, n_bits]
		for i, neuron in enumerate(self.neurons):
			neuron.train_write(gathered[:, i, :], target_bits[:, i])