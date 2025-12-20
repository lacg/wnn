# --------------------------------------------------------------------
# Author: Luiz Alberto Crispiniano Garcia
# Weightless RAM neural network primitives:
# - RAMNeuron: bool output, uint8-backed memory table (classic RAM neuron) - Originally, it was necessary. It evolved to be not used.
# The reason is that RAMLayer now have a RAMNeuron as a tensor. This way, it is GPU friendly and much faster.
# - RAMLayer: many RAMNeurons (originally) => Neuron Tensors + connectivity matrix (random or user-specified)
# - RAMAutomaton: recurrent composition (input-layer + state-layer)
#
# Notes:
# - Each RAMNeuron with k input bits has memory size 2**k (can blow up fast).
# - For large k, use hashing (use_hashing=True) with controlled hash_size to reduce memory.
# - Memory dtype is uint8 for alignment; stored values are 0/1 logically.
# - EDRA = “Error Detection and Reconstruction Algorithm” a way to backpropagate the conflicts on higher layers to lower layers
# 	(output layer to input layer for example).
#
# Requirements: torch
# --------------------------------------------------------------------

from .DecoderEnums import OutputMode
from .TransformerBitWiseDecoder import TransformerBitWiseDecoder
from .TransformerDecoder import TransformerDecoder
from .TransformerDecoderFactory import TransformerDecoderFactory
from .TransformerHammingDecoder import TransformerHammingDecoder
from .TransformerRawDecoder import TransformerRawDecoder
