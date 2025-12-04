# wnn
PyTorch's Weightless Neuron Networks (WNN) Modules

# Overview
This module contains a WNN used by the llm-optimizer architeture project and the goal is to create a Transformer using RAM neurons.

# Architeture
It is formed with:
* RAMTransformer: contains 3 layers (input, state and output) and it uses a vectorized RAM neuron. The RAM neuron memory is formed by 2 bits, instead of the traditional 1, where 00 is False, 11 is True, 01 is weak False and 10 is weak True and all memories start at 01 (weak False). This is not the traditional RAM neuron though.
* RAMLayer: a thin wrapper to the Memory.
* Memory: contains connections and memory_words. The memory does not store the bits, but bit values on every int64. that can be 1 out of 4 possible (False, weak False, weak True and True), using 2 bits and as it is int64 instead of uint64 (PyTorch does not have a uint64 tensor), we can only use 62 bits storing 31 RAM values per word.
* MemoryValue: just an IntEnum for the FALSE, WEAK_FALSE, WEAK_TRUE and TRUE, where EMPTY (the inital memory value) is an alias to WEAK_FALSE.

# Instructions
To run, just create a virtual environment:
source wnn/bin/activate
pip install -e src/wnn
pip install --upgrade pip setuptools wheel
pip install --upgrade pyyaml torch torchvision torchaudio
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"

Then to run a toy experiment, like parity_check:
tests/parity_check.py

And to finish the virtual environment:
deactivate