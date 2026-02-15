"""
HuggingFace integration for WNN RAM language models.

Provides PreTrainedModel and PretrainedConfig wrappers so WNN genomes
can be saved/loaded via the standard HuggingFace Hub interface.

Usage:
	from wnn.hf import WNNConfig, WNNForCausalLM

	# Load from Hub
	model = WNNForCausalLM.from_pretrained("lacg/wnn-bitwise-best")

	# Save locally
	model.save_pretrained("./my_model")

	# Push to Hub
	model.push_to_hub("username/model-name")
"""

from wnn.hf.configuration_wnn import WNNConfig
from wnn.hf.modeling_wnn import WNNForCausalLM

__all__ = ["WNNConfig", "WNNForCausalLM"]
