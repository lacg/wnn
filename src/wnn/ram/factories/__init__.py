"""
RAM Factories

All factory classes for the RAM architecture, consolidated from across the codebase.
"""

# Decoder
from wnn.ram.factories.decoder import TransformerDecoderFactory

# Position
from wnn.ram.factories.position import PositionEncoderFactory

# Cost
from wnn.ram.factories.cost import CostCalculatorFactory

# Mapper/Generalization
from wnn.ram.factories.mapper import MapperFactory

# Transformer components
from wnn.ram.factories.ffn import FFNFactory
from wnn.ram.factories.attention import AttentionFactory
from wnn.ram.factories.config import StepConfigurationFactory
from wnn.ram.factories.transformer import RAMTransformerFactory

# Unified model factory
from wnn.ram.factories.models import ModelsFactory


__all__ = [
	# Decoder
	'TransformerDecoderFactory',
	# Position
	'PositionEncoderFactory',
	# Cost
	'CostCalculatorFactory',
	# Mapper/Generalization
	'MapperFactory',
	# Transformer components
	'FFNFactory',
	'AttentionFactory',
	'StepConfigurationFactory',
	'RAMTransformerFactory',
	# Unified model factory
	'ModelsFactory',
]
