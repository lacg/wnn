# --------------------------------------------------------------------
# Weightless RAM neural network primitives
#
# Structure (self-contained modules):
#   wnn.ram.core/              - Core components + enums (Memory, RAMLayer, MapperStrategy, etc.)
#   wnn.ram.core.models/       - Transformer components + enums (AttentionType, FFNType, etc.)
#   wnn.ram.architecture/      - Architecture specs (KVSpec, etc.)
#   wnn.ram.encoders_decoders/ - Encoders/decoders + enums (OutputMode, PositionMode)
#   wnn.ram.cost/              - Cost calculators + enums (CostCalculatorType)
#   wnn.ram.strategies/        - Training/Forward strategies + enums
#   wnn.ram.factories/         - Factory classes (use module-local enums)
#
# Import pattern:
#   from wnn.ram.core import MapperStrategy, RAMLayer
#   from wnn.ram.core.models import AttentionType, FFNType
#   from wnn.ram.encoders_decoders import OutputMode, PositionMode
#   from wnn.ram.cost import CostCalculatorType
# --------------------------------------------------------------------
