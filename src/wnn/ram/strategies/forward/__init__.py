"""
Forward Strategies

Pluggable inference strategies for RAM networks.

Usage:
    from wnn.ram.strategies.forward import AutoregressiveForwardStrategy

    model = RAMSeq2Seq(...)
    strategy = AutoregressiveForwardStrategy(AutoregressiveConfig(use_cache=True))
    outputs = strategy.forward(model, inputs)
"""

from wnn.ram.strategies.forward.autoregressive import (
    AutoregressiveForwardStrategy,
    ParallelForwardStrategy,
)

__all__ = [
    'AutoregressiveForwardStrategy',
    'ParallelForwardStrategy',
]
