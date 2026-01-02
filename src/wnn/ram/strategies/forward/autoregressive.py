"""
Autoregressive Forward Strategy

Token-by-token generation where each output is fed back as the next input.
Standard for language models and sequence-to-sequence tasks.
"""

from typing import TYPE_CHECKING

from torch import Tensor

from wnn.ram.strategies.base import ForwardStrategyBase
from wnn.ram.strategies.config import AutoregressiveConfig, ForwardConfig

if TYPE_CHECKING:
    from wnn.ram.core.base import RAMComponent


class AutoregressiveForwardStrategy(ForwardStrategyBase):
    """
    Autoregressive generation: token-by-token with feedback.

    At each position:
    1. Run forward pass with current inputs
    2. Take the output at current position
    3. Feed it as input to the next position

    This enables models to generate sequences based on their own outputs.

    Usage:
        strategy = AutoregressiveForwardStrategy(AutoregressiveConfig(
            use_cache=True,
            max_length=100,
        ))
        outputs = strategy.forward(model, initial_tokens)
    """

    def __init__(self, config: AutoregressiveConfig | ForwardConfig | None = None):
        if config is None:
            config = AutoregressiveConfig()
        elif isinstance(config, ForwardConfig) and not isinstance(config, AutoregressiveConfig):
            config = AutoregressiveConfig(
                max_length=config.max_length,
                return_intermediates=config.return_intermediates,
            )
        super().__init__(config)

    @property
    def autoregressive_config(self) -> AutoregressiveConfig:
        """Get config with autoregressive-specific fields."""
        return self._config  # type: ignore

    def forward(
        self,
        model: "RAMComponent",
        tokens: list[Tensor],
    ) -> list[Tensor]:
        """
        Run autoregressive forward pass.

        Each output position is computed one at a time, with previous
        outputs fed back as inputs for subsequent positions.
        """
        max_len = self.autoregressive_config.max_length or len(tokens)
        outputs: list[Tensor] = []
        current_input = list(tokens)

        for pos in range(max_len):
            # Run forward pass
            if hasattr(model, 'forward_step'):
                # Model supports incremental forward
                output = model.forward_step(current_input, pos)
            else:
                # Full forward, take current position
                all_outputs = model.forward(current_input)
                if pos < len(all_outputs):
                    output = all_outputs[pos]
                else:
                    # Generate beyond input length
                    output = all_outputs[-1] if all_outputs else tokens[0].clone().zero_()

            outputs.append(output)

            # Check stop token
            if self.autoregressive_config.stop_token is not None:
                if (output == self.autoregressive_config.stop_token).all():
                    break

            # Feed output back as input for next position
            if pos + 1 < max_len:
                if pos + 1 < len(current_input):
                    current_input[pos + 1] = output.clone()
                else:
                    current_input.append(output.clone())

        return outputs


class ParallelForwardStrategy(ForwardStrategyBase):
    """
    Parallel generation: all tokens at once.

    Runs a single forward pass and returns all outputs.
    Suitable for non-causal tasks like copy, shift, sort.

    Usage:
        strategy = ParallelForwardStrategy()
        outputs = strategy.forward(model, tokens)
    """

    def forward(
        self,
        model: "RAMComponent",
        tokens: list[Tensor],
    ) -> list[Tensor]:
        """
        Run parallel forward pass.

        Simply calls model.forward once and returns all outputs.
        """
        outputs = model.forward(tokens)

        # Limit to max_length if specified
        if self.config.max_length is not None:
            outputs = outputs[:self.config.max_length]

        return outputs
