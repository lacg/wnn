"""
Step Configuration Factory

Factory for creating block configurations for multi-step transformers.
Uses match-case for clean dispatch.
"""

from wnn.ram.enums import Step, AttentionType, FFNType
from wnn.ram.encoders_decoders import PositionMode


class StepConfigurationFactory:
    """
    Factory for creating block configurations from Step enums.

    Each step maps to a specific attention + FFN configuration
    optimized for that operation.
    """

    @staticmethod
    def create(step: Step) -> dict:
        """
        Create a block configuration for a given step.

        Args:
            step: The step type

        Returns:
            Configuration dictionary for RAMTransformerBlock
        """
        match step:
            case Step.COPY:
                return {
                    'attention_type': AttentionType.POSITION_ONLY,
                    'position_mode': PositionMode.RELATIVE,
                    'causal': False,
                    'ffn_type': FFNType.NONE,
                    'use_residual': False,
                }

            case Step.SHIFT:
                return {
                    'attention_type': AttentionType.POSITION_ONLY,
                    'position_mode': PositionMode.RELATIVE,
                    'causal': True,
                    'ffn_type': FFNType.NONE,
                    'use_residual': False,
                }

            case Step.REVERSE:
                return {
                    'attention_type': AttentionType.POSITION_ONLY,
                    'position_mode': PositionMode.BINARY,
                    'causal': False,
                    'ffn_type': FFNType.NONE,
                    'use_residual': False,
                }

            case Step.SORT:
                return {
                    'attention_type': AttentionType.SORTING,
                    'ffn_type': FFNType.NONE,
                    'use_residual': False,
                }

            case Step.INCREMENT:
                return {
                    'attention_type': AttentionType.POSITION_ONLY,
                    'position_mode': PositionMode.RELATIVE,
                    'causal': False,
                    'ffn_type': FFNType.INCREMENT,
                    'use_residual': False,
                }

            case Step.DECREMENT:
                return {
                    'attention_type': AttentionType.POSITION_ONLY,
                    'position_mode': PositionMode.RELATIVE,
                    'causal': False,
                    'ffn_type': FFNType.DECREMENT,
                    'use_residual': False,
                }

            case Step.ROT13:
                return {
                    'attention_type': AttentionType.POSITION_ONLY,
                    'position_mode': PositionMode.RELATIVE,
                    'causal': False,
                    'ffn_type': FFNType.ROT13,
                    'use_residual': False,
                }

            case Step.NEGATE:
                return {
                    'attention_type': AttentionType.POSITION_ONLY,
                    'position_mode': PositionMode.RELATIVE,
                    'causal': False,
                    'ffn_type': FFNType.NEGATE,
                    'use_residual': False,
                }

            case _:
                raise ValueError(f"Unknown step: {step}")

    @staticmethod
    def create_many(steps: list[Step]) -> list[dict]:
        """
        Create configurations for multiple steps.

        Args:
            steps: List of step types

        Returns:
            List of configuration dictionaries
        """
        return [StepConfigurationFactory.create(step) for step in steps]
