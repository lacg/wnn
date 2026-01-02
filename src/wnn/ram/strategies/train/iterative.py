"""
Iterative Training Strategy

Runs multiple backward passes per example until layers stabilize.
More accurate than greedy but slower.
"""

from torch import Tensor

from wnn.ram.strategies.base import TrainStrategyBase, StepStats
from wnn.ram.strategies.config import IterativeTrainConfig, TrainConfig


class IterativeTrainStrategy(TrainStrategyBase):
    """
    Iterative training: multiple passes until convergence.

    For each training example:
    1. Forward pass to compute outputs
    2. Repeat until stable:
       a. Backpropagate targets through all layers
       b. Train each layer on its computed target
       c. Check if updates changed

    This is slower but better captures layer dependencies.

    Usage:
        strategy = IterativeTrainStrategy(IterativeTrainConfig(
            max_iterations=5,
            convergence_threshold=0.01,
        ))
        history = strategy.train(model, dataset)
    """

    def __init__(self, config: IterativeTrainConfig | TrainConfig | None = None):
        if config is None:
            config = IterativeTrainConfig()
        elif isinstance(config, TrainConfig) and not isinstance(config, IterativeTrainConfig):
            config = IterativeTrainConfig(
                epochs=config.epochs,
                early_stop=config.early_stop,
                shuffle=config.shuffle,
                verbose=config.verbose,
            )
        super().__init__(config)

    @property
    def iterative_config(self) -> IterativeTrainConfig:
        """Get config with iterative-specific fields."""
        return self._config  # type: ignore

    def train_step(
        self,
        model,
        inputs: list[Tensor],
        targets: list[Tensor],
    ) -> StepStats:
        """
        Train on a single example with iterative refinement.

        Runs multiple passes until the model stabilizes or max_iterations reached.
        """
        inputs = [t.squeeze() if t.ndim > 1 else t for t in inputs]
        targets = [t.squeeze() if t.ndim > 1 else t for t in targets]

        total_updates = {}
        last_errors = float('inf')

        for iteration in range(self.iterative_config.max_iterations):
            # Forward pass
            outputs = model.forward(inputs)

            # Count errors
            output_errors = sum(
                1 for out, tgt in zip(outputs, targets)
                if not (out.squeeze() == tgt.squeeze()).all()
            )

            # Check convergence
            if output_errors == 0:
                break

            # Check if improvement is below threshold
            improvement = (last_errors - output_errors) / max(1, last_errors)
            if improvement < self.iterative_config.convergence_threshold and iteration > 0:
                break

            last_errors = output_errors

            # Train using model's method if available
            if hasattr(model, 'train_step'):
                result = model.train_step(inputs, targets)
                if hasattr(result, 'layers_updated'):
                    for layer, count in result.layers_updated.items():
                        total_updates[layer] = total_updates.get(layer, 0) + count
                elif isinstance(result, dict):
                    layer_updates = result.get('layers_updated', result.get('layer_updates', {}))
                    if isinstance(layer_updates, dict):
                        for layer, count in layer_updates.items():
                            total_updates[layer] = total_updates.get(layer, 0) + count

        # Final error count
        outputs = model.forward(inputs)
        final_errors = sum(
            1 for out, tgt in zip(outputs, targets)
            if not (out.squeeze() == tgt.squeeze()).all()
        )

        bit_errors = sum(
            (out.squeeze() != tgt.squeeze()).sum().item()
            for out, tgt in zip(outputs, targets)
        )

        return StepStats(
            output_errors=final_errors,
            bit_errors=bit_errors,
            layers_updated=total_updates,
        )
