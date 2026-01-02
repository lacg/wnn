"""
Contrastive Training Strategy

Trains on triplets (anchor, positive, negative) to improve discrimination.
Best used as refinement after standard training.
"""

from dataclasses import dataclass
from random import shuffle as shuffle_list, sample

from torch import Tensor

from wnn.ram.strategies.base import TrainStrategyBase, StepStats, EpochStats
from wnn.ram.strategies.config import ContrastiveTrainConfig, TrainConfig


@dataclass
class Triplet:
    """A training triplet for contrastive learning."""
    anchor_input: list[Tensor]
    anchor_target: list[Tensor]
    positive_input: list[Tensor]
    positive_target: list[Tensor]
    negative_input: list[Tensor]
    negative_target: list[Tensor]


class ContrastiveTrainStrategy(TrainStrategyBase):
    """
    Contrastive training: discrimination through triplets.

    Trains the model to produce similar outputs for similar inputs
    and different outputs for different inputs.

    NOTE: Best used as refinement AFTER standard training.
    RAM networks need explicit pattern memorization first.

    Usage:
        # First, train with standard strategy
        greedy = GreedyTrainStrategy()
        greedy.train(model, dataset)

        # Then refine with contrastive
        strategy = ContrastiveTrainStrategy(ContrastiveTrainConfig(
            hard_negative_ratio=0.5,
            margin=0.3,
        ))
        strategy.train(model, dataset)
    """

    def __init__(self, config: ContrastiveTrainConfig | TrainConfig | None = None):
        if config is None:
            config = ContrastiveTrainConfig()
        elif isinstance(config, TrainConfig) and not isinstance(config, ContrastiveTrainConfig):
            config = ContrastiveTrainConfig(
                epochs=config.epochs,
                early_stop=config.early_stop,
                shuffle=config.shuffle,
                verbose=config.verbose,
            )
        super().__init__(config)

    @property
    def contrastive_config(self) -> ContrastiveTrainConfig:
        """Get config with contrastive-specific fields."""
        return self._config  # type: ignore

    def _hamming_distance(self, t1: Tensor, t2: Tensor) -> int:
        """Compute Hamming distance between two tensors."""
        return int((t1.squeeze() != t2.squeeze()).sum().item())

    def _normalized_similarity(self, t1: Tensor, t2: Tensor) -> float:
        """Normalized Hamming similarity (1 = identical, 0 = maximally different)."""
        dist = self._hamming_distance(t1, t2)
        max_dist = t1.numel()
        return 1.0 - (dist / max_dist) if max_dist > 0 else 1.0

    def _sequence_similarity(
        self,
        seq1: list[Tensor],
        seq2: list[Tensor],
    ) -> float:
        """Average similarity across sequence positions."""
        if len(seq1) != len(seq2):
            return 0.0

        if self.contrastive_config.similarity_fn is not None:
            # Use custom similarity if provided
            sims = [
                self.contrastive_config.similarity_fn(t1, t2)
                for t1, t2 in zip(seq1, seq2)
            ]
        else:
            sims = [
                self._normalized_similarity(t1, t2)
                for t1, t2 in zip(seq1, seq2)
            ]

        return sum(sims) / len(sims) if sims else 0.0

    def _generate_triplets(
        self,
        model,
        dataset: list[tuple[list[Tensor], list[Tensor]]],
    ) -> list[Triplet]:
        """
        Generate training triplets from dataset.

        Each example becomes an anchor. Positive = same target class,
        Negative = different target class (with hard negative mining).
        """
        triplets = []

        # Group examples by target signature
        target_groups: dict[str, list[tuple[list[Tensor], list[Tensor]]]] = {}
        for inputs, targets in dataset:
            # Create signature from targets
            sig = tuple(
                tuple(t.squeeze().tolist()) if t.numel() > 1 else (t.item(),)
                for t in targets
            )
            key = str(sig)
            if key not in target_groups:
                target_groups[key] = []
            target_groups[key].append((inputs, targets))

        # Generate triplets
        all_keys = list(target_groups.keys())

        for key, group in target_groups.items():
            other_keys = [k for k in all_keys if k != key]
            if not other_keys:
                continue

            for anchor_inp, anchor_tgt in group:
                # Positive: same class (if more than one example)
                if len(group) > 1:
                    candidates = [(i, t) for i, t in group if i is not anchor_inp]
                    if candidates:
                        pos_inp, pos_tgt = candidates[0]
                    else:
                        # Use self as positive (degenerate case)
                        pos_inp, pos_tgt = anchor_inp, anchor_tgt
                else:
                    pos_inp, pos_tgt = anchor_inp, anchor_tgt

                # Negative: different class (with optional hard mining)
                if self.contrastive_config.hard_negative_ratio > 0:
                    # Hard negative: find most similar example from different class
                    neg_inp, neg_tgt = self._find_hard_negative(
                        model, anchor_inp, other_keys, target_groups
                    )
                else:
                    # Random negative
                    neg_key = sample(other_keys, 1)[0]
                    neg_inp, neg_tgt = target_groups[neg_key][0]

                triplets.append(Triplet(
                    anchor_input=anchor_inp,
                    anchor_target=anchor_tgt,
                    positive_input=pos_inp,
                    positive_target=pos_tgt,
                    negative_input=neg_inp,
                    negative_target=neg_tgt,
                ))

        return triplets

    def _find_hard_negative(
        self,
        model,
        anchor_input: list[Tensor],
        other_keys: list[str],
        target_groups: dict[str, list[tuple[list[Tensor], list[Tensor]]]],
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Find the most confusing negative example."""
        anchor_output = model.forward(anchor_input)

        best_sim = -1.0
        best_neg = None

        for key in other_keys:
            for neg_inp, neg_tgt in target_groups[key]:
                neg_output = model.forward(neg_inp)
                sim = self._sequence_similarity(anchor_output, neg_output)
                if sim > best_sim:
                    best_sim = sim
                    best_neg = (neg_inp, neg_tgt)

        if best_neg is None:
            # Fallback to random
            key = sample(other_keys, 1)[0]
            return target_groups[key][0]

        return best_neg

    def train_step(
        self,
        model,
        inputs: list[Tensor],
        targets: list[Tensor],
    ) -> StepStats:
        """
        Train on a single example (non-triplet training).

        For triplet-specific training, use train_triplet.
        """
        inputs = [t.squeeze() if t.ndim > 1 else t for t in inputs]
        targets = [t.squeeze() if t.ndim > 1 else t for t in targets]

        if hasattr(model, 'train_step') and callable(model.train_step):
            result = model.train_step(inputs, targets)
            if hasattr(result, 'output_errors'):
                return StepStats(
                    output_errors=result.output_errors,
                    bit_errors=getattr(result, 'bit_errors', 0),
                    layers_updated=getattr(result, 'layers_updated', {}),
                )
            elif isinstance(result, dict):
                layer_updates = result.get('layers_updated', result.get('layer_updates', {}))
                if isinstance(layer_updates, list):
                    layer_updates = {f"layer_{i}": u for i, u in enumerate(layer_updates) if u}
                return StepStats(
                    output_errors=result.get('output_errors', 0),
                    bit_errors=result.get('bit_errors', 0),
                    layers_updated=layer_updates if isinstance(layer_updates, dict) else {},
                )
            return StepStats(output_errors=int(result))

        outputs = model.forward(inputs)
        errors = sum(
            1 for out, tgt in zip(outputs, targets)
            if not (out.squeeze() == tgt.squeeze()).all()
        )
        return StepStats(output_errors=errors)

    def train_triplet(
        self,
        model,
        triplet: Triplet,
    ) -> tuple[bool, float]:
        """
        Train on a single triplet.

        Returns (success, margin_achieved).
        Success = positive more similar than negative by margin.
        """
        # Get outputs
        anchor_out = model.forward(triplet.anchor_input)
        pos_out = model.forward(triplet.positive_input)
        neg_out = model.forward(triplet.negative_input)

        # Compute similarities
        pos_sim = self._sequence_similarity(anchor_out, pos_out)
        neg_sim = self._sequence_similarity(anchor_out, neg_out)

        margin_achieved = pos_sim - neg_sim
        success = margin_achieved >= self.contrastive_config.margin

        if not success:
            # Train on anchor and positive to reinforce correct mapping
            self.train_step(model, triplet.anchor_input, triplet.anchor_target)
            self.train_step(model, triplet.positive_input, triplet.positive_target)

        return success, margin_achieved

    def train_epoch(
        self,
        model,
        dataset: list[tuple[list[Tensor], list[Tensor]]],
        epoch_num: int,
    ) -> EpochStats:
        """
        Train for one epoch with contrastive learning.

        Combines standard training with triplet discrimination.
        """
        # First, do standard training pass
        if self._config.shuffle:
            dataset_copy = list(dataset)
            shuffle_list(dataset_copy)
        else:
            dataset_copy = dataset

        # Standard training
        total_errors = 0
        bit_errors = 0
        total_positions = 0
        layer_updates: dict[str, int] = {}
        examples_correct = 0

        for inputs, targets in dataset_copy:
            stats = self.train_step(model, inputs, targets)
            total_errors += stats.output_errors
            bit_errors += stats.bit_errors
            total_positions += len(targets)
            for layer, count in stats.layers_updated.items():
                layer_updates[layer] = layer_updates.get(layer, 0) + count
            if stats.output_errors == 0:
                examples_correct += 1

        # Then, do triplet training
        triplets = self._generate_triplets(model, dataset)
        if self.contrastive_config.triplets_per_epoch is not None:
            triplets = triplets[:self.contrastive_config.triplets_per_epoch]

        triplet_successes = 0
        for triplet in triplets:
            success, _ = self.train_triplet(model, triplet)
            if success:
                triplet_successes += 1

        # Compute final accuracy (re-evaluate after triplet training)
        final_errors = 0
        for inputs, targets in dataset:
            outputs = model.forward(inputs)
            for out, tgt in zip(outputs, targets):
                if not (out.squeeze() == tgt.squeeze()).all():
                    final_errors += 1

        error_rate = min(1.0, final_errors / total_positions) if total_positions > 0 else 0
        accuracy = 100 * (1 - error_rate)

        return EpochStats(
            epoch=epoch_num,
            total_errors=final_errors,
            bit_errors=bit_errors,
            total_positions=total_positions,
            accuracy=accuracy,
            layer_updates=layer_updates,
            examples_correct=examples_correct,
            examples_total=len(dataset),
        )
