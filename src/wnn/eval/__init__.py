"""
Evaluation infrastructure for WNN language models.

Provides:
- EvalTask: Evaluation task specification (dataset + tokenizer)
- Checkpoint: Save/load reproducible model checkpoints
- Predefined tasks: WIKITEXT2_TEST, WIKITEXT2_VAL, WIKITEXT2_TRAIN
"""

from wnn.eval.task import (
	EvalTask,
	WIKITEXT2_TEST,
	WIKITEXT2_VAL,
	WIKITEXT2_TRAIN,
)
from wnn.eval.checkpoint import Checkpoint

__all__ = [
	"EvalTask",
	"Checkpoint",
	"WIKITEXT2_TEST",
	"WIKITEXT2_VAL",
	"WIKITEXT2_TRAIN",
]
