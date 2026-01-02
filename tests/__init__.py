# --------------------------------------------------------------------
# Author: Luiz Alberto Crispiniano Garcia
#
# Requirements: torch
# --------------------------------------------------------------------
"""
WNN Test Suite

CORE TESTS (run these for verification):
	python tests/parity_check.py      # Basic parity learning
	python tests/kv_memory.py         # Key-value memory
	python tests/benchmarks.py        # Systematic benchmarks

FEATURE TESTS:
	connectivity_optimization.py      # Garcia (2003) thesis optimization
	arithmetic.py                     # LearnedFullAdder, MultiDigitAdder, etc.
	generalization.py                 # Generalization strategies
	seq2seq.py                        # Encoder-decoder tests
	strategy_pattern.py               # TrainStrategy/ForwardStrategy usage
	contrastive_learning.py           # Triplet-based learning
	training_features.py              # Curriculum, multi-task training

STANDARD BENCHMARKS:
	babi_tasks.py                     # Facebook bAbI QA tasks
	listops_benchmark.py              # Hierarchical reasoning
	scan_benchmark.py                 # Compositional generalization
	code_completion.py                # Deterministic code prediction
	theorem_proving.py                # Logical inference
	scaling_benchmark.py              # Vocabulary/sequence scaling
	real_world_benchmark.py           # WikiText, translation, sentiment

LANGUAGE MODEL EXPERIMENTS (research iterations):
	language_model.py                 # Consolidated LM approaches
	language_model_word.py            # Word-level prediction
	ram_lm_*.py                       # Various LM architecture experiments
"""
