# --------------------------------------------------------------------
# Author: Luiz Alberto Crispiniano Garcia
# Toy Helpers:
# - KV Memory: Key-Value pairs with write and query.
# - Token Rotator: Cycling through token subsets for unbiased training.
#
# Requirements: torch
# --------------------------------------------------------------------

from .kvspec import KVSpec
from .token_rotator import TokenRotator, DatasetRotator, RotatorConfig
from .base_evaluator import BaseEvaluator, EvalResult, AdaptationConfig, OffspringSearchResult
from .tiered_evaluator import TieredEvaluator, TieredEvaluatorConfig
from .token_clustering import TokenClustering
from .multistage_evaluator import MultiStageEvaluator, compute_default_k

# Backward compatibility aliases
CachedEvaluator = TieredEvaluator
CachedEvaluatorConfig = TieredEvaluatorConfig
