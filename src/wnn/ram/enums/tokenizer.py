"""
Tokenizer type enums for language model benchmarks.

DEPRECATED: Import from wnn.tokenizers instead:
    from wnn.tokenizers import TokenizerType

This module re-exports TokenizerType for backward compatibility.
"""

import warnings
from wnn.tokenizers import TokenizerType

# Re-export with deprecation notice
__all__ = ['TokenizerType']

# Backward compatibility aliases
# The new TokenizerType uses slightly different names:
# - SIMPLE -> SIMPLE_WORD
# - WIKITEXT_WORD -> WORD
# - GPT2_BPE -> GPT2 (and there's a new trainable BPE)

# Show deprecation warning on import
warnings.warn(
	"TokenizerType from wnn.ram.enums is deprecated. "
	"Import from wnn.tokenizers instead: from wnn.tokenizers import TokenizerType",
	DeprecationWarning,
	stacklevel=2
)
