"""
RAM Transformer Block

A complete transformer block using RAM-based attention and feed-forward layers.
Designed for maximum generalization using computed operations where possible.

Architecture:
    Input → Attention → Residual(XOR) → FFN → Residual(XOR) → Output

Key Features:
    - Multiple attention modes (position-only, content-match, sorting, etc.)
    - BIT_LEVEL generalization in feed-forward layers
    - XOR-based residual connections (discrete analog of addition)
    - Stackable blocks for deep transformers

Generalization Strategy:
    - Attention: Use computed modes (position_only, XOR_EQUAL, sorting) for 100%
    - FFN: Use BIT_LEVEL for partial generalization
    - Residual: XOR is computed, generalizes 100%
"""

from torch import Tensor, zeros, uint8, cat
from torch.nn import Module, ModuleList
from enum import IntEnum

from wnn.ram.SoftRAMAttention import (
    SoftRAMAttention, SortingAttention, MinMaxAttention,
    AggregationStrategy, ContentMatchMode, AttentionCombineMode,
    ComputedArithmeticFFN, ArithmeticOp,
    _bits_to_int
)
from wnn.ram.RAMGeneralization import GeneralizingProjection, MapperStrategy
from wnn.ram.RAMLayer import RAMLayer
from wnn.ram.encoders_decoders import PositionMode


class AttentionType(IntEnum):
    """Type of attention mechanism to use."""
    SOFT_RAM = 0        # Standard SoftRAMAttention (configurable)
    SORTING = 1         # SortingAttention (computed, 100% generalization)
    MIN_MAX = 2         # MinMaxAttention (computed, 100% generalization)
    POSITION_ONLY = 3   # Position-only attention (100% generalization)
    CONTENT_MATCH = 4   # Content-matching attention (XOR_EQUAL, etc.)


class FFNType(IntEnum):
    """Type of feed-forward network."""
    # Learned FFN types (may not generalize to unseen tokens)
    NONE = 0            # No FFN (attention only)
    SINGLE = 1          # Single projection layer
    TWO_LAYER = 2       # Two-layer MLP (expand then contract)
    BIT_LEVEL = 3       # BIT_LEVEL generalization (partial)

    # Computed FFN types (100% generalization - no training needed)
    INCREMENT = 10      # Add 1 to value
    DECREMENT = 11      # Subtract 1 from value
    ADD_MOD = 12        # Add constant with modulo
    SUBTRACT_MOD = 13   # Subtract constant with modulo
    ROT13 = 14          # ROT13 cipher (add 13 mod 26)
    NEGATE = 15         # Bitwise complement (max - value)


class RAMTransformerBlock(Module):
    """
    A single RAM transformer block.

    Architecture:
        x → Attention(x) → x ⊕ attn_out → FFN → x ⊕ ffn_out → output

    Where ⊕ is XOR (discrete residual connection).
    """

    def __init__(
        self,
        input_bits: int,
        # Attention config
        attention_type: AttentionType = AttentionType.POSITION_ONLY,
        num_heads: int = 8,
        content_match: ContentMatchMode = ContentMatchMode.NONE,
        attention_combine: AttentionCombineMode = AttentionCombineMode.CONTENT_ONLY,
        position_mode: PositionMode = PositionMode.RELATIVE,
        causal: bool = True,
        # FFN config
        ffn_type: FFNType = FFNType.BIT_LEVEL,
        ffn_hidden_bits: int | None = None,  # For TWO_LAYER, defaults to 2x input
        ffn_constant: int = 1,    # For ADD_MOD/SUBTRACT_MOD computed FFN
        ffn_modulo: int | None = None,  # For ADD_MOD/SUBTRACT_MOD (None = no modulo)
        # Other
        use_residual: bool = True,
        max_seq_len: int = 16,
        rng: int | None = None,
    ):
        """
        Args:
            input_bits: Bits per token
            attention_type: Type of attention mechanism
            num_heads: Number of attention heads (for SOFT_RAM)
            content_match: Content matching mode (for CONTENT_MATCH type)
            attention_combine: How to combine content and position
            position_mode: Position encoding mode
            causal: Use causal attention mask
            ffn_type: Type of feed-forward network
            ffn_hidden_bits: Hidden dimension for TWO_LAYER FFN
            ffn_constant: Constant for computed arithmetic FFN (ADD_MOD, SUBTRACT_MOD)
            ffn_modulo: Modulo for computed arithmetic FFN (None = no modulo)
            use_residual: Use XOR residual connections
            max_seq_len: Maximum sequence length
            rng: Random seed
        """
        super().__init__()

        self.input_bits = input_bits
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.use_residual = use_residual
        self.max_seq_len = max_seq_len

        # Build attention layer
        self.attention = self._build_attention(
            attention_type=attention_type,
            input_bits=input_bits,
            num_heads=num_heads,
            content_match=content_match,
            attention_combine=attention_combine,
            position_mode=position_mode,
            causal=causal,
            max_seq_len=max_seq_len,
            rng=rng,
        )

        # Build FFN layer
        self.ffn = self._build_ffn(
            ffn_type=ffn_type,
            input_bits=input_bits,
            hidden_bits=ffn_hidden_bits,
            constant=ffn_constant,
            modulo=ffn_modulo,
            rng=rng + 1000 if rng else None,
        )

        # Summary
        attn_name = attention_type.name
        ffn_name = ffn_type.name
        residual_str = "+residual" if use_residual else ""
        print(f"[RAMTransformerBlock] {input_bits}b, attn={attn_name}, "
              f"ffn={ffn_name}{residual_str}")

    def _build_attention(
        self,
        attention_type: AttentionType,
        input_bits: int,
        num_heads: int,
        content_match: ContentMatchMode,
        attention_combine: AttentionCombineMode,
        position_mode: PositionMode,
        causal: bool,
        max_seq_len: int,
        rng: int | None,
    ) -> Module:
        """Build the attention layer based on type."""

        if attention_type == AttentionType.SORTING:
            return SortingAttention(
                input_bits=input_bits,
                descending=False,
                rng=rng,
            )

        elif attention_type == AttentionType.MIN_MAX:
            return MinMaxAttention(
                input_bits=input_bits,
                find_max=False,
                rng=rng,
            )

        elif attention_type == AttentionType.POSITION_ONLY:
            return SoftRAMAttention(
                input_bits=input_bits,
                num_heads=num_heads,
                aggregation=AggregationStrategy.TOP_1,
                value_strategy=MapperStrategy.BIT_LEVEL,
                position_mode=position_mode,
                max_seq_len=max_seq_len,
                causal=causal,
                position_only=True,
                rng=rng,
            )

        elif attention_type == AttentionType.CONTENT_MATCH:
            return SoftRAMAttention(
                input_bits=input_bits,
                num_heads=num_heads,
                aggregation=AggregationStrategy.TOP_1,
                value_strategy=MapperStrategy.BIT_LEVEL,
                position_mode=position_mode,
                max_seq_len=max_seq_len,
                causal=causal,
                position_only=False,
                content_match=content_match,
                attention_combine=attention_combine,
                rng=rng,
            )

        else:  # SOFT_RAM (default, fully configurable)
            return SoftRAMAttention(
                input_bits=input_bits,
                num_heads=num_heads,
                aggregation=AggregationStrategy.TOP_1,
                value_strategy=MapperStrategy.BIT_LEVEL,
                position_mode=position_mode,
                max_seq_len=max_seq_len,
                causal=causal,
                position_only=False,
                content_match=content_match,
                attention_combine=attention_combine,
                rng=rng,
            )

    def _build_ffn(
        self,
        ffn_type: FFNType,
        input_bits: int,
        hidden_bits: int | None,
        constant: int,
        modulo: int | None,
        rng: int | None,
    ) -> Module | None:
        """Build the feed-forward network based on type."""

        if ffn_type == FFNType.NONE:
            return None

        # Learned FFN types
        elif ffn_type == FFNType.SINGLE:
            return RAMLayer(
                total_input_bits=input_bits,
                num_neurons=input_bits,
                n_bits_per_neuron=min(input_bits, 8),
                rng=rng,
            )

        elif ffn_type == FFNType.BIT_LEVEL:
            return GeneralizingProjection(
                input_bits=input_bits,
                output_bits=input_bits,
                strategy=MapperStrategy.BIT_LEVEL,
                rng=rng,
            )

        elif ffn_type == FFNType.TWO_LAYER:
            if hidden_bits is None:
                hidden_bits = input_bits * 2
            return TwoLayerFFN(
                input_bits=input_bits,
                hidden_bits=hidden_bits,
                output_bits=input_bits,
                rng=rng,
            )

        # Computed FFN types (100% generalization)
        elif ffn_type == FFNType.INCREMENT:
            return ComputedArithmeticFFN(
                input_bits=input_bits,
                operation=ArithmeticOp.INCREMENT,
                rng=rng,
            )

        elif ffn_type == FFNType.DECREMENT:
            return ComputedArithmeticFFN(
                input_bits=input_bits,
                operation=ArithmeticOp.DECREMENT,
                rng=rng,
            )

        elif ffn_type == FFNType.ADD_MOD:
            return ComputedArithmeticFFN(
                input_bits=input_bits,
                operation=ArithmeticOp.ADD_MOD,
                constant=constant,
                modulo=modulo if modulo else 26,  # Default to alphabet size
                rng=rng,
            )

        elif ffn_type == FFNType.SUBTRACT_MOD:
            return ComputedArithmeticFFN(
                input_bits=input_bits,
                operation=ArithmeticOp.SUBTRACT_MOD,
                constant=constant,
                modulo=modulo if modulo else 26,
                rng=rng,
            )

        elif ffn_type == FFNType.ROT13:
            return ComputedArithmeticFFN(
                input_bits=input_bits,
                operation=ArithmeticOp.ROT13,
                rng=rng,
            )

        elif ffn_type == FFNType.NEGATE:
            return ComputedArithmeticFFN(
                input_bits=input_bits,
                operation=ArithmeticOp.NEGATE,
                rng=rng,
            )

        return None

    def forward(self, tokens: list[Tensor]) -> list[Tensor]:
        """
        Forward pass through the transformer block.

        Args:
            tokens: List of token tensors

        Returns:
            outputs: Transformed tokens
        """
        tokens = [t.squeeze() if t.ndim > 1 else t for t in tokens]

        # Attention
        attn_out = self.attention.forward(tokens)
        attn_out = [t.squeeze() if t.ndim > 1 else t for t in attn_out]

        # Residual connection (XOR)
        if self.use_residual:
            attn_out = [t ^ r for t, r in zip(tokens, attn_out)]

        # FFN
        if self.ffn is not None:
            ffn_out = []
            for t in attn_out:
                if isinstance(self.ffn, GeneralizingProjection):
                    out = self.ffn(t)
                else:
                    out = self.ffn(t.unsqueeze(0)).squeeze()
                ffn_out.append(out)

            # Residual connection (XOR)
            if self.use_residual:
                ffn_out = [t ^ r for t, r in zip(attn_out, ffn_out)]

            return ffn_out

        return attn_out

    def train_block(
        self,
        input_tokens: list[Tensor],
        target_tokens: list[Tensor] | None = None,
        attention_pattern: str = "copy",  # "copy", "shift", "reverse", "sort"
    ) -> int:
        """
        Train the transformer block.

        Args:
            input_tokens: Input sequence
            target_tokens: Target output (if None, inferred from pattern)
            attention_pattern: What attention pattern to train

        Returns:
            corrections: Number of corrections made
        """
        input_tokens = [t.squeeze() if t.ndim > 1 else t for t in input_tokens]
        n = len(input_tokens)
        corrections = 0

        # Infer targets if not provided
        if target_tokens is None:
            if attention_pattern == "copy":
                target_tokens = input_tokens
            elif attention_pattern == "shift":
                target_tokens = [input_tokens[0]] + input_tokens[:-1]
            elif attention_pattern == "reverse":
                target_tokens = input_tokens[::-1]
            elif attention_pattern == "sort":
                # Sort by bit value
                sorted_indices = sorted(range(n), key=lambda i: _bits_to_int(input_tokens[i]))
                target_tokens = [input_tokens[i] for i in sorted_indices]
            else:
                target_tokens = input_tokens

        target_tokens = [t.squeeze() if t.ndim > 1 else t for t in target_tokens]

        # Train attention (if trainable)
        if hasattr(self.attention, 'train_value_projection'):
            corrections += self.attention.train_value_projection(input_tokens)

            # Train attention weights based on pattern
            if attention_pattern == "copy":
                for pos in range(n):
                    weights = [0.0] * n
                    weights[pos] = 1.0
                    corrections += self.attention.train_attention_weights(input_tokens, pos, weights)

            elif attention_pattern == "shift":
                for pos in range(n):
                    weights = [0.0] * n
                    if pos > 0:
                        weights[pos - 1] = 1.0
                    else:
                        weights[0] = 1.0
                    corrections += self.attention.train_attention_weights(input_tokens, pos, weights)

            elif attention_pattern == "reverse":
                for pos in range(n):
                    weights = [0.0] * n
                    weights[n - 1 - pos] = 1.0
                    corrections += self.attention.train_attention_weights(input_tokens, pos, weights)

        # Train FFN (if trainable)
        if self.ffn is not None and hasattr(self.ffn, 'train_mapping'):
            for inp, tgt in zip(input_tokens, target_tokens):
                corrections += self.ffn.train_mapping(inp, tgt)

        return corrections


class TwoLayerFFN(Module):
    """Two-layer feed-forward network with hidden expansion."""

    def __init__(
        self,
        input_bits: int,
        hidden_bits: int,
        output_bits: int,
        rng: int | None = None,
    ):
        super().__init__()

        self.input_bits = input_bits
        self.hidden_bits = hidden_bits
        self.output_bits = output_bits

        # Up projection: input → hidden
        self.up_proj = GeneralizingProjection(
            input_bits=input_bits,
            output_bits=hidden_bits,
            strategy=MapperStrategy.BIT_LEVEL,
            rng=rng,
        )

        # Down projection: hidden → output
        self.down_proj = GeneralizingProjection(
            input_bits=hidden_bits,
            output_bits=output_bits,
            strategy=MapperStrategy.BIT_LEVEL,
            rng=rng + 500 if rng else None,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: up → activation (none for binary) → down."""
        x = x.squeeze()
        hidden = self.up_proj(x)
        output = self.down_proj(hidden)
        return output

    def train_mapping(self, inp: Tensor, target: Tensor) -> int:
        """Train both projections."""
        # For now, just train down_proj to output target
        # Hidden representation is learned implicitly
        inp = inp.squeeze()
        target = target.squeeze()

        # Forward to get hidden
        hidden = self.up_proj(inp)

        # Train down_proj
        return self.down_proj.train_mapping(hidden, target)


class RAMTransformer(Module):
    """
    Full RAM Transformer with multiple stacked blocks.

    Architecture:
        Input → Block1 → Block2 → ... → BlockN → Output

    Supports different block configurations for different layers.
    """

    def __init__(
        self,
        input_bits: int,
        num_blocks: int = 2,
        # Default block config (can be overridden per block)
        attention_type: AttentionType = AttentionType.POSITION_ONLY,
        num_heads: int = 8,
        content_match: ContentMatchMode = ContentMatchMode.NONE,
        position_mode: PositionMode = PositionMode.RELATIVE,
        causal: bool = True,
        ffn_type: FFNType = FFNType.BIT_LEVEL,
        ffn_constant: int = 1,
        ffn_modulo: int | None = None,
        use_residual: bool = True,
        max_seq_len: int = 16,
        # Per-block overrides
        block_configs: list[dict] | None = None,
        rng: int | None = None,
    ):
        """
        Args:
            input_bits: Bits per token
            num_blocks: Number of transformer blocks
            attention_type: Default attention type for all blocks
            num_heads: Default number of heads
            content_match: Default content matching mode
            position_mode: Default position encoding
            causal: Default causal mask setting
            ffn_type: Default FFN type
            ffn_constant: Default constant for computed arithmetic FFN
            ffn_modulo: Default modulo for computed arithmetic FFN
            use_residual: Default residual connection setting
            max_seq_len: Maximum sequence length
            block_configs: List of per-block config overrides
            rng: Random seed
        """
        super().__init__()

        self.input_bits = input_bits
        self.num_blocks = num_blocks
        self.max_seq_len = max_seq_len

        # Build blocks
        self.blocks = ModuleList()

        for i in range(num_blocks):
            # Get per-block config or use defaults
            if block_configs and i < len(block_configs):
                cfg = block_configs[i]
            else:
                cfg = {}

            block = RAMTransformerBlock(
                input_bits=input_bits,
                attention_type=cfg.get('attention_type', attention_type),
                num_heads=cfg.get('num_heads', num_heads),
                content_match=cfg.get('content_match', content_match),
                attention_combine=cfg.get('attention_combine', AttentionCombineMode.CONTENT_ONLY),
                position_mode=cfg.get('position_mode', position_mode),
                causal=cfg.get('causal', causal),
                ffn_type=cfg.get('ffn_type', ffn_type),
                ffn_constant=cfg.get('ffn_constant', ffn_constant),
                ffn_modulo=cfg.get('ffn_modulo', ffn_modulo),
                use_residual=cfg.get('use_residual', use_residual),
                max_seq_len=max_seq_len,
                rng=rng + i * 10000 if rng else None,
            )
            self.blocks.append(block)

        print(f"[RAMTransformer] {num_blocks} blocks, {input_bits}b tokens")

    def forward(self, tokens: list[Tensor]) -> list[Tensor]:
        """Forward pass through all blocks."""
        x = tokens
        for block in self.blocks:
            x = block.forward(x)
        return x

    def train_transformer(
        self,
        input_tokens: list[Tensor],
        target_tokens: list[Tensor] | None = None,
        attention_pattern: str = "copy",
    ) -> int:
        """Train all blocks."""
        corrections = 0
        for block in self.blocks:
            corrections += block.train_block(input_tokens, target_tokens, attention_pattern)
        return corrections


# =============================================================
# Pre-configured Transformer Architectures
# =============================================================

def create_copy_transformer(input_bits: int, max_seq_len: int = 16, rng: int = None) -> RAMTransformer:
    """
    Create a transformer optimized for copy task.
    Uses position-only attention for 100% generalization.
    """
    return RAMTransformer(
        input_bits=input_bits,
        num_blocks=1,
        attention_type=AttentionType.POSITION_ONLY,
        position_mode=PositionMode.RELATIVE,
        causal=False,
        ffn_type=FFNType.NONE,  # No FFN needed - attention alone does the copy
        use_residual=False,  # Copy doesn't need residual
        max_seq_len=max_seq_len,
        rng=rng,
    )


def create_shift_transformer(input_bits: int, max_seq_len: int = 16, rng: int = None) -> RAMTransformer:
    """
    Create a transformer optimized for shift task.
    Uses position-only attention for 100% generalization.
    """
    return RAMTransformer(
        input_bits=input_bits,
        num_blocks=1,
        attention_type=AttentionType.POSITION_ONLY,
        position_mode=PositionMode.RELATIVE,
        causal=True,
        ffn_type=FFNType.NONE,  # No FFN needed for simple shift
        use_residual=False,
        max_seq_len=max_seq_len,
        rng=rng,
    )


def create_reverse_transformer(input_bits: int, max_seq_len: int = 16, rng: int = None) -> RAMTransformer:
    """
    Create a transformer optimized for reverse task.
    Uses position-only attention with BINARY positions.
    """
    return RAMTransformer(
        input_bits=input_bits,
        num_blocks=1,
        attention_type=AttentionType.POSITION_ONLY,
        position_mode=PositionMode.BINARY,  # BINARY for absolute positions
        causal=False,
        ffn_type=FFNType.NONE,  # No FFN needed - attention alone does the reverse
        use_residual=False,
        max_seq_len=max_seq_len,
        rng=rng,
    )


def create_sorting_transformer(input_bits: int, max_seq_len: int = 16, rng: int = None) -> RAMTransformer:
    """
    Create a transformer for sorting.
    Uses computed SortingAttention for 100% generalization.
    """
    return RAMTransformer(
        input_bits=input_bits,
        num_blocks=1,
        attention_type=AttentionType.SORTING,
        ffn_type=FFNType.NONE,  # Sorting doesn't need FFN
        use_residual=False,
        max_seq_len=max_seq_len,
        rng=rng,
    )


def create_self_matching_transformer(input_bits: int, max_seq_len: int = 16, rng: int = None) -> RAMTransformer:
    """
    Create a transformer for self-matching (find duplicates).
    Uses XOR_EQUAL content matching for 100% generalization.
    """
    return RAMTransformer(
        input_bits=input_bits,
        num_blocks=1,
        attention_type=AttentionType.CONTENT_MATCH,
        content_match=ContentMatchMode.XOR_EQUAL,
        causal=False,
        ffn_type=FFNType.BIT_LEVEL,
        use_residual=False,
        max_seq_len=max_seq_len,
        rng=rng,
    )


def create_multi_step_transformer(
    input_bits: int,
    steps: list[str],  # e.g., ["shift", "reverse"]
    max_seq_len: int = 16,
    rng: int = None,
) -> RAMTransformer:
    """
    Create a multi-step transformer with different operations per block.

    Args:
        input_bits: Bits per token
        steps: List of operations, one per block
        max_seq_len: Maximum sequence length
        rng: Random seed

    Supported steps: "copy", "shift", "reverse", "sort"
    """
    block_configs = []

    for step in steps:
        if step == "copy":
            cfg = {
                'attention_type': AttentionType.POSITION_ONLY,
                'position_mode': PositionMode.RELATIVE,
                'causal': False,
                'ffn_type': FFNType.NONE,
                'use_residual': False,
            }
        elif step == "shift":
            cfg = {
                'attention_type': AttentionType.POSITION_ONLY,
                'position_mode': PositionMode.RELATIVE,
                'causal': True,
                'ffn_type': FFNType.NONE,
                'use_residual': False,
            }
        elif step == "reverse":
            cfg = {
                'attention_type': AttentionType.POSITION_ONLY,
                'position_mode': PositionMode.BINARY,
                'causal': False,
                'ffn_type': FFNType.NONE,
                'use_residual': False,
            }
        elif step == "sort":
            cfg = {
                'attention_type': AttentionType.SORTING,
                'ffn_type': FFNType.NONE,
                'use_residual': False,
            }
        elif step == "increment":
            cfg = {
                'attention_type': AttentionType.POSITION_ONLY,
                'position_mode': PositionMode.RELATIVE,
                'causal': False,
                'ffn_type': FFNType.INCREMENT,
                'use_residual': False,
            }
        elif step == "decrement":
            cfg = {
                'attention_type': AttentionType.POSITION_ONLY,
                'position_mode': PositionMode.RELATIVE,
                'causal': False,
                'ffn_type': FFNType.DECREMENT,
                'use_residual': False,
            }
        elif step == "rot13":
            cfg = {
                'attention_type': AttentionType.POSITION_ONLY,
                'position_mode': PositionMode.RELATIVE,
                'causal': False,
                'ffn_type': FFNType.ROT13,
                'use_residual': False,
            }
        elif step == "negate":
            cfg = {
                'attention_type': AttentionType.POSITION_ONLY,
                'position_mode': PositionMode.RELATIVE,
                'causal': False,
                'ffn_type': FFNType.NEGATE,
                'use_residual': False,
            }
        else:
            cfg = {}

        block_configs.append(cfg)

    return RAMTransformer(
        input_bits=input_bits,
        num_blocks=len(steps),
        ffn_type=FFNType.BIT_LEVEL,
        use_residual=False,
        max_seq_len=max_seq_len,
        block_configs=block_configs,
        rng=rng,
    )


# =============================================================
# Computed FFN Factory Functions
# =============================================================

def create_increment_transformer(input_bits: int, max_seq_len: int = 16, rng: int = None) -> RAMTransformer:
    """
    Create a transformer that increments each token value by 1.
    A→B, B→C, C→D, ... (100% generalization, no training needed)
    """
    return RAMTransformer(
        input_bits=input_bits,
        num_blocks=1,
        attention_type=AttentionType.POSITION_ONLY,
        position_mode=PositionMode.RELATIVE,
        causal=False,
        ffn_type=FFNType.INCREMENT,
        use_residual=False,
        max_seq_len=max_seq_len,
        rng=rng,
    )


def create_decrement_transformer(input_bits: int, max_seq_len: int = 16, rng: int = None) -> RAMTransformer:
    """
    Create a transformer that decrements each token value by 1.
    B→A, C→B, D→C, ... (100% generalization, no training needed)
    """
    return RAMTransformer(
        input_bits=input_bits,
        num_blocks=1,
        attention_type=AttentionType.POSITION_ONLY,
        position_mode=PositionMode.RELATIVE,
        causal=False,
        ffn_type=FFNType.DECREMENT,
        use_residual=False,
        max_seq_len=max_seq_len,
        rng=rng,
    )


def create_rot13_transformer(input_bits: int, max_seq_len: int = 16, rng: int = None) -> RAMTransformer:
    """
    Create a transformer that applies ROT13 cipher (add 13 mod 26).
    A→N, B→O, N→A, ... (100% generalization, no training needed)
    """
    return RAMTransformer(
        input_bits=input_bits,
        num_blocks=1,
        attention_type=AttentionType.POSITION_ONLY,
        position_mode=PositionMode.RELATIVE,
        causal=False,
        ffn_type=FFNType.ROT13,
        use_residual=False,
        max_seq_len=max_seq_len,
        rng=rng,
    )


def create_caesar_transformer(
    input_bits: int,
    shift: int = 3,
    max_seq_len: int = 16,
    rng: int = None
) -> RAMTransformer:
    """
    Create a transformer that applies Caesar cipher (add N mod 26).
    With shift=3: A→D, B→E, X→A, ... (100% generalization, no training needed)
    """
    return RAMTransformer(
        input_bits=input_bits,
        num_blocks=1,
        attention_type=AttentionType.POSITION_ONLY,
        position_mode=PositionMode.RELATIVE,
        causal=False,
        ffn_type=FFNType.ADD_MOD,
        ffn_constant=shift,
        ffn_modulo=26,
        use_residual=False,
        max_seq_len=max_seq_len,
        rng=rng,
    )


def create_negate_transformer(input_bits: int, max_seq_len: int = 16, rng: int = None) -> RAMTransformer:
    """
    Create a transformer that negates each token (max_value - value).
    With 5-bit tokens: A(0)→?(31), Z(25)→F(6), ... (100% generalization)
    """
    return RAMTransformer(
        input_bits=input_bits,
        num_blocks=1,
        attention_type=AttentionType.POSITION_ONLY,
        position_mode=PositionMode.RELATIVE,
        causal=False,
        ffn_type=FFNType.NEGATE,
        use_residual=False,
        max_seq_len=max_seq_len,
        rng=rng,
    )
