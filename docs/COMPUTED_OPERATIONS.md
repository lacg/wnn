# Computed Operations for 100% Generalization

This document describes the computed operations in the RAM Transformer architecture that achieve **100% generalization** to unseen tokens without any training.

## Key Insight

Traditional neural networks **learn** transformations from training data, which limits generalization to patterns seen during training. Computed operations bypass this by **calculating** the transformation directly from the input representation.

```
LEARNED:   input → lookup in trained memory → output (limited generalization)
COMPUTED:  input → mathematical function(input) → output (100% generalization)
```

## Computed Attention Operations

Located in `src/wnn/ram/SoftRAMAttention.py`:

### SortingAttention

Sorts tokens by their numeric value using computed comparisons.

```python
from wnn.ram.SoftRAMAttention import SortingAttention

model = SortingAttention(input_bits=5, descending=False)
# No training needed!
# DCBA → ABCD (100% on any tokens)
# HGFE → EFGH (100% on unseen tokens)
```

**How it works**: Converts bit patterns to integers, sorts by value, routes accordingly.

### MinMaxAttention

Outputs the minimum or maximum token at every position.

```python
from wnn.ram.SoftRAMAttention import MinMaxAttention

model = MinMaxAttention(input_bits=5, find_max=False)
# DCBA → AAAA (min at every position)
# WXYZ → WWWW (works on unseen tokens)
```

### Content Matching (XOR_EQUAL)

Attends to positions where content matches the query using XOR comparison.

```python
# In SoftRAMAttention:
content_match=ContentMatchMode.XOR_EQUAL
# ABAB: position 0 attends to [0, 2] (both are 'A')
```

## Computed FFN Operations

Located in `src/wnn/ram/SoftRAMAttention.py` (ComputedArithmeticFFN) and integrated into `RAMTransformerBlock.py`.

### ArithmeticOp Enum

```python
class ArithmeticOp(IntEnum):
    INCREMENT = 0      # value + 1
    DECREMENT = 1      # value - 1
    ADD = 2            # value + constant
    SUBTRACT = 3       # value - constant
    ADD_MOD = 4        # (value + constant) mod N
    SUBTRACT_MOD = 5   # (value - constant) mod N
    ROT13 = 6          # (value + 13) mod 26
    NEGATE = 7         # max_value - value
```

### FFNType Enum (in RAMTransformerBlock)

```python
class FFNType(IntEnum):
    # Learned (may not generalize)
    NONE = 0
    SINGLE = 1
    TWO_LAYER = 2
    BIT_LEVEL = 3

    # Computed (100% generalization)
    INCREMENT = 10
    DECREMENT = 11
    ADD_MOD = 12
    SUBTRACT_MOD = 13
    ROT13 = 14
    NEGATE = 15
```

### Direct Usage

```python
from wnn.ram.SoftRAMAttention import ComputedArithmeticFFN, ArithmeticOp

# Increment
ffn = ComputedArithmeticFFN(input_bits=5, operation=ArithmeticOp.INCREMENT)
# A → B, B → C, ... (100%)

# ROT13
ffn = ComputedArithmeticFFN(input_bits=5, operation=ArithmeticOp.ROT13)
# A → N, N → A, HELLO → URYYB (100%)

# Caesar cipher (+5)
ffn = ComputedArithmeticFFN(
    input_bits=5,
    operation=ArithmeticOp.ADD_MOD,
    constant=5,
    modulo=26
)
# A → F, V → A (100%)
```

### Via RAMTransformerBlock

```python
from wnn.ram.RAMTransformerBlock import (
    RAMTransformer, FFNType,
    create_increment_transformer,
    create_rot13_transformer,
    create_caesar_transformer,
)

# Factory functions (recommended)
model = create_increment_transformer(input_bits=5)  # A→B, B→C
model = create_rot13_transformer(input_bits=5)      # ROT13 cipher
model = create_caesar_transformer(input_bits=5, shift=3)  # Caesar +3

# Manual configuration
model = RAMTransformer(
    input_bits=5,
    ffn_type=FFNType.ADD_MOD,
    ffn_constant=7,
    ffn_modulo=26,
)
```

### Multi-Step Pipelines

```python
from wnn.ram.RAMTransformerBlock import create_multi_step_transformer

# Shift right, then increment
model = create_multi_step_transformer(
    input_bits=5,
    steps=["shift", "increment"]
)
# ABCD → AABC → BBCD (100% on unseen)

# Sort, then ROT13
model = create_multi_step_transformer(
    input_bits=5,
    steps=["sort", "rot13"]
)
# DCBA → ABCD → NOPQ (100% on unseen)
```

Supported steps: `"copy"`, `"shift"`, `"reverse"`, `"sort"`, `"increment"`, `"decrement"`, `"rot13"`, `"negate"`

## Generalization Comparison

| Operation | Learned (BIT_LEVEL) | Computed |
|-----------|---------------------|----------|
| Copy | 100% | 100% |
| Shift | 100% | 100% |
| Reverse | 100% | 100% |
| Sorting | 0% (content-dependent) | **100%** |
| Increment | ~100% (bit-regular) | **100%** |
| ROT13 | 0-30% (irregular) | **100%** |
| Caesar +N | 0-30% | **100%** |
| Find Min/Max | 0% | **100%** |

## When to Use Each

### Use Computed Operations When:
- The transformation is mathematically defined
- You need 100% generalization on unseen tokens
- The operation is: sorting, comparison, arithmetic, cipher

### Use Learned Operations When:
- The transformation must be discovered from data
- The pattern is unknown or data-dependent
- You're willing to accept limited generalization

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   RAMTransformerBlock                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Input Tokens                                           │
│        │                                                 │
│        ▼                                                 │
│   ┌─────────────────────────────────────────────────┐   │
│   │               ATTENTION                          │   │
│   │  ┌─────────────┬─────────────┬─────────────┐    │   │
│   │  │ SOFT_RAM    │ SORTING     │ MIN_MAX     │    │   │
│   │  │ (learned)   │ (computed)  │ (computed)  │    │   │
│   │  └─────────────┴─────────────┴─────────────┘    │   │
│   └──────────────────────┬──────────────────────────┘   │
│                          │                               │
│                          ⊕ ← XOR Residual (optional)    │
│                          │                               │
│                          ▼                               │
│   ┌─────────────────────────────────────────────────┐   │
│   │                  FFN                             │   │
│   │  ┌─────────────┬─────────────┬─────────────┐    │   │
│   │  │ BIT_LEVEL   │ INCREMENT   │ ROT13       │    │   │
│   │  │ (learned)   │ (computed)  │ (computed)  │    │   │
│   │  └─────────────┴─────────────┴─────────────┘    │   │
│   └──────────────────────┬──────────────────────────┘   │
│                          │                               │
│                          ⊕ ← XOR Residual (optional)    │
│                          │                               │
│                          ▼                               │
│   Output Tokens                                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Implementation Details

### Bit-to-Integer Conversion

```python
def _bits_to_int(bits: Tensor) -> int:
    """Convert bit tensor to integer (MSB first)."""
    val = 0
    for b in bits:
        val = val * 2 + int(b.item())
    return val

def _int_to_bits(value: int, num_bits: int) -> Tensor:
    """Convert integer to bit tensor (MSB first)."""
    bits = zeros(num_bits, dtype=uint8)
    for i in range(num_bits - 1, -1, -1):
        bits[num_bits - 1 - i] = (value >> i) & 1
    return bits
```

### Token Encoding Requirement

Computed arithmetic assumes tokens are encoded with **numeric ordering**:
- A = 0 (00000 in 5-bit)
- B = 1 (00001)
- ...
- Z = 25 (11001)

This allows comparison and arithmetic operations to work correctly.

## Examples

### Caesar Cipher Encrypt/Decrypt

```python
# Encrypt
encrypt = create_caesar_transformer(input_bits=5, shift=3)
# HELLO → KHOOR

# Decrypt (using SUBTRACT_MOD)
decrypt = RAMTransformer(
    input_bits=5,
    ffn_type=FFNType.SUBTRACT_MOD,
    ffn_constant=3,
    ffn_modulo=26,
)
# KHOOR → HELLO
```

### ROT13 is Self-Inverse

```python
model = create_multi_step_transformer(
    input_bits=5,
    steps=["rot13", "rot13"]
)
# HELLO → URYYB → HELLO
```

### Sort Then Transform

```python
model = create_multi_step_transformer(
    input_bits=5,
    steps=["sort", "increment"]
)
# DCBA → ABCD → BCDE (sort, then add 1 to each)
```
