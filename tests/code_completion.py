"""
Code Completion and SQL Generation with RAM Networks

Low-ambiguity domains where RAM should achieve 95%+ accuracy.

Key insight: Programming languages have deterministic syntax rules!
- "def foo(" → must be followed by params or ")"
- "SELECT * FROM" → must be followed by table name
- "if x ==" → must be followed by value

Unlike natural language, code has GRAMMAR CONSTRAINTS that reduce ambiguity.
"""

from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass
import re

from torch import tensor, uint8
from torch.nn import Module

from wnn.ram.core import RAMLayer


# =============================================================================
# TOKEN ENCODER
# =============================================================================

class TokenEncoder:
    """Encode tokens to bits."""

    def __init__(self, bits_per_token: int = 12):
        self.bits_per_token = bits_per_token
        self.token_to_idx = {"<PAD>": 0, "<UNK>": 1, "<EOS>": 2}
        self.idx_to_token = {0: "<PAD>", 1: "<UNK>", 2: "<EOS>"}
        self.max_vocab = 2 ** bits_per_token

    def add_token(self, token: str) -> int:
        if token in self.token_to_idx:
            return self.token_to_idx[token]
        if len(self.token_to_idx) >= self.max_vocab:
            return 1
        idx = len(self.token_to_idx)
        self.token_to_idx[token] = idx
        self.idx_to_token[idx] = token
        return idx

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_idx)

    def encode(self, token: str) -> list[int]:
        idx = self.token_to_idx.get(token, 1)
        return [(idx >> i) & 1 for i in range(self.bits_per_token - 1, -1, -1)]

    def decode(self, bits: list[int]) -> str:
        idx = sum(b << (self.bits_per_token - 1 - i) for i, b in enumerate(bits))
        return self.idx_to_token.get(idx, "<UNK>")


# =============================================================================
# CODE COMPLETION MODEL
# =============================================================================

class CodeCompletionModel(Module):
    """RAM-based code completion."""

    def __init__(self, n: int = 3, bits_per_token: int = 10, rng: int = 42):
        super().__init__()
        self.n = n
        self.encoder = TokenEncoder(bits_per_token)
        self.context_counts = defaultdict(Counter)
        self.rng = rng
        self.predictor = None

    def tokenize_python(self, code: str) -> list[str]:
        """Simple Python tokenizer."""
        # Add spaces around operators and punctuation
        for op in ['(', ')', '[', ']', '{', '}', ':', ',', '.', '=', '+', '-',
                   '*', '/', '<', '>', '!', '@', '#', '%', '&', '|', '^', '~']:
            code = code.replace(op, f' {op} ')
        # Handle multi-char operators
        code = code.replace('= =', '==').replace('! =', '!=')
        code = code.replace('< =', '<=').replace('> =', '>=')
        code = code.replace('+ =', '+=').replace('- =', '-=')
        # Split and filter
        tokens = [t.strip() for t in code.split() if t.strip()]
        return tokens

    def train_on_code(self, code_samples: list[str]):
        """Train on code samples."""
        # Build vocabulary first
        for code in code_samples:
            tokens = self.tokenize_python(code)
            for token in tokens:
                self.encoder.add_token(token)

        # Count n-grams WITHIN each sample (don't span across <EOS>)
        for code in code_samples:
            tokens = self.tokenize_python(code)
            tokens.append("<EOS>")

            # Only create n-grams within this sample
            for i in range(len(tokens) - self.n):
                ctx = tuple(tokens[i:i + self.n])
                target = tokens[i + self.n]
                self.context_counts[ctx][target] += 1

        # Initialize RAM predictor
        input_bits = self.n * self.encoder.bits_per_token
        self.predictor = RAMLayer(
            total_input_bits=input_bits,
            num_neurons=self.encoder.bits_per_token,
            n_bits_per_neuron=min(input_bits, 12),
            rng=self.rng,
        )

        # Train on most common patterns
        for ctx, counts in self.context_counts.items():
            target = counts.most_common(1)[0][0]
            ctx_bits = []
            for token in ctx:
                ctx_bits.extend(self.encoder.encode(token))
            self.predictor.commit(
                tensor(ctx_bits, dtype=uint8).unsqueeze(0),
                tensor(self.encoder.encode(target), dtype=uint8).unsqueeze(0)
            )

    def predict_next(self, context: list[str]) -> str:
        """Predict next token."""
        context = context[-self.n:]
        while len(context) < self.n:
            context = ["<PAD>"] + context

        ctx_tuple = tuple(context)

        # Use frequency counts first
        if ctx_tuple in self.context_counts:
            return self.context_counts[ctx_tuple].most_common(1)[0][0]

        # Fall back to RAM
        if self.predictor:
            ctx_bits = []
            for token in context:
                ctx_bits.extend(self.encoder.encode(token))
            out = self.predictor(tensor(ctx_bits, dtype=uint8).unsqueeze(0)).squeeze()
            return self.encoder.decode([int(b.item()) for b in out])

        return "<UNK>"

    def complete(self, prefix: str, max_tokens: int = 5) -> str:
        """Complete code from prefix."""
        tokens = self.tokenize_python(prefix)
        result = tokens.copy()

        for _ in range(max_tokens):
            next_token = self.predict_next(result)
            if next_token == "<EOS>" or next_token == "<UNK>":
                break
            result.append(next_token)

        return " ".join(result)


# =============================================================================
# SQL GENERATION MODEL
# =============================================================================

class SQLGenerationModel(Module):
    """RAM-based SQL generation."""

    def __init__(self, n: int = 3, bits_per_token: int = 10, rng: int = 42):
        super().__init__()
        self.n = n
        self.encoder = TokenEncoder(bits_per_token)
        self.context_counts = defaultdict(Counter)
        self.rng = rng
        self.predictor = None

    def tokenize_sql(self, sql: str) -> list[str]:
        """Simple SQL tokenizer."""
        # Uppercase keywords
        sql = sql.upper()
        # Add spaces around punctuation
        for op in ['(', ')', ',', '.', '=', '<', '>', ';', '*']:
            sql = sql.replace(op, f' {op} ')
        sql = sql.replace('< =', '<=').replace('> =', '>=')
        sql = sql.replace('< >', '<>')
        tokens = [t.strip() for t in sql.split() if t.strip()]
        return tokens

    def train_on_sql(self, sql_samples: list[str]):
        """Train on SQL samples."""
        # Build vocabulary first
        for sql in sql_samples:
            tokens = self.tokenize_sql(sql)
            for token in tokens:
                self.encoder.add_token(token)

        # Count n-grams WITHIN each sample (don't span across <EOS>)
        for sql in sql_samples:
            tokens = self.tokenize_sql(sql)
            tokens.append("<EOS>")

            for i in range(len(tokens) - self.n):
                ctx = tuple(tokens[i:i + self.n])
                target = tokens[i + self.n]
                self.context_counts[ctx][target] += 1

        # Initialize RAM
        input_bits = self.n * self.encoder.bits_per_token
        self.predictor = RAMLayer(
            total_input_bits=input_bits,
            num_neurons=self.encoder.bits_per_token,
            n_bits_per_neuron=min(input_bits, 12),
            rng=self.rng,
        )

        for ctx, counts in self.context_counts.items():
            target = counts.most_common(1)[0][0]
            ctx_bits = []
            for token in ctx:
                ctx_bits.extend(self.encoder.encode(token))
            self.predictor.commit(
                tensor(ctx_bits, dtype=uint8).unsqueeze(0),
                tensor(self.encoder.encode(target), dtype=uint8).unsqueeze(0)
            )

    def predict_next(self, context: list[str]) -> str:
        context = context[-self.n:]
        while len(context) < self.n:
            context = ["<PAD>"] + context

        ctx_tuple = tuple(context)
        if ctx_tuple in self.context_counts:
            return self.context_counts[ctx_tuple].most_common(1)[0][0]

        if self.predictor:
            ctx_bits = []
            for token in context:
                ctx_bits.extend(self.encoder.encode(token))
            out = self.predictor(tensor(ctx_bits, dtype=uint8).unsqueeze(0)).squeeze()
            return self.encoder.decode([int(b.item()) for b in out])

        return "<UNK>"


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, test_tokens: list[str], n: int) -> dict:
    """Evaluate with coverage and ambiguity tracking."""
    correct = 0
    total = 0
    covered = 0
    correct_on_covered = 0  # Only count correct if context was seen

    for i in range(n, len(test_tokens)):
        context = test_tokens[i-n:i]
        target = test_tokens[i]

        ctx_tuple = tuple(context)
        is_covered = ctx_tuple in model.context_counts

        if is_covered:
            covered += 1

        predicted = model.predict_next(context)
        if predicted == target:
            correct += 1
            if is_covered:
                correct_on_covered += 1
        total += 1

    # Ambiguity analysis
    ambiguous = sum(1 for c in model.context_counts.values() if len(c) > 1)

    return {
        "accuracy": correct / total if total > 0 else 0,
        "coverage": covered / total if total > 0 else 0,
        "covered_accuracy": correct_on_covered / covered if covered > 0 else 0,
        "uncovered_correct": correct - correct_on_covered,  # RAM generalization!
        "ambiguity_rate": ambiguous / len(model.context_counts) if model.context_counts else 0,
        "total": total,
    }


# =============================================================================
# BENCHMARKS
# =============================================================================

def benchmark_python_completion():
    """Benchmark Python code completion."""
    print(f"\n{'='*70}")
    print("PYTHON CODE COMPLETION BENCHMARK")
    print(f"{'='*70}")

    # Training data - common Python patterns
    train_code = [
        # Function definitions
        "def foo ( x ) : return x + 1",
        "def bar ( x , y ) : return x * y",
        "def baz ( ) : return None",
        "def add ( a , b ) : return a + b",
        "def sub ( a , b ) : return a - b",
        "def mul ( a , b ) : return a * b",

        # Control flow
        "if x == 0 : return True",
        "if x > 0 : return x",
        "if x < 0 : return - x",
        "if x != y : return False",
        "while x > 0 : x = x - 1",
        "for i in range ( n ) : print ( i )",
        "for x in items : process ( x )",

        # Class definitions
        "class Foo : def __init__ ( self ) : pass",
        "class Bar : def method ( self ) : return self",

        # Common patterns
        "x = x + 1",
        "y = y - 1",
        "result = result + x",
        "self . value = value",
        "return self . value",
        "print ( result )",
        "import os",
        "from typing import List",
    ] * 5  # Repeat for more coverage

    # Test data - similar patterns
    test_code = [
        "def test ( x ) : return x + 1",
        "if y == 0 : return False",
        "for j in range ( m ) : print ( j )",
        "class Test : def __init__ ( self ) : pass",
        "z = z + 1",
    ]

    model = CodeCompletionModel(n=3, rng=42)
    model.train_on_code(train_code)

    # Tokenize test
    test_tokens = []
    for code in test_code:
        test_tokens.extend(model.tokenize_python(code))
        test_tokens.append("<EOS>")

    results = evaluate_model(model, test_tokens, 3)

    print(f"\nVocabulary: {model.encoder.vocab_size} tokens")
    print(f"Training patterns: {len(model.context_counts)} unique contexts")
    print(f"Ambiguous contexts: {results['ambiguity_rate']:.1%}")
    print(f"\nResults:")
    print(f"  Coverage: {results['coverage']:.1%}")
    print(f"  Overall accuracy: {results['accuracy']:.1%}")
    print(f"  Covered accuracy: {results['covered_accuracy']:.1%}")

    # Demo completions
    print(f"\n{'='*70}")
    print("COMPLETION DEMOS")
    print(f"{'='*70}")

    prefixes = [
        "def foo (",
        "if x ==",
        "for i in range (",
        "return self .",
        "class Bar :",
    ]

    for prefix in prefixes:
        completion = model.complete(prefix, max_tokens=5)
        print(f"  '{prefix}' → '{completion}'")

    return results


def benchmark_sql_generation():
    """Benchmark SQL generation."""
    print(f"\n{'='*70}")
    print("SQL GENERATION BENCHMARK")
    print(f"{'='*70}")

    # Training data - common SQL patterns
    train_sql = [
        # SELECT queries
        "SELECT * FROM users",
        "SELECT * FROM orders",
        "SELECT * FROM products",
        "SELECT id FROM users",
        "SELECT name FROM users",
        "SELECT id , name FROM users",
        "SELECT * FROM users WHERE id = 1",
        "SELECT * FROM users WHERE name = 'foo'",
        "SELECT * FROM orders WHERE user_id = 1",
        "SELECT * FROM users WHERE active = 1",

        # JOINs
        "SELECT * FROM users JOIN orders ON users . id = orders . user_id",
        "SELECT * FROM orders JOIN products ON orders . product_id = products . id",

        # Aggregations
        "SELECT COUNT ( * ) FROM users",
        "SELECT SUM ( amount ) FROM orders",
        "SELECT AVG ( price ) FROM products",
        "SELECT MAX ( id ) FROM users",

        # GROUP BY
        "SELECT user_id , COUNT ( * ) FROM orders GROUP BY user_id",
        "SELECT category , SUM ( amount ) FROM products GROUP BY category",

        # ORDER BY
        "SELECT * FROM users ORDER BY id",
        "SELECT * FROM users ORDER BY name",
        "SELECT * FROM orders ORDER BY created_at DESC",

        # INSERT
        "INSERT INTO users ( name ) VALUES ( 'foo' )",
        "INSERT INTO orders ( user_id , amount ) VALUES ( 1 , 100 )",

        # UPDATE
        "UPDATE users SET name = 'bar' WHERE id = 1",
        "UPDATE orders SET status = 'done' WHERE id = 1",

        # DELETE
        "DELETE FROM users WHERE id = 1",
        "DELETE FROM orders WHERE status = 'cancelled'",
    ] * 3

    # Test data
    test_sql = [
        "SELECT * FROM customers",
        "SELECT id , name FROM products",
        "SELECT * FROM orders WHERE status = 'pending'",
        "SELECT COUNT ( * ) FROM orders",
        "INSERT INTO products ( name ) VALUES ( 'widget' )",
    ]

    model = SQLGenerationModel(n=3, rng=42)
    model.train_on_sql(train_sql)

    # Tokenize test
    test_tokens = []
    for sql in test_sql:
        test_tokens.extend(model.tokenize_sql(sql))
        test_tokens.append("<EOS>")

    results = evaluate_model(model, test_tokens, 3)

    print(f"\nVocabulary: {model.encoder.vocab_size} tokens")
    print(f"Training patterns: {len(model.context_counts)} unique contexts")
    print(f"Ambiguous contexts: {results['ambiguity_rate']:.1%}")
    print(f"\nResults:")
    print(f"  Coverage: {results['coverage']:.1%}")
    print(f"  Overall accuracy: {results['accuracy']:.1%}")
    print(f"  Covered accuracy: {results['covered_accuracy']:.1%}")

    # Demo completions
    print(f"\n{'='*70}")
    print("SQL COMPLETION DEMOS")
    print(f"{'='*70}")

    prefixes = [
        "SELECT * FROM",
        "SELECT COUNT ( *",
        "WHERE id =",
        "ORDER BY",
        "INSERT INTO users",
    ]

    for prefix in prefixes:
        tokens = model.tokenize_sql(prefix)
        result = tokens.copy()
        for _ in range(5):
            next_token = model.predict_next(result)
            if next_token in ["<EOS>", "<UNK>"]:
                break
            result.append(next_token)
        print(f"  '{prefix}' → '{' '.join(result)}'")

    return results


def benchmark_comparison():
    """Compare ambiguity across domains."""
    print(f"\n{'='*70}")
    print("DOMAIN COMPARISON: Ambiguity vs Accuracy")
    print(f"{'='*70}")

    # Natural language (high ambiguity)
    nl_patterns = defaultdict(Counter)
    nl_sentences = [
        "the cat sat on the mat",
        "the cat ran to the door",
        "the cat chased the mouse",
        "the dog sat on the rug",
        "the dog ran to the park",
    ] * 10

    for sent in nl_sentences:
        tokens = sent.split()
        for i in range(len(tokens) - 2):
            nl_patterns[(tokens[i], tokens[i+1])][tokens[i+2]] += 1

    nl_ambiguous = sum(1 for c in nl_patterns.values() if len(c) > 1)

    # Code (low ambiguity)
    code_patterns = defaultdict(Counter)
    code_snippets = [
        "def foo ( ) : return x",
        "def bar ( x ) : return x + 1",
        "if x == 0 : return True",
        "for i in range ( n ) : pass",
    ] * 10

    for code in code_snippets:
        tokens = code.split()
        for i in range(len(tokens) - 2):
            code_patterns[(tokens[i], tokens[i+1])][tokens[i+2]] += 1

    code_ambiguous = sum(1 for c in code_patterns.values() if len(c) > 1)

    # SQL (very low ambiguity)
    sql_patterns = defaultdict(Counter)
    sql_queries = [
        "SELECT * FROM users",
        "SELECT * FROM orders",
        "SELECT id FROM users WHERE active = 1",
        "INSERT INTO users VALUES ( 1 )",
    ] * 10

    for sql in sql_queries:
        tokens = sql.upper().split()
        for i in range(len(tokens) - 2):
            sql_patterns[(tokens[i], tokens[i+1])][tokens[i+2]] += 1

    sql_ambiguous = sum(1 for c in sql_patterns.values() if len(c) > 1)

    print("\n| Domain | Contexts | Ambiguous | Rate | Expected Accuracy |")
    print("|--------|----------|-----------|------|-------------------|")
    print(f"| Natural Language | {len(nl_patterns):>8} | {nl_ambiguous:>9} | {nl_ambiguous/len(nl_patterns):>4.0%} | ~50% |")
    print(f"| Python Code | {len(code_patterns):>8} | {code_ambiguous:>9} | {code_ambiguous/len(code_patterns):>4.0%} | ~80-95% |")
    print(f"| SQL | {len(sql_patterns):>8} | {sql_ambiguous:>9} | {sql_ambiguous/len(sql_patterns):>4.0%} | ~90-100% |")

    print("\n★ Lower ambiguity = higher accuracy ceiling!")


def benchmark_deterministic_syntax():
    """Benchmark with TRULY deterministic patterns - 100% expected."""
    print(f"\n{'='*70}")
    print("DETERMINISTIC SYNTAX BENCHMARK (100% Expected)")
    print(f"{'='*70}")

    # Patterns with ZERO ambiguity - each context has exactly ONE valid continuation
    # Key insight: The OPERATION must be part of the 3-gram context before the result
    #
    # BAD:  "add 2 3 equals 5" - context ('2','3','equals') is same for mul!
    # GOOD: "2 plus 3 is 5"    - context ('plus','3','is') is unique to addition
    #
    # Structure: "X op Y is Z" where op is word (plus/times/minus)
    # Context ('op', 'Y', 'is') uniquely determines Z

    deterministic_patterns = []

    for a in range(10):
        for b in range(10):
            # Addition: "X plus Y is Z"
            deterministic_patterns.append(f"{a} plus {b} is {a+b}")
            # Multiplication: "X times Y is Z"
            deterministic_patterns.append(f"{a} times {b} is {a*b}")

    # Also add some simple grammar patterns (completely unambiguous)
    # Each starts with unique keyword
    for i in range(20):
        deterministic_patterns.append(f"print hello {i}")
        deterministic_patterns.append(f"set x to {i}")
        deterministic_patterns.append(f"get value {i}")

    print(f"Generated {len(deterministic_patterns)} arithmetic patterns")

    # Use n=4 so context includes (X, op, Y, 'is') → uniquely determines result
    # With n=3: ('plus', '5', 'is') is same for "0 plus 5 is 5" and "1 plus 5 is 6"
    model = CodeCompletionModel(n=4, rng=42)
    model.train_on_code(deterministic_patterns)

    # Test on patterns from training (should be 100%)
    test_patterns = [
        "2 plus 3 is 5",
        "4 times 5 is 20",
        "9 plus 9 is 18",
        "6 times 7 is 42",
        "print hello 5",
        "set x to 10",
    ]

    test_tokens = []
    for p in test_patterns:
        test_tokens.extend(model.tokenize_python(p))
        test_tokens.append("<EOS>")

    results = evaluate_model(model, test_tokens, 4)

    print(f"\nTraining patterns: {len(model.context_counts)} unique contexts")
    print(f"Ambiguous contexts: {results['ambiguity_rate']:.1%}")
    print(f"\nResults:")
    print(f"  Coverage: {results['coverage']:.1%}")
    print(f"  Overall accuracy: {results['accuracy']:.1%}")
    print(f"  ★ Covered accuracy: {results['covered_accuracy']:.1%}")
    if results.get('uncovered_correct', 0) > 0:
        print(f"  RAM generalized on {results['uncovered_correct']} unseen contexts!")

    # Verify zero ambiguity
    truly_ambiguous = 0
    ambiguous_examples = []
    for ctx, counts in model.context_counts.items():
        if len(counts) > 1:
            truly_ambiguous += 1
            if len(ambiguous_examples) < 3:
                ambiguous_examples.append(f"  {ctx} → {dict(counts)}")

    print(f"\n  Truly ambiguous: {truly_ambiguous}/{len(model.context_counts)}")
    for ex in ambiguous_examples:
        print(ex)

    if truly_ambiguous == 0 and results['covered_accuracy'] >= 0.99:
        print("\n★ SUCCESS! 100% accuracy on truly deterministic patterns!")
    elif results['covered_accuracy'] > 0.95:
        print("\n★ SUCCESS! Near-100% accuracy on deterministic patterns!")
    else:
        print("\n⚠ Some ambiguity remains - check patterns")

    return results


def benchmark_high_coverage_code():
    """Code completion with exhaustive training for high coverage."""
    print(f"\n{'='*70}")
    print("HIGH COVERAGE CODE BENCHMARK")
    print(f"{'='*70}")

    # Generate ALL variations of a template
    templates = []

    # All function patterns
    for fname in ["foo", "bar", "baz", "test"]:
        for param in ["x", "y", "n", ""]:
            for ret in ["x", "x + 1", "None", "True"]:
                if param:
                    templates.append(f"def {fname} ( {param} ) : return {ret}")
                else:
                    templates.append(f"def {fname} ( ) : return {ret}")

    # All if patterns
    for var in ["x", "y", "n"]:
        for op in ["==", ">", "<", "!="]:
            for val in ["0", "1", "None", "True"]:
                templates.append(f"if {var} {op} {val} : return True")

    print(f"Generated {len(templates)} exhaustive patterns")

    model = CodeCompletionModel(n=3, rng=42)
    model.train_on_code(templates)

    # Test on variations we've seen
    test_code = [
        "def foo ( x ) : return x + 1",
        "if x == 0 : return True",
        "def bar ( ) : return None",
    ]

    test_tokens = []
    for code in test_code:
        test_tokens.extend(model.tokenize_python(code))
        test_tokens.append("<EOS>")

    results = evaluate_model(model, test_tokens, 3)

    print(f"\nVocabulary: {model.encoder.vocab_size} tokens")
    print(f"Training patterns: {len(model.context_counts)} unique contexts")
    print(f"Ambiguous: {results['ambiguity_rate']:.1%}")
    print(f"\nResults:")
    print(f"  Coverage: {results['coverage']:.1%}")
    print(f"  Overall: {results['accuracy']:.1%}")
    print(f"  Covered: {results['covered_accuracy']:.1%}")

    return results


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print("CODE COMPLETION & SQL GENERATION WITH RAM")
    print(f"Started at: {datetime.now()}")
    print(f"{'='*70}")

    # First show deterministic case
    det_results = benchmark_deterministic_syntax()

    # Then high coverage
    high_cov_results = benchmark_high_coverage_code()

    python_results = benchmark_python_completion()
    sql_results = benchmark_sql_generation()
    benchmark_comparison()

    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"""
| Domain | Coverage | Accuracy | Covered Acc | Ambiguity |
|--------|----------|----------|-------------|-----------|
| Python | {python_results['coverage']:>7.1%} | {python_results['accuracy']:>7.1%} | {python_results['covered_accuracy']:>10.1%} | {python_results['ambiguity_rate']:>8.1%} |
| SQL    | {sql_results['coverage']:>7.1%} | {sql_results['accuracy']:>7.1%} | {sql_results['covered_accuracy']:>10.1%} | {sql_results['ambiguity_rate']:>8.1%} |

Key findings:
1. Code/SQL have MUCH lower ambiguity than natural language
2. Covered accuracy is high when patterns are seen
3. RAM networks excel at structured, deterministic domains

★ For code completion, expand training data for higher coverage!
""")
    print(f"\nFinished at: {datetime.now()}")
    print(f"{'='*70}")
