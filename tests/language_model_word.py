"""
Word-Level Language Modeling with Probabilistic Outputs

Two improvements over character-level models:
1. Word-level: Predict next word instead of next character (reduces ambiguity)
2. Probabilistic: Return distribution over possible outputs, not just top-1

Key insight: Word boundaries reduce ambiguity significantly.
- Character "t" after "the ca" could be many things
- Word after "the cat" is much more constrained

| Approach | Level | Output | Expected Improvement |
|----------|-------|--------|---------------------|
| FrequencyAware | char | single | 79% baseline |
| WordLevel | word | single | Higher (less ambiguity) |
| WordLevel + Prob | word | distribution | Captures uncertainty |
"""

from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass

from torch import zeros, uint8, tensor, Tensor
from torch.nn import Module

from wnn.ram.core import RAMLayer


# =============================================================================
# WORD ENCODING
# =============================================================================

class WordEncoder:
	"""Encode/decode words to/from bit vectors."""

	def __init__(self, vocab: list[str] | None = None, bits_per_word: int = 10):
		self.bits_per_word = bits_per_word
		self.max_vocab = 2 ** bits_per_word

		if vocab is None:
			vocab = []

		# Reserve special tokens
		self.UNK = "<UNK>"
		self.PAD = "<PAD>"
		self.EOS = "<EOS>"

		# Build vocabulary
		self.word_to_idx = {self.PAD: 0, self.UNK: 1, self.EOS: 2}
		self.idx_to_word = {0: self.PAD, 1: self.UNK, 2: self.EOS}

		for word in vocab:
			if word not in self.word_to_idx and len(self.word_to_idx) < self.max_vocab:
				idx = len(self.word_to_idx)
				self.word_to_idx[word] = idx
				self.idx_to_word[idx] = word

		self.vocab_size = len(self.word_to_idx)

	def add_word(self, word: str) -> int:
		"""Add word to vocabulary, return its index."""
		if word in self.word_to_idx:
			return self.word_to_idx[word]
		if len(self.word_to_idx) >= self.max_vocab:
			return self.word_to_idx[self.UNK]
		idx = len(self.word_to_idx)
		self.word_to_idx[word] = idx
		self.idx_to_word[idx] = word
		self.vocab_size = len(self.word_to_idx)
		return idx

	def encode_word(self, word: str) -> list[int]:
		"""Encode word as bit vector."""
		idx = self.word_to_idx.get(word, self.word_to_idx[self.UNK])
		return [(idx >> i) & 1 for i in range(self.bits_per_word - 1, -1, -1)]

	def decode_bits(self, bits: list[int]) -> str:
		"""Decode bit vector to word."""
		idx = sum(b << (self.bits_per_word - 1 - i) for i, b in enumerate(bits))
		return self.idx_to_word.get(idx, self.UNK)

	def tokenize(self, text: str) -> list[str]:
		"""Simple whitespace tokenization."""
		# Normalize and split
		text = text.lower().strip()
		# Handle punctuation as separate tokens
		for punct in '.,!?;:':
			text = text.replace(punct, f' {punct} ')
		return [w for w in text.split() if w]


# =============================================================================
# PROBABILISTIC OUTPUT
# =============================================================================

@dataclass
class ProbabilisticPrediction:
	"""A prediction with probability distribution."""
	predictions: list[tuple[str, float]]  # [(word, probability), ...]

	@property
	def top1(self) -> str:
		"""Get most likely prediction."""
		return self.predictions[0][0] if self.predictions else "<UNK>"

	@property
	def top1_prob(self) -> float:
		"""Get probability of top prediction."""
		return self.predictions[0][1] if self.predictions else 0.0

	def top_k(self, k: int = 5) -> list[tuple[str, float]]:
		"""Get top-k predictions."""
		return self.predictions[:k]

	@property
	def entropy(self) -> float:
		"""Compute entropy of distribution (uncertainty measure)."""
		import math
		h = 0.0
		for _, p in self.predictions:
			if p > 0:
				h -= p * math.log2(p)
		return h

	@property
	def is_confident(self) -> bool:
		"""Check if prediction is confident (top-1 > 50%)."""
		return self.top1_prob > 0.5


# =============================================================================
# WORD-LEVEL LANGUAGE MODELS
# =============================================================================

class WordLevelNGram(Module):
	"""Word-level n-gram language model."""

	def __init__(self, n: int = 2, bits_per_word: int = 10, rng: int | None = None):
		super().__init__()
		self.n = n
		self.bits_per_word = bits_per_word
		self.encoder = WordEncoder(bits_per_word=bits_per_word)

		# Will be initialized after vocabulary is built
		self.predictor = None
		self.rng = rng
		self.patterns_trained = 0

	def _init_predictor(self):
		"""Initialize predictor after vocabulary is known."""
		input_bits = self.n * self.bits_per_word
		self.predictor = RAMLayer(
			total_input_bits=input_bits,
			num_neurons=self.bits_per_word,
			n_bits_per_neuron=min(input_bits, 12),
			rng=self.rng,
		)

	def train_on_text(self, text: str) -> int:
		"""Train on text, building vocabulary first."""
		words = self.encoder.tokenize(text)

		# Build vocabulary
		for word in words:
			self.encoder.add_word(word)

		# Initialize predictor
		self._init_predictor()

		# Train on n-grams
		errors = 0
		patterns = set()

		for i in range(len(words) - self.n):
			context = tuple(words[i:i + self.n])
			target = words[i + self.n]

			if (context, target) in patterns:
				continue
			patterns.add((context, target))

			# Encode context
			ctx_bits = []
			for word in context:
				ctx_bits.extend(self.encoder.encode_word(word))

			inp = tensor(ctx_bits, dtype=uint8)
			out = tensor(self.encoder.encode_word(target), dtype=uint8)
			errors += self.predictor.commit(inp.unsqueeze(0), out.unsqueeze(0))

		self.patterns_trained = len(patterns)
		return errors

	def predict_next(self, context: list[str]) -> str:
		"""Predict next word given context."""
		if self.predictor is None:
			return self.encoder.UNK

		# Pad or truncate context
		context = context[-self.n:]
		while len(context) < self.n:
			context = [self.encoder.PAD] + context

		# Encode
		ctx_bits = []
		for word in context:
			ctx_bits.extend(self.encoder.encode_word(word))

		out = self.predictor(tensor(ctx_bits, dtype=uint8).unsqueeze(0)).squeeze()
		return self.encoder.decode_bits([int(b.item()) for b in out])


class FrequencyAwareWordModel(Module):
	"""Word-level model with frequency tracking and probabilistic output."""

	def __init__(self, n: int = 2):
		super().__init__()
		self.n = n
		self.encoder = WordEncoder()
		self.context_counts = defaultdict(Counter)
		self.total_counts = Counter()

	def train_on_text(self, text: str):
		"""Train on text."""
		words = self.encoder.tokenize(text)

		# Build vocabulary
		for word in words:
			self.encoder.add_word(word)
			self.total_counts[word] += 1

		# Count n-gram frequencies
		for i in range(len(words) - self.n):
			context = tuple(words[i:i + self.n])
			target = words[i + self.n]
			self.context_counts[context][target] += 1

	def predict_next(self, context: list[str]) -> str:
		"""Predict single most likely next word."""
		return self.predict_probabilistic(context).top1

	def predict_probabilistic(self, context: list[str]) -> ProbabilisticPrediction:
		"""Predict next word with probability distribution."""
		# Normalize context
		context = [w.lower() for w in context[-self.n:]]
		while len(context) < self.n:
			context = [self.encoder.PAD] + context

		ctx_tuple = tuple(context)

		if ctx_tuple in self.context_counts:
			counts = self.context_counts[ctx_tuple]
			total = sum(counts.values())
			predictions = [
				(word, count / total)
				for word, count in counts.most_common()
			]
			return ProbabilisticPrediction(predictions)

		# Backoff to unigram distribution
		if self.total_counts:
			total = sum(self.total_counts.values())
			predictions = [
				(word, count / total)
				for word, count in self.total_counts.most_common(10)
			]
			return ProbabilisticPrediction(predictions)

		return ProbabilisticPrediction([(self.encoder.UNK, 1.0)])


class HierarchicalWordModel(Module):
	"""Hierarchical word-level model with backoff and probabilistic output."""

	def __init__(self, max_n: int = 3):
		super().__init__()
		self.max_n = max_n
		self.encoder = WordEncoder()
		self.ngram_counts = {n: defaultdict(Counter) for n in range(1, max_n + 1)}
		self.total_counts = Counter()

	def train_on_text(self, text: str):
		"""Train on text."""
		words = self.encoder.tokenize(text)

		# Build vocabulary and unigram counts
		for word in words:
			self.encoder.add_word(word)
			self.total_counts[word] += 1

		# Count n-grams at all levels
		for n in range(1, self.max_n + 1):
			for i in range(len(words) - n):
				context = tuple(words[i:i + n])
				target = words[i + n]
				self.ngram_counts[n][context][target] += 1

	def predict_probabilistic(self, context: list[str]) -> ProbabilisticPrediction:
		"""Predict with backoff through n-gram levels."""
		context = [w.lower() for w in context]

		# Try each n-gram level from highest to lowest
		for n in range(min(self.max_n, len(context)), 0, -1):
			ctx_tuple = tuple(context[-n:])
			if ctx_tuple in self.ngram_counts[n]:
				counts = self.ngram_counts[n][ctx_tuple]
				total = sum(counts.values())
				predictions = [
					(word, count / total)
					for word, count in counts.most_common()
				]
				return ProbabilisticPrediction(predictions)

		# Backoff to unigram
		if self.total_counts:
			total = sum(self.total_counts.values())
			predictions = [
				(word, count / total)
				for word, count in self.total_counts.most_common(10)
			]
			return ProbabilisticPrediction(predictions)

		return ProbabilisticPrediction([(self.encoder.UNK, 1.0)])

	def predict_next(self, context: list[str]) -> str:
		"""Predict single most likely next word."""
		return self.predict_probabilistic(context).top1


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, test_words: list[str], n: int) -> dict:
	"""Evaluate model on test data."""
	correct = 0
	total = 0
	top3_correct = 0
	confident_correct = 0
	confident_total = 0
	total_entropy = 0.0

	has_probabilistic = hasattr(model, 'predict_probabilistic')

	for i in range(n, len(test_words)):
		context = test_words[i-n:i]
		target = test_words[i]

		if has_probabilistic:
			pred = model.predict_probabilistic(context)
			predicted = pred.top1

			# Check top-3
			top3_words = [w for w, _ in pred.top_k(3)]
			if target in top3_words:
				top3_correct += 1

			# Track confident predictions
			if pred.is_confident:
				confident_total += 1
				if predicted == target:
					confident_correct += 1

			total_entropy += pred.entropy
		else:
			predicted = model.predict_next(context)

		if predicted == target:
			correct += 1
		total += 1

	results = {
		'accuracy': correct / total if total > 0 else 0,
		'total': total,
		'correct': correct,
	}

	if has_probabilistic:
		results['top3_accuracy'] = top3_correct / total if total > 0 else 0
		results['avg_entropy'] = total_entropy / total if total > 0 else 0
		results['confident_accuracy'] = confident_correct / confident_total if confident_total > 0 else 0
		results['confident_ratio'] = confident_total / total if total > 0 else 0

	return results


def compare_word_models():
	"""Compare word-level models."""
	print(f"\n{'='*70}")
	print("Word-Level Language Model Comparison")
	print(f"{'='*70}")

	# Training corpus - more text for word-level
	train_text = """
	the cat sat on the mat. the dog ran to the park.
	the cat chased the dog around the house.
	the dog barked at the cat in the garden.
	a bird flew over the house and landed on the tree.
	the cat watched the bird from the window.
	the dog slept on the mat near the door.
	the cat and the dog played in the yard.
	the bird sang a beautiful song in the morning.
	the cat caught a mouse in the kitchen.
	the dog fetched the ball from the park.
	"""

	# Test corpus - some overlap, some novel combinations
	test_text = """
	the cat ran to the garden. the dog sat on the mat.
	the bird flew over the park and sang a song.
	the cat watched the dog from the window.
	"""

	# Tokenize
	encoder = WordEncoder()
	train_words = encoder.tokenize(train_text)
	test_words = encoder.tokenize(test_text)

	print(f"\nTraining: {len(train_words)} words, {len(set(train_words))} unique")
	print(f"Testing: {len(test_words)} words")

	# Models to compare
	models = {
		"WordLevel N-gram (n=2)": WordLevelNGram(n=2, rng=42),
		"FrequencyAware Word (n=2)": FrequencyAwareWordModel(n=2),
		"Hierarchical Word (n=3)": HierarchicalWordModel(max_n=3),
	}

	# Train all models
	print("\nTraining models...")
	for name, model in models.items():
		model.train_on_text(train_text)

	# Evaluate
	print(f"\n{'='*70}")
	print("Results")
	print(f"{'='*70}")

	for name, model in models.items():
		n = model.n if hasattr(model, 'n') else 2
		results = evaluate_model(model, test_words, n)

		print(f"\n{name}:")
		print(f"  Top-1 Accuracy: {results['accuracy']:.1%}")

		if 'top3_accuracy' in results:
			print(f"  Top-3 Accuracy: {results['top3_accuracy']:.1%}")
			print(f"  Avg Entropy: {results['avg_entropy']:.2f} bits")
			print(f"  Confident Ratio: {results['confident_ratio']:.1%}")
			print(f"  Confident Accuracy: {results['confident_accuracy']:.1%}")


def demonstrate_probabilistic():
	"""Demonstrate probabilistic predictions."""
	print(f"\n{'='*70}")
	print("Probabilistic Prediction Demo")
	print(f"{'='*70}")

	train_text = """
	the cat sat on the mat. the cat ran to the door.
	the cat chased the mouse. the cat watched the bird.
	the dog sat on the floor. the dog ran to the park.
	"""

	model = FrequencyAwareWordModel(n=2)
	model.train_on_text(train_text)

	test_contexts = [
		["the", "cat"],
		["the", "dog"],
		["sat", "on"],
		["ran", "to"],
	]

	print("\nContext → Probabilistic Predictions:")
	print("-" * 50)

	for context in test_contexts:
		pred = model.predict_probabilistic(context)
		print(f"\n'{' '.join(context)}' →")
		print(f"  Entropy: {pred.entropy:.2f} bits ({'uncertain' if pred.entropy > 1 else 'confident'})")
		for word, prob in pred.top_k(5):
			bar = '█' * int(prob * 20)
			print(f"    {word:10s} {prob:5.1%} {bar}")


def compare_char_vs_word():
	"""Compare character-level vs word-level accuracy."""
	print(f"\n{'='*70}")
	print("Character-Level vs Word-Level Comparison")
	print(f"{'='*70}")

	train_text = """
	the cat sat on the mat. the dog ran to the park.
	the cat chased the dog around the house.
	the dog barked at the cat in the garden.
	"""

	test_text = "the cat ran to the garden"

	# Character-level (from original)
	from collections import Counter, defaultdict

	class CharFrequencyModel:
		def __init__(self, n=3):
			self.n = n
			self.context_counts = defaultdict(Counter)

		def train(self, text):
			text = text.lower()
			for i in range(len(text) - self.n):
				ctx = text[i:i+self.n]
				target = text[i+self.n]
				self.context_counts[ctx][target] += 1

		def predict(self, context):
			ctx = context[-self.n:].lower()
			if ctx in self.context_counts:
				return self.context_counts[ctx].most_common(1)[0][0]
			return ' '

	# Train both
	char_model = CharFrequencyModel(n=3)
	char_model.train(train_text)

	word_model = FrequencyAwareWordModel(n=2)
	word_model.train_on_text(train_text)

	# Evaluate character-level
	test_lower = test_text.lower()
	char_correct = 0
	for i in range(3, len(test_lower)):
		if char_model.predict(test_lower[:i]) == test_lower[i]:
			char_correct += 1
	char_acc = char_correct / (len(test_lower) - 3)

	# Evaluate word-level
	test_words = word_model.encoder.tokenize(test_text)
	word_results = evaluate_model(word_model, test_words, 2)

	print(f"\nTest: '{test_text}'")
	print(f"\nCharacter-level (n=3): {char_acc:.1%}")
	print(f"Word-level (n=2): {word_results['accuracy']:.1%}")
	print(f"Word-level top-3: {word_results.get('top3_accuracy', 0):.1%}")

	# Show why word-level is better
	print("\n" + "-"*50)
	print("Why word-level reduces ambiguity:")
	print("-"*50)
	print("\nCharacter 't' after 'the ca' could be: t, r, n, ...")
	print("Word after 'the cat' is more constrained: ran, sat, chased, ...")


def large_corpus_comparison():
	"""Compare with larger corpus where word-level should excel."""
	print(f"\n{'='*70}")
	print("Large Corpus Comparison (Word-Level Advantage)")
	print(f"{'='*70}")

	# Larger, more repetitive corpus - word-level should do better here
	train_sentences = [
		"the cat sat on the mat",
		"the dog sat on the rug",
		"the cat ran to the door",
		"the dog ran to the park",
		"the bird flew to the tree",
		"the cat chased the mouse",
		"the dog chased the cat",
		"the bird watched the cat",
		"the cat watched the bird",
		"the dog watched the bird",
		"a big cat sat on a big mat",
		"a small dog ran to a small park",
		"the black cat chased the white mouse",
		"the brown dog ran to the green park",
		"the cat and the dog played",
		"the bird and the cat watched",
	] * 5  # Repeat for more data

	train_text = ". ".join(train_sentences) + "."

	# Test with seen patterns
	test_seen = "the cat sat on the rug. the dog ran to the door."
	# Test with novel combinations
	test_novel = "the bird sat on the mat. the cat flew to the park."

	# Character-level model
	class CharFrequencyModel:
		def __init__(self, n=3):
			self.n = n
			self.context_counts = defaultdict(Counter)

		def train(self, text):
			text = text.lower()
			for i in range(len(text) - self.n):
				self.context_counts[text[i:i+self.n]][text[i+self.n]] += 1

		def predict(self, context):
			ctx = context[-self.n:].lower()
			if ctx in self.context_counts:
				return self.context_counts[ctx].most_common(1)[0][0]
			return ' '

	# Train models
	char_model = CharFrequencyModel(n=3)
	char_model.train(train_text)

	word_model = HierarchicalWordModel(max_n=3)
	word_model.train_on_text(train_text)

	def eval_char(model, text):
		text = text.lower()
		correct = 0
		for i in range(3, len(text)):
			if model.predict(text[:i]) == text[i]:
				correct += 1
		return correct / (len(text) - 3) if len(text) > 3 else 0

	# Evaluate
	print(f"\nTraining: {len(train_text)} chars, {len(word_model.encoder.tokenize(train_text))} words")

	print("\n--- Test on SEEN patterns (recombined) ---")
	char_acc = eval_char(char_model, test_seen)
	test_words = word_model.encoder.tokenize(test_seen)
	word_results = evaluate_model(word_model, test_words, 2)
	print(f"Character-level: {char_acc:.1%}")
	print(f"Word-level top-1: {word_results['accuracy']:.1%}")
	print(f"Word-level top-3: {word_results.get('top3_accuracy', 0):.1%}")

	print("\n--- Test on NOVEL patterns (unseen combinations) ---")
	char_acc = eval_char(char_model, test_novel)
	test_words = word_model.encoder.tokenize(test_novel)
	word_results = evaluate_model(word_model, test_words, 2)
	print(f"Character-level: {char_acc:.1%}")
	print(f"Word-level top-1: {word_results['accuracy']:.1%}")
	print(f"Word-level top-3: {word_results.get('top3_accuracy', 0):.1%}")


def perplexity_comparison():
	"""Compare perplexity (proper LM metric) between approaches."""
	print(f"\n{'='*70}")
	print("Perplexity Comparison (Lower is Better)")
	print(f"{'='*70}")

	import math

	train_text = """
	the cat sat on the mat. the dog ran to the park.
	the cat chased the mouse. the dog chased the cat.
	the bird flew to the tree. the cat watched the bird.
	""" * 10

	test_text = "the cat ran to the tree. the dog sat on the mat."

	model = HierarchicalWordModel(max_n=3)
	model.train_on_text(train_text)

	test_words = model.encoder.tokenize(test_text)

	# Compute perplexity
	log_prob_sum = 0.0
	n_predictions = 0

	print("\nPer-word predictions:")
	print("-" * 60)

	for i in range(2, len(test_words)):
		context = test_words[i-2:i]
		target = test_words[i]

		pred = model.predict_probabilistic(context)

		# Find probability of actual target
		prob = 0.0
		for word, p in pred.predictions:
			if word == target:
				prob = p
				break

		if prob == 0:
			prob = 0.01  # Smoothing for unseen

		log_prob_sum += math.log2(prob)
		n_predictions += 1

		# Show prediction details
		top_pred = pred.top1
		status = "✓" if top_pred == target else "✗"
		print(f"  '{' '.join(context)}' → '{target}' "
				f"(predicted: '{top_pred}', p={prob:.2f}) {status}")

	# Perplexity = 2^(-avg_log_prob)
	avg_log_prob = log_prob_sum / n_predictions
	perplexity = 2 ** (-avg_log_prob)

	print("-" * 60)
	print(f"\nPerplexity: {perplexity:.2f}")
	print(f"(Lower is better. Random over {model.encoder.vocab_size} words = {model.encoder.vocab_size:.0f})")


def accuracy_summary():
	"""Final accuracy summary with probabilistic metrics."""
	print(f"\n{'='*70}")
	print("FINAL ACCURACY SUMMARY")
	print(f"{'='*70}")

	# Large repetitive corpus
	train_text = """
	the cat sat on the mat. the dog ran to the park.
	the cat chased the mouse. the dog chased the cat.
	the bird flew over the house. the cat watched the bird.
	the dog slept on the floor. the cat slept on the bed.
	the bird sang in the morning. the dog barked at night.
	""" * 20  # Lots of repetition

	# Test with mix of seen and novel
	test_text = """
	the cat ran to the park. the dog sat on the mat.
	the bird flew over the tree. the cat chased the bird.
	"""

	# Train models
	char_model_counts = defaultdict(Counter)
	text_lower = train_text.lower()
	for i in range(len(text_lower) - 3):
		char_model_counts[text_lower[i:i+3]][text_lower[i+3]] += 1

	word_model = HierarchicalWordModel(max_n=3)
	word_model.train_on_text(train_text)

	# Evaluate character-level
	test_lower = test_text.lower()
	char_correct = 0
	char_top3_correct = 0
	for i in range(3, len(test_lower)):
		ctx = test_lower[i-3:i]
		target = test_lower[i]
		if ctx in char_model_counts:
			top = char_model_counts[ctx].most_common(3)
			if top[0][0] == target:
				char_correct += 1
			if target in [w for w, _ in top]:
				char_top3_correct += 1
	char_total = len(test_lower) - 3

	# Evaluate word-level
	test_words = word_model.encoder.tokenize(test_text)
	word_results = evaluate_model(word_model, test_words, 2)

	print("\n| Model | Top-1 | Top-3 | Confident Acc |")
	print("|-------|-------|-------|---------------|")
	print(f"| Character (n=3) | {char_correct/char_total:.1%} | {char_top3_correct/char_total:.1%} | - |")
	print(f"| Word (n=3) | {word_results['accuracy']:.1%} | {word_results.get('top3_accuracy', 0):.1%} | {word_results.get('confident_accuracy', 0):.1%} |")

	print("\n--- Key Insight ---")
	print("Word-level with probabilistic output provides:")
	print("1. Top-3 accuracy: Captures 'reasonable' predictions")
	print("2. Confidence filtering: Higher accuracy on confident predictions")
	print("3. Entropy: Quantifies prediction uncertainty")


if __name__ == "__main__":
	print(f"\n{'='*70}")
	print("Word-Level Language Modeling with Probabilistic Outputs")
	print(f"Started at: {datetime.now()}")
	print(f"{'='*70}")

	compare_word_models()
	demonstrate_probabilistic()
	large_corpus_comparison()
	perplexity_comparison()
	accuracy_summary()

	print(f"\n{'='*70}")
	print("CONCLUSIONS")
	print(f"{'='*70}")
	print("""
1. WORD-LEVEL vs CHARACTER-LEVEL:
	 - Word-level needs more training data (sparsity problem)
	 - But reduces ambiguity when data is sufficient
	 - Better for repetitive/formulaic text

2. PROBABILISTIC OUTPUT:
	 - Top-3 accuracy >> Top-1 accuracy (captures uncertainty)
	 - Confident predictions have higher accuracy
	 - Entropy measures "how sure" the model is

3. BEST APPROACH:
	 - Use probabilistic word-level for structured text
	 - Filter by confidence for higher precision
	 - Fall back to character-level for novel contexts
""")
	print(f"\nFinished at: {datetime.now()}")
	print(f"{'='*70}")
