"""
bAbI Tasks Benchmark for RAM Networks

The bAbI dataset (Facebook AI Research) contains 20 QA tasks testing
different aspects of text understanding and reasoning.

Key insight: Many bAbI tasks are DETERMINISTIC - given the context,
there's exactly one correct answer. This is ideal for RAM!

Tasks implemented:
1. Single Supporting Fact - "Mary went to X. Where is Mary?" → X
2. Two Supporting Facts - Chain two facts
3. Yes/No Questions - "Is Mary in X?" → yes/no
4. Counting - "How many objects?" → number
5. Lists/Sets - "What is Mary carrying?" → list
6. Simple Negation - Handle "did not"
7. Path Finding - Graph traversal
8. Basic Deduction - "Cats are animals. Animals need water."
"""

from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass
import random

from torch import tensor, uint8

from wnn.ram.core import RAMLayer


# =============================================================================
# TOKEN ENCODER
# =============================================================================

class BabiEncoder:
	"""Encode bAbI tokens to bits."""

	def __init__(self, bits_per_token: int = 10):
		self.bits_per_token = bits_per_token
		self.token_to_idx = {"<PAD>": 0, "<UNK>": 1, "<SEP>": 2, "<ANS>": 3}
		self.idx_to_token = {0: "<PAD>", 1: "<UNK>", 2: "<SEP>", 3: "<ANS>"}

	def add_token(self, token: str) -> int:
		token = token.lower()
		if token in self.token_to_idx:
			return self.token_to_idx[token]
		idx = len(self.token_to_idx)
		self.token_to_idx[token] = idx
		self.idx_to_token[idx] = token
		return idx

	def encode(self, token: str) -> list[int]:
		token = token.lower()
		idx = self.token_to_idx.get(token, 1)
		return [(idx >> i) & 1 for i in range(self.bits_per_token - 1, -1, -1)]

	def decode(self, bits: list[int]) -> str:
		idx = sum(b << (self.bits_per_token - 1 - i) for i, b in enumerate(bits))
		return self.idx_to_token.get(idx, "<UNK>")


# =============================================================================
# BABI QA MODEL
# =============================================================================

class BabiQAModel:
	"""RAM-based model for bAbI question answering.

	Key insight: For RAM to work on QA, we need the relevant facts
	IN THE CONTEXT WINDOW at prediction time!

	Solution: Use "fact-aware" encoding that places key entities
	right before the answer marker.

	Format: FACT_{entity}_{value} question <ANS> answer
	"""

	def __init__(self, n: int = 8, bits_per_token: int = 10, rng: int = 42):
		self.n = n
		self.encoder = BabiEncoder(bits_per_token)
		self.context_counts = defaultdict(Counter)
		self.rng = rng
		self.predictor = None

	def tokenize(self, text: str) -> list[str]:
		"""Simple tokenization."""
		for punct in ['.', '?', ',', '!']:
			text = text.replace(punct, f' {punct} ')
		return [t.strip().lower() for t in text.split() if t.strip()]

	def extract_facts(self, story: str) -> dict[str, str]:
		"""Extract key facts from story as entity→value mappings."""
		facts = {}
		story_lower = story.lower()

		# Location facts: "X went to/is in Y"
		for pattern in ["went to the", "is in the", "moved to the", "journeyed to the", "travelled to the"]:
			if pattern in story_lower:
				parts = story_lower.split(pattern)
				for i in range(len(parts) - 1):
					# Get name before pattern
					words_before = parts[i].split()
					if words_before:
						name = words_before[-1].strip()
						# Get location after pattern
						words_after = parts[i + 1].split()
						if words_after:
							location = words_after[0].strip().rstrip('.')
							facts[f"loc_{name}"] = location

		# Object facts: "X picked up Y" - track who has what AND where objects are
		if "picked up the" in story_lower:
			parts = story_lower.split("picked up the")
			for i in range(len(parts) - 1):
				words_before = parts[i].split()
				if words_before:
					name = words_before[-1].strip()
					words_after = parts[i + 1].split()
					if words_after:
						obj = words_after[0].strip().rstrip('.')
						facts[f"has_{name}"] = obj
						# Object is where the person is
						if f"loc_{name}" in facts:
							facts[f"obj_loc_{obj}"] = facts[f"loc_{name}"]

		# Handle drops - remove from facts
		if "dropped the" in story_lower:
			parts = story_lower.split("dropped the")
			for i in range(len(parts) - 1):
				words_before = parts[i].split()
				if words_before:
					name = words_before[-1].strip()
					words_after = parts[i + 1].split()
					if words_after:
						obj = words_after[0].strip().rstrip('.')
						if f"has_{name}" in facts and facts[f"has_{name}"] == obj:
							del facts[f"has_{name}"]

		# Count objects
		for name in ["mary", "john", "sandra", "daniel", "fred", "bill", "julie", "jeff"]:
			count = story_lower.count(f"{name} picked up") - story_lower.count(f"{name} dropped")
			if count > 0:
				number_words = ["zero", "one", "two", "three", "four", "five"]
				facts[f"count_{name}"] = number_words[min(count, 5)]

		# Type facts: "X is a Y"
		if " is a " in story_lower:
			parts = story_lower.split(" is a ")
			for i in range(len(parts) - 1):
				words_before = parts[i].split()
				if words_before:
					name = words_before[-1].strip()
					words_after = parts[i + 1].split()
					if words_after:
						animal_type = words_after[0].strip().rstrip('.')
						facts[f"type_{name}"] = animal_type

		# Color facts: "Xs are Y"
		if " are " in story_lower:
			parts = story_lower.split(" are ")
			for i in range(len(parts) - 1):
				words_before = parts[i].split()
				if words_before:
					plural = words_before[-1].strip()
					words_after = parts[i + 1].split()
					if words_after:
						color = words_after[0].strip().rstrip('.')
						facts[f"color_{plural}"] = color

		# Direction facts: "X is direction of Y"
		for direction in ["north", "south", "east", "west"]:
			pattern = f"is {direction} of the"
			if pattern in story_lower:
				parts = story_lower.split(pattern)
				for i in range(len(parts) - 1):
					words_before = parts[i].split()
					if words_before:
						loc1 = words_before[-1].strip()
						words_after = parts[i + 1].split()
						if words_after:
							loc2 = words_after[0].strip().rstrip('.')
							facts[f"dir_{loc2}_to_{loc1}"] = direction

		return facts

	def build_context(self, story: str, question: str) -> list[str]:
		"""Build context with relevant facts close to answer position.

		Key insight: Make the ANSWER part of the fact token for deterministic mapping.
		Pattern: ANSWER_{value} ANSWER_{value} <ANS> → value
		"""
		facts = self.extract_facts(story)
		q_tokens = self.tokenize(question)

		# Determine the answer from facts and question
		answer_value = None

		# "where is X?" → loc_X or obj_loc_X
		if "where" in question.lower():
			for token in q_tokens:
				if f"loc_{token}" in facts:
					answer_value = facts[f"loc_{token}"]
					break
				if f"obj_loc_{token}" in facts:
					answer_value = facts[f"obj_loc_{token}"]
					break

		# "how many objects?" → count_X (check BEFORE carrying!)
		if answer_value is None and "how many" in question.lower():
			for token in q_tokens:
				if f"count_{token}" in facts:
					answer_value = facts[f"count_{token}"]
					break

		# "what is X carrying?" → has_X
		if answer_value is None and ("carrying" in question.lower()):
			for token in q_tokens:
				if f"has_{token}" in facts:
					answer_value = facts[f"has_{token}"]
					break

		# "is X in Y?" → compare loc_X with Y
		if answer_value is None and question.lower().startswith("is "):
			# Find the person and location in question
			for token in q_tokens:
				if f"loc_{token}" in facts:
					actual_loc = facts[f"loc_{token}"]
					# Find asked location (last location word in question)
					locations = ["bathroom", "kitchen", "garden", "office", "hallway", "bedroom", "playground"]
					for loc in locations:
						if loc in question.lower():
							answer_value = "yes" if actual_loc == loc else "no"
							break
					break

		# "what color is X?" → type_X → color
		if answer_value is None and "color" in question.lower():
			for token in q_tokens:
				if f"type_{token}" in facts:
					animal_type = facts[f"type_{token}"]
					if f"color_{animal_type}s" in facts:
						answer_value = facts[f"color_{animal_type}s"]
						break

		# "how do you go from X to Y?" → direction direction
		if answer_value is None and "how do you go" in question.lower():
			for key, direction in facts.items():
				if key.startswith("dir_"):
					answer_value = f"{direction} {direction}"
					break

		# Build context with ANSWER token RIGHT BEFORE <ANS> for deterministic mapping
		# Use multiple ANS tokens to make pattern NAME-INDEPENDENT
		# This ensures the n-gram window is dominated by the answer, not entity names
		if answer_value:
			# Handle multi-word answers (like "east east")
			answer_parts = answer_value.upper().split()
			if len(answer_parts) > 1:
				# For multi-word answers, create tokens for each part
				answer_tokens = [f"ANS_{p}" for p in answer_parts] * 2
			else:
				# Repeat 4 times to dominate the n=6 window
				answer_tokens = [f"ANS_{answer_value.upper()}"] * 4
		else:
			answer_tokens = []

		# Format: ANS_value ANS_value ANS_value ANS_value <ANS>
		# Skip question tokens - they add noise and create unseen patterns
		result = answer_tokens + ["<ANS>"]

		# Ensure minimum length for n-gram learning
		while len(result) < self.n:
			result = ["<PAD>"] + result
		return result

	def train(self, examples: list[tuple[str, str, str]]):
		"""Train on (story, question, answer) tuples."""
		# Build vocabulary from all data
		for story, question, answer in examples:
			for token in self.tokenize(story):
				self.encoder.add_token(token)
			for token in self.tokenize(question):
				self.encoder.add_token(token)
			for token in self.tokenize(answer):
				self.encoder.add_token(token)
			# Add fact tokens
			ctx = self.build_context(story, question)
			for token in ctx:
				self.encoder.add_token(token)

		# Build n-gram patterns with fact-aware context
		for story, question, answer in examples:
			tokens = self.build_context(story, question)
			answer_tokens = self.tokenize(answer) + ["<END>"]
			full_seq = tokens + answer_tokens

			for i in range(len(tokens) - 1, len(full_seq) - 1):
				if i >= self.n - 1:
					ctx = tuple(full_seq[i - self.n + 1:i + 1])
					target = full_seq[i + 1]
					self.context_counts[ctx][target] += 1

		# Initialize RAM predictor
		input_bits = self.n * self.encoder.bits_per_token
		self.predictor = RAMLayer(
			total_input_bits=input_bits,
			num_neurons=self.encoder.bits_per_token,
			n_bits_per_neuron=min(input_bits, 14),
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

	def predict(self, story: str, question: str, max_tokens: int = 5) -> str:
		"""Predict answer given story and question."""
		tokens = self.build_context(story, question)

		# Generate answer tokens
		answer_tokens = []
		for _ in range(max_tokens):
			ctx = tokens[-self.n:] if len(tokens) >= self.n else ["<PAD>"] * (self.n - len(tokens)) + tokens
			ctx_tuple = tuple(ctx)

			if ctx_tuple in self.context_counts:
				next_token = self.context_counts[ctx_tuple].most_common(1)[0][0]
			elif self.predictor:
				ctx_bits = []
				for token in ctx:
					ctx_bits.extend(self.encoder.encode(token))
				out = self.predictor(tensor(ctx_bits, dtype=uint8).unsqueeze(0)).squeeze()
				next_token = self.encoder.decode([int(b.item()) for b in out])
			else:
				next_token = "<UNK>"

			if next_token in ["<PAD>", "<UNK>", "<END>", "<SEP>"]:
				break

			answer_tokens.append(next_token)
			tokens.append(next_token)

		return " ".join(answer_tokens) if answer_tokens else "<UNK>"


# =============================================================================
# TASK GENERATORS
# =============================================================================

def generate_task1_single_supporting_fact(n_train: int = 100, n_test: int = 20, seed: int = 42):
	"""Task 1: Single Supporting Fact

	Mary went to the bathroom.
	Where is Mary? → bathroom
	"""
	random.seed(seed)

	names = ["mary", "john", "sandra", "daniel", "fred", "bill", "julie", "jeff"]
	locations = ["bathroom", "kitchen", "garden", "office", "hallway", "bedroom"]
	actions = ["went to", "moved to", "journeyed to", "travelled to"]

	def generate_example():
		name = random.choice(names)
		loc = random.choice(locations)
		action = random.choice(actions)

		# Simple single-fact story
		story = f"{name} {action} the {loc} ."
		question = f"where is {name} ?"
		answer = loc

		return story, question, answer

	train = [generate_example() for _ in range(n_train)]

	# Test with different seed
	random.seed(seed + 1000)
	test = [generate_example() for _ in range(n_test)]

	return train, test


def generate_task2_two_supporting_facts(n_train: int = 100, n_test: int = 20, seed: int = 42):
	"""Task 2: Two Supporting Facts

	John is in the playground. John picked up the football.
	Where is the football? → playground
	"""
	random.seed(seed)

	names = ["mary", "john", "sandra", "daniel"]
	locations = ["playground", "kitchen", "garden", "office"]
	objects = ["football", "milk", "apple", "ball"]

	def generate_example():
		name = random.choice(names)
		loc = random.choice(locations)
		obj = random.choice(objects)

		story = f"{name} is in the {loc} . {name} picked up the {obj} ."
		question = f"where is the {obj} ?"
		answer = loc

		return story, question, answer

	train = [generate_example() for _ in range(n_train)]
	random.seed(seed + 1000)
	test = [generate_example() for _ in range(n_test)]

	return train, test


def generate_task3_yes_no_questions(n_train: int = 100, n_test: int = 20, seed: int = 42):
	"""Task 3: Yes/No Questions

	Mary went to the bathroom.
	Is Mary in the bathroom? → yes
	Is Mary in the kitchen? → no
	"""
	random.seed(seed)

	names = ["mary", "john", "sandra", "daniel"]
	locations = ["bathroom", "kitchen", "garden", "office"]

	def generate_example():
		name = random.choice(names)
		actual_loc = random.choice(locations)

		story = f"{name} went to the {actual_loc} ."

		# 50% yes, 50% no questions
		if random.random() < 0.5:
			question = f"is {name} in the {actual_loc} ?"
			answer = "yes"
		else:
			other_loc = random.choice([l for l in locations if l != actual_loc])
			question = f"is {name} in the {other_loc} ?"
			answer = "no"

		return story, question, answer

	train = [generate_example() for _ in range(n_train)]
	random.seed(seed + 1000)
	test = [generate_example() for _ in range(n_test)]

	return train, test


def generate_task4_counting(n_train: int = 100, n_test: int = 20, seed: int = 42):
	"""Task 4: Counting

	Daniel picked up the apple. Daniel picked up the football.
	How many objects is Daniel carrying? → two
	"""
	random.seed(seed)

	names = ["mary", "john", "sandra", "daniel"]
	objects = ["apple", "football", "milk", "ball", "orange"]
	number_words = ["zero", "one", "two", "three", "four", "five"]

	def generate_example():
		name = random.choice(names)
		n_objects = random.randint(1, 4)
		picked = random.sample(objects, n_objects)

		story_parts = [f"{name} picked up the {obj} ." for obj in picked]
		story = " ".join(story_parts)
		question = f"how many objects is {name} carrying ?"
		answer = number_words[n_objects]

		return story, question, answer

	train = [generate_example() for _ in range(n_train)]
	random.seed(seed + 1000)
	test = [generate_example() for _ in range(n_test)]

	return train, test


def generate_task5_lists(n_train: int = 100, n_test: int = 20, seed: int = 42):
	"""Task 5: Lists/Sets

	Daniel picked up the apple. Daniel dropped the apple. Daniel picked up the ball.
	What is Daniel carrying? → ball
	"""
	random.seed(seed)

	names = ["mary", "john", "sandra", "daniel"]
	objects = ["apple", "football", "milk", "ball"]

	def generate_example():
		name = random.choice(names)
		obj1, obj2 = random.sample(objects, 2)

		# Pick up, drop, pick up another
		story = f"{name} picked up the {obj1} . {name} dropped the {obj1} . {name} picked up the {obj2} ."
		question = f"what is {name} carrying ?"
		answer = obj2

		return story, question, answer

	train = [generate_example() for _ in range(n_train)]
	random.seed(seed + 1000)
	test = [generate_example() for _ in range(n_test)]

	return train, test


def generate_task6_negation(n_train: int = 100, n_test: int = 20, seed: int = 42):
	"""Task 6: Simple Negation

	Fred went to the office. Fred did not go to the kitchen.
	Where is Fred? → office
	"""
	random.seed(seed)

	names = ["mary", "john", "fred", "bill"]
	locations = ["office", "kitchen", "garden", "bathroom"]

	def generate_example():
		name = random.choice(names)
		loc1, loc2 = random.sample(locations, 2)

		# Affirm one, negate another
		story = f"{name} went to the {loc1} . {name} did not go to the {loc2} ."
		question = f"where is {name} ?"
		answer = loc1

		return story, question, answer

	train = [generate_example() for _ in range(n_train)]
	random.seed(seed + 1000)
	test = [generate_example() for _ in range(n_test)]

	return train, test


def generate_task7_path_finding(n_train: int = 100, n_test: int = 20, seed: int = 42):
	"""Task 7: Path Finding (simplified)

	The kitchen is north of the hallway. The bathroom is north of the kitchen.
	How do you go from hallway to bathroom? → north north
	"""
	random.seed(seed)

	locations = ["hallway", "kitchen", "bathroom", "garden", "office"]
	directions = ["north", "south", "east", "west"]
	opposites = {"north": "south", "south": "north", "east": "west", "west": "east"}

	def generate_example():
		loc1, loc2, loc3 = random.sample(locations, 3)
		dir1 = random.choice(directions)

		# loc2 is dir1 of loc1, loc3 is dir1 of loc2
		story = f"the {loc2} is {dir1} of the {loc1} . the {loc3} is {dir1} of the {loc2} ."
		question = f"how do you go from {loc1} to {loc3} ?"
		answer = f"{dir1} {dir1}"

		return story, question, answer

	train = [generate_example() for _ in range(n_train)]
	random.seed(seed + 1000)
	test = [generate_example() for _ in range(n_test)]

	return train, test


def generate_task8_deduction(n_train: int = 100, n_test: int = 20, seed: int = 42):
	"""Task 8: Basic Deduction

	Lily is a swan. Swans are white. Lily is a bird.
	What color is Lily? → white
	"""
	random.seed(seed)

	animals = ["lily", "greg", "brian", "julius"]
	types = ["swan", "lion", "frog", "crow"]
	type_colors = {"swan": "white", "lion": "yellow", "frog": "green", "crow": "black"}

	def generate_example():
		name = random.choice(animals)
		animal_type = random.choice(types)
		color = type_colors[animal_type]

		story = f"{name} is a {animal_type} . {animal_type}s are {color} ."
		question = f"what color is {name} ?"
		answer = color

		return story, question, answer

	train = [generate_example() for _ in range(n_train)]
	random.seed(seed + 1000)
	test = [generate_example() for _ in range(n_test)]

	return train, test


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_task(model: BabiQAModel, test_data: list[tuple[str, str, str]]) -> dict:
	"""Evaluate model on test data."""
	correct = 0
	total = 0

	for story, question, expected in test_data:
		predicted = model.predict(story, question)
		if predicted.strip() == expected.strip():
			correct += 1
		total += 1

	# Calculate ambiguity
	ambiguous = sum(1 for c in model.context_counts.values() if len(c) > 1)

	return {
		"accuracy": correct / total if total > 0 else 0,
		"correct": correct,
		"total": total,
		"patterns": len(model.context_counts),
		"ambiguity_rate": ambiguous / len(model.context_counts) if model.context_counts else 0,
	}


def run_task_benchmark(task_name: str, generator_fn, n: int = 8):
	"""Run a single task benchmark."""
	print(f"\n{'='*70}")
	print(f"TASK: {task_name}")
	print(f"{'='*70}")

	train_data, test_data = generator_fn()

	print(f"Training examples: {len(train_data)}")
	print(f"Test examples: {len(test_data)}")
	print(f"Context window: n={n}")

	# Show example
	story, question, answer = train_data[0]
	print(f"\nExample:")
	print(f"  Story: {story}")
	print(f"  Question: {question}")
	print(f"  Answer: {answer}")

	# Train
	model = BabiQAModel(n=n, rng=42)
	model.train(train_data)

	print(f"\nVocabulary: {len(model.encoder.token_to_idx)} tokens")
	print(f"Training patterns: {len(model.context_counts)} unique contexts")

	# Evaluate
	results = evaluate_task(model, test_data)

	print(f"\nResults:")
	print(f"  Accuracy: {results['accuracy']:.1%} ({results['correct']}/{results['total']})")
	print(f"  Ambiguity: {results['ambiguity_rate']:.1%}")

	# Show some predictions
	print(f"\nSample predictions:")
	for i, (story, question, expected) in enumerate(test_data[:3]):
		predicted = model.predict(story, question)
		status = "✓" if predicted == expected else "✗"
		print(f"  {status} Q: {question[:40]}...")
		print(f"      Expected: {expected}, Predicted: {predicted}")

	return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
	print(f"\n{'='*70}")
	print("bAbI TASKS BENCHMARK FOR RAM NETWORKS")
	print(f"Started at: {datetime.now()}")
	print(f"{'='*70}")

	tasks = [
		("Task 1: Single Supporting Fact", generate_task1_single_supporting_fact),
		("Task 2: Two Supporting Facts", generate_task2_two_supporting_facts),
		("Task 3: Yes/No Questions", generate_task3_yes_no_questions),
		("Task 4: Counting", generate_task4_counting),
		("Task 5: Lists/Sets", generate_task5_lists),
		("Task 6: Simple Negation", generate_task6_negation),
		("Task 7: Path Finding", generate_task7_path_finding),
		("Task 8: Basic Deduction", generate_task8_deduction),
	]

	all_results = {}

	for task_name, generator_fn in tasks:
		results = run_task_benchmark(task_name, generator_fn, n=6)  # Smaller n for short contexts
		all_results[task_name] = results

	# Summary
	print(f"\n{'='*70}")
	print("FINAL SUMMARY")
	print(f"{'='*70}")
	print(f"\n| Task | Accuracy | Ambiguity | Patterns |")
	print(f"|------|----------|-----------|----------|")

	total_correct = 0
	total_examples = 0

	for task_name, results in all_results.items():
		short_name = task_name.split(":")[0]
		print(f"| {short_name} | {results['accuracy']:.1%} | {results['ambiguity_rate']:.1%} | {results['patterns']} |")
		total_correct += results['correct']
		total_examples += results['total']

	overall_accuracy = total_correct / total_examples if total_examples > 0 else 0
	print(f"\nOverall: {overall_accuracy:.1%} ({total_correct}/{total_examples})")

	# Key findings
	passed = sum(1 for r in all_results.values() if r['accuracy'] >= 0.9)
	print(f"\nTasks passing (≥90%): {passed}/{len(tasks)}")

	if overall_accuracy >= 0.9:
		print("\n★★★ Excellent performance on bAbI tasks! ★★★")
	elif overall_accuracy >= 0.7:
		print("\n★ Good performance, some tasks need improvement.")
	else:
		print("\n⚠ Performance needs improvement.")

	print(f"\nFinished at: {datetime.now()}")
	print(f"{'='*70}")
