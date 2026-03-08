"""Compute Kneser-Ney smoothed n-gram baselines on WikiText-2 (GPT-2 tokenizer)."""
import sys, math, time
from collections import defaultdict

sys.path.insert(0, "src/wnn")
import tiktoken
from datasets import load_dataset

print("Loading data...")
enc = tiktoken.get_encoding("gpt2")
ds = load_dataset("wikitext", "wikitext-2-raw-v1")

train_text = "\n\n".join([t for t in ds["train"]["text"] if t.strip()])
test_text = "\n\n".join([t for t in ds["test"]["text"] if t.strip()])
train_tokens = enc.encode(train_text)
test_tokens = enc.encode(test_text)
V = 50257

print(f"Train: {len(train_tokens):,} tokens, Test: {len(test_tokens):,} tokens")
print(f"Vocab: {V:,}")
print()


# ── Kneser-Ney Bigram ───────────────────────────────────────────
# Interpolated KN: P_KN(w|w') = max(c(w',w) - d, 0) / c(w') + λ(w') × P_cont(w)
# where P_cont(w) = |{w': c(w',w) > 0}| / |{(w',w): c(w',w) > 0}|
# and λ(w') = d × N1+(w', •) / c(w')
# d = n1 / (n1 + 2*n2) where n1,n2 = count of bigrams appearing 1,2 times

def build_kn_bigram(tokens, vocab_size):
	"""Build interpolated Kneser-Ney smoothed bigram model."""
	t0 = time.time()

	# Count bigrams and unigrams
	bigram_counts = defaultdict(lambda: defaultdict(int))
	context_counts = defaultdict(int)
	for i in range(1, len(tokens)):
		prev, cur = tokens[i-1], tokens[i]
		bigram_counts[prev][cur] += 1
		context_counts[prev] += 1

	# Compute discount d = n1 / (n1 + 2*n2)
	n1 = 0  # bigrams appearing exactly once
	n2 = 0  # bigrams appearing exactly twice
	for prev in bigram_counts:
		for cur in bigram_counts[prev]:
			c = bigram_counts[prev][cur]
			if c == 1: n1 += 1
			elif c == 2: n2 += 1
	d = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) > 0 else 0.75
	print(f"  KN discount d = {d:.4f} (n1={n1:,}, n2={n2:,})")

	# Continuation counts: for each word w, how many unique contexts precede it
	# P_cont(w) = |{w': c(w',w) > 0}| / total_unique_bigrams
	continuation_count = defaultdict(int)  # how many unique prev tokens for each cur
	total_unique_bigrams = 0
	for prev in bigram_counts:
		for cur in bigram_counts[prev]:
			continuation_count[cur] += 1
			total_unique_bigrams += 1

	# N1+(prev, •) = number of unique continuations after prev
	n1_plus = {}
	for prev in bigram_counts:
		n1_plus[prev] = len(bigram_counts[prev])

	elapsed = time.time() - t0
	print(f"  Built KN bigram in {elapsed:.1f}s (unique bigrams: {total_unique_bigrams:,})")

	def prob(cur, prev):
		"""P_KN(cur | prev)"""
		c_prev = context_counts.get(prev, 0)
		c_bigram = bigram_counts[prev].get(cur, 0) if prev in bigram_counts else 0

		# Continuation probability (unigram backoff)
		p_cont = continuation_count.get(cur, 0) / total_unique_bigrams if total_unique_bigrams > 0 else 1.0 / vocab_size

		if c_prev == 0:
			# Unseen context: fall back to continuation prob
			return p_cont

		# Interpolated KN
		first_term = max(c_bigram - d, 0) / c_prev
		lam = d * n1_plus.get(prev, 0) / c_prev
		return first_term + lam * p_cont

	return prob


def build_kn_trigram(tokens, vocab_size, bigram_prob_fn):
	"""Build interpolated KN trigram that backs off to KN bigram."""
	t0 = time.time()

	trigram_counts = defaultdict(lambda: defaultdict(int))
	ctx2_counts = defaultdict(int)
	for i in range(2, len(tokens)):
		ctx = (tokens[i-2], tokens[i-1])
		cur = tokens[i]
		trigram_counts[ctx][cur] += 1
		ctx2_counts[ctx] += 1

	# Discount
	n1 = n2 = 0
	for ctx in trigram_counts:
		for cur in trigram_counts[ctx]:
			c = trigram_counts[ctx][cur]
			if c == 1: n1 += 1
			elif c == 2: n2 += 1
	d = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) > 0 else 0.75
	print(f"  KN trigram discount d = {d:.4f}")

	# N1+(ctx, •)
	n1_plus_tri = {}
	for ctx in trigram_counts:
		n1_plus_tri[ctx] = len(trigram_counts[ctx])

	# Continuation counts for trigram → how many unique (w'') precede bigram (w', w)
	# For simplicity, use the bigram continuation count as backoff
	bi_continuation = defaultdict(int)
	total_tri_bigrams = 0
	for ctx in trigram_counts:
		for cur in trigram_counts[ctx]:
			bi_continuation[(ctx[1], cur)] += 1
			total_tri_bigrams += 1

	elapsed = time.time() - t0
	print(f"  Built KN trigram in {elapsed:.1f}s")

	def prob(cur, ctx):
		"""P_KN(cur | ctx=(w_{i-2}, w_{i-1}))"""
		c_ctx = ctx2_counts.get(ctx, 0)
		c_tri = trigram_counts[ctx].get(cur, 0) if ctx in trigram_counts else 0

		# Backoff to KN bigram
		p_backoff = bigram_prob_fn(cur, ctx[1])

		if c_ctx == 0:
			return p_backoff

		first_term = max(c_tri - d, 0) / c_ctx
		lam = d * n1_plus_tri.get(ctx, 0) / c_ctx
		return first_term + lam * p_backoff

	return prob


# ── Build models ─────────────────────────────────────────────────
print("Building KN bigram...")
kn_bigram = build_kn_bigram(train_tokens, V)

print("Building KN trigram...")
kn_trigram = build_kn_trigram(train_tokens, V, kn_bigram)

# ── Evaluate ─────────────────────────────────────────────────────
print("\nEvaluating on test set...")

# Unigram (add-1 for reference)
uni_counts = defaultdict(int)
for t in train_tokens:
	uni_counts[t] += 1
total_uni = len(train_tokens) + V
uni_default = 1.0 / total_uni
uni_argmax = max(uni_counts, key=uni_counts.get)

uni_ce = 0.0
uni_correct = 0
for t in test_tokens:
	p = (uni_counts.get(t, 0) + 1) / total_uni
	uni_ce -= math.log(max(p, 1e-20))
	if t == uni_argmax: uni_correct += 1
uni_ce /= len(test_tokens)
uni_acc = uni_correct / len(test_tokens)
print(f"  Unigram (add-1):    CE={uni_ce:.4f}  PPL={math.exp(uni_ce):.1f}  Acc={uni_acc:.2%}")

# KN Bigram
print("  Evaluating KN bigram...")
t0 = time.time()
bi_ce = 0.0
bi_correct = 0
# Precompute bigram argmax per context
bi_argmax_cache = {}
for i in range(1, len(test_tokens)):
	prev = test_tokens[i-1]
	cur = test_tokens[i]
	p = kn_bigram(cur, prev)
	bi_ce -= math.log(max(p, 1e-20))

	# For accuracy: check if this is the argmax
	# (expensive to compute full argmax, so we track the best seen token per context)
	if prev not in bi_argmax_cache:
		# Find argmax for this context by checking all tokens that appeared after prev in train
		from collections import Counter
		best_tok = -1
		best_p = -1
		# Check tokens seen after prev + unigram top tokens
		candidates = set()
		if prev in defaultdict.__new__(defaultdict):
			pass
		# Simpler: just check if cur matches the most frequent continuation
		pass

	# Skip full argmax computation for speed - just check top continuation from train
bi_ce /= (len(test_tokens) - 1)
# For accuracy, compute per-context argmax from training bigram counts
print(f"    Computing accuracy...")
bi_counts_for_acc = defaultdict(lambda: defaultdict(int))
for i in range(1, len(train_tokens)):
	bi_counts_for_acc[train_tokens[i-1]][train_tokens[i]] += 1
bi_argmax = {}
for prev in bi_counts_for_acc:
	bi_argmax[prev] = max(bi_counts_for_acc[prev], key=bi_counts_for_acc[prev].get)
bi_correct = sum(1 for i in range(1, len(test_tokens)) if test_tokens[i] == bi_argmax.get(test_tokens[i-1], -1))
bi_acc = bi_correct / (len(test_tokens) - 1)
print(f"  KN Bigram:          CE={bi_ce:.4f}  PPL={math.exp(bi_ce):.1f}  Acc={bi_acc:.2%}  ({time.time()-t0:.1f}s)")

# KN Trigram
print("  Evaluating KN trigram...")
t0 = time.time()
tri_ce = 0.0
for i in range(2, len(test_tokens)):
	ctx = (test_tokens[i-2], test_tokens[i-1])
	cur = test_tokens[i]
	p = kn_trigram(cur, ctx)
	tri_ce -= math.log(max(p, 1e-20))
tri_ce /= (len(test_tokens) - 2)
# Accuracy from train trigram counts
tri_counts_for_acc = defaultdict(lambda: defaultdict(int))
for i in range(2, len(train_tokens)):
	ctx = (train_tokens[i-2], train_tokens[i-1])
	tri_counts_for_acc[ctx][train_tokens[i]] += 1
tri_argmax = {}
for ctx in tri_counts_for_acc:
	tri_argmax[ctx] = max(tri_counts_for_acc[ctx], key=tri_counts_for_acc[ctx].get)
tri_correct = sum(1 for i in range(2, len(test_tokens)) if test_tokens[i] == tri_argmax.get((test_tokens[i-2], test_tokens[i-1]), -1))
tri_acc = tri_correct / (len(test_tokens) - 2)
print(f"  KN Trigram:         CE={tri_ce:.4f}  PPL={math.exp(tri_ce):.1f}  Acc={tri_acc:.2%}  ({time.time()-t0:.1f}s)")


# ── Summary ──────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"  {'Model':<30} {'CE':>8} {'PPL':>10} {'Acc':>8}")
print(f"  {'─'*30} {'─'*8} {'─'*10} {'─'*8}")
print(f"  {'Unigram (add-1)':<30} {uni_ce:>8.4f} {math.exp(uni_ce):>10.1f} {uni_acc:>7.2%}")
print(f"  {'KN Bigram':<30} {bi_ce:>8.4f} {math.exp(bi_ce):>10.1f} {bi_acc:>7.2%}")
print(f"  {'KN Trigram':<30} {tri_ce:>8.4f} {math.exp(tri_ce):>10.1f} {tri_acc:>7.2%}")
print(f"  {'─'*30} {'─'*8} {'─'*10} {'─'*8}")
print(f"  {'WNN K512 selector (λ=0)':<30} {'8.52':>8} {'5023':>10} {'10.28%':>8}")
print(f"  {'WNN + unigram (λ=0.8)':<30} {'7.35':>8} {'1566':>10} {'10.79%':>8}")
print(f"  {'GPT-2 small (124M)':<30} {'~3.5':>8} {'~29':>10} {'~37%':>8}")
print("=" * 70)
