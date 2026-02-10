//
// Metal Compute Shader for Bitwise RAMLM Reconstruction + Cross-Entropy
//
// Given per-bit probabilities P(bit_i=1) from cluster scoring,
// reconstructs token log-probabilities and computes cross-entropy.
//
// Each thread processes one (genome, eval_example) pair.
//
// Reconstruction formula (optimized):
//   diff[b] = log(p_b) - log(1 - p_b)
//   base    = sum_b(log(1 - p_b))
//   log_prob[t] = base + sum_b(token_bits[t,b] * diff[b])
//
// CE via online log-sum-exp:
//   CE = log_sum_exp(log_probs) - log_prob[target]
//

#include <metal_stdlib>
using namespace metal;

struct BitwiseCEParams {
	uint num_examples;
	uint num_bits;
	uint vocab_size;
	uint num_genomes;
};

kernel void bitwise_reconstruct_ce(
	device const float* bit_scores   [[buffer(0)]],  // [num_genomes * num_examples * num_bits]
	device const uchar* token_bits   [[buffer(1)]],  // [vocab_size * num_bits]
	device const uint*  targets      [[buffer(2)]],  // [num_examples] (shared across genomes)
	constant BitwiseCEParams& params [[buffer(3)]],
	device float* ce_out             [[buffer(4)]],  // [num_genomes * num_examples]
	device uint*  correct_out        [[buffer(5)]],  // [num_genomes * num_examples]
	uint thread_idx [[thread_position_in_grid]]
) {
	uint total = params.num_genomes * params.num_examples;
	if (thread_idx >= total) return;

	uint genome_idx = thread_idx / params.num_examples;
	uint ex_idx = thread_idx % params.num_examples;
	uint target = targets[ex_idx];

	uint score_base = (genome_idx * params.num_examples + ex_idx) * params.num_bits;

	// Precompute diff[b] and base from per-bit scores
	float base = 0.0f;
	float diff[32];  // max 32 bits
	float eps = 1e-7f;

	for (uint b = 0; b < params.num_bits; b++) {
		float p = clamp(bit_scores[score_base + b], eps, 1.0f - eps);
		float lp1 = log(p);
		float lp0 = log(1.0f - p);
		diff[b] = lp1 - lp0;
		base += lp0;
	}

	// Iterate over vocab: online log-sum-exp + track target and argmax
	float max_lp = -INFINITY;
	float sum_exp = 0.0f;
	float target_lp = 0.0f;
	uint predicted = 0;
	float predicted_lp = -INFINITY;

	for (uint t = 0; t < params.vocab_size; t++) {
		uint tb_base = t * params.num_bits;
		float lp = base;
		for (uint b = 0; b < params.num_bits; b++) {
			lp += float(token_bits[tb_base + b]) * diff[b];
		}

		if (t == target) target_lp = lp;
		if (lp > predicted_lp) { predicted_lp = lp; predicted = t; }

		// Online log-sum-exp (numerically stable)
		if (lp > max_lp) {
			sum_exp = sum_exp * exp(max_lp - lp) + 1.0f;
			max_lp = lp;
		} else {
			sum_exp += exp(lp - max_lp);
		}
	}

	uint out_idx = genome_idx * params.num_examples + ex_idx;
	ce_out[out_idx] = max_lp + log(sum_exp + 1e-10f) - target_lp;
	correct_out[out_idx] = (predicted == target) ? 1 : 0;
}
