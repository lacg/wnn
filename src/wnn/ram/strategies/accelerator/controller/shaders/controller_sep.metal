//! controller_sep.metal — ISOLATED bit-exact Pearson kernel for the P3 Type-2
//! bidirectional accumulator search (detect_accumulator_bidir). Kept in its OWN
//! file, compiled as a SEPARATE Metal library with fast-math DISABLED + FP
//! contraction OFF, so its div/sqrt are IEEE and `sxy += dx*dy` is NOT fused into
//! an FMA — both required to bit-match Rust's f32 two-pass pearson. This isolation
//! is deliberate: applying contract(off)/fast-math-off to the physics shader
//! (controller_rollout.metal) could perturb the forward-roll parity that already
//! passes, so the strict-FP pragma lives ONLY here.
//!
//! The heavy O(bits^2 * instances) pair-correlation runs on the GPU (one thread =
//! one (conflict, up-bit, dn-bit) pair); the host argmaxes the small corr table.

#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

struct BidirParams
{
	uint num_conflicts;
	uint num_bits; // candidate_bits length B
};

// One thread = one (conflict c, up-bit ai, dn-bit bi). Computes the signed Pearson
// correlation of the NET window count (count_up - count_dn) vs the disagreeing-motor
// PWM scalar, replicating controller_split::pearson EXACTLY (two-pass: means, then
// centered products, summed sequentially in instance order). ai==bi → 0 (skipped by
// the host, which mirrors the CPU `ai==bi continue`).
kernel void controller_sep_bidir(
		device const uint *conf_inst_count [[buffer(0)]], // [C]
		device const uint *conf_inst_base [[buffer(1)]],	// [C] scalar base (aligns w/ instances)
		device const uint *block_base [[buffer(2)]],			// [C] counts block base
		device const float *counts [[buffer(3)]],					// [sum B*n] from controller_sep_counts
		device const float *scalar [[buffer(4)]],					// [total_inst] per-instance PWM
		device float *out_corr [[buffer(5)]],							// [C*B*B] OUT
		constant BidirParams &P [[buffer(6)]],
		uint3 tid [[thread_position_in_grid]])
{
	uint c = tid.x, ai = tid.y, bi = tid.z;
	if (c >= P.num_conflicts || ai >= P.num_bits || bi >= P.num_bits)
		return;
	uint oidx = (c * P.num_bits + ai) * P.num_bits + bi;
	if (ai == bi)
	{
		out_corr[oidx] = 0.0f;
		return;
	}

	uint n = conf_inst_count[c];
	if (n == 0u)
	{
		out_corr[oidx] = 0.0f;
		return;
	}
	uint bb = block_base[c];
	device const float *ca = counts + bb + ai * n; // count_up per instance
	device const float *cb = counts + bb + bi * n; // count_dn per instance
	device const float *y = scalar + conf_inst_base[c];

	// pass 1: means (sequential sum, matches Rust .sum() left-to-right order).
	float sx = 0.0f, sy = 0.0f;
	for (uint k = 0u; k < n; k++)
	{
		sx += (ca[k] - cb[k]);
		sy += y[k];
	}
	float nf = float(n);
	float mx = sx / nf;
	float my = sy / nf;

	// pass 2: centered products.
	float sxy = 0.0f, sxx = 0.0f, syy = 0.0f;
	for (uint k = 0u; k < n; k++)
	{
		float dx = (ca[k] - cb[k]) - mx;
		float dy = y[k] - my;
		sxy += dx * dy;
		sxx += dx * dx;
		syy += dy * dy;
	}

	float corr;
	if (sxx <= 1e-9f || syy <= 1e-9f)
		corr = 0.0f;
	else
		corr = sxy / (sqrt(sxx) * sqrt(syy));
	out_corr[oidx] = corr;
}
