//
// Metal Compute Shader for CE/Accuracy Reduction
//
// Given pre-computed scores for all (example, cluster) pairs, computes:
// - Cross-entropy loss using softmax
// - Accuracy (predicted == target)
//
// This is the final step after multi-group evaluation - all scores are
// already in GPU memory, we just need to reduce to CE/accuracy.
//

#include <metal_stdlib>
using namespace metal;

struct CEReduceParams {
    uint num_examples;
    uint num_clusters;
};

//
// CE Reduction Kernel
//
// Each thread processes ONE example:
// 1. Finds max score (for numerical stability) and predicted cluster
// 2. Computes softmax: exp(score - max) / sum(exp(score - max))
// 3. Computes CE: -log(softmax_prob_target)
// 4. Checks if prediction is correct (argmax of raw scores)
//
// Grid: (num_examples)
// Input: scores[num_examples * num_clusters] - pre-computed scores (probabilities 0-1)
// Output: ce_out[num_examples], correct_out[num_examples]
//
// NOTE: The CPU path applies exp() to scores before normalizing, even though
// scores are probabilities (0-1). We match this behavior for consistency.
//
kernel void reduce_scores_to_ce(
    device const float* scores [[buffer(0)]],       // [num_examples * num_clusters]
    device const int* targets [[buffer(1)]],        // [num_examples]
    constant CEReduceParams& params [[buffer(2)]],
    device float* ce_out [[buffer(3)]],             // [num_examples]
    device uint* correct_out [[buffer(4)]],         // [num_examples]
    uint example_idx [[thread_position_in_grid]]
) {
    if (example_idx >= params.num_examples) return;

    int target_cluster = targets[example_idx];
    device const float* ex_scores = scores + example_idx * params.num_clusters;

    // Single pass: find max, predicted, and compute sum_exp simultaneously
    // NOTE: Use >= to match Rust's max_by behavior (returns LAST maximum on ties)
    float max_score = ex_scores[0];
    uint predicted_cluster = 0;
    float sum_exp = 1.0f;  // exp(0) for first element

    for (uint c = 1; c < params.num_clusters; c++) {
        float score = ex_scores[c];
        if (score >= max_score) {  // >= to pick LAST max on ties (matches Rust max_by)
            // Rescale previous sum_exp
            sum_exp *= exp(max_score - score);
            max_score = score;
            predicted_cluster = c;
        }
        sum_exp += exp(score - max_score);
    }

    // Compute CE: -log(exp(target_score - max) / sum_exp)
    float target_score = ex_scores[target_cluster];
    float target_exp = exp(target_score - max_score);
    float target_prob = target_exp / sum_exp;
    float ce = -log(target_prob + 1e-10f);

    ce_out[example_idx] = ce;
    correct_out[example_idx] = (predicted_cluster == uint(target_cluster)) ? 1 : 0;
}

//
// Batch scatter kernel
// Copies scores from a group's output to the correct positions in the full scores buffer
//
// This allows groups to write to separate buffers, then combine at the end.
//
kernel void scatter_group_scores(
    device const float* group_scores [[buffer(0)]],  // [num_examples * num_group_clusters]
    device float* all_scores [[buffer(1)]],          // [num_examples * num_total_clusters]
    device const uint* cluster_ids [[buffer(2)]],    // [num_group_clusters] - mapping to global cluster IDs
    constant uint& num_examples [[buffer(3)]],
    constant uint& num_group_clusters [[buffer(4)]],
    constant uint& num_total_clusters [[buffer(5)]],
    uint2 thread_pos [[thread_position_in_grid]]     // (group_cluster_idx, example_idx)
) {
    uint group_cluster_idx = thread_pos.x;
    uint example_idx = thread_pos.y;

    if (group_cluster_idx >= num_group_clusters) return;
    if (example_idx >= num_examples) return;

    uint global_cluster_id = cluster_ids[group_cluster_idx];
    uint src_idx = example_idx * num_group_clusters + group_cluster_idx;
    uint dst_idx = example_idx * num_total_clusters + global_cluster_id;

    all_scores[dst_idx] = group_scores[src_idx];
}
