// Iteration model - a generation/iteration within a phase

import Foundation

struct Iteration: Codable, Identifiable {
    let id: Int64
    let phase_id: Int64
    let iteration_num: Int32
    let best_ce: Double
    let best_accuracy: Double?
    let avg_ce: Double?
    let avg_accuracy: Double?
    let elite_count: Int32?
    let offspring_count: Int32?
    let offspring_viable: Int32?
    let fitness_threshold: Double?
    let elapsed_secs: Double?
    let baseline_ce: Double?
    let delta_baseline: Double?
    let delta_previous: Double?
    let patience_counter: Int32?
    let patience_max: Int32?
    let candidates_total: Int32?
    let created_at: String

    var createdDate: Date? {
        ISO8601DateFormatter().date(from: created_at)
    }

    /// Delta from previous iteration (negative is improvement for CE)
    var ceDelta: Double? {
        delta_previous
    }

    /// Format accuracy as percentage
    var accuracyPercent: Double? {
        best_accuracy.map { $0 * 100 }
    }

    /// Patience status string
    var patienceStatus: String? {
        guard let counter = patience_counter, let max = patience_max else { return nil }
        return "\(counter)/\(max)"
    }
}
