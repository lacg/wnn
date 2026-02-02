// Iteration model - a generation within an experiment

import Foundation

public struct Iteration: Codable, Identifiable {
    public let id: Int64
    public let experiment_id: Int64?
    public let iteration_num: Int32
    public let best_ce: Double
    public let best_accuracy: Double?
    public let avg_ce: Double?
    public let avg_accuracy: Double?
    public let elite_count: Int32?
    public let offspring_count: Int32?
    public let offspring_viable: Int32?
    public let fitness_threshold: Double?
    public let elapsed_secs: Double?
    public let baseline_ce: Double?
    public let delta_baseline: Double?
    public let delta_previous: Double?
    public let patience_counter: Int32?
    public let patience_max: Int32?
    public let candidates_total: Int32?
    public let created_at: String

    public var createdDate: Date? { ISO8601DateFormatter().date(from: created_at) }
    public var ceDelta: Double? { delta_previous }
    public var accuracyPercent: Double? { best_accuracy.map { $0 * 100 } }

    public var patienceStatus: String? {
        guard let counter = patience_counter, let max = patience_max else { return nil }
        return "\(counter)/\(max)"
    }
}
