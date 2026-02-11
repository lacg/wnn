// Experiment model - a single optimization run

import Foundation

public struct Experiment: Codable, Identifiable, Hashable {
    public let id: Int64
    public let flow_id: Int64?
    public let sequence_order: Int32?
    public let name: String
    public let status: ExperimentStatus
    public let fitness_calculator: FitnessCalculator
    public let fitness_weight_ce: Double
    public let fitness_weight_acc: Double
    public let tier_config: String?
    public let context_size: Int32
    public let population_size: Int32
    public let pid: Int32?
    public let last_iteration: Int32?
    public let resume_checkpoint_id: Int64?
    public let created_at: String
    public let started_at: String?
    public let ended_at: String?
    public let paused_at: String?
    // Fields moved from Phase (simplified model)
    public let phase_type: String?
    public let max_iterations: Int32?
    public let current_iteration: Int32?
    public let best_ce: Double?
    public let best_accuracy: Double?
    public let cluster_type: String?

    public var createdDate: Date? { DateFormatters.parse( created_at) }
    public var startedDate: Date? { started_at.flatMap { DateFormatters.parse( $0) } }
    public var endedDate: Date? { ended_at.flatMap { DateFormatters.parse( $0) } }

    public var duration: TimeInterval? {
        guard let start = startedDate else { return nil }
        return (endedDate ?? Date()).timeIntervalSince(start)
    }

    public var progress: Double {
        guard let max = max_iterations, max > 0, let current = current_iteration else { return 0 }
        return Double(current) / Double(max)
    }

    public var typePrefix: String {
        guard let pt = phase_type?.lowercased() else { return "" }
        if pt.contains("ga") { return "GA" }
        if pt.contains("ts") { return "TS" }
        return ""
    }

    /// Format "ts_bits" → "TS Bits", "ga_neurons" → "GA Neurons"
    public var formattedPhaseType: String? {
        guard let pt = phase_type else { return nil }
        let parts = pt.split(separator: "_")
        return parts.enumerated().map { i, part in
            let s = String(part)
            if i == 0 && (s == "ga" || s == "ts") { return s.uppercased() }
            return s.capitalized
        }.joined(separator: " ")
    }

    public var parsedTierConfig: [TierConfigEntry]? {
        guard let config = tier_config else { return nil }
        return config.split(separator: ";").compactMap { tier in
            let parts = tier.split(separator: ",")
            guard parts.count == 3,
                  let neurons = Int(parts[1]),
                  let bits = Int(parts[2]) else { return nil }
            return TierConfigEntry(clusters: String(parts[0]), neurons: neurons, bits: bits)
        }
    }
}

public struct TierConfigEntry {
    public let clusters: String
    public let neurons: Int
    public let bits: Int

    public var clusterCount: Int? { Int(clusters) }
    public var isRest: Bool { clusters.lowercased() == "rest" }
}
