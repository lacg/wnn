// Experiment model - a single optimization run

import Foundation

struct Experiment: Codable, Identifiable {
    let id: Int64
    let flow_id: Int64?
    let sequence_order: Int32?
    let name: String
    let status: ExperimentStatus
    let fitness_calculator: FitnessCalculator
    let fitness_weight_ce: Double
    let fitness_weight_acc: Double
    let tier_config: String?
    let context_size: Int32
    let population_size: Int32
    let pid: Int32?
    let last_phase_id: Int64?
    let last_iteration: Int32?
    let resume_checkpoint_id: Int64?
    let created_at: String
    let started_at: String?
    let ended_at: String?
    let paused_at: String?

    var createdDate: Date? {
        ISO8601DateFormatter().date(from: created_at)
    }

    var startedDate: Date? {
        started_at.flatMap { ISO8601DateFormatter().date(from: $0) }
    }

    var endedDate: Date? {
        ended_at.flatMap { ISO8601DateFormatter().date(from: $0) }
    }

    var duration: TimeInterval? {
        guard let start = startedDate else { return nil }
        let end = endedDate ?? Date()
        return end.timeIntervalSince(start)
    }

    /// Parse tier config string into structured data
    /// Format: "100,15,20;400,10,12;rest,5,8"
    var parsedTierConfig: [TierConfigEntry]? {
        guard let config = tier_config else { return nil }
        return config.split(separator: ";").compactMap { tier in
            let parts = tier.split(separator: ",")
            guard parts.count == 3 else { return nil }
            let clusters = String(parts[0])
            guard let neurons = Int(parts[1]),
                  let bits = Int(parts[2]) else { return nil }
            return TierConfigEntry(clusters: clusters, neurons: neurons, bits: bits)
        }
    }
}

struct TierConfigEntry {
    let clusters: String  // Can be number or "rest"
    let neurons: Int
    let bits: Int

    var clusterCount: Int? {
        Int(clusters)
    }

    var isRest: Bool {
        clusters.lowercased() == "rest"
    }
}
