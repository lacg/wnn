// Phase model - a stage within an experiment

import Foundation

/// Validation result at end of a phase
public struct PhaseResult: Codable, Identifiable {
    public let id: Int64
    public let phase_id: Int64
    public let metric_type: String  // "best_ce", "best_acc", "top_k_mean"
    public let ce: Double
    public let accuracy: Double
    public let memory_bytes: Int64?
    public let improvement_pct: Double
}

public struct Phase: Codable, Identifiable {
    public let id: Int64
    public let experiment_id: Int64
    public let name: String
    public let phase_type: String
    public let sequence_order: Int32
    public let status: PhaseStatus
    public let max_iterations: Int32
    public let population_size: Int32?
    public let current_iteration: Int32
    public let best_ce: Double?
    public let best_accuracy: Double?
    public let created_at: String
    public let started_at: String?
    public let ended_at: String?
    public let results: [PhaseResult]?

    public var createdDate: Date? { ISO8601DateFormatter().date(from: created_at) }
    public var startedDate: Date? { started_at.flatMap { ISO8601DateFormatter().date(from: $0) } }
    public var endedDate: Date? { ended_at.flatMap { ISO8601DateFormatter().date(from: $0) } }

    public var duration: TimeInterval? {
        guard let start = startedDate else { return nil }
        return (endedDate ?? Date()).timeIntervalSince(start)
    }

    public var progress: Double {
        guard max_iterations > 0 else { return 0 }
        return Double(current_iteration) / Double(max_iterations)
    }

    public var shortName: String {
        if let match = name.range(of: #"\d+[a-z]?"#, options: .regularExpression) {
            return String(name[match])
        }
        return name
    }

    public var typePrefix: String {
        if phase_type.lowercased().contains("ga") { return "GA" }
        if phase_type.lowercased().contains("ts") { return "TS" }
        return ""
    }
}
