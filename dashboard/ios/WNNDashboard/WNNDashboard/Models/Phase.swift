// Phase model - a stage within an experiment (GA-Neurons, TS-Bits, etc.)

import Foundation

struct Phase: Codable, Identifiable {
    let id: Int64
    let experiment_id: Int64
    let name: String
    let phase_type: String
    let sequence_order: Int32
    let status: PhaseStatus
    let max_iterations: Int32
    let population_size: Int32?
    let current_iteration: Int32
    let best_ce: Double?
    let best_accuracy: Double?
    let created_at: String
    let started_at: String?
    let ended_at: String?

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

    var progress: Double {
        guard max_iterations > 0 else { return 0 }
        return Double(current_iteration) / Double(max_iterations)
    }

    /// Short display name for the phase
    var shortName: String {
        // Extract the main part (e.g., "1a" from "Phase 1a - GA Neurons")
        if let match = name.range(of: #"\d+[a-z]?"#, options: .regularExpression) {
            return String(name[match])
        }
        return name
    }

    /// Phase type display (e.g., "GA" or "TS")
    var typePrefix: String {
        if phase_type.lowercased().contains("ga") {
            return "GA"
        } else if phase_type.lowercased().contains("ts") {
            return "TS"
        }
        return ""
    }
}
