// Genome and GenomeEvaluation models

import Foundation

struct Genome: Codable, Identifiable {
    let id: Int64
    let experiment_id: Int64
    let config_hash: String
    let tiers_json: String
    let total_clusters: Int32
    let total_neurons: Int32
    let total_memory_bytes: Int64
    let created_at: String

    var createdDate: Date? {
        ISO8601DateFormatter().date(from: created_at)
    }

    /// Parse tiers_json into structured tier configs
    var tiers: [TierConfig]? {
        guard let data = tiers_json.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode([TierConfig].self, from: data)
    }

    /// Memory size formatted as human-readable string
    var formattedMemorySize: String {
        let bytes = total_memory_bytes
        if bytes < 1024 {
            return "\(bytes) B"
        } else if bytes < 1024 * 1024 {
            return String(format: "%.1f KB", Double(bytes) / 1024)
        } else if bytes < 1024 * 1024 * 1024 {
            return String(format: "%.1f MB", Double(bytes) / (1024 * 1024))
        } else {
            return String(format: "%.2f GB", Double(bytes) / (1024 * 1024 * 1024))
        }
    }
}

struct TierConfig: Codable {
    let tier: Int32
    let clusters: Int32
    let neurons: Int32
    let bits: Int32
}

struct GenomeEvaluation: Codable, Identifiable {
    let id: Int64
    let iteration_id: Int64
    let genome_id: Int64
    let position: Int32
    let role: GenomeRole
    let elite_rank: Int32?
    let ce: Double
    let accuracy: Double
    let fitness_score: Double?
    let eval_time_ms: Int32?
    let created_at: String

    var createdDate: Date? {
        ISO8601DateFormatter().date(from: created_at)
    }

    /// Accuracy as percentage
    var accuracyPercent: Double {
        accuracy * 100
    }

    /// Evaluation time formatted
    var formattedEvalTime: String? {
        guard let ms = eval_time_ms else { return nil }
        if ms < 1000 {
            return "\(ms)ms"
        } else {
            return String(format: "%.1fs", Double(ms) / 1000)
        }
    }
}
