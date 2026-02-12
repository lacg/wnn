// Genome and GenomeEvaluation models

import Foundation

public struct Genome: Codable, Identifiable {
    public let id: Int64
    public let experiment_id: Int64
    public let config_hash: String
    public let tiers_json: String
    public let total_clusters: Int32
    public let total_neurons: Int32
    public let total_memory_bytes: Int64
    public let created_at: String

    public var createdDate: Date? { ISO8601DateFormatter().date(from: created_at) }

    public var tiers: [TierConfig]? {
        guard let data = tiers_json.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode([TierConfig].self, from: data)
    }

    public var formattedMemorySize: String {
        ByteCountFormatter.string(fromByteCount: total_memory_bytes, countStyle: .binary)
    }
}

public struct TierConfig: Codable {
    public let tier: Int32
    public let clusters: Int32
    public let neurons: Int32
    public let bits: Int32
}

public struct GenomeEvaluation: Codable, Identifiable {
    public let id: Int64
    public let iteration_id: Int64
    public let genome_id: Int64
    public let position: Int32
    public let role: GenomeRole
    public let elite_rank: Int32?
    public let ce: Double
    public let accuracy: Double
    public let fitness_score: Double?
    public let eval_time_ms: Int32?
    public let created_at: String

    public var createdDate: Date? { ISO8601DateFormatter().date(from: created_at) }
    public var accuracyPercent: Double { accuracy * 100 }

    public var formattedEvalTime: String? {
        guard let ms = eval_time_ms else { return nil }
        return ms < 1000 ? "\(ms)ms" : String(format: "%.1fs", Double(ms) / 1000)
    }
}
