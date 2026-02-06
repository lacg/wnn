// Checkpoint model - saved state for resume

import Foundation

public struct TierStats: Codable, Identifiable, Hashable {
    public var id: Int { tier_index }
    public let tier_index: Int
    public let cluster_count: Int
    public let start_cluster: Int
    public let end_cluster: Int
    public let avg_bits: Double
    public let avg_neurons: Double
    public let min_bits: Int
    public let max_bits: Int
    public let min_neurons: Int
    public let max_neurons: Int
    public let total_neurons: Int
    public let total_connections: Int?
}

public struct GenomeStats: Codable, Hashable {
    public let num_clusters: Int
    public let total_neurons: Int
    public let total_connections: Int
    public let bits_range: [Int]
    public let neurons_range: [Int]
    public let tier_stats: [TierStats]?
}

public struct Checkpoint: Codable, Identifiable {
    public let id: Int64
    public let experiment_id: Int64
    public let iteration_id: Int64?
    public let name: String
    public let file_path: String
    public let file_size_bytes: Int64?
    public let checkpoint_type: CheckpointType
    public let best_ce: Double?
    public let best_accuracy: Double?
    public let genome_stats: GenomeStats?
    public let created_at: String

    public var createdDate: Date? { ISO8601DateFormatter().date(from: created_at) }

    public var formattedFileSize: String? {
        guard let bytes = file_size_bytes else { return nil }
        return ByteCountFormatter.string(fromByteCount: bytes, countStyle: .binary)
    }

    public var accuracyPercent: Double? { best_accuracy.map { $0 * 100 } }
}

public struct HealthCheck: Codable, Identifiable {
    public let id: Int64
    public let iteration_id: Int64
    public let k: Int32
    public let top_k_ce: Double
    public let top_k_accuracy: Double
    public let best_ce: Double?
    public let best_ce_accuracy: Double?
    public let best_acc_ce: Double?
    public let best_acc_accuracy: Double?
    public let patience_remaining: Int32?
    public let patience_status: String?
    public let created_at: String

    public var createdDate: Date? { ISO8601DateFormatter().date(from: created_at) }
    public var topKAccuracyPercent: Double { top_k_accuracy * 100 }
}
