// Checkpoint model - saved state for resume

import Foundation

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
