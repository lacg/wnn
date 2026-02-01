// Checkpoint model - saved state for resume

import Foundation

struct Checkpoint: Codable, Identifiable {
    let id: Int64
    let experiment_id: Int64
    let phase_id: Int64?
    let iteration_id: Int64?
    let name: String
    let file_path: String
    let file_size_bytes: Int64?
    let checkpoint_type: CheckpointType
    let best_ce: Double?
    let best_accuracy: Double?
    let created_at: String

    var createdDate: Date? {
        ISO8601DateFormatter().date(from: created_at)
    }

    /// File size formatted as human-readable string
    var formattedFileSize: String? {
        guard let bytes = file_size_bytes else { return nil }
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

    /// Accuracy as percentage
    var accuracyPercent: Double? {
        best_accuracy.map { $0 * 100 }
    }
}
