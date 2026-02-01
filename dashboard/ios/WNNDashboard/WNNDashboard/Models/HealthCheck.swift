// HealthCheck model - periodic full validation

import Foundation

struct HealthCheck: Codable, Identifiable {
    let id: Int64
    let iteration_id: Int64
    let k: Int32
    let top_k_ce: Double
    let top_k_accuracy: Double
    let best_ce: Double?
    let best_ce_accuracy: Double?
    let best_acc_ce: Double?
    let best_acc_accuracy: Double?
    let patience_remaining: Int32?
    let patience_status: String?
    let created_at: String

    var createdDate: Date? {
        ISO8601DateFormatter().date(from: created_at)
    }

    /// Top-K accuracy as percentage
    var topKAccuracyPercent: Double {
        top_k_accuracy * 100
    }

    /// Best CE accuracy as percentage
    var bestCEAccuracyPercent: Double? {
        best_ce_accuracy.map { $0 * 100 }
    }

    /// Best accuracy as percentage
    var bestAccuracyPercent: Double? {
        best_acc_accuracy.map { $0 * 100 }
    }
}
