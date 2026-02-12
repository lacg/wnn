// GatingRun model - gating analysis runs for experiments

import Foundation

public struct GatingConfig: Codable, Hashable {
    public let neurons_per_gate: Int
    public let bits_per_neuron: Int
    public let threshold: Double

    public var description: String {
        "\(neurons_per_gate)n/\(bits_per_neuron)b @ \(String(format: "%.1f", threshold))"
    }
}

public struct GatingResult: Codable, Identifiable, Hashable {
    public var id: String { genome_type }
    public let genome_type: String
    public let ce: Double
    public let acc: Double
    public let gated_ce: Double
    public let gated_acc: Double
    public let gating_config: GatingConfig

    public var ceImprovement: Double {
        guard ce > 0 else { return 0 }
        return ((ce - gated_ce) / ce) * 100
    }

    public var accImprovement: Double {
        guard acc > 0 else { return 0 }
        return ((gated_acc - acc) / acc) * 100
    }

    public var genomeTypeDisplay: String {
        switch genome_type {
        case "best_ce": return "Best CE"
        case "best_acc": return "Best Accuracy"
        case "best_fitness": return "Best Fitness"
        default: return genome_type
        }
    }
}

public struct GatingRun: Codable, Identifiable, Hashable {
    public let id: Int64
    public let experiment_id: Int64
    public let status: GatingStatus
    public let config: GatingConfig?
    public let genomes_tested: Int32?
    public let results: [GatingResult]?
    public let error: String?
    public let created_at: String
    public let started_at: String?
    public let completed_at: String?

    public var createdDate: Date? { ISO8601DateFormatter().date(from: created_at) }
    public var startedDate: Date? { started_at.flatMap { ISO8601DateFormatter().date(from: $0) } }
    public var completedDate: Date? { completed_at.flatMap { ISO8601DateFormatter().date(from: $0) } }

    public var duration: TimeInterval? {
        guard let start = startedDate else { return nil }
        return (completedDate ?? Date()).timeIntervalSince(start)
    }

    public var hasResults: Bool {
        results != nil && !(results?.isEmpty ?? true)
    }
}
