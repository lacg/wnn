// TestSummary - Full-dataset test results at checkpoints

import Foundation

public enum TestPoint: String, Codable {
    case `init`, final

    public var displayName: String {
        switch self {
        case .`init`: return "Init"
        case .final: return "Final"
        }
    }
}

public enum GenomeTestType: String, Codable {
    case best_ce, best_acc, best_fitness

    public var displayName: String {
        switch self {
        case .best_ce: return "Best CE"
        case .best_acc: return "Best Acc"
        case .best_fitness: return "Best Fitness"
        }
    }
}

public struct TestSummary: Codable, Identifiable, Hashable {
    public let id: Int64
    public let flow_id: Int64?
    public let experiment_id: Int64
    public let test_point: TestPoint
    public let genome_type: GenomeTestType
    public let genome_hash: String
    public let ce: Double
    public let accuracy: Double
    public let created_at: String

    public var accuracyPercent: Double {
        accuracy * 100
    }
}
