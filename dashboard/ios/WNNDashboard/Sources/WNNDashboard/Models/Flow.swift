// Flow model - a sequence of experiments

import Foundation

public struct Flow: Codable, Identifiable {
    public let id: Int64
    public let name: String
    public let description: String?
    public let config: FlowConfig
    public let created_at: String
    public let started_at: String?
    public let completed_at: String?
    public let status: FlowStatus
    public let seed_checkpoint_id: Int64?
    public let pid: Int64?

    public var createdDate: Date? { ISO8601DateFormatter().date(from: created_at) }
    public var startedDate: Date? { started_at.flatMap { ISO8601DateFormatter().date(from: $0) } }
    public var completedDate: Date? { completed_at.flatMap { ISO8601DateFormatter().date(from: $0) } }

    public var duration: TimeInterval? {
        guard let start = startedDate else { return nil }
        return (completedDate ?? Date()).timeIntervalSince(start)
    }
}

public struct FlowConfig: Codable {
    public let experiments: [ExperimentSpec]
    public let template: String?
    public let params: [String: AnyCodable]

    public init(experiments: [ExperimentSpec], template: String? = nil, params: [String: AnyCodable] = [:]) {
        self.experiments = experiments
        self.template = template
        self.params = params
    }
}

public struct ExperimentSpec: Codable, Identifiable {
    public var id: String { name }
    public let name: String
    public let experiment_type: ExperimentType
    public let optimize_bits: Bool
    public let optimize_neurons: Bool
    public let optimize_connections: Bool
    public let params: [String: AnyCodable]

    public init(name: String, experiment_type: ExperimentType, optimize_bits: Bool = false,
                optimize_neurons: Bool = false, optimize_connections: Bool = false,
                params: [String: AnyCodable] = [:]) {
        self.name = name
        self.experiment_type = experiment_type
        self.optimize_bits = optimize_bits
        self.optimize_neurons = optimize_neurons
        self.optimize_connections = optimize_connections
        self.params = params
    }
}

public struct CreateFlowRequest: Codable {
    public let name: String
    public let description: String?
    public let config: FlowConfig
    public let seed_checkpoint_id: Int64?

    public init(name: String, description: String? = nil, config: FlowConfig, seed_checkpoint_id: Int64? = nil) {
        self.name = name
        self.description = description
        self.config = config
        self.seed_checkpoint_id = seed_checkpoint_id
    }
}
