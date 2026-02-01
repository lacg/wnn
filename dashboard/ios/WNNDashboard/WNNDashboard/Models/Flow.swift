// Flow model - a sequence of experiments

import Foundation

struct Flow: Codable, Identifiable {
    let id: Int64
    let name: String
    let description: String?
    let config: FlowConfig
    let created_at: String
    let started_at: String?
    let completed_at: String?
    let status: FlowStatus
    let seed_checkpoint_id: Int64?
    let pid: Int64?

    var createdDate: Date? {
        ISO8601DateFormatter().date(from: created_at)
    }

    var startedDate: Date? {
        started_at.flatMap { ISO8601DateFormatter().date(from: $0) }
    }

    var completedDate: Date? {
        completed_at.flatMap { ISO8601DateFormatter().date(from: $0) }
    }

    var duration: TimeInterval? {
        guard let start = startedDate else { return nil }
        let end = completedDate ?? Date()
        return end.timeIntervalSince(start)
    }
}

struct FlowConfig: Codable {
    let experiments: [ExperimentSpec]
    let template: String?
    let params: [String: AnyCodable]

    init(experiments: [ExperimentSpec], template: String? = nil, params: [String: AnyCodable] = [:]) {
        self.experiments = experiments
        self.template = template
        self.params = params
    }
}

struct ExperimentSpec: Codable, Identifiable {
    var id: String { name }

    let name: String
    let experiment_type: ExperimentType
    let optimize_bits: Bool
    let optimize_neurons: Bool
    let optimize_connections: Bool
    let params: [String: AnyCodable]

    init(
        name: String,
        experiment_type: ExperimentType,
        optimize_bits: Bool = false,
        optimize_neurons: Bool = false,
        optimize_connections: Bool = false,
        params: [String: AnyCodable] = [:]
    ) {
        self.name = name
        self.experiment_type = experiment_type
        self.optimize_bits = optimize_bits
        self.optimize_neurons = optimize_neurons
        self.optimize_connections = optimize_connections
        self.params = params
    }
}

// Request model for creating a new flow
struct CreateFlowRequest: Codable {
    let name: String
    let description: String?
    let config: FlowConfig
    let seed_checkpoint_id: Int64?

    init(name: String, description: String? = nil, config: FlowConfig, seed_checkpoint_id: Int64? = nil) {
        self.name = name
        self.description = description
        self.config = config
        self.seed_checkpoint_id = seed_checkpoint_id
    }
}
