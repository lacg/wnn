// WebSocket message types - tagged enum matching Rust backend

import Foundation

/// WebSocket message from server
/// Rust uses: #[serde(tag = "type", content = "data")]
enum WsMessage: Codable {
    case snapshot(DashboardSnapshot)
    case iterationCompleted(Iteration)
    case genomeEvaluations(iterationId: Int64, evaluations: [GenomeEvaluation])
    case phaseStarted(Phase)
    case phaseCompleted(Phase)
    case healthCheck(HealthCheck)
    case experimentStatusChanged(Experiment)
    case flowStarted(Flow)
    case flowCompleted(Flow)
    case flowFailed(flow: Flow, error: String)
    case flowCancelled(Flow)
    case flowQueued(Flow)
    case checkpointCreated(Checkpoint)
    case checkpointDeleted(id: Int64)

    private enum CodingKeys: String, CodingKey {
        case type
        case data
    }

    private enum MessageType: String, Codable {
        case Snapshot
        case IterationCompleted
        case GenomeEvaluations
        case PhaseStarted
        case PhaseCompleted
        case HealthCheck
        case ExperimentStatusChanged
        case FlowStarted
        case FlowCompleted
        case FlowFailed
        case FlowCancelled
        case FlowQueued
        case CheckpointCreated
        case CheckpointDeleted
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(MessageType.self, forKey: .type)

        switch type {
        case .Snapshot:
            let data = try container.decode(DashboardSnapshot.self, forKey: .data)
            self = .snapshot(data)

        case .IterationCompleted:
            let data = try container.decode(Iteration.self, forKey: .data)
            self = .iterationCompleted(data)

        case .GenomeEvaluations:
            let data = try container.decode(GenomeEvaluationsData.self, forKey: .data)
            self = .genomeEvaluations(iterationId: data.iteration_id, evaluations: data.evaluations)

        case .PhaseStarted:
            let data = try container.decode(Phase.self, forKey: .data)
            self = .phaseStarted(data)

        case .PhaseCompleted:
            let data = try container.decode(Phase.self, forKey: .data)
            self = .phaseCompleted(data)

        case .HealthCheck:
            let data = try container.decode(HealthCheck.self, forKey: .data)
            self = .healthCheck(data)

        case .ExperimentStatusChanged:
            let data = try container.decode(Experiment.self, forKey: .data)
            self = .experimentStatusChanged(data)

        case .FlowStarted:
            let data = try container.decode(Flow.self, forKey: .data)
            self = .flowStarted(data)

        case .FlowCompleted:
            let data = try container.decode(Flow.self, forKey: .data)
            self = .flowCompleted(data)

        case .FlowFailed:
            let data = try container.decode(FlowFailedData.self, forKey: .data)
            self = .flowFailed(flow: data.flow, error: data.error)

        case .FlowCancelled:
            let data = try container.decode(Flow.self, forKey: .data)
            self = .flowCancelled(data)

        case .FlowQueued:
            let data = try container.decode(Flow.self, forKey: .data)
            self = .flowQueued(data)

        case .CheckpointCreated:
            let data = try container.decode(Checkpoint.self, forKey: .data)
            self = .checkpointCreated(data)

        case .CheckpointDeleted:
            let data = try container.decode(CheckpointDeletedData.self, forKey: .data)
            self = .checkpointDeleted(id: data.id)
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        switch self {
        case .snapshot(let data):
            try container.encode(MessageType.Snapshot, forKey: .type)
            try container.encode(data, forKey: .data)

        case .iterationCompleted(let data):
            try container.encode(MessageType.IterationCompleted, forKey: .type)
            try container.encode(data, forKey: .data)

        case .genomeEvaluations(let iterationId, let evaluations):
            try container.encode(MessageType.GenomeEvaluations, forKey: .type)
            try container.encode(GenomeEvaluationsData(iteration_id: iterationId, evaluations: evaluations), forKey: .data)

        case .phaseStarted(let data):
            try container.encode(MessageType.PhaseStarted, forKey: .type)
            try container.encode(data, forKey: .data)

        case .phaseCompleted(let data):
            try container.encode(MessageType.PhaseCompleted, forKey: .type)
            try container.encode(data, forKey: .data)

        case .healthCheck(let data):
            try container.encode(MessageType.HealthCheck, forKey: .type)
            try container.encode(data, forKey: .data)

        case .experimentStatusChanged(let data):
            try container.encode(MessageType.ExperimentStatusChanged, forKey: .type)
            try container.encode(data, forKey: .data)

        case .flowStarted(let data):
            try container.encode(MessageType.FlowStarted, forKey: .type)
            try container.encode(data, forKey: .data)

        case .flowCompleted(let data):
            try container.encode(MessageType.FlowCompleted, forKey: .type)
            try container.encode(data, forKey: .data)

        case .flowFailed(let flow, let error):
            try container.encode(MessageType.FlowFailed, forKey: .type)
            try container.encode(FlowFailedData(flow: flow, error: error), forKey: .data)

        case .flowCancelled(let data):
            try container.encode(MessageType.FlowCancelled, forKey: .type)
            try container.encode(data, forKey: .data)

        case .flowQueued(let data):
            try container.encode(MessageType.FlowQueued, forKey: .type)
            try container.encode(data, forKey: .data)

        case .checkpointCreated(let data):
            try container.encode(MessageType.CheckpointCreated, forKey: .type)
            try container.encode(data, forKey: .data)

        case .checkpointDeleted(let id):
            try container.encode(MessageType.CheckpointDeleted, forKey: .type)
            try container.encode(CheckpointDeletedData(id: id), forKey: .data)
        }
    }
}

// Helper structs for complex message data
private struct GenomeEvaluationsData: Codable {
    let iteration_id: Int64
    let evaluations: [GenomeEvaluation]
}

private struct FlowFailedData: Codable {
    let flow: Flow
    let error: String
}

private struct CheckpointDeletedData: Codable {
    let id: Int64
}
