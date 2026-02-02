// WebSocket message types - tagged enum matching Rust backend

import Foundation

public enum WsMessage: Codable {
    case snapshot(DashboardSnapshot)
    case iterationCompleted(Iteration)
    case genomeEvaluations(iterationId: Int64, evaluations: [GenomeEvaluation])
    case healthCheck(HealthCheck)
    case experimentStatusChanged(Experiment)
    case flowStarted(Flow)
    case flowCompleted(Flow)
    case flowFailed(flow: Flow, error: String)
    case flowCancelled(Flow)
    case flowQueued(Flow)
    case checkpointCreated(Checkpoint)
    case checkpointDeleted(id: Int64)

    private enum CodingKeys: String, CodingKey { case type, data }

    private enum MessageType: String, Codable {
        case Snapshot, IterationCompleted, GenomeEvaluations
        case HealthCheck, ExperimentStatusChanged, FlowStarted, FlowCompleted, FlowFailed
        case FlowCancelled, FlowQueued, CheckpointCreated, CheckpointDeleted
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(MessageType.self, forKey: .type)

        switch type {
        case .Snapshot: self = .snapshot(try container.decode(DashboardSnapshot.self, forKey: .data))
        case .IterationCompleted: self = .iterationCompleted(try container.decode(Iteration.self, forKey: .data))
        case .GenomeEvaluations:
            let data = try container.decode(GenomeEvaluationsData.self, forKey: .data)
            self = .genomeEvaluations(iterationId: data.iteration_id, evaluations: data.evaluations)
        case .HealthCheck: self = .healthCheck(try container.decode(HealthCheck.self, forKey: .data))
        case .ExperimentStatusChanged: self = .experimentStatusChanged(try container.decode(Experiment.self, forKey: .data))
        case .FlowStarted: self = .flowStarted(try container.decode(Flow.self, forKey: .data))
        case .FlowCompleted: self = .flowCompleted(try container.decode(Flow.self, forKey: .data))
        case .FlowFailed:
            let data = try container.decode(FlowFailedData.self, forKey: .data)
            self = .flowFailed(flow: data.flow, error: data.error)
        case .FlowCancelled: self = .flowCancelled(try container.decode(Flow.self, forKey: .data))
        case .FlowQueued: self = .flowQueued(try container.decode(Flow.self, forKey: .data))
        case .CheckpointCreated: self = .checkpointCreated(try container.decode(Checkpoint.self, forKey: .data))
        case .CheckpointDeleted:
            let data = try container.decode(CheckpointDeletedData.self, forKey: .data)
            self = .checkpointDeleted(id: data.id)
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .snapshot(let d): try container.encode(MessageType.Snapshot, forKey: .type); try container.encode(d, forKey: .data)
        case .iterationCompleted(let d): try container.encode(MessageType.IterationCompleted, forKey: .type); try container.encode(d, forKey: .data)
        case .genomeEvaluations(let id, let evals): try container.encode(MessageType.GenomeEvaluations, forKey: .type); try container.encode(GenomeEvaluationsData(iteration_id: id, evaluations: evals), forKey: .data)
        case .healthCheck(let d): try container.encode(MessageType.HealthCheck, forKey: .type); try container.encode(d, forKey: .data)
        case .experimentStatusChanged(let d): try container.encode(MessageType.ExperimentStatusChanged, forKey: .type); try container.encode(d, forKey: .data)
        case .flowStarted(let d): try container.encode(MessageType.FlowStarted, forKey: .type); try container.encode(d, forKey: .data)
        case .flowCompleted(let d): try container.encode(MessageType.FlowCompleted, forKey: .type); try container.encode(d, forKey: .data)
        case .flowFailed(let f, let e): try container.encode(MessageType.FlowFailed, forKey: .type); try container.encode(FlowFailedData(flow: f, error: e), forKey: .data)
        case .flowCancelled(let d): try container.encode(MessageType.FlowCancelled, forKey: .type); try container.encode(d, forKey: .data)
        case .flowQueued(let d): try container.encode(MessageType.FlowQueued, forKey: .type); try container.encode(d, forKey: .data)
        case .checkpointCreated(let d): try container.encode(MessageType.CheckpointCreated, forKey: .type); try container.encode(d, forKey: .data)
        case .checkpointDeleted(let id): try container.encode(MessageType.CheckpointDeleted, forKey: .type); try container.encode(CheckpointDeletedData(id: id), forKey: .data)
        }
    }
}

private struct GenomeEvaluationsData: Codable { let iteration_id: Int64; let evaluations: [GenomeEvaluation] }
private struct FlowFailedData: Codable { let flow: Flow; let error: String }
private struct CheckpointDeletedData: Codable { let id: Int64 }
