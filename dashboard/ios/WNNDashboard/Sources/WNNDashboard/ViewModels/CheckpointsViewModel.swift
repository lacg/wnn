// CheckpointsViewModel - manages checkpoints grouped by flow → experiment

import Foundation
import Combine

/// A group of checkpoints belonging to one experiment (within a flow)
public struct ExperimentCheckpointGroup: Identifiable {
	public let experimentId: Int64
	public let experimentName: String
	public let experimentStatus: ExperimentStatus
	public let bestCE: Double?
	public let bestAccuracy: Double?
	public let checkpoints: [Checkpoint]

	public var id: Int64 { experimentId }
	public var totalSize: Int64 { checkpoints.compactMap(\.file_size_bytes).reduce(0, +) }
	public var latestDate: Date? { checkpoints.compactMap(\.createdDate).max() }
	public var count: Int { checkpoints.count }
}

/// A group of experiments belonging to one flow
public struct FlowCheckpointGroup: Identifiable {
	public let flowId: Int64
	public let flowName: String
	public let flowStatus: FlowStatus
	public let experiments: [ExperimentCheckpointGroup]

	public var id: Int64 { flowId }
	public var totalCheckpoints: Int { experiments.reduce(0) { $0 + $1.count } }
	public var totalSize: Int64 { experiments.reduce(0) { $0 + $1.totalSize } }
	public var latestDate: Date? { experiments.compactMap(\.latestDate).max() }
	public var bestCE: Double? { experiments.compactMap(\.bestCE).min() }
	public var bestAccuracy: Double? { experiments.compactMap(\.bestAccuracy).max() }
	public var experimentCount: Int { experiments.count }
}

@MainActor
public final class CheckpointsViewModel: ObservableObject {
	@Published public private(set) var checkpoints: [Checkpoint] = []
	@Published public private(set) var experiments: [Experiment] = []
	@Published public private(set) var flows: [Flow] = []
	@Published public private(set) var isLoading = false
	@Published public private(set) var error: String?
	@Published public var typeFilter: CheckpointType?

	public var filteredCheckpoints: [Checkpoint] {
		var result = checkpoints
		if let t = typeFilter { result = result.filter { $0.checkpoint_type == t } }
		return result
	}

	/// Checkpoints grouped by flow → experiment, flows ordered newest to oldest
	public var flowGroups: [FlowCheckpointGroup] {
		let filtered = filteredCheckpoints

		// Group checkpoints by flow_id (nil → standalone)
		let byFlow = Dictionary(grouping: filtered) { $0.flow_id ?? -1 }

		var groups: [FlowCheckpointGroup] = []

		for (flowId, flowCheckpoints) in byFlow where flowId > 0 {
			let flow = flows.first { $0.id == flowId }
			let expGroups = buildExperimentGroups(from: flowCheckpoints)
			groups.append(FlowCheckpointGroup(
				flowId: flowId,
				flowName: flow?.name ?? flowCheckpoints.first?.flow_name ?? "Flow #\(flowId)",
				flowStatus: flow?.status ?? .completed,
				experiments: expGroups
			))
		}

		// Sort flows by newest checkpoint first
		groups.sort { ($0.latestDate ?? .distantPast) > ($1.latestDate ?? .distantPast) }

		// Standalone checkpoints (no flow) go at the end as a pseudo-flow
		if let standalone = byFlow[-1], !standalone.isEmpty {
			let expGroups = buildExperimentGroups(from: standalone)
			groups.append(FlowCheckpointGroup(
				flowId: -1,
				flowName: "Standalone Experiments",
				flowStatus: .completed,
				experiments: expGroups
			))
		}

		return groups
	}

	private func buildExperimentGroups(from checkpoints: [Checkpoint]) -> [ExperimentCheckpointGroup] {
		let byExp = Dictionary(grouping: checkpoints, by: \.experiment_id)
		return byExp.map { expId, ckpts in
			let exp = experiments.first { $0.id == expId }
			return ExperimentCheckpointGroup(
				experimentId: expId,
				experimentName: exp?.name ?? "Experiment #\(expId)",
				experimentStatus: exp?.status ?? .completed,
				bestCE: ckpts.compactMap(\.best_ce).min(),
				bestAccuracy: ckpts.compactMap(\.best_accuracy).max(),
				checkpoints: ckpts.sorted { $0.created_at > $1.created_at }
			)
		}.sorted { ($0.latestDate ?? .distantPast) > ($1.latestDate ?? .distantPast) }
	}

	public var totalSize: Int64 { filteredCheckpoints.compactMap(\.file_size_bytes).reduce(0, +) }

	private let apiClient: APIClient
	private let wsManager: WebSocketManager
	private var cancellables = Set<AnyCancellable>()

	public init(apiClient: APIClient, wsManager: WebSocketManager) {
		self.apiClient = apiClient
		self.wsManager = wsManager
		wsManager.addMessageHandler(id: "checkpoints") { [weak self] msg in Task { @MainActor in self?.handleMessage(msg) } }
	}

	private func handleMessage(_ msg: WsMessage) {
		switch msg {
		case .checkpointCreated(let c): checkpoints.insert(c, at: 0)
		case .checkpointDeleted(let id): checkpoints.removeAll { $0.id == id }
		default: break
		}
	}

	public func loadCheckpoints() async {
		isLoading = true; error = nil
		do {
			async let ckpts = apiClient.getCheckpoints()
			async let exps = apiClient.getExperiments()
			async let fls = apiClient.getFlows()
			checkpoints = try await ckpts.sorted { $0.created_at > $1.created_at }
			experiments = try await exps
			flows = try await fls
		} catch let err {
			self.error = err.localizedDescription
		}
		isLoading = false
	}

	public func deleteCheckpoint(_ id: Int64) async {
		do { try await apiClient.deleteCheckpoint(id); checkpoints.removeAll { $0.id == id } }
		catch let err { self.error = err.localizedDescription }
	}

	public func downloadURL(for checkpoint: Checkpoint) -> URL? { apiClient.checkpointDownloadURL(checkpoint.id) }
	public func refresh() async { await loadCheckpoints() }
	public func clearError() { error = nil }
}
