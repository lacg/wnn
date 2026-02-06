// CheckpointsViewModel - manages checkpoints grouped by experiment

import Foundation
import Combine

/// A group of checkpoints belonging to one experiment
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

@MainActor
public final class CheckpointsViewModel: ObservableObject {
	@Published public private(set) var checkpoints: [Checkpoint] = []
	@Published public private(set) var experiments: [Experiment] = []
	@Published public private(set) var isLoading = false
	@Published public private(set) var error: String?
	@Published public var experimentFilter: Int64?
	@Published public var typeFilter: CheckpointType?

	public var filteredCheckpoints: [Checkpoint] {
		var result = checkpoints
		if let e = experimentFilter { result = result.filter { $0.experiment_id == e } }
		if let t = typeFilter { result = result.filter { $0.checkpoint_type == t } }
		return result
	}

	/// Checkpoints grouped by experiment, ordered by most recent checkpoint date
	public var experimentGroups: [ExperimentCheckpointGroup] {
		let grouped = Dictionary(grouping: filteredCheckpoints, by: \.experiment_id)
		return grouped.map { expId, ckpts in
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

	public var totalSize: Int64 { filteredCheckpoints.compactMap { $0.file_size_bytes }.reduce(0, +) }

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
			checkpoints = try await ckpts.sorted { $0.created_at > $1.created_at }
			experiments = try await exps
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
