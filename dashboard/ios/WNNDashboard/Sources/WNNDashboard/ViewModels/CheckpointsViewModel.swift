// CheckpointsViewModel - manages checkpoints list

import Foundation
import Combine

@MainActor
public final class CheckpointsViewModel: ObservableObject {
    @Published public private(set) var checkpoints: [Checkpoint] = []
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

    public var totalSize: Int64 { checkpoints.compactMap { $0.file_size_bytes }.reduce(0, +) }

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
        do { checkpoints = try await apiClient.getCheckpoints().sorted { $0.created_at > $1.created_at } } catch let err { self.error = err.localizedDescription }
        isLoading = false
    }

    public func deleteCheckpoint(_ id: Int64) async { do { try await apiClient.deleteCheckpoint(id); checkpoints.removeAll { $0.id == id } } catch let err { self.error = err.localizedDescription } }
    public func downloadURL(for checkpoint: Checkpoint) -> URL? { apiClient.checkpointDownloadURL(checkpoint.id) }
    public func refresh() async { await loadCheckpoints() }
    public func clearError() { error = nil }
}
