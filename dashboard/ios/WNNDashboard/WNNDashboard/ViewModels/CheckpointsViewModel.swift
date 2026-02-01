// CheckpointsViewModel - manages checkpoints list

import Foundation
import Combine

/// ViewModel for checkpoints management
@MainActor
final class CheckpointsViewModel: ObservableObject {
    // MARK: - Published State

    @Published private(set) var checkpoints: [Checkpoint] = []
    @Published private(set) var isLoading = false
    @Published private(set) var error: String?

    // Filters
    @Published var experimentFilter: Int64?
    @Published var typeFilter: CheckpointType?

    // MARK: - Computed Properties

    var filteredCheckpoints: [Checkpoint] {
        var result = checkpoints

        if let expId = experimentFilter {
            result = result.filter { $0.experiment_id == expId }
        }

        if let type = typeFilter {
            result = result.filter { $0.checkpoint_type == type }
        }

        return result
    }

    var groupedByExperiment: [Int64: [Checkpoint]] {
        Dictionary(grouping: filteredCheckpoints) { $0.experiment_id }
    }

    var totalSize: Int64 {
        checkpoints.compactMap { $0.file_size_bytes }.reduce(0, +)
    }

    // MARK: - Dependencies

    private let apiClient: APIClient
    private let wsManager: WebSocketManager
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Initialization

    init(apiClient: APIClient, wsManager: WebSocketManager) {
        self.apiClient = apiClient
        self.wsManager = wsManager

        setupBindings()
    }

    private func setupBindings() {
        wsManager.onMessage = { [weak self] message in
            Task { @MainActor in
                self?.handleMessage(message)
            }
        }
    }

    private func handleMessage(_ message: WsMessage) {
        switch message {
        case .checkpointCreated(let checkpoint):
            checkpoints.insert(checkpoint, at: 0)

        case .checkpointDeleted(let id):
            checkpoints.removeAll { $0.id == id }

        default:
            break
        }
    }

    // MARK: - Actions

    /// Load all checkpoints
    func loadCheckpoints() async {
        isLoading = true
        error = nil

        do {
            checkpoints = try await apiClient.getCheckpoints()
                .sorted { $0.created_at > $1.created_at }
        } catch {
            self.error = error.localizedDescription
        }

        isLoading = false
    }

    /// Delete a checkpoint
    func deleteCheckpoint(_ id: Int64) async {
        do {
            try await apiClient.deleteCheckpoint(id)
            checkpoints.removeAll { $0.id == id }
        } catch {
            self.error = error.localizedDescription
        }
    }

    /// Get download URL for a checkpoint
    func downloadURL(for checkpoint: Checkpoint) -> URL? {
        apiClient.checkpointDownloadURL(checkpoint.id)
    }

    /// Refresh checkpoints
    func refresh() async {
        await loadCheckpoints()
    }

    /// Clear error
    func clearError() {
        error = nil
    }
}
