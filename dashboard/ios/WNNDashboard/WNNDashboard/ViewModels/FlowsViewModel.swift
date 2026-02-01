// FlowsViewModel - manages flows list and control

import Foundation
import Combine

/// ViewModel for flows list and management
@MainActor
final class FlowsViewModel: ObservableObject {
    // MARK: - Published State

    @Published private(set) var flows: [Flow] = []
    @Published private(set) var isLoading = false
    @Published private(set) var error: String?
    @Published var showingNewFlowSheet = false

    // Selected flow for detail view
    @Published var selectedFlow: Flow?
    @Published var selectedFlowExperiments: [Experiment] = []

    // Filter
    @Published var statusFilter: FlowStatus?

    // MARK: - Computed Properties

    var filteredFlows: [Flow] {
        if let filter = statusFilter {
            return flows.filter { $0.status == filter }
        }
        return flows
    }

    var runningFlows: [Flow] {
        flows.filter { $0.status == .running }
    }

    var completedFlows: [Flow] {
        flows.filter { $0.status == .completed }
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
        // Listen for flow updates via WebSocket
        wsManager.onMessage = { [weak self] message in
            Task { @MainActor in
                self?.handleMessage(message)
            }
        }
    }

    private func handleMessage(_ message: WsMessage) {
        switch message {
        case .flowStarted(let flow), .flowCompleted(let flow),
             .flowCancelled(let flow), .flowQueued(let flow):
            updateFlow(flow)

        case .flowFailed(let flow, _):
            updateFlow(flow)

        default:
            break
        }
    }

    private func updateFlow(_ updatedFlow: Flow) {
        if let index = flows.firstIndex(where: { $0.id == updatedFlow.id }) {
            flows[index] = updatedFlow
        } else {
            // New flow, add it to the list
            flows.insert(updatedFlow, at: 0)
        }

        // Update selected flow if it matches
        if selectedFlow?.id == updatedFlow.id {
            selectedFlow = updatedFlow
        }
    }

    // MARK: - Actions

    /// Load all flows
    func loadFlows() async {
        isLoading = true
        error = nil

        do {
            flows = try await apiClient.getFlows()
                .sorted { ($0.created_at) > ($1.created_at) }
        } catch {
            self.error = error.localizedDescription
        }

        isLoading = false
    }

    /// Load experiments for a specific flow
    func loadFlowExperiments(_ flowId: Int64) async {
        do {
            selectedFlowExperiments = try await apiClient.getFlowExperiments(flowId)
        } catch {
            print("Failed to load flow experiments: \(error)")
            selectedFlowExperiments = []
        }
    }

    /// Create a new flow
    func createFlow(_ request: CreateFlowRequest) async throws -> Flow {
        let flow = try await apiClient.createFlow(request)
        flows.insert(flow, at: 0)
        return flow
    }

    /// Stop a running flow
    func stopFlow(_ id: Int64) async {
        do {
            try await apiClient.stopFlow(id)
            // The WebSocket will notify us of the status change
        } catch {
            self.error = error.localizedDescription
        }
    }

    /// Restart a flow
    func restartFlow(_ id: Int64) async {
        do {
            try await apiClient.restartFlow(id)
        } catch {
            self.error = error.localizedDescription
        }
    }

    /// Delete a flow
    func deleteFlow(_ id: Int64) async {
        do {
            try await apiClient.deleteFlow(id)
            flows.removeAll { $0.id == id }
        } catch {
            self.error = error.localizedDescription
        }
    }

    /// Refresh flows list
    func refresh() async {
        await loadFlows()
    }
}
