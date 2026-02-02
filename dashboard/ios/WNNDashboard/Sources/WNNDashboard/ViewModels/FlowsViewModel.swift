// FlowsViewModel - manages flows list and control

import Foundation
import Combine

@MainActor
public final class FlowsViewModel: ObservableObject {
    @Published public private(set) var flows: [Flow] = []
    @Published public private(set) var isLoading = false
    @Published public private(set) var error: String?
    @Published public var showingNewFlowSheet = false
    @Published public var selectedFlow: Flow?
    @Published public var selectedFlowExperiments: [Experiment] = []
    @Published public var statusFilter: FlowStatus?

    public var filteredFlows: [Flow] { statusFilter.map { s in flows.filter { $0.status == s } } ?? flows }
    public var runningFlows: [Flow] { flows.filter { $0.status == .running } }

    private let apiClient: APIClient
    private let wsManager: WebSocketManager
    private var cancellables = Set<AnyCancellable>()

    public init(apiClient: APIClient, wsManager: WebSocketManager) {
        self.apiClient = apiClient
        self.wsManager = wsManager
        wsManager.onMessage = { [weak self] msg in Task { @MainActor in self?.handleMessage(msg) } }
    }

    private func handleMessage(_ msg: WsMessage) {
        switch msg {
        case .flowStarted(let f), .flowCompleted(let f), .flowCancelled(let f), .flowQueued(let f): updateFlow(f)
        case .flowFailed(let f, _): updateFlow(f)
        default: break
        }
    }

    private func updateFlow(_ f: Flow) {
        if let i = flows.firstIndex(where: { $0.id == f.id }) { flows[i] = f } else { flows.insert(f, at: 0) }
        if selectedFlow?.id == f.id { selectedFlow = f }
    }

    public func loadFlows() async {
        isLoading = true; error = nil
        do { flows = try await apiClient.getFlows().sorted { $0.created_at > $1.created_at } } catch let err { self.error = err.localizedDescription }
        isLoading = false
    }

    public func loadFlowExperiments(_ flowId: Int64) async {
        do { selectedFlowExperiments = try await apiClient.getFlowExperiments(flowId) } catch { selectedFlowExperiments = [] }
    }

    public func createFlow(_ req: CreateFlowRequest) async throws -> Flow {
        let flow = try await apiClient.createFlow(req)
        flows.insert(flow, at: 0)
        return flow
    }

    public func stopFlow(_ id: Int64) async { do { try await apiClient.stopFlow(id) } catch let err { self.error = err.localizedDescription } }
    public func restartFlow(_ id: Int64) async { do { try await apiClient.restartFlow(id) } catch let err { self.error = err.localizedDescription } }
    public func deleteFlow(_ id: Int64) async { do { try await apiClient.deleteFlow(id); flows.removeAll { $0.id == id } } catch let err { self.error = err.localizedDescription } }
    public func refresh() async { await loadFlows() }
}
