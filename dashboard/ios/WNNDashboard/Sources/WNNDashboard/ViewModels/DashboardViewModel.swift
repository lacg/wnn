// DashboardViewModel - manages live dashboard state

import Foundation
import Combine

@MainActor
public final class DashboardViewModel: ObservableObject {
    @Published public private(set) var snapshot: DashboardSnapshot = .empty
    @Published public private(set) var isLoading = false
    @Published public private(set) var error: String?
    @Published public var selectedIteration: Iteration?
    @Published public var selectedIterationGenomes: [GenomeEvaluation] = []

    public var currentExperiment: Experiment? { snapshot.current_experiment }
    public var currentPhase: Phase? { snapshot.current_phase }
    public var phases: [Phase] { snapshot.phases }
    public var iterations: [Iteration] { snapshot.iterations }
    public var bestCE: Double { snapshot.best_ce }
    public var bestAccuracy: Double { snapshot.best_accuracy }
    public var isRunning: Bool { currentExperiment?.status == .running }
    public var phaseProgress: Double { currentPhase?.progress ?? 0 }
    public var currentIterationNumber: Int32 { currentPhase?.current_iteration ?? 0 }
    public var maxIterations: Int32 { currentPhase?.max_iterations ?? 0 }

    private let apiClient: APIClient
    private let wsManager: WebSocketManager
    private var cancellables = Set<AnyCancellable>()

    public init(apiClient: APIClient, wsManager: WebSocketManager) {
        self.apiClient = apiClient
        self.wsManager = wsManager
        wsManager.$snapshot.compactMap { $0 }.receive(on: DispatchQueue.main).sink { [weak self] in self?.snapshot = $0 }.store(in: &cancellables)
        wsManager.onMessage = { [weak self] msg in Task { @MainActor in self?.handleMessage(msg) } }
    }

    private func handleMessage(_ msg: WsMessage) {
        if case .genomeEvaluations(let id, let evals) = msg, selectedIteration?.id == id {
            selectedIterationGenomes = evals.sorted { $0.position < $1.position }
        }
    }

    public func loadSnapshot() async {
        isLoading = true; error = nil
        do { snapshot = try await apiClient.getSnapshot() } catch { self.error = error.localizedDescription }
        isLoading = false
    }

    public func loadGenomes(for iteration: Iteration) async {
        selectedIteration = iteration
        do { selectedIterationGenomes = try await apiClient.getGenomes(iterationId: iteration.id).sorted { $0.position < $1.position } }
        catch { selectedIterationGenomes = [] }
    }

    public func refresh() async { await loadSnapshot() }
}
