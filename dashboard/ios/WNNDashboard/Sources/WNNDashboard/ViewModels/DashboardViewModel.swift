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
    @Published public private(set) var historyIterations: [Iteration] = []
    @Published public private(set) var recentExperiments: [Experiment] = []
    @Published public var selectedHistoryExperiment: Experiment?
    @Published public private(set) var historyPhases: [Phase] = []

    public var currentExperiment: Experiment? { snapshot.current_experiment }
    public var currentPhase: Phase? { snapshot.current_phase }
    public var phases: [Phase] { selectedHistoryExperiment != nil ? historyPhases : snapshot.phases }
    public var iterations: [Iteration] { snapshot.iterations }
    public var bestCE: Double { snapshot.best_ce }
    public var bestAccuracy: Double { snapshot.best_accuracy }
    public var isRunning: Bool { currentExperiment?.status == .running }
    public var phaseProgress: Double { currentPhase?.progress ?? 0 }
    public var currentIterationNumber: Int32 { currentPhase?.current_iteration ?? 0 }
    public var maxIterations: Int32 { currentPhase?.max_iterations ?? 0 }

    public let apiClient: APIClient
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

    public func loadHistoryIterations(for phase: Phase) async {
        do {
            historyIterations = try await apiClient.getPhaseIterations(phaseId: phase.id)
                .sorted { $0.iteration_num < $1.iteration_num }
        } catch {
            historyIterations = []
            self.error = error.localizedDescription
        }
    }

    public func loadRecentExperiments() async {
        do {
            let all = try await apiClient.getExperiments()
            recentExperiments = all
                .filter { $0.status == .completed || $0.status == .failed }
                .prefix(10)
                .map { $0 }
        } catch {
            recentExperiments = []
        }
    }

    public func selectHistoryExperiment(_ exp: Experiment) async {
        if selectedHistoryExperiment?.id == exp.id {
            // Toggle off
            clearHistoryExperiment()
            return
        }

        selectedHistoryExperiment = exp

        do {
            // Load phases and iterations for this experiment
            async let phasesTask = apiClient.getPhases(experimentId: exp.id)
            async let iterationsTask = apiClient.getIterations(experimentId: exp.id)

            historyPhases = try await phasesTask.sorted { $0.sequence_order < $1.sequence_order }
            historyIterations = try await iterationsTask.sorted { $0.iteration_num < $1.iteration_num }
        } catch {
            historyPhases = []
            historyIterations = []
            self.error = error.localizedDescription
        }
    }

    public func clearHistoryExperiment() {
        selectedHistoryExperiment = nil
        historyPhases = []
        historyIterations = []
    }
}
