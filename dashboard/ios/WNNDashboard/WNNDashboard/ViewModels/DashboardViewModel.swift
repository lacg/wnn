// DashboardViewModel - manages live dashboard state

import Foundation
import Combine

/// ViewModel for the main dashboard view
@MainActor
final class DashboardViewModel: ObservableObject {
    // MARK: - Published State

    @Published private(set) var snapshot: DashboardSnapshot = .empty
    @Published private(set) var isLoading = false
    @Published private(set) var error: String?

    // Selected iteration for detail sheet
    @Published var selectedIteration: Iteration?
    @Published var selectedIterationGenomes: [GenomeEvaluation] = []

    // MARK: - Computed Properties

    var currentExperiment: Experiment? { snapshot.current_experiment }
    var currentPhase: Phase? { snapshot.current_phase }
    var phases: [Phase] { snapshot.phases }
    var iterations: [Iteration] { snapshot.iterations }
    var bestCE: Double { snapshot.best_ce }
    var bestAccuracy: Double { snapshot.best_accuracy }

    var isRunning: Bool {
        currentExperiment?.status == .running
    }

    var phaseProgress: Double {
        currentPhase?.progress ?? 0
    }

    var currentIterationNumber: Int32 {
        currentPhase?.current_iteration ?? 0
    }

    var maxIterations: Int32 {
        currentPhase?.max_iterations ?? 0
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
        // Subscribe to WebSocket snapshot updates
        wsManager.$snapshot
            .compactMap { $0 }
            .receive(on: DispatchQueue.main)
            .sink { [weak self] newSnapshot in
                self?.snapshot = newSnapshot
            }
            .store(in: &cancellables)

        // Handle specific message types
        wsManager.onMessage = { [weak self] message in
            Task { @MainActor in
                self?.handleMessage(message)
            }
        }
    }

    // MARK: - Message Handling

    private func handleMessage(_ message: WsMessage) {
        switch message {
        case .genomeEvaluations(let iterationId, let evaluations):
            // Update genomes if this is the selected iteration
            if selectedIteration?.id == iterationId {
                selectedIterationGenomes = evaluations.sorted { $0.position < $1.position }
            }
        default:
            break
        }
    }

    // MARK: - Actions

    /// Load initial snapshot
    func loadSnapshot() async {
        isLoading = true
        error = nil

        do {
            snapshot = try await apiClient.getSnapshot()
        } catch {
            self.error = error.localizedDescription
        }

        isLoading = false
    }

    /// Load genomes for a specific iteration
    func loadGenomes(for iteration: Iteration) async {
        selectedIteration = iteration

        do {
            selectedIterationGenomes = try await apiClient.getGenomes(iterationId: iteration.id)
                .sorted { $0.position < $1.position }
        } catch {
            print("Failed to load genomes: \(error)")
            selectedIterationGenomes = []
        }
    }

    /// Refresh data
    func refresh() async {
        await loadSnapshot()
    }
}
