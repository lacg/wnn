// DashboardViewModel - manages live dashboard state with expandable experiments

import Foundation
import Combine

@MainActor
public final class DashboardViewModel: ObservableObject {
	@Published public private(set) var snapshot: DashboardSnapshot = .empty
	@Published public private(set) var isLoading = false
	@Published public private(set) var error: String?
	@Published public var selectedIteration: Iteration?
	@Published public var selectedIterationGenomes: [GenomeEvaluation] = []
	@Published public private(set) var recentExperiments: [Experiment] = []
	/// Cached iterations per experiment (keyed by experiment ID)
	@Published public private(set) var experimentIterations: [Int64: [Iteration]] = [:]
	/// Which experiment is selected for chart display
	@Published public var chartExperimentId: Int64?

	public var currentExperiment: Experiment? { snapshot.current_experiment }
	public var liveIterations: [Iteration] { snapshot.iterations }
	public var bestCE: Double { snapshot.best_ce }
	public var bestAccuracy: Double { snapshot.best_accuracy }
	public var isRunning: Bool { currentExperiment?.status == .running }
	public var experimentProgress: Double { currentExperiment?.progress ?? 0 }
	public var currentIterationNumber: Int32 {
		liveIterations.map(\.iteration_num).max() ?? currentExperiment?.current_iteration ?? 0
	}
	public var maxIterations: Int32 { currentExperiment?.max_iterations ?? 0 }

	/// Flow name for the current flow (if running experiment belongs to one)
	public var currentFlowName: String? {
		guard let flowId = currentExperiment?.flow_id else { return nil }
		return currentFlowExperiments.isEmpty ? nil : "Flow #\(flowId)"
	}

	/// Experiments for the current flow, or recent experiments if no flow
	@Published public private(set) var currentFlowExperiments: [Experiment] = []

	/// All experiments to display: flow experiments (ordered newest first) or running + recent
	public var displayExperiments: [Experiment] {
		if !currentFlowExperiments.isEmpty {
			return currentFlowExperiments
				.sorted { ($0.endedDate ?? $0.startedDate ?? $0.createdDate ?? .distantPast) > ($1.endedDate ?? $1.startedDate ?? $1.createdDate ?? .distantPast) }
		}
		var result: [Experiment] = []
		if let running = currentExperiment { result.append(running) }
		let runningId = currentExperiment?.id
		result += recentExperiments.filter { $0.id != runningId }
		return result
	}

	/// Iterations for a given experiment (live for running, cached for historical)
	public func iterations(for experimentId: Int64) -> [Iteration] {
		if experimentId == currentExperiment?.id {
			return liveIterations.sorted { $0.iteration_num > $1.iteration_num }
		}
		return (experimentIterations[experimentId] ?? []).sorted { $0.iteration_num > $1.iteration_num }
	}

	/// Iterations currently shown in chart
	public var chartIterations: [Iteration] {
		guard let chartId = chartExperimentId else {
			return liveIterations
		}
		return iterations(for: chartId)
	}

	/// Name for chart title
	public var chartTitle: String {
		if let chartId = chartExperimentId {
			return displayExperiments.first { $0.id == chartId }?.name ?? "Experiment"
		}
		return "Best So Far"
	}

	public let apiClient: APIClient
	private let wsManager: WebSocketManager
	private var cancellables = Set<AnyCancellable>()

	public init(apiClient: APIClient, wsManager: WebSocketManager) {
		self.apiClient = apiClient
		self.wsManager = wsManager
		wsManager.$snapshot.compactMap { $0 }.receive(on: DispatchQueue.main).sink { [weak self] in self?.snapshot = $0 }.store(in: &cancellables)
		wsManager.addMessageHandler(id: "dashboard") { [weak self] msg in Task { @MainActor in self?.handleMessage(msg) } }
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

	public func loadRecentExperiments() async {
		// If running experiment belongs to a flow, load that flow's experiments
		if let flowId = currentExperiment?.flow_id {
			do {
				currentFlowExperiments = try await apiClient.getFlowExperiments(flowId)
			} catch {
				currentFlowExperiments = []
			}
		} else {
			currentFlowExperiments = []
		}

		// Also load recent experiments as fallback (for when no flow is running)
		do {
			let all = try await apiClient.getExperiments()
			recentExperiments = all
				.filter { $0.status == .completed || $0.status == .failed }
				.sorted { ($0.endedDate ?? $0.createdDate ?? .distantPast) > ($1.endedDate ?? $1.createdDate ?? .distantPast) }
				.prefix(10)
				.map { $0 }
		} catch {
			recentExperiments = []
		}
	}

	/// Load iterations for a historical experiment (lazy, on expand)
	public func loadExperimentIterations(_ experimentId: Int64) async {
		// Don't re-fetch if already cached or if it's the running experiment
		if experimentId == currentExperiment?.id { return }
		if experimentIterations[experimentId] != nil { return }
		do {
			experimentIterations[experimentId] = try await apiClient.getIterations(experimentId: experimentId)
		} catch {
			experimentIterations[experimentId] = []
			self.error = error.localizedDescription
		}
	}

	/// Select an experiment for chart display
	public func selectChartExperiment(_ experimentId: Int64) {
		chartExperimentId = chartExperimentId == experimentId ? nil : experimentId
	}

	public func clearError() { error = nil }
}
