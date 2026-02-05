// ExperimentIterationsView - Shows iterations for a selected experiment

import SwiftUI
import Charts

public struct ExperimentIterationsView: View {
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @EnvironmentObject var dashboardViewModel: DashboardViewModel
    @EnvironmentObject var webSocketManager: WebSocketManager

    public let experiment: Experiment

    @State private var iterations: [Iteration] = []
    @State private var gatingRuns: [GatingRun] = []
    @State private var isLoading = true
    @State private var isCreatingGatingRun = false

    public init(experiment: Experiment) {
        self.experiment = experiment
    }

    public var body: some View {
        Group {
            if isLoading {
                ProgressView("Loading iterations...")
            } else if iterations.isEmpty {
                ContentUnavailableView("No Iterations", systemImage: "chart.line.uptrend.xyaxis", description: Text("This experiment has no recorded iterations yet"))
            } else {
                content
            }
        }
        .navigationTitle(experiment.name)
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #endif
        .task { await loadData() }
        .onReceive(webSocketManager.$lastMessage) { message in
            guard let message = message else { return }
            handleWebSocketMessage(message)
        }
    }

    private func handleWebSocketMessage(_ message: WsMessage) {
        switch message {
        case .gatingRunCreated(let run) where run.experiment_id == experiment.id:
            if !gatingRuns.contains(where: { $0.id == run.id }) {
                gatingRuns.insert(run, at: 0)
            }
        case .gatingRunUpdated(let run) where run.experiment_id == experiment.id:
            if let index = gatingRuns.firstIndex(where: { $0.id == run.id }) {
                gatingRuns[index] = run
            }
        default:
            break
        }
    }

    @ViewBuilder
    private var content: some View {
        if horizontalSizeClass == .regular {
            iPadLayout
        } else {
            iPhoneLayout
        }
    }

    // MARK: - iPhone Layout

    private var iPhoneLayout: some View {
        ScrollView {
            VStack(spacing: 16) {
                experimentHeader
                metricsSection
                if experiment.status == .completed {
                    gatingSection
                }
                chartSection
                iterationsSection
            }
            .padding()
        }
    }

    // MARK: - iPad Layout

    private var iPadLayout: some View {
        HStack(alignment: .top, spacing: 20) {
            // Left: Header, Metrics, Gating
            VStack(spacing: 16) {
                experimentHeader
                metricsSection
                if experiment.status == .completed {
                    gatingSection
                }
                Spacer()
            }
            .frame(width: LayoutConstants.iPadSidebarWidth)

            // Right: Chart + Iterations
            VStack(spacing: 16) {
                chartSection
                    .frame(height: LayoutConstants.chartHeight(for: horizontalSizeClass))
                iPadIterationsSection
            }
        }
        .padding()
    }

    // MARK: - Sections

    private var experimentHeader: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(experiment.name)
                    .font(.headline)
                Spacer()
                StatusBadge(text: experiment.status.displayName, color: Theme.statusColor(experiment.status))
            }
            if let tierConfig = experiment.tier_config {
                Text("Tier: \(tierConfig)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 16) {
                Label("Context: \(experiment.context_size)", systemImage: "number")
                Label("Pop: \(experiment.population_size)", systemImage: "person.3")
                if let phaseType = experiment.phase_type {
                    Label(phaseType, systemImage: "gearshape.2")
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding()
        .glassCard()
    }

    private var metricsSection: some View {
        let bestCE = iterations.map(\.best_ce).min()
        let bestAcc = iterations.compactMap(\.best_accuracy).max()

        return LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
            MetricsCardView(title: "Best CE", value: NumberFormatters.formatCE(bestCE), icon: "arrow.down.circle", color: .blue)
            MetricsCardView(title: "Best Accuracy", value: NumberFormatters.formatAccuracy(bestAcc), icon: "arrow.up.circle", color: .green)
            MetricsCardView(title: "Type", value: experiment.typePrefix.isEmpty ? "-" : experiment.typePrefix, icon: "gearshape.2", color: .purple)
            MetricsCardView(title: "Iterations", value: "\(iterations.count)", icon: "number", color: .orange)
        }
    }

    private var chartSection: some View {
        DualAxisChartView(iterations: iterations, title: experiment.name)
            .frame(height: horizontalSizeClass == .regular ? nil : 280)
    }

    private var iterationsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Iterations")
                .font(.headline)
                .padding(.horizontal)
            IterationsListView(iterations: Array(iterations.prefix(50))) { iter in
                Task { await dashboardViewModel.loadGenomes(for: iter) }
            }
        }
    }

    private var iPadIterationsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Iterations")
                .font(.headline)
            IterationsTableView(iterations: Array(iterations.prefix(100))) { iter in
                Task { await dashboardViewModel.loadGenomes(for: iter) }
            }
        }
    }

    // MARK: - Gating Section

    private var gatingSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Gating Analysis")
                    .font(.headline)
                Spacer()
                if !hasActiveGatingRun {
                    Button {
                        Task { await createGatingRun() }
                    } label: {
                        if isCreatingGatingRun {
                            ProgressView()
                                .controlSize(.small)
                        } else {
                            Label("Run", systemImage: "play.fill")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    .disabled(isCreatingGatingRun)
                }
            }

            if gatingRuns.isEmpty {
                Text("No gating analysis runs yet")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(gatingRuns) { run in
                    GatingRunRow(run: run)
                }
            }
        }
        .padding()
        .glassCard()
    }

    private var hasActiveGatingRun: Bool {
        gatingRuns.contains { $0.status.isActive }
    }

    // MARK: - Helpers

    private func loadData() async {
        isLoading = true
        do {
            async let iterationsTask = dashboardViewModel.apiClient.getIterations(experimentId: experiment.id)
            async let gatingTask = dashboardViewModel.apiClient.getGatingRuns(experimentId: experiment.id)

            iterations = try await iterationsTask.sorted { $0.iteration_num > $1.iteration_num }
            gatingRuns = try await gatingTask.sorted { $0.id > $1.id }
        } catch {
            print("Failed to load experiment data: \(error)")
        }
        isLoading = false
    }

    private func createGatingRun() async {
        isCreatingGatingRun = true
        do {
            let run = try await dashboardViewModel.apiClient.createGatingRun(experimentId: experiment.id)
            gatingRuns.insert(run, at: 0)
        } catch {
            print("Failed to create gating run: \(error)")
        }
        isCreatingGatingRun = false
    }
}

// MARK: - Gating Run Row

private struct GatingRunRow: View {
    let run: GatingRun

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                StatusBadge(text: run.status.displayName, color: statusColor)
                Spacer()
                if let date = run.createdDate {
                    Text(date, style: .relative)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            if let results = run.results, !results.isEmpty {
                gatingResultsTable(results)
            } else if run.status == .running {
                HStack {
                    ProgressView()
                        .controlSize(.small)
                    Text("Running gating analysis...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else if let error = run.error {
                Text("Error: \(error)")
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
        .padding(8)
        #if os(iOS)
        .background(Color(.secondarySystemBackground).opacity(0.5))
        #else
        .background(Color.secondary.opacity(0.1))
        #endif
        .cornerRadius(8)
    }

    private var statusColor: Color {
        switch run.status {
        case .pending: return .orange
        case .running: return .blue
        case .completed: return .green
        case .failed: return .red
        }
    }

    @ViewBuilder
    private func gatingResultsTable(_ results: [GatingResult]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            // Header
            HStack {
                Text("Genome")
                    .frame(width: 80, alignment: .leading)
                Text("CE")
                    .frame(width: 50, alignment: .trailing)
                Text("→")
                    .frame(width: 20)
                Text("Gated")
                    .frame(width: 50, alignment: .trailing)
                Text("Δ")
                    .frame(width: 45, alignment: .trailing)
            }
            .font(.caption2.bold())
            .foregroundStyle(.secondary)

            Divider()

            // Results
            ForEach(results) { result in
                HStack {
                    Text(result.genomeTypeDisplay)
                        .frame(width: 80, alignment: .leading)
                    Text(NumberFormatters.formatCE(result.ce))
                        .frame(width: 50, alignment: .trailing)
                    Text("→")
                        .frame(width: 20)
                        .foregroundStyle(.secondary)
                    Text(NumberFormatters.formatCE(result.gated_ce))
                        .frame(width: 50, alignment: .trailing)
                    Text(String(format: "%+.1f%%", -result.ceImprovement))
                        .frame(width: 45, alignment: .trailing)
                        .foregroundStyle(result.ceImprovement > 0 ? .green : .red)
                }
                .font(.caption)
            }
        }
    }
}
