// ExperimentIterationsView - Shows iterations for a selected experiment

import SwiftUI
import Charts

public struct ExperimentIterationsView: View {
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @EnvironmentObject var dashboardViewModel: DashboardViewModel

    public let experiment: Experiment

    @State private var iterations: [Iteration] = []
    @State private var isLoading = true

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
        .navigationBarTitleDisplayMode(.inline)
        .task { await loadData() }
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
                chartSection
                iterationsSection
            }
            .padding()
        }
    }

    // MARK: - iPad Layout

    private var iPadLayout: some View {
        HStack(alignment: .top, spacing: 20) {
            // Left: Header, Metrics
            VStack(spacing: 16) {
                experimentHeader
                metricsSection
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

    // MARK: - Helpers

    private func loadData() async {
        isLoading = true
        do {
            iterations = try await dashboardViewModel.apiClient.getIterations(experimentId: experiment.id)
                .sorted { $0.iteration_num > $1.iteration_num }
        } catch {
            print("Failed to load experiment data: \(error)")
        }
        isLoading = false
    }
}
