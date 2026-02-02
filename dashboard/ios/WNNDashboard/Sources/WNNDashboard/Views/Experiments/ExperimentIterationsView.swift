// ExperimentIterationsView - Shows iterations for a selected experiment

import SwiftUI
import Charts

public struct ExperimentIterationsView: View {
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @EnvironmentObject var dashboardViewModel: DashboardViewModel

    public let experiment: Experiment

    @State private var iterations: [Iteration] = []
    @State private var phases: [Phase] = []
    @State private var isLoading = true
    @State private var selectedPhase: Phase?

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
                if !phases.isEmpty { phaseSection }
                iterationsSection
            }
            .padding()
        }
    }

    // MARK: - iPad Layout

    private var iPadLayout: some View {
        HStack(alignment: .top, spacing: 20) {
            // Left: Header, Metrics, Phases
            VStack(spacing: 16) {
                experimentHeader
                metricsSection
                if !phases.isEmpty { phaseSection }
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
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding()
        .glassCard()
    }

    private var metricsSection: some View {
        let bestCE = displayedIterations.map(\.best_ce).min()
        let bestAcc = displayedIterations.compactMap(\.best_accuracy).max()

        return LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
            MetricsCardView(title: "Best CE", value: NumberFormatters.formatCE(bestCE), icon: "arrow.down.circle", color: .blue)
            MetricsCardView(title: "Best Accuracy", value: NumberFormatters.formatAccuracy(bestAcc), icon: "arrow.up.circle", color: .green)
            MetricsCardView(title: "Phases", value: "\(phases.filter { $0.status == .completed }.count)/\(phases.count)", icon: "clock.badge.checkmark", color: .purple)
            MetricsCardView(title: "Iterations", value: "\(displayedIterations.count)", icon: "number", color: .orange)
        }
    }

    private var chartSection: some View {
        DualAxisChartView(iterations: displayedIterations, title: selectedPhase?.shortName ?? "All Phases")
            .frame(height: horizontalSizeClass == .regular ? nil : 280)
    }

    private var phaseSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Phases").font(.headline)
                Spacer()
                if selectedPhase != nil {
                    Button("Show All") { selectedPhase = nil }
                        .font(.caption)
                        .foregroundColor(.blue)
                }
            }

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 100))], spacing: 8) {
                ForEach(phases.sorted { $0.sequence_order < $1.sequence_order }) { phase in
                    PhaseChip(phase: phase, isSelected: selectedPhase?.id == phase.id)
                        .onTapGesture {
                            if phase.status == .completed || phase.status == .running {
                                selectedPhase = selectedPhase?.id == phase.id ? nil : phase
                            }
                        }
                }
            }
        }
        .padding()
        .glassCard()
    }

    private var iterationsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(selectedPhase != nil ? "Phase Iterations" : "All Iterations")
                .font(.headline)
                .padding(.horizontal)
            IterationsListView(iterations: Array(displayedIterations.prefix(50))) { iter in
                Task { await dashboardViewModel.loadGenomes(for: iter) }
            }
        }
    }

    private var iPadIterationsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(selectedPhase != nil ? "Phase Iterations" : "All Iterations")
                .font(.headline)
            IterationsTableView(iterations: Array(displayedIterations.prefix(100))) { iter in
                Task { await dashboardViewModel.loadGenomes(for: iter) }
            }
        }
    }

    // MARK: - Helpers

    private var displayedIterations: [Iteration] {
        if let phase = selectedPhase {
            return iterations.filter { $0.phase_id == phase.id }
        }
        return iterations
    }

    private func loadData() async {
        isLoading = true
        do {
            async let phasesTask = dashboardViewModel.apiClient.getPhases(experimentId: experiment.id)
            async let iterationsTask = dashboardViewModel.apiClient.getIterations(experimentId: experiment.id)
            phases = try await phasesTask
            iterations = try await iterationsTask.sorted { $0.iteration_num > $1.iteration_num }
        } catch {
            print("Failed to load experiment data: \(error)")
        }
        isLoading = false
    }
}

// MARK: - Phase Chip

private struct PhaseChip: View {
    let phase: Phase
    let isSelected: Bool

    var body: some View {
        VStack(spacing: 4) {
            Text(phase.shortName)
                .font(.caption)
                .fontWeight(.medium)
            if let ce = phase.best_ce {
                Text(NumberFormatters.formatCE(ce))
                    .font(.caption2)
                    .fontDesign(.monospaced)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(isSelected ? Color.blue.opacity(0.2) : Color.secondary.opacity(0.1))
        .foregroundColor(isSelected ? .blue : .primary)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isSelected ? Color.blue : Color.clear, lineWidth: 2)
        )
        .opacity(phase.status == .completed || phase.status == .running ? 1.0 : 0.5)
    }
}
