// DashboardView - Main iterations monitoring screen

import SwiftUI
import Charts

public struct DashboardView: View {
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @EnvironmentObject var viewModel: DashboardViewModel
    @EnvironmentObject var wsManager: WebSocketManager

    public init() {}

    public var body: some View {
        NavigationStack {
            Group {
                if horizontalSizeClass == .regular {
                    iPadLayout
                } else {
                    iPhoneLayout
                }
            }
            .navigationTitle("Iterations")
            #if os(iOS)
            .toolbar { ToolbarItem(placement: .navigationBarTrailing) { connectionIndicator } }
            #else
            .toolbar { ToolbarItem(placement: .automatic) { connectionIndicator } }
            #endif
            .refreshable { await viewModel.refresh() }
            .task { await viewModel.loadRecentExperiments() }
            .sheet(item: $viewModel.selectedIteration) { iter in
                IterationDetailSheet(iteration: iter, genomes: viewModel.selectedIterationGenomes)
            }
        }
    }

    // MARK: - iPhone Layout (existing vertical scroll)

    private var iPhoneLayout: some View {
        ScrollView {
            VStack(spacing: 16) {
                headerSection
                metricsSection
                if viewModel.isRunning && viewModel.selectedHistoryExperiment == nil { progressSection }
                if !viewModel.iterations.isEmpty || !viewModel.historyIterations.isEmpty { chartSection }
                if !viewModel.iterations.isEmpty || !viewModel.historyIterations.isEmpty { iterationsSection }
            }
            .padding()
        }
    }

    // MARK: - iPad Layout (side-by-side)

    private var iPadLayout: some View {
        HStack(alignment: .top, spacing: 20) {
            // Left column: Header, Metrics
            VStack(spacing: 16) {
                headerSection
                metricsSection
                if viewModel.isRunning && viewModel.selectedHistoryExperiment == nil { progressSection }
                Spacer()
            }
            .frame(width: LayoutConstants.iPadSidebarWidth)

            // Right column: Chart + Iterations
            VStack(spacing: 16) {
                if !viewModel.iterations.isEmpty || !viewModel.historyIterations.isEmpty {
                    chartSection
                        .frame(height: LayoutConstants.chartHeight(for: horizontalSizeClass))
                }
                if !viewModel.iterations.isEmpty || !viewModel.historyIterations.isEmpty {
                    iPadIterationsSection
                }
            }
        }
        .padding()
    }

    private var iPadIterationsSection: some View {
        let iters = viewModel.selectedHistoryExperiment != nil
            ? viewModel.historyIterations
            : viewModel.iterations
        let title = viewModel.selectedHistoryExperiment != nil
            ? "Experiment Iterations"
            : "Recent Iterations"

        return VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.headline)
            IterationsTableView(iterations: Array(iters.prefix(50))) { iter in
                Task { await viewModel.loadGenomes(for: iter) }
            }
        }
    }

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let exp = viewModel.currentExperiment {
                // Running experiment
                HStack {
                    Text(exp.name).font(.headline)
                    Spacer()
                    StatusBadge(text: exp.status.displayName, color: Theme.statusColor(exp.status))
                }
                if let current = exp.current_iteration, let max = exp.max_iterations {
                    HStack {
                        if let phaseType = exp.phase_type {
                            Text(phaseType).font(.subheadline).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text("Iteration \(current)/\(max)").font(.caption).foregroundStyle(.secondary)
                    }
                }
            } else if let histExp = viewModel.selectedHistoryExperiment {
                // Viewing historical experiment
                HStack {
                    Text("History").font(.caption).fontWeight(.medium).foregroundStyle(.orange).textCase(.uppercase)
                    Text("›").foregroundStyle(.secondary)
                    Text(histExp.name).font(.headline)
                    Spacer()
                    Button { viewModel.clearHistoryExperiment() } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                }
            } else {
                // No experiment running - show selection
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Image(systemName: "pause.circle").foregroundStyle(.secondary)
                        Text("No experiment running").font(.headline).foregroundStyle(.secondary)
                    }

                    if !viewModel.recentExperiments.isEmpty {
                        Text("View completed experiments:").font(.caption).foregroundStyle(.secondary)
                        experimentChipsView
                    }
                }
            }
        }
        .padding()
        .background(
            viewModel.selectedHistoryExperiment != nil
                ? AnyShapeStyle(Color.orange.opacity(0.1))
                : AnyShapeStyle(.ultraThinMaterial),
            in: RoundedRectangle(cornerRadius: 16)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16)
                .stroke(viewModel.selectedHistoryExperiment != nil ? Color.orange.opacity(0.3) : Color.clear, lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.1), radius: 8, x: 0, y: 4)
    }

    private var experimentChipsView: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(viewModel.recentExperiments) { exp in
                    Button {
                        Task { await viewModel.selectHistoryExperiment(exp) }
                    } label: {
                        HStack(spacing: 4) {
                            Text(exp.name)
                                .lineLimit(1)
                                .font(.caption)
                            Image(systemName: exp.status == .completed ? "checkmark.circle.fill" : "xmark.circle.fill")
                                .font(.caption2)
                                .foregroundColor(exp.status == .completed ? .green : .red)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(Color.secondary.opacity(0.15))
                        .cornerRadius(16)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private var metricsSection: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
            MetricsCardView(title: "Best CE", value: NumberFormatters.formatCE(viewModel.bestCE), icon: "arrow.down.circle", color: .blue)
            MetricsCardView(title: "Best Accuracy", value: NumberFormatters.formatAccuracy(viewModel.bestAccuracy), icon: "arrow.up.circle", color: .green)
            MetricsCardView(title: "Type", value: viewModel.currentExperiment?.typePrefix ?? "-", icon: "gearshape.2", color: .purple)
            MetricsCardView(title: "Iteration", value: "\(viewModel.currentIterationNumber)", icon: "number", color: .orange)
        }
    }

    private var progressSection: some View {
        ProgressBarView(progress: viewModel.experimentProgress, current: Int(viewModel.currentIterationNumber), max: Int(viewModel.maxIterations))
    }

    private var chartSection: some View {
        let iterations: [Iteration]
        let title: String

        if let histExp = viewModel.selectedHistoryExperiment {
            iterations = viewModel.historyIterations
            title = histExp.name
        } else {
            iterations = viewModel.iterations
            title = "Best So Far"
        }

        return DualAxisChartView(iterations: iterations, title: title)
            .frame(height: 280)
    }

    private var iterationsSection: some View {
        let iters = viewModel.selectedHistoryExperiment != nil
            ? viewModel.historyIterations
            : viewModel.iterations
        let title = viewModel.selectedHistoryExperiment != nil
            ? "Experiment Iterations"
            : "Recent Iterations"

        return VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.headline).padding(.horizontal)
            IterationsListView(iterations: Array(iters.prefix(50))) { iter in
                Task { await viewModel.loadGenomes(for: iter) }
            }
        }
    }

    private var connectionIndicator: some View {
        HStack(spacing: 4) {
            Circle().fill(wsManager.connectionState.isConnected ? Color.green : Color.red).frame(width: 8, height: 8)
            Text(wsManager.connectionState.isConnected ? "Live" : "Offline").font(.caption).foregroundColor(.secondary)
        }
    }
}
