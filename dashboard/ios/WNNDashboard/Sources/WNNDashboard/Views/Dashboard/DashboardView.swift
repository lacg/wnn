// DashboardView - Main live monitoring screen

import SwiftUI
import Charts

public struct DashboardView: View {
    @EnvironmentObject var viewModel: DashboardViewModel
    @EnvironmentObject var wsManager: WebSocketManager

    public init() {}

    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    headerSection
                    metricsSection
                    if viewModel.isRunning { progressSection }
                    if !viewModel.iterations.isEmpty { chartSection }
                    if !viewModel.phases.isEmpty { phaseTimelineSection }
                    if !viewModel.iterations.isEmpty { iterationsSection }
                }
                .padding()
            }
            .navigationTitle("Dashboard")
            .toolbar { ToolbarItem(placement: .navigationBarTrailing) { connectionIndicator } }
            .refreshable { await viewModel.refresh() }
            .sheet(item: $viewModel.selectedIteration) { iter in
                IterationDetailSheet(iteration: iter, genomes: viewModel.selectedIterationGenomes)
            }
        }
    }

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let exp = viewModel.currentExperiment {
                HStack {
                    Text(exp.name).font(.headline)
                    Spacer()
                    StatusBadge(text: exp.status.displayName, color: Theme.statusColor(exp.status))
                }
                if let phase = viewModel.currentPhase {
                    HStack {
                        Text(phase.name).font(.subheadline).foregroundColor(.secondary)
                        Spacer()
                        Text("Iteration \(phase.current_iteration)/\(phase.max_iterations)").font(.caption).foregroundColor(.secondary)
                    }
                }
            } else {
                Text("No experiment running").font(.headline).foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(12)
    }

    private var metricsSection: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
            MetricsCardView(title: "Best CE", value: NumberFormatters.formatCE(viewModel.bestCE), icon: "arrow.down.circle", color: .blue)
            MetricsCardView(title: "Best Accuracy", value: NumberFormatters.formatAccuracy(viewModel.bestAccuracy), icon: "arrow.up.circle", color: .green)
            MetricsCardView(title: "Phase", value: "\(viewModel.phases.filter { $0.status == .completed }.count + 1)/\(max(viewModel.phases.count, 1))", icon: "clock.badge.checkmark", color: .purple)
            MetricsCardView(title: "Iteration", value: "\(viewModel.currentIterationNumber)", icon: "number", color: .orange)
        }
    }

    private var progressSection: some View {
        ProgressBarView(progress: viewModel.phaseProgress, current: Int(viewModel.currentIterationNumber), max: Int(viewModel.maxIterations))
    }

    private var chartSection: some View {
        DualAxisChartView(iterations: viewModel.iterations)
            .frame(height: 200)
            .padding()
            .background(Theme.cardBackground)
            .cornerRadius(12)
    }

    private var phaseTimelineSection: some View {
        PhaseTimelineView(phases: viewModel.phases, currentPhase: viewModel.currentPhase)
    }

    private var iterationsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Recent Iterations").font(.headline).padding(.horizontal)
            IterationsListView(iterations: Array(viewModel.iterations.prefix(20))) { iter in
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
