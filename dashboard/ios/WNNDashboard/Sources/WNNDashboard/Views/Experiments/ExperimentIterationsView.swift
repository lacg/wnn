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
    @State private var flowExperiments: [Experiment] = []
    @State private var flowValidations: [ValidationSummary] = []
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
                if !validationProgression.isEmpty {
                    validationProgressionSection
                }
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
            // Left: Header, Metrics, Validation, Gating
            VStack(spacing: 16) {
                experimentHeader
                metricsSection
                if !validationProgression.isEmpty {
                    validationProgressionSection
                }
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

    // MARK: - Validation Progression

    private struct ValidationProgressionPoint: Identifiable {
        let id = UUID()
        let label: String
        let expId: Int64
        let sequenceOrder: Int32
        let validationPoint: ValidationPoint
        let summaries: [ValidationSummary]

        var bestCE: ValidationSummary? { summaries.first { $0.genome_type == .best_ce } }
        var bestAcc: ValidationSummary? { summaries.first { $0.genome_type == .best_acc } }
        var bestFitness: ValidationSummary? { summaries.first { $0.genome_type == .best_fitness } }
    }

    private var validationProgression: [ValidationProgressionPoint] {
        guard !flowValidations.isEmpty, !flowExperiments.isEmpty else { return [] }

        let currentSeqOrder = experiment.sequence_order ?? 0
        let expMap = Dictionary(uniqueKeysWithValues: flowExperiments.map { ($0.id, $0) })

        // Filter to only include experiments up to current
        let relevantValidations = flowValidations.filter { v in
            guard let exp = expMap[v.experiment_id] else { return false }
            return (exp.sequence_order ?? 0) <= currentSeqOrder
        }

        // Group by (experiment_id, validation_point)
        var grouped: [String: [ValidationSummary]] = [:]
        for v in relevantValidations {
            let key = "\(v.experiment_id)-\(v.validation_point.rawValue)"
            grouped[key, default: []].append(v)
        }

        // Convert to progression points
        var points: [ValidationProgressionPoint] = []
        for (key, validations) in grouped {
            let parts = key.split(separator: "-")
            guard parts.count == 2,
                  let expId = Int64(parts[0]),
                  let exp = expMap[expId] else { continue }

            let point = parts[1] == "init" ? ValidationPoint.`init` : ValidationPoint.final
            let seqOrder = exp.sequence_order ?? 0

            // Label: "Init" for first init, otherwise phase name
            let label: String
            if point == .`init` && seqOrder == 0 {
                label = "Init"
            } else if point == .`init` {
                continue // Skip non-first init points
            } else {
                label = exp.name.replacingOccurrences(of: #"^Phase \d+[ab]: "#, with: "", options: .regularExpression)
            }

            points.append(ValidationProgressionPoint(
                label: label,
                expId: expId,
                sequenceOrder: seqOrder,
                validationPoint: point,
                summaries: validations
            ))
        }

        // Sort by sequence order, then init before final
        return points.sorted { a, b in
            if a.sequenceOrder != b.sequenceOrder { return a.sequenceOrder < b.sequenceOrder }
            return a.validationPoint == .`init`
        }
    }

    private var validationProgressionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("📈 Validation Progression")
                    .font(.headline)
                Spacer()
                Text("\(validationProgression.count) checkpoints")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            ScrollView(.horizontal, showsIndicators: false) {
                VStack(spacing: 0) {
                    // Header row 1: Genome type groups
                    HStack(spacing: 0) {
                        Text("Phase")
                            .frame(width: 70, alignment: .leading)
                        Spacer().frame(width: 8)
                        Text("Best CE")
                            .frame(width: 90, alignment: .center)
                            .foregroundStyle(.blue)
                        Spacer().frame(width: 8)
                        Text("Best Acc")
                            .frame(width: 90, alignment: .center)
                            .foregroundStyle(.green)
                        Spacer().frame(width: 8)
                        Text("Best Fit")
                            .frame(width: 90, alignment: .center)
                            .foregroundStyle(.purple)
                    }
                    .font(.caption2.bold())
                    .padding(.vertical, 4)

                    // Header row 2: CE/Acc sub-columns
                    HStack(spacing: 0) {
                        Text("")
                            .frame(width: 70)
                        Spacer().frame(width: 8)
                        HStack(spacing: 4) {
                            Text("CE").frame(width: 43, alignment: .trailing)
                            Text("Acc").frame(width: 43, alignment: .trailing)
                        }
                        .frame(width: 90)
                        Spacer().frame(width: 8)
                        HStack(spacing: 4) {
                            Text("CE").frame(width: 43, alignment: .trailing)
                            Text("Acc").frame(width: 43, alignment: .trailing)
                        }
                        .frame(width: 90)
                        Spacer().frame(width: 8)
                        HStack(spacing: 4) {
                            Text("CE").frame(width: 43, alignment: .trailing)
                            Text("Acc").frame(width: 43, alignment: .trailing)
                        }
                        .frame(width: 90)
                    }
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 2)

                    Divider()

                    // Data rows
                    ForEach(Array(validationProgression.enumerated()), id: \.element.id) { idx, point in
                        let isCurrentPhase = point.expId == experiment.id && point.validationPoint == .final

                        HStack(spacing: 0) {
                            HStack(spacing: 2) {
                                Text(point.label)
                                    .lineLimit(1)
                                if isCurrentPhase {
                                    Text("◀")
                                        .foregroundStyle(.blue)
                                        .font(.caption2)
                                }
                            }
                            .frame(width: 70, alignment: .leading)
                            .font(point.validationPoint == .`init` ? .caption.italic() : .caption)
                            .foregroundStyle(point.validationPoint == .`init` ? .blue : .primary)

                            Spacer().frame(width: 8)

                            // Best CE genome
                            HStack(spacing: 4) {
                                Text(formatCEShort(point.bestCE?.ce))
                                    .frame(width: 43, alignment: .trailing)
                                Text(formatAccShort(point.bestCE?.accuracy))
                                    .frame(width: 43, alignment: .trailing)
                                    .foregroundStyle(.secondary)
                            }
                            .frame(width: 90)
                            .background(Color.blue.opacity(0.05))

                            Spacer().frame(width: 8)

                            // Best Acc genome
                            HStack(spacing: 4) {
                                Text(formatCEShort(point.bestAcc?.ce))
                                    .frame(width: 43, alignment: .trailing)
                                Text(formatAccShort(point.bestAcc?.accuracy))
                                    .frame(width: 43, alignment: .trailing)
                                    .foregroundStyle(.secondary)
                            }
                            .frame(width: 90)
                            .background(Color.green.opacity(0.05))

                            Spacer().frame(width: 8)

                            // Best Fitness genome
                            HStack(spacing: 4) {
                                Text(formatCEShort(point.bestFitness?.ce))
                                    .frame(width: 43, alignment: .trailing)
                                Text(formatAccShort(point.bestFitness?.accuracy))
                                    .frame(width: 43, alignment: .trailing)
                                    .foregroundStyle(.secondary)
                            }
                            .frame(width: 90)
                            .background(Color.purple.opacity(0.05))
                        }
                        .font(.caption)
                        .fontDesign(.monospaced)
                        .padding(.vertical, 6)
                        .background(isCurrentPhase ? Color.blue.opacity(0.1) : Color.clear)

                        if idx < validationProgression.count - 1 {
                            Divider()
                        }
                    }
                }
                .padding(8)
            }
            #if os(iOS)
            .background(Color(.secondarySystemBackground).opacity(0.5))
            #else
            .background(Color.secondary.opacity(0.1))
            #endif
            .cornerRadius(8)
        }
        .padding()
        .glassCard()
    }

    private func formatCEShort(_ ce: Double?) -> String {
        guard let ce = ce else { return "—" }
        return String(format: "%.2f", ce)
    }

    private func formatAccShort(_ acc: Double?) -> String {
        guard let acc = acc else { return "—" }
        return String(format: "%.2f%%", acc * 100)
    }

    // MARK: - Helpers

    private func loadData() async {
        isLoading = true
        do {
            async let iterationsTask = dashboardViewModel.apiClient.getIterations(experimentId: experiment.id)
            async let gatingTask = dashboardViewModel.apiClient.getGatingRuns(experimentId: experiment.id)

            iterations = try await iterationsTask.sorted { $0.iteration_num > $1.iteration_num }
            gatingRuns = try await gatingTask.sorted { $0.id > $1.id }

            // Load flow validations if experiment belongs to a flow
            if let flowId = experiment.flow_id {
                async let flowExpsTask = dashboardViewModel.apiClient.getFlowExperiments(flowId)
                async let flowValsTask = dashboardViewModel.apiClient.getFlowValidations(flowId: flowId)

                flowExperiments = try await flowExpsTask
                flowValidations = try await flowValsTask
            }
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
