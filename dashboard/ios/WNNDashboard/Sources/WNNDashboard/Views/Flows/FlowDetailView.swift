// FlowDetailView - Detailed view of a single flow

import SwiftUI

public struct FlowDetailView: View {
    public let flow: Flow

    @EnvironmentObject var viewModel: FlowsViewModel
    @Environment(\.dismiss) private var dismiss
    @State private var navigationPath = NavigationPath()

    public init(flow: Flow) { self.flow = flow }

    public var body: some View {
        NavigationStack(path: $navigationPath) {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    statusSection
                    configSection
                    experimentsSection
                    timingSection
                }
                .padding()
            }
            .navigationTitle(flow.name)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .navigationBarTrailing) { Button("Done") { dismiss() } } }
            .navigationDestination(for: Experiment.self) { experiment in
                ExperimentIterationsView(experiment: experiment)
            }
        }
    }

    private var statusSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack { Text("Status").font(.headline); Spacer(); StatusBadge(text: flow.status.displayName, color: Theme.statusColor(flow.status)) }
            if let desc = flow.description { Text(desc).font(.subheadline).foregroundStyle(.secondary) }
            if let pid = flow.pid, flow.status == .running { HStack { Text("Process ID").foregroundStyle(.secondary); Spacer(); Text("\(pid)").fontDesign(.monospaced) }.font(.subheadline) }
        }
        .padding()
        .glassCard()
    }

    private var configSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Configuration").font(.headline)
            if let t = flow.config.template { configRow("Template", value: t) }
            configRow("Experiments", value: "\(flow.config.experiments.count)")
            if let c = flow.seed_checkpoint_id { configRow("Seed Checkpoint", value: "#\(c)") }
        }
        .padding()
        .glassCard()
    }

    private var experimentsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack { Text("Experiment Specs").font(.headline); Spacer(); Text("\(flow.config.experiments.count)").font(.caption).foregroundStyle(.secondary) }
            ForEach(Array(flow.config.experiments.enumerated()), id: \.offset) { i, spec in ExperimentSpecRow(index: i + 1, spec: spec) }
            if !viewModel.selectedFlowExperiments.isEmpty {
                Divider().padding(.vertical, 8)
                HStack {
                    Text("Experiments").font(.subheadline).foregroundStyle(.secondary)
                    Spacer()
                    Text("Tap to view iterations").font(.caption2).foregroundStyle(.tertiary)
                }
                ForEach(viewModel.selectedFlowExperiments) { exp in
                    ExperimentRow(experiment: exp, isNavigable: exp.status == .completed || exp.status == .running)
                        .onTapGesture {
                            if exp.status == .completed || exp.status == .running {
                                navigationPath.append(exp)
                            }
                        }
                }
            }
        }
        .padding()
        .glassCard()
    }

    private var timingSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Timing").font(.headline)
            if let d = flow.createdDate { configRow("Created", value: DateFormatters.shortDateTime(d)) }
            if let d = flow.startedDate { configRow("Started", value: DateFormatters.shortDateTime(d)) }
            if let d = flow.completedDate { configRow("Completed", value: DateFormatters.shortDateTime(d)) }
            if let dur = flow.duration { configRow("Duration", value: DateFormatters.duration(dur)) }
        }
        .padding()
        .glassCard()
    }

    private func configRow(_ label: String, value: String) -> some View {
        HStack { Text(label).foregroundStyle(.secondary); Spacer(); Text(value).fontDesign(.monospaced) }.font(.subheadline)
    }
}

private struct ExperimentSpecRow: View {
    let index: Int; let spec: ExperimentSpec
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("\(index). \(spec.name)").font(.subheadline).fontWeight(.medium)
                Spacer()
                Text(spec.experiment_type.displayName).font(.caption).padding(.horizontal, 6).padding(.vertical, 2).background(spec.experiment_type == .ga ? Color.blue.opacity(0.2) : Color.purple.opacity(0.2)).foregroundStyle(spec.experiment_type == .ga ? .blue : .purple).clipShape(RoundedRectangle(cornerRadius: 4))
            }
            HStack(spacing: 8) {
                if spec.optimize_neurons { badge("Neurons") }
                if spec.optimize_bits { badge("Bits") }
                if spec.optimize_connections { badge("Connections") }
            }
        }
        .padding()
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }
    private func badge(_ text: String) -> some View { Text(text).font(.caption2).padding(.horizontal, 6).padding(.vertical, 2).background(Color.green.opacity(0.2)).foregroundStyle(.green).clipShape(RoundedRectangle(cornerRadius: 4)) }
}

struct ExperimentRow: View {
    let experiment: Experiment
    var isNavigable: Bool = false

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(experiment.name).font(.subheadline)
                if let iter = experiment.last_iteration { Text("Iteration \(iter)").font(.caption).foregroundStyle(.secondary) }
            }
            Spacer()
            StatusBadge(text: experiment.status.displayName, color: Theme.statusColor(experiment.status))
            if isNavigable {
                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 12)
        .background(isNavigable ? Color.secondary.opacity(0.08) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .contentShape(Rectangle())
        .opacity(isNavigable ? 1.0 : 0.7)
    }
}
