// FlowDetailView - Detailed view of a single flow

import SwiftUI

struct FlowDetailView: View {
    let flow: Flow

    @EnvironmentObject var viewModel: FlowsViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Status header
                    statusSection

                    // Config section
                    configSection

                    // Experiments section
                    experimentsSection

                    // Timing section
                    timingSection
                }
                .padding()
            }
            .navigationTitle(flow.name)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    // MARK: - Sections

    private var statusSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Status")
                    .font(.headline)

                Spacer()

                StatusBadge(
                    text: flow.status.displayName,
                    color: Theme.statusColor(flow.status)
                )
            }

            if let description = flow.description {
                Text(description)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }

            // PID if running
            if let pid = flow.pid, flow.status == .running {
                HStack {
                    Text("Process ID")
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("\(pid)")
                        .fontDesign(.monospaced)
                }
                .font(.subheadline)
            }
        }
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(12)
    }

    private var configSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Configuration")
                .font(.headline)

            // Template
            if let template = flow.config.template {
                configRow("Template", value: template)
            }

            // Experiments count
            configRow("Experiments", value: "\(flow.config.experiments.count)")

            // Seed checkpoint
            if let checkpointId = flow.seed_checkpoint_id {
                configRow("Seed Checkpoint", value: "#\(checkpointId)")
            }

            // Additional params
            if !flow.config.params.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Parameters")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    ForEach(Array(flow.config.params.keys.sorted()), id: \.self) { key in
                        if let value = flow.config.params[key] {
                            HStack {
                                Text(key)
                                    .font(.caption)
                                Spacer()
                                Text("\(String(describing: value.value))")
                                    .font(.caption)
                                    .fontDesign(.monospaced)
                            }
                        }
                    }
                }
                .padding(.top, 4)
            }
        }
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(12)
    }

    private var experimentsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Experiment Specs")
                    .font(.headline)

                Spacer()

                Text("\(flow.config.experiments.count)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            ForEach(Array(flow.config.experiments.enumerated()), id: \.offset) { index, spec in
                ExperimentSpecRow(index: index + 1, spec: spec)
            }

            // Actual experiments from backend
            if !viewModel.selectedFlowExperiments.isEmpty {
                Divider()
                    .padding(.vertical, 8)

                Text("Completed Experiments")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                ForEach(viewModel.selectedFlowExperiments) { experiment in
                    ExperimentRow(experiment: experiment)
                }
            }
        }
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(12)
    }

    private var timingSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Timing")
                .font(.headline)

            if let created = flow.createdDate {
                configRow("Created", value: DateFormatters.shortDateTime(created))
            }

            if let started = flow.startedDate {
                configRow("Started", value: DateFormatters.shortDateTime(started))
            }

            if let completed = flow.completedDate {
                configRow("Completed", value: DateFormatters.shortDateTime(completed))
            }

            if let duration = flow.duration {
                configRow("Duration", value: DateFormatters.duration(duration))
            }
        }
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(12)
    }

    // MARK: - Helpers

    private func configRow(_ label: String, value: String) -> some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .fontDesign(.monospaced)
        }
        .font(.subheadline)
    }
}

// MARK: - Supporting Views

private struct ExperimentSpecRow: View {
    let index: Int
    let spec: ExperimentSpec

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("\(index). \(spec.name)")
                    .font(.subheadline)
                    .fontWeight(.medium)

                Spacer()

                Text(spec.experiment_type.displayName)
                    .font(.caption)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(spec.experiment_type == .ga ? Color.blue.opacity(0.2) : Color.purple.opacity(0.2))
                    .foregroundColor(spec.experiment_type == .ga ? .blue : .purple)
                    .cornerRadius(4)
            }

            // Optimization flags
            HStack(spacing: 8) {
                if spec.optimize_neurons {
                    optimizationBadge("Neurons")
                }
                if spec.optimize_bits {
                    optimizationBadge("Bits")
                }
                if spec.optimize_connections {
                    optimizationBadge("Connections")
                }
            }
        }
        .padding()
        .background(Color.gray.opacity(0.1))
        .cornerRadius(8)
    }

    private func optimizationBadge(_ text: String) -> some View {
        Text(text)
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Color.green.opacity(0.2))
            .foregroundColor(.green)
            .cornerRadius(4)
    }
}

private struct ExperimentRow: View {
    let experiment: Experiment

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(experiment.name)
                    .font(.subheadline)

                if let iteration = experiment.last_iteration {
                    Text("Iteration \(iteration)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            Spacer()

            StatusBadge(
                text: experiment.status.displayName,
                color: Theme.statusColor(experiment.status)
            )
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    let sampleFlow = Flow(
        id: 1,
        name: "4ngram Baseline",
        description: "Testing asymmetric tier configuration with extended optimization",
        config: FlowConfig(
            experiments: [
                ExperimentSpec(name: "GA Neurons", experiment_type: .ga, optimize_neurons: true),
                ExperimentSpec(name: "TS Neurons", experiment_type: .ts, optimize_neurons: true),
                ExperimentSpec(name: "GA Bits", experiment_type: .ga, optimize_bits: true),
            ],
            template: "standard-6-phase",
            params: ["population": AnyCodable(50), "patience": AnyCodable(10)]
        ),
        created_at: "2026-01-31T18:45:23Z",
        started_at: "2026-01-31T18:47:12Z",
        completed_at: nil,
        status: .running,
        seed_checkpoint_id: nil,
        pid: 12345
    )

    FlowDetailView(flow: sampleFlow)
        .environmentObject(FlowsViewModel(
            apiClient: APIClient(connectionManager: ConnectionManager(settings: SettingsStore())),
            wsManager: WebSocketManager(connectionManager: ConnectionManager(settings: SettingsStore()))
        ))
}
