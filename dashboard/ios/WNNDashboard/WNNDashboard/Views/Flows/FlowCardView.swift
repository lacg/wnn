// FlowCardView - Summary card for a single flow

import SwiftUI

struct FlowCardView: View {
    let flow: Flow
    let onSelect: () -> Void

    @EnvironmentObject var viewModel: FlowsViewModel
    @State private var showingStopConfirmation = false
    @State private var showingDeleteConfirmation = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(flow.name)
                        .font(.headline)

                    if let description = flow.description {
                        Text(description)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                    }
                }

                Spacer()

                StatusBadge(
                    text: flow.status.displayName,
                    color: Theme.statusColor(flow.status)
                )
            }

            // Info row
            HStack(spacing: 16) {
                // Experiment count
                Label("\(flow.config.experiments.count) experiments", systemImage: "flask")
                    .font(.caption)
                    .foregroundColor(.secondary)

                // Duration/time
                if let duration = flow.duration {
                    Label(DateFormatters.durationCompact(duration), systemImage: "clock")
                        .font(.caption)
                        .foregroundColor(.secondary)
                } else if let created = flow.createdDate {
                    Label(DateFormatters.relative(created), systemImage: "calendar")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                // Template badge
                if let template = flow.config.template {
                    Text(template)
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.purple.opacity(0.2))
                        .foregroundColor(.purple)
                        .cornerRadius(4)
                }
            }

            // Action buttons
            HStack(spacing: 12) {
                Spacer()

                // Stop button (for running flows)
                if flow.status == .running {
                    Button {
                        showingStopConfirmation = true
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                }

                // Restart button (for completed/failed/cancelled flows)
                if flow.status.isTerminal {
                    Button {
                        Task { await viewModel.restartFlow(flow.id) }
                    } label: {
                        Label("Restart", systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(.bordered)
                }

                // More options menu
                Menu {
                    Button {
                        onSelect()
                    } label: {
                        Label("View Details", systemImage: "info.circle")
                    }

                    if flow.status != .running {
                        Divider()

                        Button(role: .destructive) {
                            showingDeleteConfirmation = true
                        } label: {
                            Label("Delete", systemImage: "trash")
                        }
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
        }
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(12)
        .contentShape(Rectangle())
        .onTapGesture {
            onSelect()
        }
        .confirmationDialog(
            "Stop Flow",
            isPresented: $showingStopConfirmation,
            titleVisibility: .visible
        ) {
            Button("Stop Flow", role: .destructive) {
                Task { await viewModel.stopFlow(flow.id) }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This will stop the running experiment. Are you sure?")
        }
        .confirmationDialog(
            "Delete Flow",
            isPresented: $showingDeleteConfirmation,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                Task { await viewModel.deleteFlow(flow.id) }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This will permanently delete the flow and its history.")
        }
    }
}

// MARK: - Compact Flow Card

struct CompactFlowCard: View {
    let flow: Flow

    var body: some View {
        HStack(spacing: 12) {
            // Status indicator
            Circle()
                .fill(Theme.statusColor(flow.status))
                .frame(width: 8, height: 8)

            // Name
            Text(flow.name)
                .font(.subheadline)
                .lineLimit(1)

            Spacer()

            // Status badge
            Text(flow.status.displayName)
                .font(.caption2)
                .foregroundColor(.secondary)

            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 8)
    }
}

#Preview {
    let sampleFlow = Flow(
        id: 1,
        name: "4ngram Baseline",
        description: "Testing asymmetric tier configuration",
        config: FlowConfig(
            experiments: [
                ExperimentSpec(name: "Phase 1", experiment_type: .ga, optimize_neurons: true),
                ExperimentSpec(name: "Phase 2", experiment_type: .ts, optimize_bits: true),
            ],
            template: "standard-6-phase"
        ),
        created_at: "2026-01-31T18:45:23Z",
        started_at: "2026-01-31T18:47:12Z",
        completed_at: nil,
        status: .running,
        seed_checkpoint_id: nil,
        pid: 12345
    )

    FlowCardView(flow: sampleFlow, onSelect: {})
        .environmentObject(FlowsViewModel(
            apiClient: APIClient(connectionManager: ConnectionManager(settings: SettingsStore())),
            wsManager: WebSocketManager(connectionManager: ConnectionManager(settings: SettingsStore()))
        ))
        .padding()
}
