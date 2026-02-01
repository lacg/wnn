// FlowCardView - Summary card for a single flow

import SwiftUI

public struct FlowCardView: View {
    public let flow: Flow
    public let onSelect: () -> Void

    @EnvironmentObject var viewModel: FlowsViewModel
    @State private var showingStopConfirmation = false
    @State private var showingDeleteConfirmation = false

    public init(flow: Flow, onSelect: @escaping () -> Void) { self.flow = flow; self.onSelect = onSelect }

    public var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(flow.name).font(.headline)
                    if let desc = flow.description { Text(desc).font(.caption).foregroundColor(.secondary).lineLimit(1) }
                }
                Spacer()
                StatusBadge(text: flow.status.displayName, color: Theme.statusColor(flow.status))
            }
            HStack(spacing: 16) {
                Label("\(flow.config.experiments.count) experiments", systemImage: "flask").font(.caption).foregroundColor(.secondary)
                if let dur = flow.duration { Label(DateFormatters.durationCompact(dur), systemImage: "clock").font(.caption).foregroundColor(.secondary) }
                else if let date = flow.createdDate { Label(DateFormatters.relative(date), systemImage: "calendar").font(.caption).foregroundColor(.secondary) }
                Spacer()
                if let t = flow.config.template { Text(t).font(.caption2).padding(.horizontal, 6).padding(.vertical, 2).background(Color.purple.opacity(0.2)).foregroundColor(.purple).cornerRadius(4) }
            }
            HStack(spacing: 12) {
                Spacer()
                if flow.status == .running { Button { showingStopConfirmation = true } label: { Label("Stop", systemImage: "stop.fill") }.buttonStyle(.bordered).tint(.red) }
                if flow.status.isTerminal { Button { Task { await viewModel.restartFlow(flow.id) } } label: { Label("Restart", systemImage: "arrow.clockwise") }.buttonStyle(.bordered) }
                Menu { Button { onSelect() } label: { Label("View Details", systemImage: "info.circle") }; if flow.status != .running { Divider(); Button(role: .destructive) { showingDeleteConfirmation = true } label: { Label("Delete", systemImage: "trash") } } } label: { Image(systemName: "ellipsis.circle") }
            }
        }
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(12)
        .contentShape(Rectangle())
        .onTapGesture { onSelect() }
        .confirmationDialog("Stop Flow", isPresented: $showingStopConfirmation, titleVisibility: .visible) { Button("Stop Flow", role: .destructive) { Task { await viewModel.stopFlow(flow.id) } }; Button("Cancel", role: .cancel) {} } message: { Text("This will stop the running experiment.") }
        .confirmationDialog("Delete Flow", isPresented: $showingDeleteConfirmation, titleVisibility: .visible) { Button("Delete", role: .destructive) { Task { await viewModel.deleteFlow(flow.id) } }; Button("Cancel", role: .cancel) {} } message: { Text("This will permanently delete the flow.") }
    }
}
