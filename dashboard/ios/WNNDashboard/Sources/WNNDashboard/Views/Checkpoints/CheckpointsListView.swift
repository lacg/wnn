// CheckpointsListView - List of all checkpoints

import SwiftUI

public struct CheckpointsListView: View {
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @EnvironmentObject var viewModel: CheckpointsViewModel

    public init() {}

    public var body: some View {
        NavigationStack {
            Group {
                if viewModel.isLoading && viewModel.checkpoints.isEmpty { ProgressView("Loading checkpoints...") }
                else if viewModel.checkpoints.isEmpty { emptyState }
                else { checkpointsList }
            }
            .frame(maxWidth: LayoutConstants.formMaxWidth(for: horizontalSizeClass))
            .navigationTitle("Checkpoints")
            .toolbar { ToolbarItem(placement: .navigationBarTrailing) { filterMenu } }
            .refreshable { await viewModel.refresh() }
            .alert("Error", isPresented: .constant(viewModel.error != nil)) { Button("OK") { viewModel.clearError() } } message: { Text(viewModel.error ?? "") }
        }
        .task { if viewModel.checkpoints.isEmpty { await viewModel.loadCheckpoints() } }
    }

    private var checkpointsList: some View {
        List {
            Section {
                HStack {
                    VStack(alignment: .leading) { Text("Total Size").font(.caption).foregroundColor(.secondary); Text(ByteCountFormatter.string(fromByteCount: viewModel.totalSize, countStyle: .binary)).font(.headline).fontDesign(.monospaced) }
                    Spacer()
                    VStack(alignment: .trailing) { Text("Count").font(.caption).foregroundColor(.secondary); Text("\(viewModel.filteredCheckpoints.count)").font(.headline) }
                }
            }
            ForEach(CheckpointType.allCases, id: \.self) { type in
                let typeCheckpoints = viewModel.filteredCheckpoints.filter { $0.checkpoint_type == type }
                if !typeCheckpoints.isEmpty {
                    Section(type.displayName) { ForEach(typeCheckpoints) { cp in CheckpointRow(checkpoint: cp, onDelete: { Task { await viewModel.deleteCheckpoint(cp.id) } }, downloadURL: viewModel.downloadURL(for: cp)) } }
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "externaldrive").font(.system(size: 48)).foregroundColor(.secondary)
            Text("No Checkpoints").font(.headline)
            Text("Checkpoints are created automatically during experiments").font(.subheadline).foregroundColor(.secondary).multilineTextAlignment(.center)
        }
        .padding()
    }

    private var filterMenu: some View {
        Menu {
            Button { viewModel.typeFilter = nil } label: { Label("All Types", systemImage: viewModel.typeFilter == nil ? "checkmark" : "") }
            Divider()
            ForEach(CheckpointType.allCases, id: \.self) { t in Button { viewModel.typeFilter = t } label: { Label(t.displayName, systemImage: viewModel.typeFilter == t ? "checkmark" : "") } }
        } label: { Image(systemName: "line.3.horizontal.decrease.circle") }
    }
}

struct CheckpointRow: View {
    let checkpoint: Checkpoint; let onDelete: () -> Void; let downloadURL: URL?
    @State private var showingDeleteConfirmation = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack { Text(checkpoint.name).font(.subheadline).fontWeight(.medium).lineLimit(1); Spacer(); CheckpointTypeBadge(type: checkpoint.checkpoint_type) }
            HStack(spacing: 16) {
                if let ce = checkpoint.best_ce { Label(NumberFormatters.formatCE(ce), systemImage: "chart.line.downtrend.xyaxis").font(.caption).foregroundColor(.secondary) }
                if let acc = checkpoint.accuracyPercent { Label(NumberFormatters.formatPercent(acc), systemImage: "chart.line.uptrend.xyaxis").font(.caption).foregroundColor(.secondary) }
                Spacer()
                if let size = checkpoint.formattedFileSize { Text(size).font(.caption).foregroundColor(.secondary) }
            }
            if let date = checkpoint.createdDate { Text(DateFormatters.relative(date)).font(.caption2).foregroundColor(.secondary) }
        }
        .padding(.vertical, 4)
        .swipeActions(edge: .trailing, allowsFullSwipe: false) {
            Button(role: .destructive) { showingDeleteConfirmation = true } label: { Label("Delete", systemImage: "trash") }
            if let url = downloadURL { ShareLink(item: url) { Label("Download", systemImage: "arrow.down.circle") }.tint(.blue) }
        }
        .confirmationDialog("Delete Checkpoint", isPresented: $showingDeleteConfirmation, titleVisibility: .visible) { Button("Delete", role: .destructive) { onDelete() }; Button("Cancel", role: .cancel) {} } message: { Text("This will permanently delete the checkpoint file.") }
    }
}

struct CheckpointTypeBadge: View {
    let type: CheckpointType
    var body: some View {
        Text(type.displayName).font(.caption2).padding(.horizontal, 6).padding(.vertical, 2).background(bgColor.opacity(0.2)).foregroundColor(bgColor).cornerRadius(4)
    }
    private var bgColor: Color { switch type { case .auto: return .gray; case .user: return .blue; case .phase_end: return .green; case .experiment_end: return .purple } }
}
