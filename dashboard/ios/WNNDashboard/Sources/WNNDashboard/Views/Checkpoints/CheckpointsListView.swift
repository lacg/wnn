// CheckpointsListView - Checkpoints grouped by experiment (expandable)

import SwiftUI

public struct CheckpointsListView: View {
	@Environment(\.horizontalSizeClass) private var horizontalSizeClass
	@EnvironmentObject var viewModel: CheckpointsViewModel
	@State private var expandedExperiments: Set<Int64> = []

	public init() {}

	public var body: some View {
		NavigationStack {
			Group {
				if viewModel.isLoading && viewModel.checkpoints.isEmpty { ProgressView("Loading checkpoints...") }
				else if viewModel.checkpoints.isEmpty { emptyState }
				else { checkpointsList }
			}
			.navigationTitle("Checkpoints")
			#if os(iOS)
			.toolbar { ToolbarItem(placement: .navigationBarTrailing) { filterMenu } }
			#else
			.toolbar { ToolbarItem(placement: .automatic) { filterMenu } }
			#endif
			.refreshable { await viewModel.refresh() }
			.alert("Error", isPresented: Binding(get: { viewModel.error != nil }, set: { if !$0 { viewModel.clearError() } })) {
				Button("OK") { viewModel.clearError() }
			} message: { Text(viewModel.error ?? "") }
		}
		.task { if viewModel.checkpoints.isEmpty { await viewModel.loadCheckpoints() } }
	}

	// MARK: - Main List

	private var checkpointsList: some View {
		List {
			summarySection
			ForEach(viewModel.experimentGroups) { group in
				experimentSection(group)
			}
		}
		#if os(iOS)
		.listStyle(.insetGrouped)
		#endif
	}

	private var summarySection: some View {
		Section {
			HStack {
				VStack(alignment: .leading) {
					Text("Total Size").font(.caption).foregroundColor(.secondary)
					Text(ByteCountFormatter.string(fromByteCount: viewModel.totalSize, countStyle: .binary))
						.font(.headline).fontDesign(.monospaced)
				}
				Spacer()
				VStack(alignment: .trailing) {
					Text("Checkpoints").font(.caption).foregroundColor(.secondary)
					Text("\(viewModel.filteredCheckpoints.count)").font(.headline)
				}
				Spacer()
				VStack(alignment: .trailing) {
					Text("Experiments").font(.caption).foregroundColor(.secondary)
					Text("\(viewModel.experimentGroups.count)").font(.headline)
				}
			}
		}
	}

	// MARK: - Experiment Section

	private func experimentSection(_ group: ExperimentCheckpointGroup) -> some View {
		Section {
			DisclosureGroup(
				isExpanded: Binding(
					get: { expandedExperiments.contains(group.experimentId) },
					set: { if $0 { expandedExperiments.insert(group.experimentId) } else { expandedExperiments.remove(group.experimentId) } }
				)
			) {
				ForEach(group.checkpoints) { cp in
					CheckpointRow(
						checkpoint: cp,
						onDelete: { Task { await viewModel.deleteCheckpoint(cp.id) } },
						downloadURL: viewModel.downloadURL(for: cp)
					)
				}
			} label: {
				experimentHeader(group)
			}
		}
	}

	private func experimentHeader(_ group: ExperimentCheckpointGroup) -> some View {
		VStack(alignment: .leading, spacing: 6) {
			HStack {
				Text(group.experimentName)
					.font(.subheadline).fontWeight(.semibold)
					.lineLimit(1)
				Spacer()
				StatusBadge(text: group.experimentStatus.displayName, color: Theme.statusColor(group.experimentStatus))
			}
			HStack(spacing: 12) {
				Label("\(group.count)", systemImage: "doc.circle")
					.font(.caption).foregroundColor(.secondary)
				if let ce = group.bestCE {
					Label(NumberFormatters.formatCE(ce), systemImage: "chart.line.downtrend.xyaxis")
						.font(.caption).foregroundColor(.secondary)
				}
				if let acc = group.bestAccuracy {
					Label(NumberFormatters.formatPercent(acc * 100), systemImage: "chart.line.uptrend.xyaxis")
						.font(.caption).foregroundColor(.secondary)
				}
				Spacer()
				if group.totalSize > 0 {
					Text(ByteCountFormatter.string(fromByteCount: group.totalSize, countStyle: .binary))
						.font(.caption).foregroundColor(.secondary)
				}
			}
			if let date = group.latestDate {
				Text(DateFormatters.relative(date))
					.font(.caption2).foregroundStyle(.tertiary)
			}
		}
		.padding(.vertical, 4)
	}

	// MARK: - Empty & Filter

	private var emptyState: some View {
		VStack(spacing: 16) {
			Image(systemName: "externaldrive").font(.system(size: 48)).foregroundColor(.secondary)
			Text("No Checkpoints").font(.headline)
			Text("Checkpoints are created automatically during experiments")
				.font(.subheadline).foregroundColor(.secondary).multilineTextAlignment(.center)
		}
		.padding()
	}

	private var filterMenu: some View {
		Menu {
			Button { viewModel.typeFilter = nil } label: {
				Label("All Types", systemImage: viewModel.typeFilter == nil ? "checkmark" : "")
			}
			Divider()
			ForEach(CheckpointType.allCases, id: \.self) { t in
				Button { viewModel.typeFilter = t } label: {
					Label(t.displayName, systemImage: viewModel.typeFilter == t ? "checkmark" : "")
				}
			}
		} label: { Image(systemName: "line.3.horizontal.decrease.circle") }
	}
}

// MARK: - Checkpoint Row

struct CheckpointRow: View {
	let checkpoint: Checkpoint; let onDelete: () -> Void; let downloadURL: URL?
	@State private var showingDeleteConfirmation = false

	var body: some View {
		VStack(alignment: .leading, spacing: 6) {
			HStack {
				Text(checkpoint.name).font(.subheadline).fontWeight(.medium).lineLimit(1)
				Spacer()
				CheckpointTypeBadge(type: checkpoint.checkpoint_type)
			}
			HStack(spacing: 16) {
				if let ce = checkpoint.best_ce {
					Label(NumberFormatters.formatCE(ce), systemImage: "chart.line.downtrend.xyaxis")
						.font(.caption).foregroundColor(.secondary)
				}
				if let acc = checkpoint.accuracyPercent {
					Label(NumberFormatters.formatPercent(acc), systemImage: "chart.line.uptrend.xyaxis")
						.font(.caption).foregroundColor(.secondary)
				}
				Spacer()
				if let size = checkpoint.formattedFileSize {
					Text(size).font(.caption).foregroundColor(.secondary)
				}
			}
			if let date = checkpoint.createdDate {
				Text(DateFormatters.relative(date)).font(.caption2).foregroundColor(.secondary)
			}
		}
		.padding(.vertical, 2)
		.swipeActions(edge: .trailing, allowsFullSwipe: false) {
			Button(role: .destructive) { showingDeleteConfirmation = true } label: { Label("Delete", systemImage: "trash") }
			if let url = downloadURL {
				ShareLink(item: url) { Label("Download", systemImage: "arrow.down.circle") }.tint(.blue)
			}
		}
		.confirmationDialog("Delete Checkpoint", isPresented: $showingDeleteConfirmation, titleVisibility: .visible) {
			Button("Delete", role: .destructive) { onDelete() }
			Button("Cancel", role: .cancel) {}
		} message: { Text("This will permanently delete the checkpoint file.") }
	}
}

struct CheckpointTypeBadge: View {
	let type: CheckpointType
	var body: some View {
		Text(type.displayName).font(.caption2).padding(.horizontal, 6).padding(.vertical, 2)
			.background(bgColor.opacity(0.2)).foregroundColor(bgColor).cornerRadius(4)
	}
	private var bgColor: Color {
		switch type {
		case .auto: return .gray
		case .user: return .blue
		case .phase_end: return .green
		case .experiment_end: return .purple
		}
	}
}
