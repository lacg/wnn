// LeaderboardView - Ranked genome table with filters

import SwiftUI

public struct LeaderboardView: View {
	@ObservedObject var viewModel: LeaderboardViewModel
	@State private var expandedId: Int64?

	public init(viewModel: LeaderboardViewModel) {
		self.viewModel = viewModel
	}

	public var body: some View {
		NavigationStack {
			VStack(spacing: 0) {
				filterBar
				genomeList
			}
			.navigationTitle("Leaderboard")
			.task { await viewModel.loadInitial() }
			.refreshable { await viewModel.refresh() }
		}
	}

	// MARK: - Filter Bar

	private var filterBar: some View {
		ScrollView(.horizontal, showsIndicators: false) {
			HStack(spacing: 12) {
				filterPicker(
					label: "Task",
					selection: $viewModel.taskType,
					options: LeaderboardViewModel.taskTypes
				)
				filterPicker(
					label: "Stage",
					selection: $viewModel.stage,
					options: LeaderboardViewModel.stages
				)
				filterPicker(
					label: "Metric",
					selection: $viewModel.metric,
					options: LeaderboardViewModel.metrics
				)
			}
			.padding(.horizontal)
			.padding(.vertical, 8)
		}
	}

	private func filterPicker(
		label: String,
		selection: Binding<String>,
		options: [(String, String)]
	) -> some View {
		Menu {
			ForEach(options, id: \.0) { value, display in
				Button(display) { selection.wrappedValue = value }
			}
		} label: {
			HStack(spacing: 4) {
				Text(label)
					.font(.body)
					.foregroundStyle(.secondary)
				Text(options.first { $0.0 == selection.wrappedValue }?.1 ?? selection.wrappedValue)
					.font(.body.weight(.medium))
			}
			.padding(.horizontal, 12)
			.padding(.vertical, 8)
			.glassEffect(.regular, in: .capsule)
		}
	}

	// MARK: - Genome List

	@ViewBuilder
	private var genomeList: some View {
		if viewModel.isLoading && viewModel.genomes.isEmpty {
			ProgressView("Loading...")
				.font(.body)
				.frame(maxWidth: .infinity, maxHeight: .infinity)
		} else if let error = viewModel.error {
			ContentUnavailableView {
				Label("Error", systemImage: "exclamationmark.triangle")
			} description: {
				Text(error).font(.body)
			} actions: {
				Button("Retry") { Task { await viewModel.refresh() } }
			}
		} else if viewModel.genomes.isEmpty {
			ContentUnavailableView {
				Label("No Genomes", systemImage: "chart.bar")
			} description: {
				Text("No genomes found for this category.").font(.body)
			}
		} else {
			List {
				ForEach(viewModel.genomes) { genome in
					GenomeRow(
						genome: genome,
						isExpanded: expandedId == genome.id,
						onTap: { toggleExpand(genome.id) }
					)
				}
			}
			.listStyle(.plain)
		}
	}

	private func toggleExpand(_ id: Int64) {
		withAnimation(.snappy) {
			expandedId = expandedId == id ? nil : id
		}
	}
}

// MARK: - Genome Row

private struct GenomeRow: View {
	let genome: BestGenome
	let isExpanded: Bool
	let onTap: () -> Void

	var body: some View {
		VStack(alignment: .leading, spacing: 12) {
			// Summary row
			Button(action: onTap) {
				HStack(spacing: 12) {
					RankBadge(rank: genome.rank)

					VStack(alignment: .leading, spacing: 4) {
						HStack(spacing: 8) {
							TaskBadge(taskType: genome.task_type)
							Text(genome.stageLabel)
								.font(.body.weight(.medium))
							Spacer()
						}
						HStack(spacing: 16) {
							MetricValue(
								label: "CE",
								value: Formatters.ce(genome.ce),
								color: Theme.ceColor
							)
							MetricValue(
								label: "Acc",
								value: Formatters.accuracy(genome.accuracy),
								color: Theme.accuracyColor
							)
							if let f1 = genome.f1_macro {
								MetricValue(
									label: "F1",
									value: Formatters.percent(f1),
									color: Theme.f1Color
								)
							}
						}
					}
				}
			}
			.buttonStyle(.plain)

			// Expanded details
			if isExpanded {
				expandedDetails
			}
		}
		.padding(.vertical, 4)
	}

	@ViewBuilder
	private var expandedDetails: some View {
		VStack(alignment: .leading, spacing: 8) {
			Divider()

			LabeledContent("Genome Hash") {
				Text(String(genome.genome_hash.prefix(16)) + "...")
					.font(.body)
					.fontDesign(.monospaced)
					.foregroundStyle(.secondary)
			}
			.font(.body)

			if let clusters = genome.total_clusters, let neurons = genome.total_neurons {
				LabeledContent("Architecture") {
					Text("\(clusters) clusters, \(neurons) neurons")
						.font(.body)
						.foregroundStyle(.secondary)
				}
				.font(.body)
			}

			if let fpr = genome.fpr {
				LabeledContent("FPR") {
					Text(Formatters.percent(fpr))
						.font(.body)
						.foregroundStyle(Theme.fprColor)
				}
				.font(.body)
			}

			if let hfRepo = genome.hf_repo_id, !hfRepo.isEmpty {
				LabeledContent("HuggingFace") {
					Text(hfRepo)
						.font(.body)
						.foregroundStyle(.blue)
				}
				.font(.body)
			}

			Text("Updated \(Formatters.relativeDate(genome.updated_at))")
				.font(.body)
				.foregroundStyle(.tertiary)
		}
		.padding(.leading, 44)
		.transition(.opacity.combined(with: .move(edge: .top)))
	}
}
