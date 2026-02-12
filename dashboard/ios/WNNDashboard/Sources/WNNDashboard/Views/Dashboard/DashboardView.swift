// DashboardView - Main iterations monitoring with expandable experiment sections

import SwiftUI
import Charts

public struct DashboardView: View {
	@Environment(\.horizontalSizeClass) private var horizontalSizeClass
	@EnvironmentObject var viewModel: DashboardViewModel
	@EnvironmentObject var wsManager: WebSocketManager
	@State private var expandedExperiments: Set<Int64> = []

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
			.task {
				await viewModel.loadRecentExperiments()
				// Auto-expand the running experiment
				if let running = viewModel.currentExperiment {
					expandedExperiments.insert(running.id)
				}
			}
			.sheet(item: $viewModel.selectedIteration) { iter in
				IterationDetailSheet(iteration: iter, genomes: viewModel.selectedIterationGenomes)
			}
		}
	}

	// MARK: - iPhone Layout

	private var iPhoneLayout: some View {
		ScrollView {
			VStack(spacing: 16) {
				metricsSection
				if viewModel.isRunning { progressSection }
				if hasIterations { chartSection }
				experimentsSection
			}
			.padding()
		}
	}

	// MARK: - iPad Layout (full-width, no split)

	private var iPadLayout: some View {
		ScrollView {
			VStack(spacing: 20) {
				// 4-column metrics on iPad
				LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 4), spacing: 12) {
					MetricsCardView(title: "Best CE", value: NumberFormatters.formatCE(viewModel.bestCE), icon: "arrow.down.circle", color: .blue)
					MetricsCardView(title: "Best Accuracy", value: NumberFormatters.formatAccuracy(viewModel.bestAccuracy), icon: "arrow.up.circle", color: .green)
					MetricsCardView(title: "Type", value: viewModel.currentExperiment?.typePrefix ?? "-", icon: "gearshape.2", color: .purple)
					MetricsCardView(title: "Iteration", value: "\(viewModel.currentIterationNumber)", icon: "number", color: .orange)
				}
				if viewModel.isRunning { progressSection }
				if hasIterations {
					chartSection
						.frame(height: LayoutConstants.chartHeight(for: horizontalSizeClass))
				}
				experimentsSection
			}
			.padding()
		}
	}

	// MARK: - Metrics

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

	// MARK: - Chart

	private var hasIterations: Bool {
		!viewModel.chartIterations.isEmpty
	}

	private var chartSection: some View {
		DualAxisChartView(iterations: viewModel.chartIterations, title: viewModel.chartTitle)
	}

	// MARK: - Expandable Experiments

	private var experimentsSection: some View {
		VStack(alignment: .leading, spacing: 12) {
			if !viewModel.currentFlowExperiments.isEmpty, let exp = viewModel.currentExperiment {
				HStack {
					Text(exp.flow_id.map { _ in "Flow Experiments" } ?? "Experiments").font(.headline)
					Spacer()
					Text("\(viewModel.displayExperiments.count)").font(.caption).foregroundStyle(.secondary)
				}
			} else {
				Text("Experiments").font(.headline)
			}

			if viewModel.displayExperiments.isEmpty && !viewModel.isLoading {
				VStack(spacing: 12) {
					Image(systemName: "flask").font(.system(size: 36)).foregroundColor(.secondary)
					Text("No experiments yet").font(.subheadline).foregroundColor(.secondary)
				}
				.frame(maxWidth: .infinity)
				.padding(.vertical, 24)
			} else {
				ForEach(viewModel.displayExperiments) { exp in
					experimentDisclosureGroup(exp)
				}
			}
		}
	}

	private func experimentDisclosureGroup(_ exp: Experiment) -> some View {
		let isRunning = exp.id == viewModel.currentExperiment?.id
		let isChartSelected = viewModel.chartExperimentId == exp.id || (viewModel.chartExperimentId == nil && isRunning)

		return VStack(spacing: 0) {
			DisclosureGroup(
				isExpanded: Binding(
					get: { expandedExperiments.contains(exp.id) },
					set: { newValue in
						if newValue {
							expandedExperiments.insert(exp.id)
							Task { await viewModel.loadExperimentIterations(exp.id) }
						} else {
							expandedExperiments.remove(exp.id)
						}
					}
				)
			) {
				let iters = viewModel.iterations(for: exp.id)
				if iters.isEmpty {
					HStack {
						Spacer()
						ProgressView().padding(.vertical, 8)
						Spacer()
					}
				} else if horizontalSizeClass == .regular {
					iPadIterationsContent(iters)
				} else {
					iPhoneIterationsContent(iters)
				}
			} label: {
				experimentHeader(exp, isRunning: isRunning)
			}
		}
		.padding()
		.background(
			isChartSelected
				? AnyShapeStyle(Color.accentColor.opacity(0.08))
				: AnyShapeStyle(.ultraThinMaterial),
			in: RoundedRectangle(cornerRadius: 12)
		)
		.overlay(
			RoundedRectangle(cornerRadius: 12)
				.stroke(isChartSelected ? Color.accentColor.opacity(0.3) : Color.clear, lineWidth: 1)
		)
		.onTapGesture { viewModel.selectChartExperiment(exp.id) }
	}

	private func experimentHeader(_ exp: Experiment, isRunning: Bool) -> some View {
		VStack(alignment: .leading, spacing: 6) {
			HStack {
				if isRunning {
					Image(systemName: "bolt.fill").font(.caption).foregroundStyle(.orange)
				}
				Text(exp.name)
					.font(.subheadline).fontWeight(.semibold)
					.lineLimit(1)
				Spacer()
				StatusBadge(text: exp.status.displayName, color: Theme.statusColor(exp.status))
			}
			HStack(spacing: 12) {
				if let phaseType = exp.formattedPhaseType {
					Text(phaseType).font(.caption).foregroundStyle(.secondary)
				}
				if let ce = exp.best_ce {
					Label(NumberFormatters.formatCE(ce), systemImage: "chart.line.downtrend.xyaxis")
						.font(.caption).foregroundStyle(.secondary)
				}
				if let acc = exp.best_accuracy {
					Label(NumberFormatters.formatPercent(acc * 100), systemImage: "chart.line.uptrend.xyaxis")
						.font(.caption).foregroundStyle(.secondary)
				}
				Spacer()
				if let iter = exp.current_iteration ?? exp.last_iteration {
					Text("#\(iter)").font(.caption).foregroundStyle(.secondary).fontDesign(.monospaced)
				}
			}
			if let date = exp.endedDate ?? exp.startedDate ?? exp.createdDate {
				Text(DateFormatters.relative(date))
					.font(.caption2).foregroundStyle(.tertiary)
			}
		}
		.padding(.vertical, 4)
		.contentShape(Rectangle())
	}

	// MARK: - iPhone Iterations (compact rows)

	private func iPhoneIterationsContent(_ iters: [Iteration]) -> some View {
		VStack(spacing: 6) {
			ForEach(Array(iters.prefix(20))) { iter in
				compactIterationRow(iter)
					.onTapGesture { Task { await viewModel.loadGenomes(for: iter) } }
			}
			if iters.count > 20 {
				Text("\(iters.count - 20) more iterations...")
					.font(.caption2).foregroundStyle(.secondary)
					.padding(.top, 4)
			}
		}
		.padding(.top, 8)
	}

	private func compactIterationRow(_ iter: Iteration) -> some View {
		HStack(spacing: 8) {
			Text("#\(iter.iteration_num)")
				.font(.caption).fontWeight(.medium).fontDesign(.monospaced)
				.frame(width: 40, alignment: .leading)
			Text(NumberFormatters.formatCE(iter.best_ce))
				.font(.caption).fontDesign(.monospaced)
			if let acc = iter.best_accuracy {
				Text(NumberFormatters.formatAccuracy(acc))
					.font(.caption).fontDesign(.monospaced)
			}
			Spacer()
			if let delta = iter.delta_previous {
				HStack(spacing: 1) {
					Image(systemName: delta < 0 ? "arrow.down" : delta > 0 ? "arrow.up" : "minus").font(.system(size: 8))
					Text(String(format: "%.4f", abs(delta))).font(.caption2).fontDesign(.monospaced)
				}
				.foregroundColor(Theme.deltaColor(delta))
			}
			if let date = iter.createdDate {
				Text(DateFormatters.relative(date)).font(.caption2).foregroundStyle(.secondary)
			}
			Image(systemName: "chevron.right").font(.system(size: 8)).foregroundStyle(.tertiary)
		}
		.padding(.vertical, 4)
		.padding(.horizontal, 8)
		.background(Color.secondary.opacity(0.06))
		.cornerRadius(6)
	}

	// MARK: - iPad Iterations Table (10-column)

	private func iPadIterationsContent(_ iters: [Iteration]) -> some View {
		VStack(spacing: 0) {
			iPadIterationTableHeader
			Divider().padding(.vertical, 4)
			VStack(spacing: 2) {
				ForEach(Array(iters.prefix(50))) { iter in
					iPadIterationRow(iter)
						.onTapGesture { Task { await viewModel.loadGenomes(for: iter) } }
				}
				if iters.count > 50 {
					Text("\(iters.count - 50) more iterations...")
						.font(.caption2).foregroundStyle(.secondary)
						.padding(.top, 4)
				}
			}
		}
		.padding(.top, 8)
	}

	private var iPadIterationTableHeader: some View {
		HStack(spacing: 0) {
			Text("ITER").frame(width: 44, alignment: .leading)
			Text("TIMESTAMP").frame(width: 82, alignment: .leading)
			Text("BEST CE").frame(width: 74, alignment: .trailing)
			Text("BEST ACC").frame(width: 64, alignment: .trailing)
			Text("AVG CE").frame(width: 74, alignment: .trailing)
			Text("AVG ACC").frame(width: 64, alignment: .trailing)
			Text("THRESHOLD").frame(width: 74, alignment: .trailing)
			Text("Δ PREV").frame(width: 72, alignment: .trailing)
			Text("PATIENCE").frame(width: 56, alignment: .center)
			Text("TIME").frame(width: 52, alignment: .trailing)
		}
		.font(.caption2)
		.fontWeight(.medium)
		.foregroundStyle(.secondary)
		.padding(.horizontal, 6)
	}

	private func iPadIterationRow(_ iter: Iteration) -> some View {
		HStack(spacing: 0) {
			Text("#\(iter.iteration_num)")
				.fontWeight(.medium)
				.frame(width: 44, alignment: .leading)

			Group {
				if let date = iter.createdDate {
					Text(DateFormatters.timeOnly(date))
				} else {
					Text("-")
				}
			}
			.frame(width: 82, alignment: .leading)

			Text(NumberFormatters.formatCE(iter.best_ce))
				.frame(width: 74, alignment: .trailing)

			Text(NumberFormatters.formatAccuracy(iter.best_accuracy))
				.frame(width: 64, alignment: .trailing)

			Text(NumberFormatters.formatCE(iter.avg_ce))
				.frame(width: 74, alignment: .trailing)

			Text(NumberFormatters.formatAccuracy(iter.avg_accuracy))
				.frame(width: 64, alignment: .trailing)

			Group {
				if let threshold = iter.fitness_threshold {
					Text(String(format: "%.2f", threshold))
				} else {
					Text("-")
				}
			}
			.frame(width: 74, alignment: .trailing)

			Group {
				if let delta = iter.delta_previous {
					HStack(spacing: 1) {
						Image(systemName: delta < 0 ? "arrow.down" : delta > 0 ? "arrow.up" : "minus")
							.font(.system(size: 7))
						Text(String(format: "%.4f", abs(delta)))
					}
					.foregroundColor(Theme.deltaColor(delta))
				} else {
					Text("-")
				}
			}
			.frame(width: 72, alignment: .trailing)

			Text(iter.patienceStatus ?? "-")
				.frame(width: 56, alignment: .center)

			Group {
				if let secs = iter.elapsed_secs {
					Text(DateFormatters.durationCompact(secs))
				} else {
					Text("-")
				}
			}
			.frame(width: 52, alignment: .trailing)
		}
		.font(.caption)
		.fontDesign(.monospaced)
		.padding(.vertical, 3)
		.padding(.horizontal, 6)
		.background(Color.secondary.opacity(0.04))
		.cornerRadius(4)
		.contentShape(Rectangle())
	}

	// MARK: - Connection Indicator

	private var connectionIndicator: some View {
		HStack(spacing: 4) {
			Circle().fill(wsManager.connectionState.isConnected ? Color.green : Color.red).frame(width: 8, height: 8)
			Text(wsManager.connectionState.isConnected ? "Live" : "Offline").font(.caption).foregroundColor(.secondary)
		}
	}
}
