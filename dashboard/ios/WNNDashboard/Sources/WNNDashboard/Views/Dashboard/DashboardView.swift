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

	// MARK: - iPad Layout (side-by-side)

	private var iPadLayout: some View {
		HStack(alignment: .top, spacing: 20) {
			// Left column: Metrics + Experiments list
			ScrollView {
				VStack(spacing: 16) {
					metricsSection
					if viewModel.isRunning { progressSection }
					experimentsSection
				}
			}
			.frame(width: LayoutConstants.iPadSidebarWidth)

			// Right column: Chart + selected experiment iterations table
			VStack(spacing: 16) {
				if hasIterations {
					chartSection
						.frame(height: LayoutConstants.chartHeight(for: horizontalSizeClass))
				}
				if let chartId = viewModel.chartExperimentId ?? viewModel.currentExperiment?.id {
					iPadIterationsTable(for: chartId)
				}
			}
		}
		.padding()
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
			Text("Experiments").font(.headline)

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
				} else {
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

	// MARK: - Compact Iteration Row (inside DisclosureGroup)

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

	// MARK: - iPad Iterations Table (right column)

	private func iPadIterationsTable(for experimentId: Int64) -> some View {
		let iters = viewModel.iterations(for: experimentId)
		let expName = viewModel.displayExperiments.first { $0.id == experimentId }?.name ?? "Experiment"

		return VStack(alignment: .leading, spacing: 8) {
			Text(expName).font(.headline)
			if !iters.isEmpty {
				IterationsTableView(iterations: Array(iters.prefix(50))) { iter in
					Task { await viewModel.loadGenomes(for: iter) }
				}
			}
		}
	}

	// MARK: - Connection Indicator

	private var connectionIndicator: some View {
		HStack(spacing: 4) {
			Circle().fill(wsManager.connectionState.isConnected ? Color.green : Color.red).frame(width: 8, height: 8)
			Text(wsManager.connectionState.isConnected ? "Live" : "Offline").font(.caption).foregroundColor(.secondary)
		}
	}
}
