// HFExportView - HuggingFace model export for completed experiments

import SwiftUI

public struct HFExportView: View {
	let experiment: Experiment
	let checkpointId: Int64

	@EnvironmentObject var viewModel: DashboardViewModel
	@State private var isExporting = false
	@State private var exportResult: ExportResult?
	@State private var error: String?

	public init(experiment: Experiment, checkpointId: Int64) {
		self.experiment = experiment
		self.checkpointId = checkpointId
	}

	public var body: some View {
		VStack(alignment: .leading, spacing: 16) {
			headerSection
			detailsSection

			if let result = exportResult {
				successSection(result)
			} else if let error = error {
				errorSection(error)
			}

			exportButton
		}
		.padding()
		.glassCard()
	}

	// MARK: - Header

	private var headerSection: some View {
		HStack {
			Image(systemName: "square.and.arrow.up")
				.font(.title2)
				.foregroundStyle(.blue)
			VStack(alignment: .leading) {
				Text("Export to HuggingFace")
					.font(.headline)
				Text("Save as PreTrainedModel for sharing and reuse")
					.font(.caption)
					.foregroundStyle(.secondary)
			}
			Spacer()
		}
	}

	// MARK: - Details

	private var detailsSection: some View {
		VStack(alignment: .leading, spacing: 8) {
			detailRow("Architecture", value: experiment.architecture_type?.displayName ?? "Tiered")
			detailRow("Context Size", value: "\(experiment.context_size)")

			if let ce = experiment.best_ce {
				detailRow("Best CE", value: String(format: "%.4f", ce))
			}
			if let acc = experiment.best_accuracy {
				detailRow("Best Accuracy", value: String(format: "%.2f%%", acc * 100))
			}
		}
	}

	private func detailRow(_ label: String, value: String) -> some View {
		HStack {
			Text(label)
				.foregroundStyle(.secondary)
			Spacer()
			Text(value)
				.monospacedDigit()
				.fontDesign(.monospaced)
		}
		.font(.callout)
	}

	// MARK: - Export Button

	private var exportButton: some View {
		Button {
			exportToHF()
		} label: {
			HStack {
				if isExporting {
					ProgressView()
						.controlSize(.small)
				} else {
					Image(systemName: "arrow.up.circle.fill")
				}
				Text(isExporting ? "Exporting..." : "Export Model")
			}
			.frame(maxWidth: .infinity)
		}
		.buttonStyle(.borderedProminent)
		.disabled(isExporting || experiment.status != .completed)
	}

	// MARK: - Success/Error sections

	private func successSection(_ result: ExportResult) -> some View {
		VStack(alignment: .leading, spacing: 4) {
			Label("Export Complete", systemImage: "checkmark.circle.fill")
				.foregroundStyle(.green)
				.font(.callout.bold())
			Text(result.outputDir)
				.font(.caption)
				.fontDesign(.monospaced)
				.foregroundStyle(.secondary)
		}
		.padding(12)
		.frame(maxWidth: .infinity, alignment: .leading)
		#if os(iOS)
		.background(Color(.secondarySystemBackground))
		#else
		.background(Color.green.opacity(0.1))
		#endif
		.cornerRadius(8)
	}

	private func errorSection(_ message: String) -> some View {
		VStack(alignment: .leading, spacing: 4) {
			Label("Export Failed", systemImage: "xmark.circle.fill")
				.foregroundStyle(.red)
				.font(.callout.bold())
			Text(message)
				.font(.caption)
				.foregroundStyle(.secondary)
		}
		.padding(12)
		.frame(maxWidth: .infinity, alignment: .leading)
		#if os(iOS)
		.background(Color(.secondarySystemBackground))
		#else
		.background(Color.red.opacity(0.1))
		#endif
		.cornerRadius(8)
	}

	// MARK: - Export Action

	private func exportToHF() {
		isExporting = true
		error = nil
		exportResult = nil

		Task {
			do {
				let result = try await viewModel.apiClient.exportCheckpointHF(checkpointId: checkpointId)
				await MainActor.run {
					exportResult = result
					isExporting = false
				}
			} catch {
				await MainActor.run {
					self.error = error.localizedDescription
					isExporting = false
				}
			}
		}
	}
}

// MARK: - Export Result

public struct ExportResult: Codable {
	public let checkpointId: Int64
	public let outputDir: String
	public let architectureType: String
	public let bestCe: Double?
	public let bestAccuracy: Double?

	private enum CodingKeys: String, CodingKey {
		case checkpointId = "checkpoint_id"
		case outputDir = "output_dir"
		case architectureType = "architecture_type"
		case bestCe = "best_ce"
		case bestAccuracy = "best_accuracy"
	}
}
