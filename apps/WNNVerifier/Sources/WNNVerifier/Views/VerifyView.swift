// VerifyView - Download and verify WNN models on-device

import SwiftUI

public struct VerifyView: View {
	@ObservedObject var viewModel: VerifyViewModel

	public init(viewModel: VerifyViewModel) {
		self.viewModel = viewModel
	}

	public var body: some View {
		NavigationStack {
			ScrollView {
				VStack(spacing: 20) {
					inputSection
					statusSection
					if let result = viewModel.result {
						resultsSection(result)
					}
					if let error = viewModel.error {
						errorSection(error)
					}
				}
				.padding()
			}
			.navigationTitle("Verify")
		}
	}

	// MARK: - Input Section

	private var inputSection: some View {
		VStack(alignment: .leading, spacing: 12) {
			Text("Model Source")
				.font(.headline)

			HStack {
				TextField("HuggingFace repo (e.g. lacg/wnn-ids-best)", text: $viewModel.hfRepoId)
					.font(.body)
					.textFieldStyle(.roundedBorder)
					.disabled(!viewModel.canVerify)

				Button("Verify") {
					Task { await viewModel.verifyFromRepo() }
				}
				.font(.body.weight(.medium))
				.buttonStyle(.borderedProminent)
				.disabled(!viewModel.canVerify || viewModel.hfRepoId.isEmpty)
			}

			HStack(spacing: 8) {
				Image(systemName: viewModel.hasGPU ? "gpu" : "cpu")
					.foregroundStyle(viewModel.hasGPU ? .green : .orange)
				Text(viewModel.hasGPU ? "Metal GPU available" : "CPU mode (no GPU)")
					.font(.body)
					.foregroundStyle(.secondary)
			}
		}
		.padding()
		.glassEffect(.regular, in: .rect(cornerRadius: 16))
	}

	// MARK: - Status Section

	@ViewBuilder
	private var statusSection: some View {
		if viewModel.status != .idle {
			VStack(spacing: 12) {
				HStack {
					Circle()
						.fill(Theme.verificationColor(viewModel.status))
						.frame(width: 10, height: 10)
					Text(viewModel.statusMessage)
						.font(.body.weight(.medium))
					Spacer()
					if viewModel.status == .downloading || viewModel.status == .evaluating {
						ProgressView()
					}
				}

				ProgressView(value: viewModel.progress)
					.tint(Theme.verificationColor(viewModel.status))

				if viewModel.status == .completed || viewModel.status == .failed {
					Button("Reset") { viewModel.reset() }
						.font(.body)
						.buttonStyle(.bordered)
				}
			}
			.padding()
			.glassEffect(.regular, in: .rect(cornerRadius: 16))
		}
	}

	// MARK: - Results Section

	private func resultsSection(_ result: VerificationResult) -> some View {
		VStack(alignment: .leading, spacing: 16) {
			Text("Results")
				.font(.headline)

			LazyVGrid(columns: [
				GridItem(.flexible()),
				GridItem(.flexible()),
			], spacing: 16) {
				MetricValue(
					label: "Cross-Entropy",
					value: Formatters.ce(result.crossEntropy),
					color: Theme.ceColor
				)
				MetricValue(
					label: "Perplexity",
					value: String(format: "%.1f", result.perplexity),
					color: Theme.ceColor
				)
				MetricValue(
					label: "Accuracy",
					value: Formatters.accuracy(result.accuracy),
					color: Theme.accuracyColor
				)
				MetricValue(
					label: "Predictions",
					value: Formatters.compact(result.numPredictions),
					color: .primary
				)
			}

			HStack {
				Image(systemName: "clock")
				Text("Evaluated in \(Formatters.duration(result.elapsedSeconds))")
					.font(.body)
				Spacer()
				Text("\(result.dataset) / \(result.split)")
					.font(.body)
					.foregroundStyle(.secondary)
			}
			.foregroundStyle(.secondary)

			if let f1 = result.f1Macro {
				MetricValue(
					label: "F1 Macro",
					value: Formatters.percent(f1),
					color: Theme.f1Color
				)
			}
		}
		.padding()
		.glassEffect(.regular, in: .rect(cornerRadius: 16))
	}

	// MARK: - Error Section

	private func errorSection(_ error: String) -> some View {
		HStack {
			Image(systemName: "exclamationmark.triangle.fill")
				.foregroundStyle(.red)
			Text(error)
				.font(.body)
				.foregroundStyle(.red)
		}
		.padding()
		.glassEffect(.regular, in: .rect(cornerRadius: 16))
	}
}
