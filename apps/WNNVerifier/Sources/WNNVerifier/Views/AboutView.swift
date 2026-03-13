// AboutView - Project information and links

import SwiftUI

public struct AboutView: View {
	public init() {}

	public var body: some View {
		NavigationStack {
			ScrollView {
				VStack(spacing: 20) {
					headerSection
					whatIsWNNSection
					howItWorksSection
					resultsSection
					verificationSection
				}
				.padding()
			}
			.navigationTitle("About")
		}
	}

	// MARK: - Header

	private var headerSection: some View {
		VStack(spacing: 12) {
			Image(systemName: "brain.head.profile")
				.font(.system(size: 48))
				.foregroundStyle(.tint)

			Text("WNN Verifier")
				.font(.title.weight(.bold))

			Text("Independent verification of Weightless Neural Network results")
				.font(.body)
				.foregroundStyle(.secondary)
				.multilineTextAlignment(.center)
		}
		.padding()
		.frame(maxWidth: .infinity)
		.glassEffect(.regular, in: .rect(cornerRadius: 16))
	}

	// MARK: - What is WNN

	private var whatIsWNNSection: some View {
		VStack(alignment: .leading, spacing: 12) {
			Label("What is WNN?", systemImage: "questionmark.circle")
				.font(.headline)

			Text("""
			Weightless Neural Networks (WNNs) use RAM-based neurons instead of \
			traditional weighted connections. Each neuron is a lookup table that \
			maps input bit patterns to output values.
			""")
			.font(.body)

			Text("""
			Unlike conventional neural networks that learn continuous weights via \
			gradient descent, WNNs learn by writing entries into memory tables. \
			The architecture (which input bits each neuron observes) is optimized \
			via genetic algorithms and tabu search.
			""")
			.font(.body)
		}
		.padding()
		.frame(maxWidth: .infinity, alignment: .leading)
		.glassEffect(.regular, in: .rect(cornerRadius: 16))
	}

	// MARK: - How It Works

	private var howItWorksSection: some View {
		VStack(alignment: .leading, spacing: 12) {
			Label("How Verification Works", systemImage: "checkmark.shield")
				.font(.headline)

			StepRow(number: 1, text: "Download a trained genome from HuggingFace Hub")
			StepRow(number: 2, text: "Load the model architecture and memory tables")
			StepRow(number: 3, text: "Evaluate on-device using Metal GPU acceleration")
			StepRow(number: 4, text: "Compare metrics against published results")
		}
		.padding()
		.frame(maxWidth: .infinity, alignment: .leading)
		.glassEffect(.regular, in: .rect(cornerRadius: 16))
	}

	// MARK: - Results

	private var resultsSection: some View {
		VStack(alignment: .leading, spacing: 12) {
			Label("Key Results", systemImage: "chart.bar")
				.font(.headline)

			HStack(spacing: 20) {
				MetricValue(
					label: "IDS Accuracy",
					value: "90.88%",
					color: Theme.accuracyColor
				)
				MetricValue(
					label: "IDS F1-Macro",
					value: "90.83%",
					color: Theme.f1Color
				)
			}

			Text("Beats the 2024 BiLSTM state-of-the-art on UNSW-NB15.")
				.font(.body)
				.foregroundStyle(.secondary)
		}
		.padding()
		.frame(maxWidth: .infinity, alignment: .leading)
		.glassEffect(.regular, in: .rect(cornerRadius: 16))
	}

	// MARK: - Verification

	private var verificationSection: some View {
		VStack(alignment: .leading, spacing: 12) {
			Label("Reproduce Results", systemImage: "terminal")
				.font(.headline)

			Text("CLI verification (Python):")
				.font(.body.weight(.medium))

			Text("pip install wnn && wnn-verify lacg/wnn-ids-best-ce")
				.font(.body)
				.fontDesign(.monospaced)
				.padding(12)
				.frame(maxWidth: .infinity, alignment: .leading)
				.background(.ultraThinMaterial, in: .rect(cornerRadius: 8))
		}
		.padding()
		.frame(maxWidth: .infinity, alignment: .leading)
		.glassEffect(.regular, in: .rect(cornerRadius: 16))
	}
}

// MARK: - Step Row

private struct StepRow: View {
	let number: Int
	let text: String

	var body: some View {
		HStack(alignment: .top, spacing: 12) {
			Text("\(number)")
				.font(.body.weight(.bold))
				.frame(width: 28, height: 28)
				.glassEffect(.regular, in: .circle)

			Text(text)
				.font(.body)
		}
	}
}
