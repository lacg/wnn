// BitwiseConfigEditorView - Configuration editor for bitwise 7-phase flows

import SwiftUI

public struct BitwiseConfigEditorView: View {
	@Binding var numClusters: Int
	@Binding var minBits: Int
	@Binding var maxBits: Int
	@Binding var minNeurons: Int
	@Binding var maxNeurons: Int
	@Binding var memoryMode: String
	@Binding var neuronSampleRate: Double

	public init(
		numClusters: Binding<Int>,
		minBits: Binding<Int>,
		maxBits: Binding<Int>,
		minNeurons: Binding<Int>,
		maxNeurons: Binding<Int>,
		memoryMode: Binding<String>,
		neuronSampleRate: Binding<Double>
	) {
		self._numClusters = numClusters
		self._minBits = minBits
		self._maxBits = maxBits
		self._minNeurons = minNeurons
		self._maxNeurons = maxNeurons
		self._memoryMode = memoryMode
		self._neuronSampleRate = neuronSampleRate
	}

	private let memoryModes = ["QUAD_WEIGHTED", "QUAD_BINARY", "TERNARY"]

	public var body: some View {
		Section("Bitwise Architecture") {
			Stepper("Clusters: \(numClusters)", value: $numClusters, in: 1...256)

			Picker("Memory Mode", selection: $memoryMode) {
				ForEach(memoryModes, id: \.self) { mode in
					Text(mode.replacingOccurrences(of: "_", with: " ").capitalized).tag(mode)
				}
			}

			HStack {
				Text("Neuron Sample Rate")
				Spacer()
				Text(String(format: "%.0f%%", neuronSampleRate * 100))
					.foregroundStyle(.secondary)
					.monospacedDigit()
			}
			Slider(value: $neuronSampleRate, in: 0.05...1.0, step: 0.05)
		}

		Section("Bits Range") {
			Stepper("Min Bits: \(minBits)", value: $minBits, in: 1...maxBits)
			Stepper("Max Bits: \(maxBits)", value: $maxBits, in: minBits...64)
			Text("Address space per neuron: 2^\(minBits) to 2^\(maxBits)")
				.font(.caption)
				.foregroundStyle(.secondary)
		}

		Section("Neurons Range") {
			Stepper("Min Neurons: \(minNeurons)", value: $minNeurons, in: 1...maxNeurons, step: 10)
			Stepper("Max Neurons: \(maxNeurons)", value: $maxNeurons, in: minNeurons...1000, step: 10)
			Text("Per-cluster neuron count range for GA/TS search")
				.font(.caption)
				.foregroundStyle(.secondary)
		}
	}
}
