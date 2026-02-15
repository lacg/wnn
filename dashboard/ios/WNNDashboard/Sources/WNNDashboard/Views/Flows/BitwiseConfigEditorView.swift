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
	@Binding var synaptogenesis: Bool
	@Binding var neurogenesis: Bool
	@Binding var adaptWarmup: Int
	@Binding var adaptCooldown: Int

	public init(
		numClusters: Binding<Int>,
		minBits: Binding<Int>,
		maxBits: Binding<Int>,
		minNeurons: Binding<Int>,
		maxNeurons: Binding<Int>,
		memoryMode: Binding<String>,
		neuronSampleRate: Binding<Double>,
		synaptogenesis: Binding<Bool> = .constant(false),
		neurogenesis: Binding<Bool> = .constant(false),
		adaptWarmup: Binding<Int> = .constant(10),
		adaptCooldown: Binding<Int> = .constant(5)
	) {
		self._numClusters = numClusters
		self._minBits = minBits
		self._maxBits = maxBits
		self._minNeurons = minNeurons
		self._maxNeurons = maxNeurons
		self._memoryMode = memoryMode
		self._neuronSampleRate = neuronSampleRate
		self._synaptogenesis = synaptogenesis
		self._neurogenesis = neurogenesis
		self._adaptWarmup = adaptWarmup
		self._adaptCooldown = adaptCooldown
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

		Section {
			Toggle(isOn: $synaptogenesis) {
				VStack(alignment: .leading) {
					Text("Synaptogenesis")
					Text("Prune unused synapses, grow new ones on active neurons")
						.font(.caption).foregroundStyle(.secondary)
				}
			}
			Toggle(isOn: $neurogenesis) {
				VStack(alignment: .leading) {
					Text("Neurogenesis")
					Text("Add neurons to struggling clusters, remove redundant ones")
						.font(.caption).foregroundStyle(.secondary)
				}
			}
			if synaptogenesis || neurogenesis {
				Stepper("Warmup: \(adaptWarmup) gen", value: $adaptWarmup, in: 0...100)
				Text("Let evolution stabilize before adaptation kicks in")
					.font(.caption).foregroundStyle(.secondary)
				Stepper("Cooldown: \(adaptCooldown) iter", value: $adaptCooldown, in: 0...50)
				Text("Freeze a neuron after adapting it, preventing prune/grow oscillation")
					.font(.caption).foregroundStyle(.secondary)
			}
		} header: {
			Text("Adaptation (Baldwin Effect)")
		} footer: {
			Text("Each genome is adapted during evaluation, then scored. Evolution selects architectures that respond well to adaptation.")
		}
	}
}
