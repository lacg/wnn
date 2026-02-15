// NewFlowView - Form for creating a new flow with editable phases

import SwiftUI

// UUID-identifiable wrapper so ForEach can track phases through reorders
struct EditablePhase: Identifiable {
	let id = UUID()
	var name: String
	var experimentType: ExperimentType
	var optimizeBits: Bool
	var optimizeNeurons: Bool
	var optimizeConnections: Bool
	var isGridSearch: Bool

	func toExperimentSpec() -> ExperimentSpec {
		var params: [String: AnyCodable] = [:]
		if isGridSearch { params["phase_type"] = AnyCodable("grid_search") }
		return ExperimentSpec(
			name: name,
			experiment_type: experimentType,
			optimize_bits: optimizeBits,
			optimize_neurons: optimizeNeurons,
			optimize_connections: optimizeConnections,
			params: params
		)
	}
}

public struct NewFlowView: View {
	@EnvironmentObject var viewModel: FlowsViewModel
	@Environment(\.dismiss) private var dismiss

	@State private var name = ""
	@State private var description = ""
	@State private var selectedTemplate: FlowTemplate = .standard6Phase
	@State private var phaseOrder: PhaseOrder = .neuronsFirst
	@State private var gaGenerations = 250
	@State private var tsIterations = 250
	@State private var patience = 10
	@State private var population = 50
	@State private var tierConfig = "100,15,20;400,10,12;rest,5,8"
	@State private var tier0Only = true
	@State private var isSubmitting = false
	@State private var error: String?
	@State private var phases: [EditablePhase] = []

	// Bitwise-specific config
	@State private var bitwiseNumClusters = 16
	@State private var bitwiseMinBits = 10
	@State private var bitwiseMaxBits = 24
	@State private var bitwiseMinNeurons = 10
	@State private var bitwiseMaxNeurons = 300
	@State private var bitwiseMemoryMode = "QUAD_WEIGHTED"
	@State private var bitwiseNeuronSampleRate = 0.25

	// Adaptation (Baldwin/Lamarckian)
	@State private var synaptogenesis = false
	@State private var neurogenesis = false
	@State private var adaptWarmup = 10
	@State private var adaptCooldown = 5

	// Add-phase form state
	@State private var newPhaseType: ExperimentType = .ga
	@State private var newPhaseNeurons = true
	@State private var newPhaseBits = false
	@State private var newPhaseConnections = false
	@State private var newPhaseGridSearch = false

	private var isBitwise: Bool { selectedTemplate == .bitwise7Phase }

	public init() {}

	public var body: some View {
		NavigationStack {
			Form {
				Section("Flow Details") {
					TextField("Name", text: $name)
					TextField("Description (optional)", text: $description, axis: .vertical).lineLimit(2...4)
				}

				Section("Template") {
					Picker("Template", selection: $selectedTemplate) {
						ForEach(FlowTemplate.allCases, id: \.self) { t in
							Text(t.displayName).tag(t)
						}
					}
					Picker("Phase Order", selection: $phaseOrder) {
						ForEach(PhaseOrder.allCases, id: \.self) { o in
							Text(o.displayName).tag(o)
						}
					}
					.disabled(selectedTemplate == .empty)
				}

				// Editable phases list
				Section {
					if phases.isEmpty {
						Text("No phases. Use a template or add manually.")
							.foregroundStyle(.secondary)
							.italic()
					} else {
						ForEach(phases) { phase in
							HStack(spacing: 8) {
								Text(phase.name)
								Spacer()
								Text(phase.experimentType.displayName)
									.fontWeight(.semibold)
									.padding(.horizontal, 8)
									.padding(.vertical, 4)
									.background(phase.experimentType == .ga ? Color.blue.opacity(0.15) : Color.green.opacity(0.15))
									.foregroundStyle(phase.experimentType == .ga ? .blue : .green)
									.clipShape(RoundedRectangle(cornerRadius: 4))
							}
						}
						.onMove { phases.move(fromOffsets: $0, toOffset: $1) }
						.onDelete { phases.remove(atOffsets: $0) }
					}
				} header: {
					HStack {
						Text("Phases (\(phases.count))")
						Spacer()
						#if os(iOS)
						if !phases.isEmpty {
							EditButton().textCase(nil)
						}
						#endif
					}
				}

				// Add Phase inline form
				Section("Add Phase") {
					if isBitwise {
						Toggle("Grid Search", isOn: $newPhaseGridSearch)
					}
					if !newPhaseGridSearch {
						Picker("Type", selection: $newPhaseType) {
							Text("GA").tag(ExperimentType.ga)
							Text("TS").tag(ExperimentType.ts)
						}
						.pickerStyle(.segmented)
						Toggle("Neurons", isOn: $newPhaseNeurons)
						Toggle("Bits", isOn: $newPhaseBits)
						Toggle("Connections", isOn: $newPhaseConnections)
					}
					Button("Add Phase") { addPhase() }
						.disabled(!newPhaseGridSearch && !newPhaseNeurons && !newPhaseBits && !newPhaseConnections)
				}

				Section("Optimization Parameters") {
					Stepper("GA Generations: \(gaGenerations)", value: $gaGenerations, in: 10...2000, step: 50)
					Stepper("TS Iterations: \(tsIterations)", value: $tsIterations, in: 10...2000, step: 50)
					Stepper("Patience: \(patience)", value: $patience, in: 1...50)
					Stepper("Population: \(population)", value: $population, in: 10...200, step: 10)
				}

				if isBitwise {
					BitwiseConfigEditorView(
						numClusters: $bitwiseNumClusters,
						minBits: $bitwiseMinBits,
						maxBits: $bitwiseMaxBits,
						minNeurons: $bitwiseMinNeurons,
						maxNeurons: $bitwiseMaxNeurons,
						memoryMode: $bitwiseMemoryMode,
						neuronSampleRate: $bitwiseNeuronSampleRate,
						synaptogenesis: $synaptogenesis,
						neurogenesis: $neurogenesis,
						adaptWarmup: $adaptWarmup,
						adaptCooldown: $adaptCooldown
					)
				} else {
					Section("Tier Configuration") {
						#if os(iOS)
						TextField("Tier Config", text: $tierConfig).fontDesign(.monospaced).textInputAutocapitalization(.never)
						#else
						TextField("Tier Config", text: $tierConfig).fontDesign(.monospaced)
						#endif
						Toggle("Tier 0 Only", isOn: $tier0Only)
						Text("Format: clusters,neurons,bits;...").foregroundColor(.secondary)
					}
				}

				if let error = error { Section { Text(error).foregroundColor(.red) } }
			}
			.navigationTitle("New Flow")
			#if os(iOS)
			.navigationBarTitleDisplayMode(.inline)
			.toolbar {
				ToolbarItem(placement: .navigationBarLeading) { Button("Cancel") { dismiss() } }
				ToolbarItem(placement: .navigationBarTrailing) { Button("Create") { createFlow() }.disabled(name.isEmpty || isSubmitting) }
			}
			#else
			.toolbar {
				ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
				ToolbarItem(placement: .confirmationAction) { Button("Create") { createFlow() }.disabled(name.isEmpty || isSubmitting) }
			}
			#endif
			.disabled(isSubmitting)
			.overlay { if isSubmitting { ProgressView("Creating flow...").padding().background(.ultraThinMaterial).cornerRadius(12) } }
			.onAppear { phases = generatePhases(selectedTemplate, phaseOrder) }
			.onChange(of: selectedTemplate) { _, newValue in phases = generatePhases(newValue, phaseOrder) }
			.onChange(of: phaseOrder) { _, newValue in phases = generatePhases(selectedTemplate, newValue) }
		}
	}

	// MARK: - Phase generation from templates

	private func generatePhases(_ template: FlowTemplate, _ order: PhaseOrder) -> [EditablePhase] {
		switch template {
		case .empty: return []
		case .quick4Phase:
			return order == .bitsFirst ? bitsPhases() + neuronsPhases() : neuronsPhases() + bitsPhases()
		case .standard6Phase:
			let c = connectionsPhases()
			return order == .bitsFirst ? bitsPhases() + neuronsPhases() + c : neuronsPhases() + bitsPhases() + c
		case .bitwise7Phase:
			let grid = [EditablePhase(name: "Grid Search (neurons × bits)", experimentType: .ga,
									  optimizeBits: true, optimizeNeurons: true, optimizeConnections: false, isGridSearch: true)]
			let c = connectionsPhases()
			return order == .bitsFirst ? grid + bitsPhases() + neuronsPhases() + c : grid + neuronsPhases() + bitsPhases() + c
		}
	}

	private func neuronsPhases() -> [EditablePhase] {
		[
			EditablePhase(name: "GA Neurons", experimentType: .ga,
						  optimizeBits: false, optimizeNeurons: true, optimizeConnections: false, isGridSearch: false),
			EditablePhase(name: "TS Neurons (refine)", experimentType: .ts,
						  optimizeBits: false, optimizeNeurons: true, optimizeConnections: false, isGridSearch: false),
		]
	}

	private func bitsPhases() -> [EditablePhase] {
		[
			EditablePhase(name: "GA Bits", experimentType: .ga,
						  optimizeBits: true, optimizeNeurons: false, optimizeConnections: false, isGridSearch: false),
			EditablePhase(name: "TS Bits (refine)", experimentType: .ts,
						  optimizeBits: true, optimizeNeurons: false, optimizeConnections: false, isGridSearch: false),
		]
	}

	private func connectionsPhases() -> [EditablePhase] {
		[
			EditablePhase(name: "GA Connections", experimentType: .ga,
						  optimizeBits: false, optimizeNeurons: false, optimizeConnections: true, isGridSearch: false),
			EditablePhase(name: "TS Connections (refine)", experimentType: .ts,
						  optimizeBits: false, optimizeNeurons: false, optimizeConnections: true, isGridSearch: false),
		]
	}

	// MARK: - Add phase

	private func generatePhaseName(type: ExperimentType, neurons: Bool, bits: Bool, connections: Bool) -> String {
		var targets: [String] = []
		if neurons { targets.append("Neurons") }
		if bits { targets.append("Bits") }
		if connections { targets.append("Connections") }
		return "\(type.displayName) \(targets.joined(separator: " + "))"
	}

	private func addPhase() {
		if newPhaseGridSearch {
			phases.append(EditablePhase(
				name: "Grid Search (neurons × bits)", experimentType: .ga,
				optimizeBits: true, optimizeNeurons: true, optimizeConnections: false, isGridSearch: true
			))
		} else {
			guard newPhaseNeurons || newPhaseBits || newPhaseConnections else { return }
			phases.append(EditablePhase(
				name: generatePhaseName(type: newPhaseType, neurons: newPhaseNeurons, bits: newPhaseBits, connections: newPhaseConnections),
				experimentType: newPhaseType,
				optimizeBits: newPhaseBits, optimizeNeurons: newPhaseNeurons, optimizeConnections: newPhaseConnections,
				isGridSearch: false
			))
		}
	}

	// MARK: - Create flow

	private func createFlow() {
		isSubmitting = true; error = nil

		var params: [String: AnyCodable] = [
			"ga_gens": AnyCodable(gaGenerations),
			"ts_iters": AnyCodable(tsIterations),
			"patience": AnyCodable(patience),
			"population": AnyCodable(population),
			"phase_order": AnyCodable(phaseOrder.rawValue),
		]

		if isBitwise {
			params["architecture_type"] = AnyCodable("bitwise")
			params["num_clusters"] = AnyCodable(bitwiseNumClusters)
			params["min_bits"] = AnyCodable(bitwiseMinBits)
			params["max_bits"] = AnyCodable(bitwiseMaxBits)
			params["min_neurons"] = AnyCodable(bitwiseMinNeurons)
			params["max_neurons"] = AnyCodable(bitwiseMaxNeurons)
			params["memory_mode"] = AnyCodable(bitwiseMemoryMode)
			params["neuron_sample_rate"] = AnyCodable(bitwiseNeuronSampleRate)
			// Adaptation (Baldwin/Lamarckian)
			if synaptogenesis || neurogenesis {
				params["synaptogenesis"] = AnyCodable(synaptogenesis)
				params["neurogenesis"] = AnyCodable(neurogenesis)
				params["adapt_warmup"] = AnyCodable(adaptWarmup)
				params["adapt_cooldown"] = AnyCodable(adaptCooldown)
			}
		} else {
			params["tier_config"] = AnyCodable(tierConfig)
			params["tier0_only"] = AnyCodable(tier0Only)
		}

		let experimentSpecs = phases.map { $0.toExperimentSpec() }
		let config = FlowConfig(experiments: experimentSpecs, template: selectedTemplate.rawValue, params: params)
		let request = CreateFlowRequest(name: name, description: description.isEmpty ? nil : description, config: config)
		Task {
			do { _ = try await viewModel.createFlow(request); await MainActor.run { dismiss() } }
			catch { await MainActor.run { self.error = error.localizedDescription; self.isSubmitting = false } }
		}
	}
}

// MARK: - Supporting enums

enum PhaseOrder: String, CaseIterable {
	case neuronsFirst = "neurons_first"
	case bitsFirst = "bits_first"

	var displayName: String {
		switch self {
		case .neuronsFirst: return "Neurons First"
		case .bitsFirst: return "Bits First"
		}
	}
}

enum FlowTemplate: String, CaseIterable {
	case standard6Phase = "standard-6-phase"
	case quick4Phase = "quick-4-phase"
	case bitwise7Phase = "bitwise-7-phase"
	case empty = "empty"

	var displayName: String {
		switch self {
		case .standard6Phase: return "Standard 6-Phase (Tiered)"
		case .quick4Phase: return "Quick 4-Phase (Tiered)"
		case .bitwise7Phase: return "Bitwise 7-Phase"
		case .empty: return "Empty (manual)"
		}
	}
}
