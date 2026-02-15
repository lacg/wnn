// NewFlowView - Form for creating a new flow

import SwiftUI

public struct NewFlowView: View {
    @EnvironmentObject var viewModel: FlowsViewModel
    @Environment(\.dismiss) private var dismiss

    @State private var name = ""
    @State private var description = ""
    @State private var selectedTemplate: FlowTemplate = .standard6Phase
    @State private var gaGenerations = 100
    @State private var tsIterations = 200
    @State private var patience = 10
    @State private var population = 50
    @State private var tierConfig = "100,15,20;400,10,12;rest,5,8"
    @State private var tier0Only = true
    @State private var isSubmitting = false
    @State private var error: String?

    // Bitwise-specific config
    @State private var bitwiseNumClusters = 16
    @State private var bitwiseMinBits = 10
    @State private var bitwiseMaxBits = 24
    @State private var bitwiseMinNeurons = 10
    @State private var bitwiseMaxNeurons = 300
    @State private var bitwiseMemoryMode = "QUAD_WEIGHTED"
    @State private var bitwiseNeuronSampleRate = 0.25

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
                    Picker("Template", selection: $selectedTemplate) { ForEach(FlowTemplate.allCases, id: \.self) { t in Text(t.displayName).tag(t) } }
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
                        neuronSampleRate: $bitwiseNeuronSampleRate
                    )
                } else {
                    Section("Tier Configuration") {
                        #if os(iOS)
                        TextField("Tier Config", text: $tierConfig).fontDesign(.monospaced).textInputAutocapitalization(.never)
                        #else
                        TextField("Tier Config", text: $tierConfig).fontDesign(.monospaced)
                        #endif
                        Toggle("Tier 0 Only", isOn: $tier0Only)
                        Text("Format: clusters,neurons,bits;...").font(.caption).foregroundColor(.secondary)
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
        }
    }

    private func createFlow() {
        isSubmitting = true; error = nil

        var params: [String: AnyCodable] = [
            "ga_gens": AnyCodable(gaGenerations),
            "ts_iters": AnyCodable(tsIterations),
            "patience": AnyCodable(patience),
            "population": AnyCodable(population),
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
        } else {
            params["tier_config"] = AnyCodable(tierConfig)
            params["tier0_only"] = AnyCodable(tier0Only)
        }

        let config = FlowConfig(experiments: selectedTemplate.defaultExperiments, template: selectedTemplate.rawValue, params: params)
        let request = CreateFlowRequest(name: name, description: description.isEmpty ? nil : description, config: config)
        Task {
            do { _ = try await viewModel.createFlow(request); await MainActor.run { dismiss() } }
            catch { await MainActor.run { self.error = error.localizedDescription; self.isSubmitting = false } }
        }
    }
}

enum FlowTemplate: String, CaseIterable {
    case standard6Phase = "standard-6-phase"
    case bitwise7Phase = "bitwise-7-phase"
    case neuronsOnly = "neurons-only"
    case bitsOnly = "bits-only"
    case fullOptimization = "full-optimization"

    var displayName: String {
        switch self {
        case .standard6Phase: return "Standard 6-Phase (Tiered)"
        case .bitwise7Phase: return "Bitwise 7-Phase"
        case .neuronsOnly: return "Neurons Only"
        case .bitsOnly: return "Bits Only"
        case .fullOptimization: return "Full Optimization"
        }
    }

    var defaultExperiments: [ExperimentSpec] {
        switch self {
        case .standard6Phase: return [
            ExperimentSpec(name: "1a GA Neurons", experiment_type: .ga, optimize_neurons: true),
            ExperimentSpec(name: "1b TS Neurons", experiment_type: .ts, optimize_neurons: true),
            ExperimentSpec(name: "2a GA Bits", experiment_type: .ga, optimize_bits: true),
            ExperimentSpec(name: "2b TS Bits", experiment_type: .ts, optimize_bits: true),
            ExperimentSpec(name: "3a GA Connections", experiment_type: .ga, optimize_connections: true),
            ExperimentSpec(name: "3b TS Connections", experiment_type: .ts, optimize_connections: true)]
        case .bitwise7Phase: return [
            ExperimentSpec(name: "1 Grid Search", experiment_type: .ga, optimize_bits: true, optimize_neurons: true, params: ["phase_type": AnyCodable("grid_search")]),
            ExperimentSpec(name: "2 GA Neurons", experiment_type: .ga, optimize_neurons: true),
            ExperimentSpec(name: "3 TS Neurons", experiment_type: .ts, optimize_neurons: true),
            ExperimentSpec(name: "4 GA Bits", experiment_type: .ga, optimize_bits: true),
            ExperimentSpec(name: "5 TS Bits", experiment_type: .ts, optimize_bits: true),
            ExperimentSpec(name: "6 GA Connections", experiment_type: .ga, optimize_connections: true),
            ExperimentSpec(name: "7 TS Connections", experiment_type: .ts, optimize_connections: true)]
        case .neuronsOnly: return [ExperimentSpec(name: "GA Neurons", experiment_type: .ga, optimize_neurons: true), ExperimentSpec(name: "TS Neurons", experiment_type: .ts, optimize_neurons: true)]
        case .bitsOnly: return [ExperimentSpec(name: "GA Bits", experiment_type: .ga, optimize_bits: true), ExperimentSpec(name: "TS Bits", experiment_type: .ts, optimize_bits: true)]
        case .fullOptimization: return [ExperimentSpec(name: "Full GA", experiment_type: .ga, optimize_bits: true, optimize_neurons: true, optimize_connections: true), ExperimentSpec(name: "Full TS", experiment_type: .ts, optimize_bits: true, optimize_neurons: true, optimize_connections: true)]
        }
    }
}
