// NewFlowView - Form for creating a new flow

import SwiftUI

struct NewFlowView: View {
    @EnvironmentObject var viewModel: FlowsViewModel
    @Environment(\.dismiss) private var dismiss

    // Form state
    @State private var name = ""
    @State private var description = ""
    @State private var selectedTemplate: FlowTemplate = .standard6Phase
    @State private var useCustomExperiments = false
    @State private var experiments: [ExperimentFormSpec] = []

    // GA/TS parameters
    @State private var gaGenerations = 100
    @State private var tsIterations = 200
    @State private var patience = 10
    @State private var population = 50

    // Tier config
    @State private var tierConfig = "100,15,20;400,10,12;rest,5,8"
    @State private var tier0Only = true

    @State private var isSubmitting = false
    @State private var error: String?

    var body: some View {
        NavigationStack {
            Form {
                // Basic info
                Section("Flow Details") {
                    TextField("Name", text: $name)

                    TextField("Description (optional)", text: $description, axis: .vertical)
                        .lineLimit(2...4)
                }

                // Template selection
                Section("Template") {
                    Picker("Template", selection: $selectedTemplate) {
                        ForEach(FlowTemplate.allCases, id: \.self) { template in
                            Text(template.displayName).tag(template)
                        }
                    }

                    Toggle("Customize Experiments", isOn: $useCustomExperiments)
                }

                // Custom experiments (if enabled)
                if useCustomExperiments {
                    Section("Experiments") {
                        ForEach($experiments) { $spec in
                            ExperimentSpecEditor(spec: $spec)
                        }
                        .onDelete { indices in
                            experiments.remove(atOffsets: indices)
                        }

                        Button {
                            experiments.append(ExperimentFormSpec())
                        } label: {
                            Label("Add Experiment", systemImage: "plus")
                        }
                    }
                }

                // Optimization parameters
                Section("Optimization Parameters") {
                    Stepper("GA Generations: \(gaGenerations)", value: $gaGenerations, in: 10...2000, step: 50)

                    Stepper("TS Iterations: \(tsIterations)", value: $tsIterations, in: 10...2000, step: 50)

                    Stepper("Patience: \(patience)", value: $patience, in: 1...50)

                    Stepper("Population: \(population)", value: $population, in: 10...200, step: 10)
                }

                // Tier configuration
                Section("Tier Configuration") {
                    TextField("Tier Config", text: $tierConfig)
                        .fontDesign(.monospaced)
                        .autocapitalization(.none)

                    Toggle("Tier 0 Only", isOn: $tier0Only)

                    Text("Format: clusters,neurons,bits;...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                // Error display
                if let error = error {
                    Section {
                        Text(error)
                            .foregroundColor(.red)
                    }
                }
            }
            .navigationTitle("New Flow")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") { dismiss() }
                }

                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Create") {
                        createFlow()
                    }
                    .disabled(name.isEmpty || isSubmitting)
                }
            }
            .disabled(isSubmitting)
            .overlay {
                if isSubmitting {
                    ProgressView("Creating flow...")
                        .padding()
                        .background(.ultraThinMaterial)
                        .cornerRadius(12)
                }
            }
        }
    }

    // MARK: - Actions

    private func createFlow() {
        isSubmitting = true
        error = nil

        let config = buildFlowConfig()
        let request = CreateFlowRequest(
            name: name,
            description: description.isEmpty ? nil : description,
            config: config
        )

        Task {
            do {
                _ = try await viewModel.createFlow(request)
                await MainActor.run {
                    dismiss()
                }
            } catch {
                await MainActor.run {
                    self.error = error.localizedDescription
                    self.isSubmitting = false
                }
            }
        }
    }

    private func buildFlowConfig() -> FlowConfig {
        let experimentSpecs: [ExperimentSpec]

        if useCustomExperiments {
            experimentSpecs = experiments.map { spec in
                ExperimentSpec(
                    name: spec.name,
                    experiment_type: spec.type,
                    optimize_bits: spec.optimizeBits,
                    optimize_neurons: spec.optimizeNeurons,
                    optimize_connections: spec.optimizeConnections
                )
            }
        } else {
            experimentSpecs = selectedTemplate.defaultExperiments
        }

        let params: [String: AnyCodable] = [
            "ga_gens": AnyCodable(gaGenerations),
            "ts_iters": AnyCodable(tsIterations),
            "patience": AnyCodable(patience),
            "population": AnyCodable(population),
            "tier_config": AnyCodable(tierConfig),
            "tier0_only": AnyCodable(tier0Only)
        ]

        return FlowConfig(
            experiments: experimentSpecs,
            template: useCustomExperiments ? nil : selectedTemplate.rawValue,
            params: params
        )
    }
}

// MARK: - Flow Templates

enum FlowTemplate: String, CaseIterable {
    case standard6Phase = "standard-6-phase"
    case neuronsOnly = "neurons-only"
    case bitsOnly = "bits-only"
    case fullOptimization = "full-optimization"

    var displayName: String {
        switch self {
        case .standard6Phase: return "Standard 6-Phase"
        case .neuronsOnly: return "Neurons Only"
        case .bitsOnly: return "Bits Only"
        case .fullOptimization: return "Full Optimization"
        }
    }

    var defaultExperiments: [ExperimentSpec] {
        switch self {
        case .standard6Phase:
            return [
                ExperimentSpec(name: "1a GA Neurons", experiment_type: .ga, optimize_neurons: true),
                ExperimentSpec(name: "1b TS Neurons", experiment_type: .ts, optimize_neurons: true),
                ExperimentSpec(name: "2a GA Bits", experiment_type: .ga, optimize_bits: true),
                ExperimentSpec(name: "2b TS Bits", experiment_type: .ts, optimize_bits: true),
                ExperimentSpec(name: "3a GA Connections", experiment_type: .ga, optimize_connections: true),
                ExperimentSpec(name: "3b TS Connections", experiment_type: .ts, optimize_connections: true),
            ]
        case .neuronsOnly:
            return [
                ExperimentSpec(name: "GA Neurons", experiment_type: .ga, optimize_neurons: true),
                ExperimentSpec(name: "TS Neurons", experiment_type: .ts, optimize_neurons: true),
            ]
        case .bitsOnly:
            return [
                ExperimentSpec(name: "GA Bits", experiment_type: .ga, optimize_bits: true),
                ExperimentSpec(name: "TS Bits", experiment_type: .ts, optimize_bits: true),
            ]
        case .fullOptimization:
            return [
                ExperimentSpec(name: "Full GA", experiment_type: .ga, optimize_bits: true, optimize_neurons: true, optimize_connections: true),
                ExperimentSpec(name: "Full TS", experiment_type: .ts, optimize_bits: true, optimize_neurons: true, optimize_connections: true),
            ]
        }
    }
}

// MARK: - Experiment Form Spec

struct ExperimentFormSpec: Identifiable {
    let id = UUID()
    var name = "New Experiment"
    var type: ExperimentType = .ga
    var optimizeNeurons = false
    var optimizeBits = false
    var optimizeConnections = false
}

struct ExperimentSpecEditor: View {
    @Binding var spec: ExperimentFormSpec

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("Name", text: $spec.name)

            Picker("Type", selection: $spec.type) {
                Text("GA").tag(ExperimentType.ga)
                Text("TS").tag(ExperimentType.ts)
            }
            .pickerStyle(.segmented)

            HStack {
                Toggle("Neurons", isOn: $spec.optimizeNeurons)
                    .toggleStyle(.button)
                    .buttonStyle(.bordered)

                Toggle("Bits", isOn: $spec.optimizeBits)
                    .toggleStyle(.button)
                    .buttonStyle(.bordered)

                Toggle("Connections", isOn: $spec.optimizeConnections)
                    .toggleStyle(.button)
                    .buttonStyle(.bordered)
            }
            .font(.caption)
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    NewFlowView()
        .environmentObject(FlowsViewModel(
            apiClient: APIClient(connectionManager: ConnectionManager(settings: SettingsStore())),
            wsManager: WebSocketManager(connectionManager: ConnectionManager(settings: SettingsStore()))
        ))
}
