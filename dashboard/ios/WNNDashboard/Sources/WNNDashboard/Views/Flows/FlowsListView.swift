// FlowsListView - List of all flows

import SwiftUI

public struct FlowsListView: View {
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @EnvironmentObject var viewModel: FlowsViewModel
    @State private var selectedFlowForDetail: Flow?

    public init() {}

    public var body: some View {
        Group {
            if horizontalSizeClass == .regular {
                iPadLayout
            } else {
                iPhoneLayout
            }
        }
        .sheet(isPresented: $viewModel.showingNewFlowSheet) { NewFlowView() }
    }

    // MARK: - iPhone Layout

    private var iPhoneLayout: some View {
        NavigationStack {
            flowsContent
                .navigationTitle("Flows")
                .toolbar {
                    ToolbarItem(placement: .navigationBarTrailing) { Button { viewModel.showingNewFlowSheet = true } label: { Image(systemName: "plus") } }
                    ToolbarItem(placement: .navigationBarLeading) { filterMenu }
                }
                .refreshable { await viewModel.refresh() }
                .sheet(item: $viewModel.selectedFlow) { flow in FlowDetailView(flow: flow) }
        }
    }

    // MARK: - iPad Layout (Master-Detail)

    private var iPadLayout: some View {
        NavigationSplitView {
            flowsContent
                .navigationTitle("Flows")
                .toolbar {
                    ToolbarItem(placement: .navigationBarTrailing) { Button { viewModel.showingNewFlowSheet = true } label: { Image(systemName: "plus") } }
                    ToolbarItem(placement: .navigationBarLeading) { filterMenu }
                }
                .refreshable { await viewModel.refresh() }
        } detail: {
            if let flow = selectedFlowForDetail {
                FlowDetailContentView(flow: flow)
            } else {
                VStack(spacing: 16) {
                    Image(systemName: "arrow.triangle.branch")
                        .font(.system(size: 48))
                        .foregroundColor(.secondary)
                    Text("Select a Flow")
                        .font(.headline)
                    Text("Choose a flow from the list to see its details")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
            }
        }
    }

    @ViewBuilder
    private var flowsContent: some View {
        if viewModel.isLoading && viewModel.flows.isEmpty {
            ProgressView("Loading flows...")
        } else if viewModel.flows.isEmpty {
            emptyState
        } else {
            flowsList
        }
    }

    private var flowsList: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                if !viewModel.runningFlows.isEmpty {
                    Section { ForEach(viewModel.runningFlows) { f in FlowCardView(flow: f) { selectFlow(f) } } } header: { sectionHeader("Running", count: viewModel.runningFlows.count) }
                }
                Section { ForEach(viewModel.filteredFlows) { f in FlowCardView(flow: f) { selectFlow(f) } } } header: { sectionHeader("All Flows", count: viewModel.filteredFlows.count) }
            }
            .padding()
        }
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "arrow.triangle.branch").font(.system(size: 48)).foregroundColor(.secondary)
            Text("No Flows").font(.headline)
            Text("Create a flow to start running experiments").font(.subheadline).foregroundColor(.secondary)
            Button("Create Flow") { viewModel.showingNewFlowSheet = true }.buttonStyle(.borderedProminent)
        }
        .padding()
    }

    private var filterMenu: some View {
        Menu {
            Button { viewModel.statusFilter = nil } label: { Label("All", systemImage: viewModel.statusFilter == nil ? "checkmark" : "") }
            Divider()
            ForEach(FlowStatus.allCases, id: \.self) { s in Button { viewModel.statusFilter = s } label: { Label(s.displayName, systemImage: viewModel.statusFilter == s ? "checkmark" : "") } }
        } label: { Image(systemName: "line.3.horizontal.decrease.circle") }
    }

    private func sectionHeader(_ title: String, count: Int) -> some View {
        HStack { Text(title).font(.subheadline).fontWeight(.semibold).foregroundColor(.secondary); Text("(\(count))").font(.caption).foregroundColor(.secondary); Spacer() }.padding(.top, 8)
    }

    private func selectFlow(_ flow: Flow) {
        if horizontalSizeClass == .regular {
            selectedFlowForDetail = flow
        } else {
            viewModel.selectedFlow = flow
        }
        Task { await viewModel.loadFlowExperiments(flow.id) }
    }
}

// MARK: - Flow Detail Content (without NavigationStack wrapper for iPad)

struct FlowDetailContentView: View {
    let flow: Flow
    @EnvironmentObject var viewModel: FlowsViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                statusSection
                configSection
                experimentsSection
                timingSection
            }
            .padding()
        }
        .navigationTitle(flow.name)
    }

    private var statusSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack { Text("Status").font(.headline); Spacer(); StatusBadge(text: flow.status.displayName, color: Theme.statusColor(flow.status)) }
            if let desc = flow.description { Text(desc).font(.subheadline).foregroundColor(.secondary) }
            if let pid = flow.pid, flow.status == .running { HStack { Text("Process ID").foregroundColor(.secondary); Spacer(); Text("\(pid)").fontDesign(.monospaced) }.font(.subheadline) }
        }
        .padding().background(Theme.cardBackground).cornerRadius(12)
    }

    private var configSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Configuration").font(.headline)
            if let t = flow.config.template { configRow("Template", value: t) }
            configRow("Experiments", value: "\(flow.config.experiments.count)")
            if let c = flow.seed_checkpoint_id { configRow("Seed Checkpoint", value: "#\(c)") }
        }
        .padding().background(Theme.cardBackground).cornerRadius(12)
    }

    private var experimentsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack { Text("Experiment Specs").font(.headline); Spacer(); Text("\(flow.config.experiments.count)").font(.caption).foregroundColor(.secondary) }
            ForEach(Array(flow.config.experiments.enumerated()), id: \.offset) { i, spec in
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("\(i + 1). \(spec.name)").font(.subheadline).fontWeight(.medium)
                        Spacer()
                        Text(spec.experiment_type.displayName).font(.caption).padding(.horizontal, 6).padding(.vertical, 2).background(spec.experiment_type == .ga ? Color.blue.opacity(0.2) : Color.purple.opacity(0.2)).foregroundColor(spec.experiment_type == .ga ? .blue : .purple).cornerRadius(4)
                    }
                    HStack(spacing: 8) {
                        if spec.optimize_neurons { badge("Neurons") }
                        if spec.optimize_bits { badge("Bits") }
                        if spec.optimize_connections { badge("Connections") }
                    }
                }
                .padding().background(Color.gray.opacity(0.1)).cornerRadius(8)
            }
            if !viewModel.selectedFlowExperiments.isEmpty {
                Divider().padding(.vertical, 8)
                Text("Completed Experiments").font(.subheadline).foregroundColor(.secondary)
                ForEach(viewModel.selectedFlowExperiments) { exp in
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(exp.name).font(.subheadline)
                            if let iter = exp.last_iteration { Text("Iteration \(iter)").font(.caption).foregroundColor(.secondary) }
                        }
                        Spacer()
                        StatusBadge(text: exp.status.displayName, color: Theme.statusColor(exp.status))
                    }
                    .padding(.vertical, 4)
                }
            }
        }
        .padding().background(Theme.cardBackground).cornerRadius(12)
    }

    private var timingSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Timing").font(.headline)
            if let d = flow.createdDate { configRow("Created", value: DateFormatters.shortDateTime(d)) }
            if let d = flow.startedDate { configRow("Started", value: DateFormatters.shortDateTime(d)) }
            if let d = flow.completedDate { configRow("Completed", value: DateFormatters.shortDateTime(d)) }
            if let dur = flow.duration { configRow("Duration", value: DateFormatters.duration(dur)) }
        }
        .padding().background(Theme.cardBackground).cornerRadius(12)
    }

    private func configRow(_ label: String, value: String) -> some View {
        HStack { Text(label).foregroundColor(.secondary); Spacer(); Text(value).fontDesign(.monospaced) }.font(.subheadline)
    }

    private func badge(_ text: String) -> some View {
        Text(text).font(.caption2).padding(.horizontal, 6).padding(.vertical, 2).background(Color.green.opacity(0.2)).foregroundColor(.green).cornerRadius(4)
    }
}
