// FlowsListView - List of all flows with filtering

import SwiftUI

struct FlowsListView: View {
    @EnvironmentObject var viewModel: FlowsViewModel

    var body: some View {
        NavigationStack {
            Group {
                if viewModel.isLoading && viewModel.flows.isEmpty {
                    ProgressView("Loading flows...")
                } else if viewModel.flows.isEmpty {
                    emptyState
                } else {
                    flowsList
                }
            }
            .navigationTitle("Flows")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        viewModel.showingNewFlowSheet = true
                    } label: {
                        Image(systemName: "plus")
                    }
                }

                ToolbarItem(placement: .navigationBarLeading) {
                    filterMenu
                }
            }
            .refreshable {
                await viewModel.refresh()
            }
            .sheet(isPresented: $viewModel.showingNewFlowSheet) {
                NewFlowView()
            }
            .sheet(item: $viewModel.selectedFlow) { flow in
                FlowDetailView(flow: flow)
            }
        }
    }

    // MARK: - Components

    private var flowsList: some View {
        ScrollView {
            LazyVStack(spacing: 12) {
                // Running flows section
                if !viewModel.runningFlows.isEmpty {
                    Section {
                        ForEach(viewModel.runningFlows) { flow in
                            FlowCardView(flow: flow, onSelect: { selectFlow(flow) })
                        }
                    } header: {
                        sectionHeader("Running", count: viewModel.runningFlows.count)
                    }
                }

                // All flows section
                Section {
                    ForEach(viewModel.filteredFlows) { flow in
                        FlowCardView(flow: flow, onSelect: { selectFlow(flow) })
                    }
                } header: {
                    sectionHeader("All Flows", count: viewModel.filteredFlows.count)
                }
            }
            .padding()
        }
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "arrow.triangle.branch")
                .font(.system(size: 48))
                .foregroundColor(.secondary)

            Text("No Flows")
                .font(.headline)

            Text("Create a flow to start running experiments")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            Button("Create Flow") {
                viewModel.showingNewFlowSheet = true
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }

    private var filterMenu: some View {
        Menu {
            Button {
                viewModel.statusFilter = nil
            } label: {
                Label("All", systemImage: viewModel.statusFilter == nil ? "checkmark" : "")
            }

            Divider()

            ForEach(FlowStatus.allCases, id: \.self) { status in
                Button {
                    viewModel.statusFilter = status
                } label: {
                    Label(status.displayName, systemImage: viewModel.statusFilter == status ? "checkmark" : "")
                }
            }
        } label: {
            Image(systemName: "line.3.horizontal.decrease.circle")
        }
    }

    private func sectionHeader(_ title: String, count: Int) -> some View {
        HStack {
            Text(title)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundColor(.secondary)

            Text("(\(count))")
                .font(.caption)
                .foregroundColor(.secondary)

            Spacer()
        }
        .padding(.top, 8)
    }

    private func selectFlow(_ flow: Flow) {
        viewModel.selectedFlow = flow
        Task {
            await viewModel.loadFlowExperiments(flow.id)
        }
    }
}

#Preview {
    FlowsListView()
        .environmentObject(FlowsViewModel(
            apiClient: APIClient(connectionManager: ConnectionManager(settings: SettingsStore())),
            wsManager: WebSocketManager(connectionManager: ConnectionManager(settings: SettingsStore()))
        ))
}
