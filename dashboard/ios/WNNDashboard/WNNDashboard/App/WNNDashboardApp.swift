// WNNDashboardApp - App entry point

import SwiftUI

@main
struct WNNDashboardApp: App {
    // MARK: - State Objects

    @StateObject private var settings = SettingsStore()
    @StateObject private var connectionManager: ConnectionManager
    @StateObject private var wsManager: WebSocketManager
    @StateObject private var apiClient: APIClient

    // ViewModels
    @StateObject private var dashboardVM: DashboardViewModel
    @StateObject private var flowsVM: FlowsViewModel
    @StateObject private var checkpointsVM: CheckpointsViewModel

    // MARK: - Initialization

    init() {
        // Create services
        let settings = SettingsStore()
        let connectionManager = ConnectionManager(settings: settings)
        let wsManager = WebSocketManager(connectionManager: connectionManager)
        let apiClient = APIClient(connectionManager: connectionManager)

        // Create ViewModels
        let dashboardVM = DashboardViewModel(apiClient: apiClient, wsManager: wsManager)
        let flowsVM = FlowsViewModel(apiClient: apiClient, wsManager: wsManager)
        let checkpointsVM = CheckpointsViewModel(apiClient: apiClient, wsManager: wsManager)

        // Initialize state objects
        _settings = StateObject(wrappedValue: settings)
        _connectionManager = StateObject(wrappedValue: connectionManager)
        _wsManager = StateObject(wrappedValue: wsManager)
        _apiClient = StateObject(wrappedValue: apiClient)
        _dashboardVM = StateObject(wrappedValue: dashboardVM)
        _flowsVM = StateObject(wrappedValue: flowsVM)
        _checkpointsVM = StateObject(wrappedValue: checkpointsVM)
    }

    // MARK: - Body

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(settings)
                .environmentObject(connectionManager)
                .environmentObject(wsManager)
                .environmentObject(dashboardVM)
                .environmentObject(flowsVM)
                .environmentObject(checkpointsVM)
                .task {
                    await setupConnection()
                }
        }
    }

    // MARK: - Setup

    @MainActor
    private func setupConnection() async {
        // Connect to backend
        await connectionManager.connect()

        if connectionManager.connectionState.isConnected {
            // Connect WebSocket
            wsManager.connect()

            // Load initial data
            await dashboardVM.loadSnapshot()
            await flowsVM.loadFlows()
        }
    }
}
