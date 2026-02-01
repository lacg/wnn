// WNNDashboardApp - App entry point

import SwiftUI

@main
public struct WNNDashboardApp: App {
    @StateObject private var appState = AppState()

    public init() {}

    public var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appState.settings)
                .environmentObject(appState.connectionManager)
                .environmentObject(appState.wsManager)
                .environmentObject(appState.dashboardVM)
                .environmentObject(appState.flowsVM)
                .environmentObject(appState.checkpointsVM)
                .task {
                    await appState.connect()
                }
        }
    }
}

/// Centralized app state to avoid initialization order issues
@MainActor
public final class AppState: ObservableObject {
    let settings: SettingsStore
    let connectionManager: ConnectionManager
    let wsManager: WebSocketManager
    let apiClient: APIClient
    let dashboardVM: DashboardViewModel
    let flowsVM: FlowsViewModel
    let checkpointsVM: CheckpointsViewModel

    public init() {
        let settings = SettingsStore()
        let connectionManager = ConnectionManager(settings: settings)
        let wsManager = WebSocketManager(connectionManager: connectionManager)
        let apiClient = APIClient(connectionManager: connectionManager)

        self.settings = settings
        self.connectionManager = connectionManager
        self.wsManager = wsManager
        self.apiClient = apiClient
        self.dashboardVM = DashboardViewModel(apiClient: apiClient, wsManager: wsManager)
        self.flowsVM = FlowsViewModel(apiClient: apiClient, wsManager: wsManager)
        self.checkpointsVM = CheckpointsViewModel(apiClient: apiClient, wsManager: wsManager)
    }

    func connect() async {
        await connectionManager.connect()
        if connectionManager.connectionState.isConnected {
            wsManager.connect()
            await dashboardVM.loadSnapshot()
            await flowsVM.loadFlows()
        }
    }
}
