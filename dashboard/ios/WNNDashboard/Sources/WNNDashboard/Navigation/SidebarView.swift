// SidebarView - iPad sidebar navigation

import SwiftUI

public struct SidebarView: View {
    @Binding public var selection: NavigationDestination?
    @EnvironmentObject var connectionManager: ConnectionManager
    @EnvironmentObject var wsManager: WebSocketManager

    public init(selection: Binding<NavigationDestination?>) {
        self._selection = selection
    }

    public var body: some View {
        List(selection: $selection) {
            ForEach(NavigationDestination.allCases) { destination in
                Label(destination.title, systemImage: destination.icon)
                    .tag(destination)
            }
        }
        .navigationTitle("WNN Dashboard")
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                connectionIndicator
            }
        }
    }

    private var connectionIndicator: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(wsManager.connectionState.isConnected ? Color.green : Color.red)
                .frame(width: 8, height: 8)
            Text(wsManager.connectionState.isConnected ? "Live" : "Offline")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}
