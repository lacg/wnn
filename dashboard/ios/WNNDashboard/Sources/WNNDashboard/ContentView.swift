// ContentView - Root tab navigation

import SwiftUI

public struct ContentView: View {
    @EnvironmentObject var connectionManager: ConnectionManager
    @EnvironmentObject var wsManager: WebSocketManager

    @State private var selectedTab = 0

    public init() {}

    public var body: some View {
        TabView(selection: $selectedTab) {
            DashboardView()
                .tabItem {
                    Label("Iterations", systemImage: "chart.line.uptrend.xyaxis")
                }
                .tag(0)

            FlowsListView()
                .tabItem {
                    Label("Flows", systemImage: "arrow.triangle.branch")
                }
                .tag(1)

            CheckpointsListView()
                .tabItem {
                    Label("Checkpoints", systemImage: "externaldrive")
                }
                .tag(2)

            SettingsView()
                .tabItem {
                    Label("Settings", systemImage: "gear")
                }
                .tag(3)
        }
        .overlay(alignment: .top) {
            ConnectionStatusBar()
        }
    }
}

struct ConnectionStatusBar: View {
    @EnvironmentObject var connectionManager: ConnectionManager
    @EnvironmentObject var wsManager: WebSocketManager

    var body: some View {
        if !connectionManager.connectionState.isConnected || !wsManager.connectionState.isConnected {
            HStack(spacing: 8) {
                if connectionManager.connectionState == .connecting ||
                   wsManager.connectionState == .connecting ||
                   wsManager.connectionState == .reconnecting {
                    ProgressView()
                        .scaleEffect(0.8)
                }

                Text(statusMessage)
                    .font(.caption)

                Spacer()

                if let error = connectionManager.lastError ?? wsManager.lastError {
                    Text(error)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)
            .background(statusColor.opacity(0.2))
            .foregroundColor(statusColor)
        }
    }

    private var statusMessage: String {
        if !connectionManager.connectionState.isConnected {
            return connectionManager.connectionState.displayName
        } else {
            return "WebSocket: \(wsManager.connectionState.displayName)"
        }
    }

    private var statusColor: Color {
        switch (connectionManager.connectionState, wsManager.connectionState) {
        case (.connecting, _), (_, .connecting), (_, .reconnecting):
            return .orange
        case (.disconnected, _), (_, .disconnected):
            return .red
        default:
            return .green
        }
    }
}
