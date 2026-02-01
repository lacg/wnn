// SettingsView - Connection configuration

import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var settings: SettingsStore
    @EnvironmentObject var connectionManager: ConnectionManager
    @EnvironmentObject var wsManager: WebSocketManager

    @State private var isTesting = false
    @State private var testResults: [ConnectionMode: Bool] = [:]

    var body: some View {
        NavigationStack {
            Form {
                // Connection status
                Section("Connection Status") {
                    connectionStatusRow
                }

                // Connection mode
                Section {
                    Picker("Mode", selection: $settings.connectionMode) {
                        ForEach(ConnectionMode.allCases) { mode in
                            Label(mode.displayName, systemImage: mode.icon)
                                .tag(mode)
                        }
                    }
                    .pickerStyle(.inline)
                } header: {
                    Text("Connection Mode")
                } footer: {
                    Text(settings.connectionMode.description)
                }

                // Local server URL
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        TextField("http://192.168.1.100:3000", text: $settings.localServerURL)
                            .keyboardType(.URL)
                            .autocapitalization(.none)
                            .autocorrectionDisabled()
                            .fontDesign(.monospaced)

                        testConnectionButton(mode: .local)
                    }
                } header: {
                    Label("Local Server", systemImage: "wifi")
                } footer: {
                    Text("URL for when you're on the same network as your server (e.g., home WiFi)")
                }

                // Remote server URL
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        TextField("http://hostname:3000", text: $settings.remoteServerURL)
                            .keyboardType(.URL)
                            .autocapitalization(.none)
                            .autocorrectionDisabled()
                            .fontDesign(.monospaced)

                        testConnectionButton(mode: .remote)
                    }
                } header: {
                    Label("Remote Server", systemImage: "globe")
                } footer: {
                    Text("URL for when you're away from home (Tailscale, public URL, VPN, etc.)")
                }

                // Actions
                Section {
                    Button {
                        reconnect()
                    } label: {
                        Label("Reconnect", systemImage: "arrow.clockwise")
                    }

                    Button(role: .destructive) {
                        settings.resetToDefaults()
                        testResults = [:]
                    } label: {
                        Label("Reset to Defaults", systemImage: "arrow.counterclockwise")
                    }
                }

                // About
                Section("About") {
                    LabeledContent("Version", value: "1.0.0")

                    LabeledContent("Active Connection") {
                        HStack(spacing: 4) {
                            Image(systemName: connectionManager.activeMode.icon)
                            Text(connectionManager.activeMode.displayName)
                        }
                        .foregroundColor(.secondary)
                    }

                    if let url = connectionManager.baseURL {
                        LabeledContent("Connected To") {
                            Text(url.absoluteString)
                                .font(.caption)
                                .fontDesign(.monospaced)
                                .foregroundColor(.secondary)
                                .lineLimit(1)
                        }
                    }
                }
            }
            .navigationTitle("Settings")
        }
    }

    // MARK: - Components

    private var connectionStatusRow: some View {
        VStack(alignment: .leading, spacing: 12) {
            // HTTP connection
            HStack(spacing: 12) {
                statusIndicator(isConnected: connectionManager.connectionState.isConnected)

                VStack(alignment: .leading, spacing: 2) {
                    Text("HTTP")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(connectionManager.connectionState.displayName)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                if connectionManager.isOnWiFi {
                    Label("WiFi", systemImage: "wifi")
                        .font(.caption)
                        .foregroundColor(.blue)
                } else if connectionManager.isOnCellular {
                    Label("Cellular", systemImage: "antenna.radiowaves.left.and.right")
                        .font(.caption)
                        .foregroundColor(.orange)
                }
            }

            Divider()

            // WebSocket connection
            HStack(spacing: 12) {
                statusIndicator(isConnected: wsManager.connectionState.isConnected)

                VStack(alignment: .leading, spacing: 2) {
                    Text("WebSocket")
                        .font(.subheadline)
                        .fontWeight(.medium)
                    Text(wsManager.connectionState.displayName)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                Spacer()

                if wsManager.connectionState.isConnected {
                    Text("Live")
                        .font(.caption)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 2)
                        .background(Color.green.opacity(0.2))
                        .foregroundColor(.green)
                        .cornerRadius(4)
                }
            }

            // Error message
            if let error = connectionManager.lastError ?? wsManager.lastError {
                Divider()
                Text(error)
                    .font(.caption)
                    .foregroundColor(.red)
            }
        }
    }

    private func statusIndicator(isConnected: Bool) -> some View {
        Circle()
            .fill(isConnected ? Color.green : Color.red)
            .frame(width: 12, height: 12)
            .overlay(
                Circle()
                    .stroke(isConnected ? Color.green.opacity(0.3) : Color.red.opacity(0.3), lineWidth: 4)
            )
    }

    private func testConnectionButton(mode: ConnectionMode) -> some View {
        Button {
            testConnection(mode: mode)
        } label: {
            HStack {
                Text("Test Connection")
                    .font(.subheadline)

                Spacer()

                if isTesting {
                    ProgressView()
                        .scaleEffect(0.8)
                } else if let success = testResults[mode] {
                    Image(systemName: success ? "checkmark.circle.fill" : "xmark.circle.fill")
                        .foregroundColor(success ? .green : .red)
                }
            }
        }
        .disabled(isTesting || !settings.isConfigured(mode: mode))
    }

    // MARK: - Actions

    private func testConnection(mode: ConnectionMode) {
        isTesting = true
        testResults[mode] = nil

        Task {
            let success = await connectionManager.testConnection(mode: mode)
            await MainActor.run {
                testResults[mode] = success
                isTesting = false
            }
        }
    }

    private func reconnect() {
        Task {
            wsManager.disconnect()
            await connectionManager.connect()

            if connectionManager.connectionState.isConnected {
                wsManager.connect()
            }
        }
    }
}

#Preview {
    SettingsView()
        .environmentObject(SettingsStore())
        .environmentObject(ConnectionManager(settings: SettingsStore()))
        .environmentObject(WebSocketManager(connectionManager: ConnectionManager(settings: SettingsStore())))
}
