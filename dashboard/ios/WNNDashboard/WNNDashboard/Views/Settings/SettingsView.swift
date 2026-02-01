// SettingsView - Connection configuration

import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var settings: SettingsStore
    @EnvironmentObject var connectionManager: ConnectionManager
    @EnvironmentObject var wsManager: WebSocketManager

    @State private var isTesting = false
    @State private var testResult: TestResult?

    var body: some View {
        NavigationStack {
            Form {
                // Connection status
                Section("Connection Status") {
                    connectionStatusRow
                }

                // Connection mode
                Section("Connection Mode") {
                    Picker("Mode", selection: $settings.connectionMode) {
                        ForEach(ConnectionMode.allCases) { mode in
                            VStack(alignment: .leading) {
                                Text(mode.displayName)
                            }
                            .tag(mode)
                        }
                    }
                    .pickerStyle(.inline)
                }

                // Local network settings
                Section("Local Network") {
                    TextField("IP Address", text: $settings.localHost)
                        .keyboardType(.numbersAndPunctuation)
                        .autocapitalization(.none)

                    testButton(mode: .local)
                }

                // Tailscale settings
                Section("Tailscale") {
                    TextField("Hostname", text: $settings.tailscaleHost)
                        .keyboardType(.URL)
                        .autocapitalization(.none)

                    testButton(mode: .tailscale)
                }

                // Port settings
                Section("Server") {
                    HStack {
                        Text("Port")
                        Spacer()
                        TextField("Port", value: $settings.port, format: .number)
                            .keyboardType(.numberPad)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 80)
                    }

                    Toggle("Use HTTPS", isOn: $settings.useHTTPS)
                }

                // Actions
                Section {
                    Button("Reconnect") {
                        reconnect()
                    }

                    Button("Reset to Defaults", role: .destructive) {
                        settings.resetToDefaults()
                    }
                }

                // About
                Section("About") {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }

                    HStack {
                        Text("Active Mode")
                        Spacer()
                        Text(connectionManager.activeMode.displayName)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("Settings")
        }
    }

    // MARK: - Components

    private var connectionStatusRow: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Circle()
                        .fill(connectionManager.connectionState.isConnected ? Color.green : Color.red)
                        .frame(width: 10, height: 10)

                    Text("HTTP: \(connectionManager.connectionState.displayName)")
                        .font(.subheadline)
                }

                HStack(spacing: 8) {
                    Circle()
                        .fill(wsManager.connectionState.isConnected ? Color.green : Color.red)
                        .frame(width: 10, height: 10)

                    Text("WebSocket: \(wsManager.connectionState.displayName)")
                        .font(.subheadline)
                }
            }

            Spacer()

            if connectionManager.isOnWiFi {
                Image(systemName: "wifi")
                    .foregroundColor(.blue)
            } else if connectionManager.isOnCellular {
                Image(systemName: "antenna.radiowaves.left.and.right")
                    .foregroundColor(.orange)
            }
        }
    }

    private func testButton(mode: ConnectionMode) -> some View {
        Button {
            testConnection(mode: mode)
        } label: {
            HStack {
                Text("Test Connection")

                Spacer()

                if isTesting && testResult?.mode == mode {
                    ProgressView()
                        .scaleEffect(0.8)
                } else if let result = testResult, result.mode == mode {
                    Image(systemName: result.success ? "checkmark.circle.fill" : "xmark.circle.fill")
                        .foregroundColor(result.success ? .green : .red)
                }
            }
        }
        .disabled(isTesting || (mode == .tailscale && settings.tailscaleHost.isEmpty))
    }

    // MARK: - Actions

    private func testConnection(mode: ConnectionMode) {
        isTesting = true
        testResult = nil

        Task {
            let success = await connectionManager.testConnection(mode: mode)
            await MainActor.run {
                testResult = TestResult(mode: mode, success: success)
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

// MARK: - Test Result

private struct TestResult {
    let mode: ConnectionMode
    let success: Bool
}

#Preview {
    SettingsView()
        .environmentObject(SettingsStore())
        .environmentObject(ConnectionManager(settings: SettingsStore()))
        .environmentObject(WebSocketManager(connectionManager: ConnectionManager(settings: SettingsStore())))
}
