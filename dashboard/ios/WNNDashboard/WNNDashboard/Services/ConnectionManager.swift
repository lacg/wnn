// ConnectionManager - handles local/Tailscale connection switching

import Foundation
import Network

/// Manages connection state and automatic switching between local and Tailscale
@MainActor
final class ConnectionManager: ObservableObject {
    // MARK: - Published State

    @Published private(set) var connectionState: ConnectionState = .disconnected
    @Published private(set) var activeMode: ConnectionMode = .local
    @Published private(set) var lastError: String?

    // MARK: - Dependencies

    private let settings: SettingsStore
    private let networkMonitor = NWPathMonitor()
    private let monitorQueue = DispatchQueue(label: "NetworkMonitor")

    @Published private(set) var isOnWiFi = false
    @Published private(set) var isOnCellular = false

    // MARK: - URLs

    var baseURL: URL? {
        settings.baseURL(for: activeMode)
    }

    var wsURL: URL? {
        settings.wsURL(for: activeMode)
    }

    // MARK: - Initialization

    init(settings: SettingsStore) {
        self.settings = settings
        self.activeMode = settings.connectionMode == .auto ? .local : settings.connectionMode
        startNetworkMonitoring()
    }

    deinit {
        networkMonitor.cancel()
    }

    // MARK: - Network Monitoring

    private func startNetworkMonitoring() {
        networkMonitor.pathUpdateHandler = { [weak self] path in
            Task { @MainActor in
                self?.isOnWiFi = path.usesInterfaceType(.wifi)
                self?.isOnCellular = path.usesInterfaceType(.cellular)
            }
        }
        networkMonitor.start(queue: monitorQueue)
    }

    // MARK: - Connection Testing

    /// Test connection to a specific mode
    func testConnection(mode: ConnectionMode) async -> Bool {
        guard let url = settings.baseURL(for: mode)?.appendingPathComponent("api/snapshot") else {
            return false
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 5

        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            if let httpResponse = response as? HTTPURLResponse {
                return (200...299).contains(httpResponse.statusCode)
            }
            return false
        } catch {
            return false
        }
    }

    /// Attempt to establish connection with auto-detection
    func connect() async {
        connectionState = .connecting

        switch settings.connectionMode {
        case .local:
            await tryConnect(mode: .local)

        case .remote:
            await tryConnect(mode: .remote)

        case .auto:
            // Try local first, then remote
            if settings.isConfigured(mode: .local) && await testConnection(mode: .local) {
                activeMode = .local
                connectionState = .connected
                lastError = nil
            } else if settings.isConfigured(mode: .remote) && await testConnection(mode: .remote) {
                activeMode = .remote
                connectionState = .connected
                lastError = nil
            } else {
                connectionState = .disconnected
                lastError = "Could not connect to server"
            }
        }
    }

    private func tryConnect(mode: ConnectionMode) async {
        if await testConnection(mode: mode) {
            activeMode = mode
            connectionState = .connected
            lastError = nil
        } else {
            connectionState = .disconnected
            lastError = "Could not connect to \(mode.displayName)"
        }
    }

    /// Disconnect and reset state
    func disconnect() {
        connectionState = .disconnected
        lastError = nil
    }

    /// Switch to a specific mode
    func switchMode(to mode: ConnectionMode) async {
        activeMode = mode
        await connect()
    }
}

/// Connection state
enum ConnectionState: Equatable {
    case disconnected
    case connecting
    case connected
    case reconnecting

    var displayName: String {
        switch self {
        case .disconnected: return "Disconnected"
        case .connecting: return "Connecting..."
        case .connected: return "Connected"
        case .reconnecting: return "Reconnecting..."
        }
    }

    var isConnected: Bool {
        self == .connected
    }
}
