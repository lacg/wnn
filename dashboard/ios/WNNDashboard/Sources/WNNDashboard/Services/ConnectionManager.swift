// ConnectionManager - handles local/remote connection switching

import Foundation
import Network

@MainActor
public final class ConnectionManager: ObservableObject {
    @Published public private(set) var connectionState: ConnectionState = .disconnected
    @Published public private(set) var activeMode: ConnectionMode = .local
    @Published public private(set) var lastError: String?
    @Published public private(set) var isOnWiFi = false
    @Published public private(set) var isOnCellular = false

    private let settings: SettingsStore
    private let networkMonitor = NWPathMonitor()
    private let monitorQueue = DispatchQueue(label: "NetworkMonitor")

    public var baseURL: URL? { settings.baseURL(for: activeMode) }
    public var wsURL: URL? { settings.wsURL(for: activeMode) }

    public init(settings: SettingsStore) {
        self.settings = settings
        self.activeMode = settings.connectionMode == .auto ? .local : settings.connectionMode
        networkMonitor.pathUpdateHandler = { [weak self] path in
            Task { @MainActor in
                self?.isOnWiFi = path.usesInterfaceType(.wifi)
                self?.isOnCellular = path.usesInterfaceType(.cellular)
            }
        }
        networkMonitor.start(queue: monitorQueue)
    }

    deinit { networkMonitor.cancel() }

    public func testConnection(mode: ConnectionMode) async -> Bool {
        guard let url = settings.baseURL(for: mode)?.appendingPathComponent("api/snapshot") else { return false }
        var request = URLRequest(url: url)
        request.timeoutInterval = 5
        do {
            let (_, response) = try await URLSession.shared.data(for: request)
            return (response as? HTTPURLResponse).map { (200...299).contains($0.statusCode) } ?? false
        } catch { return false }
    }

    public func connect() async {
        connectionState = .connecting
        switch settings.connectionMode {
        case .local: await tryConnect(mode: .local)
        case .remote: await tryConnect(mode: .remote)
        case .auto:
            if settings.isConfigured(mode: .local) && await testConnection(mode: .local) {
                activeMode = .local; connectionState = .connected; lastError = nil
            } else if settings.isConfigured(mode: .remote) && await testConnection(mode: .remote) {
                activeMode = .remote; connectionState = .connected; lastError = nil
            } else {
                connectionState = .disconnected; lastError = "Could not connect to server"
            }
        }
    }

    private func tryConnect(mode: ConnectionMode) async {
        if await testConnection(mode: mode) {
            activeMode = mode; connectionState = .connected; lastError = nil
        } else {
            connectionState = .disconnected; lastError = "Could not connect to \(mode.displayName)"
        }
    }

    public func disconnect() { connectionState = .disconnected; lastError = nil }
}

public enum ConnectionState: Equatable {
    case disconnected, connecting, connected, reconnecting

    public var displayName: String {
        switch self {
        case .disconnected: return "Disconnected"
        case .connecting: return "Connecting..."
        case .connected: return "Connected"
        case .reconnecting: return "Reconnecting..."
        }
    }

    public var isConnected: Bool { self == .connected }
}
