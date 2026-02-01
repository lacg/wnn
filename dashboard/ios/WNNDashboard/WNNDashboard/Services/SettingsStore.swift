// SettingsStore - UserDefaults persistence for connection settings

import Foundation
import SwiftUI

/// Persists app settings to UserDefaults
final class SettingsStore: ObservableObject {
    private let defaults = UserDefaults.standard

    // MARK: - Keys
    private enum Keys {
        static let localHost = "localHost"
        static let tailscaleHost = "tailscaleHost"
        static let port = "port"
        static let connectionMode = "connectionMode"
        static let useHTTPS = "useHTTPS"
    }

    // MARK: - Published Properties

    /// Local network IP address (e.g., "192.168.1.100")
    @Published var localHost: String {
        didSet { defaults.set(localHost, forKey: Keys.localHost) }
    }

    /// Tailscale hostname (e.g., "mac-studio.tail12345.ts.net")
    @Published var tailscaleHost: String {
        didSet { defaults.set(tailscaleHost, forKey: Keys.tailscaleHost) }
    }

    /// Server port (default 3000)
    @Published var port: Int {
        didSet { defaults.set(port, forKey: Keys.port) }
    }

    /// Connection mode preference
    @Published var connectionMode: ConnectionMode {
        didSet { defaults.set(connectionMode.rawValue, forKey: Keys.connectionMode) }
    }

    /// Whether to use HTTPS (default false for local development)
    @Published var useHTTPS: Bool {
        didSet { defaults.set(useHTTPS, forKey: Keys.useHTTPS) }
    }

    // MARK: - Initialization

    init() {
        // Load from UserDefaults with sensible defaults
        self.localHost = defaults.string(forKey: Keys.localHost) ?? "192.168.1.100"
        self.tailscaleHost = defaults.string(forKey: Keys.tailscaleHost) ?? ""
        self.port = defaults.object(forKey: Keys.port) as? Int ?? 3000
        self.useHTTPS = defaults.bool(forKey: Keys.useHTTPS)

        // Load connection mode
        if let modeString = defaults.string(forKey: Keys.connectionMode),
           let mode = ConnectionMode(rawValue: modeString) {
            self.connectionMode = mode
        } else {
            self.connectionMode = .auto
        }
    }

    // MARK: - Computed URLs

    /// HTTP/HTTPS base URL for the current connection mode
    func baseURL(for mode: ConnectionMode) -> URL? {
        let host = hostForMode(mode)
        guard !host.isEmpty else { return nil }
        let scheme = useHTTPS ? "https" : "http"
        return URL(string: "\(scheme)://\(host):\(port)")
    }

    /// WebSocket URL for the current connection mode
    func wsURL(for mode: ConnectionMode) -> URL? {
        let host = hostForMode(mode)
        guard !host.isEmpty else { return nil }
        let scheme = useHTTPS ? "wss" : "ws"
        return URL(string: "\(scheme)://\(host):\(port)/ws")
    }

    /// Get host for a specific mode
    func hostForMode(_ mode: ConnectionMode) -> String {
        switch mode {
        case .local:
            return localHost
        case .tailscale:
            return tailscaleHost
        case .auto:
            // Auto mode uses local as primary
            return localHost
        }
    }

    /// Reset to defaults
    func resetToDefaults() {
        localHost = "192.168.1.100"
        tailscaleHost = ""
        port = 3000
        connectionMode = .auto
        useHTTPS = false
    }
}

/// Connection mode for switching between local and Tailscale
enum ConnectionMode: String, CaseIterable, Identifiable {
    case local
    case tailscale
    case auto

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .local: return "Local Network"
        case .tailscale: return "Tailscale"
        case .auto: return "Auto (Local first)"
        }
    }

    var description: String {
        switch self {
        case .local: return "Connect via local IP"
        case .tailscale: return "Connect via Tailscale VPN"
        case .auto: return "Try local first, fallback to Tailscale"
        }
    }
}
