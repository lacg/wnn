// SettingsStore - UserDefaults persistence for connection settings

import Foundation
import SwiftUI

/// Persists app settings to UserDefaults
final class SettingsStore: ObservableObject {
    private let defaults = UserDefaults.standard

    // MARK: - Keys
    private enum Keys {
        static let localServerURL = "localServerURL"
        static let remoteServerURL = "remoteServerURL"
        static let connectionMode = "connectionMode"
    }

    // MARK: - Published Properties

    /// Local server URL (e.g., "http://192.168.1.100:3000")
    /// Used when on the same network as the server
    @Published var localServerURL: String {
        didSet {
            defaults.set(localServerURL, forKey: Keys.localServerURL)
        }
    }

    /// Remote server URL (e.g., "http://mac-studio.tail12345.ts.net:3000")
    /// Used when away from local network (Tailscale, public URL, etc.)
    @Published var remoteServerURL: String {
        didSet {
            defaults.set(remoteServerURL, forKey: Keys.remoteServerURL)
        }
    }

    /// Connection mode preference
    @Published var connectionMode: ConnectionMode {
        didSet {
            defaults.set(connectionMode.rawValue, forKey: Keys.connectionMode)
        }
    }

    // MARK: - Initialization

    init() {
        // Load from UserDefaults with sensible defaults
        self.localServerURL = defaults.string(forKey: Keys.localServerURL) ?? "http://192.168.1.100:3000"
        self.remoteServerURL = defaults.string(forKey: Keys.remoteServerURL) ?? ""

        // Load connection mode
        if let modeString = defaults.string(forKey: Keys.connectionMode),
           let mode = ConnectionMode(rawValue: modeString) {
            self.connectionMode = mode
        } else {
            self.connectionMode = .auto
        }
    }

    // MARK: - Computed URLs

    /// HTTP/HTTPS base URL for the specified connection mode
    func baseURL(for mode: ConnectionMode) -> URL? {
        let urlString: String
        switch mode {
        case .local:
            urlString = localServerURL
        case .remote:
            urlString = remoteServerURL
        case .auto:
            urlString = localServerURL  // Auto tries local first
        }

        guard !urlString.isEmpty else { return nil }

        // Ensure URL has scheme
        var normalized = urlString.trimmingCharacters(in: .whitespacesAndNewlines)
        if !normalized.hasPrefix("http://") && !normalized.hasPrefix("https://") {
            normalized = "http://" + normalized
        }

        // Remove trailing slash
        if normalized.hasSuffix("/") {
            normalized = String(normalized.dropLast())
        }

        return URL(string: normalized)
    }

    /// WebSocket URL for the specified connection mode
    func wsURL(for mode: ConnectionMode) -> URL? {
        guard let base = baseURL(for: mode),
              var components = URLComponents(url: base, resolvingAgainstBaseURL: false) else {
            return nil
        }

        // Convert http(s) to ws(s)
        if components.scheme == "https" {
            components.scheme = "wss"
        } else {
            components.scheme = "ws"
        }

        components.path = "/ws"
        return components.url
    }

    /// Get the URL string for display
    func urlString(for mode: ConnectionMode) -> String {
        switch mode {
        case .local: return localServerURL
        case .remote: return remoteServerURL
        case .auto: return localServerURL
        }
    }

    /// Check if a mode has a valid URL configured
    func isConfigured(mode: ConnectionMode) -> Bool {
        switch mode {
        case .local: return !localServerURL.isEmpty
        case .remote: return !remoteServerURL.isEmpty
        case .auto: return !localServerURL.isEmpty
        }
    }

    /// Reset to defaults
    func resetToDefaults() {
        localServerURL = "http://192.168.1.100:3000"
        remoteServerURL = ""
        connectionMode = .auto
    }
}

/// Connection mode for switching between local and remote servers
enum ConnectionMode: String, CaseIterable, Identifiable {
    case local
    case remote
    case auto

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .local: return "Local Server"
        case .remote: return "Remote Server"
        case .auto: return "Auto (Local first)"
        }
    }

    var description: String {
        switch self {
        case .local: return "Connect via local network"
        case .remote: return "Connect via internet (Tailscale, etc.)"
        case .auto: return "Try local first, fallback to remote"
        }
    }

    var icon: String {
        switch self {
        case .local: return "wifi"
        case .remote: return "globe"
        case .auto: return "arrow.triangle.2.circlepath"
        }
    }
}
