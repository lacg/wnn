// SettingsStore - UserDefaults persistence for connection settings

import Foundation
import SwiftUI

@MainActor
public final class SettingsStore: ObservableObject {
    private let defaults = UserDefaults.standard

    private enum Keys {
        static let localServerURL = "localServerURL"
        static let remoteServerURL = "remoteServerURL"
        static let connectionMode = "connectionMode"
    }

    @Published public var localServerURL: String {
        didSet { defaults.set(localServerURL, forKey: Keys.localServerURL) }
    }

    @Published public var remoteServerURL: String {
        didSet { defaults.set(remoteServerURL, forKey: Keys.remoteServerURL) }
    }

    @Published public var connectionMode: ConnectionMode {
        didSet { defaults.set(connectionMode.rawValue, forKey: Keys.connectionMode) }
    }

    public init() {
        self.localServerURL = defaults.string(forKey: Keys.localServerURL) ?? "http://192.168.1.100:3000"
        self.remoteServerURL = defaults.string(forKey: Keys.remoteServerURL) ?? ""
        if let modeString = defaults.string(forKey: Keys.connectionMode),
           let mode = ConnectionMode(rawValue: modeString) {
            self.connectionMode = mode
        } else {
            self.connectionMode = .auto
        }
    }

    public func baseURL(for mode: ConnectionMode) -> URL? {
        var urlString = (mode == .remote ? remoteServerURL : localServerURL).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !urlString.isEmpty else { return nil }
        if !urlString.hasPrefix("http://") && !urlString.hasPrefix("https://") { urlString = "http://" + urlString }
        if urlString.hasSuffix("/") { urlString = String(urlString.dropLast()) }
        return URL(string: urlString)
    }

    public func wsURL(for mode: ConnectionMode) -> URL? {
        guard let base = baseURL(for: mode), var components = URLComponents(url: base, resolvingAgainstBaseURL: false) else { return nil }
        components.scheme = components.scheme == "https" ? "wss" : "ws"
        components.path = "/ws"
        return components.url
    }

    public func isConfigured(mode: ConnectionMode) -> Bool {
        mode == .remote ? !remoteServerURL.isEmpty : !localServerURL.isEmpty
    }

    public func resetToDefaults() {
        localServerURL = "http://192.168.1.100:3000"
        remoteServerURL = ""
        connectionMode = .auto
    }
}

public enum ConnectionMode: String, CaseIterable, Identifiable {
    case local, remote, auto

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .local: return "Local Server"
        case .remote: return "Remote Server"
        case .auto: return "Auto (Local first)"
        }
    }

    public var description: String {
        switch self {
        case .local: return "Connect via local network"
        case .remote: return "Connect via internet"
        case .auto: return "Try local first, fallback to remote"
        }
    }

    public var icon: String {
        switch self {
        case .local: return "wifi"
        case .remote: return "globe"
        case .auto: return "arrow.triangle.2.circlepath"
        }
    }
}
