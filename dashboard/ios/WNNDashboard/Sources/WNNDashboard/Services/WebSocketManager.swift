// WebSocketManager - real-time updates via WebSocket

import Foundation

@MainActor
public final class WebSocketManager: ObservableObject {
    @Published public private(set) var connectionState: WebSocketState = .disconnected
    @Published public private(set) var lastMessage: WsMessage?
    @Published public private(set) var lastError: String?
    @Published public var snapshot: DashboardSnapshot?

    private let connectionManager: ConnectionManager
    private var webSocket: URLSessionWebSocketTask?
    private var pingTimer: Timer?
    private let decoder = JSONDecoder()

    private var reconnectAttempts = 0
    private let maxReconnectAttempts = 5
    private let baseReconnectDelay: TimeInterval = 1.0
    private let maxReconnectDelay: TimeInterval = 30.0
    private let jitterFactor: Double = 0.3

    private var messageHandlers: [String: (WsMessage) -> Void] = [:]

    public func addMessageHandler(id: String, handler: @escaping (WsMessage) -> Void) {
        messageHandlers[id] = handler
    }

    public init(connectionManager: ConnectionManager) {
        self.connectionManager = connectionManager
    }

    public func connect() {
        guard connectionManager.connectionState.isConnected, let wsURL = connectionManager.wsURL else {
            connectionState = .disconnected
            lastError = connectionManager.connectionState.isConnected ? "Invalid WebSocket URL" : "HTTP not connected"
            return
        }
        disconnect()
        connectionState = .connecting
        webSocket = TrustingURLSession.shared.webSocketTask(with: wsURL)
        webSocket?.resume()
        connectionState = .connected
        reconnectAttempts = 0
        lastError = nil
        startReceiving()
        startPingTimer()
    }

    public func disconnect() {
        stopPingTimer()
        webSocket?.cancel(with: .normalClosure, reason: nil)
        webSocket = nil
        connectionState = .disconnected
    }

    private func startReceiving() {
        webSocket?.receive { [weak self] result in
            Task { @MainActor in self?.handleReceiveResult(result) }
        }
    }

    private func handleReceiveResult(_ result: Result<URLSessionWebSocketTask.Message, Error>) {
        switch result {
        case .success(let msg):
            if case .string(let text) = msg { parseMessage(text) }
            else if case .data(let data) = msg, let text = String(data: data, encoding: .utf8) { parseMessage(text) }
            startReceiving()
        case .failure(let error):
            lastError = error.localizedDescription
            connectionState = .disconnected
            if reconnectAttempts < maxReconnectAttempts { attemptReconnect() }
        }
    }

    private func parseMessage(_ text: String) {
        guard let data = text.data(using: .utf8), let msg = try? decoder.decode(WsMessage.self, from: data) else { return }
        lastMessage = msg
        processMessage(msg)
        for handler in messageHandlers.values { handler(msg) }
    }

    private func processMessage(_ msg: WsMessage) {
        switch msg {
        case .snapshot(let s): snapshot = s
        case .iterationCompleted(let iter):
            if let s = snapshot {
                snapshot = DashboardSnapshot(current_experiment: s.current_experiment,
                    iterations: [iter] + Array(s.iterations.prefix(99)),
                    best_ce: min(s.best_ce, iter.best_ce),
                    best_accuracy: max(s.best_accuracy, iter.best_accuracy ?? 0))
            }
        case .experimentStatusChanged(let e):
            if let s = snapshot, s.current_experiment?.id == e.id {
                snapshot = DashboardSnapshot(current_experiment: e,
                    iterations: s.iterations, best_ce: s.best_ce, best_accuracy: s.best_accuracy)
            }
        default: break
        }
    }

    private func attemptReconnect() {
        reconnectAttempts += 1
        connectionState = .reconnecting
        let delay = min(baseReconnectDelay * pow(2, Double(reconnectAttempts - 1)), maxReconnectDelay) * (1.0 + Double.random(in: -jitterFactor...jitterFactor))
        Task {
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            await MainActor.run { if self.connectionState == .reconnecting { self.connect() } }
        }
    }

    private func startPingTimer() {
        pingTimer = Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.webSocket?.sendPing { [weak self] error in
                    if let error = error {
                        Task { @MainActor [weak self] in self?.lastError = error.localizedDescription }
                    }
                }
            }
        }
    }

    private func stopPingTimer() { pingTimer?.invalidate(); pingTimer = nil }
}

public enum WebSocketState: Equatable {
    case disconnected, connecting, connected, reconnecting
    public var isConnected: Bool { self == .connected }
    public var displayName: String {
        switch self {
        case .disconnected: return "Disconnected"
        case .connecting: return "Connecting..."
        case .connected: return "Connected"
        case .reconnecting: return "Reconnecting..."
        }
    }
}
