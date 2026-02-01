// WebSocketManager - real-time updates via WebSocket

import Foundation

/// Manages WebSocket connection for real-time updates
@MainActor
final class WebSocketManager: ObservableObject {
    // MARK: - Published State

    @Published private(set) var connectionState: WebSocketState = .disconnected
    @Published private(set) var lastMessage: WsMessage?
    @Published private(set) var lastError: String?

    // Snapshot data updated from messages
    @Published private(set) var snapshot: DashboardSnapshot?

    // MARK: - Dependencies

    private let connectionManager: ConnectionManager
    private var webSocket: URLSessionWebSocketTask?
    private var pingTimer: Timer?
    private let decoder = JSONDecoder()

    // Reconnection settings with exponential backoff + jitter
    private var reconnectAttempts = 0
    private let maxReconnectAttempts = 5
    private let baseReconnectDelay: TimeInterval = 1.0
    private let maxReconnectDelay: TimeInterval = 30.0
    private let jitterFactor: Double = 0.3  // +/- 30% randomization

    // MARK: - Callbacks

    /// Called when a message is received
    var onMessage: ((WsMessage) -> Void)?

    // MARK: - Initialization

    init(connectionManager: ConnectionManager) {
        self.connectionManager = connectionManager
    }

    // MARK: - Connection Management

    /// Connect to WebSocket
    func connect() {
        guard connectionManager.connectionState.isConnected else {
            connectionState = .disconnected
            lastError = "HTTP connection not established"
            return
        }

        guard let wsURL = connectionManager.wsURL else {
            connectionState = .disconnected
            lastError = "Invalid WebSocket URL"
            return
        }

        disconnect()
        connectionState = .connecting

        let session = URLSession(configuration: .default)
        webSocket = session.webSocketTask(with: wsURL)
        webSocket?.resume()

        connectionState = .connected
        reconnectAttempts = 0
        lastError = nil

        startReceiving()
        startPingTimer()
    }

    /// Disconnect from WebSocket
    func disconnect() {
        stopPingTimer()
        webSocket?.cancel(with: .normalClosure, reason: nil)
        webSocket = nil
        connectionState = .disconnected
    }

    // MARK: - Message Receiving

    private func startReceiving() {
        webSocket?.receive { [weak self] result in
            Task { @MainActor in
                self?.handleReceiveResult(result)
            }
        }
    }

    private func handleReceiveResult(_ result: Result<URLSessionWebSocketTask.Message, Error>) {
        switch result {
        case .success(let message):
            handleMessage(message)
            // Continue receiving
            startReceiving()

        case .failure(let error):
            handleError(error)
        }
    }

    private func handleMessage(_ message: URLSessionWebSocketTask.Message) {
        switch message {
        case .string(let text):
            parseMessage(text)

        case .data(let data):
            if let text = String(data: data, encoding: .utf8) {
                parseMessage(text)
            }

        @unknown default:
            break
        }
    }

    private func parseMessage(_ text: String) {
        guard let data = text.data(using: .utf8) else { return }

        do {
            let message = try decoder.decode(WsMessage.self, from: data)
            lastMessage = message
            processMessage(message)
            onMessage?(message)
        } catch {
            print("WebSocket parse error: \(error)")
            // Don't set lastError for parse errors, just log them
        }
    }

    // MARK: - Message Processing

    private func processMessage(_ message: WsMessage) {
        switch message {
        case .snapshot(let newSnapshot):
            snapshot = newSnapshot

        case .iterationCompleted(let iteration):
            // Prepend new iteration to the list
            if var current = snapshot {
                current = DashboardSnapshot(
                    current_experiment: current.current_experiment,
                    current_phase: current.current_phase,
                    phases: current.phases,
                    iterations: [iteration] + current.iterations.prefix(99),
                    best_ce: min(current.best_ce, iteration.best_ce),
                    best_accuracy: max(current.best_accuracy, iteration.best_accuracy ?? 0)
                )
                snapshot = current
            }

        case .phaseStarted(let phase):
            if var current = snapshot {
                var phases = current.phases
                if let index = phases.firstIndex(where: { $0.id == phase.id }) {
                    phases[index] = phase
                } else {
                    phases.append(phase)
                }
                snapshot = DashboardSnapshot(
                    current_experiment: current.current_experiment,
                    current_phase: phase,
                    phases: phases,
                    iterations: current.iterations,
                    best_ce: current.best_ce,
                    best_accuracy: current.best_accuracy
                )
            }

        case .phaseCompleted(let phase):
            if var current = snapshot {
                var phases = current.phases
                if let index = phases.firstIndex(where: { $0.id == phase.id }) {
                    phases[index] = phase
                }
                snapshot = DashboardSnapshot(
                    current_experiment: current.current_experiment,
                    current_phase: current.current_phase?.id == phase.id ? nil : current.current_phase,
                    phases: phases,
                    iterations: current.iterations,
                    best_ce: current.best_ce,
                    best_accuracy: current.best_accuracy
                )
            }

        case .experimentStatusChanged(let experiment):
            if var current = snapshot {
                if current.current_experiment?.id == experiment.id {
                    snapshot = DashboardSnapshot(
                        current_experiment: experiment,
                        current_phase: current.current_phase,
                        phases: current.phases,
                        iterations: current.iterations,
                        best_ce: current.best_ce,
                        best_accuracy: current.best_accuracy
                    )
                }
            }

        case .genomeEvaluations, .healthCheck, .flowStarted, .flowCompleted,
             .flowFailed, .flowCancelled, .flowQueued, .checkpointCreated, .checkpointDeleted:
            // These don't update the snapshot directly
            // ViewModels can listen to onMessage for specific handling
            break
        }
    }

    // MARK: - Error Handling

    private func handleError(_ error: Error) {
        lastError = error.localizedDescription
        connectionState = .disconnected

        // Attempt reconnection
        if reconnectAttempts < maxReconnectAttempts {
            attemptReconnect()
        }
    }

    private func attemptReconnect() {
        reconnectAttempts += 1
        connectionState = .reconnecting

        // Exponential backoff with jitter
        let exponentialDelay = baseReconnectDelay * pow(2, Double(reconnectAttempts - 1))
        let cappedDelay = min(exponentialDelay, maxReconnectDelay)

        // Add jitter: random value in range [1-jitter, 1+jitter]
        let jitter = 1.0 + Double.random(in: -jitterFactor...jitterFactor)
        let finalDelay = cappedDelay * jitter

        Task {
            try? await Task.sleep(nanoseconds: UInt64(finalDelay * 1_000_000_000))
            await MainActor.run {
                if self.connectionState == .reconnecting {
                    self.connect()
                }
            }
        }
    }

    // MARK: - Ping/Pong

    private func startPingTimer() {
        pingTimer = Timer.scheduledTimer(withTimeInterval: 30, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.sendPing()
            }
        }
    }

    private func stopPingTimer() {
        pingTimer?.invalidate()
        pingTimer = nil
    }

    private func sendPing() {
        webSocket?.sendPing { [weak self] error in
            if let error = error {
                Task { @MainActor in
                    self?.handleError(error)
                }
            }
        }
    }
}

/// WebSocket connection state
enum WebSocketState: Equatable {
    case disconnected
    case connecting
    case connected
    case reconnecting

    var isConnected: Bool {
        self == .connected
    }
}
