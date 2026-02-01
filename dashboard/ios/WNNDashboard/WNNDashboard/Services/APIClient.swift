// APIClient - REST API calls to the backend

import Foundation

/// HTTP client for REST API calls
final class APIClient {
    private let connectionManager: ConnectionManager
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder

    init(connectionManager: ConnectionManager) {
        self.connectionManager = connectionManager

        self.decoder = JSONDecoder()
        // Use default snake_case conversion
        self.decoder.keyDecodingStrategy = .useDefaultKeys

        self.encoder = JSONEncoder()
        self.encoder.keyEncodingStrategy = .useDefaultKeys
    }

    // MARK: - Base Request

    private var baseURL: URL? {
        connectionManager.baseURL
    }

    private func request<T: Decodable>(
        path: String,
        method: String = "GET",
        body: (any Encodable)? = nil
    ) async throws -> T {
        guard let base = baseURL else {
            throw APIError.notConnected
        }

        var request = URLRequest(url: base.appendingPathComponent(path))
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        if let body = body {
            request.httpBody = try encoder.encode(body)
        }

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            // Try to decode error message
            if let errorMessage = try? decoder.decode(ErrorResponse.self, from: data) {
                throw APIError.serverError(httpResponse.statusCode, errorMessage.error)
            }
            throw APIError.httpError(httpResponse.statusCode)
        }

        return try decoder.decode(T.self, from: data)
    }

    private func requestVoid(
        path: String,
        method: String = "POST",
        body: (any Encodable)? = nil
    ) async throws {
        guard let base = baseURL else {
            throw APIError.notConnected
        }

        var request = URLRequest(url: base.appendingPathComponent(path))
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        if let body = body {
            request.httpBody = try encoder.encode(body)
        }

        let (data, response) = try await URLSession.shared.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            if let errorMessage = try? decoder.decode(ErrorResponse.self, from: data) {
                throw APIError.serverError(httpResponse.statusCode, errorMessage.error)
            }
            throw APIError.httpError(httpResponse.statusCode)
        }
    }

    // MARK: - Dashboard

    /// Get current dashboard snapshot
    func getSnapshot() async throws -> DashboardSnapshot {
        try await request(path: "api/snapshot")
    }

    // MARK: - Experiments

    /// Get all experiments
    func getExperiments() async throws -> [Experiment] {
        try await request(path: "api/experiments")
    }

    /// Get current running experiment
    func getCurrentExperiment() async throws -> Experiment? {
        // The API returns 404 if no current experiment, so we handle that
        do {
            return try await request(path: "api/experiments/current")
        } catch APIError.httpError(404) {
            return nil
        } catch APIError.serverError(404, _) {
            return nil
        }
    }

    /// Get specific experiment by ID
    func getExperiment(_ id: Int64) async throws -> Experiment {
        try await request(path: "api/experiments/\(id)")
    }

    /// Get phases for an experiment
    func getPhases(experimentId: Int64) async throws -> [Phase] {
        try await request(path: "api/experiments/\(experimentId)/phases")
    }

    /// Get recent iterations for an experiment
    func getIterations(experimentId: Int64) async throws -> [Iteration] {
        try await request(path: "api/experiments/\(experimentId)/iterations")
    }

    // MARK: - Phases

    /// Get iterations for a specific phase
    func getPhaseIterations(phaseId: Int64) async throws -> [Iteration] {
        try await request(path: "api/phases/\(phaseId)/iterations")
    }

    // MARK: - Iterations

    /// Get genome evaluations for an iteration
    func getGenomes(iterationId: Int64) async throws -> [GenomeEvaluation] {
        try await request(path: "api/iterations/\(iterationId)/genomes")
    }

    // MARK: - Flows

    /// Get all flows
    func getFlows(status: FlowStatus? = nil) async throws -> [Flow] {
        var path = "api/flows"
        if let status = status {
            path += "?status=\(status.rawValue)"
        }
        return try await request(path: path)
    }

    /// Get specific flow by ID
    func getFlow(_ id: Int64) async throws -> Flow {
        try await request(path: "api/flows/\(id)")
    }

    /// Create a new flow
    func createFlow(_ request: CreateFlowRequest) async throws -> Flow {
        try await self.request(path: "api/flows", method: "POST", body: request)
    }

    /// Stop a running flow
    func stopFlow(_ id: Int64) async throws {
        try await requestVoid(path: "api/flows/\(id)/stop", method: "POST")
    }

    /// Restart a flow
    func restartFlow(_ id: Int64) async throws {
        try await requestVoid(path: "api/flows/\(id)/restart", method: "POST")
    }

    /// Delete a flow
    func deleteFlow(_ id: Int64) async throws {
        try await requestVoid(path: "api/flows/\(id)", method: "DELETE")
    }

    /// Get experiments in a flow
    func getFlowExperiments(_ flowId: Int64) async throws -> [Experiment] {
        try await request(path: "api/flows/\(flowId)/experiments")
    }

    // MARK: - Checkpoints

    /// Get all checkpoints
    func getCheckpoints(experimentId: Int64? = nil, type: CheckpointType? = nil) async throws -> [Checkpoint] {
        var params: [String] = []
        if let experimentId = experimentId {
            params.append("experiment_id=\(experimentId)")
        }
        if let type = type {
            params.append("type=\(type.rawValue)")
        }

        var path = "api/checkpoints"
        if !params.isEmpty {
            path += "?" + params.joined(separator: "&")
        }
        return try await request(path: path)
    }

    /// Get specific checkpoint
    func getCheckpoint(_ id: Int64) async throws -> Checkpoint {
        try await request(path: "api/checkpoints/\(id)")
    }

    /// Delete a checkpoint
    func deleteCheckpoint(_ id: Int64) async throws {
        try await requestVoid(path: "api/checkpoints/\(id)", method: "DELETE")
    }

    /// Get download URL for a checkpoint
    func checkpointDownloadURL(_ id: Int64) -> URL? {
        baseURL?.appendingPathComponent("api/checkpoints/\(id)/download")
    }
}

// MARK: - Error Types

enum APIError: LocalizedError {
    case notConnected
    case invalidResponse
    case httpError(Int)
    case serverError(Int, String)
    case decodingError(Error)

    var errorDescription: String? {
        switch self {
        case .notConnected:
            return "Not connected to server"
        case .invalidResponse:
            return "Invalid response from server"
        case .httpError(let code):
            return "HTTP error: \(code)"
        case .serverError(_, let message):
            return message
        case .decodingError(let error):
            return "Decoding error: \(error.localizedDescription)"
        }
    }
}

private struct ErrorResponse: Decodable {
    let error: String
}

// MARK: - AnyEncodable wrapper for generic encoding

private struct AnyEncodable: Encodable {
    private let _encode: (Encoder) throws -> Void

    init<T: Encodable>(_ wrapped: T) {
        _encode = wrapped.encode
    }

    func encode(to encoder: Encoder) throws {
        try _encode(encoder)
    }
}
