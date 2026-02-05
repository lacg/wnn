// APIClient - REST API calls to the backend

import Foundation

@MainActor
public final class APIClient {
    private let connectionManager: ConnectionManager
    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()

    public init(connectionManager: ConnectionManager) {
        self.connectionManager = connectionManager
    }

    private var baseURL: URL? { connectionManager.baseURL }

    private func request<T: Decodable>(path: String, method: String = "GET", body: (any Encodable)? = nil) async throws -> T {
        guard let base = baseURL else { throw APIError.notConnected }
        var request = URLRequest(url: base.appendingPathComponent(path))
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let body = body { request.httpBody = try encoder.encode(body) }

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        guard (200...299).contains(http.statusCode) else {
            if let err = try? decoder.decode(ErrorResponse.self, from: data) { throw APIError.serverError(http.statusCode, err.error) }
            throw APIError.httpError(http.statusCode)
        }
        return try decoder.decode(T.self, from: data)
    }

    private func requestVoid(path: String, method: String = "POST", body: (any Encodable)? = nil) async throws {
        guard let base = baseURL else { throw APIError.notConnected }
        var request = URLRequest(url: base.appendingPathComponent(path))
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let body = body { request.httpBody = try encoder.encode(body) }

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        guard (200...299).contains(http.statusCode) else {
            if let err = try? decoder.decode(ErrorResponse.self, from: data) { throw APIError.serverError(http.statusCode, err.error) }
            throw APIError.httpError(http.statusCode)
        }
    }

    // Dashboard
    public func getSnapshot() async throws -> DashboardSnapshot { try await request(path: "api/snapshot") }

    // Experiments
    public func getExperiments() async throws -> [Experiment] { try await request(path: "api/experiments") }
    public func getCurrentExperiment() async throws -> Experiment? {
        do { return try await request(path: "api/experiments/current") }
        catch APIError.httpError(404), APIError.serverError(404, _) { return nil }
    }
    public func getExperiment(_ id: Int64) async throws -> Experiment { try await request(path: "api/experiments/\(id)") }
    public func getIterations(experimentId: Int64) async throws -> [Iteration] { try await request(path: "api/experiments/\(experimentId)/iterations") }

    // Iterations
    public func getGenomes(iterationId: Int64) async throws -> [GenomeEvaluation] { try await request(path: "api/iterations/\(iterationId)/genomes") }

    // Flows
    public func getFlows(status: FlowStatus? = nil) async throws -> [Flow] {
        let path = status.map { "api/flows?status=\($0.rawValue)" } ?? "api/flows"
        return try await request(path: path)
    }
    public func getFlow(_ id: Int64) async throws -> Flow { try await request(path: "api/flows/\(id)") }
    public func createFlow(_ req: CreateFlowRequest) async throws -> Flow { try await request(path: "api/flows", method: "POST", body: req) }
    public func stopFlow(_ id: Int64) async throws { try await requestVoid(path: "api/flows/\(id)/stop") }
    public func restartFlow(_ id: Int64) async throws { try await requestVoid(path: "api/flows/\(id)/restart") }
    public func deleteFlow(_ id: Int64) async throws { try await requestVoid(path: "api/flows/\(id)", method: "DELETE") }
    public func getFlowExperiments(_ flowId: Int64) async throws -> [Experiment] { try await request(path: "api/flows/\(flowId)/experiments") }

    // Checkpoints
    public func getCheckpoints(experimentId: Int64? = nil, type: CheckpointType? = nil) async throws -> [Checkpoint] {
        var params: [String] = []
        if let e = experimentId { params.append("experiment_id=\(e)") }
        if let t = type { params.append("type=\(t.rawValue)") }
        let path = params.isEmpty ? "api/checkpoints" : "api/checkpoints?" + params.joined(separator: "&")
        return try await request(path: path)
    }
    public func deleteCheckpoint(_ id: Int64) async throws { try await requestVoid(path: "api/checkpoints/\(id)", method: "DELETE") }
    public func checkpointDownloadURL(_ id: Int64) -> URL? { baseURL?.appendingPathComponent("api/checkpoints/\(id)/download") }

    // Gating Runs
    public func getGatingRuns(experimentId: Int64) async throws -> [GatingRun] {
        try await request(path: "api/experiments/\(experimentId)/gating")
    }
    public func createGatingRun(experimentId: Int64) async throws -> GatingRun {
        try await request(path: "api/experiments/\(experimentId)/gating", method: "POST")
    }
    public func getGatingRun(experimentId: Int64, gatingId: Int64) async throws -> GatingRun {
        try await request(path: "api/experiments/\(experimentId)/gating/\(gatingId)")
    }
}

public enum APIError: LocalizedError {
    case notConnected, invalidResponse, httpError(Int), serverError(Int, String), decodingError(Error)

    public var errorDescription: String? {
        switch self {
        case .notConnected: return "Not connected to server"
        case .invalidResponse: return "Invalid response from server"
        case .httpError(let code): return "HTTP error: \(code)"
        case .serverError(_, let msg): return msg
        case .decodingError(let err): return "Decoding error: \(err.localizedDescription)"
        }
    }
}

private struct ErrorResponse: Decodable { let error: String }
