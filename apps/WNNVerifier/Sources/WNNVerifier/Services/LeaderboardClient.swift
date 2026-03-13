// LeaderboardClient - Fetches best genomes from dashboard API

import Foundation

public actor LeaderboardClient {
	private let baseURL: URL
	private let session: URLSession

	public init(baseURL: URL = URL(string: "https://localhost:3000")!) {
		self.baseURL = baseURL
		// Trust self-signed certs for local development
		let config = URLSessionConfiguration.default
		config.timeoutIntervalForRequest = 10
		self.session = URLSession(configuration: config)
	}

	/// Fetch best genomes with optional filters
	public func fetchBestGenomes(
		taskType: String? = nil,
		stage: String? = nil,
		metric: String? = nil,
		limit: Int = 50
	) async throws -> [BestGenome] {
		var components = URLComponents(url: baseURL.appendingPathComponent("/api/best-genomes"), resolvingAgainstBaseURL: false)!
		var queryItems = [URLQueryItem(name: "limit", value: "\(limit)")]
		if let taskType { queryItems.append(.init(name: "task_type", value: taskType)) }
		if let stage { queryItems.append(.init(name: "stage", value: stage)) }
		if let metric { queryItems.append(.init(name: "metric", value: metric)) }
		components.queryItems = queryItems

		let (data, response) = try await session.data(from: components.url!)
		guard let httpResponse = response as? HTTPURLResponse,
			  httpResponse.statusCode == 200 else {
			throw LeaderboardError.fetchFailed
		}

		return try JSONDecoder().decode([BestGenome].self, from: data)
	}

	/// Fetch full genome data for download/verification
	public func fetchGenomeData(id: Int64) async throws -> Data {
		let url = baseURL.appendingPathComponent("/api/best-genomes/\(id)/download")
		let (data, response) = try await session.data(from: url)
		guard let httpResponse = response as? HTTPURLResponse,
			  httpResponse.statusCode == 200 else {
			throw LeaderboardError.downloadFailed
		}
		return data
	}
}

public enum LeaderboardError: Error, LocalizedError {
	case fetchFailed
	case downloadFailed

	public var errorDescription: String? {
		switch self {
		case .fetchFailed: return "Failed to fetch leaderboard"
		case .downloadFailed: return "Failed to download genome data"
		}
	}
}
