// HuggingFaceClient - Download WNN models from HuggingFace Hub

import Foundation

public actor HuggingFaceClient {
	private let session: URLSession
	private static let hubBaseURL = "https://huggingface.co"
	private static let apiBaseURL = "https://huggingface.co/api/models"

	public init() {
		let config = URLSessionConfiguration.default
		config.timeoutIntervalForRequest = 30
		self.session = URLSession(configuration: config)
	}

	/// Search for WNN models on HuggingFace Hub
	public func searchModels(query: String = "wnn-ram") async throws -> [HFModelInfo] {
		var components = URLComponents(string: Self.apiBaseURL)!
		components.queryItems = [
			.init(name: "search", value: query),
			.init(name: "library", value: "wnn"),
			.init(name: "limit", value: "50"),
		]

		let (data, response) = try await session.data(from: components.url!)
		guard let httpResponse = response as? HTTPURLResponse,
			  httpResponse.statusCode == 200 else {
			throw HFError.searchFailed
		}

		return try JSONDecoder().decode([HFModelInfo].self, from: data)
	}

	/// Download a model's config.json
	public func downloadConfig(repoId: String) async throws -> Data {
		let url = URL(string: "\(Self.hubBaseURL)/\(repoId)/resolve/main/config.json")!
		let (data, response) = try await session.data(from: url)
		guard let httpResponse = response as? HTTPURLResponse,
			  httpResponse.statusCode == 200 else {
			throw HFError.downloadFailed("config.json")
		}
		return data
	}

	/// Download model weights (safetensors or pytorch_model.bin)
	public func downloadWeights(repoId: String, progress: @Sendable @escaping (Double) -> Void) async throws -> URL {
		// Try safetensors first, then fall back to pytorch_model.bin
		let files = ["model.safetensors", "pytorch_model.bin"]

		for filename in files {
			let url = URL(string: "\(Self.hubBaseURL)/\(repoId)/resolve/main/\(filename)")!

			do {
				let (localURL, response) = try await session.download(from: url)
				guard let httpResponse = response as? HTTPURLResponse,
					  httpResponse.statusCode == 200 else { continue }

				// Move to a persistent location
				let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
				let modelDir = cacheDir.appendingPathComponent("WNNModels").appendingPathComponent(repoId.replacingOccurrences(of: "/", with: "_"))
				try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

				let destURL = modelDir.appendingPathComponent(filename)
				if FileManager.default.fileExists(atPath: destURL.path) {
					try FileManager.default.removeItem(at: destURL)
				}
				try FileManager.default.moveItem(at: localURL, to: destURL)

				progress(1.0)
				return destURL
			} catch {
				continue
			}
		}

		throw HFError.downloadFailed("model weights")
	}

	/// Download full model (config + weights) and parse into WNNModel
	public func downloadModel(
		repoId: String,
		progress: @Sendable @escaping (String, Double) -> Void
	) async throws -> WNNModel {
		progress("Downloading config...", 0.1)

		let configData = try await downloadConfig(repoId: repoId)
		let config = try JSONSerialization.jsonObject(with: configData) as? [String: Any] ?? [:]

		progress("Downloading weights...", 0.3)

		_ = try await downloadWeights(repoId: repoId, progress: { p in
			progress("Downloading weights...", 0.3 + p * 0.5)
		})

		progress("Loading model...", 0.9)

		// Build WNNModel from config
		var modelDict = config
		modelDict["name"] = repoId
		let model = try WNNModel.from(dict: modelDict)

		progress("Ready", 1.0)
		return model
	}
}

// MARK: - HuggingFace Hub Models

public struct HFModelInfo: Codable, Identifiable, Sendable {
	public var id: String { modelId }
	public let modelId: String
	public let author: String?
	public let lastModified: String?
	public let tags: [String]?
	public let downloads: Int?
	public let likes: Int?

	enum CodingKeys: String, CodingKey {
		case modelId
		case author
		case lastModified
		case tags
		case downloads
		case likes
	}
}

public enum HFError: Error, LocalizedError {
	case searchFailed
	case downloadFailed(String)
	case invalidModel

	public var errorDescription: String? {
		switch self {
		case .searchFailed: return "Failed to search HuggingFace Hub"
		case .downloadFailed(let file): return "Failed to download \(file)"
		case .invalidModel: return "Invalid WNN model format"
		}
	}
}
