// WNNModel - Loaded WNN genome for on-device inference

import Foundation

/// A loaded WNN model ready for inference.
/// Contains the architecture (connections) and trained memory tables.
public struct WNNModel: Sendable {
	public let name: String
	public let contextSize: Int
	public let vocabSize: Int
	public let numClusters: Int
	public let bitsPerNeuron: [Int]
	public let neuronsPerCluster: [Int]
	public let connections: [Int64]         // flat: each neuron's bit indices
	public let memoryTables: [Float]        // flat: all neurons' memory values
	public let memoryOffsets: [Int]          // offset into memoryTables per neuron
	public let memorySizes: [Int]           // 2^bits per neuron

	public var totalNeurons: Int {
		neuronsPerCluster.reduce(0, +)
	}

	/// Parse a WNN model from downloaded genome data (JSON format).
	public static func from(json data: Data) throws -> WNNModel {
		let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
		return try from(dict: obj)
	}

	public static func from(dict obj: [String: Any]) throws -> WNNModel {
		let contextSize = obj["context_size"] as? Int ?? 4
		let vocabSize = obj["vocab_size"] as? Int ?? 50257
		let bpn = (obj["bits_per_neuron"] as? [Int]) ?? []
		let npc = (obj["neurons_per_cluster"] as? [Int]) ?? []
		let connArray = (obj["connections"] as? [Int]) ?? []

		let totalNeurons = npc.reduce(0, +)
		var memoryTables: [Float] = []
		var memoryOffsets: [Int] = []
		var memorySizes: [Int] = []

		// Pre-compute memory layout
		for i in 0..<totalNeurons {
			let bits = i < bpn.count ? bpn[i] : 16
			let size = 1 << bits
			memoryOffsets.append(memoryTables.count)
			memorySizes.append(size)
			// Initialize memory with 0.5 (empty/neutral)
			memoryTables.append(contentsOf: [Float](repeating: 0.5, count: size))
		}

		// Load memory values if present
		if let memValues = obj["memory_values"] as? [Double] {
			for (i, v) in memValues.enumerated() where i < memoryTables.count {
				memoryTables[i] = Float(v)
			}
		}

		return WNNModel(
			name: obj["name"] as? String ?? "WNN Model",
			contextSize: contextSize,
			vocabSize: vocabSize,
			numClusters: npc.count,
			bitsPerNeuron: bpn,
			neuronsPerCluster: npc,
			connections: connArray.map { Int64($0) },
			memoryTables: memoryTables,
			memoryOffsets: memoryOffsets,
			memorySizes: memorySizes
		)
	}
}
