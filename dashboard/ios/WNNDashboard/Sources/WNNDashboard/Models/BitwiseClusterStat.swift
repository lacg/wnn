// BitwiseClusterStat - per-cluster stats for bitwise WNN genomes

import Foundation

public struct BitwiseClusterStat: Codable, Hashable, Identifiable {
	public var id: Int { cluster }

	public let cluster: Int
	public let bits: Int
	public let neurons: Int
	public let connections: Int
	public let memoryWords: Int

	enum CodingKeys: String, CodingKey {
		case cluster, bits, neurons, connections
		case memoryWords = "memory_words"
	}

	/// Memory size in bytes (8 bytes per word)
	public var memoryBytes: Int64 { Int64(memoryWords) * 8 }

	/// Formatted memory size string
	public var formattedMemory: String {
		ByteCountFormatter.string(fromByteCount: memoryBytes, countStyle: .binary)
	}

	/// Memory size in megabytes (for color-coding thresholds)
	public var memoryMB: Double { Double(memoryBytes) / (1024 * 1024) }
}
