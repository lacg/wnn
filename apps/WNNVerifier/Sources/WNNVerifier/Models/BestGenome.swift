// BestGenome - Leaderboard entry model

import Foundation

public struct BestGenome: Codable, Identifiable, Sendable {
	public let id: Int64
	public let task_type: String
	public let stage: String
	public let metric: String
	public let genome_id: Int64
	public let genome_hash: String
	public let rank: Int?
	public let ce: Double
	public let accuracy: Double
	public let f1_macro: Double?
	public let fpr: Double?
	public let flow_id: Int64?
	public let experiment_id: Int64?
	public let hf_repo_id: String?
	public let hf_exported_at: String?
	public let created_at: String
	public let updated_at: String
	// Joined from genomes table
	public let tiers_json: String?
	public let total_clusters: Int?
	public let total_neurons: Int?
	public let architecture_type_str: String?
}

// MARK: - Display Helpers

extension BestGenome {
	public var stageLabel: String {
		switch stage {
		case "stage_0": return "S0"
		case "stage_1": return "S1"
		case "stage_2": return "S2"
		case "combined": return "Combined"
		default: return stage
		}
	}

	public var metricLabel: String {
		switch metric {
		case "ce": return "CE"
		case "accuracy": return "Acc"
		case "f1_macro": return "F1"
		default: return metric
		}
	}

	public var architectureLabel: String {
		architecture_type_str ?? "unknown"
	}
}
