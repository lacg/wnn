// LeaderboardViewModel - Manages leaderboard state and filtering

import Foundation

@MainActor
public final class LeaderboardViewModel: ObservableObject {
	@Published public var genomes: [BestGenome] = []
	@Published public var isLoading = false
	@Published public var error: String?

	// Filters
	@Published public var taskType: String = "ids" {
		didSet { Task { await refresh() } }
	}
	@Published public var stage: String = "stage_0" {
		didSet { Task { await refresh() } }
	}
	@Published public var metric: String = "ce" {
		didSet { Task { await refresh() } }
	}

	private let client: LeaderboardClient

	public init(client: LeaderboardClient = LeaderboardClient()) {
		self.client = client
	}

	public func refresh() async {
		isLoading = true
		error = nil
		do {
			genomes = try await client.fetchBestGenomes(
				taskType: taskType,
				stage: stage,
				metric: metric,
				limit: 50
			)
		} catch {
			self.error = error.localizedDescription
			genomes = []
		}
		isLoading = false
	}

	public func loadInitial() async {
		guard genomes.isEmpty else { return }
		await refresh()
	}

	// MARK: - Filter Options

	public static let taskTypes = [
		("ids", "IDS"),
		("lm", "Language Model"),
	]

	public static let stages = [
		("stage_0", "Stage 0"),
		("stage_1", "Stage 1"),
		("stage_2", "Stage 2"),
		("combined", "Combined"),
	]

	public static let metrics = [
		("ce", "Cross-Entropy"),
		("accuracy", "Accuracy"),
		("f1_macro", "F1 Macro"),
	]
}
