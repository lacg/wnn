// VerificationResult - Evaluation results and status

import Foundation

/// Status of a verification run
public enum VerificationStatus: Sendable {
	case idle
	case downloading
	case training
	case evaluating
	case completed
	case failed
}

/// Result of a verification evaluation
public struct VerificationResult: Sendable {
	public let crossEntropy: Double
	public let perplexity: Double
	public let accuracy: Double
	public let f1Macro: Double?
	public let fpr: Double?
	public let numPredictions: Int
	public let elapsedSeconds: Double
	public let dataset: String
	public let split: String

	public init(
		crossEntropy: Double,
		perplexity: Double,
		accuracy: Double,
		f1Macro: Double? = nil,
		fpr: Double? = nil,
		numPredictions: Int,
		elapsedSeconds: Double,
		dataset: String = "WikiText-2",
		split: String = "test"
	) {
		self.crossEntropy = crossEntropy
		self.perplexity = perplexity
		self.accuracy = accuracy
		self.f1Macro = f1Macro
		self.fpr = fpr
		self.numPredictions = numPredictions
		self.elapsedSeconds = elapsedSeconds
		self.dataset = dataset
		self.split = split
	}
}
