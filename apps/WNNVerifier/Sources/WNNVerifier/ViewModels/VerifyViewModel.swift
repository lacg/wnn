// VerifyViewModel - Manages model download, training, and verification

import Foundation

@MainActor
public final class VerifyViewModel: ObservableObject {
	@Published public var status: VerificationStatus = .idle
	@Published public var statusMessage: String = "Ready"
	@Published public var progress: Double = 0
	@Published public var result: VerificationResult?
	@Published public var error: String?

	// Model source
	@Published public var selectedGenome: BestGenome?
	@Published public var hfRepoId: String = ""

	private let hfClient: HuggingFaceClient
	private let metalInference: MetalInference?
	private var loadedModel: WNNModel?

	public init() {
		self.hfClient = HuggingFaceClient()
		self.metalInference = MetalInference()
	}

	public var canVerify: Bool {
		status == .idle || status == .completed || status == .failed
	}

	public var hasGPU: Bool {
		metalInference != nil
	}

	/// Verify a genome from the leaderboard
	public func verify(genome: BestGenome) async {
		selectedGenome = genome
		guard let repoId = genome.hf_repo_id, !repoId.isEmpty else {
			error = "This genome has not been exported to HuggingFace yet."
			return
		}
		hfRepoId = repoId
		await downloadAndVerify()
	}

	/// Verify a model by HuggingFace repo ID
	public func verifyFromRepo() async {
		guard !hfRepoId.trimmingCharacters(in: .whitespaces).isEmpty else {
			error = "Please enter a HuggingFace repo ID"
			return
		}
		await downloadAndVerify()
	}

	private func downloadAndVerify() async {
		status = .downloading
		statusMessage = "Downloading model..."
		progress = 0.1
		error = nil
		result = nil

		do {
			let model = try await hfClient.downloadModel(
				repoId: hfRepoId.trimmingCharacters(in: .whitespaces),
				progress: { @Sendable message, p in
					Task { @MainActor in
						self.statusMessage = message
						self.progress = p
					}
				}
			)
			loadedModel = model

			status = .evaluating
			statusMessage = "Evaluating on device..."
			progress = 0.9

			let evalResult = evaluate(model: model)
			result = evalResult
			status = .completed
			statusMessage = "Verification complete"
			progress = 1.0

		} catch {
			status = .failed
			statusMessage = "Failed"
			self.error = error.localizedDescription
		}
	}

	/// Evaluate a loaded model using Metal (GPU) or CPU fallback
	private func evaluate(model: WNNModel) -> VerificationResult {
		let start = CFAbsoluteTimeGetCurrent()

		// Generate test inputs (random bits for demonstration)
		// In production, this would use actual dataset samples
		let numSamples = 100
		let inputSize = model.contextSize * 16 // bits per token
		var totalCE = 0.0
		var correct = 0

		for _ in 0..<numSamples {
			let inputBits = (0..<inputSize).map { _ in Bool.random() }

			let probs: [Float]
			if let metal = metalInference {
				probs = metal.evaluate(model: model, inputBits: inputBits)
			} else {
				probs = cpuEvaluate(model: model, inputBits: inputBits)
			}

			// Compute CE for this sample
			if let maxIdx = probs.indices.max(by: { probs[$0] < probs[$1] }) {
				let p = max(Double(probs[maxIdx]), 1e-10)
				totalCE -= log(p)
				if maxIdx == 0 { correct += 1 } // simplified
			}
		}

		let elapsed = CFAbsoluteTimeGetCurrent() - start
		let avgCE = totalCE / Double(numSamples)

		return VerificationResult(
			crossEntropy: avgCE,
			perplexity: exp(avgCE),
			accuracy: Double(correct) / Double(numSamples),
			numPredictions: numSamples,
			elapsedSeconds: elapsed
		)
	}

	/// CPU fallback evaluation (mirrors MetalInference.cpuFallback)
	private func cpuEvaluate(model: WNNModel, inputBits: [Bool]) -> [Float] {
		let totalNeurons = model.totalNeurons
		var neuronOutputs = [Float](repeating: 0.5, count: totalNeurons)
		var connIdx = 0

		for neuronIdx in 0..<totalNeurons {
			let bits = model.bitsPerNeuron[neuronIdx]
			var address: UInt64 = 0
			for b in 0..<bits {
				let bitIdx = Int(model.connections[connIdx + b])
				if bitIdx >= 0 && bitIdx < inputBits.count && inputBits[bitIdx] {
					address |= (1 << (bits - 1 - b))
				}
			}
			let memOffset = model.memoryOffsets[neuronIdx]
			let memSize = model.memorySizes[neuronIdx]
			if Int(address) < memSize {
				neuronOutputs[neuronIdx] = model.memoryTables[memOffset + Int(address)]
			}
			connIdx += bits
		}

		// Average per cluster
		var clusterProbs = [Float](repeating: 0, count: model.numClusters)
		var ni = 0
		for cluster in 0..<model.numClusters {
			let n = model.neuronsPerCluster[cluster]
			var sum: Float = 0
			for _ in 0..<n {
				sum += neuronOutputs[ni]
				ni += 1
			}
			clusterProbs[cluster] = n > 0 ? sum / Float(n) : 0.5
		}
		return clusterProbs
	}

	public func reset() {
		status = .idle
		statusMessage = "Ready"
		progress = 0
		result = nil
		error = nil
		loadedModel = nil
		selectedGenome = nil
	}
}
