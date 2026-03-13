// MetalInference - GPU-accelerated WNN evaluation via Metal

import Foundation
import Metal

/// Metal-accelerated WNN inference engine.
/// Performs per-neuron memory lookups on GPU for fast evaluation.
public final class MetalInference: @unchecked Sendable {
	private let device: MTLDevice
	private let commandQueue: MTLCommandQueue
	private let evaluatePipeline: MTLComputePipelineState

	/// Initialize Metal inference engine.
	/// Compiles the WNN evaluation shader and creates the compute pipeline.
	public init?() {
		guard let device = MTLCreateSystemDefaultDevice(),
			  let commandQueue = device.makeCommandQueue() else {
			return nil
		}
		self.device = device
		self.commandQueue = commandQueue

		// Load shader from bundle
		guard let library = try? device.makeDefaultLibrary(bundle: Bundle.module),
			  let function = library.makeFunction(name: "wnn_evaluate_neurons") else {
			// Fallback: try embedded source
			guard let library = device.makeDefaultLibrary(),
				  let function = library.makeFunction(name: "wnn_evaluate_neurons") else {
				return nil
			}
			do {
				self.evaluatePipeline = try device.makeComputePipelineState(function: function)
			} catch { return nil }
			return
		}

		do {
			self.evaluatePipeline = try device.makeComputePipelineState(function: function)
		} catch {
			return nil
		}
	}

	/// Evaluate all neurons for a single input, returning per-cluster probabilities.
	///
	/// - Parameters:
	///   - model: The WNN model with connections and memory tables
	///   - inputBits: Boolean input vector (context tokens encoded as bits)
	/// - Returns: Per-cluster probability [numClusters]
	public func evaluate(model: WNNModel, inputBits: [Bool]) -> [Float] {
		let totalNeurons = model.totalNeurons

		// Pack input bits into UInt32 array for GPU
		let inputWords = packBits(inputBits)

		// Create GPU buffers
		guard let inputBuffer = device.makeBuffer(
			bytes: inputWords,
			length: inputWords.count * MemoryLayout<UInt32>.stride,
			options: .storageModeShared
		) else { return cpuFallback(model: model, inputBits: inputBits) }

		guard let connectionsBuffer = device.makeBuffer(
			bytes: model.connections,
			length: model.connections.count * MemoryLayout<Int64>.stride,
			options: .storageModeShared
		) else { return cpuFallback(model: model, inputBits: inputBits) }

		guard let memoryBuffer = device.makeBuffer(
			bytes: model.memoryTables,
			length: model.memoryTables.count * MemoryLayout<Float>.stride,
			options: .storageModeShared
		) else { return cpuFallback(model: model, inputBits: inputBits) }

		guard let bitsBuffer = device.makeBuffer(
			bytes: model.bitsPerNeuron.map { UInt32($0) },
			length: totalNeurons * MemoryLayout<UInt32>.stride,
			options: .storageModeShared
		) else { return cpuFallback(model: model, inputBits: inputBits) }

		guard let offsetsBuffer = device.makeBuffer(
			bytes: model.memoryOffsets.map { UInt32($0) },
			length: totalNeurons * MemoryLayout<UInt32>.stride,
			options: .storageModeShared
		) else { return cpuFallback(model: model, inputBits: inputBits) }

		// Connection offsets (cumulative bits per neuron)
		var connOffsets = [UInt32](repeating: 0, count: totalNeurons)
		var offset: UInt32 = 0
		for i in 0..<totalNeurons {
			connOffsets[i] = offset
			offset += UInt32(model.bitsPerNeuron[i])
		}
		guard let connOffsetsBuffer = device.makeBuffer(
			bytes: connOffsets,
			length: totalNeurons * MemoryLayout<UInt32>.stride,
			options: .storageModeShared
		) else { return cpuFallback(model: model, inputBits: inputBits) }

		// Output buffer: one float per neuron
		guard let outputBuffer = device.makeBuffer(
			length: totalNeurons * MemoryLayout<Float>.stride,
			options: .storageModeShared
		) else { return cpuFallback(model: model, inputBits: inputBits) }

		// Dispatch compute
		guard let commandBuffer = commandQueue.makeCommandBuffer(),
			  let encoder = commandBuffer.makeComputeCommandEncoder() else {
			return cpuFallback(model: model, inputBits: inputBits)
		}

		encoder.setComputePipelineState(evaluatePipeline)
		encoder.setBuffer(inputBuffer, offset: 0, index: 0)
		encoder.setBuffer(connectionsBuffer, offset: 0, index: 1)
		encoder.setBuffer(memoryBuffer, offset: 0, index: 2)
		encoder.setBuffer(bitsBuffer, offset: 0, index: 3)
		encoder.setBuffer(offsetsBuffer, offset: 0, index: 4)
		encoder.setBuffer(connOffsetsBuffer, offset: 0, index: 5)
		encoder.setBuffer(outputBuffer, offset: 0, index: 6)

		let threadGroupSize = min(evaluatePipeline.maxTotalThreadsPerThreadgroup, totalNeurons)
		let gridSize = MTLSize(width: totalNeurons, height: 1, depth: 1)
		let threadGroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
		encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroup)
		encoder.endEncoding()

		commandBuffer.commit()
		commandBuffer.waitUntilCompleted()

		// Read back per-neuron outputs
		let outputPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: totalNeurons)
		let neuronOutputs = Array(UnsafeBufferPointer(start: outputPtr, count: totalNeurons))

		// Average per cluster
		return averagePerCluster(neuronOutputs: neuronOutputs, model: model)
	}

	// MARK: - CPU Fallback

	/// CPU fallback when Metal buffers can't be allocated
	private func cpuFallback(model: WNNModel, inputBits: [Bool]) -> [Float] {
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

		return averagePerCluster(neuronOutputs: neuronOutputs, model: model)
	}

	// MARK: - Helpers

	private func packBits(_ bits: [Bool]) -> [UInt32] {
		let numWords = (bits.count + 31) / 32
		var words = [UInt32](repeating: 0, count: numWords)
		for (i, bit) in bits.enumerated() where bit {
			words[i / 32] |= (1 << (i % 32))
		}
		return words
	}

	private func averagePerCluster(neuronOutputs: [Float], model: WNNModel) -> [Float] {
		var clusterProbs = [Float](repeating: 0, count: model.numClusters)
		var neuronIdx = 0
		for cluster in 0..<model.numClusters {
			let n = model.neuronsPerCluster[cluster]
			var sum: Float = 0
			for _ in 0..<n {
				sum += neuronOutputs[neuronIdx]
				neuronIdx += 1
			}
			clusterProbs[cluster] = n > 0 ? sum / Float(n) : 0.5
		}
		return clusterProbs
	}
}
