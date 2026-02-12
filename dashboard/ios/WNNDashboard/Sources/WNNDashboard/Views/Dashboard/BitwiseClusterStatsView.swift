// BitwiseClusterStatsView - Per-cluster stats table for bitwise WNN genomes

import SwiftUI

public struct BitwiseClusterStatsView: View {
	public let clusterStats: [BitwiseClusterStat]

	public init(clusterStats: [BitwiseClusterStat]) {
		self.clusterStats = clusterStats
	}

	public var body: some View {
		if !clusterStats.isEmpty {
			statsContent
		}
	}

	// MARK: - Content

	private var statsContent: some View {
		VStack(alignment: .leading, spacing: 12) {
			headerRow

			ScrollView(.horizontal, showsIndicators: false) {
				VStack(alignment: .leading, spacing: 0) {
					tableHeader
					Divider()
					tableRows
					Divider()
					totalsRow
				}
				.padding(8)
			}
			#if os(iOS)
			.background(Color(.secondarySystemBackground).opacity(0.5))
			#else
			.background(Color.secondary.opacity(0.1))
			#endif
			.cornerRadius(8)
		}
		.padding()
		.glassCard()
	}

	// MARK: - Header

	private var headerRow: some View {
		HStack {
			Text("Cluster Stats")
				.font(.headline)
			Spacer()
			Text("\(clusterStats.count) clusters")
				.font(.caption)
				.foregroundStyle(.secondary)
		}
	}

	// MARK: - Table Header

	private var tableHeader: some View {
		HStack(spacing: 0) {
			Text("Cluster")
				.frame(width: columnWidths.cluster, alignment: .leading)
			Text("Bits")
				.frame(width: columnWidths.bits, alignment: .trailing)
			Text("Neurons")
				.frame(width: columnWidths.neurons, alignment: .trailing)
			Text("Connections")
				.frame(width: columnWidths.connections, alignment: .trailing)
			Text("Memory")
				.frame(width: columnWidths.memory, alignment: .trailing)
		}
		.font(.caption2.bold())
		.foregroundStyle(.secondary)
		.padding(.vertical, 4)
	}

	// MARK: - Table Rows

	private var tableRows: some View {
		ForEach(clusterStats) { stat in
			HStack(spacing: 0) {
				Text("\(stat.cluster)")
					.frame(width: columnWidths.cluster, alignment: .leading)
				Text("\(stat.bits)")
					.frame(width: columnWidths.bits, alignment: .trailing)
				Text("\(stat.neurons)")
					.frame(width: columnWidths.neurons, alignment: .trailing)
				Text(NumberFormatters.formatCompact(stat.connections))
					.frame(width: columnWidths.connections, alignment: .trailing)
				Text(stat.formattedMemory)
					.frame(width: columnWidths.memory, alignment: .trailing)
					.foregroundStyle(memoryColor(for: stat))
			}
			.font(.caption)
			.fontDesign(.monospaced)
			.monospacedDigit()
			.padding(.vertical, 4)
		}
	}

	// MARK: - Totals Row

	private var totalsRow: some View {
		let totalBits = clusterStats.map(\.bits).reduce(0, +)
		let totalNeurons = clusterStats.map(\.neurons).reduce(0, +)
		let totalConnections = clusterStats.map(\.connections).reduce(0, +)
		let totalMemoryWords = clusterStats.map(\.memoryWords).reduce(0, +)
		let totalMemoryBytes = Int64(totalMemoryWords) * 8
		let totalMemoryMB = Double(totalMemoryBytes) / (1024 * 1024)

		let avgBits: String = clusterStats.isEmpty ? "-" : String(format: "%.1f", Double(totalBits) / Double(clusterStats.count))

		return HStack(spacing: 0) {
			Text("Total")
				.fontWeight(.semibold)
				.frame(width: columnWidths.cluster, alignment: .leading)
			Text(avgBits)
				.frame(width: columnWidths.bits, alignment: .trailing)
			Text(NumberFormatters.formatCompact(totalNeurons))
				.frame(width: columnWidths.neurons, alignment: .trailing)
			Text(NumberFormatters.formatCompact(totalConnections))
				.frame(width: columnWidths.connections, alignment: .trailing)
			Text(ByteCountFormatter.string(fromByteCount: totalMemoryBytes, countStyle: .binary))
				.frame(width: columnWidths.memory, alignment: .trailing)
				.foregroundStyle(memoryColorForMB(totalMemoryMB))
		}
		.font(.caption)
		.fontDesign(.monospaced)
		.monospacedDigit()
		.padding(.vertical, 6)
	}

	// MARK: - Memory Color Coding

	/// Green < 1MB, Yellow 1-10MB, Red > 10MB (8 bytes per word)
	private func memoryColor(for stat: BitwiseClusterStat) -> Color {
		memoryColorForMB(stat.memoryMB)
	}

	private func memoryColorForMB(_ mb: Double) -> Color {
		if mb < 1 { return .green }
		if mb <= 10 { return .yellow }
		return .red
	}

	// MARK: - Column Widths

	private var columnWidths: (cluster: CGFloat, bits: CGFloat, neurons: CGFloat, connections: CGFloat, memory: CGFloat) {
		(cluster: 60, bits: 50, neurons: 65, connections: 80, memory: 80)
	}
}
