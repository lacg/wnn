// Theme - Liquid Glass styling for iOS 26 / macOS 26

import SwiftUI

public enum Theme {
	// MARK: - Status Colors

	public static func rankColor(_ rank: Int?) -> Color {
		switch rank {
		case 1: return .yellow
		case 2: return .gray
		case 3: return .orange
		default: return .secondary
		}
	}

	public static func taskColor(_ taskType: String) -> Color {
		switch taskType {
		case "lm": return .blue
		case "ids": return .green
		default: return .secondary
		}
	}

	// MARK: - Metric Colors

	public static let ceColor = Color.blue
	public static let accuracyColor = Color.green
	public static let f1Color = Color.purple
	public static let fprColor = Color.orange

	// MARK: - Verification Status Colors

	public static func verificationColor(_ status: VerificationStatus) -> Color {
		switch status {
		case .idle: return .secondary
		case .downloading: return .blue
		case .training: return .orange
		case .evaluating: return .purple
		case .completed: return .green
		case .failed: return .red
		}
	}
}

// MARK: - Liquid Glass Card Modifier

public struct LiquidGlassCard: ViewModifier {
	public func body(content: Content) -> some View {
		content
			.glassEffect(.regular, in: .rect(cornerRadius: 16))
	}
}

public extension View {
	func liquidGlassCard() -> some View {
		modifier(LiquidGlassCard())
	}
}

// MARK: - Rank Badge

public struct RankBadge: View {
	public let rank: Int?

	public init(rank: Int?) { self.rank = rank }

	public var body: some View {
		if let rank {
			Text("\(rank)")
				.font(.body.weight(.bold))
				.fontDesign(.monospaced)
				.frame(width: 32, height: 32)
				.glassEffect(.regular, in: .circle)
				.foregroundStyle(Theme.rankColor(rank))
		} else {
			Text("-")
				.font(.body)
				.frame(width: 32, height: 32)
				.foregroundStyle(.secondary)
		}
	}
}

// MARK: - Task Badge

public struct TaskBadge: View {
	public let taskType: String

	public init(taskType: String) { self.taskType = taskType }

	public var body: some View {
		Text(taskType.uppercased())
			.font(.body.weight(.semibold))
			.padding(.horizontal, 8)
			.padding(.vertical, 4)
			.foregroundStyle(Theme.taskColor(taskType))
			.glassEffect(.regular, in: .capsule)
	}
}

// MARK: - Metric Value

public struct MetricValue: View {
	public let label: String
	public let value: String
	public let color: Color

	public init(label: String, value: String, color: Color = .primary) {
		self.label = label
		self.value = value
		self.color = color
	}

	public var body: some View {
		VStack(alignment: .leading, spacing: 4) {
			Text(label)
				.font(.body)
				.foregroundStyle(.secondary)
			Text(value)
				.font(.title3.weight(.semibold))
				.fontDesign(.monospaced)
				.foregroundStyle(color)
		}
	}
}
