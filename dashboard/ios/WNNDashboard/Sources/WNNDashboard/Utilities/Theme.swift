// Theme - colors and styling

import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

public enum Theme {
    public static func statusColor(_ status: FlowStatus) -> Color {
        switch status {
        case .pending, .cancelled: return .gray
        case .queued: return .orange
        case .running: return .blue
        case .paused: return .yellow
        case .completed: return .green
        case .failed: return .red
        }
    }

    public static func statusColor(_ status: ExperimentStatus) -> Color {
        switch status {
        case .pending, .cancelled: return .gray
        case .queued: return .orange
        case .running: return .blue
        case .paused: return .yellow
        case .completed: return .green
        case .failed: return .red
        }
    }

    public static func statusColor(_ status: PhaseStatus) -> Color {
        switch status {
        case .pending, .skipped, .cancelled: return .gray
        case .running: return .blue
        case .paused: return .yellow
        case .completed: return .green
        case .failed: return .red
        }
    }

    public static func deltaColor(_ value: Double) -> Color { value < 0 ? .green : value == 0 ? .gray : .red }
    public static func roleColor(_ role: GenomeRole) -> Color {
        switch role {
        case .elite: return .yellow
        case .offspring: return .blue
        case .`init`: return .gray
        case .top_k: return .purple
        case .neighbor: return .orange
        case .current: return .green
        }
    }

    public static let ceLineColor = Color.blue
    public static let accuracyLineColor = Color.green
    public static let bestPointColor = Color.yellow
    #if canImport(UIKit)
    public static let cardBackground = Color(uiColor: UIColor.systemGray6)
    #else
    public static let cardBackground = Color.gray.opacity(0.1)
    #endif
}

public struct StatusBadge: View {
    public let text: String
    public let color: Color

    public init(text: String, color: Color) { self.text = text; self.color = color }

    public var body: some View {
        Text(text)
            .font(.caption2)
            .fontWeight(.medium)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(color.opacity(0.2))
            .foregroundColor(color)
            .cornerRadius(4)
    }
}
