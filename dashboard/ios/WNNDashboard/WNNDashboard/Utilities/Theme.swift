// Theme - colors and styling

import SwiftUI

/// App theme colors
enum Theme {
    // MARK: - Status Colors

    static func statusColor(_ status: FlowStatus) -> Color {
        switch status {
        case .pending: return .gray
        case .queued: return .orange
        case .running: return .blue
        case .paused: return .yellow
        case .completed: return .green
        case .failed: return .red
        case .cancelled: return .gray
        }
    }

    static func statusColor(_ status: ExperimentStatus) -> Color {
        switch status {
        case .pending: return .gray
        case .queued: return .orange
        case .running: return .blue
        case .paused: return .yellow
        case .completed: return .green
        case .failed: return .red
        case .cancelled: return .gray
        }
    }

    static func statusColor(_ status: PhaseStatus) -> Color {
        switch status {
        case .pending: return .gray
        case .running: return .blue
        case .paused: return .yellow
        case .completed: return .green
        case .skipped: return .gray
        case .failed: return .red
        case .cancelled: return .gray
        }
    }

    // MARK: - Metric Colors

    /// Color for CE values (lower is better, so red = bad, green = good)
    static func ceColor(_ value: Double, best: Double) -> Color {
        let ratio = value / best
        if ratio < 1.1 {
            return .green
        } else if ratio < 1.5 {
            return .yellow
        } else {
            return .red
        }
    }

    /// Color for accuracy values (higher is better)
    static func accuracyColor(_ value: Double) -> Color {
        if value >= 0.1 {
            return .green
        } else if value >= 0.05 {
            return .yellow
        } else {
            return .red
        }
    }

    /// Color for delta values (negative is improvement for CE)
    static func deltaColor(_ value: Double) -> Color {
        if value < 0 {
            return .green  // Improvement
        } else if value == 0 {
            return .gray   // No change
        } else {
            return .red    // Regression
        }
    }

    // MARK: - Role Colors

    static func roleColor(_ role: GenomeRole) -> Color {
        switch role {
        case .elite: return .yellow
        case .offspring: return .blue
        case .`init`: return .gray
        case .top_k: return .purple
        case .neighbor: return .orange
        case .current: return .green
        }
    }

    // MARK: - Chart Colors

    static let ceLineColor = Color.blue
    static let accuracyLineColor = Color.green
    static let bestPointColor = Color.yellow

    // MARK: - Background Colors

    static let cardBackground = Color(.systemGray6)
    static let sectionBackground = Color(.systemBackground)

    // MARK: - Connection Status

    static func connectionColor(_ isConnected: Bool) -> Color {
        isConnected ? .green : .red
    }
}

// MARK: - Status Badge View

struct StatusBadge: View {
    let text: String
    let color: Color

    var body: some View {
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

// MARK: - Flow Status Badge

extension View {
    func flowStatusBadge(_ status: FlowStatus) -> some View {
        StatusBadge(text: status.displayName, color: Theme.statusColor(status))
    }

    func experimentStatusBadge(_ status: ExperimentStatus) -> some View {
        StatusBadge(text: status.displayName, color: Theme.statusColor(status))
    }

    func phaseStatusBadge(_ status: PhaseStatus) -> some View {
        StatusBadge(text: status.displayName, color: Theme.statusColor(status))
    }
}
