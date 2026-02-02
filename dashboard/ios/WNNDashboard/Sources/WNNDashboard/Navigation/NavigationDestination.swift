// NavigationDestination - Enum for navigation targets

import SwiftUI

public enum NavigationDestination: String, CaseIterable, Identifiable {
    case dashboard
    case flows
    case checkpoints
    case settings

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .dashboard: return "Iterations"
        case .flows: return "Flows"
        case .checkpoints: return "Checkpoints"
        case .settings: return "Settings"
        }
    }

    public var icon: String {
        switch self {
        case .dashboard: return "chart.line.uptrend.xyaxis"
        case .flows: return "arrow.triangle.branch"
        case .checkpoints: return "externaldrive"
        case .settings: return "gear"
        }
    }
}
