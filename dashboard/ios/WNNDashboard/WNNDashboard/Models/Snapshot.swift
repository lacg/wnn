// Dashboard snapshot - full state for initial load and reconnection

import Foundation

struct DashboardSnapshot: Codable {
    let current_experiment: Experiment?
    let current_phase: Phase?
    let phases: [Phase]
    let iterations: [Iteration]
    let best_ce: Double
    let best_accuracy: Double

    /// Best accuracy as percentage
    var bestAccuracyPercent: Double {
        best_accuracy * 100
    }

    /// Empty snapshot for initial state
    static var empty: DashboardSnapshot {
        DashboardSnapshot(
            current_experiment: nil,
            current_phase: nil,
            phases: [],
            iterations: [],
            best_ce: 0,
            best_accuracy: 0
        )
    }
}
