// Dashboard snapshot - full state for initial load

import Foundation

public struct DashboardSnapshot: Codable {
    public let current_experiment: Experiment?
    public let current_phase: Phase?
    public let phases: [Phase]
    public let iterations: [Iteration]
    public let best_ce: Double
    public let best_accuracy: Double

    public var bestAccuracyPercent: Double { best_accuracy * 100 }

    public static var empty: DashboardSnapshot {
        DashboardSnapshot(current_experiment: nil, current_phase: nil, phases: [], iterations: [], best_ce: 0, best_accuracy: 0)
    }

    public init(current_experiment: Experiment?, current_phase: Phase?, phases: [Phase], iterations: [Iteration], best_ce: Double, best_accuracy: Double) {
        self.current_experiment = current_experiment
        self.current_phase = current_phase
        self.phases = phases
        self.iterations = iterations
        self.best_ce = best_ce
        self.best_accuracy = best_accuracy
    }
}
