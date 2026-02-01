// Status enums matching Rust backend (serde rename_all = "snake_case")

import Foundation

enum FlowStatus: String, Codable, CaseIterable {
    case pending
    case queued
    case running
    case paused
    case completed
    case failed
    case cancelled

    var displayName: String {
        rawValue.capitalized
    }

    var isActive: Bool {
        self == .running || self == .queued
    }

    var isTerminal: Bool {
        self == .completed || self == .failed || self == .cancelled
    }
}

enum ExperimentStatus: String, Codable, CaseIterable {
    case pending
    case queued
    case running
    case paused
    case completed
    case failed
    case cancelled

    var displayName: String {
        rawValue.capitalized
    }

    var isActive: Bool {
        self == .running || self == .queued
    }
}

enum PhaseStatus: String, Codable, CaseIterable {
    case pending
    case running
    case paused
    case completed
    case skipped
    case failed
    case cancelled

    var displayName: String {
        rawValue.capitalized
    }

    var isActive: Bool {
        self == .running
    }
}

enum FitnessCalculator: String, Codable {
    case ce
    case harmonic_rank
    case weighted_harmonic

    var displayName: String {
        switch self {
        case .ce: return "CE"
        case .harmonic_rank: return "Harmonic Rank"
        case .weighted_harmonic: return "Weighted Harmonic"
        }
    }
}

enum GenomeRole: String, Codable {
    case elite
    case offspring
    case `init`
    case top_k
    case neighbor
    case current

    var displayName: String {
        switch self {
        case .elite: return "Elite"
        case .offspring: return "Offspring"
        case .`init`: return "Init"
        case .top_k: return "Top-K"
        case .neighbor: return "Neighbor"
        case .current: return "Current"
        }
    }
}

enum CheckpointType: String, Codable {
    case auto
    case user
    case phase_end
    case experiment_end

    var displayName: String {
        switch self {
        case .auto: return "Auto"
        case .user: return "User"
        case .phase_end: return "Phase End"
        case .experiment_end: return "Experiment End"
        }
    }
}

enum ExperimentType: String, Codable {
    case ga
    case ts

    var displayName: String {
        switch self {
        case .ga: return "GA"
        case .ts: return "TS"
        }
    }
}
