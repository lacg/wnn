// Status enums matching Rust backend (serde rename_all = "snake_case")

import Foundation

public enum FlowStatus: String, Codable, CaseIterable {
    case pending, queued, running, paused, completed, failed, cancelled

    public var displayName: String { rawValue.capitalized }
    public var isActive: Bool { self == .running || self == .queued }
    public var isTerminal: Bool { self == .completed || self == .failed || self == .cancelled }
}

public enum ExperimentStatus: String, Codable, CaseIterable {
    case pending, queued, running, paused, completed, failed, cancelled

    public var displayName: String { rawValue.capitalized }
    public var isActive: Bool { self == .running || self == .queued }
}

public enum PhaseStatus: String, Codable, CaseIterable {
    case pending, queued, running, paused, completed, failed, cancelled, skipped

    public var displayName: String { rawValue.capitalized }
    public var isActive: Bool { self == .running || self == .queued }
}

public enum FitnessCalculator: String, Codable {
    case ce, harmonic_rank, weighted_harmonic, unknown

    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = FitnessCalculator(rawValue: raw) ?? .unknown
    }

    public var displayName: String {
        switch self {
        case .ce: return "CE"
        case .harmonic_rank: return "Harmonic Rank"
        case .weighted_harmonic: return "Weighted Harmonic"
        case .unknown: return "Unknown"
        }
    }
}

public enum GenomeRole: String, Codable {
    case elite, offspring, `init`, top_k, neighbor, current

    public var displayName: String {
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

public enum CheckpointType: String, Codable, CaseIterable {
    case auto, user, phase_end, experiment_end

    public var displayName: String {
        switch self {
        case .auto: return "Auto"
        case .user: return "User"
        case .phase_end: return "Phase End"
        case .experiment_end: return "Experiment End"
        }
    }
}

public enum ExperimentType: String, Codable {
    case ga, ts, unknown

    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = ExperimentType(rawValue: raw) ?? .unknown
    }

    public var displayName: String {
        switch self {
        case .ga: return "GA"
        case .ts: return "TS"
        case .unknown: return "?"
        }
    }
}

public enum ArchitectureType: String, Codable {
    case tiered, bitwise

    public var displayName: String {
        switch self {
        case .tiered: return "Tiered"
        case .bitwise: return "Bitwise"
        }
    }

    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = ArchitectureType(rawValue: raw) ?? .tiered
    }
}

public enum GatingStatus: String, Codable, CaseIterable {
    case pending, running, completed, failed

    public var displayName: String { rawValue.capitalized }
    public var isActive: Bool { self == .running || self == .pending }
    public var isTerminal: Bool { self == .completed || self == .failed }
}
