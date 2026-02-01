// Formatters for numbers, dates, and durations

import Foundation

/// Number formatters for CE and accuracy values
enum NumberFormatters {
    /// Format CE value (4 decimal places)
    static func formatCE(_ value: Double) -> String {
        String(format: "%.4f", value)
    }

    /// Format accuracy as percentage (2 decimal places)
    static func formatAccuracy(_ value: Double) -> String {
        String(format: "%.2f%%", value * 100)
    }

    /// Format accuracy that's already a percentage
    static func formatPercent(_ value: Double) -> String {
        String(format: "%.2f%%", value)
    }

    /// Format delta value with sign
    static func formatDelta(_ value: Double) -> String {
        if value > 0 {
            return String(format: "+%.4f", value)
        } else {
            return String(format: "%.4f", value)
        }
    }

    /// Format large numbers with K/M suffix
    static func formatCompact(_ value: Int) -> String {
        if value >= 1_000_000 {
            return String(format: "%.1fM", Double(value) / 1_000_000)
        } else if value >= 1_000 {
            return String(format: "%.1fK", Double(value) / 1_000)
        }
        return "\(value)"
    }

    /// Format bytes as human-readable size
    static func formatBytes(_ bytes: Int64) -> String {
        if bytes < 1024 {
            return "\(bytes) B"
        } else if bytes < 1024 * 1024 {
            return String(format: "%.1f KB", Double(bytes) / 1024)
        } else if bytes < 1024 * 1024 * 1024 {
            return String(format: "%.1f MB", Double(bytes) / (1024 * 1024))
        } else {
            return String(format: "%.2f GB", Double(bytes) / (1024 * 1024 * 1024))
        }
    }
}

/// Date formatters for timestamps and durations
enum DateFormatters {
    private static let iso8601: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private static let iso8601NoFraction: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    private static let relativeFormatter: RelativeDateTimeFormatter = {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter
    }()

    private static let shortTime: DateFormatter = {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        formatter.dateStyle = .none
        return formatter
    }()

    private static let shortDateTime: DateFormatter = {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        formatter.dateStyle = .short
        return formatter
    }()

    /// Parse ISO8601 date string
    static func parse(_ string: String) -> Date? {
        iso8601.date(from: string) ?? iso8601NoFraction.date(from: string)
    }

    /// Format date as relative time (e.g., "2h ago")
    static func relative(_ date: Date) -> String {
        relativeFormatter.localizedString(for: date, relativeTo: Date())
    }

    /// Format date as short time (e.g., "3:45 PM")
    static func shortTime(_ date: Date) -> String {
        shortTime.string(from: date)
    }

    /// Format date as short date/time (e.g., "1/15/26, 3:45 PM")
    static func shortDateTime(_ date: Date) -> String {
        shortDateTime.string(from: date)
    }

    /// Format duration in seconds as human-readable string
    static func duration(_ seconds: TimeInterval) -> String {
        if seconds < 60 {
            return String(format: "%.0fs", seconds)
        } else if seconds < 3600 {
            let mins = Int(seconds) / 60
            let secs = Int(seconds) % 60
            return secs > 0 ? "\(mins)m \(secs)s" : "\(mins)m"
        } else if seconds < 86400 {
            let hours = Int(seconds) / 3600
            let mins = (Int(seconds) % 3600) / 60
            return mins > 0 ? "\(hours)h \(mins)m" : "\(hours)h"
        } else {
            let days = Int(seconds) / 86400
            let hours = (Int(seconds) % 86400) / 3600
            return hours > 0 ? "\(days)d \(hours)h" : "\(days)d"
        }
    }

    /// Format duration as compact string (e.g., "2h 30m")
    static func durationCompact(_ seconds: TimeInterval) -> String {
        let hours = Int(seconds) / 3600
        let mins = (Int(seconds) % 3600) / 60

        if hours > 0 {
            return mins > 0 ? "\(hours)h \(mins)m" : "\(hours)h"
        } else if mins > 0 {
            return "\(mins)m"
        } else {
            return String(format: "%.0fs", seconds)
        }
    }
}
