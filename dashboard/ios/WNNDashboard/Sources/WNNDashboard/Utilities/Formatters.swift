// Formatters for numbers, dates, and durations

import Foundation

public enum NumberFormatters {
    public static func formatCE(_ value: Double) -> String { String(format: "%.4f", value) }
    public static func formatCE(_ value: Double?) -> String { value.map { String(format: "%.4f", $0) } ?? "-" }
    public static func formatAccuracy(_ value: Double) -> String { String(format: "%.2f%%", value * 100) }
    public static func formatAccuracy(_ value: Double?) -> String { value.map { String(format: "%.2f%%", $0 * 100) } ?? "-" }
    public static func formatPercent(_ value: Double) -> String { String(format: "%.2f%%", value) }
    public static func formatDelta(_ value: Double) -> String { value > 0 ? String(format: "+%.4f", value) : String(format: "%.4f", value) }
    public static func formatCompact(_ value: Int) -> String {
        if value >= 1_000_000 { return String(format: "%.1fM", Double(value) / 1_000_000) }
        if value >= 1_000 { return String(format: "%.1fK", Double(value) / 1_000) }
        return "\(value)"
    }
}

public enum DateFormatters {
    private static let iso8601: ISO8601DateFormatter = { let f = ISO8601DateFormatter(); f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]; return f }()
    private static let relativeFormatter: RelativeDateTimeFormatter = { let f = RelativeDateTimeFormatter(); f.unitsStyle = .abbreviated; return f }()
    private static let shortDateTime: DateFormatter = { let f = DateFormatter(); f.timeStyle = .short; f.dateStyle = .short; return f }()
    private static let timeOnly: DateFormatter = { let f = DateFormatter(); f.timeStyle = .medium; f.dateStyle = .none; return f }()

    public static func parse(_ string: String) -> Date? { iso8601.date(from: string) }
    public static func relative(_ date: Date) -> String { relativeFormatter.localizedString(for: date, relativeTo: Date()) }
    public static func shortDateTime(_ date: Date) -> String { shortDateTime.string(from: date) }
    public static func timeOnly(_ date: Date) -> String { timeOnly.string(from: date) }

    public static func duration(_ seconds: TimeInterval) -> String {
        if seconds < 60 { return String(format: "%.0fs", seconds) }
        if seconds < 3600 { let m = Int(seconds) / 60, s = Int(seconds) % 60; return s > 0 ? "\(m)m \(s)s" : "\(m)m" }
        if seconds < 86400 { let h = Int(seconds) / 3600, m = (Int(seconds) % 3600) / 60; return m > 0 ? "\(h)h \(m)m" : "\(h)h" }
        let d = Int(seconds) / 86400, h = (Int(seconds) % 86400) / 3600; return h > 0 ? "\(d)d \(h)h" : "\(d)d"
    }

    public static func durationCompact(_ seconds: TimeInterval) -> String {
        let h = Int(seconds) / 3600, m = (Int(seconds) % 3600) / 60
        if h > 0 { return m > 0 ? "\(h)h \(m)m" : "\(h)h" }
        if m > 0 { return "\(m)m" }
        return String(format: "%.0fs", seconds)
    }
}
