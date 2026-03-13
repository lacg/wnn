// Formatters - Number and date formatting utilities

import Foundation

public enum Formatters {
	public static func ce(_ value: Double) -> String {
		String(format: "%.4f", value)
	}

	public static func accuracy(_ value: Double) -> String {
		String(format: "%.2f%%", value * 100)
	}

	public static func percent(_ value: Double?) -> String {
		guard let value else { return "-" }
		return String(format: "%.2f%%", value * 100)
	}

	public static func compact(_ value: Int) -> String {
		if value >= 1_000_000 { return String(format: "%.1fM", Double(value) / 1_000_000) }
		if value >= 1_000 { return String(format: "%.1fK", Double(value) / 1_000) }
		return "\(value)"
	}

	public static func duration(_ seconds: Double) -> String {
		if seconds < 60 { return String(format: "%.0fs", seconds) }
		if seconds < 3600 {
			let m = Int(seconds) / 60
			let s = Int(seconds) % 60
			return "\(m)m \(s)s"
		}
		let h = Int(seconds) / 3600
		let m = (Int(seconds) % 3600) / 60
		return "\(h)h \(m)m"
	}

	private static let isoFormatter: ISO8601DateFormatter = {
		let f = ISO8601DateFormatter()
		f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
		return f
	}()

	private static let isoFormatterNoFrac: ISO8601DateFormatter = {
		let f = ISO8601DateFormatter()
		f.formatOptions = [.withInternetDateTime]
		return f
	}()

	public static func parseDate(_ string: String) -> Date? {
		isoFormatter.date(from: string) ?? isoFormatterNoFrac.date(from: string)
	}

	public static func relativeDate(_ string: String) -> String {
		guard let date = parseDate(string) else { return string }
		let formatter = RelativeDateTimeFormatter()
		formatter.unitsStyle = .abbreviated
		return formatter.localizedString(for: date, relativeTo: .now)
	}
}
