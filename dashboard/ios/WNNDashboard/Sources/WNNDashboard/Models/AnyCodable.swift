// AnyCodable wrapper for handling dynamic JSON values

import Foundation

public struct AnyCodable: Codable, Equatable, Hashable {
    public let value: Any

    public init(_ value: Any) { self.value = value }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { value = NSNull() }
        else if let b = try? container.decode(Bool.self) { value = b }
        else if let i = try? container.decode(Int.self) { value = i }
        else if let d = try? container.decode(Double.self) { value = d }
        else if let s = try? container.decode(String.self) { value = s }
        else if let a = try? container.decode([AnyCodable].self) { value = a.map { $0.value } }
        else if let d = try? container.decode([String: AnyCodable].self) { value = d.mapValues { $0.value } }
        else { throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unable to decode") }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case is NSNull: try container.encodeNil()
        case let b as Bool: try container.encode(b)
        case let i as Int: try container.encode(i)
        case let d as Double: try container.encode(d)
        case let s as String: try container.encode(s)
        case let a as [Any]: try container.encode(a.map { AnyCodable($0) })
        case let d as [String: Any]: try container.encode(d.mapValues { AnyCodable($0) })
        default: throw EncodingError.invalidValue(value, .init(codingPath: [], debugDescription: "Unable to encode"))
        }
    }

    public static func == (lhs: AnyCodable, rhs: AnyCodable) -> Bool {
        switch (lhs.value, rhs.value) {
        case is (NSNull, NSNull): return true
        case let (l as Bool, r as Bool): return l == r
        case let (l as Int, r as Int): return l == r
        case let (l as Double, r as Double): return l == r
        case let (l as String, r as String): return l == r
        default: return false
        }
    }

    public func hash(into hasher: inout Hasher) {
        switch value {
        case is NSNull: hasher.combine(0)
        case let b as Bool: hasher.combine(b)
        case let i as Int: hasher.combine(i)
        case let d as Double: hasher.combine(d)
        case let s as String: hasher.combine(s)
        default: hasher.combine(1)
        }
    }

    public var stringValue: String? { value as? String }
    public var intValue: Int? { value as? Int }
    public var doubleValue: Double? { value as? Double }
    public var boolValue: Bool? { value as? Bool }
}
