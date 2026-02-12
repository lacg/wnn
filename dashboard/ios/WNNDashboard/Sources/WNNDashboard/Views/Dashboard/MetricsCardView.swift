// MetricsCardView - Individual metric display card

import SwiftUI

public struct MetricsCardView: View {
    public let title: String
    public let value: String
    public let icon: String
    public let color: Color

    public init(title: String, value: String, icon: String, color: Color) {
        self.title = title; self.value = value; self.icon = icon; self.color = color
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon).foregroundColor(color)
                Text(title).font(.caption).foregroundColor(.secondary)
            }
            Text(value).font(.title2).fontWeight(.semibold).fontDesign(.monospaced)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(12)
    }
}
