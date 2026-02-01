// ProgressBarView - Phase progress indicator

import SwiftUI

public struct ProgressBarView: View {
    public let progress: Double
    public let current: Int
    public let max: Int

    public init(progress: Double, current: Int, max: Int) {
        self.progress = progress; self.current = current; self.max = max
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Phase Progress").font(.caption).foregroundColor(.secondary)
                Spacer()
                Text("\(current)/\(max)").font(.caption).fontDesign(.monospaced).foregroundColor(.secondary)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 6).fill(Color.gray.opacity(0.2))
                    RoundedRectangle(cornerRadius: 6)
                        .fill(LinearGradient(colors: [.blue, .purple], startPoint: .leading, endPoint: .trailing))
                        .frame(width: geo.size.width * min(progress, 1.0))
                }
            }
            .frame(height: 12)
            Text(String(format: "%.1f%%", progress * 100)).font(.caption2).foregroundColor(.secondary).frame(maxWidth: .infinity, alignment: .trailing)
        }
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(12)
    }
}
