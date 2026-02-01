// ProgressBarView - Phase progress indicator

import SwiftUI

struct ProgressBarView: View {
    let progress: Double
    let current: Int
    let max: Int

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Phase Progress")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Spacer()

                Text("\(current)/\(max)")
                    .font(.caption)
                    .fontDesign(.monospaced)
                    .foregroundColor(.secondary)
            }

            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    // Background
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Color.gray.opacity(0.2))

                    // Progress
                    RoundedRectangle(cornerRadius: 6)
                        .fill(progressGradient)
                        .frame(width: geometry.size.width * min(progress, 1.0))
                }
            }
            .frame(height: 12)

            // Percentage
            Text(String(format: "%.1f%%", progress * 100))
                .font(.caption2)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .trailing)
        }
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(12)
    }

    private var progressGradient: LinearGradient {
        LinearGradient(
            colors: [.blue, .purple],
            startPoint: .leading,
            endPoint: .trailing
        )
    }
}

// MARK: - Animated Progress Bar

struct AnimatedProgressBar: View {
    let progress: Double

    @State private var animatedProgress: Double = 0

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.gray.opacity(0.2))

                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.blue)
                    .frame(width: geometry.size.width * animatedProgress)
            }
        }
        .frame(height: 8)
        .onAppear {
            withAnimation(.easeInOut(duration: 0.5)) {
                animatedProgress = progress
            }
        }
        .onChange(of: progress) { _, newValue in
            withAnimation(.easeInOut(duration: 0.3)) {
                animatedProgress = newValue
            }
        }
    }
}

#Preview {
    VStack(spacing: 20) {
        ProgressBarView(progress: 0.65, current: 65, max: 100)

        AnimatedProgressBar(progress: 0.75)
            .frame(height: 8)
            .padding()
    }
    .padding()
}
