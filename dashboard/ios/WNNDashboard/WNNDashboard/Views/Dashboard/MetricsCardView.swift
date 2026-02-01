// MetricsCardView - Individual metric display card

import SwiftUI

struct MetricsCardView: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    var subtitle: String? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)

                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Text(value)
                .font(.title2)
                .fontWeight(.semibold)
                .fontDesign(.monospaced)

            if let subtitle = subtitle {
                Text(subtitle)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(12)
    }
}

// MARK: - Specialized Metric Cards

struct CEMetricCard: View {
    let value: Double
    let delta: Double?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "arrow.down.circle")
                    .foregroundColor(.blue)

                Text("Best CE")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Text(NumberFormatters.formatCE(value))
                .font(.title2)
                .fontWeight(.semibold)
                .fontDesign(.monospaced)

            if let delta = delta {
                HStack(spacing: 2) {
                    Image(systemName: delta < 0 ? "arrow.down" : "arrow.up")
                        .font(.caption2)

                    Text(NumberFormatters.formatDelta(delta))
                        .font(.caption2)
                        .fontDesign(.monospaced)
                }
                .foregroundColor(Theme.deltaColor(delta))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(12)
    }
}

struct AccuracyMetricCard: View {
    let value: Double
    let delta: Double?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "arrow.up.circle")
                    .foregroundColor(.green)

                Text("Best Accuracy")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Text(NumberFormatters.formatAccuracy(value))
                .font(.title2)
                .fontWeight(.semibold)
                .fontDesign(.monospaced)

            if let delta = delta {
                HStack(spacing: 2) {
                    Image(systemName: delta > 0 ? "arrow.up" : "arrow.down")
                        .font(.caption2)

                    Text(NumberFormatters.formatDelta(delta * 100))
                        .font(.caption2)
                        .fontDesign(.monospaced)
                }
                .foregroundColor(Theme.deltaColor(-delta))  // For accuracy, positive is good
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(12)
    }
}

#Preview {
    VStack(spacing: 16) {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
            MetricsCardView(
                title: "Best CE",
                value: "10.6543",
                icon: "arrow.down.circle",
                color: .blue
            )

            MetricsCardView(
                title: "Best Accuracy",
                value: "1.23%",
                icon: "arrow.up.circle",
                color: .green
            )
        }

        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
            CEMetricCard(value: 10.6543, delta: -0.0012)
            AccuracyMetricCard(value: 0.0123, delta: 0.0005)
        }
    }
    .padding()
}
