// IterationsListView - Scrollable list of iterations

import SwiftUI

struct IterationsListView: View {
    let iterations: [Iteration]
    let onSelect: (Iteration) -> Void

    var body: some View {
        LazyVStack(spacing: 8) {
            ForEach(iterations) { iteration in
                IterationRow(iteration: iteration)
                    .onTapGesture {
                        onSelect(iteration)
                    }
            }
        }
    }
}

struct IterationRow: View {
    let iteration: Iteration

    var body: some View {
        HStack(spacing: 12) {
            // Iteration number
            Text("#\(iteration.iteration_num)")
                .font(.subheadline)
                .fontWeight(.medium)
                .fontDesign(.monospaced)
                .frame(width: 50, alignment: .leading)

            // CE
            VStack(alignment: .leading, spacing: 2) {
                Text("CE")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Text(NumberFormatters.formatCE(iteration.best_ce))
                    .font(.caption)
                    .fontDesign(.monospaced)
            }
            .frame(width: 70, alignment: .leading)

            // Accuracy
            VStack(alignment: .leading, spacing: 2) {
                Text("Acc")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                if let acc = iteration.best_accuracy {
                    Text(NumberFormatters.formatAccuracy(acc))
                        .font(.caption)
                        .fontDesign(.monospaced)
                } else {
                    Text("-")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .frame(width: 60, alignment: .leading)

            Spacer()

            // Delta indicator
            if let delta = iteration.delta_previous {
                deltaIndicator(delta)
            }

            // Time
            if let date = iteration.createdDate {
                Text(DateFormatters.relative(date))
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(8)
        .padding(.horizontal)
    }

    private func deltaIndicator(_ delta: Double) -> some View {
        HStack(spacing: 2) {
            Image(systemName: delta < 0 ? "arrow.down" : delta > 0 ? "arrow.up" : "minus")
                .font(.caption2)

            Text(String(format: "%.4f", abs(delta)))
                .font(.caption2)
                .fontDesign(.monospaced)
        }
        .foregroundColor(Theme.deltaColor(delta))
    }
}

// MARK: - Compact Iteration Row

struct CompactIterationRow: View {
    let iteration: Iteration

    var body: some View {
        HStack(spacing: 8) {
            Text("#\(iteration.iteration_num)")
                .font(.caption)
                .fontWeight(.medium)
                .fontDesign(.monospaced)

            Spacer()

            Text(NumberFormatters.formatCE(iteration.best_ce))
                .font(.caption)
                .fontDesign(.monospaced)

            if let acc = iteration.best_accuracy {
                Text(NumberFormatters.formatAccuracy(acc))
                    .font(.caption)
                    .fontDesign(.monospaced)
            }
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    let sampleIterations = (1...5).map { i in
        Iteration(
            id: Int64(i),
            phase_id: 1,
            iteration_num: Int32(100 - i),
            best_ce: 10.5 + Double(i) * 0.01,
            best_accuracy: 0.012 - Double(i) * 0.001,
            avg_ce: 11.0,
            avg_accuracy: 0.01,
            elite_count: 5,
            offspring_count: 45,
            offspring_viable: 42,
            fitness_threshold: nil,
            elapsed_secs: 12.5,
            baseline_ce: nil,
            delta_baseline: nil,
            delta_previous: i % 2 == 0 ? -0.01 : 0.005,
            patience_counter: 3,
            patience_max: 10,
            candidates_total: 50,
            created_at: "2026-01-31T18:45:23Z"
        )
    }

    ScrollView {
        IterationsListView(iterations: sampleIterations) { _ in }
    }
}
