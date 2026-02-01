// IterationsListView - Scrollable list of iterations

import SwiftUI

public struct IterationsListView: View {
    public let iterations: [Iteration]
    public let onSelect: (Iteration) -> Void

    public init(iterations: [Iteration], onSelect: @escaping (Iteration) -> Void) {
        self.iterations = iterations; self.onSelect = onSelect
    }

    public var body: some View {
        LazyVStack(spacing: 8) {
            ForEach(iterations) { iter in
                IterationRow(iteration: iter).onTapGesture { onSelect(iter) }
            }
        }
    }
}

struct IterationRow: View {
    let iteration: Iteration

    var body: some View {
        HStack(spacing: 12) {
            Text("#\(iteration.iteration_num)").font(.subheadline).fontWeight(.medium).fontDesign(.monospaced).frame(width: 50, alignment: .leading)
            VStack(alignment: .leading, spacing: 2) {
                Text("CE").font(.caption2).foregroundColor(.secondary)
                Text(NumberFormatters.formatCE(iteration.best_ce)).font(.caption).fontDesign(.monospaced)
            }
            .frame(width: 70, alignment: .leading)
            VStack(alignment: .leading, spacing: 2) {
                Text("Acc").font(.caption2).foregroundColor(.secondary)
                if let acc = iteration.best_accuracy { Text(NumberFormatters.formatAccuracy(acc)).font(.caption).fontDesign(.monospaced) }
                else { Text("-").font(.caption).foregroundColor(.secondary) }
            }
            .frame(width: 60, alignment: .leading)
            Spacer()
            if let delta = iteration.delta_previous {
                HStack(spacing: 2) {
                    Image(systemName: delta < 0 ? "arrow.down" : delta > 0 ? "arrow.up" : "minus").font(.caption2)
                    Text(String(format: "%.4f", abs(delta))).font(.caption2).fontDesign(.monospaced)
                }
                .foregroundColor(Theme.deltaColor(delta))
            }
            if let date = iteration.createdDate { Text(DateFormatters.relative(date)).font(.caption2).foregroundColor(.secondary) }
            Image(systemName: "chevron.right").font(.caption).foregroundColor(.secondary)
        }
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(8)
        .padding(.horizontal)
    }
}
