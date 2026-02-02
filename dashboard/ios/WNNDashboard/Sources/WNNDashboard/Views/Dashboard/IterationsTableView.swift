// IterationsTableView - iPad sortable table for iterations

import SwiftUI

@available(iOS 16.0, *)
public struct IterationsTableView: View {
    public let iterations: [Iteration]
    public let onSelect: (Iteration) -> Void

    @State private var sortOrder: [KeyPathComparator<Iteration>] = [
        .init(\.iteration_num, order: .reverse)
    ]

    public init(iterations: [Iteration], onSelect: @escaping (Iteration) -> Void) {
        self.iterations = iterations
        self.onSelect = onSelect
    }

    public var body: some View {
        Table(sortedIterations, sortOrder: $sortOrder) {
            TableColumn("#", value: \.iteration_num) { iteration in
                Text("\(iteration.iteration_num)")
                    .fontDesign(.monospaced)
            }
            .width(60)

            TableColumn("CE", value: \.best_ce) { iteration in
                Text(NumberFormatters.formatCE(iteration.best_ce))
                    .fontDesign(.monospaced)
            }
            .width(100)

            TableColumn("Accuracy") { iteration in
                if let acc = iteration.best_accuracy {
                    Text(NumberFormatters.formatAccuracy(acc))
                        .fontDesign(.monospaced)
                } else {
                    Text("-")
                        .foregroundColor(.secondary)
                }
            }
            .width(100)

            TableColumn("Delta") { iteration in
                if let delta = iteration.delta_previous {
                    HStack(spacing: 2) {
                        Image(systemName: delta < 0 ? "arrow.down" : delta > 0 ? "arrow.up" : "minus")
                            .font(.caption2)
                        Text(String(format: "%.4f", abs(delta)))
                            .fontDesign(.monospaced)
                    }
                    .foregroundColor(Theme.deltaColor(delta))
                } else {
                    Text("-")
                        .foregroundColor(.secondary)
                }
            }
            .width(100)

            TableColumn("Time") { iteration in
                if let date = iteration.createdDate {
                    Text(DateFormatters.relative(date))
                        .foregroundColor(.secondary)
                } else {
                    Text("-")
                        .foregroundColor(.secondary)
                }
            }
        }
        .tableStyle(.inset)
    }

    private var sortedIterations: [Iteration] {
        iterations.sorted(using: sortOrder)
    }
}
