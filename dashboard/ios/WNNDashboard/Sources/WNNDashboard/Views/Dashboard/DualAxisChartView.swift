// DualAxisChartView - CE vs Accuracy chart

import SwiftUI
import Charts

public struct DualAxisChartView: View {
    public let iterations: [Iteration]

    public init(iterations: [Iteration]) { self.iterations = iterations }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Performance Over Time").font(.headline)
            chartContent
            HStack(spacing: 16) {
                HStack(spacing: 4) { Circle().fill(Theme.ceLineColor).frame(width: 8, height: 8); Text("CE").font(.caption2).foregroundColor(.secondary) }
                HStack(spacing: 4) { Circle().fill(Theme.accuracyLineColor).frame(width: 8, height: 8); Text("Accuracy").font(.caption2).foregroundColor(.secondary) }
            }
        }
    }

    @ViewBuilder
    private var chartContent: some View {
        if #available(iOS 16.0, *) {
            Chart {
                ForEach(chartData, id: \.iterationNum) { pt in
                    LineMark(x: .value("Iter", pt.iterationNum), y: .value("CE", pt.ceNorm), series: .value("M", "CE")).foregroundStyle(Theme.ceLineColor).interpolationMethod(.monotone)
                }
                ForEach(chartData, id: \.iterationNum) { pt in
                    LineMark(x: .value("Iter", pt.iterationNum), y: .value("Acc", pt.accNorm), series: .value("M", "Acc")).foregroundStyle(Theme.accuracyLineColor).interpolationMethod(.monotone)
                }
            }
            .chartYScale(domain: 0...1)
        } else {
            Text("Charts require iOS 16+").foregroundColor(.secondary)
        }
    }

    private var chartData: [(iterationNum: Int, ceNorm: Double, accNorm: Double)] {
        let sorted = iterations.sorted { $0.iteration_num < $1.iteration_num }
        let ces = sorted.map { $0.best_ce }, accs = sorted.compactMap { $0.best_accuracy }
        let minCE = ces.min() ?? 0, maxCE = ces.max() ?? 1, ceRange = Swift.max(maxCE - minCE, 0.001)
        let minAcc = accs.min() ?? 0, maxAcc = accs.max() ?? 1, accRange = Swift.max(maxAcc - minAcc, 0.001)
        return sorted.map { iter in
            let ceNorm = 1.0 - (iter.best_ce - minCE) / ceRange
            let accNorm = ((iter.best_accuracy ?? 0) - minAcc) / accRange
            return (Int(iter.iteration_num), ceNorm, accNorm)
        }
    }
}
