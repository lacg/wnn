// DualAxisChartView - CE vs Accuracy chart using Swift Charts

import SwiftUI
import Charts

struct DualAxisChartView: View {
    let iterations: [Iteration]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Performance Over Time")
                .font(.headline)

            if #available(iOS 16.0, *) {
                chartContent
            } else {
                // Fallback for iOS 15
                fallbackChart
            }

            // Legend
            HStack(spacing: 16) {
                HStack(spacing: 4) {
                    Circle()
                        .fill(Theme.ceLineColor)
                        .frame(width: 8, height: 8)
                    Text("CE (lower=better)")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }

                HStack(spacing: 4) {
                    Circle()
                        .fill(Theme.accuracyLineColor)
                        .frame(width: 8, height: 8)
                    Text("Accuracy (higher=better)")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
        }
    }

    @available(iOS 16.0, *)
    private var chartContent: some View {
        Chart {
            // CE line (inverted for visual comparison)
            ForEach(chartData, id: \.iterationNum) { point in
                LineMark(
                    x: .value("Iteration", point.iterationNum),
                    y: .value("CE", point.ceNormalized),
                    series: .value("Metric", "CE")
                )
                .foregroundStyle(Theme.ceLineColor)
                .interpolationMethod(.monotone)
            }

            // Accuracy line
            ForEach(chartData, id: \.iterationNum) { point in
                LineMark(
                    x: .value("Iteration", point.iterationNum),
                    y: .value("Accuracy", point.accuracyNormalized),
                    series: .value("Metric", "Accuracy")
                )
                .foregroundStyle(Theme.accuracyLineColor)
                .interpolationMethod(.monotone)
            }

            // Best CE point marker
            if let bestCE = bestCEPoint {
                PointMark(
                    x: .value("Iteration", bestCE.iterationNum),
                    y: .value("CE", bestCE.ceNormalized)
                )
                .foregroundStyle(Theme.bestPointColor)
                .symbolSize(100)
            }

            // Best accuracy point marker
            if let bestAcc = bestAccuracyPoint {
                PointMark(
                    x: .value("Iteration", bestAcc.iterationNum),
                    y: .value("Accuracy", bestAcc.accuracyNormalized)
                )
                .foregroundStyle(Theme.bestPointColor)
                .symbolSize(100)
            }
        }
        .chartYScale(domain: 0...1)
        .chartXAxisLabel("Iteration")
        .chartYAxisLabel("Normalized")
    }

    private var fallbackChart: some View {
        // Simple fallback for iOS 15
        GeometryReader { geometry in
            Path { path in
                guard !chartData.isEmpty else { return }

                let width = geometry.size.width
                let height = geometry.size.height
                let stepX = width / CGFloat(max(chartData.count - 1, 1))

                for (index, point) in chartData.enumerated() {
                    let x = CGFloat(index) * stepX
                    let y = height * (1 - CGFloat(point.ceNormalized))

                    if index == 0 {
                        path.move(to: CGPoint(x: x, y: y))
                    } else {
                        path.addLine(to: CGPoint(x: x, y: y))
                    }
                }
            }
            .stroke(Theme.ceLineColor, lineWidth: 2)
        }
    }

    // MARK: - Data Processing

    private struct ChartPoint {
        let iterationNum: Int
        let ce: Double
        let accuracy: Double
        let ceNormalized: Double
        let accuracyNormalized: Double
    }

    private var chartData: [ChartPoint] {
        let sortedIterations = iterations.sorted { $0.iteration_num < $1.iteration_num }

        // Find min/max for normalization
        let ceValues = sortedIterations.map { $0.best_ce }
        let accValues = sortedIterations.compactMap { $0.best_accuracy }

        let minCE = ceValues.min() ?? 0
        let maxCE = ceValues.max() ?? 1
        let ceRange = max(maxCE - minCE, 0.001)

        let minAcc = accValues.min() ?? 0
        let maxAcc = accValues.max() ?? 1
        let accRange = max(maxAcc - minAcc, 0.001)

        return sortedIterations.map { iteration in
            // Normalize CE (inverted: lower CE = higher normalized value)
            let ceNorm = 1.0 - (iteration.best_ce - minCE) / ceRange

            // Normalize accuracy
            let acc = iteration.best_accuracy ?? 0
            let accNorm = (acc - minAcc) / accRange

            return ChartPoint(
                iterationNum: Int(iteration.iteration_num),
                ce: iteration.best_ce,
                accuracy: acc,
                ceNormalized: ceNorm,
                accuracyNormalized: accNorm
            )
        }
    }

    private var bestCEPoint: ChartPoint? {
        chartData.min { $0.ce < $1.ce }
    }

    private var bestAccuracyPoint: ChartPoint? {
        chartData.max { $0.accuracy < $1.accuracy }
    }
}

#Preview {
    let sampleIterations = (1...20).map { i in
        Iteration(
            id: Int64(i),
            phase_id: 1,
            iteration_num: Int32(i),
            best_ce: 12.0 - Double(i) * 0.1 + Double.random(in: -0.05...0.05),
            best_accuracy: 0.01 + Double(i) * 0.002 + Double.random(in: -0.001...0.001),
            avg_ce: nil,
            avg_accuracy: nil,
            elite_count: 5,
            offspring_count: 45,
            offspring_viable: 42,
            fitness_threshold: nil,
            elapsed_secs: 12.5,
            baseline_ce: nil,
            delta_baseline: nil,
            delta_previous: -0.001,
            patience_counter: nil,
            patience_max: nil,
            candidates_total: 50,
            created_at: "2026-01-31T18:45:23Z"
        )
    }

    DualAxisChartView(iterations: sampleIterations)
        .frame(height: 200)
        .padding()
}
