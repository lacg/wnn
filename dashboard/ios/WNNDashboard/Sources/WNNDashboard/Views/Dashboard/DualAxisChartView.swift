// DualAxisChartView - CE vs Accuracy chart with tooltips

import SwiftUI
import Charts

/// Data point for chart display
public struct ChartPoint: Identifiable {
    public let id: Int
    public let iterationNum: Int
    public let ce: Double
    public let acc: Double?
    public let avgCe: Double?
    public let avgAcc: Double?

    // Normalized values for dual-axis display
    public var ceNorm: Double = 0
    public var accNorm: Double = 0
    public var avgCeNorm: Double = 0
    public var avgAccNorm: Double = 0
}

public struct DualAxisChartView: View {
    public let iterations: [Iteration]
    public let title: String
    public let height: CGFloat

    @State private var selectedPoint: ChartPoint?
    @State private var tooltipPosition: CGPoint = .zero

    public init(iterations: [Iteration], title: String = "Best So Far", height: CGFloat = 280) {
        self.iterations = iterations
        self.title = title
        self.height = height
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Title
            Text(title)
                .font(.headline)
                .padding(.horizontal)

            // Chart with tooltip overlay
            ZStack(alignment: .topLeading) {
                chartContent

                // Tooltip overlay
                if let point = selectedPoint {
                    tooltipView(for: point)
                        .offset(x: tooltipPosition.x, y: tooltipPosition.y)
                }
            }
            .frame(height: height)

            // Legend
            legendView
                .padding(.horizontal)
        }
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(12)
    }

    @ViewBuilder
    private var chartContent: some View {
        if #available(iOS 16.0, *) {
            Chart {
                // Best CE line (solid blue)
                ForEach(chartData) { pt in
                    LineMark(
                        x: .value("Iter", pt.iterationNum),
                        y: .value("CE", pt.ceNorm),
                        series: .value("Metric", "CE")
                    )
                    .foregroundStyle(Theme.ceLineColor)
                    .interpolationMethod(.monotone)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                }

                // Best Accuracy line (solid green)
                ForEach(chartData.filter { $0.acc != nil }) { pt in
                    LineMark(
                        x: .value("Iter", pt.iterationNum),
                        y: .value("Acc", pt.accNorm),
                        series: .value("Metric", "Acc")
                    )
                    .foregroundStyle(Theme.accuracyLineColor)
                    .interpolationMethod(.monotone)
                    .lineStyle(StrokeStyle(lineWidth: 2))
                }

                // Avg CE line (dashed blue)
                ForEach(chartData.filter { $0.avgCe != nil }) { pt in
                    LineMark(
                        x: .value("Iter", pt.iterationNum),
                        y: .value("AvgCE", pt.avgCeNorm),
                        series: .value("Metric", "AvgCE")
                    )
                    .foregroundStyle(Theme.ceLineColor.opacity(0.5))
                    .interpolationMethod(.monotone)
                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [5, 3]))
                }

                // Avg Accuracy line (dashed green)
                ForEach(chartData.filter { $0.avgAcc != nil }) { pt in
                    LineMark(
                        x: .value("Iter", pt.iterationNum),
                        y: .value("AvgAcc", pt.avgAccNorm),
                        series: .value("Metric", "AvgAcc")
                    )
                    .foregroundStyle(Theme.accuracyLineColor.opacity(0.5))
                    .interpolationMethod(.monotone)
                    .lineStyle(StrokeStyle(lineWidth: 1.5, dash: [5, 3]))
                }

                // Best CE point marker
                if let bestCEPoint = chartData.min(by: { $0.ce < $1.ce }) {
                    PointMark(
                        x: .value("Iter", bestCEPoint.iterationNum),
                        y: .value("CE", bestCEPoint.ceNorm)
                    )
                    .foregroundStyle(Theme.ceLineColor)
                    .symbolSize(60)
                }
            }
            .chartYScale(domain: 0...1)
            .chartYAxis {
                AxisMarks(position: .leading) { _ in
                    AxisGridLine()
                }
            }
            .chartXAxis {
                AxisMarks { value in
                    AxisValueLabel()
                }
            }
            .chartOverlay { proxy in
                GeometryReader { geo in
                    Rectangle()
                        .fill(Color.clear)
                        .contentShape(Rectangle())
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    handleTouch(at: value.location, in: geo, proxy: proxy)
                                }
                                .onEnded { _ in
                                    selectedPoint = nil
                                }
                        )
                }
            }
        } else {
            Text("Charts require iOS 16+")
                .foregroundColor(.secondary)
                .frame(height: height)
        }
    }

    private func handleTouch(at location: CGPoint, in geo: GeometryProxy, proxy: ChartProxy) {
        guard let plotFrame = proxy.plotFrame else { return }
        let frame = geo[plotFrame]

        // Convert touch location to chart value
        guard let iterValue = proxy.value(atX: location.x - frame.origin.x, as: Int.self) else { return }

        // Find closest data point
        if let closest = chartData.min(by: { abs($0.iterationNum - iterValue) < abs($1.iterationNum - iterValue) }) {
            selectedPoint = closest

            // Calculate tooltip position (offset from touch)
            let xPos = min(max(location.x - 80, 10), geo.size.width - 170)
            let yPos = max(location.y - 120, 10)
            tooltipPosition = CGPoint(x: xPos, y: yPos)
        }
    }

    private func tooltipView(for point: ChartPoint) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Iteration \(point.iterationNum)")
                .font(.caption.bold())

            Divider()

            HStack {
                Text("Best CE:")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Spacer()
                Text(NumberFormatters.formatCE(point.ce))
                    .font(.caption.monospaced().bold())
                    .foregroundColor(Theme.ceLineColor)
            }

            if let acc = point.acc {
                HStack {
                    Text("Best Acc:")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(NumberFormatters.formatAccuracy(acc))
                        .font(.caption.monospaced().bold())
                        .foregroundColor(Theme.accuracyLineColor)
                }
            }

            if let avgCe = point.avgCe {
                HStack {
                    Text("Avg CE:")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(NumberFormatters.formatCE(avgCe))
                        .font(.caption.monospaced())
                        .foregroundColor(Theme.ceLineColor.opacity(0.7))
                }
            }

            if let avgAcc = point.avgAcc {
                HStack {
                    Text("Avg Acc:")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(NumberFormatters.formatAccuracy(avgAcc))
                        .font(.caption.monospaced())
                        .foregroundColor(Theme.accuracyLineColor.opacity(0.7))
                }
            }
        }
        .padding(10)
        .background(Theme.cardBackground)
        .cornerRadius(8)
        .shadow(color: Color.black.opacity(0.2), radius: 4, x: 0, y: 2)
        .frame(width: 160)
    }

    private var legendView: some View {
        HStack(spacing: 16) {
            legendItem(color: Theme.ceLineColor, text: "Best CE", dashed: false)
            legendItem(color: Theme.accuracyLineColor, text: "Best Acc", dashed: false)
            legendItem(color: Theme.ceLineColor.opacity(0.5), text: "Avg CE", dashed: true)
            legendItem(color: Theme.accuracyLineColor.opacity(0.5), text: "Avg Acc", dashed: true)
        }
        .font(.caption2)
    }

    private func legendItem(color: Color, text: String, dashed: Bool) -> some View {
        HStack(spacing: 4) {
            if dashed {
                Rectangle()
                    .stroke(color, style: StrokeStyle(lineWidth: 2, dash: [4, 2]))
                    .frame(width: 16, height: 2)
            } else {
                Rectangle()
                    .fill(color)
                    .frame(width: 16, height: 2)
            }
            Text(text)
                .foregroundColor(.secondary)
        }
    }

    private var chartData: [ChartPoint] {
        let sorted = iterations.sorted { $0.iteration_num < $1.iteration_num }
        guard !sorted.isEmpty else { return [] }

        // Compute cumulative best CE for smooth descending line
        var cumulativeBest: [Double] = []
        var runningBest = Double.infinity
        for iter in sorted {
            runningBest = min(runningBest, iter.best_ce)
            cumulativeBest.append(runningBest)
        }

        // Shared ranges so related metrics are visually comparable
        // Best CE + Avg CE share one scale; Best Acc + Avg Acc share another
        let allCEs = cumulativeBest + sorted.compactMap(\.avg_ce)
        let allAccs = sorted.compactMap(\.best_accuracy) + sorted.compactMap(\.avg_accuracy)

        // CE: dynamic range from data (values are near theoretical max ~10.8, so 0-based would flatten)
        let minCE = allCEs.min() ?? 0
        let maxCE = allCEs.max() ?? 1
        let ceRange = Swift.max(maxCE - minCE, 0.001)

        // Acc: floor at 0 so the chart gives honest perspective (3.34% shows as 3.34% up, not full height)
        let maxAcc = allAccs.max() ?? 1
        let accRange = Swift.max(maxAcc, 0.001)

        return sorted.enumerated().map { index, iter in
            var point = ChartPoint(
                id: index,
                iterationNum: Int(iter.iteration_num),
                ce: cumulativeBest[index],
                acc: iter.best_accuracy,
                avgCe: iter.avg_ce,
                avgAcc: iter.avg_accuracy
            )

            // CE: lower = better = lower on chart (going down is improving)
            // Acc: higher = better = higher on chart (going up is improving)
            // Both best and avg use the SAME shared range so they're visually comparable
            point.ceNorm = (cumulativeBest[index] - minCE) / ceRange
            point.accNorm = iter.best_accuracy.map { $0 / accRange } ?? 0
            point.avgCeNorm = iter.avg_ce.map { ($0 - minCE) / ceRange } ?? 0
            point.avgAccNorm = iter.avg_accuracy.map { $0 / accRange } ?? 0

            return point
        }
    }
}
