// IterationDetailSheet - Modal with iteration details and genomes

import SwiftUI

public struct IterationDetailSheet: View {
    public let iteration: Iteration
    public let genomes: [GenomeEvaluation]

    @Environment(\.dismiss) private var dismiss

    public init(iteration: Iteration, genomes: [GenomeEvaluation]) { self.iteration = iteration; self.genomes = genomes }

    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    summarySection
                    statsSection
                    if !genomes.isEmpty { genomesSection }
                }
                .padding()
            }
            .navigationTitle("Iteration #\(iteration.iteration_num)")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .navigationBarTrailing) { Button("Done") { dismiss() } } }
            #else
            .toolbar { ToolbarItem(placement: .automatic) { Button("Done") { dismiss() } } }
            #endif
        }
    }

    private var summarySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Summary").font(.headline)
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                StatCard(label: "Best CE", value: NumberFormatters.formatCE(iteration.best_ce))
                if let acc = iteration.best_accuracy { StatCard(label: "Best Accuracy", value: NumberFormatters.formatAccuracy(acc)) }
                if let avg = iteration.avg_ce { StatCard(label: "Avg CE", value: NumberFormatters.formatCE(avg)) }
                if let avg = iteration.avg_accuracy { StatCard(label: "Avg Accuracy", value: NumberFormatters.formatAccuracy(avg)) }
            }
        }
    }

    private var statsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Statistics").font(.headline)
            VStack(spacing: 8) {
                if let e = iteration.elite_count { StatRow(label: "Elite Count", value: "\(e)") }
                if let o = iteration.offspring_count { StatRow(label: "Offspring", value: "\(o)") }
                if let v = iteration.offspring_viable { StatRow(label: "Viable", value: "\(v)") }
                if let t = iteration.candidates_total { StatRow(label: "Total", value: "\(t)") }
                if let e = iteration.elapsed_secs { StatRow(label: "Elapsed", value: DateFormatters.duration(e)) }
                if let p = iteration.patienceStatus { StatRow(label: "Patience", value: p) }
                if let d = iteration.delta_previous { StatRow(label: "Delta", value: NumberFormatters.formatDelta(d), color: Theme.deltaColor(d)) }
            }
            .padding()
            .background(Theme.cardBackground)
            .cornerRadius(12)
        }
    }

    private var genomesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack { Text("Genomes").font(.headline); Spacer(); Text("\(genomes.count) evaluated").font(.caption).foregroundColor(.secondary) }
            HStack(spacing: 8) {
                Text("#").frame(width: 30, alignment: .leading)
                Text("Role").frame(width: 60, alignment: .leading)
                Text("CE").frame(width: 70, alignment: .trailing)
                Text("Acc").frame(width: 50, alignment: .trailing)
                Text("Fitness").frame(width: 60, alignment: .trailing)
            }
            .font(.caption2).fontWeight(.medium).foregroundColor(.secondary).padding(.horizontal)
            Divider()
            ForEach(genomes) { g in GenomeRow(genome: g) }
        }
    }
}

private struct StatCard: View {
    let label: String; let value: String
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label).font(.caption).foregroundColor(.secondary)
            Text(value).font(.subheadline).fontWeight(.semibold).fontDesign(.monospaced)
        }
        .frame(maxWidth: .infinity, alignment: .leading).padding().background(Theme.cardBackground).cornerRadius(8)
    }
}

private struct StatRow: View {
    let label: String; let value: String; var color: Color = .primary
    var body: some View { HStack { Text(label).foregroundColor(.secondary); Spacer(); Text(value).fontDesign(.monospaced).foregroundColor(color) }.font(.subheadline) }
}

private struct GenomeRow: View {
    let genome: GenomeEvaluation
    var body: some View {
        HStack(spacing: 8) {
            Text("\(genome.position)").frame(width: 30, alignment: .leading)
            Text(genome.role.displayName).font(.caption2).padding(.horizontal, 6).padding(.vertical, 2).background(Theme.roleColor(genome.role).opacity(0.2)).foregroundColor(Theme.roleColor(genome.role)).cornerRadius(4).frame(width: 60, alignment: .leading)
            Text(NumberFormatters.formatCE(genome.ce)).frame(width: 70, alignment: .trailing)
            Text(NumberFormatters.formatAccuracy(genome.accuracy)).frame(width: 50, alignment: .trailing)
            if let f = genome.fitness_score { Text(String(format: "%.2f", f)).frame(width: 60, alignment: .trailing) }
            else { Text("-").frame(width: 60, alignment: .trailing).foregroundColor(.secondary) }
        }
        .font(.caption).fontDesign(.monospaced).padding(.horizontal).padding(.vertical, 4)
    }
}
