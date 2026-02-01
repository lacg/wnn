// IterationDetailSheet - Modal with iteration details and genomes

import SwiftUI

struct IterationDetailSheet: View {
    let iteration: Iteration
    let genomes: [GenomeEvaluation]

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Summary section
                    summarySection

                    // Stats section
                    statsSection

                    // Genomes table
                    if !genomes.isEmpty {
                        genomesSection
                    }
                }
                .padding()
            }
            .navigationTitle("Iteration #\(iteration.iteration_num)")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    // MARK: - Sections

    private var summarySection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Summary")
                .font(.headline)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                StatCard(label: "Best CE", value: NumberFormatters.formatCE(iteration.best_ce))

                if let acc = iteration.best_accuracy {
                    StatCard(label: "Best Accuracy", value: NumberFormatters.formatAccuracy(acc))
                }

                if let avgCE = iteration.avg_ce {
                    StatCard(label: "Avg CE", value: NumberFormatters.formatCE(avgCE))
                }

                if let avgAcc = iteration.avg_accuracy {
                    StatCard(label: "Avg Accuracy", value: NumberFormatters.formatAccuracy(avgAcc))
                }
            }
        }
    }

    private var statsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Statistics")
                .font(.headline)

            VStack(spacing: 8) {
                if let elite = iteration.elite_count {
                    StatRow(label: "Elite Count", value: "\(elite)")
                }

                if let offspring = iteration.offspring_count {
                    StatRow(label: "Offspring", value: "\(offspring)")
                }

                if let viable = iteration.offspring_viable {
                    StatRow(label: "Viable Offspring", value: "\(viable)")
                }

                if let candidates = iteration.candidates_total {
                    StatRow(label: "Total Candidates", value: "\(candidates)")
                }

                if let elapsed = iteration.elapsed_secs {
                    StatRow(label: "Elapsed Time", value: DateFormatters.duration(elapsed))
                }

                if let patience = iteration.patienceStatus {
                    StatRow(label: "Patience", value: patience)
                }

                if let delta = iteration.delta_previous {
                    StatRow(label: "Delta (previous)", value: NumberFormatters.formatDelta(delta), color: Theme.deltaColor(delta))
                }

                if let delta = iteration.delta_baseline {
                    StatRow(label: "Delta (baseline)", value: NumberFormatters.formatDelta(delta), color: Theme.deltaColor(delta))
                }
            }
            .padding()
            .background(Theme.cardBackground)
            .cornerRadius(12)
        }
    }

    private var genomesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Genomes")
                    .font(.headline)

                Spacer()

                Text("\(genomes.count) evaluated")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            // Table header
            HStack(spacing: 8) {
                Text("#")
                    .frame(width: 30, alignment: .leading)
                Text("Role")
                    .frame(width: 60, alignment: .leading)
                Text("CE")
                    .frame(width: 70, alignment: .trailing)
                Text("Acc")
                    .frame(width: 50, alignment: .trailing)
                Text("Fitness")
                    .frame(width: 60, alignment: .trailing)
            }
            .font(.caption2)
            .fontWeight(.medium)
            .foregroundColor(.secondary)
            .padding(.horizontal)

            Divider()

            // Genome rows
            ForEach(genomes) { genome in
                GenomeRow(genome: genome)
            }
        }
    }
}

// MARK: - Supporting Views

private struct StatCard: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)

            Text(value)
                .font(.subheadline)
                .fontWeight(.semibold)
                .fontDesign(.monospaced)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(8)
    }
}

private struct StatRow: View {
    let label: String
    let value: String
    var color: Color = .primary

    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)

            Spacer()

            Text(value)
                .fontDesign(.monospaced)
                .foregroundColor(color)
        }
        .font(.subheadline)
    }
}

private struct GenomeRow: View {
    let genome: GenomeEvaluation

    var body: some View {
        HStack(spacing: 8) {
            Text("\(genome.position)")
                .frame(width: 30, alignment: .leading)

            RoleBadge(role: genome.role)
                .frame(width: 60, alignment: .leading)

            Text(NumberFormatters.formatCE(genome.ce))
                .frame(width: 70, alignment: .trailing)

            Text(NumberFormatters.formatAccuracy(genome.accuracy))
                .frame(width: 50, alignment: .trailing)

            if let fitness = genome.fitness_score {
                Text(String(format: "%.2f", fitness))
                    .frame(width: 60, alignment: .trailing)
            } else {
                Text("-")
                    .frame(width: 60, alignment: .trailing)
                    .foregroundColor(.secondary)
            }
        }
        .font(.caption)
        .fontDesign(.monospaced)
        .padding(.horizontal)
        .padding(.vertical, 4)
    }
}

private struct RoleBadge: View {
    let role: GenomeRole

    var body: some View {
        Text(role.displayName)
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Theme.roleColor(role).opacity(0.2))
            .foregroundColor(Theme.roleColor(role))
            .cornerRadius(4)
    }
}

#Preview {
    let sampleIteration = Iteration(
        id: 1,
        phase_id: 1,
        iteration_num: 42,
        best_ce: 10.5432,
        best_accuracy: 0.0123,
        avg_ce: 11.234,
        avg_accuracy: 0.0098,
        elite_count: 5,
        offspring_count: 45,
        offspring_viable: 42,
        fitness_threshold: nil,
        elapsed_secs: 12.5,
        baseline_ce: 12.0,
        delta_baseline: -1.4568,
        delta_previous: -0.0234,
        patience_counter: 3,
        patience_max: 10,
        candidates_total: 50,
        created_at: "2026-01-31T18:45:23Z"
    )

    let sampleGenomes = (1...10).map { i in
        GenomeEvaluation(
            id: Int64(i),
            iteration_id: 1,
            genome_id: Int64(i),
            position: Int32(i),
            role: i <= 5 ? .elite : .offspring,
            elite_rank: i <= 5 ? Int32(i) : nil,
            ce: 10.5 + Double(i) * 0.1,
            accuracy: 0.012 - Double(i) * 0.001,
            fitness_score: Double(100 - i * 5),
            eval_time_ms: Int32(100 + i * 10),
            created_at: "2026-01-31T18:45:23Z"
        )
    }

    IterationDetailSheet(iteration: sampleIteration, genomes: sampleGenomes)
}
