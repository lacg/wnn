// PhaseTimelineView - Horizontal scrolling phase cards

import SwiftUI

struct PhaseTimelineView: View {
    let phases: [Phase]
    let currentPhase: Phase?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Phases")
                .font(.headline)
                .padding(.horizontal)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(sortedPhases) { phase in
                        PhaseCard(
                            phase: phase,
                            isCurrent: phase.id == currentPhase?.id
                        )
                    }
                }
                .padding(.horizontal)
            }
        }
    }

    private var sortedPhases: [Phase] {
        phases.sorted { $0.sequence_order < $1.sequence_order }
    }
}

struct PhaseCard: View {
    let phase: Phase
    let isCurrent: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header
            HStack {
                Text(phase.shortName)
                    .font(.headline)
                    .fontWeight(.semibold)

                Spacer()

                statusIcon
            }

            // Type badge
            Text(phase.phase_type)
                .font(.caption2)
                .foregroundColor(.secondary)

            Spacer()

            // Metrics
            if let ce = phase.best_ce {
                HStack {
                    Text("CE:")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    Text(NumberFormatters.formatCE(ce))
                        .font(.caption)
                        .fontDesign(.monospaced)
                }
            }

            if let acc = phase.best_accuracy {
                HStack {
                    Text("Acc:")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    Text(NumberFormatters.formatAccuracy(acc))
                        .font(.caption)
                        .fontDesign(.monospaced)
                }
            }

            // Progress
            if phase.status == .running {
                ProgressView(value: phase.progress)
                    .progressViewStyle(.linear)

                Text("\(phase.current_iteration)/\(phase.max_iterations)")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
        .frame(width: 120, height: 140)
        .padding()
        .background(isCurrent ? Color.blue.opacity(0.1) : Theme.cardBackground)
        .cornerRadius(12)
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(isCurrent ? Color.blue : Color.clear, lineWidth: 2)
        )
    }

    private var statusIcon: some View {
        Group {
            switch phase.status {
            case .completed:
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
            case .running:
                ProgressView()
                    .scaleEffect(0.7)
            case .failed:
                Image(systemName: "xmark.circle.fill")
                    .foregroundColor(.red)
            case .skipped:
                Image(systemName: "forward.fill")
                    .foregroundColor(.gray)
            case .pending:
                Image(systemName: "circle")
                    .foregroundColor(.gray)
            case .paused:
                Image(systemName: "pause.circle.fill")
                    .foregroundColor(.yellow)
            case .cancelled:
                Image(systemName: "xmark.circle")
                    .foregroundColor(.gray)
            }
        }
        .font(.caption)
    }
}

#Preview {
    let samplePhases = [
        Phase(id: 1, experiment_id: 1, name: "Phase 1a - GA Neurons", phase_type: "ga_neurons",
              sequence_order: 1, status: .completed, max_iterations: 100, population_size: 50,
              current_iteration: 100, best_ce: 10.5, best_accuracy: 0.012,
              created_at: "2026-01-31T18:45:23Z", started_at: "2026-01-31T18:45:23Z",
              ended_at: "2026-01-31T19:45:23Z"),
        Phase(id: 2, experiment_id: 1, name: "Phase 1b - TS Neurons", phase_type: "ts_neurons",
              sequence_order: 2, status: .running, max_iterations: 200, population_size: 50,
              current_iteration: 65, best_ce: 10.3, best_accuracy: 0.015,
              created_at: "2026-01-31T19:45:23Z", started_at: "2026-01-31T19:45:23Z",
              ended_at: nil),
        Phase(id: 3, experiment_id: 1, name: "Phase 2 - GA Bits", phase_type: "ga_bits",
              sequence_order: 3, status: .pending, max_iterations: 100, population_size: 50,
              current_iteration: 0, best_ce: nil, best_accuracy: nil,
              created_at: "2026-01-31T18:45:23Z", started_at: nil, ended_at: nil),
    ]

    PhaseTimelineView(phases: samplePhases, currentPhase: samplePhases[1])
}
