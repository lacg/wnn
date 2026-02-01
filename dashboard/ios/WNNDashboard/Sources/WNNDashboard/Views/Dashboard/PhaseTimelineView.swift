// PhaseTimelineView - Horizontal scrolling phase cards

import SwiftUI

public struct PhaseTimelineView: View {
    public let phases: [Phase]
    public let currentPhase: Phase?

    public init(phases: [Phase], currentPhase: Phase?) { self.phases = phases; self.currentPhase = currentPhase }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Phases").font(.headline).padding(.horizontal)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(phases.sorted { $0.sequence_order < $1.sequence_order }) { phase in
                        PhaseCard(phase: phase, isCurrent: phase.id == currentPhase?.id)
                    }
                }
                .padding(.horizontal)
            }
        }
    }
}

struct PhaseCard: View {
    let phase: Phase
    let isCurrent: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(phase.shortName).font(.headline).fontWeight(.semibold)
                Spacer()
                statusIcon
            }
            Text(phase.phase_type).font(.caption2).foregroundColor(.secondary)
            Spacer()
            if let ce = phase.best_ce { HStack { Text("CE:").font(.caption2).foregroundColor(.secondary); Text(NumberFormatters.formatCE(ce)).font(.caption).fontDesign(.monospaced) } }
            if let acc = phase.best_accuracy { HStack { Text("Acc:").font(.caption2).foregroundColor(.secondary); Text(NumberFormatters.formatAccuracy(acc)).font(.caption).fontDesign(.monospaced) } }
            if phase.status == .running {
                ProgressView(value: phase.progress).progressViewStyle(.linear)
                Text("\(phase.current_iteration)/\(phase.max_iterations)").font(.caption2).foregroundColor(.secondary)
            }
        }
        .frame(width: 120, height: 140)
        .padding()
        .background(isCurrent ? Color.blue.opacity(0.1) : Theme.cardBackground)
        .cornerRadius(12)
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(isCurrent ? Color.blue : Color.clear, lineWidth: 2))
    }

    @ViewBuilder
    private var statusIcon: some View {
        switch phase.status {
        case .completed: Image(systemName: "checkmark.circle.fill").foregroundColor(.green).font(.caption)
        case .running: ProgressView().scaleEffect(0.7)
        case .failed: Image(systemName: "xmark.circle.fill").foregroundColor(.red).font(.caption)
        case .skipped: Image(systemName: "forward.fill").foregroundColor(.gray).font(.caption)
        case .pending: Image(systemName: "circle").foregroundColor(.gray).font(.caption)
        case .paused: Image(systemName: "pause.circle.fill").foregroundColor(.yellow).font(.caption)
        case .cancelled: Image(systemName: "xmark.circle").foregroundColor(.gray).font(.caption)
        }
    }
}
