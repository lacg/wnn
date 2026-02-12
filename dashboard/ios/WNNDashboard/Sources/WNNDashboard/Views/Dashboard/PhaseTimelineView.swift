// PhaseTimelineView - Horizontal scrolling phase cards with tap to view history

import SwiftUI

public struct PhaseTimelineView: View {
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    public let phases: [Phase]
    public let currentPhase: Phase?
    public let selectedPhase: Phase?
    public let onPhaseSelected: ((Phase?) -> Void)?

    public init(
        phases: [Phase],
        currentPhase: Phase?,
        selectedPhase: Phase? = nil,
        onPhaseSelected: ((Phase?) -> Void)? = nil
    ) {
        self.phases = phases
        self.currentPhase = currentPhase
        self.selectedPhase = selectedPhase
        self.onPhaseSelected = onPhaseSelected
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Phases").font(.headline)
                Spacer()
                if selectedPhase != nil {
                    Button("Show Live") {
                        onPhaseSelected?(nil)
                    }
                    .font(.caption)
                    .foregroundColor(.blue)
                }
            }
            .padding(.horizontal)

            if horizontalSizeClass == .regular {
                // iPad: Grid layout
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 140))], spacing: 12) {
                    ForEach(phases.sorted { $0.sequence_order < $1.sequence_order }) { phase in
                        phaseCardView(phase: phase)
                    }
                }
                .padding(.horizontal)
            } else {
                // iPhone: Horizontal scroll
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        ForEach(phases.sorted { $0.sequence_order < $1.sequence_order }) { phase in
                            phaseCardView(phase: phase)
                        }
                    }
                    .padding(.horizontal)
                }
            }
        }
    }

    @ViewBuilder
    private func phaseCardView(phase: Phase) -> some View {
        PhaseCard(
            phase: phase,
            isCurrent: phase.id == currentPhase?.id,
            isSelected: phase.id == selectedPhase?.id,
            isSelectable: phase.status == .completed
        )
        .onTapGesture {
            if phase.status == .completed {
                // Toggle selection: tap again to deselect
                if selectedPhase?.id == phase.id {
                    onPhaseSelected?(nil)
                } else {
                    onPhaseSelected?(phase)
                }
            }
        }
    }
}

struct PhaseCard: View {
    let phase: Phase
    let isCurrent: Bool
    let isSelected: Bool
    let isSelectable: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(phase.shortName).font(.headline).fontWeight(.semibold)
                Spacer()
                statusIcon
            }
            Text(phase.phase_type).font(.caption2).foregroundColor(.secondary)

            if isSelected {
                Text("Viewing History")
                    .font(.caption2)
                    .foregroundColor(.orange)
            }

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
        .background(cardBackground)
        .cornerRadius(12)
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(borderColor, lineWidth: isSelected ? 3 : (isCurrent ? 2 : 0)))
        .opacity(isSelectable ? 1.0 : (isCurrent ? 1.0 : 0.6))
    }

    private var cardBackground: Color {
        if isSelected { return Color.orange.opacity(0.15) }
        if isCurrent { return Color.blue.opacity(0.1) }
        return Theme.cardBackground
    }

    private var borderColor: Color {
        if isSelected { return Color.orange }
        if isCurrent { return Color.blue }
        return Color.clear
    }

    @ViewBuilder
    private var statusIcon: some View {
        switch phase.status {
        case .completed: Image(systemName: "checkmark.circle.fill").foregroundColor(.green).font(.caption)
        case .running: ProgressView().scaleEffect(0.7)
        case .queued: Image(systemName: "clock").foregroundColor(.orange).font(.caption)
        case .failed: Image(systemName: "xmark.circle.fill").foregroundColor(.red).font(.caption)
        case .skipped: Image(systemName: "forward.fill").foregroundColor(.gray).font(.caption)
        case .pending: Image(systemName: "circle").foregroundColor(.gray).font(.caption)
        case .paused: Image(systemName: "pause.circle.fill").foregroundColor(.yellow).font(.caption)
        case .cancelled: Image(systemName: "xmark.circle").foregroundColor(.gray).font(.caption)
        }
    }
}
