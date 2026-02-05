import { writable, derived, get } from 'svelte/store';
import type {
  Experiment, Phase, Iteration, WsMessage, HealthCheck, Flow, Checkpoint, DashboardSnapshot
} from './types';

// =============================================================================
// Core stores
// =============================================================================

// Current experiment being viewed
export const currentExperiment = writable<Experiment | null>(null);

// All phases for current experiment
export const phases = writable<Phase[]>([]);

// Current phase being viewed
export const currentPhase = writable<Phase | null>(null);

// Iterations for current phase (keeps last 500)
export const iterations = writable<Iteration[]>([]);

// WebSocket connection
export const wsConnected = writable(false);

// CE history for charts (includes avg metrics)
export const ceHistory = writable<{
  iter: number;
  ce: number;
  acc: number | null;
  avgCe: number | null;
  avgAcc: number | null;
}[]>([]);

// Latest health check
export const latestHealthCheck = writable<HealthCheck | null>(null);

// Flow and checkpoint stores
export const flows = writable<Flow[]>([]);
export const currentFlow = writable<Flow | null>(null);
export const checkpoints = writable<Checkpoint[]>([]);

// Gating status updates (for real-time notifications)
export const gatingStatusUpdates = writable<{ experiment_id: number; status: string } | null>(null);

// Best metrics seen so far
export const bestMetrics = writable({
  bestCE: Infinity,
  bestCEAcc: 0,
  bestAcc: 0,
  bestAccCE: Infinity,
  baseline: 10.5801, // Default baseline
});

// =============================================================================
// Derived stores
// =============================================================================

// Derived: current phase progress (0-100)
export const phaseProgress = derived(
  [currentPhase, iterations],
  ([$phase, $iters]) => {
    if (!$phase || $iters.length === 0) return 0;
    const latest = $iters[$iters.length - 1];
    return (latest.iteration_num / $phase.max_iterations) * 100;
  }
);

// Derived: current iteration number
export const currentIteration = derived(
  iterations,
  ($iters) => $iters.length > 0 ? $iters[$iters.length - 1].iteration_num : 0
);

// Derived: improvement from baseline
export const improvement = derived(
  bestMetrics,
  ($best) => (($best.baseline - $best.bestCE) / $best.baseline) * 100
);

// =============================================================================
// WebSocket manager
// =============================================================================

let ws: WebSocket | null = null;
let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;

export function connectWebSocket() {
  // Clear any pending reconnect
  if (reconnectTimeout) {
    clearTimeout(reconnectTimeout);
    reconnectTimeout = null;
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  console.log('Connecting to WebSocket:', wsUrl);
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    wsConnected.set(true);
    console.log('WebSocket connected');
  };

  ws.onclose = () => {
    wsConnected.set(false);
    console.log('WebSocket disconnected, reconnecting in 3s...');
    reconnectTimeout = setTimeout(connectWebSocket, 3000);
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };

  ws.onmessage = (event) => {
    try {
      const msg: WsMessage = JSON.parse(event.data);
      handleWsMessage(msg);
    } catch (e) {
      console.error('Failed to parse WebSocket message:', e);
    }
  };
}

function handleWsMessage(msg: WsMessage) {
  console.log('WS message:', msg.type);

  switch (msg.type) {
    case 'Snapshot': {
      const snapshot = msg.data;
      console.log('Received snapshot:', snapshot.phases?.length, 'phases', snapshot.iterations?.length, 'iterations');

      currentExperiment.set(snapshot.current_experiment);
      phases.set(snapshot.phases || []);

      // If current_phase is null but we have phases, use the most recent one
      let snapshotPhase = snapshot.current_phase;
      if (!snapshotPhase && snapshot.phases && snapshot.phases.length > 0) {
        // Find running phase, or fall back to most recent by sequence_order
        snapshotPhase = snapshot.phases.find((p: Phase) => p.status === 'running')
          || snapshot.phases.reduce((a: Phase, b: Phase) => a.sequence_order > b.sequence_order ? a : b);
        console.log('No current_phase in snapshot, using fallback:', snapshotPhase?.name);
      }
      currentPhase.set(snapshotPhase || null);

      // Filter iterations to only show those from the current phase
      // This prevents mixing iterations from different phases
      const allIterations = snapshot.iterations || [];
      const currentPhaseId = snapshotPhase?.id;
      const filteredIterations = currentPhaseId
        ? allIterations.filter((iter: Iteration) => iter.phase_id === currentPhaseId)
        : allIterations;
      iterations.set(filteredIterations);

      // Build CE history from iterations (filter to current phase only)
      // Reuse currentPhaseId and filteredIterations from above
      const history = filteredIterations.map((iter: Iteration) => ({
        iter: iter.iteration_num,
        ce: iter.best_ce,
        acc: iter.best_accuracy,
        avgCe: iter.avg_ce,
        avgAcc: iter.avg_accuracy
      }));
      ceHistory.set(history);

      // Compute best metrics from iterations for accurate tracking
      let bestCE = snapshot.best_ce || Infinity;
      let bestCEAcc = 0;
      let bestAcc = 0;
      let bestAccCE = Infinity;

      const snapshotIterations = snapshot.iterations || [];
      for (const iter of snapshotIterations) {
        if (iter.best_ce < bestCE) {
          bestCE = iter.best_ce;
          bestCEAcc = iter.best_accuracy || 0;
        }
        if (iter.best_accuracy && iter.best_accuracy > bestAcc) {
          bestAcc = iter.best_accuracy;
          bestAccCE = iter.best_ce;
        }
      }

      // Fall back to snapshot values if no iterations
      if (snapshotIterations.length === 0) {
        bestCE = snapshot.best_ce || Infinity;
        bestCEAcc = snapshot.best_accuracy || 0;
        bestAcc = snapshot.best_accuracy || 0;
        bestAccCE = snapshot.best_ce || Infinity;
      }

      bestMetrics.set({
        bestCE,
        bestCEAcc,
        bestAcc,
        bestAccCE,
        baseline: 10.5801,
      });
      break;
    }

    case 'IterationCompleted': {
      const iter = msg.data;
      console.log(`[Iter ${iter.iteration_num}] CE=${iter.best_ce?.toFixed(4)}, Acc=${iter.best_accuracy?.toFixed(4) ?? 'null'}%, AvgCE=${iter.avg_ce?.toFixed(4) ?? 'null'}, AvgAcc=${iter.avg_accuracy?.toFixed(4) ?? 'null'}%`);

      // Only add iterations from the current phase to prevent mixing
      const currentPhaseValue = get(currentPhase);
      if (currentPhaseValue && iter.phase_id !== currentPhaseValue.id) {
        console.log(`[Iter ${iter.iteration_num}] Skipping - different phase (${iter.phase_id} vs ${currentPhaseValue.id})`);
        break;
      }

      // Add to iterations (keep last 500, deduplicate by id)
      iterations.update((iters) => {
        // Check if iteration already exists (by id)
        if (iters.some(i => i.id === iter.id)) {
          return iters; // Already exists, don't add duplicate
        }
        const updated = [...iters, iter];
        return updated.slice(-500);
      });

      // Update CE history for chart
      ceHistory.update((history) => {
        return [...history, {
          iter: iter.iteration_num,
          ce: iter.best_ce,
          acc: iter.best_accuracy,
          avgCe: iter.avg_ce,
          avgAcc: iter.avg_accuracy
        }];
      });

      // Update best metrics
      bestMetrics.update((best) => {
        const newBest = { ...best };
        if (iter.best_ce < best.bestCE) {
          newBest.bestCE = iter.best_ce;
          newBest.bestCEAcc = iter.best_accuracy || 0;
        }
        if (iter.best_accuracy && iter.best_accuracy > best.bestAcc) {
          newBest.bestAcc = iter.best_accuracy;
          newBest.bestAccCE = iter.best_ce;
        }
        return newBest;
      });
      break;
    }

    case 'PhaseStarted': {
      const phase = msg.data;
      phases.update((p) => [...p, phase]);
      currentPhase.set(phase);
      iterations.set([]); // Clear iterations for new phase
      ceHistory.set([]); // Clear chart for new phase
      console.log('Phase started:', phase.name);
      break;
    }

    case 'PhaseCompleted': {
      const phase = msg.data;
      phases.update((p) =>
        p.map((existing) =>
          existing.id === phase.id ? phase : existing
        )
      );
      console.log('Phase completed:', phase.name);
      break;
    }

    case 'HealthCheck': {
      latestHealthCheck.set(msg.data);
      console.log('Health check:', msg.data);
      break;
    }

    case 'ExperimentStatusChanged': {
      const exp = msg.data;
      currentExperiment.set(exp);
      console.log('Experiment status changed:', exp.status);
      break;
    }

    case 'FlowStarted': {
      const flow = msg.data;
      flows.update((f) => {
        const existing = f.find((x) => x.id === flow.id);
        if (existing) {
          return f.map((x) => (x.id === flow.id ? flow : x));
        }
        return [...f, flow];
      });
      currentFlow.set(flow);
      console.log('Flow started:', flow.name);
      break;
    }

    case 'FlowCompleted': {
      const flow = msg.data;
      flows.update((f) =>
        f.map((x) => (x.id === flow.id ? flow : x))
      );
      if (flow.id === get(currentFlow)?.id) {
        currentFlow.set(flow);
      }
      console.log('Flow completed:', flow.name);
      break;
    }

    case 'FlowFailed': {
      const { flow, error } = msg.data;
      flows.update((f) =>
        f.map((x) => (x.id === flow.id ? flow : x))
      );
      console.error('Flow failed:', flow.name, error);
      break;
    }

    case 'FlowQueued': {
      const flow = msg.data;
      flows.update((f) => {
        const existing = f.find((x) => x.id === flow.id);
        if (existing) {
          return f.map((x) => (x.id === flow.id ? flow : x));
        }
        return [...f, flow];
      });
      currentFlow.set(flow);
      console.log('Flow queued:', flow.name);
      break;
    }

    case 'FlowCancelled': {
      const flow = msg.data;
      flows.update((f) =>
        f.map((x) => (x.id === flow.id ? flow : x))
      );
      if (get(currentFlow)?.id === flow.id) {
        currentFlow.set(flow);
      }
      console.log('Flow cancelled:', flow.name);
      break;
    }

    case 'CheckpointCreated': {
      const checkpoint = msg.data;
      checkpoints.update((c) => [...c, checkpoint]);
      console.log('Checkpoint created:', checkpoint.name);
      break;
    }

    case 'CheckpointDeleted': {
      const { id } = msg.data;
      checkpoints.update((c) => c.filter((x) => x.id !== id));
      console.log('Checkpoint deleted:', id);
      break;
    }

    case 'GatingStatusChanged': {
      const { experiment_id, status } = msg.data;
      gatingStatusUpdates.set({ experiment_id, status });
      console.log(`Gating status changed for experiment ${experiment_id}:`, status);
      break;
    }
  }
}

export function disconnectWebSocket() {
  if (reconnectTimeout) {
    clearTimeout(reconnectTimeout);
    reconnectTimeout = null;
  }
  ws?.close();
  ws = null;
}

// Reset all stores (for new experiment)
export function resetStores() {
  currentExperiment.set(null);
  phases.set([]);
  currentPhase.set(null);
  iterations.set([]);
  ceHistory.set([]);
  latestHealthCheck.set(null);
  bestMetrics.set({
    bestCE: Infinity,
    bestCEAcc: 0,
    bestAcc: 0,
    bestAccCE: Infinity,
    baseline: 10.5801,
  });
  flows.set([]);
  currentFlow.set(null);
  checkpoints.set([]);
}
