import { writable, derived, get } from 'svelte/store';
import { browser } from '$app/environment';
import type {
  Experiment, Phase, Iteration, WsMessage, HealthCheck, Flow, Checkpoint, PhaseSummary,
  ExperimentV2, PhaseV2, IterationV2, WsMessageV2, DashboardSnapshotV2
} from './types';

// Current experiment being viewed
export const currentExperiment = writable<Experiment | null>(null);

// All phases for current experiment
export const phases = writable<Phase[]>([]);

// Current phase being viewed
export const currentPhase = writable<Phase | null>(null);

// Iterations for current phase (keeps last 500)
export const iterations = writable<Iteration[]>([]);

// All iterations across all phases (for history)
export const allIterations = writable<Iteration[]>([]);

// WebSocket connection
export const wsConnected = writable(false);

// V2 mode toggle (persisted in localStorage, defaults to V2)
function createV2ModeStore() {
  const storedMode = browser ? localStorage.getItem('wnn-mode') : null;
  // Default to V2 mode if no preference is stored
  const defaultToV2 = storedMode === null || storedMode === 'v2';
  const { subscribe, set, update } = writable(defaultToV2);

  return {
    subscribe,
    set: (value: boolean) => {
      if (browser) {
        localStorage.setItem('wnn-mode', value ? 'v2' : 'v1');
      }
      set(value);
    },
    toggle: () => {
      update(v => {
        const newValue = !v;
        if (browser) {
          localStorage.setItem('wnn-mode', newValue ? 'v2' : 'v1');
        }
        return newValue;
      });
    }
  };
}
export const useV2Mode = createV2ModeStore();

// Latest CE values for live chart (rolling window)
export const ceHistory = writable<{ iter: number; ce: number; acc: number | null }[]>([]);

// Latest health check
export const latestHealthCheck = writable<HealthCheck | null>(null);

// Flow and checkpoint stores
export const flows = writable<Flow[]>([]);
export const currentFlow = writable<Flow | null>(null);
export const checkpoints = writable<Checkpoint[]>([]);

// Final phase comparison summary
export const phaseSummary = writable<PhaseSummary | null>(null);

// Best metrics seen so far
export const bestMetrics = writable({
  bestCE: Infinity,
  bestCEAcc: 0,
  bestAcc: 0,
  bestAccCE: Infinity,
  baseline: 10.5801, // Default baseline
});

// Derived: current phase progress (0-100)
export const phaseProgress = derived(
  [currentPhase, iterations],
  ([$phase, $iters]) => {
    if (!$phase || $iters.length === 0) return 0;
    const latest = $iters[$iters.length - 1];
    return (latest.iteration_num / 250) * 100;
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

// WebSocket manager
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
      // Debug: log iteration data to help diagnose chart issues
      if (msg.type === 'IterationUpdate') {
        const iter = msg.data as any;
        console.log(`[Iter ${iter.iteration_num}] CE=${iter.best_ce?.toFixed(4)}, Acc=${iter.best_accuracy?.toFixed(4) ?? 'null'}%`);
      }
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
      // Initial state from server - load all historical data
      const snapshot = msg.data;
      console.log('Received snapshot:', snapshot.phases?.length, 'phases', snapshot.iterations?.length, 'iterations');

      phases.set(snapshot.phases || []);
      currentPhase.set(snapshot.current_phase || null);
      iterations.set(snapshot.iterations || []);

      // Build CE history from iterations (all data)
      const history = (snapshot.iterations || []).map((iter: any) => ({
        iter: iter.iteration_num,
        ce: iter.best_ce,
        acc: iter.best_accuracy
      }));

      // Debug: log accuracy distribution to diagnose chart issues
      const accValues = history.map((h: any) => h.acc).filter((a: any) => a !== null && a !== undefined);
      const nullCount = history.length - accValues.length;
      const maxAcc = accValues.length > 0 ? Math.max(...accValues) : null;
      const minAcc = accValues.length > 0 ? Math.min(...accValues) : null;
      console.log(`[Snapshot ceHistory] total=${history.length}, with_acc=${accValues.length}, null_acc=${nullCount}, acc_range=[${minAcc?.toFixed(4) ?? 'null'}, ${maxAcc?.toFixed(4) ?? 'null'}]`);

      ceHistory.set(history);

      // Set best metrics from snapshot
      if (snapshot.best_ce && snapshot.best_ce !== Infinity) {
        bestMetrics.set({
          bestCE: snapshot.best_ce,
          bestCEAcc: snapshot.best_ce_acc || 0,
          bestAcc: snapshot.best_acc || 0,
          bestAccCE: snapshot.best_acc_ce || Infinity,
          baseline: 10.5801,
        });
      }
      break;
    }

    case 'IterationUpdate': {
      const iter = msg.data;

      // Add to current phase iterations (keep last 500)
      iterations.update((iters) => {
        const updated = [...iters, iter];
        return updated.slice(-500);
      });

      // Add to all iterations history
      allIterations.update((all) => [...all, iter]);

      // Update CE history for chart (keep all for current phase)
      ceHistory.update((history) => {
        return [...history, {
          iter: iter.iteration_num,
          ce: iter.best_ce,
          acc: iter.best_accuracy
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
      // Note: Clearing iterations/ceHistory means chart only shows current phase data
      // bestMetrics persists across phases (stores all-time best)
      iterations.set([]); // Clear iterations for new phase
      ceHistory.set([]); // Clear chart for new phase
      console.log('Phase started:', phase.name, '- ceHistory and iterations cleared (bestMetrics preserved)');
      break;
    }

    case 'PhaseCompleted': {
      const { phase } = msg.data;
      phases.update((p) =>
        p.map((existing) =>
          existing.id === phase.id ? { ...existing, status: 'completed', ended_at: phase.ended_at } : existing
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

    case 'PhaseSummary': {
      phaseSummary.set(msg.data);
      console.log('Phase summary received:', msg.data.rows.length, 'rows');
      break;
    }

    case 'ExperimentCompleted': {
      currentExperiment.set(msg.data);
      console.log('Experiment completed');
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
      if (flow.id === (currentFlow as any)?.$?.id) {
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
  allIterations.set([]);
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
  phaseSummary.set(null);
}

// =============================================================================
// V2 Stores: Database as source of truth (no log parsing)
// =============================================================================

// V2 stores
export const currentExperimentV2 = writable<ExperimentV2 | null>(null);
export const phasesV2 = writable<PhaseV2[]>([]);
export const currentPhaseV2 = writable<PhaseV2 | null>(null);
export const iterationsV2 = writable<IterationV2[]>([]);
export const wsConnectedV2 = writable(false);

// V2 CE history for charts (includes avg_accuracy)
export const ceHistoryV2 = writable<{
  iter: number;
  ce: number;
  acc: number | null;
  avgCe: number | null;
  avgAcc: number | null;
}[]>([]);

// V2 Best metrics
export const bestMetricsV2 = writable({
  bestCE: Infinity,
  bestCEAcc: 0,
  bestAcc: 0,
  bestAccCE: Infinity,
  baseline: 10.5801,
});

// V2 Derived: current phase progress
export const phaseProgressV2 = derived(
  [currentPhaseV2, iterationsV2],
  ([$phase, $iters]) => {
    if (!$phase || $iters.length === 0) return 0;
    const latest = $iters[$iters.length - 1];
    return (latest.iteration_num / $phase.max_iterations) * 100;
  }
);

// V2 Derived: current iteration number
export const currentIterationV2 = derived(
  iterationsV2,
  ($iters) => $iters.length > 0 ? $iters[$iters.length - 1].iteration_num : 0
);

// V2 Derived: improvement from baseline
export const improvementV2 = derived(
  bestMetricsV2,
  ($best) => (($best.baseline - $best.bestCE) / $best.baseline) * 100
);

// V2 WebSocket manager
let wsV2: WebSocket | null = null;
let reconnectTimeoutV2: ReturnType<typeof setTimeout> | null = null;

export function connectWebSocketV2() {
  // Clear any pending reconnect
  if (reconnectTimeoutV2) {
    clearTimeout(reconnectTimeoutV2);
    reconnectTimeoutV2 = null;
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/v2`;

  console.log('[V2] Connecting to WebSocket:', wsUrl);
  wsV2 = new WebSocket(wsUrl);

  wsV2.onopen = () => {
    wsConnectedV2.set(true);
    console.log('[V2] WebSocket connected');
  };

  wsV2.onclose = () => {
    wsConnectedV2.set(false);
    console.log('[V2] WebSocket disconnected, reconnecting in 3s...');
    reconnectTimeoutV2 = setTimeout(connectWebSocketV2, 3000);
  };

  wsV2.onerror = (error) => {
    console.error('[V2] WebSocket error:', error);
  };

  wsV2.onmessage = (event) => {
    try {
      const msg: WsMessageV2 = JSON.parse(event.data);
      handleWsMessageV2(msg);
    } catch (e) {
      console.error('[V2] Failed to parse WebSocket message:', e);
    }
  };
}

function handleWsMessageV2(msg: WsMessageV2) {
  console.log('[V2] WS message:', msg.type);

  switch (msg.type) {
    case 'Snapshot': {
      const snapshot = msg.data;
      console.log('[V2] Received snapshot:', snapshot.phases?.length, 'phases', snapshot.iterations?.length, 'iterations');

      currentExperimentV2.set(snapshot.current_experiment);
      phasesV2.set(snapshot.phases || []);

      // If current_phase is null but we have phases, use the most recent one
      let currentPhase = snapshot.current_phase;
      if (!currentPhase && snapshot.phases && snapshot.phases.length > 0) {
        // Find running phase, or fall back to most recent by sequence_order
        currentPhase = snapshot.phases.find((p: PhaseV2) => p.status === 'running')
          || snapshot.phases.reduce((a: PhaseV2, b: PhaseV2) => a.sequence_order > b.sequence_order ? a : b);
        console.log('[V2] No current_phase in snapshot, using fallback:', currentPhase?.name);
      }
      currentPhaseV2.set(currentPhase || null);
      iterationsV2.set(snapshot.iterations || []);

      // Build CE history from iterations
      const history = (snapshot.iterations || []).map((iter: IterationV2) => ({
        iter: iter.iteration_num,
        ce: iter.best_ce,
        acc: iter.best_accuracy,
        avgCe: iter.avg_ce,
        avgAcc: iter.avg_accuracy
      }));
      ceHistoryV2.set(history);

      // Compute best metrics from iterations for accurate tracking
      // - bestCE: minimum CE seen
      // - bestCEAcc: accuracy at the iteration with best CE
      // - bestAcc: maximum accuracy seen
      // - bestAccCE: CE at the iteration with best accuracy
      let bestCE = snapshot.best_ce || Infinity;
      let bestCEAcc = 0;
      let bestAcc = 0;
      let bestAccCE = Infinity;

      const iterations = snapshot.iterations || [];
      for (const iter of iterations) {
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
      if (iterations.length === 0) {
        bestCE = snapshot.best_ce || Infinity;
        bestCEAcc = snapshot.best_accuracy || 0;
        bestAcc = snapshot.best_accuracy || 0;
        bestAccCE = snapshot.best_ce || Infinity;
      }

      bestMetricsV2.set({
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
      console.log(`[V2] [Iter ${iter.iteration_num}] CE=${iter.best_ce?.toFixed(4)}, Acc=${iter.best_accuracy?.toFixed(4) ?? 'null'}%, AvgCE=${iter.avg_ce?.toFixed(4) ?? 'null'}, AvgAcc=${iter.avg_accuracy?.toFixed(4) ?? 'null'}%`);

      // Add to iterations (keep last 500, deduplicate by id)
      iterationsV2.update((iters) => {
        // Check if iteration already exists (by id)
        if (iters.some(i => i.id === iter.id)) {
          return iters; // Already exists, don't add duplicate
        }
        const updated = [...iters, iter];
        return updated.slice(-500);
      });

      // Update CE history for chart
      ceHistoryV2.update((history) => {
        return [...history, {
          iter: iter.iteration_num,
          ce: iter.best_ce,
          acc: iter.best_accuracy,
          avgCe: iter.avg_ce,
          avgAcc: iter.avg_accuracy
        }];
      });

      // Update best metrics
      bestMetricsV2.update((best) => {
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
      phasesV2.update((p) => [...p, phase]);
      currentPhaseV2.set(phase);
      iterationsV2.set([]); // Clear iterations for new phase
      ceHistoryV2.set([]); // Clear chart for new phase
      console.log('[V2] Phase started:', phase.name);
      break;
    }

    case 'PhaseCompleted': {
      const phase = msg.data;
      phasesV2.update((p) =>
        p.map((existing) =>
          existing.id === phase.id ? phase : existing
        )
      );
      console.log('[V2] Phase completed:', phase.name);
      break;
    }

    case 'ExperimentStatusChanged': {
      const exp = msg.data;
      currentExperimentV2.set(exp);
      console.log('[V2] Experiment status changed:', exp.status);
      break;
    }

    // Flow status updates (shared with V1)
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
      console.log('[V2] Flow started:', flow.name);
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
      console.log('[V2] Flow queued:', flow.name);
      break;
    }

    case 'FlowCompleted': {
      const flow = msg.data;
      flows.update((f) =>
        f.map((x) => (x.id === flow.id ? flow : x))
      );
      console.log('[V2] Flow completed:', flow.name);
      break;
    }

    case 'FlowFailed': {
      const { flow, error } = msg.data;
      flows.update((f) =>
        f.map((x) => (x.id === flow.id ? flow : x))
      );
      console.error('[V2] Flow failed:', flow.name, error);
      break;
    }
  }
}

export function disconnectWebSocketV2() {
  if (reconnectTimeoutV2) {
    clearTimeout(reconnectTimeoutV2);
    reconnectTimeoutV2 = null;
  }
  wsV2?.close();
  wsV2 = null;
}

// Reset V2 stores
export function resetStoresV2() {
  currentExperimentV2.set(null);
  phasesV2.set([]);
  currentPhaseV2.set(null);
  iterationsV2.set([]);
  ceHistoryV2.set([]);
  bestMetricsV2.set({
    bestCE: Infinity,
    bestCEAcc: 0,
    bestAcc: 0,
    bestAccCE: Infinity,
    baseline: 10.5801,
  });
}
