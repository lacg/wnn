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

// Gating run updates (for real-time notifications)
import type { GatingRun } from './types';
export const gatingRunUpdates = writable<GatingRun | null>(null);

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
//
// `DashboardWebSocket` encapsulates one auto-reconnecting WebSocket connection
// to the dashboard backend. Each lifecycle (start..stop) owns a private
// shutdown promise that is resolved exactly once — when the WebSocket has
// fully closed in response to `stop()`. This prevents the previously-observed
// zombie-reconnect race, where `ws.close()` would trigger `onclose`
// asynchronously *after* the disconnect call had returned, re-arming a
// reconnect timer that the caller could not cancel.
//
// Re-entrancy:
//   - `start()` while already running: no-op (returns immediately).
//   - `stop()` while already stopping: returns the same in-flight Promise.
//   - `start()` after a completed `stop()`: opens a fresh connection with a
//     fresh shutdown signal.

class DashboardWebSocket {
  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  // Shutdown signal: resolved exactly once when the WS finishes closing
  // following a `stop()` call. The synchronous `isShuttingDown` flag is the
  // source of truth for "should the onclose handler skip reconnect?"; the
  // promise exists so callers can `await stop()` and know the close handshake
  // has fully completed.
  private isShuttingDown = false;
  private shutdownResolve: (() => void) | null = null;
  private shutdownPromise: Promise<void> = Promise.resolve();

  // Backoff state: incremented on each unsuccessful reconnect, reset to 0
  // when the connection successfully opens (onopen). Used by
  // computeBackoffDelay() below to determine the next setTimeout delay.
  private reconnectAttempt = 0;

  /** Open the WebSocket and enable auto-reconnect on unexpected close. */
  start(): void {
    if (this.ws !== null && !this.isShuttingDown) {
      return; // already connected (or connecting)
    }

    // Fresh shutdown signal for this lifecycle.
    this.isShuttingDown = false;
    this.shutdownPromise = new Promise<void>((resolve) => {
      this.shutdownResolve = resolve;
    });
    this.reconnectAttempt = 0;

    this.openSocket();
  }

  /**
   * Compute the delay (in ms) before the next reconnect attempt, given the
   * 0-based attempt counter (0 = first retry after the initial connection
   * was lost). Returns the delay to pass to `setTimeout`.
   *
   * TODO: implement the backoff policy.
   *
   * Considerations:
   *   - Base delay: how fast is the first retry? Common: 500-1000 ms.
   *   - Growth shape: classic doubling (`base * 2 ** attempt`) is standard;
   *     Fibonacci or 1.5x are gentler if you want to retry more before
   *     hitting the cap.
   *   - Maximum cap: how long is the longest delay? 30 s is a good default
   *     for a research dashboard — long enough to back off during a real
   *     outage, short enough that the user doesn't wait forever after the
   *     server is back. Use Math.min(cap, computedDelay).
   *   - Jitter: prevents synchronized reconnects across clients and against
   *     periodic server restarts. ±20% is a typical, harmless amount:
   *     delay * (0.8 + Math.random() * 0.4). Skip if you don't care.
   *
   * Reference example (doubling + 30s cap + ±20% jitter):
   *   const base = 1000;
   *   const max = 30_000;
   *   const raw = Math.min(max, base * 2 ** attempt);
   *   return Math.floor(raw * (0.8 + Math.random() * 0.4));
   */
  private computeBackoffDelay(attempt: number): number {
    // Standard policy: doubling growth, 30 s cap, ±20% jitter.
    // Gives approximately 1s, 2s, 4s, 8s, 16s, then ~30s thereafter,
    // each jittered into a ±20% window around the deterministic value.
    const base = 1000;
    const max = 30_000;
    const raw = Math.min(max, base * 2 ** attempt);
    return Math.floor(raw * (0.8 + Math.random() * 0.4));
  }

  /**
   * Disarm reconnect and close the WebSocket. Returns a Promise that resolves
   * once the close handshake has completed (i.e., `onclose` has fired).
   * Idempotent: calling while already shutting down returns the in-flight promise.
   */
  async stop(): Promise<void> {
    if (this.isShuttingDown) {
      return this.shutdownPromise;
    }
    this.isShuttingDown = true;

    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    // Important: setting isShuttingDown BEFORE close() is what defeats the
    // zombie-reconnect race. The async onclose handler will see the flag and
    // skip scheduling a reconnect.
    if (this.ws !== null) {
      this.ws.close();
      // Don't null ws here — let onclose do it after the actual close.
    } else {
      // No socket was open (e.g., stop() called between reconnect attempts);
      // resolve the shutdown promise immediately.
      this.shutdownResolve?.();
      this.shutdownResolve = null;
    }

    return this.shutdownPromise;
  }

  private openSocket(): void {
    if (this.isShuttingDown) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    console.log('Connecting to WebSocket:', wsUrl);
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      wsConnected.set(true);
      this.reconnectAttempt = 0; // healthy connection — reset backoff
      console.log('WebSocket connected');
    };

    this.ws.onclose = () => {
      wsConnected.set(false);
      this.ws = null;

      if (this.isShuttingDown) {
        // Shutdown was requested — settle the promise and stop here.
        console.log('WebSocket closed (shutdown complete)');
        this.shutdownResolve?.();
        this.shutdownResolve = null;
        return;
      }

      // Unexpected close — schedule a reconnect with backoff. The shutdown
      // flag is re-checked at fire time in case stop() races between now
      // and then.
      const delayMs = this.computeBackoffDelay(this.reconnectAttempt);
      this.reconnectAttempt += 1;
      console.log(
        `WebSocket disconnected, reconnecting in ${delayMs}ms ` +
        `(attempt ${this.reconnectAttempt})...`
      );
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null;
        if (this.isShuttingDown) return;
        this.openSocket();
      }, delayMs);
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onmessage = (event) => {
      try {
        const msg: WsMessage = JSON.parse(event.data);
        handleWsMessage(msg);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };
  }
}

// Singleton instance — preserves the previous module-level semantics so
// existing callers (currently just +layout.svelte) keep working unchanged.
const dashboardWs = new DashboardWebSocket();

/** Open the dashboard WebSocket (with auto-reconnect). Idempotent. */
export function connectWebSocket(): void {
  dashboardWs.start();
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

    case 'GatingRunCreated':
    case 'GatingRunUpdated': {
      const run = msg.data;
      gatingRunUpdates.set(run);
      console.log(`Gating run ${msg.type === 'GatingRunCreated' ? 'created' : 'updated'} for experiment ${run.experiment_id}:`, run.status);
      break;
    }
  }
}

/**
 * Disarm reconnect and close the dashboard WebSocket. Returns a Promise that
 * resolves once the close handshake completes. Callers may ignore the return
 * value (e.g., Svelte's onDestroy doesn't await its callback) — the shutdown
 * still happens correctly either way.
 */
export function disconnectWebSocket(): Promise<void> {
  return dashboardWs.stop();
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
