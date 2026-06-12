<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { Flow, Experiment } from '$lib/types';
  import { formatCE, formatPercent } from '$lib/format';

  export let flow: Flow;
  export let displayExperiments: Experiment[] = [];
  /** Raw (unsorted) experiments — canEditExperiment indexes into this list,
   *  matching the page's pre-extraction behavior. */
  export let experiments: Experiment[] = [];
  export let isIDS: boolean = false;
  export let saving: boolean = false;
  export let actionInFlight: boolean = false;

  const dispatch = createEventDispatcher<{
    move: { index: number; direction: -1 | 1 };
    delete: number;
    updateIterations: { expId: number; iterations: number };
    stop: void;
    restartFrom: number;
  }>();

  function canEditExperiment(index: number): boolean {
    if (!flow) return false;
    // Can always edit if flow is pending, queued, or failed
    if (flow.status === 'pending' || flow.status === 'queued' || flow.status === 'failed') return true;
    // Can't edit completed or cancelled flows
    if (flow.status !== 'running') return false;

    // For running flows, check if this experiment has started
    const exp = experiments[index];
    if (!exp) return false;
    return exp.status === 'pending';
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'queued': return 'var(--accent-yellow, #f59e0b)';
      case 'running': return 'var(--accent-blue)';
      case 'completed': return 'var(--accent-green)';
      case 'failed': return 'var(--accent-red)';
      case 'cancelled': return 'var(--text-tertiary)';
      default: return 'var(--text-secondary)';
    }
  }

  // Get default iterations based on experiment type (GA vs TS)
  function getDefaultIterations(expType: string): number {
    return expType === 'GA'
      ? (flow?.config.params.ga_generations ?? 250)
      : (flow?.config.params.ts_iterations ?? 250);
  }

  // Get the link URL for an experiment - all experiments are viewable
  function getExperimentLink(exp: Experiment): string {
    return `/experiments/${exp.id}`;
  }
</script>

<div class="experiments-table">
  <table>
    <thead>
      <tr>
        <th class="col-reorder"></th>
        <th class="col-order">#</th>
        <th class="col-name">Name</th>
        <th class="col-type">Type</th>
        <th class="col-iters">Iterations</th>
        <th class="col-status">Status</th>
        {#if isIDS}
          <th class="col-ce">Best F1</th>
          <th class="col-acc">Best Acc</th>
        {:else}
          <th class="col-ce">Best CE</th>
          <th class="col-acc">Best Acc</th>
        {/if}
        <th class="col-actions">Actions</th>
      </tr>
    </thead>
    <tbody>
      {#each displayExperiments as exp, i}
        {@const isRunning = exp.status === 'running'}
        {@const isCompleted = exp.status === 'completed'}
        {@const isPending = exp.status === 'pending'}
        {@const canEdit = canEditExperiment(i)}
        {@const isGridSearch = exp.phase_type === 'grid_search'}
        {@const isAdapt = ['neurogenesis', 'synaptogenesis', 'axonogenesis'].includes(exp.phase_type ?? '')}
        {@const expType = isGridSearch ? 'GRID' : isAdapt ? exp.phase_type?.toUpperCase()?.slice(0, 5) ?? '—' : exp.phase_type?.startsWith('ga') ? 'GA' : exp.phase_type?.startsWith('ts') ? 'TS' : '—'}
        {@const optimizeTarget = isGridSearch ? '' : isAdapt ? '' : exp.phase_type?.includes('bits') ? 'Bits' : exp.phase_type?.includes('neurons') ? 'Neurons' : exp.phase_type?.includes('connections') ? 'Conn' : '—'}
        {@const expLink = getExperimentLink(exp)}
        {@const phaseGroup = isGridSearch ? 'grid' : (exp.phase_type?.includes('neuron') || exp.phase_type === 'neurogenesis') ? 'neurons' : (exp.phase_type?.includes('bits') || exp.phase_type === 'synaptogenesis') ? 'bits' : (exp.phase_type?.includes('connection') || exp.phase_type === 'axonogenesis') ? 'connections' : 'other'}
        {@const prevExp = i > 0 ? displayExperiments[i - 1] : null}
        {@const prevGroup = prevExp ? (prevExp.phase_type === 'grid_search' ? 'grid' : (prevExp.phase_type?.includes('neuron') || prevExp.phase_type === 'neurogenesis') ? 'neurons' : (prevExp.phase_type?.includes('bits') || prevExp.phase_type === 'synaptogenesis') ? 'bits' : (prevExp.phase_type?.includes('connection') || prevExp.phase_type === 'axonogenesis') ? 'connections' : 'other') : null}
        {@const showDivider = i > 0 && phaseGroup !== prevGroup}
        {#if showDivider}
          <tr class="phase-divider"><td colspan="9"><div class="divider-line"><span class="divider-label">{phaseGroup === 'neurons' ? 'Neurons' : phaseGroup === 'bits' ? 'Bits' : phaseGroup === 'connections' ? 'Connections' : phaseGroup}</span></div></td></tr>
        {/if}
        <tr class:row-running={isRunning} class:row-completed={isCompleted} class:row-pending={isPending}>
          <td class="col-reorder">
            {#if isPending && canEdit}
              <div class="reorder-buttons">
                <button class="move-btn" on:click={() => dispatch('move', { index: i, direction: -1 })} disabled={i === 0 || saving} title="Move up">&uarr;</button>
                <button class="move-btn" on:click={() => dispatch('move', { index: i, direction: 1 })} disabled={i === displayExperiments.length - 1 || saving} title="Move down">&darr;</button>
              </div>
            {/if}
          </td>
          <td class="col-order">
            <span class="order-badge" class:order-completed={isCompleted} class:order-running={isRunning}>
              {#if isCompleted}✓{:else}{i + 1}{/if}
            </span>
          </td>
          <td class="col-name clickable-cell">
            <a href={expLink} class="cell-link">
              {exp.name}
              {#if isRunning}
                <span class="live-badge"><span class="pulse"></span>Live</span>
              {/if}
            </a>
          </td>
          <td class="col-type">
            <span class="type-badge" class:type-ga={expType === 'GA'} class:type-ts={expType === 'TS'} class:type-grid={isGridSearch} class:type-adapt={isAdapt}>{expType}</span>
            {#if optimizeTarget}<span class="target-badge">{optimizeTarget}</span>{/if}
          </td>
          <td class="col-iters">
            {#if isPending && canEdit}
              <input
                type="number"
                class="iters-input"
                value={exp.max_iterations ?? getDefaultIterations(expType)}
                min="10"
                max="10000"
                on:change={(e) => dispatch('updateIterations', { expId: exp.id, iterations: parseInt(e.currentTarget.value) })}
              />
            {:else if isRunning}
              <span class="iters-progress">{exp.current_iteration ?? 0}/{exp.max_iterations ?? '?'}</span>
            {:else}
              <span class="mono">{exp.current_iteration ?? exp.max_iterations ?? '—'}</span>
            {/if}
          </td>
          <td class="col-status">
            <span class="status-pill" style="background: {getStatusColor(exp.status)}">{exp.status}</span>
          </td>
          {#if isIDS}
            <td class="col-ce mono">{exp.extra_metrics?.f1_macro != null ? (exp.extra_metrics.f1_macro * 100).toFixed(2) + '%' : '—'}</td>
          {:else}
            <td class="col-ce mono">{formatCE(exp.best_ce)}</td>
          {/if}
          <td class="col-acc mono">{formatPercent(exp.best_accuracy)}</td>
          <td class="col-actions">
            <div class="action-buttons">
              {#if canEdit}
                <button class="btn-icon btn-danger" title="Delete" on:click={() => dispatch('delete', i)}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="3 6 5 6 21 6"></polyline>
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                  </svg>
                </button>
              {/if}
              {#if isRunning}
                <button class="btn-icon btn-danger" title="Stop" on:click={() => dispatch('stop')}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="6" y="6" width="12" height="12" rx="2"></rect>
                  </svg>
                </button>
              {/if}
              {#if (isCompleted || isRunning) && (flow.status === 'running' || flow.status === 'failed' || flow.status === 'cancelled' || flow.status === 'completed')}
                <button class="btn-icon" title="Restart from here" on:click={() => dispatch('restartFrom', i)} disabled={actionInFlight}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="1 4 1 10 7 10"></polyline>
                    <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path>
                  </svg>
                </button>
              {/if}
            </div>
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
</div>

<style>
  .pulse {
    width: 8px;
    height: 8px;
    background: white;
    border-radius: 50%;
    animation: pulse 1.5s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.2); }
  }

  /* Experiments Table */
  .experiments-table {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    overflow: hidden;
  }

  .experiments-table table {
    width: 100%;
    border-collapse: collapse;
  }

  .experiments-table th,
  .experiments-table td {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--glass-border);
  }

  .experiments-table th {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-tertiary);
    text-align: center;
    text-transform: uppercase;
    background: rgba(51, 65, 85, 0.4);
  }

  .experiments-table td {
    font-size: 1rem;
    color: var(--text-primary);
  }

  .experiments-table tr:last-child td {
    border-bottom: none;
  }

  .experiments-table .col-reorder { width: 44px; text-align: center; padding: 0.25rem !important; }
  .experiments-table .col-order { width: 40px; text-align: center; }
  .experiments-table .col-name { min-width: 200px; text-align: left; }
  .experiments-table .col-type { width: 140px; white-space: nowrap; text-align: center; }
  .experiments-table .col-iters { width: 100px; text-align: center; }
  .experiments-table .col-status { width: 100px; text-align: center; }
  .experiments-table .col-ce { width: 100px; text-align: right; }
  .experiments-table .col-acc { width: 100px; text-align: right; }
  .experiments-table .col-actions { width: 120px; text-align: center; }

  .reorder-buttons {
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .move-btn {
    background: rgba(51, 65, 85, 0.4);
    border: 1px solid var(--glass-border);
    border-radius: 3px;
    color: var(--text-secondary);
    cursor: pointer;
    padding: 0 4px;
    font-size: 1rem;
    line-height: 1.2;
    transition: all 0.15s;
  }

  .move-btn:hover:not(:disabled) {
    background: var(--border);
    color: var(--text-primary);
  }

  .move-btn:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }

  .iters-input {
    width: 70px;
    padding: 0.25rem 0.5rem;
    border: 1px solid var(--glass-border);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 1rem;
    text-align: center;
  }

  .iters-input:hover {
    border-color: var(--accent-blue);
  }

  .iters-input:focus {
    outline: none;
    border-color: var(--accent-blue);
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
  }

  .iters-progress {
    font-size: 1rem;
    font-family: monospace;
    color: var(--accent-blue);
  }

  .experiments-table .mono {
    font-family: monospace;
  }

  .order-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: rgba(51, 65, 85, 0.4);
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-secondary);
  }

  .order-badge.order-completed {
    background: var(--accent-green);
    color: white;
  }

  .order-badge.order-running {
    background: var(--accent-blue);
    color: white;
  }

  .clickable-cell {
    padding: 0 !important;
  }

  .cell-link {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;
    height: 100%;
    padding: 0.75rem 1rem;
    color: var(--text-primary);
    text-decoration: none;
    transition: background-color 0.15s, color 0.15s;
  }

  .cell-link:hover {
    background: rgba(51, 65, 85, 0.4);
    color: var(--accent-blue);
  }

  .live-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 1rem;
    font-weight: 600;
    color: var(--accent-blue);
    text-transform: uppercase;
    background: rgba(59, 130, 246, 0.1);
    padding: 0.125rem 0.375rem;
    border-radius: 3px;
  }

  .type-badge {
    font-size: 1rem;
    font-weight: 600;
    padding: 0.125rem 0.375rem;
    border-radius: 3px;
  }

  .type-badge.type-ga {
    background: rgba(59, 130, 246, 0.15);
    color: var(--accent-blue);
  }

  .type-badge.type-ts {
    background: rgba(16, 185, 129, 0.15);
    color: var(--accent-green);
  }

  .type-badge.type-grid {
    background: rgba(245, 158, 11, 0.15);
    color: var(--accent-yellow, #f59e0b);
  }

  .type-badge.type-adapt {
    background: rgba(139, 92, 246, 0.15);
    color: var(--accent-purple, #8b5cf6);
  }

  .target-badge {
    font-size: 1rem;
    color: var(--text-tertiary);
    margin-left: 0.25rem;
  }

  .status-pill {
    display: inline-block;
    font-size: 1rem;
    font-weight: 600;
    padding: 0.25rem 0.5rem;
    border-radius: 9999px;
    color: white;
    text-transform: uppercase;
  }

  .action-buttons {
    display: flex;
    gap: 0.25rem;
    justify-content: flex-end;
  }

  .row-running {
    background: rgba(59, 130, 246, 0.05);
  }

  .row-completed {
    background: rgba(34, 197, 94, 0.05); /* subtle green tint, mirrors .row-running blue */
  }

  .row-pending {
    opacity: 0.7;
  }

  .phase-divider td {
    padding: 0;
    border-bottom: none;
  }

  .divider-line {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.25rem 1rem;
  }

  .divider-line::before, .divider-line::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--glass-border);
  }

  .divider-label {
    font-size: 1rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    white-space: nowrap;
  }

  .btn-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border: none;
    border-radius: 4px;
    background: rgba(51, 65, 85, 0.4);
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn-icon:hover {
    background: var(--border);
    color: var(--text-primary);
  }

  .btn-icon.btn-danger:hover {
    background: var(--accent-red);
    color: white;
  }
</style>
