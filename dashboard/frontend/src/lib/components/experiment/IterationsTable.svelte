<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { Iteration } from '$lib/types';
  import { formatDate } from '$lib/dateFormat';
  import { formatCE } from '$lib/format';
  import { formatAccShort, formatF1, formatFPR } from './metricFormat';

  export let displayIterations: Iteration[] = [];
  export let maxIterations: number | null = null;
  export let isIDS: boolean = false;
  export let bestCE: number = Infinity;
  export let bestAcc: number | null = null;
  export let bestF1: number | null = null;
  export let bestFpr: number | null = null;

  const dispatch = createEventDispatcher<{ openDetails: Iteration }>();
</script>

<div class="card">
  <div class="card-header">
    <span class="card-title">Iterations</span>
    <span class="count">{displayIterations.length}{#if maxIterations} / {maxIterations}{/if} iterations</span>
  </div>
  {#if displayIterations.length === 0}
    <div class="empty-state">No iterations recorded</div>
  {:else}
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th>Iter</th>
            <th>Timestamp</th>
            {#if isIDS}
              <th>Best F1</th>
              <th>Best FPR</th>
              <th>Best Acc</th>
            {:else}
              <th>Best CE</th>
              <th>Best Acc</th>
              <th>Avg CE</th>
              <th>Avg Acc</th>
            {/if}
            <th>Threshold</th>
            <th>Δ Prev</th>
            <th>Patience</th>
            <th>Time</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {#each [...displayIterations].reverse() as iter}
            <tr
              class="clickable"
              on:click={() => dispatch('openDetails', iter)}
              on:keydown={(e) => e.key === 'Enter' && dispatch('openDetails', iter)}
              tabindex={0}
              role="button"
            >
              <td>{iter.iteration_num}</td>
              <td class="timestamp">{formatDate(iter.created_at)}</td>
              {#if isIDS}
                <td class:best={iter.best_f1 !== null && bestF1 !== null && iter.best_f1 === bestF1}>{formatF1(iter.best_f1)}</td>
                <td class:best={iter.best_fpr !== null && bestFpr !== null && iter.best_fpr === bestFpr}>{formatFPR(iter.best_fpr)}</td>
                <td class:best={iter.best_accuracy !== null && iter.best_accuracy === bestAcc}>{formatAccShort(iter.best_accuracy)}</td>
              {:else}
                <td class:best={iter.best_ce === bestCE}>{formatCE(iter.best_ce)}</td>
                <td class:best={iter.best_accuracy !== null && iter.best_accuracy === bestAcc}>{formatAccShort(iter.best_accuracy)}</td>
                <td class="secondary">{iter.avg_ce ? formatCE(iter.avg_ce) : '—'}</td>
                <td class="secondary">{formatAccShort(iter.avg_accuracy)}</td>
              {/if}
              <td class="secondary">{iter.fitness_threshold !== null ? formatAccShort(iter.fitness_threshold) : '—'}</td>
              <td class:delta-positive={iter.delta_previous && iter.delta_previous < 0} class:delta-negative={iter.delta_previous && iter.delta_previous > 0}>
                {iter.delta_previous !== null ? (iter.delta_previous < 0 ? '↓' : iter.delta_previous > 0 ? '↑' : '') + Math.abs(iter.delta_previous).toFixed(4) : '—'}
              </td>
              <td>{iter.patience_counter !== null && iter.patience_max ? `${iter.patience_max - iter.patience_counter}/${iter.patience_max}` : '—'}</td>
              <td>{iter.elapsed_secs ? iter.elapsed_secs.toFixed(1) + 's' : '—'}</td>
              <td class="view-link">View →</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</div>

<style>
  /* Card styles */
  .card {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: 0.5rem;
    margin-bottom: 1.5rem;
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    border-bottom: 1px solid var(--glass-border);
  }

  .card-title {
    font-weight: 600;
    color: var(--text-primary);
  }

  .count {
    font-size: 1rem;
    color: var(--text-primary);
  }

  /* Table */
  .table-scroll {
    max-height: 500px;
    overflow-y: auto;
  }

  table {
    width: 100%;
    border-collapse: collapse;
  }

  th, td {
    padding: 0.5rem 0.625rem;
    text-align: left;
    border-bottom: 1px solid var(--glass-border);
  }

  th {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    background: rgba(51, 65, 85, 0.4);
    position: sticky;
    top: 0;
    z-index: 1;
  }

  td {
    font-size: 1rem;
    font-family: monospace;
  }

  tr:last-child td {
    border-bottom: none;
  }

  tr.clickable {
    cursor: pointer;
    transition: background-color 0.15s;
  }

  tr.clickable:hover {
    background: rgba(59, 130, 246, 0.1);
  }

  .best {
    color: var(--accent-green);
    font-weight: 600;
  }

  .secondary {
    color: var(--text-secondary);
  }

  .timestamp {
    color: var(--text-secondary);
    font-family: monospace;
    font-size: 1rem;
  }

  .delta-positive {
    color: var(--accent-green);
  }

  .delta-negative {
    color: var(--accent-red);
  }

  .view-link {
    color: var(--accent-blue);
    font-size: 1rem;
    opacity: 0.7;
  }

  tr.clickable:hover .view-link {
    opacity: 1;
  }

  .empty-state {
    padding: 2rem;
    text-align: center;
    color: var(--text-secondary);
  }
</style>
