<script lang="ts">
  import { onMount } from 'svelte';
  import type { Experiment } from '$lib/types';
  import { formatDate } from '$lib/dateFormat';
  import { formatCE, formatDuration, formatPercent } from '$lib/format';
  import { getStatusColor } from '$lib/statusColors';

  let experiments: Experiment[] = [];
  let loading = true;
  let error: string | null = null;

  onMount(async () => {
    try {
      const response = await fetch('/api/experiments');
      if (!response.ok) throw new Error('Failed to fetch experiments');
      experiments = await response.json();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Unknown error';
    } finally {
      loading = false;
    }
  });
</script>

<div class="container">
  <div class="page-header">
    <h1>Experiment History</h1>
  </div>

  {#if loading}
    <div class="loading">Loading experiments...</div>
  {:else if error}
    <div class="error">{error}</div>
  {:else if experiments.length === 0}
    <div class="empty">
      <p>No experiments yet.</p>
      <p class="hint">Start a flow or run an optimization to create experiments.</p>
    </div>
  {:else}
    <div class="experiments-table">
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Status</th>
            <th class="col-ce">Best CE</th>
            <th class="col-acc">Best Acc</th>
            <th>Started</th>
            <th>Duration</th>
            <th>Config</th>
          </tr>
        </thead>
        <tbody>
          {#each experiments as exp}
            <tr>
              <td>
                <a href="/experiments/{exp.id}" class="exp-link">{exp.name}</a>
              </td>
              <td>
                <span class="status-badge" style="background: {getStatusColor(exp.status)}">
                  {exp.status}
                </span>
              </td>
              <td class="col-ce mono">{formatCE(exp.best_ce)}</td>
              <td class="col-acc mono">{formatPercent(exp.best_accuracy)}</td>
              <td>{exp.started_at ? formatDate(exp.started_at) : '—'}</td>
              <td>{!exp.started_at ? '—' : exp.ended_at ? formatDuration(exp.started_at, exp.ended_at) : 'Running...'}</td>
              <td>
                <div class="config-preview">
                  {#if exp.tier_config}
                    <span class="config-tag">Tiered</span>
                  {/if}
                  {#if exp.phase_type}
                    <span class="config-tag">{exp.phase_type}</span>
                  {/if}
                  {#if exp.max_iterations}
                    <span class="config-tag">{exp.max_iterations} iters</span>
                  {/if}
                </div>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</div>

<style>
  .page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    padding-top: 2rem;
  }

  h1 {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-primary);
  }

  .loading, .error, .empty {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--text-secondary);
  }

  .error {
    color: var(--accent-red);
  }

  .hint {
    font-size: 1rem;
    margin-top: 0.5rem;
    opacity: 0.7;
  }

  .experiments-table {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    box-shadow: var(--glass-shadow), var(--glass-inset);
    overflow: hidden;
  }

  table {
    width: 100%;
    border-collapse: collapse;
  }

  th, td {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid rgba(71, 85, 105, 0.4);
  }

  th {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-tertiary);
    text-transform: uppercase;
    background: rgba(30, 41, 59, 0.3);
  }

  td {
    font-size: 1rem;
    color: var(--text-primary);
  }

  tr:last-child td {
    border-bottom: none;
  }

  .exp-link {
    color: var(--accent-blue);
    text-decoration: none;
    font-weight: 500;
  }

  .exp-link:hover {
    text-decoration: underline;
  }

  .status-badge {
    display: inline-block;
    font-size: 1rem;
    padding: 0.25rem 0.5rem;
    border-radius: 8px;
    color: white;
    text-transform: capitalize;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  .config-preview {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .config-tag {
    font-size: 1rem;
    color: var(--text-tertiary);
    background: rgba(51, 65, 85, 0.4);
    border: 1px solid var(--glass-border);
    padding: 0.25rem 0.5rem;
    border-radius: 8px;
  }

  .col-ce, .col-acc {
    text-align: right;
    width: 100px;
  }

  .mono {
    font-family: monospace;
  }
</style>
