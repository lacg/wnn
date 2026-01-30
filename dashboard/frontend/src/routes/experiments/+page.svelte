<script lang="ts">
  import { onMount } from 'svelte';
  import type { Experiment } from '$lib/types';
  import { formatDate } from '$lib/dateFormat';

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

  function getStatusColor(status: string): string {
    switch (status) {
      case 'running': return 'var(--accent-blue)';
      case 'completed': return 'var(--accent-green)';
      case 'failed': return 'var(--accent-red)';
      case 'cancelled': return 'var(--text-tertiary)';
      default: return 'var(--text-secondary)';
    }
  }

  function formatDuration(start: string, end: string | null): string {
    if (!end) return 'Running...';
    const startDate = new Date(start);
    const endDate = new Date(end);
    const seconds = Math.floor((endDate.getTime() - startDate.getTime()) / 1000);

    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${mins}m`;
  }
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
              <td>{formatDate(exp.started_at)}</td>
              <td>{formatDuration(exp.started_at, exp.ended_at)}</td>
              <td>
                <div class="config-preview">
                  {#if exp.config.tier_config}
                    <span class="config-tag">Tiered</span>
                  {/if}
                  {#if exp.config.ga_generations}
                    <span class="config-tag">GA: {exp.config.ga_generations}</span>
                  {/if}
                  {#if exp.config.ts_iterations}
                    <span class="config-tag">TS: {exp.config.ts_iterations}</span>
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
    font-size: 0.875rem;
    margin-top: 0.5rem;
    opacity: 0.7;
  }

  .experiments-table {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }

  table {
    width: 100%;
    border-collapse: collapse;
  }

  th, td {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }

  th {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-tertiary);
    text-transform: uppercase;
    background: var(--bg-tertiary);
  }

  td {
    font-size: 0.875rem;
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
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    color: white;
    text-transform: capitalize;
  }

  .config-preview {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .config-tag {
    font-size: 0.75rem;
    color: var(--text-tertiary);
    background: var(--bg-tertiary);
    padding: 0.125rem 0.375rem;
    border-radius: 3px;
  }
</style>
