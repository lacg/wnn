<script lang="ts">
  import { onMount } from 'svelte';
  import { flows } from '$lib/stores';
  import type { Flow } from '$lib/types';
  import { formatDate } from '$lib/dateFormat';

  // Helper: get experiments count safely (handles {} stored instead of [])
  function getExperimentsCount(flow: Flow): number {
    const exps = flow?.config?.experiments;
    if (Array.isArray(exps)) return exps.length;
    return 0;
  }

  let loading = true;
  let error: string | null = null;
  let deleting: number | null = null;

  onMount(async () => {
    await loadFlows();
  });

  async function loadFlows() {
    try {
      const response = await fetch('/api/flows');
      if (!response.ok) throw new Error('Failed to fetch flows');
      const data = await response.json();
      flows.set(data);
    } catch (e) {
      error = e instanceof Error ? e.message : 'Unknown error';
    } finally {
      loading = false;
    }
  }

  async function deleteFlow(event: MouseEvent, flow: Flow) {
    event.preventDefault();
    event.stopPropagation();

    if (!confirm(`Delete flow "${flow.name}"? This cannot be undone.`)) return;

    deleting = flow.id;
    try {
      const response = await fetch(`/api/flows/${flow.id}`, { method: 'DELETE' });
      if (!response.ok) throw new Error('Failed to delete flow');
      await loadFlows();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to delete';
    } finally {
      deleting = null;
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'running': return 'var(--accent-blue)';
      case 'completed': return 'var(--accent-green)';
      case 'failed': return 'var(--accent-red)';
      case 'cancelled': return 'var(--text-tertiary)';
      default: return 'var(--text-secondary)';
    }
  }

</script>

<div class="container">
  <div class="page-header">
    <h1>Flows</h1>
    <a href="/flows/new" class="btn btn-primary">New Flow</a>
  </div>

  {#if loading}
    <div class="loading">Loading flows...</div>
  {:else if error}
    <div class="error">{error}</div>
  {:else if $flows.length === 0}
    <div class="empty">
      <p>No flows yet.</p>
      <p class="hint">Create a flow to start a sequence of experiments.</p>
    </div>
  {:else}
    <div class="flows-grid">
      {#each $flows as flow}
        <a href="/flows/{flow.id}" class="flow-card" class:deleting={deleting === flow.id}>
          <div class="flow-header">
            <h3 class="flow-name">{flow.name}</h3>
            <div class="flow-header-actions">
              <span class="status-badge" style="background: {getStatusColor(flow.status)}">
                {flow.status}
              </span>
              <button
                class="btn-delete"
                title="Delete flow"
                on:click={(e) => deleteFlow(e, flow)}
                disabled={deleting === flow.id}
              >
                {deleting === flow.id ? '...' : '×'}
              </button>
            </div>
          </div>

          {#if flow.description}
            <p class="flow-description">{flow.description}</p>
          {/if}

          <div class="flow-meta">
            <div class="meta-item">
              <span class="meta-label">Experiments</span>
              <span class="meta-value">{getExperimentsCount(flow)}</span>
            </div>
            {#if flow.config.template}
              <div class="meta-item">
                <span class="meta-label">Template</span>
                <span class="meta-value">{flow.config.template}</span>
              </div>
            {/if}
          </div>

          <div class="flow-dates">
            <span class="date-item">Created: {formatDate(flow.created_at)}</span>
            {#if flow.completed_at}
              <span class="date-item">Completed: {formatDate(flow.completed_at)}</span>
            {:else if flow.started_at}
              <span class="date-item">Started: {formatDate(flow.started_at)}</span>
            {/if}
          </div>
        </a>
      {/each}
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

  .btn {
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-size: 1rem;
    font-weight: 500;
    text-decoration: none;
    transition: all 0.2s;
  }

  .btn-primary {
    background: var(--accent-blue);
    color: white;
  }

  .btn-primary:hover {
    opacity: 0.9;
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

  .flows-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 1rem;
  }

  .flow-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    text-decoration: none;
    transition: all 0.2s;
  }

  .flow-card:hover {
    border-color: var(--accent-blue);
    transform: translateY(-2px);
  }

  .flow-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
    min-width: 0;
  }

  .flow-name {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .status-badge {
    font-size: 0.75rem;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    color: white;
    text-transform: capitalize;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .flow-description {
    font-size: 1rem;
    color: var(--text-secondary);
    margin: 0 0 1rem 0;
    line-height: 1.4;
  }

  .flow-meta {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1rem;
  }

  .meta-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .meta-label {
    font-size: 1rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
  }

  .meta-value {
    font-size: 1rem;
    color: var(--text-primary);
  }

  .flow-dates {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 1rem;
    color: var(--text-tertiary);
  }

  .flow-header-actions {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-shrink: 0;
  }

  .btn-delete {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    width: 24px;
    height: 24px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
    line-height: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
  }

  .btn-delete:hover {
    background: var(--accent-red);
    border-color: var(--accent-red);
    color: white;
  }

  .btn-delete:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .flow-card.deleting {
    opacity: 0.5;
    pointer-events: none;
  }
</style>
