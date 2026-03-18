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

  // Filtering & pagination
  let statusFilter: string = 'all';
  let searchQuery: string = '';
  let pageSize = 30;
  let currentPage = 0;

  onMount(async () => {
    await loadFlows();
  });

  async function loadFlows() {
    try {
      const response = await fetch('/api/flows?limit=500');
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

  // Reset to page 0 when filters change
  $: statusFilter, searchQuery, currentPage = 0;

  $: filteredFlows = $flows.filter(f => {
    if (statusFilter !== 'all' && f.status !== statusFilter) return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      return f.name.toLowerCase().includes(q) || (f.description || '').toLowerCase().includes(q);
    }
    return true;
  });

  $: totalPages = Math.ceil(filteredFlows.length / pageSize);
  $: pagedFlows = filteredFlows.slice(currentPage * pageSize, (currentPage + 1) * pageSize);

  // Status counts for filter badges
  $: statusCounts = $flows.reduce((acc: Record<string, number>, f) => {
    acc[f.status] = (acc[f.status] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
</script>

<div class="container">
  <div class="page-header">
    <h1>Flows</h1>
    <a href="/flows/new" class="btn btn-primary">New Flow</a>
  </div>

  {#if !loading && !error && $flows.length > 0}
    <div class="filters">
      <div class="status-filters">
        <button class="filter-btn" class:active={statusFilter === 'all'} on:click={() => statusFilter = 'all'}>
          All <span class="count">{$flows.length}</span>
        </button>
        {#each ['running', 'queued', 'completed', 'failed', 'cancelled', 'pending'] as status}
          {#if statusCounts[status]}
            <button class="filter-btn" class:active={statusFilter === status} on:click={() => statusFilter = status}>
              <span class="status-dot" style="background: {getStatusColor(status)}"></span>
              {status} <span class="count">{statusCounts[status]}</span>
            </button>
          {/if}
        {/each}
      </div>
      <input
        type="text"
        class="search-input"
        placeholder="Search flows..."
        bind:value={searchQuery}
      />
    </div>
  {/if}

  {#if loading}
    <div class="loading">Loading flows...</div>
  {:else if error}
    <div class="error">{error}</div>
  {:else if $flows.length === 0}
    <div class="empty">
      <p>No flows yet.</p>
      <p class="hint">Create a flow to start a sequence of experiments.</p>
    </div>
  {:else if filteredFlows.length === 0}
    <div class="empty">
      <p>No flows match your filter.</p>
    </div>
  {:else}
    <div class="results-info">
      Showing {currentPage * pageSize + 1}-{Math.min((currentPage + 1) * pageSize, filteredFlows.length)} of {filteredFlows.length} flows
    </div>

    <div class="flows-grid">
      {#each pagedFlows as flow}
        <a href="/flows/{flow.id}" class="flow-card" class:deleting={deleting === flow.id}>
          <span class="flow-id-label">F{flow.id}</span>
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

    {#if totalPages > 1}
      <div class="pagination">
        <button class="page-btn" disabled={currentPage === 0} on:click={() => currentPage = 0}>First</button>
        <button class="page-btn" disabled={currentPage === 0} on:click={() => currentPage--}>Prev</button>
        <span class="page-info">Page {currentPage + 1} of {totalPages}</span>
        <button class="page-btn" disabled={currentPage >= totalPages - 1} on:click={() => currentPage++}>Next</button>
        <button class="page-btn" disabled={currentPage >= totalPages - 1} on:click={() => currentPage = totalPages - 1}>Last</button>
      </div>
    {/if}
  {/if}
</div>

<style>
  .page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    padding-top: 2rem;
  }

  h1 {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-primary);
  }

  .btn {
    padding: 0.5rem 1rem;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 500;
    text-decoration: none;
    transition: all 0.2s;
  }

  .btn-primary {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.85), rgba(99, 102, 241, 0.85));
    border: 1px solid rgba(59, 130, 246, 0.4);
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    color: white;
  }

  .btn-primary:hover {
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.2);
    transform: translateY(-1px);
  }

  .filters {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
    align-items: center;
    flex-wrap: wrap;
  }

  .status-filters {
    display: flex;
    gap: 0.25rem;
    flex-wrap: wrap;
  }

  .filter-btn {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.35rem 0.75rem;
    border-radius: 8px;
    border: 1px solid var(--glass-border);
    background: var(--glass-bg);
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 1rem;
    transition: all 0.2s;
  }

  .filter-btn:hover {
    border-color: rgba(59, 130, 246, 0.3);
    color: var(--text-primary);
  }

  .filter-btn.active {
    background: rgba(59, 130, 246, 0.15);
    border-color: rgba(59, 130, 246, 0.4);
    color: var(--text-primary);
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
  }

  .count {
    opacity: 0.6;
    font-size: 1rem;
  }

  .search-input {
    padding: 0.35rem 0.75rem;
    border-radius: 8px;
    border: 1px solid var(--glass-border);
    background: var(--glass-bg);
    color: var(--text-primary);
    font-size: 1rem;
    min-width: 200px;
    outline: none;
    transition: border-color 0.2s;
  }

  .search-input:focus {
    border-color: rgba(59, 130, 246, 0.4);
  }

  .search-input::placeholder {
    color: var(--text-tertiary);
  }

  .results-info {
    font-size: 1rem;
    color: var(--text-tertiary);
    margin-bottom: 0.75rem;
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

  .flow-id-label {
    font-size: 1rem;
    font-weight: 600;
    color: #b91c1c;
    opacity: 0.8;
  }

  .flow-card {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    box-shadow: var(--glass-shadow), var(--glass-inset);
    padding: 1.25rem;
    text-decoration: none;
    transition: all 0.2s;
  }

  .flow-card:hover {
    border-color: rgba(59, 130, 246, 0.4);
    box-shadow: var(--glass-shadow-hover), var(--glass-inset), 0 0 20px rgba(59, 130, 246, 0.1);
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
    font-size: 1rem;
    padding: 0.25rem 0.5rem;
    border-radius: 8px;
    color: white;
    text-transform: capitalize;
    white-space: nowrap;
    flex-shrink: 0;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.1);
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
    border: 1px solid var(--glass-border);
    color: var(--text-secondary);
    width: 24px;
    height: 24px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 1rem;
    line-height: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
  }

  .btn-delete:hover {
    background: rgba(239, 68, 68, 0.7);
    border-color: rgba(239, 68, 68, 0.3);
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

  .pagination {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0.75rem;
    margin-top: 1.5rem;
    padding-bottom: 2rem;
  }

  .page-btn {
    padding: 0.35rem 0.75rem;
    border-radius: 8px;
    border: 1px solid var(--glass-border);
    background: var(--glass-bg);
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 1rem;
    transition: all 0.2s;
  }

  .page-btn:hover:not(:disabled) {
    border-color: rgba(59, 130, 246, 0.4);
    color: var(--text-primary);
  }

  .page-btn:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }

  .page-info {
    font-size: 1rem;
    color: var(--text-secondary);
  }
</style>
