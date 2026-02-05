<script lang="ts">
  import { onMount } from 'svelte';
  import { checkpoints } from '$lib/stores';
  import type { Checkpoint } from '$lib/types';

  let loading = true;
  let error: string | null = null;
  let filterFinalOnly = false;
  let confirmDelete: number | null = null;

  onMount(async () => {
    await loadCheckpoints();
  });

  async function loadCheckpoints() {
    loading = true;
    try {
      const params = new URLSearchParams();
      if (filterFinalOnly) params.set('is_final', 'true');
      params.set('limit', '100');

      const response = await fetch(`/api/checkpoints?${params}`);
      if (!response.ok) throw new Error('Failed to fetch checkpoints');
      const data = await response.json();
      checkpoints.set(data);
    } catch (e) {
      error = e instanceof Error ? e.message : 'Unknown error';
    } finally {
      loading = false;
    }
  }

  async function deleteCheckpoint(id: number, force: boolean = false) {
    try {
      const params = force ? '?force=true' : '';
      const response = await fetch(`/api/checkpoints/${id}${params}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to delete');
      }

      checkpoints.update(c => c.filter(x => x.id !== id));
      confirmDelete = null;
    } catch (e) {
      error = e instanceof Error ? e.message : 'Unknown error';
    }
  }

  function formatBytes(bytes: number | null): string {
    if (bytes === null) return '-';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function formatDate(dateStr: string): string {
    const date = new Date(dateStr);
    return date.toLocaleString();
  }
</script>

<div class="container">
  <div class="page-header">
    <h1>Checkpoints</h1>
    <div class="filters">
      <label class="filter-checkbox">
        <input type="checkbox" bind:checked={filterFinalOnly} on:change={loadCheckpoints} />
        Final only
      </label>
    </div>
  </div>

  {#if loading}
    <div class="loading">Loading checkpoints...</div>
  {:else if error}
    <div class="error">{error}</div>
  {:else if $checkpoints.length === 0}
    <div class="empty">
      <p>No checkpoints found.</p>
      <p class="hint">Checkpoints are saved automatically during optimization runs.</p>
    </div>
  {:else}
    <div class="checkpoints-table">
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Fitness (CE)</th>
            <th>Accuracy</th>
            <th>Iterations</th>
            <th>Size</th>
            <th>Created</th>
            <th>Refs</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {#each $checkpoints as ckpt}
            <tr>
              <td>
                <div class="ckpt-name">
                  {ckpt.name}
                  {#if ckpt.is_final}
                    <span class="final-badge">Final</span>
                  {/if}
                </div>
                <div class="ckpt-path">{ckpt.file_path}</div>
              </td>
              <td>{ckpt.final_fitness?.toFixed(4) ?? '-'}</td>
              <td>{ckpt.final_accuracy ? `${(ckpt.final_accuracy * 100).toFixed(2)}%` : '-'}</td>
              <td>{ckpt.iterations_run ?? '-'}</td>
              <td>{formatBytes(ckpt.file_size_bytes)}</td>
              <td>{formatDate(ckpt.created_at)}</td>
              <td>
                {#if ckpt.reference_count > 0}
                  <span class="ref-count">{ckpt.reference_count}</span>
                {:else}
                  -
                {/if}
              </td>
              <td>
                <div class="actions">
                  <a href="/api/checkpoints/{ckpt.id}/download" class="action-btn" title="Download">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                      <polyline points="7 10 12 15 17 10"/>
                      <line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                  </a>

                  {#if confirmDelete === ckpt.id}
                    <button class="action-btn danger" on:click={() => deleteCheckpoint(ckpt.id, ckpt.reference_count > 0)}>
                      Confirm
                    </button>
                    <button class="action-btn" on:click={() => confirmDelete = null}>
                      Cancel
                    </button>
                  {:else}
                    <button
                      class="action-btn"
                      title={ckpt.reference_count > 0 ? 'Has references' : 'Delete'}
                      on:click={() => confirmDelete = ckpt.id}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"/>
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
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

    {#if $checkpoints.some(c => c.genome_stats)}
      <section class="genome-stats-section">
        <h2>Genome Statistics</h2>
        <div class="stats-grid">
          {#each $checkpoints.filter(c => c.genome_stats) as ckpt}
            <div class="stats-card">
              <h3>{ckpt.name}</h3>
              {#if ckpt.genome_stats}
                <div class="stat-row">
                  <span class="stat-label">Clusters</span>
                  <span class="stat-value">{ckpt.genome_stats.num_clusters.toLocaleString()}</span>
                </div>
                <div class="stat-row">
                  <span class="stat-label">Total Neurons</span>
                  <span class="stat-value">{ckpt.genome_stats.total_neurons.toLocaleString()}</span>
                </div>
                <div class="stat-row">
                  <span class="stat-label">Total Connections</span>
                  <span class="stat-value">{ckpt.genome_stats.total_connections?.toLocaleString() ?? '—'}</span>
                </div>
                <div class="stat-row">
                  <span class="stat-label">Bits Range</span>
                  <span class="stat-value">{ckpt.genome_stats.bits_range[0]} - {ckpt.genome_stats.bits_range[1]}</span>
                </div>
                <div class="stat-row">
                  <span class="stat-label">Neurons Range</span>
                  <span class="stat-value">{ckpt.genome_stats.neurons_range[0]} - {ckpt.genome_stats.neurons_range[1]}</span>
                </div>

                {#if ckpt.genome_stats.tier_stats && ckpt.genome_stats.tier_stats.length > 0}
                  <div class="tier-stats">
                    <h4>Per-Tier Stats</h4>
                    <table class="tier-table">
                      <thead>
                        <tr>
                          <th>Tier</th>
                          <th>Clusters</th>
                          <th>Bits</th>
                          <th>Neurons</th>
                          <th>Connections</th>
                        </tr>
                      </thead>
                      <tbody>
                        {#each ckpt.genome_stats.tier_stats as tier}
                          <tr>
                            <td>{tier.tier_index}</td>
                            <td>{tier.cluster_count}</td>
                            <td>{tier.min_bits}-{tier.max_bits}</td>
                            <td>{tier.min_neurons}-{tier.max_neurons}</td>
                            <td>{tier.total_connections?.toLocaleString() ?? '—'}</td>
                          </tr>
                        {/each}
                      </tbody>
                    </table>
                  </div>
                {/if}
              {/if}
            </div>
          {/each}
        </div>
      </section>
    {/if}
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

  .filters {
    display: flex;
    gap: 1rem;
  }

  .filter-checkbox {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1rem;
    color: var(--text-secondary);
    cursor: pointer;
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

  .checkpoints-table {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow-x: auto;
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
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-tertiary);
    text-transform: uppercase;
    background: var(--bg-tertiary);
  }

  td {
    font-size: 1rem;
    color: var(--text-primary);
  }

  tr:last-child td {
    border-bottom: none;
  }

  .ckpt-name {
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .ckpt-path {
    font-size: 1rem;
    color: var(--text-tertiary);
    font-family: monospace;
    max-width: 300px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .final-badge {
    font-size: 1rem;
    background: var(--accent-green);
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    text-transform: uppercase;
  }

  .ref-count {
    background: var(--bg-tertiary);
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    font-size: 1rem;
  }

  .actions {
    display: flex;
    gap: 0.5rem;
  }

  .action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0.375rem;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    border-radius: 4px;
    text-decoration: none;
    font-size: 1rem;
  }

  .action-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .action-btn.danger {
    color: var(--accent-red);
  }

  .genome-stats-section {
    margin-top: 2rem;
  }

  h2 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
  }

  .stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem;
  }

  .stats-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    min-width: 0;
  }

  .stats-card h3 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 0.75rem 0;
  }

  .stat-row {
    display: flex;
    justify-content: space-between;
    font-size: 1rem;
    padding: 0.25rem 0;
  }

  .stat-label {
    color: var(--text-secondary);
  }

  .stat-value {
    color: var(--text-primary);
    font-weight: 500;
  }

  .tier-stats {
    margin-top: 1rem;
    padding-top: 1rem;
    padding-bottom: 0.25rem;
    border-top: 1px solid var(--border);
    overflow-x: auto;
  }

  .tier-stats h4 {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin: 0 0 0.5rem 0;
  }

  .tier-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.7rem;
    table-layout: fixed;
    margin-bottom: 0.5rem;
  }

  .tier-table th, .tier-table td {
    padding: 0.2rem 0.25rem;
    text-align: center;
    border: 1px solid var(--border);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .tier-table th {
    background: var(--bg-tertiary);
    color: var(--text-tertiary);
    font-weight: 600;
  }

  .tier-table td {
    color: var(--text-primary);
  }
</style>
