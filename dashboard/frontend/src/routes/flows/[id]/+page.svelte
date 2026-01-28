<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import type { Flow, Experiment } from '$lib/types';

  let flow: Flow | null = null;
  let experiments: Experiment[] = [];
  let loading = true;
  let error: string | null = null;
  let saving = false;
  let editMode = false;

  // Edit form state
  let editConfig = {
    patience: 10,
    ga_generations: 250,
    ts_iterations: 250,
    population_size: 50,
    neighbors_per_iter: 50,
    fitness_percentile: 0.75,
    tier_config: '',
    optimize_tier0_only: false,
    phase_order: 'neurons_first'
  };

  $: flowId = $page.params.id;

  onMount(async () => {
    await loadFlow();
  });

  async function loadFlow() {
    loading = true;
    try {
      const [flowRes, expsRes] = await Promise.all([
        fetch(`/api/flows/${flowId}`),
        fetch(`/api/flows/${flowId}/experiments`)
      ]);

      if (!flowRes.ok) throw new Error('Flow not found');

      flow = await flowRes.json();
      experiments = await expsRes.json();

      // Populate edit form from config
      if (flow?.config?.params) {
        const p = flow.config.params;
        editConfig.patience = p.patience ?? 10;
        editConfig.ga_generations = p.ga_generations ?? 250;
        editConfig.ts_iterations = p.ts_iterations ?? 250;
        editConfig.population_size = p.population_size ?? 50;
        editConfig.neighbors_per_iter = p.neighbors_per_iter ?? 50;
        editConfig.fitness_percentile = p.fitness_percentile ?? 0.75;
        editConfig.optimize_tier0_only = p.optimize_tier0_only ?? false;
        editConfig.phase_order = p.phase_order ?? 'neurons_first';
        if (p.tier_config) {
          editConfig.tier_config = p.tier_config
            .map((t: any) => `${t[0] ?? 'rest'},${t[1]},${t[2]}`)
            .join(';');
        }
      }
    } catch (e) {
      error = e instanceof Error ? e.message : 'Unknown error';
    } finally {
      loading = false;
    }
  }

  async function saveChanges() {
    if (!flow) return;
    saving = true;

    try {
      // Parse tier_config string
      let tier_config = null;
      if (editConfig.tier_config.trim()) {
        tier_config = editConfig.tier_config.split(';').map(tier => {
          const parts = tier.trim().split(',');
          return [
            parts[0] === 'rest' ? null : parseInt(parts[0]),
            parseInt(parts[1]),
            parseInt(parts[2])
          ];
        });
      }

      const updatedConfig = {
        ...flow.config,
        params: {
          ...flow.config.params,
          patience: editConfig.patience,
          ga_generations: editConfig.ga_generations,
          ts_iterations: editConfig.ts_iterations,
          population_size: editConfig.population_size,
          neighbors_per_iter: editConfig.neighbors_per_iter,
          fitness_percentile: editConfig.fitness_percentile,
          optimize_tier0_only: editConfig.optimize_tier0_only,
          phase_order: editConfig.phase_order,
          tier_config
        }
      };

      const res = await fetch(`/api/flows/${flowId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: updatedConfig })
      });

      if (!res.ok) throw new Error('Failed to save');

      editMode = false;
      await loadFlow();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to save';
    } finally {
      saving = false;
    }
  }

  function cancelEdit() {
    editMode = false;
    // Reload to reset form
    loadFlow();
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

  function formatDate(dateStr: string | null): string {
    if (!dateStr) return '-';
    const date = new Date(dateStr);
    return date.toLocaleString();
  }
</script>

<div class="container">
  {#if loading}
    <div class="loading">Loading flow...</div>
  {:else if error}
    <div class="error">{error}</div>
  {:else if flow}
    <div class="flow-header">
      <div class="header-left">
        <a href="/flows" class="back-link">&larr; Flows</a>
        <h1>{flow.name}</h1>
        <span class="status-badge" style="background: {getStatusColor(flow.status)}">
          {flow.status}
        </span>
      </div>
      <div class="header-actions">
        {#if !editMode && flow.status === 'pending'}
          <button class="btn btn-secondary" on:click={() => editMode = true}>
            Edit Config
          </button>
        {/if}
      </div>
    </div>

    {#if flow.description}
      <p class="description">{flow.description}</p>
    {/if}

    <div class="info-cards">
      <div class="info-card">
        <span class="info-label">Created</span>
        <span class="info-value">{formatDate(flow.created_at)}</span>
      </div>
      {#if flow.started_at}
        <div class="info-card">
          <span class="info-label">Started</span>
          <span class="info-value">{formatDate(flow.started_at)}</span>
        </div>
      {/if}
      {#if flow.completed_at}
        <div class="info-card">
          <span class="info-label">Completed</span>
          <span class="info-value">{formatDate(flow.completed_at)}</span>
        </div>
      {/if}
      {#if flow.config.template}
        <div class="info-card">
          <span class="info-label">Template</span>
          <span class="info-value">{flow.config.template}</span>
        </div>
      {/if}
    </div>

    {#if editMode}
      <section class="section edit-section">
        <h2>Edit Configuration</h2>
        <div class="edit-form">
          <div class="form-row">
            <div class="form-group">
              <label for="patience">Patience</label>
              <input type="number" id="patience" bind:value={editConfig.patience} min="1" max="100" />
              <span class="form-hint">Early stopping patience</span>
            </div>
            <div class="form-group">
              <label for="phase_order">Phase Order</label>
              <select id="phase_order" bind:value={editConfig.phase_order}>
                <option value="neurons_first">Neurons First</option>
                <option value="bits_first">Bits First</option>
              </select>
            </div>
          </div>

          <div class="form-row">
            <div class="form-group">
              <label for="ga_generations">GA Generations</label>
              <input type="number" id="ga_generations" bind:value={editConfig.ga_generations} min="10" max="10000" />
            </div>
            <div class="form-group">
              <label for="ts_iterations">TS Iterations</label>
              <input type="number" id="ts_iterations" bind:value={editConfig.ts_iterations} min="10" max="10000" />
            </div>
          </div>

          <div class="form-row">
            <div class="form-group">
              <label for="population_size">Population Size</label>
              <input type="number" id="population_size" bind:value={editConfig.population_size} min="10" max="500" />
            </div>
            <div class="form-group">
              <label for="neighbors_per_iter">Neighbors/Iter</label>
              <input type="number" id="neighbors_per_iter" bind:value={editConfig.neighbors_per_iter} min="10" max="500" />
            </div>
          </div>

          <div class="form-row">
            <div class="form-group">
              <label for="fitness_percentile">Fitness Percentile</label>
              <input type="number" id="fitness_percentile" bind:value={editConfig.fitness_percentile} min="0" max="1" step="0.05" />
              <span class="form-hint">Keep top N% by fitness (0.75 = top 75%)</span>
            </div>
            <div class="form-group checkbox-group">
              <label>
                <input type="checkbox" bind:checked={editConfig.optimize_tier0_only} />
                Optimize Tier0 Only
              </label>
              <span class="form-hint">Only mutate most frequent tokens</span>
            </div>
          </div>

          <div class="form-group full-width">
            <label for="tier_config">Tier Config</label>
            <input type="text" id="tier_config" bind:value={editConfig.tier_config}
                   placeholder="100,15,20;400,10,12;rest,5,8" />
            <span class="form-hint">Format: clusters,neurons,bits per tier (semicolon separated)</span>
          </div>

          <div class="form-actions">
            <button class="btn btn-secondary" on:click={cancelEdit} disabled={saving}>
              Cancel
            </button>
            <button class="btn btn-primary" on:click={saveChanges} disabled={saving}>
              {saving ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </div>
      </section>
    {:else if flow.config.params}
      <section class="section">
        <h2>Parameters</h2>
        <div class="params-grid">
          <div class="param-item">
            <span class="param-label">Patience</span>
            <span class="param-value">{flow.config.params.patience ?? '-'}</span>
          </div>
          <div class="param-item">
            <span class="param-label">Phase Order</span>
            <span class="param-value">{flow.config.params.phase_order ?? 'neurons_first'}</span>
          </div>
          <div class="param-item">
            <span class="param-label">GA Generations</span>
            <span class="param-value">{flow.config.params.ga_generations ?? '-'}</span>
          </div>
          <div class="param-item">
            <span class="param-label">TS Iterations</span>
            <span class="param-value">{flow.config.params.ts_iterations ?? '-'}</span>
          </div>
          <div class="param-item">
            <span class="param-label">Population</span>
            <span class="param-value">{flow.config.params.population_size ?? '-'}</span>
          </div>
          <div class="param-item">
            <span class="param-label">Neighbors/Iter</span>
            <span class="param-value">{flow.config.params.neighbors_per_iter ?? '-'}</span>
          </div>
          <div class="param-item">
            <span class="param-label">Fitness %</span>
            <span class="param-value">{flow.config.params.fitness_percentile ?? '-'}</span>
          </div>
          <div class="param-item">
            <span class="param-label">Tier0 Only</span>
            <span class="param-value">{flow.config.params.optimize_tier0_only ? 'Yes' : 'No'}</span>
          </div>
          {#if flow.config.params.tier_config}
            <div class="param-item full-width">
              <span class="param-label">Tier Config</span>
              <span class="param-value mono">
                {flow.config.params.tier_config.map((t: any) => `${t[0] ?? 'rest'},${t[1]},${t[2]}`).join('; ')}
              </span>
            </div>
          {/if}
        </div>
      </section>
    {/if}

    <section class="section">
      <h2>Experiments ({flow.config.experiments.length})</h2>
      <div class="experiments-list">
        {#each flow.config.experiments as exp, i}
          <div class="experiment-item">
            <div class="exp-order">{i + 1}</div>
            <div class="exp-content">
              <div class="exp-name">{exp.name}</div>
              <div class="exp-meta">
                <span class="exp-type">{exp.experiment_type.toUpperCase()}</span>
                {#if exp.optimize_bits}
                  <span class="exp-tag">Bits</span>
                {/if}
                {#if exp.optimize_neurons}
                  <span class="exp-tag">Neurons</span>
                {/if}
                {#if exp.optimize_connections}
                  <span class="exp-tag">Connections</span>
                {/if}
              </div>
            </div>
          </div>
        {/each}
      </div>
    </section>

    {#if experiments.length > 0}
      <section class="section">
        <h2>Completed Runs</h2>
        <div class="runs-table">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Started</th>
                <th>Ended</th>
              </tr>
            </thead>
            <tbody>
              {#each experiments as exp}
                <tr>
                  <td>{exp.name}</td>
                  <td>
                    <span class="status-dot" style="background: {getStatusColor(exp.status)}"></span>
                    {exp.status}
                  </td>
                  <td>{formatDate(exp.started_at)}</td>
                  <td>{formatDate(exp.ended_at)}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </section>
    {/if}
  {/if}
</div>

<style>
  .loading, .error {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--text-secondary);
  }

  .error {
    color: var(--accent-red);
  }

  .flow-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1.5rem;
    padding-top: 2rem;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .header-actions {
    display: flex;
    gap: 0.5rem;
  }

  .back-link {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 0.875rem;
  }

  .back-link:hover {
    color: var(--text-primary);
  }

  h1 {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }

  .status-badge {
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    color: white;
    text-transform: capitalize;
  }

  .description {
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
  }

  .info-cards {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 2rem;
  }

  .info-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .info-label {
    font-size: 0.75rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
  }

  .info-value {
    font-size: 0.875rem;
    color: var(--text-primary);
  }

  .section {
    margin-bottom: 2rem;
  }

  h2 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
  }

  /* Edit Form Styles */
  .edit-section {
    background: var(--bg-secondary);
    border: 1px solid var(--accent-blue);
    border-radius: 8px;
    padding: 1.5rem;
  }

  .edit-form {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .form-row {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
  }

  .form-group {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
  }

  .form-group.full-width {
    grid-column: 1 / -1;
  }

  .form-group.checkbox-group {
    flex-direction: row;
    align-items: center;
    gap: 0.5rem;
  }

  .form-group.checkbox-group label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
  }

  label {
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--text-primary);
  }

  input[type="number"],
  input[type="text"],
  select {
    padding: 0.5rem 0.75rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 0.875rem;
  }

  input[type="number"]:focus,
  input[type="text"]:focus,
  select:focus {
    outline: none;
    border-color: var(--accent-blue);
  }

  .form-hint {
    font-size: 0.75rem;
    color: var(--text-tertiary);
  }

  .form-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
  }

  .btn {
    padding: 0.5rem 1rem;
    border-radius: 4px;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    border: none;
    transition: background 0.15s;
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn-primary {
    background: var(--accent-blue);
    color: white;
  }

  .btn-primary:hover:not(:disabled) {
    background: #2563eb;
  }

  .btn-secondary {
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border);
  }

  .btn-secondary:hover:not(:disabled) {
    background: var(--border);
  }

  /* Parameters Grid */
  .params-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
  }

  .param-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .param-item.full-width {
    grid-column: 1 / -1;
  }

  .param-label {
    font-size: 0.75rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
  }

  .param-value {
    font-size: 0.875rem;
    color: var(--text-primary);
  }

  .param-value.mono {
    font-family: monospace;
    font-size: 0.8125rem;
  }

  /* Experiments List */
  .experiments-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .experiment-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
  }

  .exp-order {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--bg-tertiary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
  }

  .exp-content {
    flex: 1;
  }

  .exp-name {
    font-weight: 500;
    color: var(--text-primary);
  }

  .exp-meta {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.25rem;
  }

  .exp-type {
    font-size: 0.75rem;
    color: var(--accent-blue);
    font-weight: 600;
  }

  .exp-tag {
    font-size: 0.75rem;
    color: var(--text-tertiary);
    background: var(--bg-tertiary);
    padding: 0.125rem 0.375rem;
    border-radius: 3px;
  }

  /* Runs Table */
  .runs-table {
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

  .status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 0.5rem;
  }

  @media (max-width: 768px) {
    .form-row {
      grid-template-columns: 1fr;
    }

    .params-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }
</style>
