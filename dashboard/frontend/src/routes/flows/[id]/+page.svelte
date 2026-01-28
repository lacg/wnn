<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import type { Flow, Experiment } from '$lib/types';

  let flow: Flow | null = null;
  let experiments: Experiment[] = [];
  let loading = true;
  let error: string | null = null;

  $: flowId = $page.params.id;

  onMount(async () => {
    try {
      const [flowRes, expsRes] = await Promise.all([
        fetch(`/api/flows/${flowId}`),
        fetch(`/api/flows/${flowId}/experiments`)
      ]);

      if (!flowRes.ok) throw new Error('Flow not found');

      flow = await flowRes.json();
      experiments = await expsRes.json();
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
</style>
