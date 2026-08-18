<script lang="ts">
	import { createEventDispatcher } from 'svelte'
	import type { Experiment, Flow, GatingRun } from '$lib/types'
	import { getStatusColor } from '$lib/statusColors'

	export let experiment: Experiment
	export let flow: Flow | null = null
	export let latestGatingRun: GatingRun | null = null
	export let hasActiveGating: boolean = false
	export let hasCompletedGating: boolean = false
	export let gatingLoading: boolean = false

	const dispatch = createEventDispatcher<{ runGating: void }>()
</script>

<div class="experiment-header">
	<div class="header-left">
		{#if experiment.flow_id}
			<a href="/flows/{experiment.flow_id}" class="back-link">&larr; Back to Flow</a>
		{:else}
			<a href="/flows" class="back-link">&larr; Flows</a>
		{/if}
		{#if flow}
			<span class="flow-name-label"><a href="/flows/{flow.id}">{flow.name}</a> /</span>
		{/if}
		<h1>{experiment.name}</h1>
		<span class="status-badge" style="background: {getStatusColor(experiment.status)}">
			{experiment.status}
		</span>
	</div>
	<div class="header-right">
		{#if experiment.status === 'completed' && !hasActiveGating}
			<button class="btn-secondary" on:click={() => dispatch('runGating')} disabled={gatingLoading}>
				{gatingLoading ? '⏳ Starting...' : hasCompletedGating ? '🔄 Re-run Gating' : '🎯 Run Gating Analysis'}
			</button>
		{:else if hasActiveGating}
			<span class="gating-status running">⏳ Gating {latestGatingRun?.status}...</span>
		{/if}
	</div>
</div>

<style>
  .experiment-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1rem;
    padding-top: 1rem;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
  }

  .back-link {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 1rem;
  }

  .back-link:hover {
    color: var(--text-primary);
  }

  .flow-name-label {
    font-size: 1.125rem;
    color: var(--text-secondary);
    font-weight: 500;
  }

  .flow-name-label a {
    color: var(--text-secondary);
    text-decoration: none;
  }

  .flow-name-label a:hover {
    color: var(--text-primary);
  }

  h1 {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }

  .status-badge {
    font-size: 1rem;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    color: white;
    text-transform: capitalize;
  }

  /* Header actions */
  .header-right {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .btn-secondary {
    padding: 0.5rem 1rem;
    font-size: 1rem;
    border: 1px solid var(--accent-blue);
    border-radius: 0.375rem;
    background: transparent;
    color: var(--accent-blue);
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn-secondary:hover:not(:disabled) {
    background: var(--accent-blue);
    color: white;
  }

  .btn-secondary:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .gating-status {
    font-size: 1rem;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
  }

  .gating-status.running {
    background: rgba(59, 130, 246, 0.15);
    color: var(--accent-blue);
  }
</style>
