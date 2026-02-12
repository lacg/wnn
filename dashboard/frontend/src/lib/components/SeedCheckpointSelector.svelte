<script lang="ts">
	import { createEventDispatcher, onMount } from 'svelte';
	import type { Flow, Experiment, Checkpoint } from '$lib/types';

	export let value: number | null = null;

	const dispatch = createEventDispatcher<{ change: number | null }>();

	let flows: Flow[] = [];
	let experiments: Experiment[] = [];

	let selectedFlowId: number | null = null;
	let selectedExperimentId: number | null = null;
	let resolvedCheckpoint: Checkpoint | null = null;

	let loadingFlows = false;
	let loadingExperiments = false;
	let loadingCheckpoint = false;
	let checkpointError: string | null = null;

	onMount(loadFlows);

	async function loadFlows() {
		loadingFlows = true;
		try {
			const res = await fetch('/api/flows');
			if (res.ok) {
				const data = await res.json();
				flows = Array.isArray(data) ? data : [];
				// Sort by id descending (most recent first)
				flows.sort((a, b) => b.id - a.id);
			}
		} catch (e) {
			console.error('Failed to load flows:', e);
		} finally {
			loadingFlows = false;
		}
	}

	async function onFlowChange() {
		// Reset downstream
		experiments = [];
		selectedExperimentId = null;
		resolvedCheckpoint = null;
		checkpointError = null;
		setValue(null);

		if (!selectedFlowId) return;

		loadingExperiments = true;
		try {
			const res = await fetch(`/api/flows/${selectedFlowId}/experiments`);
			if (res.ok) {
				const data = await res.json();
				experiments = Array.isArray(data) ? data : [];
			}
		} catch (e) {
			console.error('Failed to load experiments:', e);
		} finally {
			loadingExperiments = false;
		}
	}

	async function onExperimentChange() {
		// Reset checkpoint
		resolvedCheckpoint = null;
		checkpointError = null;
		setValue(null);

		if (!selectedExperimentId) return;

		loadingCheckpoint = true;
		try {
			const res = await fetch(`/api/checkpoints?experiment_id=${selectedExperimentId}&is_final=true&limit=10`);
			if (res.ok) {
				const checkpoints: Checkpoint[] = await res.json();
				if (checkpoints.length > 0) {
					// Pick the final checkpoint (or most recent)
					resolvedCheckpoint = checkpoints[0];
					setValue(resolvedCheckpoint.id);
				} else {
					checkpointError = 'No final checkpoint found for this experiment';
				}
			}
		} catch (e) {
			checkpointError = 'Failed to load checkpoints';
			console.error(e);
		} finally {
			loadingCheckpoint = false;
		}
	}

	function clear() {
		selectedFlowId = null;
		selectedExperimentId = null;
		experiments = [];
		resolvedCheckpoint = null;
		checkpointError = null;
		setValue(null);
	}

	function setValue(id: number | null) {
		value = id;
		dispatch('change', id);
	}

	function formatStatus(status: string): string {
		return status.charAt(0).toUpperCase() + status.slice(1);
	}

	function formatCE(ce: number | null): string {
		if (ce === null) return '';
		return `CE: ${ce.toFixed(4)}`;
	}

	function formatAcc(acc: number | null): string {
		if (acc === null) return '';
		return `Acc: ${(acc * 100).toFixed(2)}%`;
	}
</script>

<div class="seed-selector">
	<div class="selector-row">
		<div class="selector-field">
			<label for="seed-flow">Flow</label>
			<select id="seed-flow" bind:value={selectedFlowId} on:change={onFlowChange} disabled={loadingFlows}>
				<option value={null}>Select a flow...</option>
				{#each flows as f}
					<option value={f.id}>
						{f.name} ({f.status})
					</option>
				{/each}
			</select>
			{#if loadingFlows}
				<span class="loading-hint">Loading flows...</span>
			{/if}
		</div>

		<div class="selector-field">
			<label for="seed-experiment">Experiment</label>
			<select
				id="seed-experiment"
				bind:value={selectedExperimentId}
				on:change={onExperimentChange}
				disabled={!selectedFlowId || loadingExperiments}
			>
				<option value={null}>Select an experiment...</option>
				{#each experiments as exp}
					<option value={exp.id}>
						{exp.name}
						{#if exp.best_ce !== null} ({formatCE(exp.best_ce)}){/if}
					</option>
				{/each}
			</select>
			{#if loadingExperiments}
				<span class="loading-hint">Loading experiments...</span>
			{/if}
		</div>
	</div>

	{#if resolvedCheckpoint}
		<div class="resolved-checkpoint">
			<span class="checkpoint-info">
				Seed: <strong>{resolvedCheckpoint.name}</strong>
				{#if resolvedCheckpoint.best_ce !== null}
					<span class="checkpoint-metric">{formatCE(resolvedCheckpoint.best_ce)}</span>
				{/if}
				{#if resolvedCheckpoint.best_accuracy !== null}
					<span class="checkpoint-metric">{formatAcc(resolvedCheckpoint.best_accuracy)}</span>
				{/if}
			</span>
			<button class="btn-clear" on:click={clear}>Clear</button>
		</div>
	{:else if loadingCheckpoint}
		<div class="resolved-checkpoint muted">Resolving checkpoint...</div>
	{:else if checkpointError}
		<div class="resolved-checkpoint error">{checkpointError}</div>
	{/if}
</div>

<style>
	.seed-selector {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
	}

	.selector-row {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1rem;
	}

	.selector-field {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.selector-field label {
		font-size: 1rem;
		font-weight: 500;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--text-secondary);
	}

	.selector-field select {
		width: 100%;
		padding: 0.5rem 0.75rem;
		border: 1px solid var(--border);
		border-radius: 6px;
		background: var(--bg-primary);
		color: var(--text-primary);
		font-size: 1rem;
		font-family: inherit;
	}

	.selector-field select:focus {
		outline: none;
		border-color: var(--accent-blue);
	}

	.selector-field select:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}

	.loading-hint {
		font-size: 1rem;
		color: var(--text-secondary);
		font-style: italic;
	}

	.resolved-checkpoint {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.5rem 0.75rem;
		background: rgba(16, 185, 129, 0.08);
		border: 1px solid rgba(16, 185, 129, 0.2);
		border-radius: 6px;
		font-size: 1rem;
	}

	.resolved-checkpoint.muted {
		background: var(--bg-tertiary);
		border-color: var(--border);
		color: var(--text-secondary);
		font-style: italic;
	}

	.resolved-checkpoint.error {
		background: rgba(239, 68, 68, 0.08);
		border-color: rgba(239, 68, 68, 0.2);
		color: var(--accent-red);
	}

	.checkpoint-info {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		flex-wrap: wrap;
	}

	.checkpoint-metric {
		font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
		font-size: 1rem;
		color: var(--text-secondary);
	}

	.btn-clear {
		background: none;
		border: 1px solid var(--border);
		color: var(--text-secondary);
		cursor: pointer;
		padding: 0.25rem 0.5rem;
		border-radius: 4px;
		font-size: 1rem;
		flex-shrink: 0;
	}

	.btn-clear:hover {
		color: var(--accent-red);
		border-color: var(--accent-red);
	}
</style>
