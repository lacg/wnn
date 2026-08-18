<script lang="ts">
	import type { Flow } from '$lib/types'
	import { formatDate } from '$lib/dateFormat'

	export let flow: Flow
	export let isIDS: boolean = false
	export let isController: boolean = false

	// Display labels for known IDS datasets; unknown ids fall through to the
  // raw id from the flow config (see `info-card` for the Dataset row).
	const DATASET_LABELS: Record<string, string> = {
		'unsw-nb15': 'UNSW-NB15',
		'cicids2017': 'CICIDS2017',
		'ciciot2023': 'CIC-IoT-2023 (1.3M)',
		'ciciot2023_full': 'CIC-IoT-2023 (46M)',
	}
</script>

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
	{#if isIDS}
		<div class="info-card">
			<span class="info-label">Task</span>
			<span class="info-value">IDS {flow.config.params?.ids_classification ?? 'binary'}</span>
		</div>
		<div class="info-card">
			<span class="info-label">Dataset</span>
			<span class="info-value">{DATASET_LABELS[flow.config.params?.ids_dataset] ?? flow.config.params?.ids_dataset ?? 'UNSW-NB15'} ({flow.config.params?.ids_split ?? 'standard'})</span>
		</div>
		<div class="info-card">
			<span class="info-label">Encoding</span>
			<span class="info-value">{flow.config.params?.ids_n_bits ?? 8}-bit thermometer</span>
		</div>
		<div class="info-card">
			<span class="info-label">Bits</span>
			<span class="info-value">{flow.config.params?.min_bits ?? 4}–{flow.config.params?.max_bits ?? 24}</span>
		</div>
		<div class="info-card">
			<span class="info-label">Neurons</span>
			<span class="info-value">{flow.config.params?.min_neurons ?? 5}–{flow.config.params?.max_neurons ?? 300}</span>
		</div>
	{:else if isController}
		<div class="info-card">
			<span class="info-label">Task</span>
			<span class="info-value">Controller (attitude)</span>
		</div>
		<div class="info-card">
			<span class="info-label">Motors</span>
			<span class="info-value">{flow.config.params?.controller_num_motors ?? 4} × {flow.config.params?.controller_levels_per_motor ?? 16} lvls</span>
		</div>
		<div class="info-card">
			<span class="info-label">State Neurons</span>
			<span class="info-value">{flow.config.params?.controller_state_neurons ?? 4}</span>
		</div>
		<div class="info-card">
			<span class="info-label">Episodes</span>
			<span class="info-value">{flow.config.params?.controller_eval_episodes ?? 20} × {flow.config.params?.controller_steps ?? 1500} steps</span>
		</div>
	{/if}
</div>

<style>
  .info-cards {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 2rem;
  }

  .info-card {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .info-label {
    font-size: 1rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
  }

  .info-value {
    font-size: 1rem;
    color: var(--text-primary);
  }
</style>
