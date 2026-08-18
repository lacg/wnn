<script lang="ts">
	import type { Experiment, Flow } from '$lib/types'
	import { formatCE, formatDuration } from '$lib/format'
	import { formatAcc, formatF1, formatFPR } from './metricFormat'

	export let experiment: Experiment
	export let flow: Flow | null = null
	export let isIDS: boolean = false
	export let isController: boolean = false
	export let isGridSearch: boolean = false
	export let bestCE: number = Infinity
	export let bestAcc: number | null = null
	export let bestF1: number | null = null
	export let bestFpr: number | null = null
	export let bestAttitudeDeg: number | null = null
	export let ceImprovement: number | null = null
	export let accImprovement: number | null = null
	export let f1Improvement: number | null = null
	export let fprImprovement: number | null = null
	export let iterationsCount: number = 0
	export let gridConfigsCount: number = 0
	export let maxIterations: number | null = null
	export let avgSecsPerIter: number | null = null
</script>

<div class="info-cards">
	{#if isIDS}
		<div class="info-card">
			<span class="info-label">Best F1-macro</span>
			<span class="info-value best">{formatF1(bestF1)}</span>
			{#if f1Improvement !== null}
				<span class="info-delta" class:improved={f1Improvement > 0} class:worsened={f1Improvement < 0}>
					{f1Improvement > 0 ? '↑' : '↓'}{Math.abs(f1Improvement).toFixed(2)}%
				</span>
			{/if}
		</div>
		<div class="info-card">
			<span class="info-label">Best FPR</span>
			<span class="info-value">{formatFPR(bestFpr)}</span>
			{#if fprImprovement !== null}
				<span class="info-delta" class:improved={fprImprovement > 0} class:worsened={fprImprovement < 0}>
					{fprImprovement > 0 ? '↓' : '↑'}{Math.abs(fprImprovement).toFixed(2)}%
				</span>
			{/if}
		</div>
		<div class="info-card">
			<span class="info-label">Best Acc</span>
			<span class="info-value">{formatAcc(bestAcc)}</span>
			{#if accImprovement !== null}
				<span class="info-delta" class:improved={accImprovement > 0} class:worsened={accImprovement < 0}>
					{accImprovement > 0 ? '↑' : '↓'}{Math.abs(accImprovement).toFixed(2)}%
				</span>
			{/if}
		</div>
	{:else if isController}
		<div class="info-card">
			<span class="info-label">Best Reward</span>
			<span class="info-value best">{bestCE !== null && bestCE !== undefined ? (-bestCE).toFixed(2) : '—'}</span>
			<span class="info-subvalue">ce = −reward</span>
		</div>
		<div class="info-card">
			<span class="info-label">Stable %</span>
			<span class="info-value">{formatAcc(bestAcc)}</span>
		</div>
		<div class="info-card">
			<span class="info-label">Best Attitude°</span>
			<span class="info-value">{bestAttitudeDeg !== null ? bestAttitudeDeg.toFixed(2) + '°' : '—'}</span>
		</div>
	{:else}
		<div class="info-card">
			<span class="info-label">Best CE</span>
			<span class="info-value best">{formatCE(bestCE)}</span>
			{#if ceImprovement !== null}
				<span class="info-delta" class:improved={ceImprovement > 0} class:worsened={ceImprovement < 0}>
					{ceImprovement > 0 ? '↓' : '↑'}{Math.abs(ceImprovement).toFixed(2)}%
				</span>
			{/if}
		</div>
		<div class="info-card">
			<span class="info-label">Best Acc</span>
			<span class="info-value">{formatAcc(bestAcc)}</span>
			{#if accImprovement !== null}
				<span class="info-delta" class:improved={accImprovement > 0} class:worsened={accImprovement < 0}>
					{accImprovement > 0 ? '↑' : '↓'}{Math.abs(accImprovement).toFixed(2)}%
				</span>
			{/if}
		</div>
	{/if}
	<div class="info-card">
		<span class="info-label">{isGridSearch ? 'Configs Tested' : 'Iterations'}</span>
		<span class="info-value">{isGridSearch ? gridConfigsCount : iterationsCount}{#if !isGridSearch && maxIterations}/{maxIterations}{/if}</span>
		{#if avgSecsPerIter !== null && !isGridSearch}
			<span class="info-subvalue">{avgSecsPerIter.toFixed(1)}s/iter</span>
		{/if}
	</div>
	<div class="info-card">
		<span class="info-label">Duration</span>
		<span class="info-value">{formatDuration(experiment.started_at, experiment.ended_at)}</span>
		{#if flow}
			<span class="info-subvalue">Flow: {formatDuration(flow.started_at, flow.completed_at)}</span>
		{/if}
	</div>
</div>

<style>
  /* Info Cards */
  .info-cards {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
  }

  @media (max-width: 768px) {
    .info-cards {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  .info-card {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: 0.5rem;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .info-label {
    font-size: 1rem;
    color: var(--text-primary);
  }

  .info-value {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
    font-family: monospace;
  }

  .info-value.best {
    color: var(--accent-green);
  }

  .info-delta {
    font-size: 1rem;
    font-family: monospace;
  }

  .info-delta.improved {
    color: var(--accent-green);
  }

  .info-delta.worsened {
    color: var(--accent-red);
  }

  .info-subvalue {
    font-size: 1rem;
    color: var(--text-primary);
    font-family: monospace;
  }
</style>
