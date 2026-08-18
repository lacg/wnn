<script lang="ts">
	import type { GatingRun } from '$lib/types'

	export let latestGatingRun: GatingRun | null = null
	export let gatingRunsCount: number = 0
	export let hasCompletedGating: boolean = false
</script>

{#if hasCompletedGating && latestGatingRun?.results}
	<div class="gating-section">
		<div class="gating-header">
			<span class="gating-title">🎯 Gating Analysis Results</span>
			<span class="gating-meta">
				{latestGatingRun.genomes_tested ?? latestGatingRun.results.length} genomes tested
				{#if latestGatingRun.started_at && latestGatingRun.completed_at}
					{@const startMs = new Date(latestGatingRun.started_at).getTime()}
					{@const endMs = new Date(latestGatingRun.completed_at).getTime()}
					{@const durationSec = Math.round((endMs - startMs) / 1000)}
					{@const durationMin = Math.floor(durationSec / 60)}
					{@const durationRemSec = durationSec % 60}
					· Duration: {durationMin}m {durationRemSec}s
				{/if}
				{#if gatingRunsCount > 1}
					· Run #{latestGatingRun.id}
				{/if}
			</span>
		</div>
		<div class="gating-table-container">
			<table class="gating-table">
				<thead>
					<tr>
						<th>Genome</th>
						<th>CE (no gate)</th>
						<th>CE (gated)</th>
						<th>Δ CE</th>
						<th>Acc (no gate)</th>
						<th>Acc (gated)</th>
						<th>Δ Acc</th>
					</tr>
				</thead>
				<tbody>
					{#each latestGatingRun.results as result}
						{@const ceDelta = result.gated_ce - result.ce}
						{@const accDelta = result.gated_acc - result.acc}
						<tr>
							<td class="genome-type">{result.genome_type.replace('_', ' ')}</td>
							<td class="mono">{result.ce.toFixed(4)}</td>
							<td class="mono">{result.gated_ce.toFixed(4)}</td>
							<td class="mono" class:delta-positive={ceDelta < 0} class:delta-negative={ceDelta > 0}>
								{ceDelta < 0 ? '↑' : ceDelta > 0 ? '↓' : ''}{Math.abs(ceDelta).toFixed(4)}
							</td>
							<td class="mono">{(result.acc * 100).toFixed(2)}%</td>
							<td class="mono">{(result.gated_acc * 100).toFixed(2)}%</td>
							<td class="mono" class:delta-positive={accDelta > 0} class:delta-negative={accDelta < 0}>
								{accDelta > 0 ? '↑' : accDelta < 0 ? '↓' : ''}{Math.abs(accDelta * 100).toFixed(2)}%
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
		{#if latestGatingRun.error}
			<div class="gating-error">
				Error: {latestGatingRun.error}
			</div>
		{/if}
	</div>
{:else if latestGatingRun?.status === 'failed'}
	<div class="gating-section">
		<div class="gating-header">
			<span class="gating-title">🎯 Gating Analysis</span>
			<span class="gating-meta">Run #{latestGatingRun.id}</span>
		</div>
		<div class="gating-error">
			Error: {latestGatingRun.error ?? 'Gating analysis failed'}
		</div>
	</div>
{/if}

<style>
  /* Base table styles (inherited from the page before extraction) */
  table {
    width: 100%;
    border-collapse: collapse;
  }

  th, td {
    padding: 0.5rem 0.625rem;
    text-align: left;
    border-bottom: 1px solid var(--glass-border);
  }

  th {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    background: rgba(51, 65, 85, 0.4);
    position: sticky;
    top: 0;
    z-index: 1;
  }

  td {
    font-size: 1rem;
    font-family: monospace;
  }

  tr:last-child td {
    border-bottom: none;
  }

  .delta-positive {
    color: var(--accent-green);
  }

  .delta-negative {
    color: var(--accent-red);
  }

  /* Gating Results Section */
  .gating-section {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: 0.5rem;
    padding: 1rem;
    margin-bottom: 1.5rem;
    border-left: 4px solid var(--accent-purple, #9b59b6);
  }

  .gating-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--glass-border);
  }

  .gating-title {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 1.1rem;
  }

  .gating-meta {
    font-size: 1rem;
    color: var(--text-secondary);
  }

  .gating-table-container {
    overflow-x: auto;
  }

  .gating-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 1rem;
  }

  .gating-table th {
    background: rgba(51, 65, 85, 0.4);
    padding: 0.5rem 0.75rem;
    text-align: center;
    font-weight: 600;
    color: var(--text-secondary);
    font-size: 1rem;
    text-transform: uppercase;
    border-bottom: 1px solid var(--glass-border);
  }

  .gating-table td {
    padding: 0.5rem 0.75rem;
    text-align: center;
    border-bottom: 1px solid var(--glass-border);
  }

  .gating-table tr:last-child td {
    border-bottom: none;
  }

  .gating-table .genome-type {
    text-transform: capitalize;
    font-weight: 500;
    color: var(--text-primary);
    text-align: left;
  }

  .gating-table .mono {
    font-family: monospace;
  }

  .gating-error {
    margin-top: 1rem;
    padding: 0.75rem;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid var(--accent-red);
    border-radius: 0.25rem;
    color: var(--accent-red);
    font-size: 1rem;
  }
</style>
