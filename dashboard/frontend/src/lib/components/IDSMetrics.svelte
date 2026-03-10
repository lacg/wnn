<script lang="ts">
	import type { IDSMetrics } from '$lib/types';

	export let metrics: IDSMetrics;
	export let title: string = 'IDS Metrics';

	$: perClassF1 = metrics.per_class_f1 ?? {};
	$: hasPerClass = Object.keys(perClassF1).length > 0;
	$: classNames = Object.keys(perClassF1);

	function fmt(v: number | undefined, decimals = 2): string {
		if (v === undefined || v === null) return '—';
		return (v * 100).toFixed(decimals) + '%';
	}

	function f1Val(cls: string): number {
		return perClassF1[cls] ?? 0;
	}
</script>

<div class="ids-metrics">
	{#if title}
		<h3>{title}</h3>
	{/if}

	<div class="metric-cards">
		<div class="metric-card">
			<span class="metric-label">Accuracy</span>
			<span class="metric-value">{fmt(metrics.accuracy)}</span>
		</div>
		<div class="metric-card">
			<span class="metric-label">F1-macro</span>
			<span class="metric-value">{fmt(metrics.f1_macro)}</span>
		</div>
		{#if metrics.fpr !== undefined}
			<div class="metric-card">
				<span class="metric-label">FPR</span>
				<span class="metric-value">{fmt(metrics.fpr)}</span>
			</div>
		{/if}
	</div>

	{#if hasPerClass}
		<details class="per-class-details">
			<summary>Per-class breakdown ({classNames.length} classes)</summary>
			<table class="per-class-table">
				<thead>
					<tr>
						<th>Class</th>
						<th>Precision</th>
						<th>Recall</th>
						<th>F1</th>
					</tr>
				</thead>
				<tbody>
					{#each classNames as cls}
						<tr>
							<td>{cls}</td>
							<td>{fmt(metrics.per_class_precision?.[cls])}</td>
							<td>{fmt(metrics.per_class_recall?.[cls])}</td>
							<td class:good={f1Val(cls) >= 0.5}
								class:poor={f1Val(cls) < 0.1}>
								{fmt(f1Val(cls))}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</details>
	{/if}
</div>

<style>
	.ids-metrics {
		margin: 1rem 0;
	}

	h3 {
		font-size: 1rem;
		font-weight: 600;
		margin: 0 0 0.75rem 0;
		color: var(--text-primary);
	}

	.metric-cards {
		display: flex;
		gap: 1rem;
		flex-wrap: wrap;
	}

	.metric-card {
		background: var(--bg-secondary, #1a1a2e);
		border: 1px solid var(--border-color, #333);
		border-radius: 8px;
		padding: 0.75rem 1.25rem;
		display: flex;
		flex-direction: column;
		align-items: center;
		min-width: 100px;
	}

	.metric-label {
		font-size: 1rem;
		color: var(--text-secondary, #888);
		text-transform: uppercase;
		letter-spacing: 0.05em;
	}

	.metric-value {
		font-size: 1.25rem;
		font-weight: 700;
		color: var(--text-primary, #fff);
		font-variant-numeric: tabular-nums;
	}

	.per-class-details {
		margin-top: 1rem;
	}

	.per-class-details summary {
		cursor: pointer;
		font-size: 1rem;
		color: var(--text-secondary, #888);
		padding: 0.25rem 0;
	}

	.per-class-table {
		width: 100%;
		border-collapse: collapse;
		margin-top: 0.5rem;
		font-size: 1rem;
	}

	.per-class-table th,
	.per-class-table td {
		padding: 0.4rem 0.75rem;
		text-align: right;
		border-bottom: 1px solid var(--border-color, #333);
	}

	.per-class-table th {
		font-weight: 600;
		color: var(--text-secondary, #888);
		text-align: right;
	}

	.per-class-table th:first-child,
	.per-class-table td:first-child {
		text-align: left;
	}

	.good {
		color: var(--accent-green, #4caf50);
	}

	.poor {
		color: var(--accent-red, #f44336);
	}
</style>
