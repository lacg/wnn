<script lang="ts">
	import type { ValidationProgressionPoint } from './types'

	export let points: ValidationProgressionPoint[] = []

	// Local drill-down state (phase × decode mode × genome for the confusion matrix).
	let pointChoice: number = 0
	let modeChoice: 'argmax' | 'margin_fixed0' | 'margin_train_cal' | 'margin_val_cal' = 'argmax'
	let confusionGenomeChoice: string = 'best_fitness'

	const GENOME_COLS = [
		{ key: 'best_f1', label: 'Best F1', cls: 'best-ce-col' },
		{ key: 'best_fpr', label: 'Best FPR', cls: 'best-acc-col' },
		{ key: 'best_acc', label: 'Best Acc', cls: '' },
		{ key: 'best_ce', label: 'Best CE', cls: '' },
		{ key: 'best_fitness', label: 'Best Fitness', cls: 'best-fit-col' },
	]

	const DECODE_MODES = [
		{ key: 'argmax', label: 'argmax' },
		{ key: 'margin_fixed0', label: 'margin τ=0' },
		{ key: 'margin_train_cal', label: 'margin train-cal' },
		{ key: 'margin_val_cal', label: 'margin val-cal' },
	] as const

	type McPerClass = Record<string, { f1: number, precision: number, recall: number, support: number }>

	function mcLookup(summary: any, mode: string): McPerClass | null
	{
		return summary?.threshold_metadata?.[mode]?.per_class ?? null
	}
	function mcConfusion(summary: any, mode: string): number[][] | null
	{
		return summary?.threshold_metadata?.[mode]?.confusion ?? null
	}
	function fmtPct(v: number | null | undefined): string
	{
		return v != null ? (v * 100).toFixed(2) + '%' : '—'
	}

	/**
	 * The confusion matrix rows follow label-INDEX order (benign class 0 first,
	 * then attacks), while per_class serializes name-sorted. Recover the index
	 * order by matching per-class supports against confusion row sums; fall
	 * back through plausible orders if the benign-first guess doesn't match.
	 */
	function confusionOrder(pc: McPerClass, confusion: number[][]): string[] | null
	{
		const names = Object.keys(pc)
		if (names.length !== confusion.length) return null
		const benign = names.find(n => /^(normal|benign)$/i.test(n))
		const candidates: string[][] = []
		if (benign) candidates.push([benign, ...names.filter(n => n !== benign).sort()])
		candidates.push([...names].sort())
		candidates.push(names)
		const rowSums = confusion.map(r => r.reduce((a, b) => a + b, 0))
		for (const order of candidates)
		{
			if (order.every((n, i) => pc[n].support === rowSums[i])) return order
		}
		return null
	}

	/** Row-normalized shade: diagonal cells green, off-diagonal red, by row fraction. */
	function cellShade(count: number, rowTotal: number, isDiagonal: boolean): string
	{
		const frac = rowTotal > 0 ? count / rowTotal : 0
		const alpha = Math.min(0.85, frac * 0.85)
		return isDiagonal
			? `rgba(34, 197, 94, ${alpha.toFixed(3)})`
			: `rgba(239, 68, 68, ${alpha.toFixed(3)})`
	}

	$: mcPoints = points.map(p => ({
		label: p.label,
		summaries: Object.fromEntries(
			GENOME_COLS.map(g => [g.key, p.summaries.find(s => s.genomeType === g.key)])
		) as Record<string, any>,
	}))
	$: mcAvailablePoints = mcPoints.filter(p =>
		GENOME_COLS.some(g => mcLookup(p.summaries[g.key], modeChoice))
	)
</script>

{#if mcAvailablePoints.length > 0}
	{@const _selectedIdx = Math.min(Math.max(0, pointChoice), mcAvailablePoints.length - 1)}
	{@const selectedPoint = mcAvailablePoints[_selectedIdx]}
	{@const byGenome = Object.fromEntries(
		GENOME_COLS.map(g => [g.key, mcLookup(selectedPoint.summaries[g.key], modeChoice)])
	)}
	{@const classes = GENOME_COLS.map(g => byGenome[g.key]).find(Boolean)}
	{@const confSummary = selectedPoint.summaries[confusionGenomeChoice]}
	{@const confusion = mcConfusion(confSummary, modeChoice)}
	{@const confPc = mcLookup(confSummary, modeChoice)}
	{@const confOrder = confusion && confPc ? confusionOrder(confPc, confusion) : null}
	<details class="per-class-section" open>
		<summary class="per-class-summary">
			Per-class breakdown
			{#if classes}
				({Object.keys(classes).length} classes)
			{/if}
			<span class="pc-control-sep">—</span>
			<span class="pc-control-label">Phase:</span>
			<select bind:value={pointChoice} on:click|stopPropagation class="pc-select">
				{#each mcAvailablePoints as p, i}
					<option value={i}>{p.label}</option>
				{/each}
			</select>
			<span class="pc-control-label">at</span>
			<select bind:value={modeChoice} on:click|stopPropagation class="pc-select">
				{#each DECODE_MODES as mode}
					<option value={mode.key}>{mode.label}</option>
				{/each}
			</select>
			<span class="pc-control-label">decode</span>
		</summary>
		{#if classes}
			<table class="per-class-table">
				<thead>
					<tr>
						<th rowspan="2" class="pc-class-col">Class</th>
						<th rowspan="2" class="pc-count-col">Support</th>
						{#each GENOME_COLS as g}
							<th colspan="2" class={g.cls}>{g.label}</th>
						{/each}
					</tr>
					<tr>
						{#each GENOME_COLS as g}
							<th class={g.cls} title="Per-class F1">F1</th>
							<th class={g.cls} title="Per-class recall (detection rate)">Det</th>
						{/each}
					</tr>
				</thead>
				<tbody>
					{#each Object.entries(classes) as [clsName, entry]}
						<tr>
							<td class="pc-class-col">{clsName}</td>
							<td class="mono pc-count-col">{entry.support.toLocaleString()}</td>
							{#each GENOME_COLS as g}
								{@const e = byGenome[g.key]?.[clsName]}
								<td class="mono {g.cls}">{fmtPct(e?.f1)}</td>
								<td class="mono {g.cls}">{fmtPct(e?.recall)}</td>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>

			<div class="confusion-block">
				<div class="confusion-header">
					<span class="confusion-title">Confusion matrix</span>
					<span class="pc-control-label">for</span>
					<select bind:value={confusionGenomeChoice} class="pc-select">
						{#each GENOME_COLS as g}
							<option value={g.key}>{g.label}</option>
						{/each}
					</select>
					<span class="pc-control-label">(rows = true class, cols = predicted)</span>
				</div>
				{#if confusion && confOrder}
					<div class="confusion-scroll">
						<table class="confusion-table">
							<thead>
								<tr>
									<th class="conf-corner">true ╲ pred</th>
									{#each confOrder as name}
										<th class="conf-col-label">{name}</th>
									{/each}
								</tr>
							</thead>
							<tbody>
								{#each confOrder as trueName, i}
									{@const row = confusion[i]}
									{@const rowTotal = row.reduce((a, b) => a + b, 0)}
									<tr>
										<td class="conf-row-label">{trueName}</td>
										{#each row as count, j}
											<td
												class="mono conf-cell"
												style:background={cellShade(count, rowTotal, i === j)}
												title="{trueName} → {confOrder[j]}: {count.toLocaleString()} ({rowTotal > 0 ? ((count / rowTotal) * 100).toFixed(1) : '0.0'}% of true {trueName})"
											>{count.toLocaleString()}</td>
										{/each}
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				{:else if confusion}
					<p class="pc-empty">Confusion matrix present but class order could not be verified against per-class supports.</p>
				{:else}
					<p class="pc-empty">No confusion matrix for this genome × decode mode.</p>
				{/if}
			</div>
		{:else}
			<p class="pc-empty">No per-class data for this phase × decode mode combination.</p>
		{/if}
	</details>
{/if}

<style>
	.per-class-section {
		margin-top: 1rem;
		padding: 0.75rem 1rem;
		border-radius: 8px;
		background: rgba(255, 255, 255, 0.02);
		border: 1px solid var(--glass-border);
	}
	.per-class-section[open] {
		background: rgba(255, 255, 255, 0.03);
	}
	.per-class-summary {
		font-weight: 600;
		cursor: pointer;
		padding: 0.25rem 0;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.4rem;
	}
	.pc-control-sep {
		opacity: 0.5;
		margin: 0 0.25rem;
	}
	.pc-control-label {
		opacity: 0.65;
		font-weight: 400;
	}
	.pc-select {
		font-size: 1rem;
		padding: 0.15rem 0.4rem;
		cursor: pointer;
	}
	.pc-empty {
		opacity: 0.65;
		font-style: italic;
		margin: 0.75rem 0;
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
	}

	td {
		font-size: 1rem;
		font-family: monospace;
	}

	.per-class-table {
		border-collapse: collapse;
		margin: 0.5rem 0;
		font-size: 1rem;
		width: 100%;
	}
	.per-class-table th,
	.per-class-table td {
		padding: 0.25rem 0.75rem;
		text-align: right;
		border-bottom: 1px solid #333;
	}
	.per-class-table thead th {
		border-bottom: 1px solid #444;
		text-align: center;
		font-weight: 600;
	}
	.per-class-table .pc-class-col {
		text-align: left;
	}
	.per-class-table .pc-count-col {
		text-align: right;
	}
	.per-class-table .best-ce-col { background: rgba(59, 130, 246, 0.08); }
	.per-class-table .best-acc-col { background: rgba(34, 197, 94, 0.08); }
	.per-class-table .best-fit-col { background: rgba(155, 89, 182, 0.08); }

	.confusion-block {
		margin-top: 1rem;
	}
	.confusion-header {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.4rem;
		margin-bottom: 0.5rem;
	}
	.confusion-title {
		font-weight: 600;
		color: var(--text-primary);
	}
	.confusion-scroll {
		overflow-x: auto;
	}
	.confusion-table {
		border-collapse: collapse;
		font-size: 1rem;
	}
	.confusion-table th,
	.confusion-table td {
		padding: 0.25rem 0.5rem;
		border: 1px solid rgba(255, 255, 255, 0.08);
		text-align: right;
		white-space: nowrap;
	}
	.conf-corner {
		text-align: left;
		text-transform: none;
		font-style: italic;
		font-weight: 400;
	}
	.conf-col-label {
		text-transform: none;
		max-width: 7rem;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.conf-row-label {
		text-align: left;
		font-weight: 600;
		color: var(--text-primary);
		background: rgba(51, 65, 85, 0.4);
	}
	.conf-cell {
		color: #e2e8f0;
	}
</style>
