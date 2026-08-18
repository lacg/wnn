<script lang="ts">
	import type { ValidationProgressionPoint } from './types'
	import PerClassBreakdown from './PerClassBreakdown.svelte'
	import MulticlassBreakdown from './MulticlassBreakdown.svelte'

	export let points: ValidationProgressionPoint[] = []
	export let isIDS: boolean = false
	// Multiclass IDS flows carry decode-mode metadata (argmax/margin_*) instead
  // of binary threshold modes; f1/fpr cells then read macro-F1/benign-FPR
  // (the worker writes them as compat aliases on each mode).
	export let isMulticlass: boolean = false
	export let currentExperimentId: number
	// Per-class drill-down selection — bound through to the page so the choice
  // survives in-page navigation (state lives in the page).
	export let perClassPointChoice: number = 0
	export let perClassThresholdChoice: 'train_cal' | 'fixed_05' | 'val_cal' | 'platt' | 'beta' | 'empirical' | 'empirical_cumulative' = 'train_cal'

	// Headline row + sub-row mode keys per flow kind. Binary: fixed-0.5 headline
  // with calibration sub-rows. Multiclass: argmax headline with margin-decode
  // sub-rows.
	$: headlineKey = isMulticlass ? 'argmax' : 'fixed_05'
	$: headlineHint = isMulticlass ? 'argmax' : 'fixed 0.5'
	$: subModes = isMulticlass
		? [
			{ key: 'margin_fixed0', label: '┣ Margin τ=0', cls: 'threshold-train-row' },
			{ key: 'margin_train_cal', label: '┣ Margin train-cal', cls: 'threshold-platt-row' },
			{ key: 'margin_val_cal', label: '┣ Margin val-cal', cls: 'threshold-oracle-row' },
			{ key: 'argmax_platt', label: '┣ Argmax platt', cls: 'threshold-platt-row' },
			{ key: 'argmax_beta', label: '┗ Argmax beta', cls: 'threshold-beta-row' },
		]
		: [
			{ key: 'train_cal', label: '┣ Train-cal', cls: 'threshold-train-row' },
			{ key: 'platt', label: '┣ Platt', cls: 'threshold-platt-row' },
			{ key: 'beta', label: '┣ Beta', cls: 'threshold-beta-row' },
			{ key: 'empirical', label: '┣ Empirical', cls: 'threshold-empirical-row' },
			{ key: 'empirical_cumulative', label: '┣ Emp-cumul', cls: 'threshold-empirical-row' },
			{ key: 'val_cal', label: '┗ Oracle', cls: 'threshold-oracle-row' },
		]
	$: f1Label = isMulticlass ? 'mF1' : 'F1'
	$: fprLabel = isMulticlass ? 'bFPR' : 'FPR'
</script>

<div class="validation-section">
	<div class="validation-header">
		<span class="validation-title">📈 Validation Progression</span>
		<div class="validation-legend">
			{#if isIDS}
				<span class="legend-item"><span class="legend-marker best-ce"></span> Best {isMulticlass ? 'Macro-F1' : 'F1-macro'}</span>
				<span class="legend-item"><span class="legend-marker best-acc"></span> Best {isMulticlass ? 'Benign-FPR' : 'FPR'}</span>
				<span class="legend-item"><span class="legend-marker best-fitness"></span> Best Fitness</span>
			{:else}
				<span class="legend-item"><span class="legend-marker best-ce"></span> Best CE</span>
				<span class="legend-item"><span class="legend-marker best-acc"></span> Best Acc</span>
				<span class="legend-item"><span class="legend-marker best-fitness"></span> Best Fitness</span>
			{/if}
		</div>
	</div>
	<div class="validation-table-container">
		<table class="validation-table">
			<thead>
				{#if isIDS}
					<tr>
						<th rowspan="2">Phase</th>
						<th colspan="3">Best F1 Genome</th>
						<th colspan="3">Best FPR Genome</th>
						<th colspan="3">Best Acc Genome</th>
						<th colspan="3">Best CE Genome</th>
						<th colspan="3">Best Fitness Genome</th>
					</tr>
					<tr>
						<th>{f1Label}</th><th>{fprLabel}</th><th>Acc</th>
						<th>{f1Label}</th><th>{fprLabel}</th><th>Acc</th>
						<th>{f1Label}</th><th>{fprLabel}</th><th>Acc</th>
						<th>{f1Label}</th><th>{fprLabel}</th><th>Acc</th>
						<th>{f1Label}</th><th>{fprLabel}</th><th>Acc</th>
					</tr>
				{:else}
					<tr>
						<th rowspan="2">Phase</th>
						<th colspan="2">Best CE Genome</th>
						<th colspan="2">Best Acc Genome</th>
						<th colspan="2">Best Fitness Genome</th>
					</tr>
					<tr>
						<th>CE</th>
						<th>Acc</th>
						<th>CE</th>
						<th>Acc</th>
						<th>CE</th>
						<th>Acc</th>
					</tr>
				{/if}
			</thead>
			{#each points as point, idx}
				{@const bestF1Summary = point.summaries.find(s => s.genomeType === 'best_f1')}
				{@const bestFprSummary = point.summaries.find(s => s.genomeType === 'best_fpr')}
				{@const bestFitSummary = point.summaries.find(s => s.genomeType === 'best_fitness')}
				{@const bestAccSummary = point.summaries.find(s => s.genomeType === 'best_acc')}
				{@const bestCeSummary = point.summaries.find(s => s.genomeType === 'best_ce')}
				{@const isCurrentExp = point.expId === currentExperimentId}
				{@const hasThresholds = isIDS && (bestF1Summary?.threshold_metadata || bestFprSummary?.threshold_metadata || bestAccSummary?.threshold_metadata || bestCeSummary?.threshold_metadata || bestFitSummary?.threshold_metadata)}
				{#if idx > 0}
					<tbody class="phase-spacer"><tr><td colspan="16"></td></tr></tbody>
				{/if}
				<tbody class="phase-group" class:phase-group-current={isCurrentExp && point.validationPoint === 'final'}>
					<tr class:current-phase={isCurrentExp && point.validationPoint === 'final'}>
						<td class="phase-name" class:init-phase={point.validationPoint === 'init'}>
							{point.label}
							{#if isCurrentExp && point.validationPoint === 'final'}
								<span class="current-marker">◀</span>
							{/if}
							{#if isIDS && hasThresholds}
								<br><span class="phase-threshold-hint">{headlineHint}</span>
							{/if}
						</td>
						{#if isIDS}
							{@const f05_f1 = bestF1Summary?.threshold_metadata?.[headlineKey]}
							{@const f05_fpr = bestFprSummary?.threshold_metadata?.[headlineKey]}
							{@const f05_acc = bestAccSummary?.threshold_metadata?.[headlineKey]}
							{@const f05_ce = bestCeSummary?.threshold_metadata?.[headlineKey]}
							{@const f05_fit = bestFitSummary?.threshold_metadata?.[headlineKey]}
							<td class="mono best-ce-col">{f05_f1?.f1 != null ? (f05_f1.f1 * 100).toFixed(2) + '%' : bestF1Summary?.f1_macro != null ? (bestF1Summary.f1_macro * 100).toFixed(2) + '%' : '—'}</td>
							<td class="mono best-ce-col">{f05_f1?.fpr != null ? (f05_f1.fpr * 100).toFixed(2) + '%' : bestF1Summary?.fpr != null ? (bestF1Summary.fpr * 100).toFixed(2) + '%' : '—'}</td>
							<td class="mono best-ce-col">{f05_f1?.acc != null ? (f05_f1.acc * 100).toFixed(2) + '%' : bestF1Summary ? (bestF1Summary.accuracy * 100).toFixed(2) + '%' : '—'}</td>
							<td class="mono best-acc-col">{f05_fpr?.f1 != null ? (f05_fpr.f1 * 100).toFixed(2) + '%' : bestFprSummary?.f1_macro != null ? (bestFprSummary.f1_macro * 100).toFixed(2) + '%' : '—'}</td>
							<td class="mono best-acc-col">{f05_fpr?.fpr != null ? (f05_fpr.fpr * 100).toFixed(2) + '%' : bestFprSummary?.fpr != null ? (bestFprSummary.fpr * 100).toFixed(2) + '%' : '—'}</td>
							<td class="mono best-acc-col">{f05_fpr?.acc != null ? (f05_fpr.acc * 100).toFixed(2) + '%' : bestFprSummary ? (bestFprSummary.accuracy * 100).toFixed(2) + '%' : '—'}</td>
							<td class="mono">{f05_acc?.f1 != null ? (f05_acc.f1 * 100).toFixed(2) + '%' : bestAccSummary?.f1_macro != null ? (bestAccSummary.f1_macro * 100).toFixed(2) + '%' : '—'}</td>
							<td class="mono">{f05_acc?.fpr != null ? (f05_acc.fpr * 100).toFixed(2) + '%' : bestAccSummary?.fpr != null ? (bestAccSummary.fpr * 100).toFixed(2) + '%' : '—'}</td>
							<td class="mono">{f05_acc?.acc != null ? (f05_acc.acc * 100).toFixed(2) + '%' : bestAccSummary ? (bestAccSummary.accuracy * 100).toFixed(2) + '%' : '—'}</td>
							<td class="mono">{f05_ce?.f1 != null ? (f05_ce.f1 * 100).toFixed(2) + '%' : bestCeSummary?.f1_macro != null ? (bestCeSummary.f1_macro * 100).toFixed(2) + '%' : '—'}</td>
							<td class="mono">{f05_ce?.fpr != null ? (f05_ce.fpr * 100).toFixed(2) + '%' : bestCeSummary?.fpr != null ? (bestCeSummary.fpr * 100).toFixed(2) + '%' : '—'}</td>
							<td class="mono">{f05_ce?.acc != null ? (f05_ce.acc * 100).toFixed(2) + '%' : bestCeSummary ? (bestCeSummary.accuracy * 100).toFixed(2) + '%' : '—'}</td>
							<td class="mono best-fit-col">{f05_fit?.f1 != null ? (f05_fit.f1 * 100).toFixed(2) + '%' : bestFitSummary?.f1_macro != null ? (bestFitSummary.f1_macro * 100).toFixed(2) + '%' : '—'}</td>
							<td class="mono best-fit-col">{f05_fit?.fpr != null ? (f05_fit.fpr * 100).toFixed(2) + '%' : bestFitSummary?.fpr != null ? (bestFitSummary.fpr * 100).toFixed(2) + '%' : '—'}</td>
							<td class="mono best-fit-col">{f05_fit?.acc != null ? (f05_fit.acc * 100).toFixed(2) + '%' : bestFitSummary ? (bestFitSummary.accuracy * 100).toFixed(2) + '%' : '—'}</td>
						{:else}
							<!-- LM/tiered flows emit best_ce/best_acc/best_fitness — NOT the IDS
                   best_f1/best_fpr genome types (reading those rendered '—' here). -->
							<td class="mono best-ce-col">{bestCeSummary ? bestCeSummary.ce.toFixed(4) : '—'}</td>
							<td class="mono best-ce-col">{bestCeSummary ? (bestCeSummary.accuracy * 100).toFixed(2) + '%' : '—'}</td>
							<td class="mono best-acc-col">{bestAccSummary ? bestAccSummary.ce.toFixed(4) : '—'}</td>
							<td class="mono best-acc-col">{bestAccSummary ? (bestAccSummary.accuracy * 100).toFixed(2) + '%' : '—'}</td>
							<td class="mono best-fit-col">{bestFitSummary ? bestFitSummary.ce.toFixed(4) : '—'}</td>
							<td class="mono best-fit-col">{bestFitSummary ? (bestFitSummary.accuracy * 100).toFixed(2) + '%' : '—'}</td>
						{/if}
					</tr>
					{#if hasThresholds}
						{#each subModes as mode}
							{#if (bestF1Summary?.threshold_metadata?.[mode.key] || bestFprSummary?.threshold_metadata?.[mode.key] || bestAccSummary?.threshold_metadata?.[mode.key] || bestCeSummary?.threshold_metadata?.[mode.key] || bestFitSummary?.threshold_metadata?.[mode.key])}
								<tr class="threshold-sub-row {mode.cls}">
									<td class="phase-name threshold-mode-label">{mode.label}</td>
									<td class="mono best-ce-col">{bestF1Summary?.threshold_metadata?.[mode.key]?.f1 != null ? (bestF1Summary.threshold_metadata[mode.key].f1 * 100).toFixed(2) + '%' : '—'}</td>
									<td class="mono best-ce-col">{bestF1Summary?.threshold_metadata?.[mode.key]?.fpr != null ? (bestF1Summary.threshold_metadata[mode.key].fpr * 100).toFixed(2) + '%' : '—'}</td>
									<td class="mono best-ce-col">{bestF1Summary?.threshold_metadata?.[mode.key]?.acc != null ? (bestF1Summary.threshold_metadata[mode.key].acc * 100).toFixed(2) + '%' : '—'}</td>
									<td class="mono best-acc-col">{bestFprSummary?.threshold_metadata?.[mode.key]?.f1 != null ? (bestFprSummary.threshold_metadata[mode.key].f1 * 100).toFixed(2) + '%' : '—'}</td>
									<td class="mono best-acc-col">{bestFprSummary?.threshold_metadata?.[mode.key]?.fpr != null ? (bestFprSummary.threshold_metadata[mode.key].fpr * 100).toFixed(2) + '%' : '—'}</td>
									<td class="mono best-acc-col">{bestFprSummary?.threshold_metadata?.[mode.key]?.acc != null ? (bestFprSummary.threshold_metadata[mode.key].acc * 100).toFixed(2) + '%' : '—'}</td>
									<td class="mono">{bestAccSummary?.threshold_metadata?.[mode.key]?.f1 != null ? (bestAccSummary.threshold_metadata[mode.key].f1 * 100).toFixed(2) + '%' : '—'}</td>
									<td class="mono">{bestAccSummary?.threshold_metadata?.[mode.key]?.fpr != null ? (bestAccSummary.threshold_metadata[mode.key].fpr * 100).toFixed(2) + '%' : '—'}</td>
									<td class="mono">{bestAccSummary?.threshold_metadata?.[mode.key]?.acc != null ? (bestAccSummary.threshold_metadata[mode.key].acc * 100).toFixed(2) + '%' : '—'}</td>
									<td class="mono">{bestCeSummary?.threshold_metadata?.[mode.key]?.f1 != null ? (bestCeSummary.threshold_metadata[mode.key].f1 * 100).toFixed(2) + '%' : '—'}</td>
									<td class="mono">{bestCeSummary?.threshold_metadata?.[mode.key]?.fpr != null ? (bestCeSummary.threshold_metadata[mode.key].fpr * 100).toFixed(2) + '%' : '—'}</td>
									<td class="mono">{bestCeSummary?.threshold_metadata?.[mode.key]?.acc != null ? (bestCeSummary.threshold_metadata[mode.key].acc * 100).toFixed(2) + '%' : '—'}</td>
									<td class="mono best-fit-col">{bestFitSummary?.threshold_metadata?.[mode.key]?.f1 != null ? (bestFitSummary.threshold_metadata[mode.key].f1 * 100).toFixed(2) + '%' : '—'}</td>
									<td class="mono best-fit-col">{bestFitSummary?.threshold_metadata?.[mode.key]?.fpr != null ? (bestFitSummary.threshold_metadata[mode.key].fpr * 100).toFixed(2) + '%' : '—'}</td>
									<td class="mono best-fit-col">{bestFitSummary?.threshold_metadata?.[mode.key]?.acc != null ? (bestFitSummary.threshold_metadata[mode.key].acc * 100).toFixed(2) + '%' : '—'}</td>
								</tr>
							{/if}
						{/each}
					{/if}
				</tbody>
			{/each}
		</table>
	</div>

	<!-- Per-attack-class drill-down: separate section so it doesn't split
       the main validation table's header from its phase rows. -->
	{#if isIDS && isMulticlass}
		<MulticlassBreakdown {points} />
	{:else if isIDS}
		<PerClassBreakdown {points} bind:pointChoice={perClassPointChoice} bind:thresholdChoice={perClassThresholdChoice} />
	{/if}
</div>

<style>
  /* Validation Section */
  .validation-section {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: 0.5rem;
    padding: 1rem;
    margin-bottom: 1.5rem;
  }

  .validation-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--glass-border);
  }

  .validation-title {
    font-weight: 600;
    color: var(--text-primary);
  }

  .validation-legend {
    display: flex;
    gap: 1rem;
    font-size: 1rem;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.25rem;
  }

  .legend-marker {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 4px;
  }

  .legend-marker.best-ce { background: var(--accent-blue); }
  .legend-marker.best-acc { background: var(--accent-green); }
  .legend-marker.best-fitness { background: var(--accent-purple, #9b59b6); }

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

  /* Validation Progression Table */
  .validation-table-container {
    overflow-x: auto;
  }

  .validation-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 1rem;
  }

  .validation-table th {
    background: rgba(51, 65, 85, 0.4);
    padding: 0.4rem 0.5rem;
    text-align: center;
    font-weight: 600;
    color: var(--text-secondary);
    font-size: 1rem;
    text-transform: uppercase;
    border-bottom: 1px solid var(--glass-border);
    white-space: nowrap;
  }

  .validation-table td {
    padding: 0.4rem 0.5rem;
    text-align: center;
    border-bottom: 1px solid var(--glass-border);
  }

  .validation-table tr:last-child td {
    border-bottom: none;
  }

  .validation-table .phase-name {
    text-align: left;
    font-weight: 500;
    color: var(--text-primary);
    max-width: 140px;
    white-space: normal;
    line-height: 1.2;
  }

  .validation-table .phase-name.init-phase {
    color: var(--accent-blue);
    font-style: italic;
  }

  .validation-table tr.current-phase {
    background: rgba(59, 130, 246, 0.1);
  }

  .validation-table tr.current-phase td {
    font-weight: 600;
  }

  .current-marker {
    color: var(--accent-blue);
    margin-left: 0.5rem;
    font-size: 1rem;
  }

  .validation-table th[colspan] {
    border-bottom: 1px solid var(--glass-border);
  }

  .validation-table .best-ce-col {
    background: rgba(59, 130, 246, 0.08);
  }

  .validation-table .best-acc-col {
    background: rgba(34, 197, 94, 0.08);
  }

  .validation-table .best-fit-col {
    background: rgba(155, 89, 182, 0.08);
  }

  /* Glass panels for each genome column group */
  /* Best F1 group: cols 2-4 */
  .validation-table td:nth-child(2) {
    border-left: 1px solid rgba(59, 130, 246, 0.3) !important;
  }
  .validation-table td:nth-child(4) {
    border-right: 1px solid rgba(59, 130, 246, 0.3) !important;
  }
  /* Best FPR group: cols 5-7 */
  .validation-table td:nth-child(5) {
    border-left: 1px solid rgba(34, 197, 94, 0.3) !important;
  }
  .validation-table td:nth-child(7) {
    border-right: 1px solid rgba(34, 197, 94, 0.3) !important;
  }
  /* Best Fitness group: cols 8-10 */
  .validation-table td:nth-child(8) {
    border-left: 1px solid rgba(155, 89, 182, 0.3) !important;
  }
  .validation-table td:nth-child(10) {
    border-right: 1px solid rgba(155, 89, 182, 0.3) !important;
  }

  /* Top/bottom borders on first/last row of each phase group */
  .phase-group tr:first-child td:nth-child(n+2) {
    border-top: 1px solid var(--glass-border) !important;
  }
  .phase-group tr:first-child td:nth-child(2) { border-top-color: rgba(59, 130, 246, 0.3) !important; }
  .phase-group tr:first-child td:nth-child(3) { border-top-color: rgba(59, 130, 246, 0.3) !important; }
  .phase-group tr:first-child td:nth-child(4) { border-top-color: rgba(59, 130, 246, 0.3) !important; }
  .phase-group tr:first-child td:nth-child(5) { border-top-color: rgba(34, 197, 94, 0.3) !important; }
  .phase-group tr:first-child td:nth-child(6) { border-top-color: rgba(34, 197, 94, 0.3) !important; }
  .phase-group tr:first-child td:nth-child(7) { border-top-color: rgba(34, 197, 94, 0.3) !important; }
  .phase-group tr:first-child td:nth-child(8) { border-top-color: rgba(155, 89, 182, 0.3) !important; }
  .phase-group tr:first-child td:nth-child(9) { border-top-color: rgba(155, 89, 182, 0.3) !important; }
  .phase-group tr:first-child td:nth-child(10) { border-top-color: rgba(155, 89, 182, 0.3) !important; }

  .phase-group tr:last-child td:nth-child(n+2) {
    border-bottom: 1px solid var(--glass-border) !important;
  }
  .phase-group tr:last-child td:nth-child(2) { border-bottom-color: rgba(59, 130, 246, 0.3) !important; }
  .phase-group tr:last-child td:nth-child(3) { border-bottom-color: rgba(59, 130, 246, 0.3) !important; }
  .phase-group tr:last-child td:nth-child(4) { border-bottom-color: rgba(59, 130, 246, 0.3) !important; }
  .phase-group tr:last-child td:nth-child(5) { border-bottom-color: rgba(34, 197, 94, 0.3) !important; }
  .phase-group tr:last-child td:nth-child(6) { border-bottom-color: rgba(34, 197, 94, 0.3) !important; }
  .phase-group tr:last-child td:nth-child(7) { border-bottom-color: rgba(34, 197, 94, 0.3) !important; }
  .phase-group tr:last-child td:nth-child(8) { border-bottom-color: rgba(155, 89, 182, 0.3) !important; }
  .phase-group tr:last-child td:nth-child(9) { border-bottom-color: rgba(155, 89, 182, 0.3) !important; }
  .phase-group tr:last-child td:nth-child(10) { border-bottom-color: rgba(155, 89, 182, 0.3) !important; }

  /* Rounded corners on column group panels */
  .phase-group tr:first-child td:nth-child(2) { border-top-left-radius: 6px; }
  .phase-group tr:first-child td:nth-child(4) { border-top-right-radius: 6px; }
  .phase-group tr:first-child td:nth-child(5) { border-top-left-radius: 6px; }
  .phase-group tr:first-child td:nth-child(7) { border-top-right-radius: 6px; }
  .phase-group tr:first-child td:nth-child(8) { border-top-left-radius: 6px; }
  .phase-group tr:first-child td:nth-child(10) { border-top-right-radius: 6px; }
  .phase-group tr:last-child td:nth-child(2) { border-bottom-left-radius: 6px; }
  .phase-group tr:last-child td:nth-child(4) { border-bottom-right-radius: 6px; }
  .phase-group tr:last-child td:nth-child(5) { border-bottom-left-radius: 6px; }
  .phase-group tr:last-child td:nth-child(7) { border-bottom-right-radius: 6px; }
  .phase-group tr:last-child td:nth-child(8) { border-bottom-left-radius: 6px; }
  .phase-group tr:last-child td:nth-child(10) { border-bottom-right-radius: 6px; }

  /* Header column group styling */
  .validation-table thead tr:first-child th:nth-child(2) {
    border-left: 1px solid rgba(59, 130, 246, 0.3);
    border-top: 1px solid rgba(59, 130, 246, 0.3);
    border-top-left-radius: 6px;
    background: rgba(59, 130, 246, 0.1);
  }
  .validation-table thead tr:first-child th:nth-child(3) {
    border-left: 1px solid rgba(34, 197, 94, 0.3);
    border-top: 1px solid rgba(34, 197, 94, 0.3);
    border-top-left-radius: 6px;
    background: rgba(34, 197, 94, 0.1);
  }
  .validation-table thead tr:first-child th:nth-child(4) {
    border-left: 1px solid rgba(155, 89, 182, 0.3);
    border-top: 1px solid rgba(155, 89, 182, 0.3);
    border-top-left-radius: 6px;
    background: rgba(155, 89, 182, 0.1);
  }

  .threshold-sub-row td {
    padding: 0.2rem 0.75rem !important;
    border-bottom: none !important;
    font-size: 1rem;
  }

  .threshold-sub-row:last-child td {
    border-bottom: 1px solid var(--glass-border) !important;
  }

  .phase-threshold-hint {
    font-size: 1rem; /* accessibility: 1rem minimum for all text (CLAUDE.md) */
    color: #e2e8f0;
    font-weight: 400;
  }

  .threshold-mode-label {
    font-weight: 400 !important;
    font-size: 1rem;
    white-space: nowrap;
    font-family: monospace;
  }

  /* Actual result row (main data — fixed 0.5) — white, bold */
  .validation-table tbody tr:not(.threshold-sub-row) td.mono {
    color: #e2e8f0;
    font-weight: 600;
  }

  /* Phase name (Grid Search / GA Neurons) — white */
  .validation-table tbody tr:not(.threshold-sub-row) td.phase-name {
    color: #e2e8f0;
    font-weight: 600;
  }

  /* Platt scaling — cyan/teal */
  .threshold-platt-row td {
    color: #06b6d4 !important;
  }

  /* Beta calibration — purple (parametric, 3 params) */
  .threshold-beta-row td {
    color: #a855f7 !important;
  }

  /* Empirical table — bright green (zero assumptions, data-driven) */
  .threshold-empirical-row td {
    color: #22c55e !important;
    font-weight: 500;
  }

  /* Train-cal — light steel blue */
  .threshold-train-row td {
    color: #94a3b8 !important;
  }

  /* Oracle — orange, italic (upper bound) */
  .threshold-oracle-row td {
    color: #f59e0b !important;
    font-style: italic;
  }

  /* Glass card effect per phase group via cell borders + radius */
  .phase-group td {
    border-left: 1px solid var(--glass-border);
    border-right: 1px solid var(--glass-border);
    border-top: none;
    border-bottom: none;
  }

  .phase-group td:not(:first-child) {
    border-left: none;
  }

  .phase-group tr:first-child td {
    border-top: 1px solid var(--glass-border);
    padding-top: 0.625rem;
    background: rgba(15, 23, 42, 0.5);
  }

  .phase-group tr:first-child td:first-child {
    border-top-left-radius: 8px;
  }

  .phase-group tr:first-child td:last-child {
    border-top-right-radius: 8px;
  }

  .phase-group tr:last-child td {
    border-bottom: 1px solid var(--glass-border);
    padding-bottom: 0.625rem;
  }

  .phase-group tr:last-child td:first-child {
    border-bottom-left-radius: 8px;
  }

  .phase-group tr:last-child td:last-child {
    border-bottom-right-radius: 8px;
  }

  .phase-group-current td {
    background: rgba(59, 130, 246, 0.04);
  }

  .phase-group-current tr:first-child td {
    border-top-color: rgba(59, 130, 246, 0.3);
  }

  .phase-group-current tr:last-child td {
    border-bottom-color: rgba(59, 130, 246, 0.3);
  }

  .phase-group-current td:first-child {
    border-left-color: rgba(59, 130, 246, 0.3);
  }

  .phase-group-current td:last-child {
    border-right-color: rgba(59, 130, 246, 0.3);
  }

  /* Spacer row between phase groups */
  .phase-spacer td {
    height: 12px;
    padding: 0;
    border: none !important;
    background: transparent !important;
  }
</style>
