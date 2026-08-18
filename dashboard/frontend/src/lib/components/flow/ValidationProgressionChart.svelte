<script lang="ts">
	import type { Experiment, ValidationSummary } from '$lib/types'

	export let validationSummaries: ValidationSummary[] = []
	export let displayExperiments: Experiment[] = []

	// Validation chart tooltip state
	let validationTooltip: {
		x: number
		y: number
		label: string
		genomeType: string
		ce: number
		accuracy: number
	} | null = null

	// Build chart data points from validation summaries
  // Group by experiment and validation_point, creating x-axis labels like "Init", "Exp1", "Exp2", etc.
	$: validationChartData = (() =>
	{
		if (validationSummaries.length === 0) return []

		// Create a map of experiment_id -> sequence_order for labeling
		const expOrderMap = new Map<number, number>()
		displayExperiments.forEach((exp, idx) =>
		{
			expOrderMap.set(exp.id, idx)
		})

		// Group validations by (experiment_id, validation_point)
		const points: {
			label: string
			sortKey: number
			validations: { genomeType: string, ce: number, accuracy: number }[]
		}[] = []

		// Group by experiment and point
		const grouped = new Map<string, ValidationSummary[]>()
		for (const v of validationSummaries)
		{
			const key = `${v.experiment_id}-${v.validation_point}`
			if (!grouped.has(key)) grouped.set(key, [])
			grouped.get(key)!.push(v)
		}

		// Convert to chart points
		for (const [key, validations] of grouped)
		{
			const [expIdStr, point] = key.split('-')
			const expId = parseInt(expIdStr)
			const order = expOrderMap.get(expId) ?? 0

			// Create label: "Init" for first init, "Exp N" for finals
			let label: string
			let sortKey: number
			if (point === 'init')
			{
				label = order === 0 ? 'Init' : `Exp${order} Init`
				sortKey = order * 2  // init comes before final
			}
			else
			{
				label = `Exp${order + 1}`
				sortKey = order * 2 + 1
			}

			points.push({
				label,
				sortKey,
				validations: validations.map(v => ({
					genomeType: v.genome_type,
					ce: v.ce,
					accuracy: v.accuracy
				}))
			})
		}

		// Sort by sortKey
		return points.sort((a, b) => a.sortKey - b.sortKey)
	})()

	// Chart dimensions
	const valChartPadding = { top: 30, right: 40, bottom: 50, left: 60 }
	const valChartSvgWidth = 700
	const valChartSvgHeight = 280
	$: valChartWidth = valChartSvgWidth - valChartPadding.left - valChartPadding.right
	$: valChartHeight = valChartSvgHeight - valChartPadding.top - valChartPadding.bottom

	// Compute CE range for y-axis
	$: allCeValues = validationChartData.flatMap(p => p.validations.map(v => v.ce))
	$: valCeMin = allCeValues.length > 0 ? Math.min(...allCeValues) : 0
	$: valCeMax = allCeValues.length > 0 ? Math.max(...allCeValues) : 1
	$: valCeRange = valCeMax - valCeMin || 0.001
	// Add 10% padding to range
	$: valCeMinPadded = valCeMin - valCeRange * 0.1
	$: valCeMaxPadded = valCeMax + valCeRange * 0.1
	$: valCeRangePadded = valCeMaxPadded - valCeMinPadded

	// Genome type colors and markers
	function getGenomeTypeColor(genomeType: string): string
	{
		switch (genomeType)
		{
			case 'best_ce': return 'var(--accent-blue)'
			case 'best_acc': return 'var(--accent-green)'
			case 'best_fitness': return 'var(--accent-purple, #8b5cf6)'
			default: return 'var(--text-secondary)'
		}
	}

	function getGenomeTypeLabel(genomeType: string): string
	{
		switch (genomeType)
		{
			case 'best_ce': return 'Best CE'
			case 'best_acc': return 'Best Acc'
			case 'best_fitness': return 'Best Fitness'
			default: return genomeType
		}
	}

	// Helper to compute polyline points for a genome type
	function getPolylinePoints(genomeType: string): string
	{
		const points: { x: number, y: number }[] = []
		validationChartData.forEach((p, i) =>
		{
			const v = p.validations.find(val => val.genomeType === genomeType)
			if (v)
			{
				const x = valChartPadding.left + (i / Math.max(validationChartData.length - 1, 1)) * valChartWidth
				const y = valChartPadding.top + valChartHeight - ((v.ce - valCeMinPadded) / valCeRangePadded) * valChartHeight
				points.push({ x, y })
			}
		})
		return points.map(p => `${p.x},${p.y}`).join(' ')
	}

	// Check if we have enough points for a polyline
	function hasMultiplePoints(genomeType: string): boolean
	{
		let count = 0
		for (const p of validationChartData)
		{
			if (p.validations.some(v => v.genomeType === genomeType)) count++
			if (count > 1) return true
		}
		return false
	}
</script>

{#if validationChartData.length > 0}
	<section class="section">
		<div class="card">
			<div class="card-header">
				<span class="card-title">Validation Progression</span>
				<div class="val-chart-legend">
					<span class="legend-item"><span class="legend-marker best-ce"></span> Best CE</span>
					<span class="legend-item"><span class="legend-marker best-acc"></span> Best Acc</span>
					<span class="legend-item"><span class="legend-marker best-fitness"></span> Best Fitness</span>
				</div>
			</div>
			<div class="val-chart-container">
				<svg viewBox="-20 0 {valChartSvgWidth + 20} {valChartSvgHeight}" class="val-chart">
					<!-- Y-axis labels (CE) -->
					<text x={valChartPadding.left - 8} y={valChartPadding.top + 4} text-anchor="end" class="axis-label">{valCeMaxPadded.toFixed(2)}</text>
					<text x={valChartPadding.left - 8} y={valChartPadding.top + valChartHeight} text-anchor="end" class="axis-label">{valCeMinPadded.toFixed(2)}</text>
					<text x={valChartPadding.left - 8} y={valChartPadding.top + valChartHeight / 2} text-anchor="end" class="axis-label">{((valCeMaxPadded + valCeMinPadded) / 2).toFixed(2)}</text>

					<!-- Y-axis title -->
					<text x={-8} y={valChartPadding.top + valChartHeight / 2} text-anchor="middle" class="axis-title" transform="rotate(-90, -8, {valChartPadding.top + valChartHeight / 2})">CE (Loss)</text>

					<!-- Grid lines -->
					<line x1={valChartPadding.left} y1={valChartPadding.top} x2={valChartPadding.left + valChartWidth} y2={valChartPadding.top} stroke="var(--border)" stroke-dasharray="4" />
					<line x1={valChartPadding.left} y1={valChartPadding.top + valChartHeight} x2={valChartPadding.left + valChartWidth} y2={valChartPadding.top + valChartHeight} stroke="var(--border)" stroke-dasharray="4" />
					<line x1={valChartPadding.left} y1={valChartPadding.top + valChartHeight / 2} x2={valChartPadding.left + valChartWidth} y2={valChartPadding.top + valChartHeight / 2} stroke="var(--border)" stroke-dasharray="2" opacity="0.5" />

					<!-- X-axis line -->
					<line x1={valChartPadding.left} y1={valChartPadding.top + valChartHeight} x2={valChartPadding.left + valChartWidth} y2={valChartPadding.top + valChartHeight} stroke="var(--text-tertiary)" />

					<!-- X-axis labels -->
					{#each validationChartData as point, i}
						{@const x = valChartPadding.left + (i / Math.max(validationChartData.length - 1, 1)) * valChartWidth}
						<line x1={x} y1={valChartPadding.top + valChartHeight} x2={x} y2={valChartPadding.top + valChartHeight + 5} stroke="var(--text-tertiary)" />
						<text x={x} y={valChartPadding.top + valChartHeight + 20} text-anchor="middle" class="axis-label x-label">{point.label}</text>
					{/each}

					<!-- Connect lines for each genome type -->
					{#each ['best_ce', 'best_acc', 'best_fitness'] as genomeType}
						{#if hasMultiplePoints(genomeType)}
							<polyline
								fill="none"
								stroke={getGenomeTypeColor(genomeType)}
								stroke-width="2"
								stroke-opacity="0.5"
								points={getPolylinePoints(genomeType)}
							/>
						{/if}
					{/each}

					<!-- Data points (offset horizontally when multiple markers at same point) -->
					{#each validationChartData as point, i}
						{@const baseX = valChartPadding.left + (i / Math.max(validationChartData.length - 1, 1)) * valChartWidth}
						{@const numValidations = point.validations.length}
						{@const offsetStep = numValidations > 1 ? 12 : 0}
						{#each point.validations as v, vi}
							{@const xOffset = numValidations > 1 ? (vi - (numValidations - 1) / 2) * offsetStep : 0}
							{@const x = baseX + xOffset}
							{@const y = valChartPadding.top + valChartHeight - ((v.ce - valCeMinPadded) / valCeRangePadded) * valChartHeight}
							<!-- Marker based on genome type -->
							{#if v.genomeType === 'best_ce'}
								<circle
									cx={x} cy={y} r="6"
									fill={getGenomeTypeColor(v.genomeType)}
									stroke="white" stroke-width="2"
									class="data-point"
									role="button"
									tabindex="-1"
									on:mouseenter={() => validationTooltip = { x, y, label: point.label, genomeType: v.genomeType, ce: v.ce, accuracy: v.accuracy }}
									on:mouseleave={() => validationTooltip = null}
								/>
							{:else if v.genomeType === 'best_acc'}
								<rect
									x={x - 5} y={y - 5} width="10" height="10"
									fill={getGenomeTypeColor(v.genomeType)}
									stroke="white" stroke-width="2"
									class="data-point"
									role="button"
									tabindex="-1"
									on:mouseenter={() => validationTooltip = { x, y, label: point.label, genomeType: v.genomeType, ce: v.ce, accuracy: v.accuracy }}
									on:mouseleave={() => validationTooltip = null}
								/>
							{:else}
								<polygon
									points="{x},{y - 7} {x + 6},{y + 4} {x - 6},{y + 4}"
									fill={getGenomeTypeColor(v.genomeType)}
									stroke="white" stroke-width="2"
									class="data-point"
									role="button"
									tabindex="-1"
									on:mouseenter={() => validationTooltip = { x, y, label: point.label, genomeType: v.genomeType, ce: v.ce, accuracy: v.accuracy }}
									on:mouseleave={() => validationTooltip = null}
								/>
							{/if}
						{/each}
					{/each}

					<!-- Tooltip -->
					{#if validationTooltip}
						<g class="tooltip-group" transform="translate({validationTooltip.x}, {validationTooltip.y - 70})">
							<rect x="-100" y="-10" width="200" height="70" rx="4" fill="var(--bg-secondary)" stroke="var(--border)" />
							<text x="0" y="10" text-anchor="middle" class="tooltip-label">{validationTooltip.label}</text>
							<text x="0" y="30" text-anchor="middle" class="tooltip-type" fill={getGenomeTypeColor(validationTooltip.genomeType)}>{getGenomeTypeLabel(validationTooltip.genomeType)}</text>
							<text x="0" y="50" text-anchor="middle" class="tooltip-value">CE: {validationTooltip.ce.toFixed(4)} | Acc: {(validationTooltip.accuracy * 100).toFixed(2)}%</text>
						</g>
					{/if}
				</svg>
			</div>
		</div>
	</section>
{/if}

<style>
  .section {
    margin-bottom: 2rem;
  }

  /* Validation Progression Chart */
  .card {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    overflow: hidden;
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--glass-border);
    background: rgba(51, 65, 85, 0.4);
  }

  .card-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
  }

  .val-chart-legend {
    display: flex;
    gap: 1rem;
    font-size: 1rem;
    color: var(--text-secondary);
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.375rem;
  }

  .legend-marker {
    width: 12px;
    height: 12px;
    border-radius: 2px;
  }

  .legend-marker.best-ce {
    background: var(--accent-blue);
    border-radius: 50%;
  }

  .legend-marker.best-acc {
    background: var(--accent-green);
  }

  .legend-marker.best-fitness {
    background: var(--accent-purple, #8b5cf6);
    clip-path: polygon(50% 0%, 100% 100%, 0% 100%);
  }

  .val-chart-container {
    padding: 1rem;
  }

  .val-chart {
    width: 100%;
    height: 260px;
  }

  .val-chart .axis-label {
    font-size: 1rem;
    fill: var(--text-tertiary);
  }

  .val-chart .axis-label.x-label {
    font-size: 1rem;
  }

  .val-chart .axis-title {
    font-size: 1rem;
    fill: var(--text-secondary);
    font-weight: 500;
  }

  .val-chart .data-point {
    cursor: pointer;
  }

  .val-chart .tooltip-group {
    pointer-events: none;
  }

  .val-chart .tooltip-label {
    font-size: 1rem;
    font-weight: 600;
    fill: var(--text-primary);
  }

  .val-chart .tooltip-type {
    font-size: 1rem;
    font-weight: 500;
  }

  .val-chart .tooltip-value {
    font-size: 1rem;
    fill: var(--text-secondary);
  }

  .val-chart .tooltip-group rect {
    filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
  }
</style>
