<script lang="ts">
  export let isIDS: boolean = false;
  export let chartData: { iter: number; ce: number; acc: number | null; avgCe: number | null; avgAcc: number | null; f1: number | null; fpr: number | null }[] = [];

  // Chart tooltip state
  let tooltipData: { x: number; y: number; iter: number; ce: number; acc: number | null; avgCe: number | null; avgAcc: number | null; f1?: number | null; fpr?: number | null } | null = null;

  // Chart computed values
  $: cumulativeData = (() => {
    let minCE = Infinity;
    let maxAcc = 0;
    return chartData.map(p => {
      minCE = Math.min(minCE, p.ce);
      if (p.acc !== null) maxAcc = Math.max(maxAcc, p.acc);
      return { iter: p.iter, ce: minCE, acc: maxAcc > 0 ? maxAcc : null, avgCe: p.avgCe, avgAcc: p.avgAcc };
    });
  })();
  $: ceMin = cumulativeData.length > 0 ? Math.min(...cumulativeData.map(p => p.ce)) : 0;
  $: ceMax = chartData.length > 0 ? Math.max(...chartData.map(p => p.ce)) : 1;

  // For avg CE line, need separate range that includes avg values
  $: avgCeValues = chartData.filter(p => p.avgCe !== null && p.avgCe !== undefined).map(p => p.avgCe!);
  $: avgCeMax = avgCeValues.length > 0 ? Math.max(...avgCeValues) : ceMax;
  $: avgCeMin = avgCeValues.length > 0 ? Math.min(...avgCeValues) : ceMin;
  $: combinedCeMax = Math.max(ceMax, avgCeMax);
  $: combinedCeMin = Math.min(ceMin, avgCeMin);
  $: combinedCeRange = combinedCeMax - combinedCeMin || 0.001;

  $: accData = cumulativeData.filter(p => p.acc !== null).map(p => ({ ...p, acc: p.acc ?? 0 }));
  $: accMax = accData.length > 0 ? Math.max(...accData.map(p => p.acc)) : 1;
  $: avgAccValues = chartData.filter(p => p.avgAcc !== null && p.avgAcc !== undefined).map(p => p.avgAcc!);
  $: combinedAccMax = avgAccValues.length > 0 ? Math.max(accMax, ...avgAccValues) : accMax;
  $: accRange = combinedAccMax || 0.001;

  // IDS chart: cumulative best F1 (max) and best FPR (min)
  $: cumulativeIdsData = (() => {
    let maxF1 = 0;
    let minFpr = 100;
    return chartData.map(p => {
      if (p.f1 !== null) maxF1 = Math.max(maxF1, p.f1);
      if (p.fpr !== null) minFpr = Math.min(minFpr, p.fpr);
      return { iter: p.iter, f1: maxF1 > 0 ? maxF1 : null, fpr: minFpr < 100 ? minFpr : null };
    });
  })();
  $: f1Data = chartData.filter(p => p.f1 !== null);
  $: f1Max = f1Data.length > 0 ? Math.max(...f1Data.map(p => p.f1!)) : 100;
  $: fprData = chartData.filter(p => p.fpr !== null);
  $: fprMax = fprData.length > 0 ? Math.max(...fprData.map(p => p.fpr!)) : 100;

  // Chart dimensions
  const chartPadding = { top: 40, right: 60, bottom: 40, left: 60 };
  const chartSvgWidth = 800;
  const chartSvgHeight = 320;
  $: chartWidth = chartSvgWidth - chartPadding.left - chartPadding.right;
  $: chartHeight = chartSvgHeight - chartPadding.top - chartPadding.bottom;

  // X-axis tick positions
  $: xAxisTicks = (() => {
    const n = chartData.length;
    if (n <= 10) return chartData.map((_, i) => i);
    const step = Math.ceil(n / 10);
    return chartData.map((_, i) => i).filter(i => i % step === 0 || i === n - 1);
  })();
</script>

<div class="card">
  <div class="card-header">
    <span class="card-title">
      Progress ({chartData.length} iterations)
    </span>
    <div class="chart-legend">
      {#if isIDS}
        <span class="legend-item"><span class="legend-line ce"></span> Best F1</span>
        <span class="legend-item"><span class="legend-line acc"></span> Best FPR</span>
      {:else}
        <span class="legend-item"><span class="legend-line ce"></span> Best CE</span>
        <span class="legend-item"><span class="legend-line ce-avg"></span> Avg CE</span>
        <span class="legend-item"><span class="legend-line acc"></span> Best Acc</span>
        <span class="legend-item"><span class="legend-line acc-avg"></span> Avg Acc</span>
      {/if}
    </div>
  </div>
  <div class="chart-container">
    <svg viewBox="0 0 {chartSvgWidth} {chartSvgHeight}" class="line-chart">
      {#if isIDS}
        <!-- IDS Chart: F1 (left, blue) + FPR (right, green) -->
        <!-- Y-axis labels (F1 on left, 0-100%) -->
        <text x={chartPadding.left - 5} y={chartPadding.top + 5} text-anchor="end" class="axis-label ce-label">100%</text>
        <text x={chartPadding.left - 5} y={chartPadding.top + chartHeight - 5} text-anchor="end" class="axis-label ce-label">0%</text>

        <!-- Y-axis labels (FPR on right, 0-fprMax%) -->
        {#if fprData.length > 0}
          {@const fprAxisMax = Math.max(fprMax * 1.2, 1)}
          <text x={chartSvgWidth - chartPadding.right + 5} y={chartPadding.top + 5} text-anchor="start" class="axis-label acc-label">{fprAxisMax.toFixed(1)}%</text>
          <text x={chartSvgWidth - chartPadding.right + 5} y={chartPadding.top + chartHeight - 5} text-anchor="start" class="axis-label acc-label">0%</text>
        {/if}

        <!-- Grid lines -->
        <line x1={chartPadding.left} y1={chartPadding.top} x2={chartPadding.left + chartWidth} y2={chartPadding.top} stroke="var(--border)" stroke-dasharray="4" />
        <line x1={chartPadding.left} y1={chartPadding.top + chartHeight} x2={chartPadding.left + chartWidth} y2={chartPadding.top + chartHeight} stroke="var(--border)" stroke-dasharray="4" />
        <line x1={chartPadding.left} y1={chartPadding.top + chartHeight / 2} x2={chartPadding.left + chartWidth} y2={chartPadding.top + chartHeight / 2} stroke="var(--border)" stroke-dasharray="2" opacity="0.5" />

        <!-- X-axis line -->
        <line x1={chartPadding.left} y1={chartPadding.top + chartHeight} x2={chartPadding.left + chartWidth} y2={chartPadding.top + chartHeight} stroke="var(--text-tertiary)" />

        <!-- X-axis ticks and labels -->
        {#each xAxisTicks as tickIdx}
          {@const x = chartPadding.left + (tickIdx / Math.max(chartData.length - 1, 1)) * chartWidth}
          <line x1={x} y1={chartPadding.top + chartHeight} x2={x} y2={chartPadding.top + chartHeight + 5} stroke="var(--text-tertiary)" />
          <text x={x} y={chartPadding.top + chartHeight + 18} text-anchor="middle" class="axis-label x-label">{chartData[tickIdx]?.iter ?? tickIdx + 1}</text>
        {/each}

        <!-- Best F1 line (cumulative max, blue, 0-100% scale) -->
        {#if f1Data.length > 0}
          <polyline
            fill="none"
            stroke="var(--accent-blue)"
            stroke-width="2"
            points={cumulativeIdsData.map((p, i) => {
              if (p.f1 === null) return null;
              const x = chartPadding.left + (i / Math.max(cumulativeIdsData.length - 1, 1)) * chartWidth;
              const y = chartPadding.top + chartHeight - (p.f1 / 100) * chartHeight;
              return `${x},${y}`;
            }).filter(Boolean).join(' ')}
          />
        {/if}

        <!-- Best FPR line (cumulative min, green, 0-fprMax scale) -->
        {#if fprData.length > 0}
          {@const fprAxisMax = Math.max(fprMax * 1.2, 1)}
          <polyline
            fill="none"
            stroke="var(--accent-green)"
            stroke-width="2"
            points={cumulativeIdsData.map((p, i) => {
              if (p.fpr === null) return null;
              const x = chartPadding.left + (i / Math.max(cumulativeIdsData.length - 1, 1)) * chartWidth;
              const y = chartPadding.top + chartHeight - (p.fpr / fprAxisMax) * chartHeight;
              return `${x},${y}`;
            }).filter(Boolean).join(' ')}
          />
        {/if}

        <!-- Best F1 marker -->
        {#each [cumulativeIdsData.findIndex(p => p.f1 !== null && p.f1 === f1Max)] as bestIdx}
          {#if bestIdx >= 0}
            <circle cx={chartPadding.left + (bestIdx / Math.max(cumulativeIdsData.length - 1, 1)) * chartWidth} cy={chartPadding.top + chartHeight - (f1Max / 100) * chartHeight} r="5" fill="var(--accent-blue)" />
            <text x={chartPadding.left + (bestIdx / Math.max(cumulativeIdsData.length - 1, 1)) * chartWidth} y={chartPadding.top + chartHeight - (f1Max / 100) * chartHeight - 8} text-anchor="middle" class="best-label" fill="var(--accent-blue)">{f1Max.toFixed(2)}%</text>
          {/if}
        {/each}

        <!-- Hover zones -->
        {#each chartData as point, i}
          <rect
            x={chartPadding.left + (i / Math.max(chartData.length - 1, 1)) * chartWidth - chartWidth / Math.max(chartData.length, 1) / 2}
            y={chartPadding.top}
            width={chartWidth / Math.max(chartData.length, 1)}
            height={chartHeight}
            fill="transparent"
            role="button"
            tabindex="-1"
            on:mouseenter={() => {
              const cumPoint = cumulativeIdsData[i];
              const x = chartPadding.left + (i / Math.max(chartData.length - 1, 1)) * chartWidth;
              tooltipData = { x, y: chartPadding.top + chartHeight / 2, iter: point.iter, ce: 0, acc: point.acc, avgCe: null, avgAcc: null, f1: cumPoint.f1, fpr: cumPoint.fpr };
            }}
            on:mouseleave={() => tooltipData = null}
          />
        {/each}

        <!-- Tooltip -->
        {#if tooltipData}
          <g transform="translate({tooltipData.x}, {tooltipData.y})">
            <rect x="-85" y="-50" width="170" height="90" fill="var(--bg-card)" stroke="var(--border)" rx="6" class="tooltip-bg" />
            <text x="0" y="-30" text-anchor="middle" class="tooltip-title">Iter {tooltipData.iter}</text>
            {#if tooltipData.f1 !== null && tooltipData.f1 !== undefined}
              <text x="-70" y="-5" class="tooltip-label ce-label">Best F1:</text>
              <text x="70" y="-5" text-anchor="end" class="tooltip-value ce-label">{tooltipData.f1.toFixed(2)}%</text>
            {/if}
            {#if tooltipData.fpr !== null && tooltipData.fpr !== undefined}
              <text x="-70" y="20" class="tooltip-label acc-label">Best FPR:</text>
              <text x="70" y="20" text-anchor="end" class="tooltip-value acc-label">{tooltipData.fpr.toFixed(3)}%</text>
            {/if}
          </g>
        {/if}

      {:else}
        <!-- LM Chart: CE (left, blue) + Acc (right, green) -->
        <!-- Y-axis labels (CE on left) -->
        <text x={chartPadding.left - 5} y={chartPadding.top + 5} text-anchor="end" class="axis-label ce-label">{combinedCeMax.toFixed(2)}</text>
        <text x={chartPadding.left - 5} y={chartPadding.top + chartHeight - 5} text-anchor="end" class="axis-label ce-label">{combinedCeMin.toFixed(2)}</text>

        <!-- Y-axis labels (Acc on right) -->
        {#if accData.length > 0}
          <text x={chartSvgWidth - chartPadding.right + 5} y={chartPadding.top + 5} text-anchor="start" class="axis-label acc-label">{combinedAccMax.toFixed(2)}%</text>
          <text x={chartSvgWidth - chartPadding.right + 5} y={chartPadding.top + chartHeight - 5} text-anchor="start" class="axis-label acc-label">0.00%</text>
        {/if}

        <!-- Grid lines -->
        <line x1={chartPadding.left} y1={chartPadding.top} x2={chartPadding.left + chartWidth} y2={chartPadding.top} stroke="var(--border)" stroke-dasharray="4" />
        <line x1={chartPadding.left} y1={chartPadding.top + chartHeight} x2={chartPadding.left + chartWidth} y2={chartPadding.top + chartHeight} stroke="var(--border)" stroke-dasharray="4" />
        <line x1={chartPadding.left} y1={chartPadding.top + chartHeight / 2} x2={chartPadding.left + chartWidth} y2={chartPadding.top + chartHeight / 2} stroke="var(--border)" stroke-dasharray="2" opacity="0.5" />

        <!-- X-axis line -->
        <line x1={chartPadding.left} y1={chartPadding.top + chartHeight} x2={chartPadding.left + chartWidth} y2={chartPadding.top + chartHeight} stroke="var(--text-tertiary)" />

        <!-- X-axis ticks and labels -->
        {#each xAxisTicks as tickIdx}
          {@const x = chartPadding.left + (tickIdx / Math.max(chartData.length - 1, 1)) * chartWidth}
          <line x1={x} y1={chartPadding.top + chartHeight} x2={x} y2={chartPadding.top + chartHeight + 5} stroke="var(--text-tertiary)" />
          <text x={x} y={chartPadding.top + chartHeight + 18} text-anchor="middle" class="axis-label x-label">{chartData[tickIdx]?.iter ?? tickIdx + 1}</text>
        {/each}

        <!-- Avg CE line (dashed, behind main line) -->
        {#if avgCeValues.length > 0}
          <polyline
            fill="none"
            stroke="var(--accent-blue)"
            stroke-width="1.5"
            stroke-dasharray="4 2"
            opacity="0.5"
            points={chartData.map((p, i) => {
              if (p.avgCe === null || p.avgCe === undefined) return null;
              const x = chartPadding.left + (i / Math.max(chartData.length - 1, 1)) * chartWidth;
              const y = chartPadding.top + chartHeight - ((p.avgCe - combinedCeMin) / combinedCeRange) * chartHeight;
              return `${x},${y}`;
            }).filter(Boolean).join(' ')}
          />
        {/if}

        <!-- Best CE line (cumulative min) -->
        <polyline
          fill="none"
          stroke="var(--accent-blue)"
          stroke-width="2"
          points={cumulativeData.map((p, i) => {
            const x = chartPadding.left + (i / Math.max(cumulativeData.length - 1, 1)) * chartWidth;
            const y = chartPadding.top + chartHeight - ((p.ce - combinedCeMin) / combinedCeRange) * chartHeight;
            return `${x},${y}`;
          }).join(' ')}
        />

        <!-- Avg Accuracy line (dashed, behind main line) -->
        {#if avgAccValues.length > 0}
          <polyline
            fill="none"
            stroke="var(--accent-green)"
            stroke-width="1.5"
            stroke-dasharray="4 2"
            opacity="0.5"
            points={chartData.map((p, i) => {
              if (p.avgAcc === null || p.avgAcc === undefined) return null;
              const x = chartPadding.left + (i / Math.max(chartData.length - 1, 1)) * chartWidth;
              const y = chartPadding.top + chartHeight - (p.avgAcc / accRange) * chartHeight;
              return `${x},${y}`;
            }).filter(Boolean).join(' ')}
          />
        {/if}

        <!-- Best Accuracy line (cumulative max) -->
        {#if accData.length > 0}
          <polyline
            fill="none"
            stroke="var(--accent-green)"
            stroke-width="2"
            points={cumulativeData.map((p, i) => {
              if (p.acc === null) return null;
              const x = chartPadding.left + (i / Math.max(cumulativeData.length - 1, 1)) * chartWidth;
              const y = chartPadding.top + chartHeight - (p.acc / accRange) * chartHeight;
              return `${x},${y}`;
            }).filter(Boolean).join(' ')}
          />
        {/if}

        <!-- Best CE marker -->
        {#each [cumulativeData.findIndex(p => p.ce === ceMin)] as bestIdx}
          {#if bestIdx >= 0}
            <circle cx={chartPadding.left + (bestIdx / Math.max(cumulativeData.length - 1, 1)) * chartWidth} cy={chartPadding.top + chartHeight - ((ceMin - combinedCeMin) / combinedCeRange) * chartHeight} r="5" fill="var(--accent-blue)" />
            <text x={chartPadding.left + (bestIdx / Math.max(cumulativeData.length - 1, 1)) * chartWidth} y={chartPadding.top + chartHeight - ((ceMin - combinedCeMin) / combinedCeRange) * chartHeight - 8} text-anchor="middle" class="best-label" fill="var(--accent-blue)">{ceMin.toFixed(4)}</text>
          {/if}
        {/each}

        <!-- Hover zones -->
        {#each chartData as point, i}
          <rect
            x={chartPadding.left + (i / Math.max(chartData.length - 1, 1)) * chartWidth - chartWidth / Math.max(chartData.length, 1) / 2}
            y={chartPadding.top}
            width={chartWidth / Math.max(chartData.length, 1)}
            height={chartHeight}
            fill="transparent"
            role="button"
            tabindex="-1"
            on:mouseenter={() => {
              const cumPoint = cumulativeData[i];
              const x = chartPadding.left + (i / Math.max(chartData.length - 1, 1)) * chartWidth;
              tooltipData = { x, y: chartPadding.top + chartHeight / 2, iter: point.iter, ce: cumPoint.ce, acc: cumPoint.acc, avgCe: point.avgCe, avgAcc: point.avgAcc };
            }}
            on:mouseleave={() => tooltipData = null}
          />
        {/each}

        <!-- Tooltip -->
        {#if tooltipData}
          <g transform="translate({tooltipData.x}, {tooltipData.y})">
            <rect x="-85" y="-70" width="170" height="120" fill="var(--bg-card)" stroke="var(--border)" rx="6" class="tooltip-bg" />
            <text x="0" y="-50" text-anchor="middle" class="tooltip-title">Iter {tooltipData.iter}</text>
            <text x="-70" y="-25" class="tooltip-label ce-label">Best CE:</text>
            <text x="70" y="-25" text-anchor="end" class="tooltip-value ce-label">{tooltipData.ce.toFixed(4)}</text>
            {#if tooltipData.avgCe !== null && tooltipData.avgCe !== undefined}
              <text x="-70" y="-5" class="tooltip-label ce-label">Avg CE:</text>
              <text x="70" y="-5" text-anchor="end" class="tooltip-value ce-label" opacity="0.7">{tooltipData.avgCe.toFixed(4)}</text>
            {/if}
            {#if tooltipData.acc !== null}
              <text x="-70" y="20" class="tooltip-label acc-label">Best Acc:</text>
              <text x="70" y="20" text-anchor="end" class="tooltip-value acc-label">{tooltipData.acc.toFixed(3)}%</text>
            {/if}
            {#if tooltipData.avgAcc !== null && tooltipData.avgAcc !== undefined}
              <text x="-70" y="40" class="tooltip-label acc-label">Avg Acc:</text>
              <text x="70" y="40" text-anchor="end" class="tooltip-value acc-label" opacity="0.7">{tooltipData.avgAcc.toFixed(3)}%</text>
            {/if}
          </g>
        {/if}
      {/if}
    </svg>
  </div>
</div>

<style>
  /* Card styles */
  .card {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: 0.5rem;
    margin-bottom: 1.5rem;
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    border-bottom: 1px solid var(--glass-border);
  }

  .card-title {
    font-weight: 600;
    color: var(--text-primary);
  }

  /* Chart */
  .chart-container {
    padding: 1rem;
  }

  .line-chart {
    width: 100%;
    height: 300px;
  }

  .chart-legend {
    display: flex;
    gap: 1rem;
    font-size: 1rem;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.25rem;
  }

  .legend-line {
    width: 1rem;
    height: 2px;
  }

  .legend-line.ce {
    background: var(--accent-blue);
  }

  .legend-line.ce-avg {
    background: var(--accent-blue);
    opacity: 0.5;
    border-top: 1px dashed var(--accent-blue);
    height: 0;
  }

  .legend-line.acc {
    background: var(--accent-green);
  }

  .legend-line.acc-avg {
    background: var(--accent-green);
    opacity: 0.5;
    border-top: 1px dashed var(--accent-green);
    height: 0;
  }

  .axis-label {
    font-size: 1rem;
    fill: var(--text-secondary);
  }

  .x-label {
    fill: var(--text-primary);
  }

  .ce-label {
    fill: var(--accent-blue);
  }

  .acc-label {
    fill: var(--accent-green);
  }

  .best-label {
    font-size: 1rem;
    font-weight: 600;
  }

  .tooltip-bg {
    filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
  }

  .tooltip-title {
    font-size: 1rem;
    font-weight: 600;
    fill: var(--text-primary);
  }

  .tooltip-label {
    font-size: 1rem;
    fill: var(--text-secondary);
  }

  .tooltip-value {
    font-size: 1rem;
    font-family: monospace;
    font-weight: 600;
  }
</style>
