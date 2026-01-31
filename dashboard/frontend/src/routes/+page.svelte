<script lang="ts">
  import { onMount } from 'svelte';
  import {
    // V1 stores
    phases,
    currentPhase,
    iterations,
    ceHistory,
    bestMetrics,
    improvement,
    currentIteration,
    phaseProgress,
    latestHealthCheck,
    phaseSummary,
    // V2 stores
    phasesV2,
    currentPhaseV2,
    iterationsV2,
    ceHistoryV2,
    bestMetricsV2,
    improvementV2,
    currentIterationV2,
    phaseProgressV2,
    currentExperimentV2,
    // Mode toggle
    useV2Mode
  } from '$lib/stores';
  import type { Flow } from '$lib/types';

  // Current running flow (fetched from API)
  let currentFlow: Flow | null = null;

  onMount(() => {
    // Fetch the current running flow
    fetchRunningFlow();
    // Refresh every 10 seconds
    const interval = setInterval(fetchRunningFlow, 10000);
    return () => clearInterval(interval);
  });

  async function fetchRunningFlow() {
    try {
      const res = await fetch('/api/flows?status=running&limit=1');
      if (res.ok) {
        const flows = await res.json();
        currentFlow = flows.length > 0 ? flows[0] : null;
      }
    } catch (e) {
      console.error('Failed to fetch running flow:', e);
    }
  }

  // Unified accessors that switch based on mode
  $: displayPhases = $useV2Mode ? $phasesV2 : $phases;
  $: displayCurrentPhase = $useV2Mode ? $currentPhaseV2 : $currentPhase;
  $: displayIterations = $useV2Mode ? $iterationsV2 : $iterations;
  $: displayCeHistory = $useV2Mode
    ? $ceHistoryV2.map(h => ({ iter: h.iter, ce: h.ce, acc: h.acc }))
    : $ceHistory;
  $: displayBestMetrics = $useV2Mode ? $bestMetricsV2 : $bestMetrics;
  $: displayImprovement = $useV2Mode ? $improvementV2 : $improvement;
  $: displayCurrentIteration = $useV2Mode ? $currentIterationV2 : $currentIteration;
  $: displayPhaseProgress = $useV2Mode ? $phaseProgressV2 : $phaseProgress;
  $: displayMaxIterations = $useV2Mode && $currentPhaseV2 ? $currentPhaseV2.max_iterations : 250;

  // V2-specific: avg_ce and avg_accuracy for display
  $: displayAvgData = $useV2Mode ? $ceHistoryV2 : null;

  function formatCE(ce: number): string {
    if (ce === Infinity) return '—';
    return ce.toFixed(4);
  }

  function formatPct(pct: number): string {
    if (!pct && pct !== 0) return '—';
    return pct.toFixed(2) + '%';
  }

  function formatAcc(acc: number | null | undefined): string {
    if (acc === null || acc === undefined) return '—';
    return acc.toFixed(2) + '%';  // Already a percentage from log parser
  }

  function phaseShortName(name: string): string {
    // "Phase 3b: TS Connections" -> "3b TS Conn"
    const match = name.match(/Phase (\d+[ab]): (GA|TS) (\w+)/);
    if (match) {
      const type = match[3].slice(0, 4); // Neurons -> Neur, Connections -> Conn
      return `${match[1]} ${match[2]} ${type}`;
    }
    return name;
  }

  // Get latest iteration for display
  $: latestIter = displayIterations[displayIterations.length - 1];
  $: hasData = displayIterations.length > 0 || displayPhases.length > 0;
</script>

<div class="container">
  <!-- Flow/Experiment Header -->
  {#if currentFlow || ($useV2Mode && $currentExperimentV2)}
    <div class="experiment-header">
      {#if currentFlow}
        <span class="flow-name">{currentFlow.name}</span>
        <span class="separator">›</span>
      {/if}
      {#if $useV2Mode && $currentExperimentV2}
        <span class="experiment-name">{$currentExperimentV2.name}</span>
      {:else if displayCurrentPhase}
        <span class="experiment-name">{displayCurrentPhase.name}</span>
      {/if}
    </div>
  {/if}

  <!-- Summary Cards -->
  <div class="grid grid-4">
    <div class="card metric">
      <div class="metric-value">{formatCE(displayBestMetrics.bestCE)}</div>
      <div class="metric-label">Best CE</div>
      {#if displayImprovement > 0}
        <div class="metric-change positive">↓ {formatPct(displayImprovement)} from baseline</div>
      {:else if displayBestMetrics.bestCE !== Infinity}
        <div class="metric-change negative">↑ {formatPct(-displayImprovement)} from baseline</div>
      {/if}
    </div>
    <div class="card metric">
      <div class="metric-value">{formatPct(displayBestMetrics.bestAcc)}</div>
      <div class="metric-label">Best Accuracy</div>
      {#if displayBestMetrics.bestAccCE !== Infinity}
        <div class="metric-change">@ CE {formatCE(displayBestMetrics.bestAccCE)}</div>
      {/if}
    </div>
    <div class="card metric">
      <div class="metric-value">{displayCurrentPhase ? phaseShortName(displayCurrentPhase.name) : '—'}</div>
      <div class="metric-label">Current Phase</div>
      {#if displayCurrentPhase}
        <div class="metric-change">{displayCurrentPhase.status}</div>
      {/if}
    </div>
    <div class="card metric">
      <div class="metric-value">{displayCurrentIteration}/{displayMaxIterations}</div>
      <div class="metric-label">Iteration</div>
      {#if latestIter && latestIter.elapsed_secs}
        <div class="metric-change">{latestIter.elapsed_secs.toFixed(1)}s/iter</div>
      {/if}
    </div>
  </div>

  <!-- Progress Bar -->
  {#if displayCurrentPhase}
    <div class="card">
      <div class="progress-header">
        <span>{displayCurrentPhase.name}</span>
        <span>{displayPhaseProgress.toFixed(0)}%</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill" style="width: {displayPhaseProgress}%"></div>
      </div>
    </div>
  {/if}

  <!-- CE & Accuracy History Chart -->
  <div class="card">
    <div class="card-header">
      <span class="card-title">Best So Far ({displayCeHistory.length} iterations)</span>
      <div class="chart-legend">
        <span class="legend-item"><span class="legend-dot ce"></span> Min CE (↓)</span>
        <span class="legend-item"><span class="legend-dot acc"></span> Max Acc (↑)</span>
      </div>
    </div>
    {#if displayCeHistory.length > 0}
      {@const chartData = displayCeHistory}
      <!-- Compute cumulative min(CE) and max(Acc) for smooth monotonic curves -->
      {@const cumulativeData = (() => {
        let minCE = Infinity;
        let maxAcc = 0;
        return chartData.map(p => {
          minCE = Math.min(minCE, p.ce);
          if (p.acc !== null && p.acc !== undefined) {
            maxAcc = Math.max(maxAcc, p.acc);
          }
          return { iter: p.iter, ce: minCE, acc: maxAcc > 0 ? maxAcc : null };
        });
      })()}
      {@const ceMin = Math.min(...cumulativeData.map(p => p.ce))}
      {@const ceMax = Math.max(...chartData.map(p => p.ce))}
      {@const ceRange = ceMax - ceMin || 0.001}
      {@const accData = cumulativeData.filter(p => p.acc !== null && p.acc !== undefined).map(p => ({ ...p, acc: p.acc ?? 0 }))}
      {@const accMin = 0}
      {@const accMax = accData.length > 0 ? Math.max(...accData.map(p => p.acc)) : 1}
      {@const accRange = accMax - accMin || 0.001}
      <!-- Debug: Log to console when values seem wrong -->
      {@const _ = (accMax === 0 && accData.length > 0) ? console.warn('[Chart Debug] accMax=0 but accData has', accData.length, 'entries. First 5:', accData.slice(0, 5).map(d => d.acc)) : null}
      {@const padding = { top: 20, right: 60, bottom: 30, left: 60 }}
      {@const width = 800}
      {@const height = 220}
      {@const chartWidth = width - padding.left - padding.right}
      {@const chartHeight = height - padding.top - padding.bottom}
      <div class="chart-container">
        <svg viewBox="0 0 {width} {height}" class="line-chart">
          <!-- Left Y-axis labels (CE values) -->
          <text x={padding.left - 5} y={padding.top + 5} text-anchor="end" class="axis-label ce-label">{ceMax.toFixed(2)}</text>
          <text x={padding.left - 5} y={padding.top + chartHeight / 2} text-anchor="end" class="axis-label ce-label">{((ceMax + ceMin) / 2).toFixed(2)}</text>
          <text x={padding.left - 5} y={padding.top + chartHeight - 5} text-anchor="end" class="axis-label ce-label">{ceMin.toFixed(2)}</text>

          <!-- Left Y-axis title (CE) -->
          <text x="12" y={height / 2} text-anchor="middle" transform="rotate(-90, 12, {height / 2})" class="axis-title ce-label">CE Loss</text>

          <!-- Right Y-axis labels (Accuracy values) -->
          {#if accData.length > 0}
            <text x={width - padding.right + 5} y={padding.top + 5} text-anchor="start" class="axis-label acc-label">{accMax.toFixed(2)}%</text>
            <text x={width - padding.right + 5} y={padding.top + chartHeight / 2} text-anchor="start" class="axis-label acc-label">{((accMax + accMin) / 2).toFixed(2)}%</text>
            <text x={width - padding.right + 5} y={padding.top + chartHeight - 5} text-anchor="start" class="axis-label acc-label">{accMin.toFixed(2)}%</text>

            <!-- Right Y-axis title (Accuracy) -->
            <text x={width - 8} y={height / 2} text-anchor="middle" transform="rotate(90, {width - 8}, {height / 2})" class="axis-title acc-label">Accuracy %</text>
          {/if}

          <!-- X-axis labels (iterations) -->
          <text x={padding.left} y={height - 5} text-anchor="start" class="axis-label">1</text>
          <text x={padding.left + chartWidth / 2} y={height - 5} text-anchor="middle" class="axis-label">{Math.floor(chartData.length / 2)}</text>
          <text x={padding.left + chartWidth} y={height - 5} text-anchor="end" class="axis-label">{chartData.length}</text>

          <!-- Grid lines -->
          <line x1={padding.left} y1={padding.top} x2={padding.left + chartWidth} y2={padding.top} stroke="var(--border)" stroke-dasharray="4" />
          <line x1={padding.left} y1={padding.top + chartHeight / 2} x2={padding.left + chartWidth} y2={padding.top + chartHeight / 2} stroke="var(--border)" stroke-dasharray="4" />
          <line x1={padding.left} y1={padding.top + chartHeight} x2={padding.left + chartWidth} y2={padding.top + chartHeight} stroke="var(--border)" stroke-dasharray="4" />

          <!-- CE line (blue) - uses cumulative min for smooth descent -->
          <polyline
            fill="none"
            stroke="var(--accent-blue)"
            stroke-width="2"
            points={cumulativeData.map((p, i) => {
              const x = padding.left + (i / Math.max(cumulativeData.length - 1, 1)) * chartWidth;
              const y = padding.top + chartHeight - ((p.ce - ceMin) / ceRange) * chartHeight;
              return `${x},${y}`;
            }).join(' ')}
          />

          <!-- Accuracy line (green) - uses cumulative max for smooth ascent -->
          {#if accData.length > 0}
            <polyline
              fill="none"
              stroke="var(--accent-green)"
              stroke-width="2"
              stroke-opacity="0.8"
              points={cumulativeData.map((p, i) => {
                if (p.acc === null || p.acc === undefined) return null;
                const x = padding.left + (i / Math.max(cumulativeData.length - 1, 1)) * chartWidth;
                const y = padding.top + chartHeight - ((p.acc - accMin) / accRange) * chartHeight;
                return `${x},${y}`;
              }).filter(Boolean).join(' ')}
            />
          {/if}

          <!-- Best CE marker (circle where min CE was first achieved) -->
          {#each [cumulativeData.findIndex(p => p.ce === ceMin)] as bestIdx}
            {#if bestIdx >= 0}
              {@const bestX = padding.left + (bestIdx / Math.max(cumulativeData.length - 1, 1)) * chartWidth}
              {@const bestY = padding.top + chartHeight}
              <circle cx={bestX} cy={bestY} r="5" fill="var(--accent-blue)" />
              <text x={bestX} y={bestY - 8} text-anchor="middle" class="best-label" fill="var(--accent-blue)">CE: {ceMin.toFixed(4)}</text>
            {/if}
          {/each}

          <!-- Best Accuracy marker (circle where max acc was first achieved) -->
          {#if accData.length > 0}
            {#each [cumulativeData.findIndex(p => p.acc === accMax)] as bestAccIdx}
              {#if bestAccIdx >= 0}
                {@const bestX = padding.left + (bestAccIdx / Math.max(cumulativeData.length - 1, 1)) * chartWidth}
                {@const bestY = padding.top}
                <circle cx={bestX} cy={bestY} r="5" fill="var(--accent-green)" />
                <text x={bestX} y={bestY + 15} text-anchor="middle" class="best-label" fill="var(--accent-green)">Acc: {accMax.toFixed(2)}%</text>
              {/if}
            {/each}
          {/if}
        </svg>
      </div>
    {:else}
      <div class="empty-state">
        <p>Waiting for iteration data...</p>
      </div>
    {/if}
  </div>

  <!-- Phase Timeline -->
  <div class="card">
    <div class="card-header">
      <span class="card-title">Phase Progress</span>
      {#if displayCurrentPhase}
        <span class="status-badge status-{displayCurrentPhase.status}">{displayCurrentPhase.status}</span>
      {/if}
    </div>
    {#if displayPhases.length > 0}
      <div class="phase-timeline">
        {#each displayPhases as phase}
          <div class="phase-item" class:completed={phase.status === 'completed'} class:running={phase.status === 'running'}>
            <div class="phase-indicator"></div>
            <div class="phase-info">
              <div class="phase-name">{phaseShortName(phase.name)}</div>
              <div class="phase-status">{phase.status}</div>
            </div>
          </div>
        {/each}
      </div>
    {:else}
      <div class="empty-state">
        <p>Waiting for experiment data...</p>
        <p class="hint">Make sure the backend is watching a log file with LOG_PATH env var</p>
      </div>
    {/if}
  </div>

  <!-- Latest Iterations Table -->
  <div class="card">
    <div class="card-header">
      <span class="card-title">Recent Iterations</span>
      <span class="count">{displayIterations.length} iterations</span>
    </div>
    {#if displayIterations.length > 0}
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Iter</th>
              <th>Best CE</th>
              <th>Accuracy</th>
              {#if $useV2Mode}
                <th>Avg CE</th>
                <th>Avg Acc</th>
              {/if}
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            {#each displayIterations.slice(-20).reverse() as iter}
              <tr>
                <td>{iter.iteration_num}</td>
                <td class:best={iter.best_ce === displayBestMetrics.bestCE}>{formatCE(iter.best_ce)}</td>
                <td>{formatAcc(iter.best_accuracy)}</td>
                {#if $useV2Mode}
                  <td>{iter.avg_ce ? formatCE(iter.avg_ce) : '—'}</td>
                  <td>{formatAcc(iter.avg_accuracy)}</td>
                {/if}
                <td>{iter.elapsed_secs ? iter.elapsed_secs.toFixed(1) + 's' : '—'}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {:else}
      <div class="empty-state">
        <p>No iterations yet</p>
      </div>
    {/if}
  </div>

  <!-- Debug: Data Inspection (collapse by default) -->
  <details class="card debug-panel">
    <summary class="card-header" style="cursor: pointer;">
      <span class="card-title">🔍 Debug: Data Inspection ({$useV2Mode ? 'V2 DB' : 'V1 Log'})</span>
    </summary>
    <div style="padding: 1rem; font-family: monospace; font-size: 0.75rem;">
      <p><strong>Mode:</strong> {$useV2Mode ? 'V2 (Database)' : 'V1 (Log Parsing)'}</p>
      <p><strong>ceHistory:</strong> {displayCeHistory.length} entries</p>
      <p><strong>iterations:</strong> {displayIterations.length} entries</p>
      {#if displayCeHistory.length > 0}
        {@const accValues = displayCeHistory.map(h => h.acc).filter(a => a !== null && a !== undefined)}
        <p><strong>Accuracy data:</strong> {accValues.length} entries with values, {displayCeHistory.length - accValues.length} null</p>
        {#if accValues.length > 0}
          <p><strong>Acc range:</strong> [{Math.min(...accValues).toFixed(4)}%, {Math.max(...accValues).toFixed(4)}%]</p>
        {/if}
        <p><strong>Last 5 ceHistory entries:</strong></p>
        <pre style="overflow-x: auto; background: var(--bg-card); padding: 0.5rem; border-radius: 4px;">{JSON.stringify(displayCeHistory.slice(-5), null, 2)}</pre>
      {/if}
      {#if displayIterations.length > 0}
        <p><strong>Last 5 iterations:</strong></p>
        <pre style="overflow-x: auto; background: var(--bg-card); padding: 0.5rem; border-radius: 4px;">{JSON.stringify(displayIterations.slice(-5).map(i => ({
          iter: i.iteration_num,
          best_ce: i.best_ce,
          best_acc: i.best_accuracy,
          avg_ce: i.avg_ce,
          avg_acc: i.avg_accuracy
        })), null, 2)}</pre>
      {/if}
      <p><strong>bestMetrics:</strong></p>
      <pre style="overflow-x: auto; background: var(--bg-card); padding: 0.5rem; border-radius: 4px;">{JSON.stringify(displayBestMetrics, null, 2)}</pre>
      {#if $useV2Mode && $currentExperimentV2}
        <p><strong>Current Experiment (V2):</strong></p>
        <pre style="overflow-x: auto; background: var(--bg-card); padding: 0.5rem; border-radius: 4px;">{JSON.stringify($currentExperimentV2, null, 2)}</pre>
      {/if}
    </div>
  </details>

  <!-- Health Check -->
  {#if $latestHealthCheck}
    <div class="card">
      <div class="card-header">
        <span class="card-title">Latest Health Check (Full Validation)</span>
      </div>
      <div class="grid grid-3">
        <div class="metric">
          <div class="metric-value">{formatCE($latestHealthCheck.top_k_ce)}</div>
          <div class="metric-label">Top-{$latestHealthCheck.k} CE</div>
        </div>
        <div class="metric">
          <div class="metric-value">{formatPct($latestHealthCheck.top_k_accuracy)}</div>
          <div class="metric-label">Top-{$latestHealthCheck.k} Acc</div>
        </div>
        <div class="metric">
          <div class="metric-value">{formatCE($latestHealthCheck.best_ce)}</div>
          <div class="metric-label">Best CE</div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Final Phase Summary -->
  {#if $phaseSummary}
    {@const groupedRows = $phaseSummary.rows.reduce((acc, row) => {
      if (!acc[row.phase_name]) acc[row.phase_name] = [];
      acc[row.phase_name].push(row);
      return acc;
    }, {})}
    <div class="card">
      <div class="card-header">
        <span class="card-title">Final Phase Comparison (Full Validation)</span>
        <span class="status-badge status-completed">Complete</span>
      </div>
      <div class="summary-table-scroll">
        <table class="summary-table">
          <thead>
            <tr>
              <th>Phase</th>
              <th>Metric</th>
              <th class="num">CE</th>
              <th class="num">PPL</th>
              <th class="num">Accuracy</th>
            </tr>
          </thead>
          <tbody>
            {#each Object.entries(groupedRows) as [phaseName, rows], phaseIdx}
              {#each rows as row, rowIdx}
                <tr class:phase-first={rowIdx === 0} class:baseline={phaseName === 'Baseline'}>
                  {#if rowIdx === 0}
                    <td rowspan={rows.length} class="phase-name-cell">
                      {phaseName === 'Baseline' ? '📊 Baseline' : phaseName}
                    </td>
                  {/if}
                  <td class="metric-type">{row.metric_type}</td>
                  <td class="num">{row.ce.toFixed(4)}</td>
                  <td class="num">{row.ppl.toFixed(1)}</td>
                  <td class="num">{row.accuracy.toFixed(2)}%</td>
                </tr>
              {/each}
            {/each}
          </tbody>
        </table>
      </div>
    </div>
  {/if}
</div>

<style>
  .experiment-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
    background: var(--bg-card);
    border-radius: 0.5rem;
    border-left: 4px solid var(--accent-blue);
  }

  .flow-name {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
  }

  .separator {
    color: var(--text-tertiary);
    font-size: 1.2rem;
  }

  .experiment-name {
    font-size: 1rem;
    color: var(--text-secondary);
  }

  .phase-timeline {
    display: flex;
    gap: 0.5rem;
    overflow-x: auto;
    padding: 1rem 0;
  }

  .phase-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 100px;
    padding: 0.5rem;
    border-radius: 0.25rem;
    background: var(--bg-card);
  }

  .phase-item.completed {
    border-left: 3px solid var(--accent-green);
  }

  .phase-item.running {
    border-left: 3px solid var(--accent-blue);
    background: rgba(59, 130, 246, 0.1);
  }

  .phase-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--border);
    margin-bottom: 0.5rem;
  }

  .phase-item.completed .phase-indicator {
    background: var(--accent-green);
  }

  .phase-item.running .phase-indicator {
    background: var(--accent-blue);
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .phase-info {
    text-align: center;
  }

  .phase-name {
    font-size: 0.75rem;
    font-weight: 500;
  }

  .phase-status {
    font-size: 0.625rem;
    color: var(--text-secondary);
    text-transform: uppercase;
  }

  .positive {
    color: var(--accent-green);
  }

  .negative {
    color: var(--accent-red);
  }

  .progress-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
  }

  .progress-bar {
    height: 8px;
    background: var(--bg-card);
    border-radius: 4px;
    overflow: hidden;
  }

  .progress-fill {
    height: 100%;
    background: var(--accent-blue);
    transition: width 0.3s ease;
  }

  .table-scroll {
    max-height: 400px;
    overflow-y: auto;
  }

  .count {
    font-size: 0.75rem;
    color: var(--text-secondary);
  }

  .best {
    color: var(--accent-green);
    font-weight: 600;
  }

  .empty-state {
    padding: 2rem;
    text-align: center;
    color: var(--text-secondary);
  }

  .empty-state .hint {
    font-size: 0.75rem;
    margin-top: 0.5rem;
    opacity: 0.7;
  }

  .chart-container {
    padding: 0.5rem 0;
  }

  .line-chart {
    width: 100%;
    height: 200px;
    background: var(--bg-card);
    border-radius: 4px;
  }

  .axis-label {
    font-size: 11px;
    fill: var(--text-secondary);
  }

  .axis-title {
    font-size: 12px;
    fill: var(--text-secondary);
    font-weight: 500;
  }

  .best-label {
    font-size: 11px;
    font-weight: 600;
  }

  .chart-legend {
    display: flex;
    gap: 1rem;
    font-size: 0.75rem;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.25rem;
  }

  .legend-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
  }

  .legend-dot.ce {
    background: var(--accent-blue);
  }

  .legend-dot.acc {
    background: var(--accent-green);
  }

  .ce-label {
    fill: var(--accent-blue);
  }

  .acc-label {
    fill: var(--accent-green);
  }

  .chart-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
  }

  .chart-label-min {
    color: var(--accent-green);
  }

  .chart-label-max {
    color: var(--text-secondary);
    font-size: 0.625rem;
    color: var(--text-secondary);
  }

  .status-running {
    background: var(--accent-blue);
    color: white;
  }

  .status-completed {
    background: var(--accent-green);
    color: white;
  }

  /* Phase Summary Table */
  .summary-table-scroll {
    overflow-x: auto;
  }

  .summary-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
  }

  .summary-table th {
    text-align: left;
    padding: 0.75rem 0.5rem;
    border-bottom: 2px solid var(--border);
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    font-size: 0.75rem;
  }

  .summary-table th.num {
    text-align: right;
  }

  .summary-table td {
    padding: 0.5rem 0.5rem;
    border-bottom: 1px solid var(--border);
  }

  .summary-table td.num {
    text-align: right;
    font-family: monospace;
  }

  .summary-table tr.phase-first td {
    border-top: 1px solid var(--border);
  }

  .summary-table tr.baseline {
    background: rgba(59, 130, 246, 0.05);
  }

  .summary-table tr.baseline td {
    font-weight: 500;
  }

  .phase-name-cell {
    font-weight: 600;
    vertical-align: top;
    background: var(--bg-secondary);
  }

  .metric-type {
    color: var(--text-secondary);
    font-size: 0.8125rem;
  }
</style>
