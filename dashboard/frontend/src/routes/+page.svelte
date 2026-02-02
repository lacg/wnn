<script lang="ts">
  import { onMount } from 'svelte';
  import {
    phases,
    currentPhase,
    iterations,
    ceHistory,
    bestMetrics,
    improvement,
    currentIteration,
    phaseProgress,
    currentExperiment,
    latestHealthCheck,
  } from '$lib/stores';
  import type { Flow, GenomeEvaluation, Iteration, Phase } from '$lib/types';

  // Current running flow (fetched from API)
  let currentFlow: Flow | null = null;

  // Iteration details modal state
  let selectedIteration: Iteration | null = null;
  let genomeEvaluations: GenomeEvaluation[] = [];
  let loadingGenomes = false;
  let showIterationModal = false;

  // Phase history state (inline display, not modal)
  let selectedHistoryPhase: Phase | null = null;
  let phaseIterations: Iteration[] = [];
  let loadingPhaseIterations = false;

  // Chart tooltip state
  let tooltipData: { x: number; y: number; iter: number; ce: number; acc: number | null; avgCe: number | null; avgAcc: number | null } | null = null;

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

  async function openIterationDetails(iter: Iteration) {
    selectedIteration = iter;
    showIterationModal = true;
    loadingGenomes = true;
    genomeEvaluations = [];

    try {
      const res = await fetch(`/api/iterations/${iter.id}/genomes`);
      if (res.ok) {
        genomeEvaluations = await res.json();
      } else {
        console.error('Failed to fetch genome evaluations:', res.status);
      }
    } catch (e) {
      console.error('Failed to fetch genome evaluations:', e);
    } finally {
      loadingGenomes = false;
    }
  }

  function closeIterationModal() {
    showIterationModal = false;
    selectedIteration = null;
    genomeEvaluations = [];
  }

  async function selectPhaseHistory(phase: Phase) {
    // If clicking the same phase or the running phase, go back to live view
    if (selectedHistoryPhase?.id === phase.id || phase.status === 'running') {
      selectedHistoryPhase = null;
      phaseIterations = [];
      return;
    }

    selectedHistoryPhase = phase;
    loadingPhaseIterations = true;
    phaseIterations = [];

    try {
      const res = await fetch(`/api/phases/${phase.id}/iterations`);
      if (res.ok) {
        phaseIterations = await res.json();
      } else {
        console.error('Failed to fetch phase iterations:', res.status);
      }
    } catch (e) {
      console.error('Failed to fetch phase iterations:', e);
    } finally {
      loadingPhaseIterations = false;
    }
  }

  function clearPhaseHistory() {
    selectedHistoryPhase = null;
    phaseIterations = [];
  }

  function formatRole(role: string): string {
    const roleMap: Record<string, string> = {
      elite: '🏆 Elite',
      offspring: '🧬 Offspring',
      init: '🌱 Init',
      top_k: '🔝 Top-K',
      neighbor: '🔄 Neighbor',
      current: '📍 Current',
    };
    return roleMap[role] || role;
  }

  // Chart data transformation (accuracy is stored as decimal, display as percentage)
  $: displayCeHistory = $ceHistory.map(h => ({
    iter: h.iter,
    ce: h.ce,
    acc: h.acc !== null && h.acc !== undefined ? h.acc * 100 : null,
    avgCe: h.avgCe,
    avgAcc: h.avgAcc !== null && h.avgAcc !== undefined ? h.avgAcc * 100 : null
  }));

  // Chart data: use historical phase data when selected, otherwise live data
  $: displayChartData = selectedHistoryPhase
    ? phaseIterations.map(iter => ({
        iter: iter.iteration_num,
        ce: iter.best_ce,
        acc: iter.best_accuracy !== null && iter.best_accuracy !== undefined ? iter.best_accuracy * 100 : null,
        avgCe: iter.avg_ce,
        avgAcc: iter.avg_accuracy !== null && iter.avg_accuracy !== undefined ? iter.avg_accuracy * 100 : null
      }))
    : displayCeHistory;

  // Chart title: indicate if viewing history
  $: chartTitle = selectedHistoryPhase
    ? `${selectedHistoryPhase.name} (${phaseIterations.length} iterations)`
    : `Best So Far (${displayCeHistory.length} iterations)`;

  // Get max iterations from current phase, or from most recent phase if current is null
  $: displayMaxIterations = (() => {
    if ($currentPhase) return $currentPhase.max_iterations;
    // Fallback to most recent phase
    if ($phases.length > 0) {
      const recent = $phases.reduce((a, b) => a.sequence_order > b.sequence_order ? a : b);
      return recent.max_iterations;
    }
    return 250; // Default fallback
  })();

  function formatCE(ce: number): string {
    if (ce === Infinity) return '—';
    return ce.toFixed(4);
  }

  function formatPct(pct: number): string {
    if (!pct && pct !== 0) return '—';
    // API returns decimal (0.00018), display as percentage
    const val = pct * 100;
    return val.toFixed(4) + '%';
  }

  function formatAcc(acc: number | null | undefined): string {
    if (acc === null || acc === undefined) return '—';
    // API returns decimal (0.00018), display as percentage
    const pct = acc * 100;
    return pct.toFixed(4) + '%';
  }

  function formatThreshold(threshold: number | null | undefined): string {
    if (threshold === null || threshold === undefined) return '—';
    return (threshold * 100).toFixed(4) + '%';
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
  $: latestIter = $iterations[$iterations.length - 1];
  $: hasData = $iterations.length > 0 || $phases.length > 0;
</script>

<div class="container">
  <!-- Flow/Experiment Header -->
  {#if currentFlow || $currentExperiment}
    <div class="experiment-header">
      {#if currentFlow}
        <span class="flow-name">{currentFlow.name}</span>
        <span class="separator">›</span>
      {/if}
      {#if $currentExperiment}
        <span class="experiment-name">{$currentExperiment.name}</span>
      {:else if $currentPhase}
        <span class="experiment-name">{$currentPhase.name}</span>
      {/if}
    </div>
  {/if}

  <!-- Summary Cards -->
  <div class="grid grid-4">
    <div class="card metric">
      <div class="metric-value">{formatCE($bestMetrics.bestCE)}</div>
      <div class="metric-label">Best CE</div>
      {#if $improvement > 0}
        <div class="metric-change positive">↓ {formatPct($improvement)} from baseline</div>
      {:else if $bestMetrics.bestCE !== Infinity}
        <div class="metric-change negative">↑ {formatPct(-$improvement)} from baseline</div>
      {/if}
    </div>
    <div class="card metric">
      <div class="metric-value">{formatPct($bestMetrics.bestAcc)}</div>
      <div class="metric-label">Best Accuracy</div>
      {#if $bestMetrics.bestAccCE !== Infinity}
        <div class="metric-change">@ CE {formatCE($bestMetrics.bestAccCE)}</div>
      {/if}
    </div>
    <div class="card metric">
      <div class="metric-value">{$currentPhase ? phaseShortName($currentPhase.name) : '—'}</div>
      <div class="metric-label">Current Phase</div>
      {#if $currentPhase}
        <div class="metric-change">{$currentPhase.status}</div>
      {/if}
    </div>
    <div class="card metric">
      <div class="metric-value">{$currentIteration}/{displayMaxIterations}</div>
      <div class="metric-label">Iteration</div>
      {#if latestIter && latestIter.elapsed_secs}
        <div class="metric-change">{latestIter.elapsed_secs.toFixed(1)}s/iter</div>
      {/if}
    </div>
  </div>

  <!-- Progress Bar -->
  {#if $currentPhase}
    <div class="card">
      <div class="progress-header">
        <span>{$currentPhase.name}</span>
        <span>{$phaseProgress.toFixed(0)}%</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill" style="width: {$phaseProgress}%"></div>
      </div>
    </div>
  {/if}

  <!-- CE & Accuracy History Chart -->
  <div class="card">
    <div class="card-header">
      <span class="card-title">{chartTitle}</span>
      {#if selectedHistoryPhase}
        <button class="btn-small" on:click={clearPhaseHistory}>← Back to Live</button>
      {/if}
      <div class="chart-legend">
        <span class="legend-item"><span class="legend-line ce"></span> Best CE (↓)</span>
        <span class="legend-item"><span class="legend-line ce dashed"></span> Avg CE</span>
        <span class="legend-item"><span class="legend-line acc"></span> Best Acc (↑)</span>
        <span class="legend-item"><span class="legend-line acc dashed"></span> Avg Acc</span>
      </div>
    </div>
    {#if displayChartData.length > 0}
      {@const chartData = displayChartData}
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

          <!-- Average CE line (blue, dashed) - shows per-iteration avg -->
          {#if chartData.filter(p => p.avgCe !== null && p.avgCe !== undefined).length > 0}
            <polyline
              fill="none"
              stroke="var(--accent-blue)"
              stroke-width="1.5"
              stroke-opacity="0.6"
              stroke-dasharray="4,3"
              points={chartData.map((p, i) => {
                if (p.avgCe === null || p.avgCe === undefined) return null;
                const x = padding.left + (i / Math.max(chartData.length - 1, 1)) * chartWidth;
                const y = padding.top + chartHeight - ((p.avgCe - ceMin) / ceRange) * chartHeight;
                return `${x},${y}`;
              }).filter(Boolean).join(' ')}
            />
          {/if}

          <!-- Average Accuracy line (green, dashed) - shows per-iteration avg -->
          {#if chartData.filter(p => p.avgAcc !== null && p.avgAcc !== undefined).length > 0}
            <polyline
              fill="none"
              stroke="var(--accent-green)"
              stroke-width="1.5"
              stroke-opacity="0.6"
              stroke-dasharray="4,3"
              points={chartData.map((p, i) => {
                if (p.avgAcc === null || p.avgAcc === undefined) return null;
                const x = padding.left + (i / Math.max(chartData.length - 1, 1)) * chartWidth;
                const y = padding.top + chartHeight - ((p.avgAcc - accMin) / accRange) * chartHeight;
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

          <!-- Invisible hover zones for tooltip -->
          {#each chartData as point, i}
            {@const x = padding.left + (i / Math.max(chartData.length - 1, 1)) * chartWidth}
            {@const barWidth = chartWidth / Math.max(chartData.length, 1)}
            <rect
              x={x - barWidth / 2}
              y={padding.top}
              width={barWidth}
              height={chartHeight}
              fill="transparent"
              on:mouseenter={() => {
                const cumPoint = cumulativeData[i];
                tooltipData = {
                  x: x,
                  y: padding.top + chartHeight / 2,
                  iter: point.iter,
                  ce: cumPoint.ce,
                  acc: cumPoint.acc,
                  avgCe: point.avgCe,
                  avgAcc: point.avgAcc
                };
              }}
              on:mouseleave={() => tooltipData = null}
            />
          {/each}

          <!-- Tooltip -->
          {#if tooltipData}
            <g transform="translate({tooltipData.x}, {tooltipData.y})">
              <rect
                x="-120"
                y="-95"
                width="240"
                height="165"
                fill="var(--bg-card)"
                stroke="var(--border)"
                rx="6"
                class="tooltip-bg"
              />
              <text x="0" y="-68" text-anchor="middle" class="tooltip-title">Iter {tooltipData.iter}</text>
              <text x="-105" y="-38" class="tooltip-label ce-label">Best CE:</text>
              <text x="105" y="-38" text-anchor="end" class="tooltip-value ce-label">{tooltipData.ce.toFixed(4)}</text>
              {#if tooltipData.acc !== null}
                <text x="-105" y="-10" class="tooltip-label acc-label">Best Acc:</text>
                <text x="105" y="-10" text-anchor="end" class="tooltip-value acc-label">{tooltipData.acc.toFixed(3)}%</text>
              {/if}
              {#if tooltipData.avgCe !== null}
                <text x="-105" y="18" class="tooltip-label ce-label">Avg CE:</text>
                <text x="105" y="18" text-anchor="end" class="tooltip-value ce-label">{tooltipData.avgCe.toFixed(4)}</text>
              {/if}
              {#if tooltipData.avgAcc !== null}
                <text x="-105" y="46" class="tooltip-label acc-label">Avg Acc:</text>
                <text x="105" y="46" text-anchor="end" class="tooltip-value acc-label">{tooltipData.avgAcc.toFixed(3)}%</text>
              {/if}
            </g>
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
      {#if $currentPhase}
        <span class="status-badge status-{$currentPhase.status}">{$currentPhase.status}</span>
      {/if}
    </div>
    {#if $phases.length > 0}
      <div class="phase-timeline">
        {#each $phases as phase}
          <button
            class="phase-item"
            class:completed={phase.status === 'completed'}
            class:running={phase.status === 'running'}
            class:selected={selectedHistoryPhase?.id === phase.id}
            class:clickable={true}
            on:click={() => selectPhaseHistory(phase)}
            title={phase.status === 'running' ? 'Click to show live iterations' : 'Click to show phase history'}
          >
            <div class="phase-indicator"></div>
            <div class="phase-info">
              <div class="phase-name">{phaseShortName(phase.name)}</div>
              <div class="phase-status">{phase.status}</div>
              {#if phase.results && phase.results.length > 0}
                <!-- Show best CE result for completed phases -->
                {@const bestCeResult = phase.results.find(r => r.metric_type === 'best_ce')}
                {#if bestCeResult}
                  <div class="phase-result best-ce">
                    <span class="result-ce">{bestCeResult.ce.toFixed(4)}</span>
                    <span class="result-acc">{(bestCeResult.accuracy * 100).toFixed(3)}%</span>
                  </div>
                {/if}
              {:else if phase.best_ce}
                <div class="phase-ce">CE: {phase.best_ce.toFixed(4)}</div>
              {/if}
            </div>
          </button>
        {/each}
      </div>
    {:else}
      <div class="empty-state">
        <p>Waiting for experiment data...</p>
      </div>
    {/if}
  </div>

  <!-- Iterations Table (Live or Historical) -->
  <div class="card">
    <div class="card-header">
      <span class="card-title">
        {#if selectedHistoryPhase}
          {selectedHistoryPhase.name}
        {:else}
          Iterations
        {/if}
      </span>
      <div class="header-right">
        {#if selectedHistoryPhase}
          <button class="btn-link" on:click={clearPhaseHistory}>← Back to Live</button>
          <span class="count">{phaseIterations.length} iterations</span>
        {:else}
          <span class="count">{$iterations.length} iterations</span>
        {/if}
      </div>
    </div>
    {#if loadingPhaseIterations}
      <div class="empty-state">
        <p>Loading iterations...</p>
      </div>
    {:else if selectedHistoryPhase}
      {#if phaseIterations.length > 0}
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Iter</th>
                <th>Best CE</th>
                <th>Best Acc</th>
                <th>Avg CE</th>
                <th>Avg Acc</th>
                <th>Δ Prev</th>
                <th>Threshold</th>
                <th>Patience</th>
                <th>Time</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {#each [...phaseIterations].reverse() as iter}
                <tr
                  class:clickable={true}
                  on:click={() => openIterationDetails(iter)}
                  on:keydown={(e) => e.key === 'Enter' && openIterationDetails(iter)}
                  tabindex={0}
                  role="button"
                >
                  <td>{iter.iteration_num}</td>
                  <td class:best={iter.best_ce === $bestMetrics.bestCE}>{formatCE(iter.best_ce)}</td>
                  <td>{formatAcc(iter.best_accuracy)}</td>
                  <td class="avg-col">{iter.avg_ce ? formatCE(iter.avg_ce) : '—'}</td>
                  <td class="avg-col">{formatAcc(iter.avg_accuracy)}</td>
                  <td class:delta-positive={iter.delta_previous && iter.delta_previous < 0} class:delta-negative={iter.delta_previous && iter.delta_previous > 0}>
                    {iter.delta_previous !== null && iter.delta_previous !== undefined ? (iter.delta_previous < 0 ? '↓' : iter.delta_previous > 0 ? '↑' : '') + Math.abs(iter.delta_previous).toFixed(4) : '—'}
                  </td>
                  <td>{formatThreshold(iter.fitness_threshold)}</td>
                  <td>{iter.patience_counter !== null && iter.patience_max ? `${iter.patience_max - iter.patience_counter}/${iter.patience_max}` : '—'}</td>
                  <td>{iter.elapsed_secs ? iter.elapsed_secs.toFixed(1) + 's' : '—'}</td>
                  <td class="view-link">View →</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {:else}
        <div class="empty-state">
          <p>No iterations recorded for this phase</p>
        </div>
      {/if}
    {:else if $iterations.length > 0}
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Iter</th>
              <th>Best CE</th>
              <th>Best Acc</th>
              <th>Avg CE</th>
              <th>Avg Acc</th>
              <th>Δ Prev</th>
              <th>Threshold</th>
              <th>Patience</th>
              <th>Time</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {#each [...$iterations].reverse() as iter}
              <tr
                class:clickable={true}
                on:click={() => openIterationDetails(iter)}
                on:keydown={(e) => e.key === 'Enter' && openIterationDetails(iter)}
                tabindex={0}
                role="button"
              >
                <td>{iter.iteration_num}</td>
                <td class:best={iter.best_ce === $bestMetrics.bestCE}>{formatCE(iter.best_ce)}</td>
                <td>{formatAcc(iter.best_accuracy)}</td>
                <td class="avg-col">{iter.avg_ce ? formatCE(iter.avg_ce) : '—'}</td>
                <td class="avg-col">{formatAcc(iter.avg_accuracy)}</td>
                <td class:delta-positive={iter.delta_previous && iter.delta_previous < 0} class:delta-negative={iter.delta_previous && iter.delta_previous > 0}>
                  {iter.delta_previous !== null && iter.delta_previous !== undefined ? (iter.delta_previous < 0 ? '↓' : iter.delta_previous > 0 ? '↑' : '') + Math.abs(iter.delta_previous).toFixed(4) : '—'}
                </td>
                <td>{formatThreshold(iter.fitness_threshold)}</td>
                <td>{iter.patience_counter !== null && iter.patience_max ? `${iter.patience_max - iter.patience_counter}/${iter.patience_max}` : '—'}</td>
                <td>{iter.elapsed_secs ? iter.elapsed_secs.toFixed(1) + 's' : '—'}</td>
                <td class="view-link">View →</td>
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
      <span class="card-title">🔍 Debug: Data Inspection</span>
    </summary>
    <div style="padding: 1rem; font-family: monospace; font-size: 0.75rem;">
      <p><strong>ceHistory:</strong> {displayCeHistory.length} entries</p>
      <p><strong>iterations:</strong> {$iterations.length} entries</p>
      {#if displayCeHistory.length > 0}
        {@const accValues = displayCeHistory.map(h => h.acc).filter(a => a !== null && a !== undefined)}
        <p><strong>Accuracy data:</strong> {accValues.length} entries with values, {displayCeHistory.length - accValues.length} null</p>
        {#if accValues.length > 0}
          <p><strong>Acc range:</strong> [{Math.min(...accValues).toFixed(4)}%, {Math.max(...accValues).toFixed(4)}%]</p>
        {/if}
        <p><strong>Last 5 ceHistory entries:</strong></p>
        <pre style="overflow-x: auto; background: var(--bg-card); padding: 0.5rem; border-radius: 4px;">{JSON.stringify(displayCeHistory.slice(-5), null, 2)}</pre>
      {/if}
      {#if $iterations.length > 0}
        <p><strong>Last 5 iterations:</strong></p>
        <pre style="overflow-x: auto; background: var(--bg-card); padding: 0.5rem; border-radius: 4px;">{JSON.stringify($iterations.slice(-5).map(i => ({
          iter: i.iteration_num,
          best_ce: i.best_ce,
          best_acc: i.best_accuracy,
          avg_ce: i.avg_ce,
          avg_acc: i.avg_accuracy
        })), null, 2)}</pre>
      {/if}
      <p><strong>bestMetrics:</strong></p>
      <pre style="overflow-x: auto; background: var(--bg-card); padding: 0.5rem; border-radius: 4px;">{JSON.stringify($bestMetrics, null, 2)}</pre>
      {#if $currentExperiment}
        <p><strong>Current Experiment:</strong></p>
        <pre style="overflow-x: auto; background: var(--bg-card); padding: 0.5rem; border-radius: 4px;">{JSON.stringify($currentExperiment, null, 2)}</pre>
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
</div>

<!-- Iteration Details Modal -->
{#if showIterationModal && selectedIteration}
  <div class="modal-overlay" on:click={closeIterationModal} on:keydown={(e) => e.key === 'Escape' && closeIterationModal()} role="dialog" aria-modal="true" tabindex="-1">
    <div class="modal" on:click|stopPropagation on:keydown|stopPropagation role="document">
      <div class="modal-header">
        <h2>Iteration {selectedIteration.iteration_num} Details</h2>
        <button class="modal-close" on:click={closeIterationModal} aria-label="Close">×</button>
      </div>

      <div class="modal-body">
        <!-- Iteration Summary -->
        <div class="iteration-summary">
          <div class="summary-item">
            <span class="label">Best CE:</span>
            <span class="value">{formatCE(selectedIteration.best_ce)}</span>
          </div>
          <div class="summary-item">
            <span class="label">Best Accuracy:</span>
            <span class="value">{formatAcc(selectedIteration.best_accuracy)}</span>
          </div>
          {#if selectedIteration.avg_ce}
            <div class="summary-item">
              <span class="label">Avg CE:</span>
              <span class="value">{formatCE(selectedIteration.avg_ce)}</span>
            </div>
          {/if}
          {#if selectedIteration.avg_accuracy !== null && selectedIteration.avg_accuracy !== undefined}
            <div class="summary-item">
              <span class="label">Avg Accuracy:</span>
              <span class="value">{formatAcc(selectedIteration.avg_accuracy)}</span>
            </div>
          {/if}
          {#if selectedIteration.elite_count}
            <div class="summary-item">
              <span class="label">Elite Count:</span>
              <span class="value">{selectedIteration.elite_count}</span>
            </div>
          {/if}
          {#if selectedIteration.offspring_count}
            <div class="summary-item">
              <span class="label">Offspring:</span>
              <span class="value">{selectedIteration.offspring_viable ?? selectedIteration.offspring_count} / {selectedIteration.offspring_count}</span>
            </div>
          {/if}
          {#if selectedIteration.candidates_total}
            <div class="summary-item">
              <span class="label">Candidates:</span>
              <span class="value">{selectedIteration.candidates_total}</span>
            </div>
          {/if}
          {#if selectedIteration.baseline_ce !== null && selectedIteration.baseline_ce !== undefined}
            <div class="summary-item">
              <span class="label">Baseline CE:</span>
              <span class="value">{formatCE(selectedIteration.baseline_ce)}</span>
            </div>
          {/if}
          {#if selectedIteration.delta_baseline !== null && selectedIteration.delta_baseline !== undefined}
            <div class="summary-item">
              <span class="label">Δ from Baseline:</span>
              <span class="value" class:delta-positive={selectedIteration.delta_baseline < 0} class:delta-negative={selectedIteration.delta_baseline > 0}>
                {selectedIteration.delta_baseline < 0 ? '↓' : '↑'}{Math.abs(selectedIteration.delta_baseline).toFixed(4)}
              </span>
            </div>
          {/if}
          {#if selectedIteration.delta_previous !== null && selectedIteration.delta_previous !== undefined}
            <div class="summary-item">
              <span class="label">Δ from Previous:</span>
              <span class="value" class:delta-positive={selectedIteration.delta_previous < 0} class:delta-negative={selectedIteration.delta_previous > 0}>
                {selectedIteration.delta_previous < 0 ? '↓' : selectedIteration.delta_previous > 0 ? '↑' : ''}{Math.abs(selectedIteration.delta_previous).toFixed(4)}
              </span>
            </div>
          {/if}
          {#if selectedIteration.patience_counter !== null && selectedIteration.patience_max}
            <div class="summary-item">
              <span class="label">Patience:</span>
              <span class="value">{selectedIteration.patience_max - selectedIteration.patience_counter} / {selectedIteration.patience_max}</span>
            </div>
          {/if}
          {#if selectedIteration.fitness_threshold !== null && selectedIteration.fitness_threshold !== undefined}
            <div class="summary-item">
              <span class="label">Threshold:</span>
              <span class="value">{formatThreshold(selectedIteration.fitness_threshold)}</span>
            </div>
          {/if}
        </div>

        <!-- Genome Evaluations - Grouped by Role -->
        {#if loadingGenomes}
          <div class="loading">Loading genome evaluations...</div>
        {:else if genomeEvaluations.length === 0}
          <div class="empty-state">
            <p>No genome evaluations recorded for this iteration.</p>
            <p class="hint">Genome tracking may not be enabled for this experiment.</p>
          </div>
        {:else}
          {@const currentGenome = genomeEvaluations.find(g => g.role === 'current')}
          {@const topKGenomes = genomeEvaluations.filter(g => g.role === 'top_k').sort((a, b) => a.position - b.position)}
          {@const newNeighbors = genomeEvaluations.filter(g => g.role === 'neighbor').sort((a, b) => a.position - b.position)}
          {@const gaElites = genomeEvaluations.filter(g => g.role === 'elite').sort((a, b) => a.position - b.position)}
          {@const gaOffspring = genomeEvaluations.filter(g => g.role === 'offspring').sort((a, b) => a.position - b.position)}

          <!-- Current Best (TS) -->
          {#if currentGenome}
            <h3>📍 Current Best</h3>
            <div class="genome-summary">
              <span class="metric">CE: <strong>{formatCE(currentGenome.ce)}</strong></span>
              <span class="metric">Acc: <strong>{formatAcc(currentGenome.accuracy)}</strong></span>
            </div>
          {/if}

          <!-- Top 50 Cache (TS) -->
          {#if topKGenomes.length > 0}
            <h3>🔝 Top {topKGenomes.length} Cache</h3>
            <div class="genome-table-scroll">
              <table class="genome-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>CE</th>
                    <th>Accuracy</th>
                  </tr>
                </thead>
                <tbody>
                  {#each topKGenomes as genome}
                    <tr class="elite">
                      <td>#{genome.elite_rank !== null ? genome.elite_rank + 1 : genome.position}</td>
                      <td class:best={genome.ce === selectedIteration.best_ce}>{formatCE(genome.ce)}</td>
                      <td>{formatAcc(genome.accuracy)}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}

          <!-- New Viable Neighbors (TS) -->
          {#if newNeighbors.length > 0}
            <h3>🔄 New Viable Neighbors ({newNeighbors.length})</h3>
            <div class="genome-table-scroll">
              <table class="genome-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>CE</th>
                    <th>Accuracy</th>
                  </tr>
                </thead>
                <tbody>
                  {#each newNeighbors as genome, idx}
                    <tr>
                      <td>{idx + 1}</td>
                      <td class:best={genome.ce === selectedIteration.best_ce}>{formatCE(genome.ce)}</td>
                      <td>{formatAcc(genome.accuracy)}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}

          <!-- GA Elites -->
          {#if gaElites.length > 0}
            <h3>🏆 Elites ({gaElites.length})</h3>
            <div class="genome-table-scroll">
              <table class="genome-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>CE</th>
                    <th>Accuracy</th>
                  </tr>
                </thead>
                <tbody>
                  {#each gaElites as genome}
                    <tr class="elite">
                      <td>#{genome.elite_rank !== null ? genome.elite_rank + 1 : genome.position + 1}</td>
                      <td class:best={genome.ce === selectedIteration.best_ce}>{formatCE(genome.ce)}</td>
                      <td>{formatAcc(genome.accuracy)}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}

          <!-- GA Offspring -->
          {#if gaOffspring.length > 0}
            <h3>🧬 Offspring ({gaOffspring.length})</h3>
            <div class="genome-table-scroll">
              <table class="genome-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>CE</th>
                    <th>Accuracy</th>
                  </tr>
                </thead>
                <tbody>
                  {#each gaOffspring as genome, idx}
                    <tr>
                      <td>{idx + 1}</td>
                      <td class:best={genome.ce === selectedIteration.best_ce}>{formatCE(genome.ce)}</td>
                      <td>{formatAcc(genome.accuracy)}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}
        {/if}
      </div>
    </div>
  </div>
{/if}

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

  .btn-small {
    padding: 0.25rem 0.5rem;
    font-size: 0.75rem;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-color);
    border-radius: 0.25rem;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn-small:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
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
    min-width: 120px;
    padding: 0.5rem;
    border-radius: 0.25rem;
    background: var(--bg-card);
    border: 1px solid transparent;
    font: inherit;
    color: inherit;
    text-align: center;
  }

  .phase-item.clickable {
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .phase-item.clickable:hover {
    border-color: var(--accent-blue);
    background: rgba(59, 130, 246, 0.1);
  }

  .phase-item.clickable:focus {
    outline: 2px solid var(--accent-blue);
    outline-offset: 2px;
  }

  .phase-item:disabled {
    cursor: default;
  }

  .phase-item.selected {
    border-color: var(--accent-yellow, #eab308);
    background: rgba(234, 179, 8, 0.15);
    box-shadow: 0 0 0 2px rgba(234, 179, 8, 0.3);
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

  .phase-ce {
    font-size: 0.6875rem;
    color: var(--accent-green);
    font-family: monospace;
    margin-top: 0.25rem;
  }

  .phase-result {
    display: flex;
    gap: 0.25rem;
    font-size: 0.7rem;
    font-family: monospace;
    margin-top: 0.125rem;
    padding: 0.2rem 0.25rem;
    background: var(--bg-secondary);
    border-radius: 2px;
    white-space: nowrap;
  }

  .phase-result.best-ce {
    background: rgba(52, 211, 153, 0.1);
  }

  .phase-result.best-acc {
    background: rgba(96, 165, 250, 0.1);
  }

  .phase-result .result-label {
    color: var(--text-secondary);
    min-width: 3rem;
  }

  .phase-result .result-ce {
    color: var(--accent-green);
  }

  .phase-result .result-acc {
    color: var(--accent-blue);
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

  .header-right {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .btn-link {
    background: none;
    border: none;
    color: var(--accent-blue);
    cursor: pointer;
    font-size: 0.875rem;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    transition: background-color 0.15s ease;
  }

  .btn-link:hover {
    background: rgba(59, 130, 246, 0.1);
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
    font-size: 1.2rem;
    font-weight: 600;
    fill: var(--text-secondary);
  }

  .axis-title {
    font-size: 1.4rem;
    fill: var(--text-secondary);
    font-weight: 700;
  }

  .best-label {
    font-size: 1.2rem;
    font-weight: 700;
  }

  .tooltip-bg {
    filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3));
  }

  .tooltip-title {
    font-size: 1.4rem;
    font-weight: 700;
    fill: var(--text-primary);
  }

  .tooltip-label {
    font-size: 1.2rem;
    font-weight: 600;
    fill: var(--text-secondary);
  }

  .tooltip-value {
    font-size: 1.2rem;
    font-family: monospace;
    font-weight: 700;
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

  .legend-line {
    width: 20px;
    height: 2px;
    display: inline-block;
  }

  .legend-line.ce {
    background: var(--accent-blue);
  }

  .legend-line.acc {
    background: var(--accent-green);
  }

  .legend-line.dashed {
    background: repeating-linear-gradient(
      90deg,
      currentColor 0px,
      currentColor 4px,
      transparent 4px,
      transparent 7px
    );
    opacity: 0.6;
  }

  .legend-line.dashed.ce {
    background: repeating-linear-gradient(
      90deg,
      var(--accent-blue) 0px,
      var(--accent-blue) 4px,
      transparent 4px,
      transparent 7px
    );
  }

  .legend-line.dashed.acc {
    background: repeating-linear-gradient(
      90deg,
      var(--accent-green) 0px,
      var(--accent-green) 4px,
      transparent 4px,
      transparent 7px
    );
  }

  .ce-label {
    fill: var(--accent-blue);
  }

  .acc-label {
    fill: var(--accent-green);
  }

  .status-running {
    background: var(--accent-blue);
    color: white;
  }

  .status-completed {
    background: var(--accent-green);
    color: white;
  }

  /* Clickable iteration rows */
  tr.clickable {
    cursor: pointer;
    transition: background-color 0.15s ease;
  }

  tr.clickable:hover {
    background-color: rgba(59, 130, 246, 0.1);
  }

  tr.clickable:focus {
    outline: 2px solid var(--accent-blue);
    outline-offset: -2px;
  }

  .view-link {
    color: var(--accent-blue);
    font-size: 0.75rem;
    opacity: 0.7;
    transition: opacity 0.15s ease;
  }

  tr.clickable:hover .view-link {
    opacity: 1;
  }

  /* Modal styles */
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    backdrop-filter: blur(4px);
  }

  .modal {
    background: var(--bg);
    border-radius: 0.75rem;
    width: 90%;
    max-width: 800px;
    max-height: 85vh;
    display: flex;
    flex-direction: column;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    border: 1px solid var(--border);
  }


  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
  }

  .modal-header h2 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 600;
  }

  .modal-close {
    background: none;
    border: none;
    font-size: 1.5rem;
    cursor: pointer;
    color: var(--text-secondary);
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    transition: background-color 0.15s ease;
  }

  .modal-close:hover {
    background-color: var(--bg-card);
    color: var(--text-primary);
  }

  .modal-body {
    padding: 1.5rem;
    overflow-y: auto;
    flex: 1;
  }

  .modal-body h3 {
    margin: 1.5rem 0 0.75rem 0;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-secondary);
  }

  .modal-body h3:first-of-type {
    margin-top: 0;
  }

  .iteration-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1rem;
    background: var(--bg-card);
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1.5rem;
  }

  .summary-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .summary-item .label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
  }

  .summary-item .value {
    font-size: 1.125rem;
    font-weight: 600;
    font-family: monospace;
  }

  .loading {
    text-align: center;
    padding: 2rem;
    color: var(--text-secondary);
  }

  /* Genome summary (current best) */
  .genome-summary {
    display: flex;
    gap: 2rem;
    padding: 0.75rem 1rem;
    background: var(--bg-card);
    border-radius: 0.5rem;
    margin-bottom: 1rem;
  }

  .genome-summary .metric {
    font-family: monospace;
    font-size: 0.9rem;
  }

  .genome-summary .metric strong {
    color: var(--accent);
  }

  /* Genome table */
  .genome-table-scroll {
    max-height: 400px;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 0.5rem;
  }

  .genome-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
  }

  .genome-table th {
    text-align: left;
    padding: 0.625rem 0.75rem;
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    font-size: 0.75rem;
    position: sticky;
    top: 0;
    z-index: 1;
  }

  .genome-table td {
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
    font-family: monospace;
  }

  .genome-table tr:last-child td {
    border-bottom: none;
  }

  .genome-table tr.elite {
    background: rgba(34, 197, 94, 0.08);
  }

  .genome-table tr:hover {
    background: rgba(59, 130, 246, 0.05);
  }

  .role-cell {
    font-family: inherit;
    white-space: nowrap;
  }

  /* Delta value colors */
  .delta-positive {
    color: var(--accent-green);
    font-weight: 500;
  }

  .delta-negative {
    color: var(--accent-red);
    font-weight: 500;
  }

  /* Avg columns styling */
  .avg-col {
    color: var(--text-secondary);
    font-size: 0.875em;
  }
</style>
