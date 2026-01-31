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
  import type { Flow, GenomeEvaluationV2, IterationV2, PhaseV2 } from '$lib/types';

  // Current running flow (fetched from API)
  let currentFlow: Flow | null = null;

  // Iteration details modal state
  let selectedIteration: IterationV2 | null = null;
  let genomeEvaluations: GenomeEvaluationV2[] = [];
  let loadingGenomes = false;
  let showIterationModal = false;

  // Phase history modal state
  let selectedPhase: PhaseV2 | null = null;
  let phaseIterations: IterationV2[] = [];
  let loadingPhaseIterations = false;
  let showPhaseModal = false;

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

  async function openIterationDetails(iter: IterationV2) {
    if (!$useV2Mode) return; // Only available in V2 mode

    selectedIteration = iter;
    showIterationModal = true;
    loadingGenomes = true;
    genomeEvaluations = [];

    try {
      const res = await fetch(`/api/v2/iterations/${iter.id}/genomes`);
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

  async function openPhaseHistory(phase: PhaseV2) {
    if (!$useV2Mode) return; // Only available in V2 mode

    selectedPhase = phase;
    showPhaseModal = true;
    loadingPhaseIterations = true;
    phaseIterations = [];

    try {
      const res = await fetch(`/api/v2/phases/${phase.id}/iterations`);
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

  function closePhaseModal() {
    showPhaseModal = false;
    selectedPhase = null;
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

  // Unified accessors that switch based on mode
  $: displayPhases = $useV2Mode ? $phasesV2 : $phases;
  $: displayCurrentPhase = $useV2Mode ? $currentPhaseV2 : $currentPhase;
  $: displayIterations = $useV2Mode ? $iterationsV2 : $iterations;

  // Debug: log when stores change
  $: if (typeof window !== 'undefined') {
    console.log('[LiveUI] Mode:', $useV2Mode ? 'V2' : 'V1');
    console.log('[LiveUI] currentPhaseV2:', $currentPhaseV2?.name || 'null', 'phases:', $phasesV2.length);
    console.log('[LiveUI] displayCurrentPhase:', displayCurrentPhase?.name || 'null');
  }
  $: displayCeHistory = $useV2Mode
    ? $ceHistoryV2.map(h => ({ iter: h.iter, ce: h.ce, acc: h.acc !== null && h.acc !== undefined ? h.acc * 100 : null }))
    : $ceHistory;
  $: displayBestMetrics = $useV2Mode ? $bestMetricsV2 : $bestMetrics;
  $: displayImprovement = $useV2Mode ? $improvementV2 : $improvement;
  $: displayCurrentIteration = $useV2Mode ? $currentIterationV2 : $currentIteration;
  $: displayPhaseProgress = $useV2Mode ? $phaseProgressV2 : $phaseProgress;
  // Get max iterations from current phase, or from most recent phase if current is null
  $: displayMaxIterations = (() => {
    if ($useV2Mode) {
      if ($currentPhaseV2) return $currentPhaseV2.max_iterations;
      // Fallback to most recent phase
      const phases = $phasesV2;
      if (phases.length > 0) {
        const recent = phases.reduce((a, b) => a.sequence_order > b.sequence_order ? a : b);
        return recent.max_iterations;
      }
    }
    return 250; // Default fallback
  })();

  // V2-specific: avg_ce and avg_accuracy for display
  $: displayAvgData = $useV2Mode ? $ceHistoryV2 : null;

  function formatCE(ce: number): string {
    if (ce === Infinity) return '—';
    return ce.toFixed(4);
  }

  function formatPct(pct: number): string {
    if (!pct && pct !== 0) return '—';
    // V2 API returns decimal (0.00018), V1 log parser returns percentage (0.018)
    const val = $useV2Mode ? pct * 100 : pct;
    return val.toFixed(2) + '%';
  }

  function formatAcc(acc: number | null | undefined): string {
    if (acc === null || acc === undefined) return '—';
    // V2 API returns decimal (0.00018), V1 log parser returns percentage (0.018)
    // V2 mode: multiply by 100 to get percentage
    const pct = $useV2Mode ? acc * 100 : acc;
    return pct.toFixed(2) + '%';
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
          <button
            class="phase-item"
            class:completed={phase.status === 'completed'}
            class:running={phase.status === 'running'}
            class:clickable={$useV2Mode}
            on:click={() => $useV2Mode && openPhaseHistory(phase)}
            disabled={!$useV2Mode}
          >
            <div class="phase-indicator"></div>
            <div class="phase-info">
              <div class="phase-name">{phaseShortName(phase.name)}</div>
              <div class="phase-status">{phase.status}</div>
              {#if phase.best_ce}
                <div class="phase-ce">CE: {phase.best_ce.toFixed(4)}</div>
              {/if}
            </div>
          </button>
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
                <th>Δ Prev</th>
                <th>Threshold</th>
                <th>Patience</th>
              {/if}
              <th>Time</th>
              {#if $useV2Mode}
                <th></th>
              {/if}
            </tr>
          </thead>
          <tbody>
            {#each displayIterations.slice(-20).reverse() as iter}
              <tr
                class:clickable={$useV2Mode}
                on:click={() => $useV2Mode && openIterationDetails(iter)}
                on:keydown={(e) => e.key === 'Enter' && $useV2Mode && openIterationDetails(iter)}
                tabindex={$useV2Mode ? 0 : -1}
                role={$useV2Mode ? 'button' : undefined}
              >
                <td>{iter.iteration_num}</td>
                <td class:best={iter.best_ce === displayBestMetrics.bestCE}>{formatCE(iter.best_ce)}</td>
                <td>{formatAcc(iter.best_accuracy)}</td>
                {#if $useV2Mode}
                  <td class:delta-positive={iter.delta_previous && iter.delta_previous < 0} class:delta-negative={iter.delta_previous && iter.delta_previous > 0}>
                    {iter.delta_previous !== null && iter.delta_previous !== undefined ? (iter.delta_previous < 0 ? '↓' : iter.delta_previous > 0 ? '↑' : '') + Math.abs(iter.delta_previous).toFixed(4) : '—'}
                  </td>
                  <td>{iter.fitness_threshold !== null && iter.fitness_threshold !== undefined ? (iter.fitness_threshold * 100).toFixed(2) + '%' : '—'}</td>
                  <td>{iter.patience_counter !== null && iter.patience_max ? `${iter.patience_max - iter.patience_counter}/${iter.patience_max}` : '—'}</td>
                {/if}
                <td>{iter.elapsed_secs ? iter.elapsed_secs.toFixed(1) + 's' : '—'}</td>
                {#if $useV2Mode}
                  <td class="view-link">View →</td>
                {/if}
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
              <span class="value">{(selectedIteration.fitness_threshold * 100).toFixed(2)}%</span>
            </div>
          {/if}
        </div>

        <!-- Genome Evaluations Table -->
        <h3>Genome Evaluations ({genomeEvaluations.length})</h3>
        {#if loadingGenomes}
          <div class="loading">Loading genome evaluations...</div>
        {:else if genomeEvaluations.length === 0}
          <div class="empty-state">
            <p>No genome evaluations recorded for this iteration.</p>
            <p class="hint">Genome tracking may not be enabled for this experiment.</p>
          </div>
        {:else}
          <div class="genome-table-scroll">
            <table class="genome-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Role</th>
                  <th>CE</th>
                  <th>Accuracy</th>
                  {#if genomeEvaluations.some(g => g.fitness_score !== null)}
                    <th>Fitness</th>
                  {/if}
                  {#if genomeEvaluations.some(g => g.elite_rank !== null)}
                    <th>Elite Rank</th>
                  {/if}
                </tr>
              </thead>
              <tbody>
                {#each genomeEvaluations as genome}
                  <tr class:elite={genome.role === 'elite' || genome.role === 'top_k'}>
                    <td>{genome.position + 1}</td>
                    <td class="role-cell">{formatRole(genome.role)}</td>
                    <td class:best={genome.ce === selectedIteration.best_ce}>{formatCE(genome.ce)}</td>
                    <td>{formatAcc(genome.accuracy)}</td>
                    {#if genomeEvaluations.some(g => g.fitness_score !== null)}
                      <td>{genome.fitness_score?.toFixed(4) ?? '—'}</td>
                    {/if}
                    {#if genomeEvaluations.some(g => g.elite_rank !== null)}
                      <td>{genome.elite_rank !== null ? `#${genome.elite_rank + 1}` : '—'}</td>
                    {/if}
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        {/if}
      </div>
    </div>
  </div>
{/if}

<!-- Phase History Modal -->
{#if showPhaseModal && selectedPhase}
  <div class="modal-overlay" on:click={closePhaseModal} on:keydown={(e) => e.key === 'Escape' && closePhaseModal()} role="dialog" aria-modal="true" tabindex="-1">
    <div class="modal modal-wide" on:click|stopPropagation on:keydown|stopPropagation role="document">
      <div class="modal-header">
        <h2>{selectedPhase.name} - History</h2>
        <button class="modal-close" on:click={closePhaseModal} aria-label="Close">×</button>
      </div>

      <div class="modal-body">
        <!-- Phase Summary -->
        <div class="iteration-summary">
          <div class="summary-item">
            <span class="label">Status:</span>
            <span class="value status-badge status-{selectedPhase.status}">{selectedPhase.status}</span>
          </div>
          <div class="summary-item">
            <span class="label">Iterations:</span>
            <span class="value">{phaseIterations.length} / {selectedPhase.max_iterations}</span>
          </div>
          {#if selectedPhase.best_ce}
            <div class="summary-item">
              <span class="label">Best CE:</span>
              <span class="value">{formatCE(selectedPhase.best_ce)}</span>
            </div>
          {/if}
          {#if selectedPhase.best_accuracy}
            <div class="summary-item">
              <span class="label">Best Accuracy:</span>
              <span class="value">{formatAcc(selectedPhase.best_accuracy)}</span>
            </div>
          {/if}
          {#if selectedPhase.started_at}
            <div class="summary-item">
              <span class="label">Started:</span>
              <span class="value">{new Date(selectedPhase.started_at).toLocaleString()}</span>
            </div>
          {/if}
          {#if selectedPhase.ended_at}
            <div class="summary-item">
              <span class="label">Ended:</span>
              <span class="value">{new Date(selectedPhase.ended_at).toLocaleString()}</span>
            </div>
          {/if}
        </div>

        <!-- Iterations Table -->
        <h3>Iterations ({phaseIterations.length})</h3>
        {#if loadingPhaseIterations}
          <div class="loading">Loading iterations...</div>
        {:else if phaseIterations.length === 0}
          <div class="empty-state">
            <p>No iterations recorded for this phase.</p>
          </div>
        {:else}
          <div class="genome-table-scroll">
            <table class="genome-table">
              <thead>
                <tr>
                  <th>Iter</th>
                  <th>Best CE</th>
                  <th>Accuracy</th>
                  <th>Avg CE</th>
                  <th>Δ Prev</th>
                  <th>Patience</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {#each phaseIterations as iter}
                  <tr
                    class:clickable={true}
                    on:click={() => openIterationDetails(iter)}
                    on:keydown={(e) => e.key === 'Enter' && openIterationDetails(iter)}
                    tabindex="0"
                    role="button"
                  >
                    <td>{iter.iteration_num}</td>
                    <td>{formatCE(iter.best_ce)}</td>
                    <td>{formatAcc(iter.best_accuracy)}</td>
                    <td>{iter.avg_ce ? formatCE(iter.avg_ce) : '—'}</td>
                    <td class:delta-positive={iter.delta_previous && iter.delta_previous < 0} class:delta-negative={iter.delta_previous && iter.delta_previous > 0}>
                      {iter.delta_previous !== null && iter.delta_previous !== undefined ? (iter.delta_previous < 0 ? '↓' : iter.delta_previous > 0 ? '↑' : '') + Math.abs(iter.delta_previous).toFixed(4) : '—'}
                    </td>
                    <td>{iter.patience_counter !== null && iter.patience_max ? `${iter.patience_max - iter.patience_counter}/${iter.patience_max}` : '—'}</td>
                    <td>{iter.elapsed_secs ? iter.elapsed_secs.toFixed(1) + 's' : '—'}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
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

  .modal.modal-wide {
    max-width: 1000px;
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
</style>
