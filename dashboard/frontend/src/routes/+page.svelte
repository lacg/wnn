<script lang="ts">
  import {
    phases,
    currentPhase,
    iterations,
    ceHistory,
    bestMetrics,
    improvement,
    currentIteration,
    phaseProgress,
    latestHealthCheck
  } from '$lib/stores';

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
  $: latestIter = $iterations[$iterations.length - 1];
  $: hasData = $iterations.length > 0 || $phases.length > 0;
</script>

<div class="container">
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
      <div class="metric-value">{$currentIteration}/250</div>
      <div class="metric-label">Iteration</div>
      {#if latestIter}
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

  <!-- CE History Chart -->
  <div class="card">
    <div class="card-header">
      <span class="card-title">CE Trend ({$ceHistory.length} iterations) — Lower is Better ↓</span>
    </div>
    {#if $ceHistory.length > 0}
      {@const chartData = $ceHistory}
      {@const min = Math.min(...chartData.map(p => p.ce))}
      {@const max = Math.max(...chartData.map(p => p.ce))}
      {@const range = max - min || 0.001}
      {@const padding = { top: 20, right: 20, bottom: 30, left: 60 }}
      {@const width = 800}
      {@const height = 220}
      {@const chartWidth = width - padding.left - padding.right}
      {@const chartHeight = height - padding.top - padding.bottom}
      <div class="chart-container">
        <svg viewBox="0 0 {width} {height}" class="line-chart">
          <!-- Y-axis labels (CE values) -->
          <text x={padding.left - 5} y={padding.top + 5} text-anchor="end" class="axis-label">{max.toFixed(2)}</text>
          <text x={padding.left - 5} y={padding.top + chartHeight / 2} text-anchor="end" class="axis-label">{((max + min) / 2).toFixed(2)}</text>
          <text x={padding.left - 5} y={padding.top + chartHeight - 5} text-anchor="end" class="axis-label">{min.toFixed(2)}</text>

          <!-- Y-axis title -->
          <text x="12" y={height / 2} text-anchor="middle" transform="rotate(-90, 12, {height / 2})" class="axis-title">CE Loss</text>

          <!-- X-axis labels (iterations) -->
          <text x={padding.left} y={height - 5} text-anchor="start" class="axis-label">1</text>
          <text x={padding.left + chartWidth / 2} y={height - 5} text-anchor="middle" class="axis-label">{Math.floor(chartData.length / 2)}</text>
          <text x={padding.left + chartWidth} y={height - 5} text-anchor="end" class="axis-label">{chartData.length}</text>

          <!-- X-axis title -->
          <text x={padding.left + chartWidth / 2} y={height - 18} text-anchor="middle" class="axis-title">Iteration</text>

          <!-- Grid lines -->
          <line x1={padding.left} y1={padding.top} x2={padding.left + chartWidth} y2={padding.top} stroke="var(--border)" stroke-dasharray="4" />
          <line x1={padding.left} y1={padding.top + chartHeight / 2} x2={padding.left + chartWidth} y2={padding.top + chartHeight / 2} stroke="var(--border)" stroke-dasharray="4" />
          <line x1={padding.left} y1={padding.top + chartHeight} x2={padding.left + chartWidth} y2={padding.top + chartHeight} stroke="var(--border)" stroke-dasharray="4" />

          <!-- CE line -->
          <polyline
            fill="none"
            stroke="var(--accent-blue)"
            stroke-width="2"
            points={chartData.map((p, i) => {
              const x = padding.left + (i / Math.max(chartData.length - 1, 1)) * chartWidth;
              const y = padding.top + chartHeight - ((p.ce - min) / range) * chartHeight;
              return `${x},${y}`;
            }).join(' ')}
          />

          <!-- Best CE marker (green dot at lowest point) -->
          {#each [chartData.findIndex(p => p.ce === min)] as bestIdx}
            {#if bestIdx >= 0}
              {@const bestX = padding.left + (bestIdx / Math.max(chartData.length - 1, 1)) * chartWidth}
              {@const bestY = padding.top + chartHeight - ((min - min) / range) * chartHeight}
              <circle cx={bestX} cy={bestY} r="6" fill="var(--accent-green)" />
              <text x={bestX} y={bestY - 10} text-anchor="middle" class="best-label">Best: {min.toFixed(4)}</text>
            {/if}
          {/each}
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
      <span class="count">{$iterations.length} iterations</span>
    </div>
    {#if $iterations.length > 0}
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Iter</th>
              <th>Best CE</th>
              <th>Accuracy</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            {#each $iterations.slice(-20).reverse() as iter}
              <tr>
                <td>{iter.iteration_num}</td>
                <td class:best={iter.best_ce === $bestMetrics.bestCE}>{formatCE(iter.best_ce)}</td>
                <td>{formatAcc(iter.best_accuracy)}</td>
                <td>{iter.elapsed_secs.toFixed(1)}s</td>
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

<style>
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
    fill: var(--accent-green);
    font-weight: 600;
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
</style>
