<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { page } from '$app/stores';
  import type { Experiment, Iteration, GenomeEvaluation, Flow, ValidationSummary, GatingResults, Checkpoint, TierStats } from '$lib/types';
  import { formatDate } from '$lib/dateFormat';

  let experiment: Experiment | null = null;
  let iterations: Iteration[] = [];
  let flowExperiments: Experiment[] = [];
  let flow: Flow | null = null;
  let validationSummaries: ValidationSummary[] = [];
  let checkpoints: Checkpoint[] = [];
  let loading = true;
  let error: string | null = null;
  let pollInterval: ReturnType<typeof setInterval> | null = null;
  let flowPollInterval: ReturnType<typeof setInterval> | null = null;

  // Iteration detail modal state
  let selectedIteration: Iteration | null = null;
  let genomeEvaluations: GenomeEvaluation[] = [];
  let loadingGenomes = false;
  let showIterationModal = false;

  // Gating state
  let gatingLoading = false;

  // Chart tooltip state
  let tooltipData: { x: number; y: number; iter: number; ce: number; acc: number | null; avgCe: number | null; avgAcc: number | null } | null = null;

  $: experimentId = $page.params.id;

  // Reload when experimentId changes (for in-page navigation)
  $: if (experimentId) {
    loadExperiment();
  }

  async function loadExperiment() {
    loading = true;
    error = null;

    try {
      const [expRes, itersRes, summariesRes, checkpointsRes] = await Promise.all([
        fetch(`/api/experiments/${experimentId}`),
        fetch(`/api/experiments/${experimentId}/iterations?limit=500`),
        fetch(`/api/experiments/${experimentId}/summaries`),
        fetch(`/api/checkpoints?experiment_id=${experimentId}`)
      ]);

      if (!expRes.ok) throw new Error('Experiment not found');

      experiment = await expRes.json();
      iterations = itersRes.ok ? await itersRes.json() : [];
      validationSummaries = summariesRes.ok ? await summariesRes.json() : [];
      checkpoints = checkpointsRes.ok ? await checkpointsRes.json() : [];

      // Ensure arrays
      if (!Array.isArray(iterations)) iterations = [];
      if (!Array.isArray(validationSummaries)) validationSummaries = [];
      if (!Array.isArray(checkpoints)) checkpoints = [];

      // Fetch flow and its experiments if this experiment belongs to a flow
      if (experiment?.flow_id) {
        const [flowRes, flowExpsRes] = await Promise.all([
          fetch(`/api/flows/${experiment.flow_id}`),
          fetch(`/api/flows/${experiment.flow_id}/experiments`)
        ]);
        if (flowRes.ok) flow = await flowRes.json();
        if (flowExpsRes.ok) {
          const exps = await flowExpsRes.json();
          flowExperiments = Array.isArray(exps) ? exps : [];
        }
      }
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to load experiment';
    } finally {
      loading = false;
    }
  }

  // Light refresh for running experiments - only fetch new iterations and status
  async function refreshRunningExperiment() {
    if (!experiment) return;

    try {
      const [expRes, itersRes] = await Promise.all([
        fetch(`/api/experiments/${experimentId}`),
        fetch(`/api/experiments/${experimentId}/iterations?limit=500`)
      ]);

      if (expRes.ok) {
        const newExp = await expRes.json();
        // Only update fields that change during execution
        experiment.status = newExp.status;
        experiment.current_iteration = newExp.current_iteration;
        experiment.best_ce = newExp.best_ce;
        experiment.best_accuracy = newExp.best_accuracy;
        experiment.ended_at = newExp.ended_at;
        experiment.gating_status = newExp.gating_status;
        experiment.gating_results = newExp.gating_results;

        // Also update this experiment's status in flowExperiments for Flow Progress bar
        if (flowExperiments.length > 0) {
          const idx = flowExperiments.findIndex(e => e.id === experiment.id);
          if (idx >= 0) {
            flowExperiments[idx].status = newExp.status;
            flowExperiments = flowExperiments; // Trigger Svelte reactivity
          }
        }
      }

      if (itersRes.ok) {
        const newIters = await itersRes.json();
        if (Array.isArray(newIters)) {
          iterations = newIters;
        }
      }
    } catch (e) {
      // Silently fail on refresh - don't disrupt the UI
      console.error('Refresh failed:', e);
    }
  }

  // Polling for running experiments - use light refresh
  $: {
    if (experiment?.status === 'running') {
      if (!pollInterval) {
        pollInterval = setInterval(refreshRunningExperiment, 3000);
      }
    } else {
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
    }
  }

  // Refresh flow experiments (for Flow Progress bar)
  async function refreshFlowExperiments() {
    if (!experiment?.flow_id) return;
    try {
      const res = await fetch(`/api/flows/${experiment.flow_id}/experiments`);
      if (res.ok) {
        const exps = await res.json();
        if (Array.isArray(exps)) {
          flowExperiments = exps;
        }
      }
    } catch (e) {
      // Silently fail
    }
  }

  // Poll flow experiments if any experiment in the flow is running/pending
  $: flowHasActiveExperiments = flowExperiments.some(e => e.status === 'running' || e.status === 'pending' || e.status === 'queued');
  $: {
    if (experiment?.flow_id && flowHasActiveExperiments) {
      if (!flowPollInterval) {
        flowPollInterval = setInterval(refreshFlowExperiments, 10000);
      }
    } else {
      if (flowPollInterval) {
        clearInterval(flowPollInterval);
        flowPollInterval = null;
      }
    }
  }

  // Cleanup on destroy
  onDestroy(() => {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
    if (flowPollInterval) {
      clearInterval(flowPollInterval);
      flowPollInterval = null;
    }
  });

  async function openIterationDetails(iter: Iteration) {
    selectedIteration = iter;
    showIterationModal = true;
    loadingGenomes = true;
    genomeEvaluations = [];

    try {
      const res = await fetch(`/api/iterations/${iter.id}/genomes`);
      if (res.ok) {
        genomeEvaluations = await res.json();
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

  async function runGating() {
    if (!experiment || gatingLoading) return;

    gatingLoading = true;
    try {
      const res = await fetch(`/api/experiments/${experimentId}/run-gating`, {
        method: 'POST'
      });

      if (res.ok) {
        // Reload experiment to get updated gating_status
        await loadExperiment();
      } else {
        const data = await res.json();
        alert(data.error || 'Failed to start gating analysis');
      }
    } catch (e) {
      console.error('Failed to start gating:', e);
      alert('Failed to start gating analysis');
    } finally {
      gatingLoading = false;
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'running': return 'var(--accent-blue)';
      case 'completed': return 'var(--accent-green)';
      case 'failed': return 'var(--accent-red)';
      case 'cancelled': return 'var(--text-tertiary)';
      default: return 'var(--text-secondary)';
    }
  }

  function formatCE(ce: number): string {
    if (ce === Infinity) return '—';
    return ce.toFixed(4);
  }

  function formatAcc(acc: number | null | undefined): string {
    if (acc === null || acc === undefined) return '—';
    return (acc * 100).toFixed(4) + '%';
  }

  function formatAccShort(acc: number | null | undefined): string {
    if (acc === null || acc === undefined) return '—';
    return (acc * 100).toFixed(2) + '%';
  }

  function formatDuration(start: string | null, end: string | null): string {
    if (!start) return '—';
    const startDate = new Date(start);
    const endDate = end ? new Date(end) : new Date();
    const seconds = Math.floor((endDate.getTime() - startDate.getTime()) / 1000);

    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${mins}m`;
  }

  function formatRole(role: string): string {
    switch (role) {
      case 'elite': return '🏆 Elite';
      case 'top_k': return '🏆 Top-K';
      case 'offspring': return '📌 Offspring';
      case 'init': return '🌱 Init';
      case 'neighbor': return '🔗 Neighbor';
      case 'current': return '⭐ Current';
      default: return role;
    }
  }

  // Flow steps directly from DB experiments (all exist with pending/running/completed status)
  $: flowSteps = flowExperiments
    .sort((a, b) => (a.sequence_order ?? 0) - (b.sequence_order ?? 0))
    .map((exp, i) => ({
      name: exp.name,
      status: exp.status,
      id: exp.id,
      index: i
    }));

  // Chart data - iterations directly from experiment
  $: displayIterations = iterations;
  $: chartData = displayIterations.map(iter => ({
    iter: iter.iteration_num,
    ce: iter.best_ce,
    acc: iter.best_accuracy !== null ? iter.best_accuracy * 100 : null,
    avgCe: iter.avg_ce,
    avgAcc: iter.avg_accuracy !== null ? iter.avg_accuracy * 100 : null
  }));

  // Metrics
  $: bestCE = iterations.length > 0 ? Math.min(...iterations.map(i => i.best_ce)) : Infinity;
  $: bestAcc = iterations.length > 0 ? Math.max(...iterations.filter(i => i.best_accuracy !== null).map(i => i.best_accuracy!)) : null;

  // Baseline values (first iteration)
  $: baselineCE = iterations.length > 0 ? iterations[0].best_ce : null;
  $: baselineAcc = iterations.length > 0 ? iterations[0].best_accuracy : null;

  // Improvement percentages
  $: ceImprovement = baselineCE !== null && bestCE !== Infinity && baselineCE > 0
    ? ((baselineCE - bestCE) / baselineCE) * 100
    : null;
  $: accImprovement = baselineAcc !== null && bestAcc !== null && baselineAcc > 0
    ? ((bestAcc - baselineAcc) / baselineAcc) * 100
    : null;

  // Max iterations from experiment config
  $: maxIterations = experiment?.max_iterations ?? null;

  // Average seconds per iteration
  $: avgSecsPerIter = iterations.length > 0
    ? iterations.reduce((sum, i) => sum + (i.elapsed_secs ?? 0), 0) / iterations.length
    : null;

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
  $: ceRange = ceMax - ceMin || 0.001;

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

  // Get tier_stats from the final checkpoint's genome_stats
  // If no checkpoint with tier_stats exists, fall back to parsing tier_config string
  $: finalCheckpoint = checkpoints.find(c => c.checkpoint_type === 'experiment_end' && c.genome_stats?.tier_stats);

  $: tierStats: TierStats[] | null = finalCheckpoint?.genome_stats?.tier_stats ?? null;

  // Parse tier_config string for the optimize flag (not in computed tier_stats)
  // Format: "100,15,20;400,10,12;rest,5,8" or "100,15,20,true;400,10,12,false;rest,5,8,false"
  $: tierConfigOptimize: boolean[] = (() => {
    if (!experiment?.tier_config) return [];
    try {
      return experiment.tier_config.split(';').map(tierStr => {
        const parts = tierStr.trim().split(',');
        // 4th part is optional optimize flag (defaults to true for backward compat)
        return parts.length >= 4 ? parts[3].trim().toLowerCase() === 'true' : true;
      });
    } catch {
      return [];
    }
  })();

  // Fallback: parse tier_config when no computed tier_stats available
  interface ParsedTier {
    clusters: string;  // number or "rest"
    neurons: number;
    bits: number;
    optimize: boolean;
  }
  $: parsedTiers: ParsedTier[] = (() => {
    if (!experiment?.tier_config) return [];
    try {
      return experiment.tier_config.split(';').map(tierStr => {
        const parts = tierStr.trim().split(',');
        if (parts.length < 3) return null;
        const clusters = parts[0].trim();
        const neurons = parseInt(parts[1].trim());
        const bits = parseInt(parts[2].trim());
        const optimize = parts.length >= 4 ? parts[3].trim().toLowerCase() === 'true' : true;
        return { clusters, neurons, bits, optimize };
      }).filter((t): t is ParsedTier => t !== null);
    } catch {
      return [];
    }
  })();
</script>

<div class="container">
  {#if loading}
    <div class="loading">Loading experiment...</div>
  {:else if error}
    <div class="error">{error}</div>
  {:else if experiment}
    <!-- Header -->
    <div class="experiment-header">
      <div class="header-left">
        {#if experiment.flow_id}
          <a href="/flows/{experiment.flow_id}" class="back-link">&larr; Back to Flow</a>
        {:else}
          <a href="/flows" class="back-link">&larr; Flows</a>
        {/if}
        <h1>{experiment.name}</h1>
        <span class="status-badge" style="background: {getStatusColor(experiment.status)}">
          {experiment.status}
        </span>
      </div>
      <div class="header-right">
        {#if experiment.status === 'completed' && !experiment.gating_status}
          <button class="btn-secondary" on:click={runGating} disabled={gatingLoading}>
            {gatingLoading ? '⏳ Starting...' : '🎯 Run Gating Analysis'}
          </button>
        {:else if experiment.gating_status === 'pending' || experiment.gating_status === 'running'}
          <span class="gating-status running">⏳ Gating {experiment.gating_status}...</span>
        {:else if experiment.gating_status === 'failed'}
          <button class="btn-secondary" on:click={runGating} disabled={gatingLoading}>
            🔄 Retry Gating
          </button>
        {/if}
      </div>
    </div>

    <!-- Flow Progress Bar -->
    {#if flowSteps.length > 0}
      <div class="flow-progress">
        <div class="flow-progress-label">Flow Progress</div>
        <div class="flow-progress-bar">
          {#each flowSteps as step, idx}
            {@const isCurrent = step.id === experiment.id}
            {@const hasId = step.id !== null}
            <div class="flow-step" class:current={isCurrent}>
              {#if hasId && !isCurrent}
                <a href="/experiments/{step.id}" class="step-link step-{step.status}">
                  <span class="step-number">{idx + 1}</span>
                  <span class="step-name">{step.name.replace(/^Phase \d+[ab]: /, '')}</span>
                </a>
              {:else}
                <div class="step-box step-{step.status}" class:step-current={isCurrent}>
                  <span class="step-number">{idx + 1}</span>
                  <span class="step-name">{step.name.replace(/^Phase \d+[ab]: /, '')}</span>
                </div>
              {/if}
            </div>
            {#if idx < flowSteps.length - 1}
              <div class="step-connector" class:connector-done={step.status === 'completed'}></div>
            {/if}
          {/each}
        </div>
      </div>
    {/if}

    <!-- Info Cards -->
    <div class="info-cards">
      <div class="info-card">
        <span class="info-label">Best CE</span>
        <span class="info-value best">{formatCE(bestCE)}</span>
        {#if ceImprovement !== null}
          <span class="info-delta" class:improved={ceImprovement > 0} class:worsened={ceImprovement < 0}>
            {ceImprovement > 0 ? '↓' : '↑'}{Math.abs(ceImprovement).toFixed(2)}%
          </span>
        {/if}
      </div>
      <div class="info-card">
        <span class="info-label">Best Acc</span>
        <span class="info-value">{formatAcc(bestAcc)}</span>
        {#if accImprovement !== null}
          <span class="info-delta" class:improved={accImprovement > 0} class:worsened={accImprovement < 0}>
            {accImprovement > 0 ? '↑' : '↓'}{Math.abs(accImprovement).toFixed(2)}%
          </span>
        {/if}
      </div>
      <div class="info-card">
        <span class="info-label">Iterations</span>
        <span class="info-value">{iterations.length}{#if maxIterations}/{maxIterations}{/if}</span>
        {#if avgSecsPerIter !== null}
          <span class="info-subvalue">{avgSecsPerIter.toFixed(1)}s/iter</span>
        {/if}
      </div>
      <div class="info-card">
        <span class="info-label">Duration</span>
        <span class="info-value">{formatDuration(experiment.started_at, experiment.ended_at)}</span>
        {#if flow}
          <span class="info-subvalue">Flow: {formatDuration(flow.started_at, flow.completed_at)}</span>
        {/if}
      </div>
    </div>

    <!-- Validation Summaries (Init vs Final) -->
    {#if validationSummaries.length > 0}
      {@const initSummaries = validationSummaries.filter(s => s.validation_point === 'init')}
      {@const finalSummaries = validationSummaries.filter(s => s.validation_point === 'final')}
      <div class="validation-section">
        <div class="validation-header">
          <span class="validation-title">Full Validation Results</span>
          <div class="validation-legend">
            <span class="legend-item"><span class="legend-marker best-ce"></span> Best CE</span>
            <span class="legend-item"><span class="legend-marker best-acc"></span> Best Acc</span>
            <span class="legend-item"><span class="legend-marker best-fitness"></span> Best Fitness</span>
          </div>
        </div>
        <div class="validation-cards">
          {#if initSummaries.length > 0}
            <div class="validation-card init">
              <div class="card-label">Init Baseline</div>
              <div class="validation-metrics">
                {#each initSummaries as summary}
                  <div class="metric-item {summary.genome_type}">
                    <span class="metric-marker"></span>
                    <span class="metric-ce">{summary.ce.toFixed(4)}</span>
                    <span class="metric-acc">{(summary.accuracy * 100).toFixed(2)}%</span>
                  </div>
                {/each}
              </div>
            </div>
          {/if}
          {#if finalSummaries.length > 0}
            <div class="validation-card final">
              <div class="card-label">Final Results</div>
              <div class="validation-metrics">
                {#each finalSummaries as summary}
                  <div class="metric-item {summary.genome_type}">
                    <span class="metric-marker"></span>
                    <span class="metric-ce">{summary.ce.toFixed(4)}</span>
                    <span class="metric-acc">{(summary.accuracy * 100).toFixed(2)}%</span>
                  </div>
                {/each}
              </div>
            </div>
          {/if}
        </div>
      </div>
    {/if}

    <!-- Gating Results -->
    {#if experiment.gating_status === 'completed' && experiment.gating_results}
      <div class="gating-section">
        <div class="gating-header">
          <span class="gating-title">🎯 Gating Analysis Results</span>
          <span class="gating-meta">
            {experiment.gating_results.genomes_tested} genomes tested
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
              {#each experiment.gating_results.results as result}
                {@const ceDelta = result.gated_ce - result.ce}
                {@const accDelta = result.gated_acc - result.acc}
                <tr>
                  <td class="genome-type">{result.genome_type.replace('_', ' ')}</td>
                  <td class="mono">{result.ce.toFixed(4)}</td>
                  <td class="mono">{result.gated_ce.toFixed(4)}</td>
                  <td class="mono" class:delta-positive={ceDelta < 0} class:delta-negative={ceDelta > 0}>
                    {ceDelta < 0 ? '↓' : ceDelta > 0 ? '↑' : ''}{Math.abs(ceDelta).toFixed(4)}
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
        {#if experiment.gating_results.error}
          <div class="gating-error">
            Error: {experiment.gating_results.error}
          </div>
        {/if}
      </div>
    {/if}

    <!-- Tier Stats (Best Genome) - shows computed averages when available -->
    {#if tierStats && tierStats.length > 0}
      <div class="gating-section">
        <div class="gating-header">
          <span class="gating-title">📊 Tier Stats (Best Genome)</span>
          <span class="gating-meta">
            {tierStats.length} tiers
          </span>
        </div>
        <div class="gating-table-container">
          <table class="gating-table">
            <thead>
              <tr>
                <th>Tier</th>
                <th>Clusters</th>
                <th>Avg Bits</th>
                <th>Avg Neurons</th>
                <th>Bit Range</th>
                <th>Neuron Range</th>
                <th>Optimize</th>
              </tr>
            </thead>
            <tbody>
              {#each tierStats as tier, i}
                {@const optimize = tierConfigOptimize[i] ?? true}
                <tr>
                  <td class="genome-type">Tier {tier.tier_index}</td>
                  <td class="mono">{tier.cluster_count}</td>
                  <td class="mono">{tier.avg_bits.toFixed(1)}</td>
                  <td class="mono">{tier.avg_neurons.toFixed(1)}</td>
                  <td class="mono">{tier.min_bits}-{tier.max_bits}</td>
                  <td class="mono">{tier.min_neurons}-{tier.max_neurons}</td>
                  <td class="mono" class:delta-positive={optimize} class:delta-negative={!optimize}>
                    {optimize ? '✓' : '✗'}
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </div>
    {:else if parsedTiers.length > 0}
      <!-- Fallback: show configured tier values when computed stats not available -->
      <div class="gating-section">
        <div class="gating-header">
          <span class="gating-title">📊 Tier Configuration</span>
          <span class="gating-meta">
            {parsedTiers.length} tiers (configured)
          </span>
        </div>
        <div class="gating-table-container">
          <table class="gating-table">
            <thead>
              <tr>
                <th>Tier</th>
                <th>Clusters</th>
                <th>Neurons</th>
                <th>Bits</th>
                <th>Optimize</th>
              </tr>
            </thead>
            <tbody>
              {#each parsedTiers as tier, i}
                <tr>
                  <td class="genome-type">Tier {i}</td>
                  <td class="mono">{tier.clusters}</td>
                  <td class="mono">{tier.neurons}</td>
                  <td class="mono">{tier.bits}</td>
                  <td class="mono" class:delta-positive={tier.optimize} class:delta-negative={!tier.optimize}>
                    {tier.optimize ? '✓' : '✗'}
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </div>
    {/if}

    <!-- Chart -->
    {#if chartData.length > 0}
      <div class="card">
        <div class="card-header">
          <span class="card-title">
            Progress ({chartData.length} iterations)
          </span>
          <div class="chart-legend">
            <span class="legend-item"><span class="legend-line ce"></span> Best CE</span>
            <span class="legend-item"><span class="legend-line ce-avg"></span> Avg CE</span>
            <span class="legend-item"><span class="legend-line acc"></span> Best Acc</span>
            <span class="legend-item"><span class="legend-line acc-avg"></span> Avg Acc</span>
          </div>
        </div>
        <div class="chart-container">
          <svg viewBox="0 0 {chartSvgWidth} {chartSvgHeight}" class="line-chart">
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
          </svg>
        </div>
      </div>
    {/if}

    <!-- Iterations Table -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">Iterations</span>
        <span class="count">{displayIterations.length}{#if maxIterations} / {maxIterations}{/if} iterations</span>
      </div>
      {#if displayIterations.length === 0}
        <div class="empty-state">No iterations recorded</div>
      {:else}
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Iter</th>
                <th>Timestamp</th>
                <th>Best CE</th>
                <th>Best Acc</th>
                <th>Avg CE</th>
                <th>Avg Acc</th>
                <th>Threshold</th>
                <th>Δ Prev</th>
                <th>Patience</th>
                <th>Time</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {#each [...displayIterations].reverse() as iter}
                <tr
                  class="clickable"
                  on:click={() => openIterationDetails(iter)}
                  on:keydown={(e) => e.key === 'Enter' && openIterationDetails(iter)}
                  tabindex={0}
                  role="button"
                >
                  <td>{iter.iteration_num}</td>
                  <td class="timestamp">{formatDate(iter.created_at)}</td>
                  <td class:best={iter.best_ce === bestCE}>{formatCE(iter.best_ce)}</td>
                  <td>{formatAccShort(iter.best_accuracy)}</td>
                  <td class="secondary">{iter.avg_ce ? formatCE(iter.avg_ce) : '—'}</td>
                  <td class="secondary">{formatAccShort(iter.avg_accuracy)}</td>
                  <td class="secondary">{iter.fitness_threshold !== null ? formatAccShort(iter.fitness_threshold) : '—'}</td>
                  <td class:delta-positive={iter.delta_previous && iter.delta_previous < 0} class:delta-negative={iter.delta_previous && iter.delta_previous > 0}>
                    {iter.delta_previous !== null ? (iter.delta_previous < 0 ? '↓' : iter.delta_previous > 0 ? '↑' : '') + Math.abs(iter.delta_previous).toFixed(4) : '—'}
                  </td>
                  <td>{iter.patience_counter !== null && iter.patience_max ? `${iter.patience_max - iter.patience_counter}/${iter.patience_max}` : '—'}</td>
                  <td>{iter.elapsed_secs ? iter.elapsed_secs.toFixed(1) + 's' : '—'}</td>
                  <td class="view-link">View →</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    </div>
  {/if}
</div>

<!-- Iteration Details Modal -->
{#if showIterationModal && selectedIteration}
  <div class="modal-overlay" on:click={closeIterationModal} on:keydown={(e) => e.key === 'Escape' && closeIterationModal()} role="dialog" aria-modal="true" tabindex="-1">
    <div class="modal" on:click|stopPropagation on:keydown|stopPropagation role="document">
      <div class="modal-header">
        <h2>Iteration {selectedIteration.iteration_num}</h2>
        <button class="modal-close" on:click={closeIterationModal} aria-label="Close">×</button>
      </div>
      <div class="modal-body">
        <div class="iteration-summary">
          <div class="summary-item">
            <span class="label">Best CE</span>
            <span class="value">{formatCE(selectedIteration.best_ce)}</span>
          </div>
          <div class="summary-item">
            <span class="label">Best Accuracy</span>
            <span class="value">{formatAcc(selectedIteration.best_accuracy)}</span>
          </div>
          {#if selectedIteration.avg_ce}
            <div class="summary-item">
              <span class="label">Avg CE</span>
              <span class="value">{formatCE(selectedIteration.avg_ce)}</span>
            </div>
          {/if}
          {#if selectedIteration.avg_accuracy !== null && selectedIteration.avg_accuracy !== undefined}
            <div class="summary-item">
              <span class="label">Avg Accuracy</span>
              <span class="value">{formatAcc(selectedIteration.avg_accuracy)}</span>
            </div>
          {/if}
          {#if selectedIteration.delta_previous !== null}
            <div class="summary-item">
              <span class="label">Δ Previous</span>
              <span class="value" class:delta-positive={selectedIteration.delta_previous < 0} class:delta-negative={selectedIteration.delta_previous > 0}>
                {selectedIteration.delta_previous < 0 ? '↓' : '↑'}{Math.abs(selectedIteration.delta_previous).toFixed(4)}
              </span>
            </div>
          {/if}
        </div>

        {#if loadingGenomes}
          <div class="loading-inline">Loading genomes...</div>
        {:else if genomeEvaluations.length === 0}
          <div class="empty-state">No genome evaluations recorded</div>
        {:else}
          {@const elites = genomeEvaluations.filter(g => g.role === 'elite' || g.role === 'top_k').sort((a, b) => a.position - b.position)}
          {@const others = genomeEvaluations.filter(g => g.role !== 'elite' && g.role !== 'top_k').sort((a, b) => a.ce - b.ce)}

          {#if elites.length > 0}
            <h3>Top Genomes ({elites.length})</h3>
            <div class="genome-table-scroll">
              <table class="genome-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>CE</th>
                    <th>Accuracy</th>
                    <th>Role</th>
                  </tr>
                </thead>
                <tbody>
                  {#each elites as genome}
                    <tr class="elite">
                      <td>{genome.elite_rank !== null ? genome.elite_rank + 1 : genome.position + 1}</td>
                      <td class:best={genome.ce === selectedIteration.best_ce}>{formatCE(genome.ce)}</td>
                      <td>{formatAcc(genome.accuracy)}</td>
                      <td>{formatRole(genome.role)}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}

          {#if others.length > 0}
            <h3>Other Genomes ({others.length})</h3>
            <div class="genome-table-scroll">
              <table class="genome-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>CE</th>
                    <th>Accuracy</th>
                    <th>Role</th>
                  </tr>
                </thead>
                <tbody>
                  {#each others as genome, idx}
                    <tr>
                      <td>{idx + 1}</td>
                      <td>{formatCE(genome.ce)}</td>
                      <td>{formatAcc(genome.accuracy)}</td>
                      <td>{formatRole(genome.role)}</td>
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
  .container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 1rem;
  }

  .loading, .error {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--text-secondary);
  }

  .error {
    color: var(--accent-red);
  }

  .experiment-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1rem;
    padding-top: 1rem;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
  }

  .back-link {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 1rem;
  }

  .back-link:hover {
    color: var(--text-primary);
  }

  h1 {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }

  .status-badge {
    font-size: 1rem;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    color: white;
    text-transform: capitalize;
  }

  /* Flow Progress Bar */
  .flow-progress {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
    overflow-x: auto;
  }

  .flow-progress-label {
    font-size: 1rem;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
  }

  .flow-progress-bar {
    display: flex;
    align-items: center;
    gap: 0;
    min-width: max-content;
  }

  .flow-step {
    flex-shrink: 0;
  }

  .step-link, .step-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 0.375rem 0.75rem;
    border-radius: 0.375rem;
    font-size: 1rem;
    text-decoration: none;
    min-width: 5rem;
    text-align: center;
    transition: all 0.15s;
  }

  .step-link {
    cursor: pointer;
  }

  .step-link:hover {
    transform: translateY(-1px);
  }

  .step-number {
    font-weight: 600;
    font-size: 1rem;
  }

  .step-name {
    font-size: 1rem;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 8rem;
  }

  .step-completed {
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid var(--accent-green);
    color: var(--accent-green);
  }

  .step-completed .step-name {
    color: var(--accent-green);
  }

  .step-running {
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid var(--accent-blue);
    color: var(--accent-blue);
  }

  .step-running .step-name {
    color: var(--accent-blue);
  }

  .step-pending {
    background: rgba(128, 128, 128, 0.2);
    border: 1px dashed var(--text-tertiary);
    color: var(--text-secondary);
  }

  .step-pending .step-name {
    color: var(--text-tertiary);
  }

  .step-current {
    box-shadow: 0 0 0 2px var(--accent-blue);
  }

  .step-connector {
    width: 1.5rem;
    height: 2px;
    background: var(--border);
    flex-shrink: 0;
  }

  .connector-done {
    background: var(--accent-green);
  }

  /* Info Cards */
  .info-cards {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
  }

  @media (max-width: 768px) {
    .info-cards {
      grid-template-columns: repeat(2, 1fr);
    }
  }

  .info-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .info-label {
    font-size: 1rem;
    color: var(--text-primary);
  }

  .info-value {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
    font-family: monospace;
  }

  .info-value.best {
    color: var(--accent-green);
  }

  .info-delta {
    font-size: 1rem;
    font-family: monospace;
  }

  .info-delta.improved {
    color: var(--accent-green);
  }

  .info-delta.worsened {
    color: var(--accent-red);
  }

  .info-subvalue {
    font-size: 1rem;
    color: var(--text-primary);
    font-family: monospace;
  }

  /* Validation Section */
  .validation-section {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
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
    border-bottom: 1px solid var(--border);
  }

  .validation-title {
    font-weight: 600;
    color: var(--text-primary);
  }

  .validation-legend {
    display: flex;
    gap: 1rem;
    font-size: 0.8rem;
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

  .validation-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
  }

  .validation-card {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    padding: 0.75rem;
  }

  .validation-card.init {
    border-top: 3px solid var(--accent-blue);
  }

  .validation-card.final {
    border-top: 3px solid var(--accent-green);
  }

  .card-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .validation-card.init .card-label { color: var(--accent-blue); }
  .validation-card.final .card-label { color: var(--accent-green); }

  .validation-metrics {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
  }

  .metric-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: monospace;
    font-size: 0.85rem;
  }

  .metric-item .metric-marker {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .metric-item.best_ce .metric-marker { background: var(--accent-blue); }
  .metric-item.best_acc .metric-marker { background: var(--accent-green); }
  .metric-item.best_fitness .metric-marker { background: var(--accent-purple, #9b59b6); }

  .metric-ce {
    color: var(--text-primary);
    min-width: 70px;
  }

  .metric-acc {
    color: var(--text-secondary);
  }

  /* Card styles */
  .card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 0.5rem;
    margin-bottom: 1.5rem;
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem;
    border-bottom: 1px solid var(--border);
  }

  .card-title {
    font-weight: 600;
    color: var(--text-primary);
  }

  .count {
    font-size: 1rem;
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

  /* Table */
  .table-scroll {
    max-height: 500px;
    overflow-y: auto;
  }

  table {
    width: 100%;
    border-collapse: collapse;
  }

  th, td {
    padding: 0.5rem 0.625rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }

  th {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    background: var(--bg-tertiary);
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

  tr.clickable {
    cursor: pointer;
    transition: background-color 0.15s;
  }

  tr.clickable:hover {
    background: rgba(59, 130, 246, 0.1);
  }

  .best {
    color: var(--accent-green);
    font-weight: 600;
  }

  .secondary {
    color: var(--text-secondary);
  }

  .timestamp {
    color: var(--text-secondary);
    font-family: monospace;
    font-size: 0.85rem;
  }

  .delta-positive {
    color: var(--accent-green);
  }

  .delta-negative {
    color: var(--accent-red);
  }

  .view-link {
    color: var(--accent-blue);
    font-size: 1rem;
    opacity: 0.7;
  }

  tr.clickable:hover .view-link {
    opacity: 1;
  }

  .loading-inline {
    padding: 2rem;
    text-align: center;
    color: var(--text-secondary);
  }

  .empty-state {
    padding: 2rem;
    text-align: center;
    color: var(--text-secondary);
  }

  /* Modal */
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
    max-width: 900px;
    max-height: 80vh;
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
  }

  .modal-close {
    background: none;
    border: none;
    font-size: 1.5rem;
    cursor: pointer;
    color: var(--text-secondary);
    padding: 0.25rem;
    border-radius: 0.25rem;
  }

  .modal-close:hover {
    background: var(--bg-card);
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
    color: var(--text-secondary);
  }

  .modal-body h3:first-of-type {
    margin-top: 0;
  }

  .iteration-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(7.5rem, 1fr));
    gap: 1rem;
    background: var(--bg-card);
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
  }

  .summary-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .summary-item .label {
    font-size: 1rem;
    color: var(--text-primary);
    text-transform: uppercase;
  }

  .summary-item .value {
    font-size: 1rem;
    font-weight: 600;
    font-family: monospace;
  }

  .genome-table-scroll {
    max-height: 15rem;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 0.25rem;
  }

  .genome-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 1rem;
  }

  .genome-table th {
    background: var(--bg-card);
    font-size: 1rem;
    position: sticky;
    top: 0;
    text-align: center;
  }

  .genome-table td {
    font-family: monospace;
    text-align: center;
  }

  .genome-table tr.elite {
    background: rgba(34, 197, 94, 0.08);
  }

  .more-hint {
    text-align: center;
    padding: 0.5rem;
    font-size: 1rem;
    color: var(--text-secondary);
  }

  /* Header actions */
  .header-right {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .btn-secondary {
    padding: 0.5rem 1rem;
    font-size: 0.9rem;
    border: 1px solid var(--accent-blue);
    border-radius: 0.375rem;
    background: transparent;
    color: var(--accent-blue);
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn-secondary:hover:not(:disabled) {
    background: var(--accent-blue);
    color: white;
  }

  .btn-secondary:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .gating-status {
    font-size: 0.9rem;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
  }

  .gating-status.running {
    background: rgba(59, 130, 246, 0.15);
    color: var(--accent-blue);
  }

  /* Gating Results Section */
  .gating-section {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
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
    border-bottom: 1px solid var(--border);
  }

  .gating-title {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 1.1rem;
  }

  .gating-meta {
    font-size: 0.85rem;
    color: var(--text-secondary);
  }

  .gating-table-container {
    overflow-x: auto;
  }

  .gating-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
  }

  .gating-table th {
    background: var(--bg-tertiary);
    padding: 0.5rem 0.75rem;
    text-align: center;
    font-weight: 600;
    color: var(--text-secondary);
    font-size: 0.8rem;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
  }

  .gating-table td {
    padding: 0.5rem 0.75rem;
    text-align: center;
    border-bottom: 1px solid var(--border);
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
    font-size: 0.9rem;
  }
</style>
