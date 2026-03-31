<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { page } from '$app/stores';
  import type { Experiment, Iteration, GenomeEvaluation, GenomeTier, Flow, ValidationSummary, GatingResults, Checkpoint, TierStats, BitwiseClusterStat } from '$lib/types';
  import { formatDate } from '$lib/dateFormat';
  import { gatingRunUpdates } from '$lib/stores';
  import BitwiseClusterStats from '$lib/components/BitwiseClusterStats.svelte';
  import type { GatingRun } from '$lib/types';

  let experiment: Experiment | null = null;
  let iterations: Iteration[] = [];
  let flowExperiments: Experiment[] = [];
  let flow: Flow | null = null;
  let validationSummaries: ValidationSummary[] = [];
  let flowValidationSummaries: ValidationSummary[] = []; // All validation summaries for the flow
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
  let gatingRuns: GatingRun[] = [];

  // Chart tooltip state
  let tooltipData: { x: number; y: number; iter: number; ce: number; acc: number | null; avgCe: number | null; avgAcc: number | null; f1?: number | null; fpr?: number | null } | null = null;

  // Live generation progress (in-memory, no DB)
  let liveProgress: { generation: number; total_generations: number; phase: string; evaluated: number; target_count: number; viable: number | null; best_ce: number; best_acc: number; elapsed_secs: number } | null = null;

  // Grid search results
  let gridSearchResults: { rank: number; neurons: number; bits: number; ce: number; accuracy: number; fitness: number | null; count: number; elapsed?: number; f1_macro?: number | null; fpr?: number | null }[] = [];
  let expandedPopulation: { rank: number; neurons: number; bits: number; ce: number; accuracy: number; fitness: number | null; f1_macro?: number | null; fpr?: number | null }[] = [];
  let seedEvalComplete = false;
  let gridSearchLoading = false;

  $: experimentId = $page.params.id;

  // Reload when experimentId changes (for in-page navigation)
  $: if (experimentId) {
    loadExperiment();
  }

  // React to gating run updates from WebSocket
  $: if ($gatingRunUpdates && experiment && $gatingRunUpdates.experiment_id === experiment.id) {
    // Update the gating runs list
    const existingIdx = gatingRuns.findIndex(r => r.id === $gatingRunUpdates!.id);
    if (existingIdx >= 0) {
      gatingRuns[existingIdx] = $gatingRunUpdates;
    } else {
      gatingRuns = [$gatingRunUpdates, ...gatingRuns];
    }
    gatingRuns = gatingRuns; // Trigger reactivity

    // Also update experiment's gating_status for backward compat display
    experiment.gating_status = $gatingRunUpdates.status;
    experiment = experiment;
  }

  async function loadExperiment() {
    loading = true;
    error = null;

    try {
      const [expRes, itersRes, summariesRes, checkpointsRes, gatingRes] = await Promise.all([
        fetch(`/api/experiments/${experimentId}`),
        fetch(`/api/experiments/${experimentId}/iterations?limit=500`),
        fetch(`/api/experiments/${experimentId}/summaries`),
        fetch(`/api/checkpoints?experiment_id=${experimentId}`),
        fetch(`/api/experiments/${experimentId}/gating`)
      ]);

      if (!expRes.ok) throw new Error('Experiment not found');

      experiment = await expRes.json();
      iterations = itersRes.ok ? await itersRes.json() : [];
      validationSummaries = summariesRes.ok ? await summariesRes.json() : [];
      checkpoints = checkpointsRes.ok ? await checkpointsRes.json() : [];
      gatingRuns = gatingRes.ok ? await gatingRes.json() : [];

      // Ensure arrays
      if (!Array.isArray(iterations)) iterations = [];
      if (!Array.isArray(validationSummaries)) validationSummaries = [];
      if (!Array.isArray(checkpoints)) checkpoints = [];
      if (!Array.isArray(gatingRuns)) gatingRuns = [];

      // Fetch flow and its experiments if this experiment belongs to a flow
      if (experiment?.flow_id) {
        const [flowRes, flowExpsRes, flowValidationsRes] = await Promise.all([
          fetch(`/api/flows/${experiment.flow_id}`),
          fetch(`/api/flows/${experiment.flow_id}/experiments`),
          fetch(`/api/flows/${experiment.flow_id}/validations`)
        ]);
        if (flowRes.ok) flow = await flowRes.json();
        if (flowExpsRes.ok) {
          const exps = await flowExpsRes.json();
          flowExperiments = Array.isArray(exps) ? exps : [];
        }
        if (flowValidationsRes.ok) {
          const validations = await flowValidationsRes.json();
          flowValidationSummaries = Array.isArray(validations) ? validations : [];
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
    const prevStatus = experiment.status;

    try {
      const [expRes, itersRes] = await Promise.all([
        fetch(`/api/experiments/${experimentId}`),
        fetch(`/api/experiments/${experimentId}/iterations?limit=500`)
      ]);

      if (expRes.ok) {
        const newExp = await expRes.json();
        // Update fields that change during execution
        experiment.status = newExp.status;
        experiment.started_at = newExp.started_at;
        experiment.current_iteration = newExp.current_iteration;
        experiment.best_ce = newExp.best_ce;
        experiment.best_accuracy = newExp.best_accuracy;
        experiment.status_message = newExp.status_message;
        experiment.ended_at = newExp.ended_at;
        experiment.gating_status = newExp.gating_status;
        experiment.gating_results = newExp.gating_results;
        experiment = experiment; // Trigger Svelte reactivity for duration display

        // Status transition detected — do a full reload to get validation summaries,
        // flow experiments, checkpoints, etc.
        if (prevStatus !== newExp.status) {
          await loadExperiment();
          return;
        }

        // Also update this experiment's status in flowExperiments for Flow Progress bar
        if (flowExperiments.length > 0) {
          const idx = flowExperiments.findIndex(e => e.id === experiment!.id);
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

      // Fetch live generation progress (in-memory on dashboard)
      try {
        const liveRes = await fetch(`/api/experiments/${experimentId}/live-progress`);
        liveProgress = liveRes.ok ? await liveRes.json() : null;
      } catch {
        liveProgress = null;
      }

      // Also refresh flow data so duration stays in sync
      if (flow) {
        const flowRes = await fetch(`/api/flows/${flow.id}`);
        if (flowRes.ok) {
          const newFlow = await flowRes.json();
          flow.started_at = newFlow.started_at;
          flow.completed_at = newFlow.completed_at;
          flow.status = newFlow.status;
          flow = flow;
        }
      }
    } catch (e) {
      // Silently fail on refresh - don't disrupt the UI
      console.error('Refresh failed:', e);
    }
  }

  // Polling for active experiments - use light refresh
  // Poll any non-terminal status so we catch pending→running transitions
  $: {
    const isActive = experiment?.status === 'running' || experiment?.status === 'pending' || experiment?.status === 'queued';
    if (isActive) {
      if (!pollInterval) {
        pollInterval = setInterval(refreshRunningExperiment, 3000);
      }
    } else {
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
      liveProgress = null;
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
      const res = await fetch(`/api/experiments/${experimentId}/gating`, {
        method: 'POST'
      });

      if (res.ok) {
        const newRun: GatingRun = await res.json();
        // Add the new run to the list
        gatingRuns = [newRun, ...gatingRuns];
        // Update experiment status for display
        experiment.gating_status = newRun.status;
        experiment = experiment;
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

  function formatF1(f1: number | null | undefined): string {
    if (f1 === null || f1 === undefined) return '—';
    return (f1 * 100).toFixed(2) + '%';
  }

  function formatFPR(fpr: number | null | undefined): string {
    if (fpr === null || fpr === undefined) return '—';
    return (fpr * 100).toFixed(3) + '%';
  }

  function formatDuration(start: string | null, end: string | null): string {
    if (!start) return '—';
    const startDate = new Date(start);
    const endDate = end ? new Date(end) : new Date();
    const seconds = Math.max(0, Math.floor((endDate.getTime() - startDate.getTime()) / 1000));

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

  function parseTier(g: GenomeEvaluation): { neurons: string; bits: string } {
    if (!g.tiers_json) return { neurons: '—', bits: '—' };
    try {
      const t: GenomeTier[] = JSON.parse(g.tiers_json);
      if (t.length === 0) return { neurons: '—', bits: '—' };
      if (t.length === 1) return { neurons: String(t[0].neurons), bits: String(t[0].bits) };
      // Multiple tiers: show per-tier values joined with +
      return {
        neurons: t.map(tier => String(tier.neurons)).join('+'),
        bits: t.map(tier => String(tier.bits)).join('+'),
      };
    } catch { return { neurons: '—', bits: '—' }; }
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
    avgAcc: iter.avg_accuracy !== null ? iter.avg_accuracy * 100 : null,
    f1: iter.best_f1 !== null && iter.best_f1 !== undefined ? iter.best_f1 * 100 : null,
    fpr: iter.best_fpr !== null && iter.best_fpr !== undefined ? iter.best_fpr * 100 : null
  }));

  // Metrics
  $: bestCE = iterations.length > 0 ? Math.min(...iterations.map(i => i.best_ce)) : Infinity;
  $: bestAcc = iterations.length > 0 ? Math.max(...iterations.filter(i => i.best_accuracy !== null).map(i => i.best_accuracy!)) : null;
  $: bestF1 = iterations.length > 0 ? Math.max(...iterations.filter(i => i.best_f1 !== null && i.best_f1 !== undefined).map(i => i.best_f1!), 0) || null : null;
  $: bestFpr = iterations.length > 0
    ? (() => {
        const fprVals = iterations.filter(i => i.best_fpr !== null && i.best_fpr !== undefined).map(i => i.best_fpr!);
        return fprVals.length > 0 ? Math.min(...fprVals) : null;
      })()
    : null;

  // Baseline values (first iteration)
  $: baselineCE = iterations.length > 0 ? iterations[0].best_ce : null;
  $: baselineAcc = iterations.length > 0 ? iterations[0].best_accuracy : null;
  $: baselineF1 = iterations.length > 0 ? iterations[0].best_f1 ?? null : null;
  $: baselineFpr = iterations.length > 0 ? iterations[0].best_fpr ?? null : null;

  // Improvement percentages
  $: ceImprovement = baselineCE !== null && bestCE !== Infinity && baselineCE > 0
    ? ((baselineCE - bestCE) / baselineCE) * 100
    : null;
  $: accImprovement = baselineAcc !== null && bestAcc !== null && baselineAcc > 0
    ? ((bestAcc - baselineAcc) / baselineAcc) * 100
    : null;
  $: f1Improvement = baselineF1 !== null && bestF1 !== null && baselineF1 > 0
    ? ((bestF1 - baselineF1) / baselineF1) * 100
    : null;
  $: fprImprovement = baselineFpr !== null && bestFpr !== null && baselineFpr > 0
    ? ((baselineFpr - bestFpr) / baselineFpr) * 100
    : null;

  // Max iterations from experiment config
  $: maxIterations = experiment?.max_iterations ?? null;

  // Experiment type detection
  $: isIDS = experiment?.architecture_type === 'ids';
  // Grid search: detect and auto-load results
  $: isGridSearch = experiment?.name?.includes('Grid Search') ?? false;

  // Auto-load grid search genome evaluations when iterations arrive or update
  let _lastGridIterCount = 0;
  $: if (isGridSearch && iterations.length > 0 && !gridSearchLoading && iterations.length !== _lastGridIterCount) {
    _lastGridIterCount = iterations.length;
    loadGridSearchResults();
  }

  async function loadGridSearchResults() {
    if (!iterations.length) return;
    gridSearchLoading = true;
    try {
      // Each per-config iteration has one genome evaluation with (neurons, bits)
      // The final iteration (N+1) has the expanded population — skip it
      const configIters = iterations.filter(i => i.candidates_total && i.candidates_total > 1);
      const perConfigIters = configIters.length > 0
        ? iterations.filter(i => i.iteration_num <= (configIters[0]?.candidates_total ?? iterations.length))
        : iterations;

      const results: { neurons: number; bits: number; ce: number; accuracy: number; fitness: number | null; count: number; elapsed: number; f1_macro?: number | null; fpr?: number | null }[] = [];

      // Fetch genome evaluations for each per-config iteration
      const fetches = perConfigIters.map(iter =>
        fetch(`/api/iterations/${iter.id}/genomes`).then(r => r.ok ? r.json() : [])
      );
      const allEvals = await Promise.all(fetches);

      for (let i = 0; i < perConfigIters.length; i++) {
        const evals: GenomeEvaluation[] = allEvals[i];
        const iter = perConfigIters[i];
        if (evals.length === 0) continue;
        const ev = evals[0];
        let neurons = 0, bits = 0;
        if (ev.tiers_json) {
          try {
            const tiers: GenomeTier[] = JSON.parse(ev.tiers_json);
            if (tiers.length > 0) { neurons = tiers[0].neurons; bits = tiers[0].bits; }
          } catch {}
        }
        results.push({
          neurons, bits, ce: ev.ce, accuracy: ev.accuracy,
          fitness: ev.fitness_score, count: 1, elapsed: iter.elapsed_secs ?? 0,
          f1_macro: ev.f1_macro ?? iter.best_f1 ?? null, fpr: ev.fpr ?? iter.best_fpr ?? null,
        });
      }

      // Sort by fitness (lower = better), fall back to CE if fitness unavailable
      results.sort((a, b) => {
        if (a.fitness != null && b.fitness != null) return a.fitness - b.fitness;
        if (a.fitness != null) return -1;
        if (b.fitness != null) return 1;
        return a.ce - b.ce;
      });
      gridSearchResults = results.map((r, i) => ({ ...r, rank: i + 1 }));

      // Load expanded population from seed iterations or final summary
      // Three iteration phases: per-config (1..N), seed (N+1..N+K), final summary (N+K+1)
      const totalConfigs = configIters.length > 0 ? configIters[0].candidates_total ?? 0 : 0;
      const afterConfigIters = iterations.filter(i => i.iteration_num > totalConfigs);
      if (afterConfigIters.length > 0) {
        // Find the max iteration_num to identify the final summary
        const maxIterNum = Math.max(...afterConfigIters.map(i => i.iteration_num));
        const maxIter = afterConfigIters.find(i => i.iteration_num === maxIterNum)!;

        // Check if this is the final summary (has multiple genome evaluations)
        // vs a seed iteration (has exactly 1). Fetch it to check.
        const maxRes = await fetch(`/api/iterations/${maxIter.id}/genomes`);
        const maxEvals: GenomeEvaluation[] = maxRes.ok ? await maxRes.json() : [];

        if (maxEvals.length > 1) {
          // Final summary iteration — show full sorted population
          const expanded: typeof expandedPopulation = [];
          for (const ev of maxEvals) {
            let neurons = 0, bits = 0;
            if (ev.tiers_json) {
              try {
                const tiers: GenomeTier[] = JSON.parse(ev.tiers_json);
                if (tiers.length > 0) { neurons = tiers[0].neurons; bits = tiers[0].bits; }
              } catch {}
            }
            expanded.push({
              rank: ev.position + 1, neurons, bits,
              ce: ev.ce, accuracy: ev.accuracy, fitness: ev.fitness_score,
              f1_macro: ev.f1_macro, fpr: ev.fpr,
            });
          }
          expandedPopulation = expanded;
          seedEvalComplete = true;
        } else {
          // Seed evaluation still in progress — gather individual seed genomes
          const seedIters = afterConfigIters.sort((a, b) => a.iteration_num - b.iteration_num);
          const seedFetches = seedIters.map(si =>
            fetch(`/api/iterations/${si.id}/genomes`).then(r => r.ok ? r.json() : [])
          );
          const seedEvals = await Promise.all(seedFetches);
          const expanded: typeof expandedPopulation = [];
          const seedTotal = seedIters.length > 0 ? seedIters[0].candidates_total ?? 0 : 0;
          for (let s = 0; s < seedIters.length; s++) {
            const evals: GenomeEvaluation[] = seedEvals[s];
            if (evals.length === 0) continue;
            const ev = evals[0];
            let neurons = 0, bits = 0;
            if (ev.tiers_json) {
              try {
                const tiers: GenomeTier[] = JSON.parse(ev.tiers_json);
                if (tiers.length > 0) { neurons = tiers[0].neurons; bits = tiers[0].bits; }
              } catch {}
            }
            expanded.push({
              rank: s + 1, neurons, bits,
              ce: ev.ce, accuracy: ev.accuracy, fitness: ev.fitness_score,
              f1_macro: ev.f1_macro, fpr: ev.fpr,
            });
          }
          // Sort by fitness (lower = better), fall back to CE
          expanded.sort((a, b) => {
            if (a.fitness != null && b.fitness != null) return a.fitness - b.fitness;
            if (a.fitness != null) return -1;
            if (b.fitness != null) return 1;
            return a.ce - b.ce;
          });
          expanded.forEach((g, i) => g.rank = i + 1);
          expandedPopulation = expanded;
          seedEvalComplete = false;
        }
      } else {
        expandedPopulation = [];
        seedEvalComplete = false;
      }
    } catch (e) {
      console.error('Failed to load grid search results:', e);
    } finally {
      gridSearchLoading = false;
    }
  }

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

  // Get tier_stats from the final checkpoint's genome_stats
  // If no checkpoint with tier_stats exists, fall back to parsing tier_config string
  $: finalCheckpoint = checkpoints.find(c => c.checkpoint_type === 'experiment_end' && c.genome_stats?.tier_stats);

  $: tierStats = finalCheckpoint?.genome_stats?.tier_stats ?? null;

  // Bitwise cluster stats from checkpoint genome_stats (when cluster_type is 'bitwise')
  $: bitwiseClusterStats = (() => {
    const cp = checkpoints.find(c => c.genome_stats?.cluster_stats);
    return cp?.genome_stats?.cluster_stats ?? null;
  })();

  // Parse tier_config string for the optimize flag (not in computed tier_stats)
  // Format: "100,15,20;400,10,12;rest,5,8" or "100,15,20,true;400,10,12,false;rest,5,8,false"
  $: tierConfigOptimize = (() => {
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
  // Latest gating run (most recent by created_at, which is already sorted DESC from API)
  $: latestGatingRun = gatingRuns.length > 0 ? gatingRuns[0] : null;
  $: hasActiveGating = latestGatingRun && (latestGatingRun.status === 'pending' || latestGatingRun.status === 'running');
  $: hasCompletedGating = latestGatingRun && latestGatingRun.status === 'completed' && latestGatingRun.results;

  $: parsedTiers = (() => {
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

  // Cumulative validation progression: all experiments up to and including current
  interface ValidationProgressionPoint {
    label: string;
    expId: number;
    sequenceOrder: number;
    validationPoint: 'init' | 'final';
    summaries: { genomeType: string; ce: number; accuracy: number; f1_macro: number | null; fpr: number | null; threshold_metadata: any | null }[];
  }

  $: cumulativeValidationProgression = (() => {
    if (!experiment || flowValidationSummaries.length === 0 || flowExperiments.length === 0) {
      // Fall back to current experiment's validations if no flow context
      if (!experiment || validationSummaries.length === 0) return [];

      const points: ValidationProgressionPoint[] = [];
      const initSummaries = validationSummaries.filter(s => s.validation_point === 'init');
      const finalSummaries = validationSummaries.filter(s => s.validation_point === 'final');

      if (initSummaries.length > 0) {
        points.push({
          label: 'Init',
          expId: experiment.id,
          sequenceOrder: experiment.sequence_order ?? 0,
          validationPoint: 'init',
          summaries: initSummaries.map(s => ({ genomeType: s.genome_type, ce: s.ce, accuracy: s.accuracy, f1_macro: s.f1_macro, fpr: s.fpr, threshold_metadata: s.threshold_metadata }))
        });
      }
      if (finalSummaries.length > 0) {
        points.push({
          label: experiment.name.replace(/^Phase \d+[ab]: /, ''),
          expId: experiment.id,
          sequenceOrder: experiment.sequence_order ?? 0,
          validationPoint: 'final',
          summaries: finalSummaries.map(s => ({ genomeType: s.genome_type, ce: s.ce, accuracy: s.accuracy, f1_macro: s.f1_macro, fpr: s.fpr, threshold_metadata: s.threshold_metadata }))
        });
      }
      return points;
    }

    // Build cumulative progression from flow validations
    const currentSeqOrder = experiment.sequence_order ?? 0;

    // Create a map of experiment_id -> experiment info
    const expMap = new Map(flowExperiments.map(e => [e.id, e]));

    // Filter validations to only include experiments up to and including current
    const relevantValidations = flowValidationSummaries.filter(v => {
      const exp = expMap.get(v.experiment_id);
      if (!exp) return false;
      return (exp.sequence_order ?? 0) <= currentSeqOrder;
    });

    // Group by (experiment_id, validation_point)
    const grouped = new Map<string, ValidationSummary[]>();
    for (const v of relevantValidations) {
      const key = `${v.experiment_id}-${v.validation_point}`;
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key)!.push(v);
    }

    // Convert to progression points
    const points: ValidationProgressionPoint[] = [];
    for (const [key, validations] of grouped) {
      const [expIdStr, point] = key.split('-');
      const expId = parseInt(expIdStr);
      const exp = expMap.get(expId);
      if (!exp) continue;

      const seqOrder = exp.sequence_order ?? 0;

      // Label: "Init" for first init, otherwise "Phase 1a", "Phase 1b", etc.
      let label: string;
      if (point === 'init' && seqOrder === 0) {
        label = 'Init';
      } else if (point === 'init') {
        // Skip non-first init points (they're the same as previous final)
        continue;
      } else {
        label = exp.name.replace(/^Phase \d+[ab]: /, '');
      }

      points.push({
        label,
        expId,
        sequenceOrder: seqOrder,
        validationPoint: point as 'init' | 'final',
        summaries: validations.map(v => ({ genomeType: v.genome_type, ce: v.ce, accuracy: v.accuracy, f1_macro: v.f1_macro, fpr: v.fpr, threshold_metadata: v.threshold_metadata }))
      });
    }

    // Sort by sequence order, then init before final
    return points.sort((a, b) => {
      if (a.sequenceOrder !== b.sequenceOrder) return a.sequenceOrder - b.sequenceOrder;
      return a.validationPoint === 'init' ? -1 : 1;
    });
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
        {#if flow}
          <span class="flow-name-label"><a href="/flows/{flow.id}">{flow.name}</a> /</span>
        {/if}
        <h1>{experiment.name}</h1>
        <span class="status-badge" style="background: {getStatusColor(experiment.status)}">
          {experiment.status}
        </span>
      </div>
      <div class="header-right">
        {#if experiment.status === 'completed' && !hasActiveGating}
          <button class="btn-secondary" on:click={runGating} disabled={gatingLoading}>
            {gatingLoading ? '⏳ Starting...' : hasCompletedGating ? '🔄 Re-run Gating' : '🎯 Run Gating Analysis'}
          </button>
        {:else if hasActiveGating}
          <span class="gating-status running">⏳ Gating {latestGatingRun?.status}...</span>
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
      {#if isIDS}
        <div class="info-card">
          <span class="info-label">Best F1-macro</span>
          <span class="info-value best">{formatF1(bestF1)}</span>
          {#if f1Improvement !== null}
            <span class="info-delta" class:improved={f1Improvement > 0} class:worsened={f1Improvement < 0}>
              {f1Improvement > 0 ? '↑' : '↓'}{Math.abs(f1Improvement).toFixed(2)}%
            </span>
          {/if}
        </div>
        <div class="info-card">
          <span class="info-label">Best FPR</span>
          <span class="info-value">{formatFPR(bestFpr)}</span>
          {#if fprImprovement !== null}
            <span class="info-delta" class:improved={fprImprovement > 0} class:worsened={fprImprovement < 0}>
              {fprImprovement > 0 ? '↓' : '↑'}{Math.abs(fprImprovement).toFixed(2)}%
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
      {:else}
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
      {/if}
      <div class="info-card">
        <span class="info-label">{isGridSearch ? 'Configs Tested' : 'Iterations'}</span>
        <span class="info-value">{isGridSearch ? gridSearchResults.length : iterations.length}{#if !isGridSearch && maxIterations}/{maxIterations}{/if}</span>
        {#if avgSecsPerIter !== null && !isGridSearch}
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

    {#if experiment.status === 'running' && experiment.status_message}
      <div class="status-message">{experiment.status_message}</div>
    {/if}

    <!-- Cumulative Validation Progression -->
    {#if cumulativeValidationProgression.length > 0}
      <div class="validation-section">
        <div class="validation-header">
          <span class="validation-title">📈 Validation Progression</span>
          <div class="validation-legend">
            {#if isIDS}
              <span class="legend-item"><span class="legend-marker best-ce"></span> Best F1-macro</span>
              <span class="legend-item"><span class="legend-marker best-acc"></span> Best FPR</span>
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
                  <th>F1</th><th>FPR</th><th>Acc</th>
                  <th>F1</th><th>FPR</th><th>Acc</th>
                  <th>F1</th><th>FPR</th><th>Acc</th>
                  <th>F1</th><th>FPR</th><th>Acc</th>
                  <th>F1</th><th>FPR</th><th>Acc</th>
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
            {#each cumulativeValidationProgression as point, idx}
              {@const bestF1Summary = point.summaries.find(s => s.genomeType === 'best_f1')}
              {@const bestFprSummary = point.summaries.find(s => s.genomeType === 'best_fpr')}
              {@const bestFitSummary = point.summaries.find(s => s.genomeType === 'best_fitness')}
              {@const bestAccSummary = point.summaries.find(s => s.genomeType === 'best_acc')}
              {@const bestCeSummary = point.summaries.find(s => s.genomeType === 'best_ce')}
              {@const isCurrentExp = point.expId === experiment?.id}
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
                  </td>
                  {#if isIDS}
                    <td class="mono best-ce-col">{bestF1Summary?.f1_macro != null ? (bestF1Summary.f1_macro * 100).toFixed(2) + '%' : '—'}</td>
                    <td class="mono best-ce-col">{bestF1Summary?.fpr != null ? (bestF1Summary.fpr * 100).toFixed(2) + '%' : '—'}</td>
                    <td class="mono best-ce-col">{bestF1Summary ? (bestF1Summary.accuracy * 100).toFixed(2) + '%' : '—'}</td>
                    <td class="mono best-acc-col">{bestFprSummary?.f1_macro != null ? (bestFprSummary.f1_macro * 100).toFixed(2) + '%' : '—'}</td>
                    <td class="mono best-acc-col">{bestFprSummary?.fpr != null ? (bestFprSummary.fpr * 100).toFixed(2) + '%' : '—'}</td>
                    <td class="mono best-acc-col">{bestFprSummary ? (bestFprSummary.accuracy * 100).toFixed(2) + '%' : '—'}</td>
                    <td class="mono">{bestAccSummary?.f1_macro != null ? (bestAccSummary.f1_macro * 100).toFixed(2) + '%' : '—'}</td>
                    <td class="mono">{bestAccSummary?.fpr != null ? (bestAccSummary.fpr * 100).toFixed(2) + '%' : '—'}</td>
                    <td class="mono">{bestAccSummary ? (bestAccSummary.accuracy * 100).toFixed(2) + '%' : '—'}</td>
                    <td class="mono">{bestCeSummary?.f1_macro != null ? (bestCeSummary.f1_macro * 100).toFixed(2) + '%' : '—'}</td>
                    <td class="mono">{bestCeSummary?.fpr != null ? (bestCeSummary.fpr * 100).toFixed(2) + '%' : '—'}</td>
                    <td class="mono">{bestCeSummary ? (bestCeSummary.accuracy * 100).toFixed(2) + '%' : '—'}</td>
                    <td class="mono best-fit-col">{bestFitSummary?.f1_macro != null ? (bestFitSummary.f1_macro * 100).toFixed(2) + '%' : '—'}</td>
                    <td class="mono best-fit-col">{bestFitSummary?.fpr != null ? (bestFitSummary.fpr * 100).toFixed(2) + '%' : '—'}</td>
                    <td class="mono best-fit-col">{bestFitSummary ? (bestFitSummary.accuracy * 100).toFixed(2) + '%' : '—'}</td>
                  {:else}
                    <td class="mono best-ce-col">{bestF1Summary ? bestF1Summary.ce.toFixed(4) : '—'}</td>
                    <td class="mono best-ce-col">{bestF1Summary ? (bestF1Summary.accuracy * 100).toFixed(2) + '%' : '—'}</td>
                    <td class="mono best-acc-col">{bestFprSummary ? bestFprSummary.ce.toFixed(4) : '—'}</td>
                    <td class="mono best-acc-col">{bestFprSummary ? (bestFprSummary.accuracy * 100).toFixed(2) + '%' : '—'}</td>
                    <td class="mono best-fit-col">{bestFitSummary ? bestFitSummary.ce.toFixed(4) : '—'}</td>
                    <td class="mono best-fit-col">{bestFitSummary ? (bestFitSummary.accuracy * 100).toFixed(2) + '%' : '—'}</td>
                  {/if}
                </tr>
                {#if hasThresholds}
                  {#each [
                    { key: 'test_cal', label: '┣ Holdout', cls: 'threshold-holdout-row' },
                    { key: 'platt', label: '┣ Platt', cls: 'threshold-platt-row' },
                    { key: 'beta', label: '┣ Beta', cls: 'threshold-beta-row' },
                    { key: 'empirical', label: '┣ Empirical', cls: 'threshold-empirical-row' },
                    { key: 'empirical_cumulative', label: '┣ Emp-cumul', cls: 'threshold-empirical-row' },
                    { key: 'train_cal', label: '┣ Train-cal', cls: 'threshold-train-row' },
                    { key: 'fixed_05', label: '┣ Fixed 0.5', cls: 'threshold-fixed-row' },
                    { key: 'val_cal', label: '┗ Oracle', cls: 'threshold-oracle-row' },
                  ] as mode}
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
      </div>
    {/if}

    <!-- Gating Results -->
    {#if hasCompletedGating && latestGatingRun?.results}
      <div class="gating-section">
        <div class="gating-header">
          <span class="gating-title">🎯 Gating Analysis Results</span>
          <span class="gating-meta">
            {latestGatingRun.genomes_tested ?? latestGatingRun.results.length} genomes tested
            {#if latestGatingRun.started_at && latestGatingRun.completed_at}
              {@const startMs = new Date(latestGatingRun.started_at).getTime()}
              {@const endMs = new Date(latestGatingRun.completed_at).getTime()}
              {@const durationSec = Math.round((endMs - startMs) / 1000)}
              {@const durationMin = Math.floor(durationSec / 60)}
              {@const durationRemSec = durationSec % 60}
              · Duration: {durationMin}m {durationRemSec}s
            {/if}
            {#if gatingRuns.length > 1}
              · Run #{latestGatingRun.id}
            {/if}
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
              {#each latestGatingRun.results as result}
                {@const ceDelta = result.gated_ce - result.ce}
                {@const accDelta = result.gated_acc - result.acc}
                <tr>
                  <td class="genome-type">{result.genome_type.replace('_', ' ')}</td>
                  <td class="mono">{result.ce.toFixed(4)}</td>
                  <td class="mono">{result.gated_ce.toFixed(4)}</td>
                  <td class="mono" class:delta-positive={ceDelta < 0} class:delta-negative={ceDelta > 0}>
                    {ceDelta < 0 ? '↑' : ceDelta > 0 ? '↓' : ''}{Math.abs(ceDelta).toFixed(4)}
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
        {#if latestGatingRun.error}
          <div class="gating-error">
            Error: {latestGatingRun.error}
          </div>
        {/if}
      </div>
    {:else if latestGatingRun?.status === 'failed'}
      <div class="gating-section">
        <div class="gating-header">
          <span class="gating-title">🎯 Gating Analysis</span>
          <span class="gating-meta">Run #{latestGatingRun.id}</span>
        </div>
        <div class="gating-error">
          Error: {latestGatingRun.error ?? 'Gating analysis failed'}
        </div>
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
                <th>Connections</th>
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
                  <td class="mono">{tier.total_connections?.toLocaleString() ?? '—'}</td>
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

    <!-- Bitwise Cluster Stats (per-cluster view for bitwise experiments) -->
    {#if experiment.architecture_type === 'bitwise' || experiment.cluster_type === 'bitwise' || bitwiseClusterStats}
      <BitwiseClusterStats clusterStats={bitwiseClusterStats} />
    {/if}

    <!-- Chart (hidden for grid search — results shown in Grid Search Results section) -->
    {#if chartData.length > 0 && !isGridSearch}
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
    {/if}

    <!-- Grid Search Results -->
    {#if isGridSearch}
      {@const totalConfigs = iterations.length > 0 && iterations[0].candidates_total ? iterations[0].candidates_total : 0}
      {@const testedCount = gridSearchResults.length}
      {@const pendingCount = totalConfigs > testedCount ? totalConfigs - testedCount : 0}
      {@const totalElapsed = gridSearchResults.reduce((s, r) => s + (r.elapsed ?? 0), 0)}
      {@const avgTimePerConfig = testedCount > 0 ? totalElapsed / testedCount : 0}
      {@const estimatedRemaining = pendingCount * avgTimePerConfig}
      {@const progressPct = totalConfigs > 0 ? (testedCount / totalConfigs) * 100 : 0}

      <div class="gating-section" style="border-left-color: var(--accent-blue);">
        <div class="gating-header">
          <span class="gating-title">Grid Search Results</span>
          <span class="gating-meta">
            {#if testedCount > 0 && pendingCount > 0}
              {testedCount} / {totalConfigs} configs
            {:else if testedCount > 0}
              {testedCount} configs tested
            {:else if experiment?.status === 'running'}
              Evaluating...
            {/if}
          </span>
        </div>

        <!-- Progress bar (shown while running) -->
        {#if experiment?.status === 'running' && totalConfigs > 0}
          <div class="grid-progress">
            <div class="grid-progress-bar">
              <div class="grid-progress-fill" style="width: {progressPct}%"></div>
            </div>
            <div class="grid-progress-info">
              <span>{testedCount} tested, {pendingCount} remaining</span>
              {#if estimatedRemaining > 0}
                <span>~{estimatedRemaining >= 60 ? Math.ceil(estimatedRemaining / 60) + 'm' : Math.ceil(estimatedRemaining) + 's'} remaining (avg {avgTimePerConfig.toFixed(1)}s/config)</span>
              {/if}
            </div>
          </div>
        {/if}

        {#if gridSearchLoading}
          <div class="empty-state">Loading grid search results...</div>
        {:else if gridSearchResults.length > 0}
          {@const topK = 5}
          <div class="table-scroll">
            <table class="gating-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Neurons</th>
                  <th>Bits</th>
                  {#if isIDS}
                    <th>F1</th>
                    <th>FPR</th>
                  {:else}
                    <th>CE</th>
                  {/if}
                  <th>Accuracy</th>
                  <th>Fitness</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {#each gridSearchResults as r}
                  <tr class:grid-top-k={r.rank <= topK}>
                    <td class="mono">
                      {#if r.rank <= topK}
                        <span class="grid-rank-star">&#9733;</span>
                      {/if}
                      {r.rank}
                    </td>
                    <td class="mono">{r.neurons.toLocaleString()}</td>
                    <td class="mono">{r.bits}</td>
                    {#if isIDS}
                      <td class="mono">{r.f1_macro != null ? (r.f1_macro * 100).toFixed(2) + '%' : '—'}</td>
                      <td class="mono">{r.fpr != null ? (r.fpr * 100).toFixed(3) + '%' : '—'}</td>
                    {:else}
                      <td class="mono">{r.ce.toFixed(4)}</td>
                    {/if}
                    <td class="mono">{(r.accuracy * 100).toFixed(2)}%</td>
                    <td class="mono">{r.fitness !== null ? r.fitness.toFixed(4) : '—'}</td>
                    <td class="mono">{r.elapsed ? r.elapsed.toFixed(1) + 's' : '—'}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        {:else if experiment?.status === 'completed'}
          <div class="empty-state">No genome tracking data available</div>
        {:else}
          <div class="empty-state">Results will appear as configs are evaluated</div>
        {/if}
      </div>

      <!-- Seeded Population (after grid search completes, top-K seeded with fresh connections) -->
      {#if expandedPopulation.length > 0}
        {@const bestCeGenome = expandedPopulation.reduce((best, g) => g.ce < best.ce ? g : best, expandedPopulation[0])}
        {@const bestAccGenome = expandedPopulation.reduce((best, g) => g.accuracy > best.accuracy ? g : best, expandedPopulation[0])}
        {@const bestF1Genome = isIDS ? expandedPopulation.filter(g => g.f1_macro != null).reduce((best, g) => (g.f1_macro ?? 0) > (best?.f1_macro ?? 0) ? g : best, expandedPopulation[0]) : null}
        <div class="gating-section" style="border-left-color: var(--accent-green);">
          <div class="gating-header">
            <span class="gating-title">Seeded Population{#if !seedEvalComplete} (evaluating...){/if}</span>
            <span class="gating-meta">
              {expandedPopulation.length} genomes{#if !seedEvalComplete}&nbsp;so far{/if} &middot;
              {#if isIDS}
                Best F1: {bestF1Genome?.f1_macro != null ? (bestF1Genome.f1_macro * 100).toFixed(2) + '%' : '—'} ({bestF1Genome?.neurons}n {bestF1Genome?.bits}b) &middot;
                Best Acc: {(bestAccGenome.accuracy * 100).toFixed(2)}% ({bestAccGenome.neurons}n {bestAccGenome.bits}b)
              {:else}
                Best CE: {bestCeGenome.ce.toFixed(4)} ({bestCeGenome.neurons}n {bestCeGenome.bits}b) &middot;
                Best Acc: {(bestAccGenome.accuracy * 100).toFixed(2)}% ({bestAccGenome.neurons}n {bestAccGenome.bits}b)
              {/if}
            </span>
          </div>

          <div class="table-scroll">
            <table class="gating-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Neurons</th>
                  <th>Bits</th>
                  {#if isIDS}
                    <th>F1</th>
                    <th>FPR</th>
                  {:else}
                    <th>CE</th>
                  {/if}
                  <th>Accuracy</th>
                  <th>Fitness</th>
                </tr>
              </thead>
              <tbody>
                {#each expandedPopulation as g}
                  <tr class:expanded-best-ce={g.ce === bestCeGenome.ce} class:expanded-best-acc={g.accuracy === bestAccGenome.accuracy}>
                    <td class="mono">{g.rank}</td>
                    <td class="mono">{g.neurons.toLocaleString()}</td>
                    <td class="mono">{g.bits}</td>
                    {#if isIDS}
                      <td class="mono">{g.f1_macro != null ? (g.f1_macro * 100).toFixed(2) + '%' : '—'}</td>
                      <td class="mono">{g.fpr != null ? (g.fpr * 100).toFixed(2) + '%' : '—'}</td>
                    {:else}
                      <td class="mono">{g.ce.toFixed(4)}</td>
                    {/if}
                    <td class="mono">{(g.accuracy * 100).toFixed(2)}%</td>
                    <td class="mono">{g.fitness !== null ? g.fitness.toFixed(4) : '—'}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        </div>
      {/if}
    {/if}

    <!-- Live Generation Progress -->
    {#if liveProgress}
    <div class="live-progress-card">
      <div class="live-progress-bar">
        <span class="live-dot"></span>
        {#if liveProgress.total_generations > 0}
          <strong>Gen {liveProgress.generation}/{liveProgress.total_generations}</strong>
        {/if}
        <span class="live-phase">{liveProgress.phase === 'ga_offspring' ? 'GA Offspring' : liveProgress.phase === 'ts_neighbors' ? 'TS Neighbors' : 'Evaluating'}</span>
        <progress value={liveProgress.evaluated} max={liveProgress.target_count}></progress>
        <span>{liveProgress.evaluated}/{liveProgress.target_count}</span>
        {#if liveProgress.viable != null}
          <span>({liveProgress.viable} viable)</span>
        {/if}
        {#if liveProgress.best_ce > 0}
          {#if !isIDS}
            <span class="live-metric">CE: {liveProgress.best_ce.toFixed(4)}</span>
          {/if}
          <span class="live-metric">Acc: {(liveProgress.best_acc * 100).toFixed(2)}%</span>
        {/if}
        <span class="live-elapsed">{liveProgress.elapsed_secs.toFixed(0)}s</span>
      </div>
    </div>
    {/if}

    <!-- Iterations Table (hidden for grid search) -->
    {#if !isGridSearch}
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
                {#if isIDS}
                  <th>Best F1</th>
                  <th>Best FPR</th>
                  <th>Best Acc</th>
                {:else}
                  <th>Best CE</th>
                  <th>Best Acc</th>
                  <th>Avg CE</th>
                  <th>Avg Acc</th>
                {/if}
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
                  {#if isIDS}
                    <td class:best={iter.best_f1 !== null && bestF1 !== null && iter.best_f1 === bestF1}>{formatF1(iter.best_f1)}</td>
                    <td class:best={iter.best_fpr !== null && bestFpr !== null && iter.best_fpr === bestFpr}>{formatFPR(iter.best_fpr)}</td>
                    <td class:best={iter.best_accuracy !== null && iter.best_accuracy === bestAcc}>{formatAccShort(iter.best_accuracy)}</td>
                  {:else}
                    <td class:best={iter.best_ce === bestCE}>{formatCE(iter.best_ce)}</td>
                    <td class:best={iter.best_accuracy !== null && iter.best_accuracy === bestAcc}>{formatAccShort(iter.best_accuracy)}</td>
                    <td class="secondary">{iter.avg_ce ? formatCE(iter.avg_ce) : '—'}</td>
                    <td class="secondary">{formatAccShort(iter.avg_accuracy)}</td>
                  {/if}
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
  {/if}
</div>

<!-- Iteration Details Modal -->
{#if showIterationModal && selectedIteration}
  <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
  <div class="modal-overlay" on:click={closeIterationModal} on:keydown={(e) => e.key === 'Escape' && closeIterationModal()} role="dialog" aria-modal="true" tabindex="-1">
    <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
    <div class="modal" on:click|stopPropagation on:keydown|stopPropagation role="document">
      <div class="modal-header">
        <h2>Iteration {selectedIteration.iteration_num}</h2>
        <button class="modal-close" on:click={closeIterationModal} aria-label="Close">×</button>
      </div>
      <div class="modal-body">
        <div class="iteration-summary">
          {#if isIDS}
            <div class="summary-item">
              <span class="label">Best F1-macro</span>
              <span class="value">{formatF1(selectedIteration.best_f1)}</span>
            </div>
            <div class="summary-item">
              <span class="label">Best FPR</span>
              <span class="value">{formatFPR(selectedIteration.best_fpr)}</span>
            </div>
            <div class="summary-item">
              <span class="label">Best Accuracy</span>
              <span class="value">{formatAcc(selectedIteration.best_accuracy)}</span>
            </div>
          {:else}
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
          {@const elites = genomeEvaluations.filter(g => g.role === 'elite' || g.role === 'top_k').sort((a, b) => {
            if (a.fitness_score !== null && b.fitness_score !== null) return a.fitness_score - b.fitness_score;
            return a.position - b.position;
          })}
          {@const others = genomeEvaluations.filter(g => g.role !== 'elite' && g.role !== 'top_k').sort((a, b) => {
            // Sort by fitness_score if available (lower = better), fall back to CE
            if (a.fitness_score !== null && b.fitness_score !== null) return a.fitness_score - b.fitness_score;
            return a.ce - b.ce;
          })}

          {@const hasFitness = [...elites, ...others].some(g => g.fitness_score !== null)}
          {@const hasTiers = [...elites, ...others].some(g => g.tiers_json)}
          {#if elites.length > 0}
            <h3>Top Genomes ({elites.length})</h3>
            <div class="genome-table-scroll">
              <table class="genome-table">
                <thead>
                  <tr>
                    <th>#</th>
                    {#if hasFitness}<th>Fitness</th>{/if}
                    {#if !isIDS}<th>CE</th>{/if}
                    <th>Accuracy</th>
                    {#if isIDS}<th>F1-Macro</th><th>FPR</th>{/if}
                    {#if hasTiers}<th>Neurons</th><th>Bits</th>{/if}
                    <th>Role</th>
                  </tr>
                </thead>
                <tbody>
                  {#each elites as genome, idx}
                    {@const tier = parseTier(genome)}
                    <tr class="elite">
                      <td>{idx + 1}</td>
                      {#if hasFitness}
                        <td class="mono">{genome.fitness_score !== null ? genome.fitness_score.toFixed(2) : '—'}</td>
                      {/if}
                      {#if !isIDS}<td class:best={genome.ce === selectedIteration.best_ce}>{formatCE(genome.ce)}</td>{/if}
                      <td>{formatAcc(genome.accuracy)}</td>
                      {#if isIDS}
                        <td>{formatF1(genome.f1_macro)}</td>
                        <td>{formatFPR(genome.fpr)}</td>
                      {/if}
                      {#if hasTiers}
                        <td class="mono">{tier.neurons}</td>
                        <td class="mono">{tier.bits}</td>
                      {/if}
                      <td>{formatRole(genome.role)}</td>
                    </tr>
                  {/each}
                </tbody>
              </table>
            </div>
          {/if}

          {#if others.length > 0}
            <h3>Offspring ({others.length})</h3>
            <div class="genome-table-scroll">
              <table class="genome-table">
                <thead>
                  <tr>
                    <th>#</th>
                    {#if hasFitness}<th>Fitness</th>{/if}
                    {#if !isIDS}<th>CE</th>{/if}
                    <th>Accuracy</th>
                    {#if isIDS}<th>F1-Macro</th><th>FPR</th>{/if}
                    {#if hasTiers}<th>Neurons</th><th>Bits</th>{/if}
                    <th>Role</th>
                  </tr>
                </thead>
                <tbody>
                  {#each others as genome, idx}
                    {@const tier = parseTier(genome)}
                    <tr>
                      <td>{idx + 1}</td>
                      {#if hasFitness}
                        <td class="mono">{genome.fitness_score !== null ? genome.fitness_score.toFixed(2) : '—'}</td>
                      {/if}
                      {#if !isIDS}<td>{formatCE(genome.ce)}</td>{/if}
                      <td>{formatAcc(genome.accuracy)}</td>
                      {#if isIDS}
                        <td>{formatF1(genome.f1_macro)}</td>
                        <td>{formatFPR(genome.fpr)}</td>
                      {/if}
                      {#if hasTiers}
                        <td class="mono">{tier.neurons}</td>
                        <td class="mono">{tier.bits}</td>
                      {/if}
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
  /* Live generation progress */
  .live-progress-card {
    background: var(--card-bg, #1e1e2e);
    border: 1px solid var(--border, #333);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
  }
  .live-progress-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 1rem;
    flex-wrap: wrap;
  }
  .live-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #4ade80;
    animation: pulse 1.5s ease-in-out infinite;
    flex-shrink: 0;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }
  .live-phase {
    color: var(--text-secondary, #aaa);
  }
  .live-progress-bar progress {
    flex: 1;
    min-width: 100px;
    max-width: 200px;
    height: 8px;
    border-radius: 4px;
    appearance: none;
  }
  .live-progress-bar progress::-webkit-progress-bar {
    background: var(--border, #333);
    border-radius: 4px;
  }
  .live-progress-bar progress::-webkit-progress-value {
    background: #4ade80;
    border-radius: 4px;
  }
  .live-metric {
    color: var(--accent, #60a5fa);
    font-variant-numeric: tabular-nums;
  }
  .live-elapsed {
    color: var(--text-secondary, #aaa);
    font-variant-numeric: tabular-nums;
  }

  .container {
    max-width: 1680px;
    margin: 0 auto;
    padding: 1rem;
  }

  .status-message {
    font-size: 1rem;
    color: var(--text-secondary);
    background: var(--bg-secondary);
    padding: 0.5rem 1rem;
    border-radius: 6px;
    margin-bottom: 1rem;
    font-family: var(--font-mono, monospace);
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

  .flow-name-label {
    font-size: 1.125rem;
    color: var(--text-secondary);
    font-weight: 500;
  }

  .flow-name-label a {
    color: var(--text-secondary);
    text-decoration: none;
  }

  .flow-name-label a:hover {
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
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
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
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
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

  .validation-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    margin-left: 0.5rem;
  }

  .validation-legend {
    display: flex;
    gap: 1rem;
    font-size: 1rem;
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
    border: 1px solid var(--glass-border);
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
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

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
    font-size: 1rem;
  }

  .metric-ce {
    color: var(--text-primary);
    min-width: 70px;
  }

  .metric-acc {
    color: var(--text-secondary);
  }

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
    font-size: 1rem;
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
    border: 1px solid var(--glass-border);
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--glass-border);
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
    border: 1px solid var(--glass-border);
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
    font-size: 1rem;
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
    font-size: 1rem;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
  }

  .gating-status.running {
    background: rgba(59, 130, 246, 0.15);
    color: var(--accent-blue);
  }

  /* Gating Results Section */
  .gating-section {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
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
    border-bottom: 1px solid var(--glass-border);
  }

  .gating-title {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 1.1rem;
  }

  .gating-meta {
    font-size: 1rem;
    color: var(--text-secondary);
  }

  .gating-table-container {
    overflow-x: auto;
  }

  .gating-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 1rem;
  }

  .gating-table th {
    background: rgba(51, 65, 85, 0.4);
    padding: 0.5rem 0.75rem;
    text-align: center;
    font-weight: 600;
    color: var(--text-secondary);
    font-size: 1rem;
    text-transform: uppercase;
    border-bottom: 1px solid var(--glass-border);
  }

  .gating-table td {
    padding: 0.5rem 0.75rem;
    text-align: center;
    border-bottom: 1px solid var(--glass-border);
  }

  .gating-table tr:last-child td {
    border-bottom: none;
  }

  .grid-top-k {
    background: rgba(59, 130, 246, 0.1);
  }

  .grid-top-k td:first-child {
    font-weight: 600;
    color: var(--accent-blue);
  }

  .grid-rank-star {
    color: var(--accent-yellow);
    margin-right: 0.25rem;
  }

  .grid-progress {
    margin-bottom: 1rem;
  }

  .grid-progress-bar {
    height: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 0.5rem;
  }

  .grid-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-green));
    border-radius: 4px;
    transition: width 0.5s ease;
  }

  .grid-progress-info {
    display: flex;
    justify-content: space-between;
    font-size: 1rem;
    color: var(--text-secondary);
  }

  .expanded-summary {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 1rem;
  }

  .expanded-best {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 8px;
    padding: 0.75rem;
    text-align: center;
  }

  .expanded-label {
    display: block;
    font-size: 1rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 0.25rem;
  }

  .expanded-value {
    display: block;
    font-size: 1.5rem;
    font-weight: 700;
    font-family: 'Berkeley Mono', monospace;
    color: var(--accent-green);
  }

  .expanded-detail {
    display: block;
    font-size: 1rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
  }

  .expanded-best-ce td {
    background: rgba(34, 197, 94, 0.1);
  }

  .expanded-best-acc td {
    background: rgba(59, 130, 246, 0.1);
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
    font-size: 1rem;
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
    max-width: 120px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
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

  .metric-acc-inline {
    color: var(--text-secondary);
    font-size: 1rem;
    margin-left: 0.25rem;
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

  .threshold-mode-label {
    font-weight: 400 !important;
    font-size: 1rem;
    white-space: nowrap;
    font-family: monospace;
  }

  /* Actual result row (main data) — bright green, bold */
  .validation-table tbody tr:not(.threshold-sub-row) td.mono {
    color: #22c55e;
    font-weight: 600;
  }

  /* Holdout — dark green */
  .threshold-holdout-row td {
    color: #16a34a !important;
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

  /* Train-cal — default dim */
  .threshold-train-row td {
    color: var(--text-dim) !important;
  }

  /* Fixed 0.5 — orange */
  .threshold-fixed-row td {
    color: #f59e0b !important;
  }

  /* Oracle — orange/amber, italic */
  .threshold-oracle-row td {
    color: #f97316 !important;
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
