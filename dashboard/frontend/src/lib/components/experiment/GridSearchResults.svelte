<script lang="ts">
  import type { Iteration, ExperimentStatus } from '$lib/types';
  import type { GridSearchRow, ExpandedGenome } from './types';

  export let iterations: Iteration[] = [];
  export let gridSearchResults: GridSearchRow[] = [];
  export let expandedPopulation: ExpandedGenome[] = [];
  export let seedEvalComplete: boolean = false;
  export let gridSearchLoading: boolean = false;
  export let isIDS: boolean = false;
  export let status: ExperimentStatus;

  $: totalConfigs = iterations.length > 0 && iterations[0].candidates_total ? iterations[0].candidates_total : 0;
  $: testedCount = gridSearchResults.length;
  $: pendingCount = totalConfigs > testedCount ? totalConfigs - testedCount : 0;
  $: totalElapsed = gridSearchResults.reduce((s, r) => s + (r.elapsed ?? 0), 0);
  $: avgTimePerConfig = testedCount > 0 ? totalElapsed / testedCount : 0;
  $: estimatedRemaining = pendingCount * avgTimePerConfig;
  $: progressPct = totalConfigs > 0 ? (testedCount / totalConfigs) * 100 : 0;
</script>

<div class="gating-section" style="border-left-color: var(--accent-blue);">
  <div class="gating-header">
    <span class="gating-title">Grid Search Results</span>
    <span class="gating-meta">
      {#if testedCount > 0 && pendingCount > 0}
        {testedCount} / {totalConfigs} configs
      {:else if testedCount > 0}
        {testedCount} configs tested
      {:else if status === 'running'}
        Evaluating...
      {/if}
    </span>
  </div>

  <!-- Progress bar (shown while running) -->
  {#if status === 'running' && totalConfigs > 0}
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
  {:else if status === 'completed'}
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

<style>
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

  .table-scroll {
    max-height: 500px;
    overflow-y: auto;
  }

  .empty-state {
    padding: 2rem;
    text-align: center;
    color: var(--text-secondary);
  }

  /* Gating-style panel (shared look) */
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

  .expanded-best-ce td {
    background: rgba(34, 197, 94, 0.1);
  }

  .expanded-best-acc td {
    background: rgba(59, 130, 246, 0.1);
  }

  .gating-table .mono {
    font-family: monospace;
  }
</style>
