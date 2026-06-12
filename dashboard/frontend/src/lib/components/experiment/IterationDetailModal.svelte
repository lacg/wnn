<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { Iteration, GenomeEvaluation } from '$lib/types';
  import { formatCE } from '$lib/format';
  import { formatAcc, formatF1, formatFPR, formatRole, parseTier } from './metricFormat';

  export let selectedIteration: Iteration;
  export let genomeEvaluations: GenomeEvaluation[] = [];
  export let loadingGenomes: boolean = false;
  export let isIDS: boolean = false;

  const dispatch = createEventDispatcher<{ close: void }>();
  const close = () => dispatch('close');
</script>

<!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
<div class="modal-overlay" on:click={close} on:keydown={(e) => e.key === 'Escape' && close()} role="dialog" aria-modal="true" tabindex="-1">
  <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
  <div class="modal" on:click|stopPropagation on:keydown|stopPropagation role="document">
    <div class="modal-header">
      <h2>Iteration {selectedIteration.iteration_num}</h2>
      <button class="modal-close" on:click={close} aria-label="Close">×</button>
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

  .best {
    color: var(--accent-green);
    font-weight: 600;
  }

  .delta-positive {
    color: var(--accent-green);
  }

  .delta-negative {
    color: var(--accent-red);
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

  .genome-table .mono {
    font-family: monospace;
  }
</style>
