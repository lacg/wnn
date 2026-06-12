<script lang="ts">
  import type { ValidationProgressionPoint } from './types';

  export let points: ValidationProgressionPoint[] = [];
  // Which validation point + threshold mode to display. Bound from the page so
  // the choice survives in-page navigation (state lives in the page).
  export let pointChoice: number = 0;
  export let thresholdChoice: 'train_cal' | 'fixed_05' | 'val_cal' | 'platt' | 'beta' | 'empirical' | 'empirical_cumulative' = 'train_cal';

  // Per-class helpers — defined here so TS narrows their parameter types properly.
  type PerClassEntry = { count: number; rate: number };
  type PerClassDict = Record<string, PerClassEntry>;
  function pcLookup(summary: any, mode: string): PerClassDict | null {
    if (!summary) return null;
    return summary?.threshold_metadata?.[mode]?.per_class
        || (mode === 'train_cal' ? summary?.threshold_metadata?.per_class : null)
        || null;
  }
  function pcRateAt(genomeData: PerClassDict | null, clsName: string): string | null {
    if (!genomeData) return null;
    const e = genomeData[clsName];
    return e == null ? null : (e.rate * 100).toFixed(2) + '%';
  }
  function pcAnySummaryHasMode(summaries: any[], mode: string): boolean {
    return summaries.some(s => Boolean(pcLookup(s, mode)));
  }

  $: pcPoints = points.map(p => ({
    label: p.label,
    summaries: {
      f1:      p.summaries.find(s => s.genomeType === 'best_f1'),
      fpr:     p.summaries.find(s => s.genomeType === 'best_fpr'),
      acc:     p.summaries.find(s => s.genomeType === 'best_acc'),
      ce:      p.summaries.find(s => s.genomeType === 'best_ce'),
      fitness: p.summaries.find(s => s.genomeType === 'best_fitness'),
    },
  }));
  $: pcAvailablePoints = pcPoints.filter(p => {
    const all = [p.summaries.f1, p.summaries.fpr, p.summaries.acc, p.summaries.ce, p.summaries.fitness];
    return pcAnySummaryHasMode(all, thresholdChoice)
        || pcAnySummaryHasMode(all, 'train_cal');
  });
</script>

{#if pcAvailablePoints.length > 0}
  {@const _selectedIdx = Math.min(Math.max(0, pointChoice), pcAvailablePoints.length - 1)}
  {@const selectedPcPoint = pcAvailablePoints[_selectedIdx]}
  {@const pcAllSummaries = [selectedPcPoint.summaries.f1, selectedPcPoint.summaries.fpr, selectedPcPoint.summaries.acc, selectedPcPoint.summaries.ce, selectedPcPoint.summaries.fitness]}
  {@const pcByGenome = {
    f1:      pcLookup(selectedPcPoint.summaries.f1, thresholdChoice),
    fpr:     pcLookup(selectedPcPoint.summaries.fpr, thresholdChoice),
    acc:     pcLookup(selectedPcPoint.summaries.acc, thresholdChoice),
    ce:      pcLookup(selectedPcPoint.summaries.ce, thresholdChoice),
    fitness: pcLookup(selectedPcPoint.summaries.fitness, thresholdChoice),
  }}
  {@const pcClasses = pcByGenome.f1 || pcByGenome.fpr || pcByGenome.acc || pcByGenome.ce || pcByGenome.fitness}
  {@const pcThresholdModes = [
    { key: 'train_cal',           label: 'train_cal' },
    { key: 'fixed_05',            label: 'fixed_05' },
    { key: 'val_cal',             label: 'val_cal (oracle)' },
    { key: 'platt',               label: 'platt' },
    { key: 'beta',                label: 'beta' },
    { key: 'empirical',           label: 'empirical' },
    { key: 'empirical_cumulative',label: 'empirical_cumulative' },
  ].filter(m => pcAnySummaryHasMode(pcAllSummaries, m.key))}
  <details class="per-class-section" open>
    <summary class="per-class-summary">
      Per-attack-class breakdown
      {#if pcClasses}
        ({Object.keys(pcClasses).length} classes)
      {/if}
      <span class="pc-control-sep">—</span>
      <span class="pc-control-label">Phase:</span>
      <select bind:value={pointChoice} on:click|stopPropagation class="pc-select">
        {#each pcAvailablePoints as p, i}
          <option value={i}>{p.label}</option>
        {/each}
      </select>
      {#if pcThresholdModes.length > 0}
        <span class="pc-control-label">at</span>
        <select bind:value={thresholdChoice} on:click|stopPropagation class="pc-select">
          {#each pcThresholdModes as mode}
            <option value={mode.key}>{mode.label}</option>
          {/each}
        </select>
        <span class="pc-control-label">threshold</span>
      {/if}
    </summary>
    {#if pcClasses}
      <table class="per-class-table">
        <thead>
          <tr>
            <th rowspan="2" class="pc-class-col">Class</th>
            <th rowspan="2" class="pc-count-col">Count</th>
            <th colspan="2" class="best-ce-col">Best F1</th>
            <th colspan="2" class="best-acc-col">Best FPR</th>
            <th colspan="2">Best Acc</th>
            <th colspan="2">Best CE</th>
            <th colspan="2" class="best-fit-col">Best Fitness</th>
          </tr>
          <tr>
            <th class="best-ce-col" title="Detection rate (TPR) — fraction of this attack class predicted as attack. Only meaningful for non-Benign rows.">Det</th>
            <th class="best-ce-col" title="FPR — fraction of Benign predicted as attack. Only meaningful on the Benign row.">FPR</th>
            <th class="best-acc-col" title="Detection rate">Det</th>
            <th class="best-acc-col" title="FPR">FPR</th>
            <th title="Detection rate">Det</th>
            <th title="FPR">FPR</th>
            <th title="Detection rate">Det</th>
            <th title="FPR">FPR</th>
            <th class="best-fit-col" title="Detection rate">Det</th>
            <th class="best-fit-col" title="FPR">FPR</th>
          </tr>
        </thead>
        <tbody>
          {#each Object.entries(pcClasses) as [clsName, anyEntry]}
            {@const isBenign = clsName === 'Benign'}
            <tr>
              <td class="pc-class-col">{clsName}</td>
              <td class="mono pc-count-col">{anyEntry.count.toLocaleString()}</td>
              <td class="mono best-ce-col" style:opacity={isBenign ? 0.4 : 1}>{isBenign ? '—' : (pcRateAt(pcByGenome.f1, clsName) ?? '—')}</td>
              <td class="mono best-ce-col" style:opacity={isBenign ? 1 : 0.4}>{isBenign ? (pcRateAt(pcByGenome.f1, clsName) ?? '—') : '—'}</td>
              <td class="mono best-acc-col" style:opacity={isBenign ? 0.4 : 1}>{isBenign ? '—' : (pcRateAt(pcByGenome.fpr, clsName) ?? '—')}</td>
              <td class="mono best-acc-col" style:opacity={isBenign ? 1 : 0.4}>{isBenign ? (pcRateAt(pcByGenome.fpr, clsName) ?? '—') : '—'}</td>
              <td class="mono" style:opacity={isBenign ? 0.4 : 1}>{isBenign ? '—' : (pcRateAt(pcByGenome.acc, clsName) ?? '—')}</td>
              <td class="mono" style:opacity={isBenign ? 1 : 0.4}>{isBenign ? (pcRateAt(pcByGenome.acc, clsName) ?? '—') : '—'}</td>
              <td class="mono" style:opacity={isBenign ? 0.4 : 1}>{isBenign ? '—' : (pcRateAt(pcByGenome.ce, clsName) ?? '—')}</td>
              <td class="mono" style:opacity={isBenign ? 1 : 0.4}>{isBenign ? (pcRateAt(pcByGenome.ce, clsName) ?? '—') : '—'}</td>
              <td class="mono best-fit-col" style:opacity={isBenign ? 0.4 : 1}>{isBenign ? '—' : (pcRateAt(pcByGenome.fitness, clsName) ?? '—')}</td>
              <td class="mono best-fit-col" style:opacity={isBenign ? 1 : 0.4}>{isBenign ? (pcRateAt(pcByGenome.fitness, clsName) ?? '—') : '—'}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {:else}
      <p class="pc-empty">No per-class data for this phase × threshold combination.</p>
    {/if}
  </details>
{/if}

<style>
  /* Per-class drill-down section (rendered BELOW the main validation table,
     not interleaved between header and phase rows). */
  .per-class-section {
    margin-top: 1rem;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--glass-border);
  }
  .per-class-section[open] {
    background: rgba(255, 255, 255, 0.03);
  }
  .per-class-summary {
    font-weight: 600;
    cursor: pointer;
    padding: 0.25rem 0;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
  }
  .pc-control-sep {
    opacity: 0.5;
    margin: 0 0.25rem;
  }
  .pc-control-label {
    opacity: 0.65;
    font-weight: 400;
  }
  .pc-select {
    font-size: 1rem;
    padding: 0.15rem 0.4rem;
    cursor: pointer;
  }
  .pc-empty {
    opacity: 0.65;
    font-style: italic;
    margin: 0.75rem 0;
  }

  /* Base table styles (inherited from the page before extraction) */
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

  /* Per-class breakdown subtable */
  .per-class-table {
    border-collapse: collapse;
    margin: 0.5rem 0;
    font-size: 1rem;
    width: 100%;
  }
  .per-class-table th,
  .per-class-table td {
    padding: 0.25rem 0.75rem;
    text-align: right;
    border-bottom: 1px solid #333;
  }
  .per-class-table thead th {
    border-bottom: 1px solid #444;
    text-align: center;
    font-weight: 600;
  }
  .per-class-table .pc-class-col {
    text-align: left;
  }
  .per-class-table .pc-count-col {
    text-align: right;
  }
</style>
