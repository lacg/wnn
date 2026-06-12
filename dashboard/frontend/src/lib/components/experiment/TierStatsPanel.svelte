<script lang="ts">
  import type { TierStats } from '$lib/types';

  interface ParsedTier {
    clusters: string;  // number or "rest"
    neurons: number;
    bits: number;
    optimize: boolean;
  }

  export let tierStats: TierStats[] | null = null;
  export let tierConfigOptimize: boolean[] = [];
  export let parsedTiers: ParsedTier[] = [];
</script>

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

  .delta-positive {
    color: var(--accent-green);
  }

  .delta-negative {
    color: var(--accent-red);
  }

  /* Gating-style panel (shared look with gating results) */
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

  .gating-table .genome-type {
    text-transform: capitalize;
    font-weight: 500;
    color: var(--text-primary);
    text-align: left;
  }

  .gating-table .mono {
    font-family: monospace;
  }
</style>
