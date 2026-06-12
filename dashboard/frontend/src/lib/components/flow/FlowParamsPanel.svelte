<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { Flow } from '$lib/types';
  import TierConfigEditor from '$lib/components/TierConfigEditor.svelte';

  export let flow: Flow;
  export let isBitwise: boolean = false;
  export let saving: boolean = false;

  const dispatch = createEventDispatcher<{
    updateCalculator: string;
    updateWeight: { field: 'fitness_weight_ce' | 'fitness_weight_acc' | 'fitness_weight_f1' | 'fitness_weight_fpr' | 'min_accuracy_floor' | 'threshold_start' | 'threshold_step'; value: number };
  }>();

  /** Format tier_config which may be a string or array of tuples */
  function formatTierConfig(tierConfig: unknown): string {
    if (typeof tierConfig === 'string') return tierConfig;
    if (Array.isArray(tierConfig)) {
      return tierConfig.map((t: (number|string|boolean)[]) => {
        const base = `${t[0] ?? 'rest'},${t[1]},${t[2]}`;
        return t.length > 3 ? `${base},${t[3]}` : base;
      }).join('; ');
    }
    return String(tierConfig);
  }
</script>

<section class="section">
  <h2>Parameters</h2>
  <div class="params-grid">
    <div class="param-item">
      <span class="param-label">Patience</span>
      <span class="param-value">{flow.config.params.patience ?? '-'}</span>
    </div>
    <div class="param-item">
      <span class="param-label">Phase Order</span>
      <span class="param-value">{flow.config.params.phase_order ?? 'neurons_first'}</span>
    </div>
    <div class="param-item">
      <span class="param-label">GA Generations</span>
      <span class="param-value">{flow.config.params.ga_generations ?? '-'}</span>
    </div>
    <div class="param-item">
      <span class="param-label">TS Iterations</span>
      <span class="param-value">{flow.config.params.ts_iterations ?? '-'}</span>
    </div>
    <div class="param-item">
      <span class="param-label">Population</span>
      <span class="param-value">{flow.config.params.population_size ?? '-'}</span>
    </div>
    <div class="param-item">
      <span class="param-label">Neighbors/Iter</span>
      <span class="param-value">{flow.config.params.neighbors_per_iter ?? flow.config.params.population_size ?? 50}</span>
    </div>
    <div class="param-item">
      <span class="param-label">Fitness %</span>
      <span class="param-value">{flow.config.params.fitness_percentile ?? 0.75}</span>
    </div>
    {#if flow.config.params.cluster_crossover_ratio}
      <div class="param-item">
        <span class="param-label">Cluster XO Ratio</span>
        <span class="param-value">{flow.config.params.cluster_crossover_ratio}</span>
      </div>
    {/if}
    {#if flow.config.params.pool_shuffle_ratio}
      <div class="param-item">
        <span class="param-label">Pool Shuffle Ratio</span>
        <span class="param-value">{flow.config.params.pool_shuffle_ratio}</span>
      </div>
    {/if}
    {#if flow.config.params.assortative_mating_ratio}
      <div class="param-item">
        <span class="param-label">Assortative Mating</span>
        <span class="param-value">{flow.config.params.assortative_mating_ratio}</span>
      </div>
    {/if}
    <div class="param-group full-width">
      <span class="param-group-label">Fitness</span>
      <div class="param-group-items">
        <div class="param-group-item">
          <span class="param-label">Calculator</span>
          <select
            class="inline-select"
            value={flow.config.params.fitness_calculator ?? 'normalized'}
            on:change={(e) => dispatch('updateCalculator', e.currentTarget.value)}
            disabled={saving}
          >
            <option value="normalized">Normalized</option>
            <option value="normalized_harmonic">Normalized Harmonic</option>
            <option value="harmonic_rank">Harmonic Rank</option>
            <option value="ce">CE Only</option>
          </select>
        </div>
        <div class="param-group-item">
          <span class="param-label">CE Weight</span>
          <input
            type="number"
            class="inline-input"
            value={flow.config.params.fitness_weight_ce ?? 1.0}
            min="0"
            max="10"
            step="0.1"
            on:change={(e) => dispatch('updateWeight', { field: 'fitness_weight_ce', value: parseFloat(e.currentTarget.value) })}
            disabled={saving}
          />
        </div>
        <div class="param-group-item">
          <span class="param-label">F1 Weight</span>
          <input
            type="number"
            class="inline-input"
            value={flow.config.params.fitness_weight_f1 ?? 0}
            min="0"
            max="10"
            step="0.1"
            on:change={(e) => dispatch('updateWeight', { field: 'fitness_weight_f1', value: parseFloat(e.currentTarget.value) })}
            disabled={saving}
          />
        </div>
        <div class="param-group-item">
          <span class="param-label">FPR Weight</span>
          <input
            type="number"
            class="inline-input"
            value={flow.config.params.fitness_weight_fpr ?? 0}
            min="0"
            max="10"
            step="0.1"
            on:change={(e) => dispatch('updateWeight', { field: 'fitness_weight_fpr', value: parseFloat(e.currentTarget.value) })}
            disabled={saving}
          />
        </div>
        <div class="param-group-item">
          <span class="param-label">Acc Weight</span>
          <input
            type="number"
            class="inline-input"
            value={flow.config.params.fitness_weight_acc ?? 1.0}
            min="0"
            max="10"
            step="0.1"
            on:change={(e) => dispatch('updateWeight', { field: 'fitness_weight_acc', value: parseFloat(e.currentTarget.value) })}
            disabled={saving}
          />
        </div>
        <div class="param-group-item">
          <span class="param-label">Acc Floor</span>
          <input
            type="number"
            class="inline-input"
            value={flow.config.params.min_accuracy_floor ?? 0}
            min="0"
            max="0.1"
            step="0.001"
            on:change={(e) => dispatch('updateWeight', { field: 'min_accuracy_floor', value: parseFloat(e.currentTarget.value) })}
            disabled={saving}
          />
        </div>
      </div>
    </div>
    {#if isBitwise}
      <div class="param-group full-width">
        <span class="param-group-label">Bitwise Architecture</span>
        <div class="param-group-items">
          <div class="param-group-item">
            <span class="param-label">Clusters</span>
            <span class="param-value mono">{flow.config.params.num_clusters ?? 16}</span>
          </div>
          <div class="param-group-item">
            <span class="param-label">Bits</span>
            <span class="param-value mono">{flow.config.params.min_bits ?? 10}–{flow.config.params.max_bits ?? 24}</span>
          </div>
          <div class="param-group-item">
            <span class="param-label">Neurons</span>
            <span class="param-value mono">{flow.config.params.min_neurons ?? 10}–{flow.config.params.max_neurons ?? 300}</span>
          </div>
          <div class="param-group-item">
            <span class="param-label">Memory</span>
            <span class="param-value mono">{flow.config.params.memory_mode ?? 'QUAD_WEIGHTED'}</span>
          </div>
          <div class="param-group-item">
            <span class="param-label">Sample Rate</span>
            <span class="param-value mono">{Math.round((flow.config.params.neuron_sample_rate ?? 0.25) * 100)}%</span>
          </div>
          {#if flow.config.params.context_size}
            <div class="param-group-item">
              <span class="param-label">Context</span>
              <span class="param-value mono">{flow.config.params.context_size}-gram</span>
            </div>
          {/if}
        </div>
      </div>
    {:else}
      {#if flow.config.params.tier_config}
        <div class="param-item full-width">
          <span class="param-label">Tier Config</span>
          <TierConfigEditor value={formatTierConfig(flow.config.params.tier_config)} readonly={true} />
        </div>
      {/if}
      {#if flow.config.params.context_size}
        <div class="param-item">
          <span class="param-label">Context Size</span>
          <span class="param-value">{flow.config.params.context_size}-gram</span>
        </div>
      {/if}
    {/if}
  </div>
</section>

<style>
  .section {
    margin-bottom: 2rem;
  }

  h2 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
  }

  /* Parameters Grid */
  .params-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 1rem;
  }

  .param-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .param-item.full-width,
  .param-group.full-width {
    grid-column: 1 / -1;
  }

  .param-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 0.75rem;
    background: rgba(51, 65, 85, 0.4);
    border-radius: 6px;
    border: 1px solid var(--glass-border);
  }

  .param-group-label {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
  }

  .param-group-items {
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
  }

  .param-group-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .param-group-item .param-label {
    font-size: 1rem;
  }

  .param-label {
    font-size: 1rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
  }

  .param-value {
    font-size: 1rem;
    color: var(--text-primary);
  }

  .param-value.mono {
    font-family: monospace;
    font-size: 1rem;
  }

  .inline-select {
    padding: 0.25rem 0.5rem;
    border: 1px solid var(--glass-border);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 1rem;
    cursor: pointer;
    transition: border-color 0.15s;
  }

  .inline-select:hover:not(:disabled) {
    border-color: var(--accent-blue);
  }

  .inline-select:focus {
    outline: none;
    border-color: var(--accent-blue);
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
  }

  .inline-select:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .inline-input {
    width: 70px;
    padding: 0.25rem 0.5rem;
    border: 1px solid var(--glass-border);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 1rem;
    text-align: right;
    transition: border-color 0.15s;
  }

  .inline-input:hover:not(:disabled) {
    border-color: var(--accent-blue);
  }

  .inline-input:focus {
    outline: none;
    border-color: var(--accent-blue);
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
  }

  .inline-input:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  @media (max-width: 768px) {
    .params-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }
</style>
