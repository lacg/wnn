<script lang="ts">
  import type { StageConfig } from '$lib/flowTemplates';
  import { GRID_DEFAULTS, ALL_DEFAULT_NEURONS, ALL_DEFAULT_BITS } from '$lib/flowTemplates';

  export let numStages = 2;
  export let selectedStage = 0;
  export let stageMode = 'input_concat';
  export let msTemplate = 'full';
  export let stageConfigs: StageConfig[] = [];
  export let msMemoryMode = 'QUAD_WEIGHTED';
  export let msNeuronSampleRate = 0.25;
  export let invalidMode = false;
  export let topM = 5;

  function stageGridMode(stageIdx: number): string {
    if (stageMode === 'selector' && stageIdx > 0) return 'selector';
    const ct = stageConfigs[stageIdx]?.clusterType;
    if (ct === 'tiered' || ct === 'semantic') return ct;
    if (ct === 'semantic_bitwise') return 'semantic_bitwise';
    return 'bitwise';
  }

  /** Update grid defaults for a stage IF the user hasn't customized them. */
  function applyGridDefaults(stageIdx: number) {
    const config = stageConfigs[stageIdx];
    if (!config) return;
    const defaults = GRID_DEFAULTS[stageGridMode(stageIdx)];
    if (ALL_DEFAULT_NEURONS.has(config.neuronsGrid)) {
      config.neuronsGrid = defaults.neurons;
    }
    if (ALL_DEFAULT_BITS.has(config.bitsGrid)) {
      config.bitsGrid = defaults.bits;
    }
    stageConfigs = stageConfigs; // trigger Svelte reactivity
  }

  function handleStageModeChange() {
    for (let i = 0; i < stageConfigs.length; i++) {
      applyGridDefaults(i);
    }
  }
</script>

<div class="form-section">
  <h2>Multi-Stage Configuration</h2>
  <div class="form-row-4">
    <div class="form-group">
      <label for="selectedStage">Edit Stage</label>
      <select id="selectedStage" bind:value={selectedStage}>
        {#each Array(numStages) as _, i}
          <option value={i}>Stage {i}</option>
        {/each}
      </select>
    </div>
    <div class="form-group">
      <label for="stageMode">Stage Connection</label>
      <select id="stageMode" bind:value={stageMode} on:change={handleStageModeChange}>
        <option value="input_concat">Input Concat</option>
        <option value="selector">Selector (cluster routing)</option>
      </select>
      <span class="field-hint">
        {#if stageMode === 'input_concat'}
          Stage N+1 sees stage N output bits
        {:else}
          Stage N output selects which cluster group to use
        {/if}
      </span>
    </div>
    <div class="form-group">
      <label for="msTemplate">Template</label>
      <select id="msTemplate" bind:value={msTemplate}>
        <option value="full">Full (10 phases/stage)</option>
        <option value="fast">Fast (2 phases/stage)</option>
      </select>
      <span class="field-hint">
        {#if msTemplate === 'fast'}
          Grid Search + GA Neurons only
        {:else}
          All 10 optimization phases
        {/if}
      </span>
    </div>
    {#each stageConfigs as config, i}
      {#if i === selectedStage}
        <div class="form-group">
          <label for="stageArch_{i}">Architecture</label>
          <select id="stageArch_{i}" bind:value={config.clusterType} on:change={() => applyGridDefaults(i)}>
            <option value="bitwise">Bitwise</option>
            <option value="tiered">Tiered</option>
            <option value="semantic">Semantic (GPT-2 embeddings)</option>
            <option value="semantic_bitwise">Semantic Bitwise (PCA bisection)</option>
          </select>
        </div>
        <div class="form-group">
          <label for="stageK_{i}">K</label>
          <input type="number" id="stageK_{i}" bind:value={config.k} min="2" max="1024" />
        </div>
        <div class="form-group">
          <label for="stageCtx_{i}">Context</label>
          <input type="number" id="stageCtx_{i}" bind:value={config.contextSize} min="2" max="16" />
        </div>
      {/if}
    {/each}
  </div>

  <div class="shared-params-header">Per-Stage Bounds (Stage {selectedStage})</div>
  {#each stageConfigs as config, i}
    {#if i === selectedStage}
      <div class="form-row-4">
        <div class="form-group">
          <label for="stageMinBits_{i}">Min Bits</label>
          <input type="number" id="stageMinBits_{i}" bind:value={config.minBits} min="1" max="64" />
        </div>
        <div class="form-group">
          <label for="stageMaxBits_{i}">Max Bits</label>
          <input type="number" id="stageMaxBits_{i}" bind:value={config.maxBits} min="1" max="64" />
        </div>
        <div class="form-group">
          <label for="stageMinNeurons_{i}">Min Neurons</label>
          <input type="number" id="stageMinNeurons_{i}" bind:value={config.minNeurons} min="1" max="1000" />
        </div>
        <div class="form-group">
          <label for="stageMaxNeurons_{i}">Max Neurons</label>
          <input type="number" id="stageMaxNeurons_{i}" bind:value={config.maxNeurons} min="1" max="1000" />
        </div>
      </div>
      <div class="form-row">
        <div class="form-group">
          <label for="stageNeuronsGrid_{i}">Neurons Grid</label>
          <input type="text" id="stageNeuronsGrid_{i}" bind:value={config.neuronsGrid}
            placeholder={GRID_DEFAULTS[stageGridMode(i)].neurons} />
          <span class="field-hint">Defaults: {stageGridMode(i)}</span>
        </div>
        <div class="form-group">
          <label for="stageBitsGrid_{i}">Bits Grid</label>
          <input type="text" id="stageBitsGrid_{i}" bind:value={config.bitsGrid}
            placeholder={GRID_DEFAULTS[stageGridMode(i)].bits} />
          <span class="field-hint">Defaults: {stageGridMode(i)}</span>
        </div>
      </div>
    {/if}
  {/each}
  <div class="form-row">
    <div class="form-group">
      <label for="msMemoryMode">Memory Mode</label>
      <select id="msMemoryMode" bind:value={msMemoryMode}>
        <option value="QUAD_WEIGHTED">Quad Weighted</option>
        <option value="QUAD_BINARY">Quad Binary</option>
        <option value="TERNARY">Ternary</option>
      </select>
    </div>
    <div class="form-group">
      <label for="msNeuronSampleRate">Neuron Sample Rate</label>
      <input type="number" id="msNeuronSampleRate" bind:value={msNeuronSampleRate} min="0.01" max="1.0" step="0.01" />
      <span class="field-hint">Fraction of neurons sampled per example</span>
    </div>
  </div>
  {#if stageMode === 'selector'}
    <div class="form-row">
      <div class="form-group">
        <label for="invalidMode">
          <input type="checkbox" id="invalidMode" bind:checked={invalidMode} />
          Invalid Token Mode
        </label>
        <span class="field-hint">S1 groups learn to reject wrong-group inputs</span>
      </div>
      {#if invalidMode}
        <div class="form-group">
          <label for="topM">Top-M Groups</label>
          <input type="number" id="topM" bind:value={topM} min="0" max="50" step="1" />
          <span class="field-hint">Groups per example in augmented training (0 = all)</span>
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .form-section {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--glass-shadow), var(--glass-inset);
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
  }

  .form-section:hover {
    box-shadow: var(--glass-shadow-hover), var(--glass-inset);
    border-color: var(--glass-border-highlight);
  }

  h2 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 0.75rem 0;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }

  .form-group {
    margin-bottom: 0.75rem;
  }

  .form-group:last-child {
    margin-bottom: 0;
  }

  .field-hint {
    display: block;
    font-size: 1rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
  }

  input[type="checkbox"] {
    width: auto;
    margin-right: 0.5rem;
    vertical-align: middle;
  }

  label:has(input[type="checkbox"]) {
    display: flex;
    align-items: center;
    cursor: pointer;
  }

  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
  }

  .form-row-4 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 0.75rem;
  }

  label {
    display: block;
    font-size: 1rem;
    font-weight: 500;
    color: var(--text-primary);
    margin-bottom: 0.375rem;
  }

  input, select {
    width: 100%;
    padding: 0.5rem 0.75rem;
    border: 1px solid var(--glass-border);
    border-radius: 8px;
    background: var(--glass-input-bg);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    color: var(--text-primary);
    font-size: 1rem;
    font-family: inherit;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.15);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }

  input:focus, select:focus {
    outline: none;
    border-color: rgba(59, 130, 246, 0.6);
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.15), 0 0 0 3px rgba(59, 130, 246, 0.15);
  }

  /* Multi-stage config */
  .shared-params-header {
    font-size: 1rem;
    font-weight: 500;
    color: var(--text-secondary);
    margin: 0.75rem 0 0.5rem 0;
    padding-top: 0.75rem;
    border-top: 1px solid rgba(148, 163, 184, 0.12);
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
</style>
