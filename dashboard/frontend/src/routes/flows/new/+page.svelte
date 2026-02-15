<script lang="ts">
  import { goto } from '$app/navigation';
  import TierConfigEditor from '$lib/components/TierConfigEditor.svelte';
  import SeedCheckpointSelector from '$lib/components/SeedCheckpointSelector.svelte';

  let name = '';
  let description = '';
  let template = 'standard-6-phase';
  let phaseOrder = 'neurons_first';

  // Bitwise-specific config
  let bitwiseNumClusters = 16;
  let bitwiseMinBits = 10;
  let bitwiseMaxBits = 24;
  let bitwiseMinNeurons = 10;
  let bitwiseMaxNeurons = 300;
  let bitwiseMemoryMode = 'QUAD_WEIGHTED';
  let bitwiseNeuronSampleRate = 0.25;

  $: isBitwise = template === 'bitwise-7-phase';
  let gaGenerations = 250;
  let tsIterations = 250;
  let populationSize = 50;
  let neighborsPerIter = 50;
  let patience = 10;
  let fitnessPercentile = 0.75;
  let fitnessCalculator = 'normalized_harmonic';  // Default to normalized harmonic with equal weights
  let fitnessWeightCe = 1.0;   // Weight for CE in harmonic calculations
  let fitnessWeightAcc = 1.0;  // Weight for accuracy in harmonic calculations
  let minAccuracyFloor = 0;  // 0 = disabled, 0.003 = 0.3% floor
  let contextSize = 4;
  let tierConfig = '100,15,20,true;400,10,12,false;rest,5,8,false';

  // Apply template defaults when template changes
  function applyTemplateDefaults(templateName: string) {
    if (templateName === 'quick-4-phase') {
      gaGenerations = 50;
      tsIterations = 50;
      populationSize = 50;
      neighborsPerIter = 50;
      patience = 2;
      fitnessPercentile = 0.75;
      fitnessCalculator = 'normalized_harmonic';
      fitnessWeightCe = 1.0;
      fitnessWeightAcc = 1.0;
      contextSize = 4;
      tierConfig = '100,15,16,true;400,10,12,false;rest,5,8,false';
      phaseOrder = 'neurons_first';
    } else if (templateName === 'standard-6-phase') {
      gaGenerations = 250;
      tsIterations = 250;
      populationSize = 50;
      neighborsPerIter = 50;
      patience = 10;
      fitnessPercentile = 0.75;
      fitnessCalculator = 'normalized_harmonic';
      fitnessWeightCe = 1.0;
      fitnessWeightAcc = 1.0;
      contextSize = 4;
      tierConfig = '100,15,20,true;400,10,12,false;rest,5,8,false';
    } else if (templateName === 'bitwise-7-phase') {
      gaGenerations = 250;
      tsIterations = 250;
      populationSize = 50;
      neighborsPerIter = 50;
      patience = 10;
      fitnessPercentile = 0.75;
      fitnessCalculator = 'harmonic_rank';
      fitnessWeightCe = 1.0;
      fitnessWeightAcc = 1.0;
      contextSize = 4;
      bitwiseNumClusters = 16;
      bitwiseMinBits = 10;
      bitwiseMaxBits = 24;
      bitwiseMinNeurons = 10;
      bitwiseMaxNeurons = 300;
      bitwiseMemoryMode = 'QUAD_WEIGHTED';
      bitwiseNeuronSampleRate = 0.25;
    }
  }

  // Watch for template changes
  $: applyTemplateDefaults(template);
  let seedCheckpointId: number | null = null;

  let loading = false;
  let error: string | null = null;

  // Phase templates - generates experiments array based on template and phase_order
  interface PhaseSpec {
    name: string;
    experiment_type: 'ga' | 'ts';
    optimize_bits: boolean;
    optimize_neurons: boolean;
    optimize_connections: boolean;
  }

  function generatePhases(templateName: string, order: string): PhaseSpec[] {
    if (templateName === 'empty') {
      return [];
    }

    // Bitwise 7-phase template (grid search first, then neurons/bits order is configurable)
    if (templateName === 'bitwise-7-phase') {
      const grid: PhaseSpec = { name: 'Phase 1: Grid Search (neurons × bits)', experiment_type: 'ga', optimize_bits: true, optimize_neurons: true, optimize_connections: false };
      const neuronsPhases: PhaseSpec[] = [
        { name: `Phase ${order === 'neurons_first' ? 2 : 4}: GA Neurons`, experiment_type: 'ga', optimize_bits: false, optimize_neurons: true, optimize_connections: false },
        { name: `Phase ${order === 'neurons_first' ? 3 : 5}: TS Neurons (refine)`, experiment_type: 'ts', optimize_bits: false, optimize_neurons: true, optimize_connections: false },
      ];
      const bitsPhases: PhaseSpec[] = [
        { name: `Phase ${order === 'bits_first' ? 2 : 4}: GA Bits`, experiment_type: 'ga', optimize_bits: true, optimize_neurons: false, optimize_connections: false },
        { name: `Phase ${order === 'bits_first' ? 3 : 5}: TS Bits (refine)`, experiment_type: 'ts', optimize_bits: true, optimize_neurons: false, optimize_connections: false },
      ];
      const connectionsPhases: PhaseSpec[] = [
        { name: 'Phase 6: GA Connections', experiment_type: 'ga', optimize_bits: false, optimize_neurons: false, optimize_connections: true },
        { name: 'Phase 7: TS Connections (refine)', experiment_type: 'ts', optimize_bits: false, optimize_neurons: false, optimize_connections: true },
      ];
      if (order === 'bits_first') {
        return [grid, ...bitsPhases, ...neuronsPhases, ...connectionsPhases];
      } else {
        return [grid, ...neuronsPhases, ...bitsPhases, ...connectionsPhases];
      }
    }

    // Quick 4-phase template (neurons first, no connections phase)
    if (templateName === 'quick-4-phase') {
      return [
        { name: 'Phase 1a: GA Neurons Only', experiment_type: 'ga', optimize_bits: false, optimize_neurons: true, optimize_connections: false },
        { name: 'Phase 1b: TS Neurons Only (refine)', experiment_type: 'ts', optimize_bits: false, optimize_neurons: true, optimize_connections: false },
        { name: 'Phase 2a: GA Bits Only', experiment_type: 'ga', optimize_bits: true, optimize_neurons: false, optimize_connections: false },
        { name: 'Phase 2b: TS Bits Only (refine)', experiment_type: 'ts', optimize_bits: true, optimize_neurons: false, optimize_connections: false },
      ];
    }

    // Standard 6-phase template
    if (order === 'bits_first') {
      return [
        { name: 'Phase 1a: GA Bits Only', experiment_type: 'ga', optimize_bits: true, optimize_neurons: false, optimize_connections: false },
        { name: 'Phase 1b: TS Bits Only (refine)', experiment_type: 'ts', optimize_bits: true, optimize_neurons: false, optimize_connections: false },
        { name: 'Phase 2a: GA Neurons Only', experiment_type: 'ga', optimize_bits: false, optimize_neurons: true, optimize_connections: false },
        { name: 'Phase 2b: TS Neurons Only (refine)', experiment_type: 'ts', optimize_bits: false, optimize_neurons: true, optimize_connections: false },
        { name: 'Phase 3a: GA Connections Only', experiment_type: 'ga', optimize_bits: false, optimize_neurons: false, optimize_connections: true },
        { name: 'Phase 3b: TS Connections Only (refine)', experiment_type: 'ts', optimize_bits: false, optimize_neurons: false, optimize_connections: true },
      ];
    } else {
      // neurons_first (default)
      return [
        { name: 'Phase 1a: GA Neurons Only', experiment_type: 'ga', optimize_bits: false, optimize_neurons: true, optimize_connections: false },
        { name: 'Phase 1b: TS Neurons Only (refine)', experiment_type: 'ts', optimize_bits: false, optimize_neurons: true, optimize_connections: false },
        { name: 'Phase 2a: GA Bits Only', experiment_type: 'ga', optimize_bits: true, optimize_neurons: false, optimize_connections: false },
        { name: 'Phase 2b: TS Bits Only (refine)', experiment_type: 'ts', optimize_bits: true, optimize_neurons: false, optimize_connections: false },
        { name: 'Phase 3a: GA Connections Only', experiment_type: 'ga', optimize_bits: false, optimize_neurons: false, optimize_connections: true },
        { name: 'Phase 3b: TS Connections Only (refine)', experiment_type: 'ts', optimize_bits: false, optimize_neurons: false, optimize_connections: true },
      ];
    }
  }

  // Reactive: regenerate phases when template or phaseOrder changes
  $: experiments = generatePhases(template, phaseOrder);

  async function handleSubmit() {
    if (!name.trim()) {
      error = 'Name is required';
      return;
    }

    loading = true;
    error = null;

    try {
      // Enrich experiments with their params (generations/iterations based on type)
      const enrichedExperiments = experiments.map((exp, i) => ({
        ...exp,
        params: {
          generations: exp.experiment_type === 'ga' ? gaGenerations : undefined,
          iterations: exp.experiment_type === 'ts' ? tsIterations : undefined,
          population_size: populationSize,
          neighbors_per_iter: neighborsPerIter,
          // First bitwise phase is a grid search (special handling in worker)
          ...(isBitwise && i === 0 ? { phase_type: 'grid_search' } : {}),
        }
      }));

      // Build params — bitwise and tiered share common search params
      // but differ in architecture-specific config
      const params: Record<string, unknown> = {
        phase_order: phaseOrder,
        ga_generations: gaGenerations,
        ts_iterations: tsIterations,
        population_size: populationSize,
        neighbors_per_iter: neighborsPerIter,
        patience,
        fitness_percentile: fitnessPercentile,
        fitness_calculator: fitnessCalculator,
        fitness_weight_ce: fitnessWeightCe,
        fitness_weight_acc: fitnessWeightAcc,
        min_accuracy_floor: minAccuracyFloor,
        context_size: contextSize,
      };

      if (isBitwise) {
        params.architecture_type = 'bitwise';
        params.num_clusters = bitwiseNumClusters;
        params.min_bits = bitwiseMinBits;
        params.max_bits = bitwiseMaxBits;
        params.min_neurons = bitwiseMinNeurons;
        params.max_neurons = bitwiseMaxNeurons;
        params.memory_mode = bitwiseMemoryMode;
        params.neuron_sample_rate = bitwiseNeuronSampleRate;
      } else {
        params.tier_config = tierConfig || null;
      }

      // Experiments are passed separately (normalized design: Flow 1:N Experiments via FK)
      const response = await fetch('/api/flows', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          description: description || null,
          config: {
            template,
            params
          },
          experiments: enrichedExperiments,
          seed_checkpoint_id: seedCheckpointId
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Failed to create flow (${response.status})`);
      }

      const flow = await response.json();
      goto(`/flows/${flow.id}`);
    } catch (e) {
      error = e instanceof Error ? e.message : 'Unknown error';
    } finally {
      loading = false;
    }
  }
</script>

<div class="container">
  <div class="page-header">
    <a href="/flows" class="back-link">&larr; Flows</a>
    <h1>New Flow</h1>
  </div>

  <form on:submit|preventDefault={handleSubmit} class="form">
    {#if error}
      <div class="error-message">{error}</div>
    {/if}

    <!-- Top row: Name + Description side by side -->
    <div class="form-section">
      <div class="form-row">
        <div class="form-group">
          <label for="name">Name *</label>
          <input
            type="text"
            id="name"
            bind:value={name}
            placeholder="e.g., Pass 1 - Initial Search"
          />
        </div>

        <div class="form-group">
          <label for="description">Description</label>
          <input
            type="text"
            id="description"
            bind:value={description}
            placeholder="Optional description..."
          />
        </div>
      </div>
    </div>

    <!-- Main two-column layout -->
    <div class="form-columns">
      <!-- Left column: Search parameters -->
      <div class="form-section">
        <h2>Search Parameters</h2>

        <div class="form-row">
          <div class="form-group">
            <label for="template">Template</label>
            <select id="template" bind:value={template}>
              <option value="quick-4-phase">Quick 4-Phase (Tiered)</option>
              <option value="standard-6-phase">Standard 6-Phase (Tiered)</option>
              <option value="bitwise-7-phase">Bitwise 7-Phase</option>
              <option value="empty">Empty (no phases)</option>
            </select>
            <span class="field-hint">
              {#if template === 'quick-4-phase'}
                Fast iteration: neurons &rarr; bits (50 gens, patience 2)
              {:else if template === 'standard-6-phase'}
                Full search: neurons &rarr; bits &rarr; connections (250 gens)
              {:else if template === 'bitwise-7-phase'}
                Grid search + 6 optimization phases (per-cluster, 250 gens)
              {:else}
                Add phases manually after creation
              {/if}
            </span>
          </div>

          <div class="form-group">
            <label for="phaseOrder">Phase Order</label>
            <select id="phaseOrder" bind:value={phaseOrder} disabled={template === 'empty'}>
              <option value="neurons_first">Neurons First</option>
              <option value="bits_first">Bits First</option>
            </select>
            <span class="field-hint">
              {#if isBitwise}
                grid &rarr; {phaseOrder === 'neurons_first' ? 'neurons → bits' : 'bits → neurons'} &rarr; connections
              {:else if template === 'quick-4-phase'}
                {phaseOrder === 'neurons_first' ? 'neurons → bits' : 'bits → neurons'} (no connections)
              {:else if phaseOrder === 'neurons_first'}
                neurons &rarr; bits &rarr; connections
              {:else}
                bits &rarr; neurons &rarr; connections
              {/if}
            </span>
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label for="gaGens">GA Generations</label>
            <input type="number" id="gaGens" bind:value={gaGenerations} min="1" />
          </div>

          <div class="form-group">
            <label for="tsIters">TS Iterations</label>
            <input type="number" id="tsIters" bind:value={tsIterations} min="1" />
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label for="popSize">Population Size</label>
            <input type="number" id="popSize" bind:value={populationSize} min="1" />
          </div>

          <div class="form-group">
            <label for="neighborsPerIter">Neighbors/Iter (TS)</label>
            <input type="number" id="neighborsPerIter" bind:value={neighborsPerIter} min="1" />
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label for="patience">Patience</label>
            <input type="number" id="patience" bind:value={patience} min="1" />
          </div>

          <div class="form-group">
            <label for="contextSize">Context Size</label>
            <input type="number" id="contextSize" bind:value={contextSize} min="2" max="16" />
            <span class="field-hint">N-gram context window (4 = 4-gram)</span>
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label for="fitnessCalculator">Fitness Calculator</label>
            <select id="fitnessCalculator" bind:value={fitnessCalculator}>
              <option value="normalized">Normalized (Recommended)</option>
              <option value="harmonic_rank">Harmonic Rank</option>
              <option value="normalized_harmonic">Normalized Harmonic</option>
              <option value="ce">CE Only</option>
            </select>
          </div>

          <div class="form-group">
            <label for="fitnessPercentile">Fitness Percentile</label>
            <input type="number" id="fitnessPercentile" bind:value={fitnessPercentile} min="0" max="1" step="0.05" />
            <span class="field-hint">Keep top N% by fitness</span>
          </div>
        </div>

        {#if fitnessCalculator === 'harmonic_rank' || fitnessCalculator === 'normalized_harmonic'}
          <div class="form-row">
            <div class="form-group">
              <label for="fitnessWeightCe">CE Weight</label>
              <input type="number" id="fitnessWeightCe" bind:value={fitnessWeightCe} min="0" max="10" step="0.1" />
              <span class="field-hint">Higher = prioritize lower CE</span>
            </div>

            <div class="form-group">
              <label for="fitnessWeightAcc">Accuracy Weight</label>
              <input type="number" id="fitnessWeightAcc" bind:value={fitnessWeightAcc} min="0" max="10" step="0.1" />
              <span class="field-hint">Higher = prioritize accuracy</span>
            </div>
          </div>
        {/if}

        <div class="form-group">
          <label for="minAccuracyFloor">Accuracy Floor</label>
          <input type="number" id="minAccuracyFloor" bind:value={minAccuracyFloor} min="0" max="0.1" step="0.001" />
          <span class="field-hint">Min accuracy threshold (0.003 = 0.3%). 0 = disabled</span>
        </div>
      </div>

      <!-- Right column: Phases + Tiers + Seed -->
      <div class="right-column">
        {#if experiments.length > 0}
          <div class="form-section">
            <h2>Phases Preview ({experiments.length})</h2>
            <div class="phase-list">
              {#each experiments as phase, i}
                <div class="phase-item">
                  <span class="phase-num">{i + 1}</span>
                  <span class="phase-name">{phase.name}</span>
                  <span class="phase-type" class:ga={phase.experiment_type === 'ga'} class:ts={phase.experiment_type === 'ts'}>
                    {phase.experiment_type.toUpperCase()}
                  </span>
                </div>
              {/each}
            </div>
          </div>
        {/if}

        {#if isBitwise}
          <div class="form-section">
            <h2>Bitwise Configuration</h2>
            <div class="form-row">
              <div class="form-group">
                <label for="bitwiseNumClusters">Clusters</label>
                <input type="number" id="bitwiseNumClusters" bind:value={bitwiseNumClusters} min="1" max="256" />
                <span class="field-hint">Output clusters (default 16)</span>
              </div>
              <div class="form-group">
                <label for="bitwiseMemoryMode">Memory Mode</label>
                <select id="bitwiseMemoryMode" bind:value={bitwiseMemoryMode}>
                  <option value="QUAD_WEIGHTED">Quad Weighted</option>
                  <option value="QUAD_BINARY">Quad Binary</option>
                  <option value="TERNARY">Ternary</option>
                </select>
              </div>
            </div>
            <div class="form-row">
              <div class="form-group">
                <label for="bitwiseMinBits">Min Bits</label>
                <input type="number" id="bitwiseMinBits" bind:value={bitwiseMinBits} min="1" max="64" />
              </div>
              <div class="form-group">
                <label for="bitwiseMaxBits">Max Bits</label>
                <input type="number" id="bitwiseMaxBits" bind:value={bitwiseMaxBits} min="1" max="64" />
              </div>
            </div>
            <div class="form-row">
              <div class="form-group">
                <label for="bitwiseMinNeurons">Min Neurons</label>
                <input type="number" id="bitwiseMinNeurons" bind:value={bitwiseMinNeurons} min="1" max="1000" />
              </div>
              <div class="form-group">
                <label for="bitwiseMaxNeurons">Max Neurons</label>
                <input type="number" id="bitwiseMaxNeurons" bind:value={bitwiseMaxNeurons} min="1" max="1000" />
              </div>
            </div>
            <div class="form-group">
              <label for="bitwiseNeuronSampleRate">Neuron Sample Rate</label>
              <input type="number" id="bitwiseNeuronSampleRate" bind:value={bitwiseNeuronSampleRate} min="0.01" max="1.0" step="0.01" />
              <span class="field-hint">Fraction of neurons sampled per example (0.25 = 25%)</span>
            </div>
          </div>
        {:else}
          <div class="form-section">
            <h2>Tier Configuration</h2>
            <TierConfigEditor bind:value={tierConfig} />
          </div>
        {/if}

        <div class="form-section">
          <h2>Seed Checkpoint</h2>
          <p class="section-hint">
            Optionally seed from a previous run's checkpoint.
          </p>
          <SeedCheckpointSelector bind:value={seedCheckpointId} />
        </div>
      </div>
    </div>

    <div class="form-actions">
      <a href="/flows" class="btn btn-secondary">Cancel</a>
      <button type="submit" class="btn btn-primary" disabled={loading}>
        {loading ? 'Creating...' : 'Create Flow'}
      </button>
    </div>
  </form>
</div>

<style>
  .page-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
    padding-top: 2rem;
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

  .form {
    max-width: 95%;
  }

  .form-columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    align-items: start;
  }

  .right-column {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  .form-section {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1.5rem;
  }

  .form-columns .form-section {
    margin-bottom: 0;
  }

  h2 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 0.75rem 0;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }

  .section-hint {
    font-size: 1rem;
    color: var(--text-secondary);
    margin: -0.5rem 0 0.75rem 0;
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

  label {
    display: block;
    font-size: 1rem;
    font-weight: 500;
    color: var(--text-primary);
    margin-bottom: 0.375rem;
  }

  input, select, textarea {
    width: 100%;
    padding: 0.5rem 0.75rem;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 1rem;
    font-family: inherit;
  }

  input:focus, select:focus, textarea:focus {
    outline: none;
    border-color: var(--accent-blue);
  }

  textarea {
    resize: vertical;
  }

  .form-actions {
    display: flex;
    gap: 1rem;
    justify-content: flex-end;
  }

  .btn {
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-size: 1rem;
    font-weight: 500;
    text-decoration: none;
    border: none;
    cursor: pointer;
    transition: all 0.2s;
  }

  .btn-primary {
    background: var(--accent-blue);
    color: white;
  }

  .btn-primary:hover:not(:disabled) {
    opacity: 0.9;
  }

  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn-secondary {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .btn-secondary:hover {
    background: var(--border);
  }

  .error-message {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid var(--accent-red);
    color: var(--accent-red);
    padding: 0.75rem 1rem;
    border-radius: 6px;
    font-size: 1rem;
    margin-bottom: 1rem;
  }

  /* Phases Preview */
  .phase-list {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
  }

  .phase-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1rem;
  }

  .phase-num {
    width: 1.5rem;
    height: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--bg-tertiary);
    border-radius: 4px;
    font-size: 1rem;
    color: var(--text-secondary);
    font-weight: 500;
    flex-shrink: 0;
  }

  .phase-name {
    flex: 1;
    color: var(--text-primary);
  }

  .phase-type {
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 1rem;
    font-weight: 600;
    text-transform: uppercase;
    flex-shrink: 0;
  }

  .phase-type.ga {
    background: rgba(59, 130, 246, 0.15);
    color: var(--accent-blue);
  }

  .phase-type.ts {
    background: rgba(16, 185, 129, 0.15);
    color: var(--accent-green);
  }

  select:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
