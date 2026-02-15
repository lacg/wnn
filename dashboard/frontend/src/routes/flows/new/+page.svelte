<script lang="ts">
  import { goto } from '$app/navigation';
  import TierConfigEditor from '$lib/components/TierConfigEditor.svelte';
  import SeedCheckpointSelector from '$lib/components/SeedCheckpointSelector.svelte';

  let name = '';
  let description = '';
  let template = 'bitwise-7-phase';
  let phaseOrder = 'neurons_first';

  // Bitwise-specific config
  let bitwiseNumClusters = 16;
  let bitwiseMinBits = 10;
  let bitwiseMaxBits = 24;
  let bitwiseMinNeurons = 10;
  let bitwiseMaxNeurons = 300;
  let bitwiseMemoryMode = 'QUAD_WEIGHTED';
  let bitwiseNeuronSampleRate = 0.25;

  // Adaptation (Baldwin/Lamarckian)
  let synaptogenesis = true;
  let neurogenesis = true;
  let adaptWarmup = 10;
  let adaptCooldown = 5;

  $: isBitwise = template === 'bitwise-7-phase';
  let gaGenerations = 250;
  let tsIterations = 250;
  let populationSize = 50;
  let neighborsPerIter = 50;
  let patience = 10;
  let fitnessPercentile = 0.75;
  let fitnessCalculator = 'harmonic_rank';  // Default matches bitwise-7-phase template
  let fitnessWeightCe = 1.0;   // Weight for CE in harmonic calculations
  let fitnessWeightAcc = 1.0;  // Weight for accuracy in harmonic calculations
  let minAccuracyFloor = 0;  // 0 = disabled, 0.003 = 0.3% floor
  let thresholdStart = 0;      // Accuracy filter at phase 1 (%)
  let thresholdStep = 1;       // Accuracy increase per phase (%)
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
      synaptogenesis = true;
      neurogenesis = true;
      adaptWarmup = 10;
      adaptCooldown = 5;
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
    phase_type?: 'grid_search';
  }

  // Add-phase form state
  let newPhaseType: 'ga' | 'ts' = 'ga';
  let newPhaseGrid = false;
  let newPhaseNeurons = true;
  let newPhaseBits = false;
  let newPhaseConnections = false;

  function generatePhases(templateName: string, order: string): PhaseSpec[] {
    if (templateName === 'empty') return [];

    const neuronsPhases: PhaseSpec[] = [
      { name: 'GA Neurons', experiment_type: 'ga', optimize_bits: false, optimize_neurons: true, optimize_connections: false },
      { name: 'TS Neurons (refine)', experiment_type: 'ts', optimize_bits: false, optimize_neurons: true, optimize_connections: false },
    ];
    const bitsPhases: PhaseSpec[] = [
      { name: 'GA Bits', experiment_type: 'ga', optimize_bits: true, optimize_neurons: false, optimize_connections: false },
      { name: 'TS Bits (refine)', experiment_type: 'ts', optimize_bits: true, optimize_neurons: false, optimize_connections: false },
    ];
    const connectionsPhases: PhaseSpec[] = [
      { name: 'GA Connections', experiment_type: 'ga', optimize_bits: false, optimize_neurons: false, optimize_connections: true },
      { name: 'TS Connections (refine)', experiment_type: 'ts', optimize_bits: false, optimize_neurons: false, optimize_connections: true },
    ];

    if (templateName === 'bitwise-7-phase') {
      const grid: PhaseSpec = { name: 'Grid Search (neurons × bits)', experiment_type: 'ga', optimize_bits: true, optimize_neurons: true, optimize_connections: false, phase_type: 'grid_search' };
      if (order === 'bits_first') return [grid, ...bitsPhases, ...neuronsPhases, ...connectionsPhases];
      return [grid, ...neuronsPhases, ...bitsPhases, ...connectionsPhases];
    }

    if (templateName === 'quick-4-phase') {
      if (order === 'bits_first') return [...bitsPhases, ...neuronsPhases];
      return [...neuronsPhases, ...bitsPhases];
    }

    // standard-6-phase (default)
    if (order === 'bits_first') return [...bitsPhases, ...neuronsPhases, ...connectionsPhases];
    return [...neuronsPhases, ...bitsPhases, ...connectionsPhases];
  }

  // Reactive: regenerate phases when template or phaseOrder changes
  $: experiments = generatePhases(template, phaseOrder);

  function generatePhaseName(type: string, neurons: boolean, bits: boolean, connections: boolean): string {
    const targets: string[] = [];
    if (neurons) targets.push('Neurons');
    if (bits) targets.push('Bits');
    if (connections) targets.push('Connections');
    return `${type.toUpperCase()} ${targets.join(' + ')}`;
  }

  function addPhase() {
    if (newPhaseGrid) {
      experiments = [...experiments, {
        name: 'Grid Search (neurons × bits)',
        experiment_type: 'ga' as const,
        optimize_bits: true,
        optimize_neurons: true,
        optimize_connections: false,
        phase_type: 'grid_search' as const,
      }];
      return;
    }
    if (!newPhaseNeurons && !newPhaseBits && !newPhaseConnections) return;
    experiments = [...experiments, {
      name: generatePhaseName(newPhaseType, newPhaseNeurons, newPhaseBits, newPhaseConnections),
      experiment_type: newPhaseType,
      optimize_bits: newPhaseBits,
      optimize_neurons: newPhaseNeurons,
      optimize_connections: newPhaseConnections,
    }];
  }

  function removePhase(index: number) {
    experiments = experiments.filter((_, i) => i !== index);
  }

  function movePhase(index: number, direction: -1 | 1) {
    const newIndex = index + direction;
    if (newIndex < 0 || newIndex >= experiments.length) return;
    const copy = [...experiments];
    [copy[index], copy[newIndex]] = [copy[newIndex], copy[index]];
    experiments = copy;
  }

  async function handleSubmit() {
    if (!name.trim()) {
      error = 'Name is required';
      return;
    }

    loading = true;
    error = null;

    try {
      // Enrich experiments with their params (generations/iterations based on type)
      const enrichedExperiments = experiments.map((exp) => ({
        ...exp,
        params: {
          // Grid search is a single step — don't pass generations/iterations
          generations: (exp.phase_type === 'grid_search') ? undefined : (exp.experiment_type === 'ga' ? gaGenerations : undefined),
          iterations: (exp.phase_type === 'grid_search') ? undefined : (exp.experiment_type === 'ts' ? tsIterations : undefined),
          population_size: populationSize,
          neighbors_per_iter: neighborsPerIter,
          ...(exp.phase_type ? { phase_type: exp.phase_type } : {}),
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
        threshold_start: thresholdStart,
        threshold_step: thresholdStep,
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
        // Adaptation (Baldwin/Lamarckian)
        if (synaptogenesis || neurogenesis) {
          params.synaptogenesis = synaptogenesis;
          params.neurogenesis = neurogenesis;
          params.adapt_warmup = adaptWarmup;
          params.adapt_cooldown = adaptCooldown;
        }
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
      <!-- Left column: Search parameters + Seed -->
      <div class="left-column">
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
                Exhaustive neurons &times; bits grid, then 6 GA/TS optimization phases
              {:else}
                Start empty, add phases manually below
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
          <span class="field-hint">Hard floor (0.003 = 0.3%). Below = fitness infinity. 0 = disabled</span>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label for="thresholdStart">Threshold Start (%)</label>
            <input type="number" id="thresholdStart" bind:value={thresholdStart} min="0" max="50" step="0.1" />
            <span class="field-hint">Accuracy filter at phase 1 (0 = no filter)</span>
          </div>
          <div class="form-group">
            <label for="thresholdStep">Threshold Increase / Phase (%)</label>
            <input type="number" id="thresholdStep" bind:value={thresholdStep} min="0" max="50" step="0.1" />
            <span class="field-hint">How much accuracy filter grows each phase</span>
          </div>
        </div>
      </div>

      <div class="form-section">
        <h2>Seed Checkpoint</h2>
        <p class="section-hint">
          Optionally seed from a previous run's checkpoint.
        </p>
        <SeedCheckpointSelector bind:value={seedCheckpointId} />
      </div>
      </div>

      <!-- Right column: Phases + Tiers -->
      <div class="right-column">
        <div class="form-section">
          <h2>Phases ({experiments.length})</h2>
          {#if experiments.length > 0}
            <div class="phase-list">
              {#each experiments as phase, i}
                <div class="phase-item">
                  <div class="phase-move">
                    <button type="button" class="move-btn" on:click={() => movePhase(i, -1)} disabled={i === 0} title="Move up">&uarr;</button>
                    <button type="button" class="move-btn" on:click={() => movePhase(i, 1)} disabled={i === experiments.length - 1} title="Move down">&darr;</button>
                  </div>
                  <span class="phase-num">{i + 1}</span>
                  <span class="phase-name">{phase.name}</span>
                  <span class="phase-type" class:ga={phase.experiment_type === 'ga'} class:ts={phase.experiment_type === 'ts'}>
                    {phase.experiment_type.toUpperCase()}
                  </span>
                  <button type="button" class="remove-btn" on:click={() => removePhase(i)} title="Remove">&times;</button>
                </div>
              {/each}
            </div>
          {:else}
            <p class="empty-phases">No phases. Use a template or add phases manually.</p>
          {/if}
          <div class="add-phase-row">
            {#if isBitwise}
              <label class="inline-check">
                <input type="checkbox" bind:checked={newPhaseGrid} /> Grid Search
              </label>
            {/if}
            {#if !newPhaseGrid}
              <select bind:value={newPhaseType} class="phase-type-select">
                <option value="ga">GA</option>
                <option value="ts">TS</option>
              </select>
              <label class="inline-check"><input type="checkbox" bind:checked={newPhaseNeurons} /> Neurons</label>
              <label class="inline-check"><input type="checkbox" bind:checked={newPhaseBits} /> Bits</label>
              <label class="inline-check"><input type="checkbox" bind:checked={newPhaseConnections} /> Connections</label>
            {/if}
            <button type="button" class="btn btn-add" on:click={addPhase}
              disabled={!newPhaseGrid && !newPhaseNeurons && !newPhaseBits && !newPhaseConnections}>
              + Add Phase
            </button>
          </div>
        </div>

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

          <div class="form-section">
            <h2>Adaptation (Baldwin Effect)</h2>
            <span class="field-hint" style="display:block; margin-top:-0.5rem; margin-bottom:0.75rem;">
              Each genome is adapted during evaluation, then scored. Evolution selects architectures that respond well to adaptation (Baldwin effect).
            </span>
            <div class="form-row">
              <div class="form-group">
                <label class="inline-check">
                  <input type="checkbox" bind:checked={synaptogenesis} /> Synaptogenesis
                </label>
                <span class="field-hint">Prune unused synapses, grow new ones on active neurons</span>
              </div>
              <div class="form-group">
                <label class="inline-check">
                  <input type="checkbox" bind:checked={neurogenesis} /> Neurogenesis
                </label>
                <span class="field-hint">Add neurons to struggling clusters, remove redundant ones</span>
              </div>
            </div>
            {#if synaptogenesis || neurogenesis}
              <div class="form-row" style="margin-top:0.75rem;">
                <div class="form-group">
                  <label for="adaptWarmup">Warmup Generations</label>
                  <input type="number" id="adaptWarmup" bind:value={adaptWarmup} min="0" max="100" />
                  <span class="field-hint">Let evolution stabilize before adaptation kicks in (0 = adapt from start)</span>
                </div>
                <div class="form-group">
                  <label for="adaptCooldown">Cooldown Iterations</label>
                  <input type="number" id="adaptCooldown" bind:value={adaptCooldown} min="0" max="50" />
                  <span class="field-hint">Freeze a neuron after adapting it, preventing prune/grow oscillation</span>
                </div>
              </div>
            {/if}
          </div>
        {:else}
          <div class="form-section">
            <h2>Tier Configuration</h2>
            <TierConfigEditor bind:value={tierConfig} />
          </div>
        {/if}

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

  .left-column, .right-column {
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

  .phase-move {
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .move-btn {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 3px;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 1rem;
    line-height: 1;
    padding: 0 0.25rem;
    transition: all 0.15s;
  }

  .move-btn:hover:not(:disabled) {
    background: var(--border);
    color: var(--text-primary);
  }

  .move-btn:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }

  .remove-btn {
    background: none;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 1.25rem;
    line-height: 1;
    padding: 0 0.25rem;
    transition: color 0.15s;
    flex-shrink: 0;
  }

  .remove-btn:hover {
    color: var(--accent-red);
  }

  .empty-phases {
    color: var(--text-secondary);
    font-size: 1rem;
    font-style: italic;
    margin: 0.5rem 0;
  }

  .add-phase-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 1px solid var(--border);
    flex-wrap: wrap;
  }

  .phase-type-select {
    width: auto;
    padding: 0.25rem 0.5rem;
    font-size: 1rem;
  }

  .inline-check {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 1rem;
    color: var(--text-primary);
    cursor: pointer;
    margin-bottom: 0;
    font-weight: 400;
  }

  .inline-check input[type="checkbox"] {
    width: auto;
    margin: 0;
  }

  .btn-add {
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    color: var(--text-primary);
    padding: 0.25rem 0.75rem;
    border-radius: 6px;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.15s;
    margin-left: auto;
  }

  .btn-add:hover:not(:disabled) {
    background: var(--border);
  }

  .btn-add:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  select:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
