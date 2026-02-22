<script lang="ts">
  import { goto } from '$app/navigation';
  import TierConfigEditor from '$lib/components/TierConfigEditor.svelte';
  import SeedCheckpointSelector from '$lib/components/SeedCheckpointSelector.svelte';

  let name = '';
  let description = '';
  let template = 'bitwise-7-phase';
  let phaseOrder = 'neurons_first';

  // Bitwise-specific config (single-stage only)
  let bitwiseNumClusters = 16;
  let bitwiseMinBits = 10;
  let bitwiseMaxBits = 24;
  let bitwiseMinNeurons = 10;
  let bitwiseMaxNeurons = 300;
  let bitwiseMemoryMode = 'QUAD_WEIGHTED';
  let bitwiseNeuronSampleRate = 0.25;

  // Multi-stage config
  let numStages = 1;
  let stageMode = 'input_concat';
  let selectedStage = 0;

  interface StageConfig {
    clusterType: string;
    k: number;
  }

  function defaultStageConfig(): StageConfig {
    return { clusterType: 'bitwise', k: 256 };
  }

  let stageConfigs: StageConfig[] = [defaultStageConfig(), defaultStageConfig()];

  // Shared multi-stage architecture params
  let msMinBits = 4;
  let msMaxBits = 24;
  let msMinNeurons = 5;
  let msMaxNeurons = 300;
  let msMemoryMode = 'QUAD_WEIGHTED';
  let msNeuronSampleRate = 0.25;

  $: isMultiStage = numStages >= 2;
  $: isBitwise = !isMultiStage && (template === 'bitwise-7-phase' || template === 'bitwise-10-phase');

  // Resize stageConfigs when numStages changes
  $: {
    while (stageConfigs.length < numStages) {
      stageConfigs = [...stageConfigs, defaultStageConfig()];
    }
    if (stageConfigs.length > numStages) {
      stageConfigs = stageConfigs.slice(0, numStages);
    }
    if (selectedStage >= numStages) {
      selectedStage = Math.max(0, numStages - 1);
    }
  }

  let gaGenerations = 250;
  let tsIterations = 250;
  let adaptationIterations = 50;
  let populationSize = 50;
  let neighborsPerIter = 50;
  let patience = 10;
  let fitnessPercentile = 0.75;
  let fitnessCalculator = 'harmonic_rank';
  let fitnessWeightCe = 1.0;
  let fitnessWeightAcc = 1.0;
  let minAccuracyFloor = 0;
  let thresholdStart = 0;
  let thresholdStep = 1;
  let contextSize = 4;
  let tierConfig = '100,15,20,true;400,10,12,false;rest,5,8,false';

  // Apply template defaults (only in single-stage mode)
  function applyTemplateDefaults(templateName: string) {
    if (isMultiStage) return;

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
    } else if (templateName === 'bitwise-10-phase') {
      gaGenerations = 250;
      tsIterations = 250;
      adaptationIterations = 50;
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

  $: applyTemplateDefaults(template);
  let seedCheckpointId: number | null = null;

  let loading = false;
  let error: string | null = null;

  // Phase spec interface
  interface PhaseSpec {
    name: string;
    experiment_type: 'ga' | 'ts' | 'neurogenesis' | 'synaptogenesis' | 'axonogenesis';
    optimize_bits: boolean;
    optimize_neurons: boolean;
    optimize_connections: boolean;
    phase_type?: 'grid_search' | 'neurogenesis' | 'synaptogenesis' | 'axonogenesis';
  }

  // Add-phase form state
  let newPhaseType: 'ga' | 'ts' | 'neurogenesis' | 'synaptogenesis' | 'axonogenesis' = 'ga';
  let newPhaseGrid = false;
  let newPhaseNeurons = true;
  let newPhaseBits = false;
  let newPhaseConnections = false;

  /** Generate the 10-phase pipeline for a single stage. */
  function generate10PhaseForStage(prefix: string): PhaseSpec[] {
    return [
      { name: `${prefix}: Grid Search`, experiment_type: 'ga', optimize_bits: true, optimize_neurons: true, optimize_connections: false, phase_type: 'grid_search' },
      { name: `${prefix}: GA Neurons`, experiment_type: 'ga', optimize_bits: false, optimize_neurons: true, optimize_connections: false },
      { name: `${prefix}: Neurogenesis`, experiment_type: 'neurogenesis', optimize_bits: false, optimize_neurons: false, optimize_connections: false, phase_type: 'neurogenesis' },
      { name: `${prefix}: TS Neurons`, experiment_type: 'ts', optimize_bits: false, optimize_neurons: true, optimize_connections: false },
      { name: `${prefix}: GA Bits`, experiment_type: 'ga', optimize_bits: true, optimize_neurons: false, optimize_connections: false },
      { name: `${prefix}: Synaptogenesis`, experiment_type: 'synaptogenesis', optimize_bits: false, optimize_neurons: false, optimize_connections: false, phase_type: 'synaptogenesis' },
      { name: `${prefix}: TS Bits`, experiment_type: 'ts', optimize_bits: true, optimize_neurons: false, optimize_connections: false },
      { name: `${prefix}: GA Connections`, experiment_type: 'ga', optimize_bits: false, optimize_neurons: false, optimize_connections: true },
      { name: `${prefix}: Axonogenesis`, experiment_type: 'axonogenesis', optimize_bits: false, optimize_neurons: false, optimize_connections: false, phase_type: 'axonogenesis' },
      { name: `${prefix}: TS Connections`, experiment_type: 'ts', optimize_bits: false, optimize_neurons: false, optimize_connections: true },
    ];
  }

  /** Generate single-stage phases from template. */
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

    if (templateName === 'bitwise-10-phase') {
      const grid: PhaseSpec = { name: 'Grid Search (neurons × bits)', experiment_type: 'ga', optimize_bits: true, optimize_neurons: true, optimize_connections: false, phase_type: 'grid_search' };
      const neurogenesisPhase: PhaseSpec = { name: 'Neurogenesis', experiment_type: 'neurogenesis', optimize_bits: false, optimize_neurons: false, optimize_connections: false, phase_type: 'neurogenesis' };
      const synaptogenesisPhase: PhaseSpec = { name: 'Synaptogenesis', experiment_type: 'synaptogenesis', optimize_bits: false, optimize_neurons: false, optimize_connections: false, phase_type: 'synaptogenesis' };
      const axonogenesisPhase: PhaseSpec = { name: 'Axonogenesis', experiment_type: 'axonogenesis', optimize_bits: false, optimize_neurons: false, optimize_connections: false, phase_type: 'axonogenesis' };
      return [
        grid,
        neuronsPhases[0], neurogenesisPhase, neuronsPhases[1],
        bitsPhases[0], synaptogenesisPhase, bitsPhases[1],
        connectionsPhases[0], axonogenesisPhase, connectionsPhases[1],
      ];
    }

    if (templateName === 'bitwise-7-phase') {
      const grid: PhaseSpec = { name: 'Grid Search (neurons × bits)', experiment_type: 'ga', optimize_bits: true, optimize_neurons: true, optimize_connections: false, phase_type: 'grid_search' };
      if (order === 'bits_first') return [grid, ...bitsPhases, ...neuronsPhases, ...connectionsPhases];
      return [grid, ...neuronsPhases, ...bitsPhases, ...connectionsPhases];
    }

    if (templateName === 'quick-4-phase') {
      if (order === 'bits_first') return [...bitsPhases, ...neuronsPhases];
      return [...neuronsPhases, ...bitsPhases];
    }

    // standard-6-phase
    if (order === 'bits_first') return [...bitsPhases, ...neuronsPhases, ...connectionsPhases];
    return [...neuronsPhases, ...bitsPhases, ...connectionsPhases];
  }

  // --- Per-stage phase storage (multi-stage only) ---
  let perStagePhases: PhaseSpec[][] = [];

  // Initialize/resize per-stage phases when numStages changes
  $: if (isMultiStage && perStagePhases.length !== numStages) {
    const updated = [...perStagePhases];
    while (updated.length < numStages) {
      updated.push(generate10PhaseForStage(`S${updated.length}`));
    }
    if (updated.length > numStages) {
      updated.length = numStages;
    }
    perStagePhases = updated;
  }

  // Single-stage phases from template
  let singleStagePhases: PhaseSpec[] = [];
  $: if (!isMultiStage) {
    singleStagePhases = generatePhases(template, phaseOrder);
  }

  // What to display in the Phases panel
  $: displayPhases = isMultiStage
    ? (perStagePhases[selectedStage] ?? [])
    : singleStagePhases;

  // All experiments flattened for submit
  $: allExperiments = isMultiStage
    ? perStagePhases.flat()
    : singleStagePhases;

  function generatePhaseName(type: string, neurons: boolean, bits: boolean, connections: boolean): string {
    const targets: string[] = [];
    if (neurons) targets.push('Neurons');
    if (bits) targets.push('Bits');
    if (connections) targets.push('Connections');
    return `${type.toUpperCase()} ${targets.join(' + ')}`;
  }

  const adaptationPhaseTypes = ['neurogenesis', 'synaptogenesis', 'axonogenesis'];
  function isAdaptationType(t: string): boolean { return adaptationPhaseTypes.includes(t); }

  function addPhase() {
    let newPhase: PhaseSpec;

    if (newPhaseGrid) {
      newPhase = {
        name: 'Grid Search (neurons × bits)',
        experiment_type: 'ga' as const,
        optimize_bits: true,
        optimize_neurons: true,
        optimize_connections: false,
        phase_type: 'grid_search' as const,
      };
    } else if (isAdaptationType(newPhaseType)) {
      const label = newPhaseType.charAt(0).toUpperCase() + newPhaseType.slice(1);
      newPhase = {
        name: label,
        experiment_type: newPhaseType as PhaseSpec['experiment_type'],
        optimize_bits: false,
        optimize_neurons: false,
        optimize_connections: false,
        phase_type: newPhaseType as PhaseSpec['phase_type'],
      };
    } else {
      if (!newPhaseNeurons && !newPhaseBits && !newPhaseConnections) return;
      newPhase = {
        name: generatePhaseName(newPhaseType, newPhaseNeurons, newPhaseBits, newPhaseConnections),
        experiment_type: newPhaseType,
        optimize_bits: newPhaseBits,
        optimize_neurons: newPhaseNeurons,
        optimize_connections: newPhaseConnections,
      };
    }

    if (isMultiStage) {
      newPhase.name = `S${selectedStage}: ${newPhase.name}`;
      perStagePhases[selectedStage] = [...perStagePhases[selectedStage], newPhase];
      perStagePhases = perStagePhases;
    } else {
      singleStagePhases = [...singleStagePhases, newPhase];
    }
  }

  function removePhase(index: number) {
    if (isMultiStage) {
      perStagePhases[selectedStage] = perStagePhases[selectedStage].filter((_, i) => i !== index);
      perStagePhases = perStagePhases;
    } else {
      singleStagePhases = singleStagePhases.filter((_, i) => i !== index);
    }
  }

  function movePhase(index: number, direction: -1 | 1) {
    const arr = isMultiStage ? perStagePhases[selectedStage] : singleStagePhases;
    const newIndex = index + direction;
    if (newIndex < 0 || newIndex >= arr.length) return;
    const copy = [...arr];
    [copy[index], copy[newIndex]] = [copy[newIndex], copy[index]];
    if (isMultiStage) {
      perStagePhases[selectedStage] = copy;
      perStagePhases = perStagePhases;
    } else {
      singleStagePhases = copy;
    }
  }

  async function handleSubmit() {
    if (!name.trim()) {
      error = 'Name is required';
      return;
    }

    loading = true;
    error = null;

    try {
      const adaptationTypes = new Set(['neurogenesis', 'synaptogenesis', 'axonogenesis']);
      const enrichedExperiments = allExperiments.map((exp) => {
        const isAdaptation = adaptationTypes.has(exp.phase_type ?? '');
        return {
          ...exp,
          params: {
            generations: (exp.phase_type === 'grid_search') ? undefined
              : isAdaptation ? adaptationIterations
              : (exp.experiment_type === 'ga' ? gaGenerations : undefined),
            iterations: (exp.phase_type === 'grid_search') ? undefined
              : isAdaptation ? adaptationIterations
              : (exp.experiment_type === 'ts' ? tsIterations : undefined),
            population_size: populationSize,
            neighbors_per_iter: neighborsPerIter,
            ...(exp.phase_type ? { phase_type: exp.phase_type } : {}),
          }
        };
      });

      const params: Record<string, unknown> = {
        phase_order: phaseOrder,
        ga_generations: gaGenerations,
        ts_iterations: tsIterations,
        adaptation_iterations: adaptationIterations,
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

      if (isMultiStage) {
        params.architecture_type = 'multi_stage';
        params.num_stages = numStages;
        params.stage_k = stageConfigs.slice(0, numStages).map(s => s.k);
        params.stage_cluster_type = stageConfigs.slice(0, numStages).map(s => s.clusterType);
        params.stage_mode = stageMode;
        params.min_bits = msMinBits;
        params.max_bits = msMaxBits;
        params.min_neurons = msMinNeurons;
        params.max_neurons = msMaxNeurons;
        params.memory_mode = msMemoryMode;
        params.neuron_sample_rate = msNeuronSampleRate;
      } else if (isBitwise) {
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

      const response = await fetch('/api/flows', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          description: description || null,
          config: {
            template: isMultiStage ? 'multi-stage' : template,
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

    <!-- Top row: Name + Description + Stages -->
    <div class="form-section">
      <div class="form-row-header">
        <div class="form-group">
          <label for="name">Name *</label>
          <input type="text" id="name" bind:value={name} placeholder="e.g., Pass 1 - Initial Search" />
        </div>
        <div class="form-group">
          <label for="description">Description</label>
          <input type="text" id="description" bind:value={description} placeholder="Optional description..." />
        </div>
        <div class="form-group">
          <label for="numStages">Stages</label>
          <input type="number" id="numStages" bind:value={numStages} min="1" max="4" />
          <span class="field-hint">
            {#if numStages === 1}
              Single-stage
            {:else}
              {numStages}-stage factorized
            {/if}
          </span>
        </div>
      </div>
    </div>

    <!-- Multi-Stage Configuration (full width, only when stages > 1) -->
    {#if isMultiStage}
      <div class="form-section">
        <h2>Multi-Stage Configuration</h2>
        <div class="form-row-3">
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
            <select id="stageMode" bind:value={stageMode}>
              <option value="input_concat">Input Concat</option>
            </select>
            <span class="field-hint">Stage N+1 sees stage N output bits</span>
          </div>
          <!-- Per-stage config -->
          {#each stageConfigs as config, i}
            {#if i === selectedStage}
              <div class="form-group stage-config-inline">
                <div class="stage-config-header">Stage {i}</div>
                <div class="stage-fields">
                  <div class="stage-field">
                    <label for="stageArch_{i}">Architecture</label>
                    <select id="stageArch_{i}" bind:value={config.clusterType}>
                      <option value="bitwise">Bitwise</option>
                      <option value="tiered">Tiered</option>
                    </select>
                  </div>
                  <div class="stage-field">
                    <label for="stageK_{i}">K</label>
                    <input type="number" id="stageK_{i}" bind:value={config.k} min="2" max="1024" />
                  </div>
                </div>
              </div>
            {/if}
          {/each}
        </div>

        <div class="shared-params-header">Shared Parameters</div>
        <div class="form-row-4">
          <div class="form-group">
            <label for="msMinBits">Min Bits</label>
            <input type="number" id="msMinBits" bind:value={msMinBits} min="1" max="64" />
          </div>
          <div class="form-group">
            <label for="msMaxBits">Max Bits</label>
            <input type="number" id="msMaxBits" bind:value={msMaxBits} min="1" max="64" />
          </div>
          <div class="form-group">
            <label for="msMinNeurons">Min Neurons</label>
            <input type="number" id="msMinNeurons" bind:value={msMinNeurons} min="1" max="1000" />
          </div>
          <div class="form-group">
            <label for="msMaxNeurons">Max Neurons</label>
            <input type="number" id="msMaxNeurons" bind:value={msMaxNeurons} min="1" max="1000" />
          </div>
        </div>
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
      </div>
    {/if}

    <!-- Main two-column layout -->
    <div class="form-columns">
      <!-- Left column: Search parameters + Seed -->
      <div class="left-column">
      <div class="form-section">
        <h2>Search Parameters</h2>

        {#if !isMultiStage}
          <div class="form-row">
            <div class="form-group">
              <label for="template">Template</label>
              <select id="template" bind:value={template}>
                <option value="quick-4-phase">Quick 4-Phase (Tiered)</option>
                <option value="standard-6-phase">Standard 6-Phase (Tiered)</option>
                <option value="bitwise-7-phase">Bitwise 7-Phase</option>
                <option value="bitwise-10-phase">Bitwise 10-Phase (+ Adaptation)</option>
                <option value="empty">Empty (no phases)</option>
              </select>
              <span class="field-hint">
                {#if template === 'quick-4-phase'}
                  Fast iteration: neurons &rarr; bits (50 gens, patience 2)
                {:else if template === 'standard-6-phase'}
                  Full search: neurons &rarr; bits &rarr; connections (250 gens)
                {:else if template === 'bitwise-7-phase'}
                  Exhaustive neurons &times; bits grid, then 6 GA/TS optimization phases
                {:else if template === 'bitwise-10-phase'}
                  Grid + GA/adapt/TS for neurons &rarr; bits &rarr; connections
                {:else}
                  Start empty, add phases manually below
                {/if}
              </span>
            </div>

            <div class="form-group">
              <label for="phaseOrder">Phase Order</label>
              <select id="phaseOrder" bind:value={phaseOrder} disabled={template === 'empty' || template === 'bitwise-10-phase'}>
                <option value="neurons_first">Neurons First</option>
                <option value="bits_first">Bits First</option>
              </select>
              <span class="field-hint">
                {#if template === 'bitwise-10-phase'}
                  Fixed: grid &rarr; neurons &rarr; bits &rarr; connections
                {:else if isBitwise}
                  grid &rarr; {phaseOrder === 'neurons_first' ? 'neurons → bits' : 'bits → neurons'} &rarr; connections
                {:else if template === 'quick-4-phase'}
                  {phaseOrder === 'neurons_first' ? 'neurons → bits' : 'bits → neurons'}
                {:else if phaseOrder === 'neurons_first'}
                  neurons &rarr; bits &rarr; connections
                {:else}
                  bits &rarr; neurons &rarr; connections
                {/if}
              </span>
            </div>
          </div>
        {/if}

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

        {#if template === 'bitwise-10-phase' || isMultiStage}
          <div class="form-row">
            <div class="form-group">
              <label for="adaptIters">Adaptation Iterations</label>
              <input type="number" id="adaptIters" bind:value={adaptationIterations} min="1" />
              <span class="field-hint">Iterations for neurogenesis, synaptogenesis, axonogenesis</span>
            </div>
          </div>
        {/if}

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

      <!-- Right column: Phases + Architecture config -->
      <div class="right-column">
        <div class="form-section">
          <h2>
            {#if isMultiStage}
              Stage {selectedStage} Phases ({displayPhases.length})
            {:else}
              Phases ({displayPhases.length})
            {/if}
          </h2>
          {#if displayPhases.length > 0}
            <div class="phase-list">
              {#each displayPhases as phase, i}
                <div class="phase-item">
                  <div class="phase-move">
                    <button type="button" class="move-btn" on:click={() => movePhase(i, -1)} disabled={i === 0} title="Move up">&uarr;</button>
                    <button type="button" class="move-btn" on:click={() => movePhase(i, 1)} disabled={i === displayPhases.length - 1} title="Move down">&darr;</button>
                  </div>
                  <span class="phase-num">{i + 1}</span>
                  <span class="phase-name">{phase.name}</span>
                  <span class="phase-type"
                    class:ga={phase.experiment_type === 'ga'}
                    class:ts={phase.experiment_type === 'ts'}
                    class:adapt={['neurogenesis', 'synaptogenesis', 'axonogenesis'].includes(phase.experiment_type)}>
                    {phase.phase_type === 'grid_search' ? 'GRID' : phase.experiment_type.toUpperCase()}
                  </span>
                  <button type="button" class="remove-btn" on:click={() => removePhase(i)} title="Remove">&times;</button>
                </div>
              {/each}
            </div>
          {:else}
            <p class="empty-phases">No phases. Use a template or add phases manually.</p>
          {/if}
          <div class="add-phase-row">
            {#if isBitwise || isMultiStage}
              <label class="inline-check">
                <input type="checkbox" bind:checked={newPhaseGrid} /> Grid Search
              </label>
            {/if}
            {#if !newPhaseGrid}
              <select bind:value={newPhaseType} class="phase-type-select">
                <option value="ga">GA</option>
                <option value="ts">TS</option>
                {#if isBitwise || isMultiStage}
                  <option value="neurogenesis">Neurogenesis</option>
                  <option value="synaptogenesis">Synaptogenesis</option>
                  <option value="axonogenesis">Axonogenesis</option>
                {/if}
              </select>
              {#if !isAdaptationType(newPhaseType)}
                <label class="inline-check"><input type="checkbox" bind:checked={newPhaseNeurons} /> Neurons</label>
                <label class="inline-check"><input type="checkbox" bind:checked={newPhaseBits} /> Bits</label>
                <label class="inline-check"><input type="checkbox" bind:checked={newPhaseConnections} /> Connections</label>
              {/if}
            {/if}
            <button type="button" class="btn btn-add" on:click={addPhase}
              disabled={!newPhaseGrid && !isAdaptationType(newPhaseType) && !newPhaseNeurons && !newPhaseBits && !newPhaseConnections}>
              + Add Phase
            </button>
          </div>
        </div>

        <!-- Architecture config (single-stage only — multi-stage config is above) -->
        {#if !isMultiStage}
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

  .form-row-header {
    display: grid;
    grid-template-columns: 2fr 2fr 1fr;
    gap: 0.75rem;
  }

  .form-row-3 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
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

  .phase-type.adapt {
    background: rgba(168, 85, 247, 0.15);
    color: #a855f7;
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

  /* Multi-stage config */
  .stage-config-inline {
    margin-bottom: 0;
  }

  .stage-config-header {
    font-size: 1rem;
    font-weight: 600;
    color: var(--accent-blue);
    margin-bottom: 0.375rem;
  }

  .stage-fields {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
  }

  .stage-field label {
    font-size: 1rem;
    font-weight: 400;
    color: var(--text-secondary);
    margin-bottom: 0.25rem;
  }

  .shared-params-header {
    font-size: 1rem;
    font-weight: 500;
    color: var(--text-secondary);
    margin: 0.75rem 0 0.5rem 0;
    padding-top: 0.75rem;
    border-top: 1px solid var(--border);
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
</style>
