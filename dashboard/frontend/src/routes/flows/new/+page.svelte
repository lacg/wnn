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

  // IDS-specific config
  let idsClassification = 'binary';
  let idsNBits = 8;
  let idsValFraction = 0.25;
  let idsKFolds = 5;
  let idsFitnessWeightF1 = 0.0;
  let idsSplit = 'standard';
  let idsMinBits = 4;
  let idsMaxBits = 14;
  let idsMinNeurons = 5;
  let idsMaxNeurons = 300;
  let idsMaxBitDelta = 0;

  // Multi-stage config
  let numStages = 1;
  let stageMode = 'input_concat';
  let selectedStage = 0;

  interface StageConfig {
    clusterType: string;
    k: number;
    contextSize: number;
    minBits: number;
    maxBits: number;
    minNeurons: number;
    maxNeurons: number;
    neuronsGrid: string;
    bitsGrid: string;
    gaGenerations: number;
    tsIterations: number;
    adaptationIterations: number;
    populationSize: number;
    neighborsPerIter: number;
    patience: number;
    fitnessPercentile: number;
    fitnessCalculator: string;
    fitnessWeightCe: number;
    fitnessWeightAcc: number;
    minAccuracyFloor: number;
    thresholdStart: number;
    thresholdStep: number;
  }

  // Mode-specific grid defaults (must match worker.py fallback grids)
  const GRID_DEFAULTS: Record<string, { neurons: string; bits: string }> = {
    bitwise:           { neurons: '5,10,25,50', bits: '4,6,8,10,12,16,20,24' },
    tiered:            { neurons: '20,30,40,50',             bits: '18,19,20,21,22,23' },
    semantic:          { neurons: '20,30,40,50',             bits: '18,19,20,21,22,23' },
    semantic_bitwise:  { neurons: '5,10,25,50', bits: '4,6,8,10,12,16,20,24' },
    selector:          { neurons: '5,10,15',                  bits: '5,6,7,8,9,10' },
  };
  const ALL_DEFAULT_NEURONS = new Set(Object.values(GRID_DEFAULTS).map(d => d.neurons));
  const ALL_DEFAULT_BITS = new Set(Object.values(GRID_DEFAULTS).map(d => d.bits));

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

  function defaultStageConfig(): StageConfig {
    return {
      clusterType: 'bitwise', k: 256, contextSize: 4,
      minBits: 4, maxBits: 24, minNeurons: 5, maxNeurons: 300,
      neuronsGrid: '5,10,25,50',
      bitsGrid: '4,6,8,10,12,16,20,24',
      gaGenerations: 250, tsIterations: 250, adaptationIterations: 50,
      populationSize: 50, neighborsPerIter: 50, patience: 10,
      fitnessPercentile: 0.75, fitnessCalculator: 'harmonic_rank',
      fitnessWeightCe: 1.0, fitnessWeightAcc: 1.0,
      minAccuracyFloor: 0, thresholdStart: 0, thresholdStep: 1,
    };
  }

  let stageConfigs: StageConfig[] = [defaultStageConfig(), defaultStageConfig()];

  // Track previous stage for save/load on switch
  let _prevStage = 0;

  function saveSearchParamsToStage(stage: number) {
    if (stage < 0 || stage >= stageConfigs.length) return;
    stageConfigs[stage] = {
      ...stageConfigs[stage],
      gaGenerations, tsIterations, adaptationIterations,
      populationSize, neighborsPerIter, patience,
      fitnessPercentile, fitnessCalculator,
      fitnessWeightCe, fitnessWeightAcc,
      minAccuracyFloor, thresholdStart, thresholdStep,
    };
  }

  function loadSearchParamsFromStage(stage: number) {
    if (stage < 0 || stage >= stageConfigs.length) return;
    const c = stageConfigs[stage];
    gaGenerations = c.gaGenerations;
    tsIterations = c.tsIterations;
    adaptationIterations = c.adaptationIterations;
    populationSize = c.populationSize;
    neighborsPerIter = c.neighborsPerIter;
    patience = c.patience;
    fitnessPercentile = c.fitnessPercentile;
    fitnessCalculator = c.fitnessCalculator;
    fitnessWeightCe = c.fitnessWeightCe;
    fitnessWeightAcc = c.fitnessWeightAcc;
    minAccuracyFloor = c.minAccuracyFloor;
    thresholdStart = c.thresholdStart;
    thresholdStep = c.thresholdStep;
  }

  // Shared multi-stage architecture params
  let msMemoryMode = 'QUAD_WEIGHTED';
  let msNeuronSampleRate = 0.25;
  let msTemplate = 'full';
  let invalidMode = false;
  let topM = 5;

  $: isMultiStage = numStages >= 2;
  $: isBitwise = !isMultiStage && (template === 'bitwise-7-phase' || template === 'bitwise-10-phase');
  $: isIDS = !isMultiStage && (template === 'ids-binary-7-phase' || template === 'ids-multi-7-phase' || template === 'ids-binary-10-phase' || template === 'ids-multi-10-phase');

  // Resize stageConfigs when numStages changes
  $: {
    const prevLen = stageConfigs.length;
    while (stageConfigs.length < numStages) {
      const newIdx = stageConfigs.length;
      const cfg = defaultStageConfig();
      // Apply mode-specific grid defaults for the new stage
      const mode = stageMode === 'selector' && newIdx > 0 ? 'selector'
        : (cfg.clusterType === 'tiered' || cfg.clusterType === 'semantic') ? cfg.clusterType
        : cfg.clusterType === 'semantic_bitwise' ? 'semantic_bitwise' : 'bitwise';
      cfg.neuronsGrid = GRID_DEFAULTS[mode].neurons;
      cfg.bitsGrid = GRID_DEFAULTS[mode].bits;
      stageConfigs = [...stageConfigs, cfg];
    }
    if (stageConfigs.length > numStages) {
      stageConfigs = stageConfigs.slice(0, numStages);
    }
    if (selectedStage >= numStages) {
      selectedStage = Math.max(0, numStages - 1);
    }
  }

  // Save/load search params when switching stages
  $: if (isMultiStage && selectedStage !== _prevStage) {
    saveSearchParamsToStage(_prevStage);
    loadSearchParamsFromStage(selectedStage);
    _prevStage = selectedStage;
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
  let reweightRounds = 0;
  let reweightMaxBoost = 4;
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
    } else if (templateName === 'ids-binary-7-phase') {
      gaGenerations = 250;
      tsIterations = 250;
      populationSize = 150;
      neighborsPerIter = 150;
      patience = 5;
      fitnessPercentile = 0.75;
      fitnessCalculator = 'ids_security';
      fitnessWeightCe = 0.3;
      fitnessWeightAcc = 1.0;
      idsClassification = 'binary';
      idsNBits = 8;
      idsValFraction = 0.25;
      idsKFolds = 5;
      idsFitnessWeightF1 = 0.0;
      idsSplit = 'standard';
    } else if (templateName === 'ids-multi-7-phase') {
      gaGenerations = 250;
      tsIterations = 250;
      populationSize = 150;
      neighborsPerIter = 150;
      patience = 5;
      fitnessPercentile = 0.75;
      fitnessCalculator = 'ids_security';
      fitnessWeightCe = 0.3;
      fitnessWeightAcc = 1.0;
      idsClassification = 'multi_tiered';
      idsNBits = 8;
      idsValFraction = 0.25;
      idsKFolds = 5;
      idsFitnessWeightF1 = 0.0;
      idsSplit = 'standard';
    } else if (templateName === 'ids-binary-10-phase') {
      gaGenerations = 250;
      tsIterations = 250;
      adaptationIterations = 50;
      populationSize = 150;
      neighborsPerIter = 150;
      patience = 5;
      fitnessPercentile = 0.75;
      fitnessCalculator = 'ids_security';
      fitnessWeightCe = 0.3;
      fitnessWeightAcc = 1.0;
      idsClassification = 'binary';
      idsNBits = 8;
      idsValFraction = 0.25;
      idsKFolds = 5;
      idsFitnessWeightF1 = 0.0;
      idsSplit = 'standard';
    } else if (templateName === 'ids-multi-10-phase') {
      gaGenerations = 250;
      tsIterations = 250;
      adaptationIterations = 50;
      populationSize = 150;
      neighborsPerIter = 150;
      patience = 5;
      fitnessPercentile = 0.75;
      fitnessCalculator = 'ids_security';
      fitnessWeightCe = 0.3;
      fitnessWeightAcc = 1.0;
      idsClassification = 'multi_tiered';
      idsNBits = 8;
      idsValFraction = 0.25;
      idsKFolds = 5;
      idsFitnessWeightF1 = 0.0;
      idsSplit = 'standard';
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

  /** Generate the 2-phase fast pipeline for a single stage. */
  function generate2PhaseForStage(prefix: string): PhaseSpec[] {
    return [
      { name: `${prefix}: Grid Search`, experiment_type: 'ga', optimize_bits: true, optimize_neurons: true, optimize_connections: false, phase_type: 'grid_search' },
      { name: `${prefix}: GA Neurons`, experiment_type: 'ga', optimize_bits: false, optimize_neurons: true, optimize_connections: false },
    ];
  }

  function generatePhasesForStage(prefix: string, tmpl: string): PhaseSpec[] {
    return tmpl === 'fast' ? generate2PhaseForStage(prefix) : generate10PhaseForStage(prefix);
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

    if (templateName === 'ids-binary-7-phase' || templateName === 'ids-multi-7-phase') {
      const grid: PhaseSpec = { name: 'Grid Search (neurons × bits)', experiment_type: 'ga', optimize_bits: true, optimize_neurons: true, optimize_connections: false, phase_type: 'grid_search' };
      return [grid, ...neuronsPhases, ...bitsPhases, ...connectionsPhases];
    }

    if (templateName === 'ids-binary-10-phase' || templateName === 'ids-multi-10-phase') {
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

  // Regenerate per-stage phases when template changes
  let _prevMsTemplate = msTemplate;
  $: if (isMultiStage && msTemplate !== _prevMsTemplate) {
    _prevMsTemplate = msTemplate;
    perStagePhases = Array.from({ length: numStages }, (_, i) =>
      generatePhasesForStage(`S${i}`, msTemplate)
    );
  }

  // Resize per-stage phases when numStages changes
  $: if (isMultiStage && perStagePhases.length !== numStages) {
    const updated = [...perStagePhases];
    while (updated.length < numStages) {
      updated.push(generatePhasesForStage(`S${updated.length}`, msTemplate));
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
      // Save current stage's search params before submit
      if (isMultiStage) {
        saveSearchParamsToStage(selectedStage);
      }

      const adaptationTypes = new Set(['neurogenesis', 'synaptogenesis', 'axonogenesis']);

      // Helper to get search params for a given experiment
      function getSearchParams(exp: PhaseSpec, stageIdx: number) {
        const cfg = isMultiStage ? stageConfigs[stageIdx] : null;
        const gens = cfg ? cfg.gaGenerations : gaGenerations;
        const tsIts = cfg ? cfg.tsIterations : tsIterations;
        const adaptIts = cfg ? cfg.adaptationIterations : adaptationIterations;
        const pop = cfg ? cfg.populationSize : populationSize;
        const neighbors = cfg ? cfg.neighborsPerIter : neighborsPerIter;

        const isAdaptation = adaptationTypes.has(exp.phase_type ?? '');
        return {
          generations: (exp.phase_type === 'grid_search') ? undefined
            : isAdaptation ? adaptIts
            : (exp.experiment_type === 'ga' ? gens : undefined),
          iterations: (exp.phase_type === 'grid_search') ? undefined
            : isAdaptation ? adaptIts
            : (exp.experiment_type === 'ts' ? tsIts : undefined),
          population_size: pop,
          neighbors_per_iter: neighbors,
          ...(exp.phase_type ? { phase_type: exp.phase_type } : {}),
        };
      }

      // Enrich experiments with per-stage search params
      let enrichedExperiments;
      if (isMultiStage) {
        enrichedExperiments = perStagePhases.flatMap((phases, stageIdx) =>
          phases.map((exp) => ({ ...exp, params: getSearchParams(exp, stageIdx) }))
        );
      } else {
        enrichedExperiments = singleStagePhases.map((exp) => ({
          ...exp, params: getSearchParams(exp, 0),
        }));
      }

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
        params.stage_context_size = stageConfigs.slice(0, numStages).map(s => s.contextSize);
        params.context_size = Math.max(...stageConfigs.slice(0, numStages).map(s => s.contextSize));
        params.stage_mode = stageMode;
        // Per-stage bounds
        params.stage_min_bits = stageConfigs.slice(0, numStages).map(s => s.minBits);
        params.stage_max_bits = stageConfigs.slice(0, numStages).map(s => s.maxBits);
        params.stage_min_neurons = stageConfigs.slice(0, numStages).map(s => s.minNeurons);
        params.stage_max_neurons = stageConfigs.slice(0, numStages).map(s => s.maxNeurons);
        // Global fallbacks (from S0 values)
        params.min_bits = stageConfigs[0].minBits;
        params.max_bits = stageConfigs[0].maxBits;
        params.min_neurons = stageConfigs[0].minNeurons;
        params.max_neurons = stageConfigs[0].maxNeurons;
        params.memory_mode = msMemoryMode;
        params.neuron_sample_rate = msNeuronSampleRate;
        params.invalid_mode = invalidMode;
        params.top_m = topM;
        // Per-stage search params
        params.stage_ga_generations = stageConfigs.slice(0, numStages).map(s => s.gaGenerations);
        params.stage_ts_iterations = stageConfigs.slice(0, numStages).map(s => s.tsIterations);
        params.stage_adaptation_iterations = stageConfigs.slice(0, numStages).map(s => s.adaptationIterations);
        params.stage_population_size = stageConfigs.slice(0, numStages).map(s => s.populationSize);
        params.stage_neighbors_per_iter = stageConfigs.slice(0, numStages).map(s => s.neighborsPerIter);
        params.stage_patience = stageConfigs.slice(0, numStages).map(s => s.patience);
        params.stage_fitness_percentile = stageConfigs.slice(0, numStages).map(s => s.fitnessPercentile);
        params.stage_fitness_calculator = stageConfigs.slice(0, numStages).map(s => s.fitnessCalculator);
        params.stage_fitness_weight_ce = stageConfigs.slice(0, numStages).map(s => s.fitnessWeightCe);
        params.stage_fitness_weight_acc = stageConfigs.slice(0, numStages).map(s => s.fitnessWeightAcc);
        // Per-stage grids (comma-separated strings → arrays of numbers)
        params.stage_neurons_grid = stageConfigs.slice(0, numStages).map(s =>
          s.neuronsGrid.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v))
        );
        params.stage_bits_grid = stageConfigs.slice(0, numStages).map(s =>
          s.bitsGrid.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v))
        );
        params.ms_template = msTemplate;
        if (reweightRounds > 0) {
          params.reweight_rounds = reweightRounds;
          params.reweight_max_boost = reweightMaxBoost;
        }
      } else if (isIDS) {
        params.architecture_type = 'ids';
        // Split combined classification into classification + ids_arch_type
        const isBitwiseIds = idsClassification === 'multi_bitwise';
        const classMap: Record<string, string> = {
          'binary': 'binary',
          'hierarchical': 'hierarchical',
          'multi_tiered': 'multi',
          'multi_bitwise': 'multi',
        };
        params.ids_classification = classMap[idsClassification] || 'binary';
        params.ids_arch_type = isBitwiseIds ? 'bitwise' : 'tiered';
        params.ids_n_bits = idsNBits;
        params.ids_val_fraction = idsValFraction;
        params.ids_num_parts = idsKFolds > 1 ? idsKFolds : 3;
        params.ids_k_folds = idsKFolds;
        params.ids_fitness_weight_f1 = idsFitnessWeightF1;
        params.ids_split = idsSplit;
        params.min_bits = idsMinBits;
        params.max_bits = idsMaxBits;
        params.min_neurons = idsMinNeurons;
        params.max_neurons = idsMaxNeurons;
        if (idsMaxBitDelta > 0) params.max_bit_delta = idsMaxBitDelta;
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
                <option value="ids-binary-7-phase">IDS Binary 7-Phase (UNSW-NB15)</option>
                <option value="ids-binary-10-phase">IDS Binary 10-Phase + *genesis (UNSW-NB15)</option>
                <option value="ids-multi-7-phase">IDS Multi-class 7-Phase (UNSW-NB15)</option>
                <option value="ids-multi-10-phase">IDS Multi-class 10-Phase + *genesis (UNSW-NB15)</option>
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
                {:else if template === 'ids-binary-7-phase'}
                  IDS binary (attack vs normal) on UNSW-NB15 &mdash; 7 phases
                {:else if template === 'ids-binary-10-phase'}
                  IDS binary + neurogenesis/synaptogenesis/axonogenesis &mdash; 10 phases
                {:else if template === 'ids-multi-7-phase'}
                  IDS 10-class classification on UNSW-NB15 &mdash; 7 phases
                {:else if template === 'ids-multi-10-phase'}
                  IDS 10-class + neurogenesis/synaptogenesis/axonogenesis &mdash; 10 phases
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

          {#if !isIDS}
          <div class="form-group">
            <label for="contextSize">Context Size</label>
            <input type="number" id="contextSize" bind:value={contextSize} min="2" max="16" />
            <span class="field-hint">N-gram context window (4 = 4-gram)</span>
          </div>
          {/if}
        </div>

        <div class="form-row">
          <div class="form-group">
            <label for="fitnessCalculator">Fitness Calculator</label>
            <select id="fitnessCalculator" bind:value={fitnessCalculator}>
              <option value="normalized">Normalized (Recommended)</option>
              <option value="harmonic_rank">Harmonic Rank</option>
              <option value="normalized_harmonic">Normalized Harmonic</option>
              <option value="ids_security">IDS Security: F1 × (1−FPR)²</option>
              <option value="ids_recall">IDS Recall: F1 × (1−FPR)¹</option>
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

        {#if isMultiStage}
          <div class="form-row">
            <div class="form-group">
              <label for="reweightRounds">S1 Re-weight Rounds</label>
              <input type="number" id="reweightRounds" bind:value={reweightRounds} min="0" max="10" />
              <span class="field-hint">Iterative re-weighting for S1 prediction (0 = off)</span>
            </div>

            <div class="form-group">
              <label for="reweightMaxBoost">S1 Max Boost</label>
              <input type="number" id="reweightMaxBoost" bind:value={reweightMaxBoost} min="1" max="10" />
              <span class="field-hint">Max nudge repeat for misclassified examples</span>
            </div>
          </div>
        {/if}
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
            {#if isBitwise || isMultiStage || isIDS}
              <label class="inline-check">
                <input type="checkbox" bind:checked={newPhaseGrid} /> Grid Search
              </label>
            {/if}
            {#if !newPhaseGrid}
              <select bind:value={newPhaseType} class="phase-type-select">
                <option value="ga">GA</option>
                <option value="ts">TS</option>
                {#if isBitwise || isMultiStage || isIDS}
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
          {#if isIDS}
            <div class="form-section">
              <h2>IDS Configuration (UNSW-NB15)</h2>
              <div class="form-row">
                <div class="form-group">
                  <label for="idsClassification">Classification</label>
                  <select id="idsClassification" bind:value={idsClassification}>
                    <option value="binary">Binary (attack vs normal)</option>
                    <option value="hierarchical">Hierarchical S0→S1 (binary + 9 attack types)</option>
                    <option value="multi_tiered">Multi-class Tiered (10 categories)</option>
                    <option value="multi_bitwise">Multi-class Bitwise (10 categories)</option>
                  </select>
                  <span class="field-hint">
                    {#if idsClassification === 'binary'}2 classes — tiered architecture
                    {:else if idsClassification === 'hierarchical'}S0: Normal vs Attack → S1: 9 attack types (separate genomes)
                    {:else if idsClassification === 'multi_tiered'}10 classes — frequency-based tier allocation
                    {:else}10 classes — per-cluster independent bits/neurons
                    {/if}
                  </span>
                </div>
                <div class="form-group">
                  <label for="idsSplit">Data Split</label>
                  <select id="idsSplit" bind:value={idsSplit}>
                    <option value="standard">Standard (paper split)</option>
                    <option value="random">Random (stratified)</option>
                  </select>
                  <span class="field-hint">Standard = original train/test split</span>
                </div>
              </div>
              <div class="form-row">
                <div class="form-group">
                  <label for="idsNBits">Thermometer Bits</label>
                  <input type="number" id="idsNBits" bind:value={idsNBits} min="4" max="16" />
                  <span class="field-hint">Bits per feature (8 = 336 total input bits)</span>
                </div>
                <div class="form-group">
                  <label for="idsValFraction">Validation Fraction</label>
                  <input type="number" id="idsValFraction" bind:value={idsValFraction} min="0" max="0.5" step="0.05" />
                  <span class="field-hint">Holdout from training for optimizer eval</span>
                </div>
              </div>
              <div class="form-row">
                <div class="form-group">
                  <label for="idsKFolds">K-Fold CV</label>
                  <input type="number" id="idsKFolds" bind:value={idsKFolds} min="1" max="10" />
                  <span class="field-hint">1 = off, 5 = default (also sets data partitions)</span>
                </div>
              </div>
              <h3>Neuron Architecture Bounds</h3>
              <div class="form-row">
                <div class="form-group">
                  <label for="idsMinBits">Min Bits</label>
                  <input type="number" id="idsMinBits" bind:value={idsMinBits} min="2" max="32" />
                  <span class="field-hint">Min address bits per neuron</span>
                </div>
                <div class="form-group">
                  <label for="idsMaxBits">Max Bits</label>
                  <input type="number" id="idsMaxBits" bind:value={idsMaxBits} min="2" max="32" />
                  <span class="field-hint">Max address bits (lower = more generalization)</span>
                </div>
                <div class="form-group">
                  <label for="idsMinNeurons">Min Neurons</label>
                  <input type="number" id="idsMinNeurons" bind:value={idsMinNeurons} min="1" max="500" />
                  <span class="field-hint">Min neurons per class</span>
                </div>
                <div class="form-group">
                  <label for="idsMaxNeurons">Max Neurons</label>
                  <input type="number" id="idsMaxNeurons" bind:value={idsMaxNeurons} min="1" max="500" />
                  <span class="field-hint">Max neurons per class</span>
                </div>
              </div>
              <div class="form-row">
                <div class="form-group">
                  <label for="idsMaxBitDelta">Max Bit Delta</label>
                  <input type="number" id="idsMaxBitDelta" bind:value={idsMaxBitDelta} min="0" max="16" />
                  <span class="field-hint">0 = auto (~10% of range). Limits bit jumps per mutation to prevent overfitting</span>
                </div>
              </div>
            </div>
          {:else if isBitwise}
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

  .form-actions {
    display: flex;
    gap: 1rem;
    justify-content: flex-end;
  }

  .btn {
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 500;
    text-decoration: none;
    border: none;
    cursor: pointer;
    transition: all 0.25s ease;
  }

  .btn-primary {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.85), rgba(99, 102, 241, 0.85));
    border: 1px solid rgba(59, 130, 246, 0.4);
    color: white;
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.15);
  }

  .btn-primary:hover:not(:disabled) {
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.2);
    transform: translateY(-1px);
  }

  .btn-primary:active:not(:disabled) {
    transform: translateY(0);
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.2);
  }

  .btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn-secondary {
    background: rgba(51, 65, 85, 0.5);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid var(--glass-border);
    color: var(--text-primary);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15), var(--glass-inset);
  }

  .btn-secondary:hover {
    background: rgba(71, 85, 105, 0.5);
    border-color: var(--glass-border-highlight);
    transform: translateY(-1px);
  }

  .error-message {
    background: rgba(239, 68, 68, 0.1);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: var(--accent-red);
    padding: 0.75rem 1rem;
    border-radius: 8px;
    font-size: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 0 16px rgba(239, 68, 68, 0.1);
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
    padding: 0.375rem 0.5rem;
    border-radius: 8px;
    background: rgba(30, 41, 59, 0.3);
    border: 1px solid rgba(148, 163, 184, 0.08);
    transition: background 0.15s ease, border-color 0.15s ease;
  }

  .phase-item:hover {
    background: rgba(30, 41, 59, 0.5);
    border-color: rgba(148, 163, 184, 0.15);
  }

  .phase-num {
    width: 1.5rem;
    height: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 6px;
    font-size: 1rem;
    color: var(--accent-blue);
    font-weight: 600;
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
    border: 1px solid rgba(59, 130, 246, 0.2);
    color: var(--accent-blue);
    box-shadow: 0 0 8px rgba(59, 130, 246, 0.1);
  }

  .phase-type.ts {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.2);
    color: var(--accent-green);
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.1);
  }

  .phase-type.adapt {
    background: rgba(168, 85, 247, 0.15);
    border: 1px solid rgba(168, 85, 247, 0.2);
    color: #a855f7;
    box-shadow: 0 0 8px rgba(168, 85, 247, 0.1);
  }

  .phase-move {
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .move-btn {
    background: rgba(51, 65, 85, 0.4);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 4px;
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
    background: rgba(51, 65, 85, 0.4);
    border: 1px solid var(--glass-border);
    color: var(--text-primary);
    padding: 0.25rem 0.75rem;
    border-radius: 8px;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.2s ease;
    margin-left: auto;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
  }

  .btn-add:hover:not(:disabled) {
    background: rgba(71, 85, 105, 0.5);
    border-color: var(--glass-border-highlight);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
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
