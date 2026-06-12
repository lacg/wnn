/**
 * Pure template → phases/defaults logic for the New Flow page (P3 decomposition).
 * Extracted verbatim from routes/flows/new/+page.svelte — no component state here.
 */

export interface StageConfig {
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

// Phase spec interface
export interface PhaseSpec {
  name: string;
  experiment_type: 'ga' | 'ts' | 'grid_search' | 'neurogenesis' | 'synaptogenesis' | 'axonogenesis';
  optimize_bits: boolean;
  optimize_neurons: boolean;
  optimize_connections: boolean;
  phase_type?: 'grid_search' | 'neurogenesis' | 'synaptogenesis' | 'axonogenesis'
    | 'ga_neurons' | 'ga_bits' | 'ga_connections' | 'ga_memory'
    | 'ts_neurons' | 'ts_bits' | 'ts_connections' | 'ts_memory';
}

// Mode-specific grid defaults (must match worker.py fallback grids)
export const GRID_DEFAULTS: Record<string, { neurons: string; bits: string }> = {
  bitwise:           { neurons: '5,10,25,50', bits: '4,6,8,10,12,16,20,24' },
  tiered:            { neurons: '20,30,40,50',             bits: '18,19,20,21,22,23' },
  semantic:          { neurons: '20,30,40,50',             bits: '18,19,20,21,22,23' },
  semantic_bitwise:  { neurons: '5,10,25,50', bits: '4,6,8,10,12,16,20,24' },
  selector:          { neurons: '5,10,15',                  bits: '5,6,7,8,9,10' },
};
export const ALL_DEFAULT_NEURONS = new Set(Object.values(GRID_DEFAULTS).map(d => d.neurons));
export const ALL_DEFAULT_BITS = new Set(Object.values(GRID_DEFAULTS).map(d => d.bits));

export function defaultStageConfig(): StageConfig {
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

/** Generate the 10-phase pipeline for a single stage. */
export function generate10PhaseForStage(prefix: string): PhaseSpec[] {
  return [
    { name: `${prefix}: Grid Search`, experiment_type: 'grid_search', optimize_bits: false, optimize_neurons: false, optimize_connections: false, phase_type: 'grid_search' },
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
export function generate2PhaseForStage(prefix: string): PhaseSpec[] {
  return [
    { name: `${prefix}: Grid Search`, experiment_type: 'grid_search', optimize_bits: false, optimize_neurons: false, optimize_connections: false, phase_type: 'grid_search' },
    { name: `${prefix}: GA Neurons`, experiment_type: 'ga', optimize_bits: false, optimize_neurons: true, optimize_connections: false },
  ];
}

export function generatePhasesForStage(prefix: string, tmpl: string): PhaseSpec[] {
  return tmpl === 'fast' ? generate2PhaseForStage(prefix) : generate10PhaseForStage(prefix);
}

/** Generate single-stage phases from template. */
export function generatePhases(templateName: string, order: string): PhaseSpec[] {
  if (templateName === 'empty') return [];

  // Controller phases use phase_type directly (the worker resolves it via
  // PHASE_TO_KIND_DIM); ga_memory is paradigm-B (no training).
  if (templateName === 'controller-ga-memory') {
    return [{ name: 'GA Memory', experiment_type: 'ga', optimize_bits: false, optimize_neurons: false, optimize_connections: false, phase_type: 'ga_memory' }];
  }
  if (templateName === 'controller-full-matrix') {
    return [
      { name: 'GA Neurons', experiment_type: 'ga', optimize_bits: false, optimize_neurons: true, optimize_connections: false, phase_type: 'ga_neurons' },
      { name: 'GA Bits', experiment_type: 'ga', optimize_bits: true, optimize_neurons: false, optimize_connections: false, phase_type: 'ga_bits' },
      { name: 'GA Connections', experiment_type: 'ga', optimize_bits: false, optimize_neurons: false, optimize_connections: true, phase_type: 'ga_connections' },
      { name: 'GA Memory', experiment_type: 'ga', optimize_bits: false, optimize_neurons: false, optimize_connections: false, phase_type: 'ga_memory' },
    ];
  }

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

  const grid: PhaseSpec = { name: 'Grid Search (neurons × bits)', experiment_type: 'grid_search', optimize_bits: false, optimize_neurons: false, optimize_connections: false, phase_type: 'grid_search' };
  const synaptogenesisPhase: PhaseSpec = { name: 'Synaptogenesis', experiment_type: 'synaptogenesis', optimize_bits: false, optimize_neurons: false, optimize_connections: false, phase_type: 'synaptogenesis' };

  if (templateName === 'bitwise-2-phase' || templateName === 'ids-binary-2-phase' || templateName === 'ids-multi-2-phase') {
    return [grid, neuronsPhases[0]];
  }

  if (templateName === 'bitwise-5-phase' || templateName === 'ids-binary-5-phase' || templateName === 'ids-multi-5-phase') {
    return [grid, neuronsPhases[0], bitsPhases[0], synaptogenesisPhase, connectionsPhases[0]];
  }

  if (templateName === 'bitwise-10-phase') {
    const neurogenesisPhase: PhaseSpec = { name: 'Neurogenesis', experiment_type: 'neurogenesis', optimize_bits: false, optimize_neurons: false, optimize_connections: false, phase_type: 'neurogenesis' };
    const axonogenesisPhase: PhaseSpec = { name: 'Axonogenesis', experiment_type: 'axonogenesis', optimize_bits: false, optimize_neurons: false, optimize_connections: false, phase_type: 'axonogenesis' };
    return [
      grid,
      neuronsPhases[0], neurogenesisPhase, neuronsPhases[1],
      bitsPhases[0], synaptogenesisPhase, bitsPhases[1],
      connectionsPhases[0], axonogenesisPhase, connectionsPhases[1],
    ];
  }

  if (templateName === 'bitwise-7-phase') {
    if (order === 'bits_first') return [grid, ...bitsPhases, ...neuronsPhases, ...connectionsPhases];
    return [grid, ...neuronsPhases, ...bitsPhases, ...connectionsPhases];
  }

  if (templateName === 'ids-binary-7-phase' || templateName === 'ids-multi-7-phase') {
    return [grid, ...neuronsPhases, ...bitsPhases, ...connectionsPhases];
  }

  if (templateName === 'ids-binary-10-phase' || templateName === 'ids-multi-10-phase') {
    const neurogenesisPhase: PhaseSpec = { name: 'Neurogenesis', experiment_type: 'neurogenesis', optimize_bits: false, optimize_neurons: false, optimize_connections: false, phase_type: 'neurogenesis' };
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

export function generatePhaseName(type: string, neurons: boolean, bits: boolean, connections: boolean): string {
  const targets: string[] = [];
  if (neurons) targets.push('Neurons');
  if (bits) targets.push('Bits');
  if (connections) targets.push('Connections');
  return `${type.toUpperCase()} ${targets.join(' + ')}`;
}
