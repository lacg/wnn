/**
 * Shared IDS-task detection. The architecture_type field can be missing on
 * older rows, so fall back to IDS-only signals (best_f1 on iterations,
 * ids-typed experiments) instead of silently rendering LM columns.
 */

import type { Experiment, Flow, Iteration } from './types';

export function isIdsExperiment(experiment: Experiment | null, iterations: Iteration[]): boolean {
  return experiment?.architecture_type === 'ids'
    || iterations.some(i => i.best_f1 != null);
}

export function isIdsFlow(flow: Flow | null, experiments: Experiment[]): boolean {
  return flow?.config?.params?.architecture_type === 'ids'
    || experiments.some(e => e.architecture_type === 'ids');
}
