/**
 * Shared row/point types for the experiment detail page and its extracted
 * child components (P3 decomposition).
 */

import type { TestFootprint } from '$lib/types'

/** One point of the cumulative test progression (init/final per phase). */
export interface TestProgressionPoint
{
	label: string
	expId: number
	sequenceOrder: number
	testPoint: 'init' | 'final'
	summaries: { genomeType: string, ce: number, accuracy: number, f1_macro: number | null, fpr: number | null, threshold_metadata: any | null, footprint: TestFootprint | null }[]
}

/** One ranked grid-search config result. */
export interface GridSearchRow
{
	rank: number
	neurons: number
	bits: number
	ce: number
	accuracy: number
	fitness: number | null
	count: number
	elapsed?: number
	f1_macro?: number | null
	fpr?: number | null
}

/** One genome of the seeded (expanded) population after grid search. */
export interface ExpandedGenome
{
	rank: number
	neurons: number
	bits: number
	ce: number
	accuracy: number
	fitness: number | null
	f1_macro?: number | null
	fpr?: number | null
}

/** In-memory live generation progress payload from the dashboard. */
export interface LiveProgress
{
	generation: number
	total_generations: number
	phase: string
	evaluated: number
	target_count: number
	viable: number | null
	best_ce: number
	best_acc: number
	elapsed_secs: number
}
