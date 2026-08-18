<script lang="ts">
	import { onDestroy } from 'svelte'
	import { page } from '$app/stores'
	import type { Experiment, Iteration, GenomeEvaluation, GenomeTier, Flow, ValidationSummary, Checkpoint } from '$lib/types'
	import { makeLatestGuard } from '$lib/api'
	import { isIdsExperiment, isMulticlassFlow, hasMulticlassMetadata } from '$lib/ids'
	import { gatingRunUpdates } from '$lib/stores'
	import BitwiseClusterStats from '$lib/components/BitwiseClusterStats.svelte'
	import ExperimentHeader from '$lib/components/experiment/ExperimentHeader.svelte'
	import FlowProgressBar from '$lib/components/experiment/FlowProgressBar.svelte'
	import ExperimentInfoCards from '$lib/components/experiment/ExperimentInfoCards.svelte'
	import ValidationProgressionTable from '$lib/components/experiment/ValidationProgressionTable.svelte'
	import GatingResultsPanel from '$lib/components/experiment/GatingResultsPanel.svelte'
	import TierStatsPanel from '$lib/components/experiment/TierStatsPanel.svelte'
	import ProgressChart from '$lib/components/experiment/ProgressChart.svelte'
	import GridSearchResults from '$lib/components/experiment/GridSearchResults.svelte'
	import LiveProgressBar from '$lib/components/experiment/LiveProgressBar.svelte'
	import IterationsTable from '$lib/components/experiment/IterationsTable.svelte'
	import IterationDetailModal from '$lib/components/experiment/IterationDetailModal.svelte'
	import type { ValidationProgressionPoint, GridSearchRow, ExpandedGenome, LiveProgress } from '$lib/components/experiment/types'
	import type { GatingRun } from '$lib/types'

	let experiment: Experiment | null = null
	let iterations: Iteration[] = []
	let flowExperiments: Experiment[] = []
	let flow: Flow | null = null
	let validationSummaries: ValidationSummary[] = []
	let flowValidationSummaries: ValidationSummary[] = [] // All validation summaries for the flow
	let checkpoints: Checkpoint[] = []
	let loading = true
	let error: string | null = null
	let pollInterval: ReturnType<typeof setInterval> | null = null
	let flowPollInterval: ReturnType<typeof setInterval> | null = null

	// Per-class breakdown: which validation point + threshold mode to display.
  // Defaults to the first available point (chronologically earliest, e.g. INIT).
  // Kept at page level so the choice survives in-page navigation; bound into
  // ValidationProgressionTable → PerClassBreakdown.
	let perClassPointChoice: number = 0
	let perClassThresholdChoice: 'train_cal' | 'fixed_05' | 'val_cal' | 'platt' | 'beta' | 'empirical' | 'empirical_cumulative' = 'train_cal'

	// Iteration detail modal state
	let selectedIteration: Iteration | null = null
	let genomeEvaluations: GenomeEvaluation[] = []
	let loadingGenomes = false
	let showIterationModal = false

	// Gating state
	let gatingLoading = false
	let gatingRuns: GatingRun[] = []

	// Live generation progress (in-memory, no DB)
	let liveProgress: LiveProgress | null = null

	// Grid search results
	let gridSearchResults: GridSearchRow[] = []
	let expandedPopulation: ExpandedGenome[] = []
	let seedEvalComplete = false
	let gridSearchLoading = false

	$: experimentId = $page.params.id

	// Drops stale responses: any newer load/refresh invalidates older in-flight ones
	const requestGuard = makeLatestGuard()

	// Reload when experimentId changes (for in-page navigation).
  // Grid-search state must not survive across experiments — equal iteration
  // counts would otherwise show the previous experiment's grid results.
	$: if (experimentId)
	{
		_lastGridIterCount = 0
		_gridRetryAfter = 0
		gridSearchResults = []
		expandedPopulation = []
		seedEvalComplete = false
		loadExperiment()
	}

	// React to gating run updates from WebSocket
	$: if ($gatingRunUpdates && experiment && $gatingRunUpdates.experiment_id === experiment.id)
	{
		// Update the gating runs list
		const existingIdx = gatingRuns.findIndex(r => r.id === $gatingRunUpdates!.id)
		if (existingIdx >= 0)
		{
			gatingRuns[existingIdx] = $gatingRunUpdates
		}
		else
		{
			gatingRuns = [$gatingRunUpdates, ...gatingRuns]
		}
		gatingRuns = gatingRuns // Trigger reactivity

		// Also update experiment's gating_status for backward compat display
		experiment.gating_status = $gatingRunUpdates.status
		experiment = experiment
	}

	async function loadExperiment()
	{
		const token = requestGuard.begin()
		loading = true
		error = null
		// Reset flow context — stale flow data must not survive navigation to a
  	// flowless experiment.
		flow = null
		flowExperiments = []
		flowValidationSummaries = []

		try
		{
			const [expRes, itersRes, summariesRes, checkpointsRes, gatingRes] = await Promise.all([
				fetch(`/api/experiments/${experimentId}`),
				fetch(`/api/experiments/${experimentId}/iterations?limit=500`),
				fetch(`/api/experiments/${experimentId}/summaries`),
				fetch(`/api/checkpoints?experiment_id=${experimentId}`),
				fetch(`/api/experiments/${experimentId}/gating`)
			])

			if (!expRes.ok) throw new Error('Experiment not found')

			const newExperiment = await expRes.json()
			let newIterations = itersRes.ok ? await itersRes.json() : []
			let newSummaries = summariesRes.ok ? await summariesRes.json() : []
			let newCheckpoints = checkpointsRes.ok ? await checkpointsRes.json() : []
			let newGatingRuns = gatingRes.ok ? await gatingRes.json() : []

			// Ensure arrays
			if (!Array.isArray(newIterations)) newIterations = []
			if (!Array.isArray(newSummaries)) newSummaries = []
			if (!Array.isArray(newCheckpoints)) newCheckpoints = []
			if (!Array.isArray(newGatingRuns)) newGatingRuns = []

			// A newer load/refresh superseded this one — drop the stale response
			if (!requestGuard.isCurrent(token)) return

			experiment = newExperiment
			iterations = newIterations
			validationSummaries = newSummaries
			checkpoints = newCheckpoints
			gatingRuns = newGatingRuns

			// Fetch flow and its experiments if this experiment belongs to a flow
			if (experiment?.flow_id)
			{
				const [flowRes, flowExpsRes, flowValidationsRes] = await Promise.all([
					fetch(`/api/flows/${experiment.flow_id}`),
					fetch(`/api/flows/${experiment.flow_id}/experiments`),
					fetch(`/api/flows/${experiment.flow_id}/validations`)
				])
				const newFlow = flowRes.ok ? await flowRes.json() : null
				const exps = flowExpsRes.ok ? await flowExpsRes.json() : []
				const validations = flowValidationsRes.ok ? await flowValidationsRes.json() : []
				if (!requestGuard.isCurrent(token)) return
				if (newFlow) flow = newFlow
				flowExperiments = Array.isArray(exps) ? exps : []
				flowValidationSummaries = Array.isArray(validations) ? validations : []
			}
		}
		catch (e)
		{
			if (!requestGuard.isCurrent(token)) return
			error = e instanceof Error ? e.message : 'Failed to load experiment'
		}
		finally
		{
			if (requestGuard.isCurrent(token)) loading = false
		}
	}

	// Light refresh for running experiments - only fetch new iterations and status
	async function refreshRunningExperiment()
	{
		if (!experiment) return
		// Never race a full load (e.g. mid-navigation): the poll would mutate the
  	// old experiment object with the new id's data and invalidate the load.
		if (loading) return
		const prevStatus = experiment.status
		const token = requestGuard.begin()
		const idAtStart = experimentId
		// A response is stale if a newer request began OR the page navigated to a
  	// different experiment while this one was in flight.
		const isStale = () => !requestGuard.isCurrent(token) || experimentId !== idAtStart

		try
		{
			const [expRes, itersRes] = await Promise.all([
				fetch(`/api/experiments/${idAtStart}`),
				fetch(`/api/experiments/${idAtStart}/iterations?limit=500`)
			])

			if (expRes.ok)
			{
				const newExp = await expRes.json()
				if (isStale()) return
				// Update fields that change during execution
				experiment.status = newExp.status
				experiment.started_at = newExp.started_at
				experiment.current_iteration = newExp.current_iteration
				experiment.best_ce = newExp.best_ce
				experiment.best_accuracy = newExp.best_accuracy
				experiment.status_message = newExp.status_message
				experiment.ended_at = newExp.ended_at
				experiment.gating_status = newExp.gating_status
				experiment.gating_results = newExp.gating_results
				experiment = experiment // Trigger Svelte reactivity for duration display

				// Status transition detected — do a full reload to get validation summaries,
  			// flow experiments, checkpoints, etc.
				if (prevStatus !== newExp.status)
				{
					await loadExperiment()
					return
				}

				// Also update this experiment's status in flowExperiments for Flow Progress bar
				if (flowExperiments.length > 0)
				{
					const idx = flowExperiments.findIndex(e => e.id === experiment!.id)
					if (idx >= 0)
					{
						flowExperiments[idx].status = newExp.status
						flowExperiments = flowExperiments // Trigger Svelte reactivity
					}
				}
			}

			if (itersRes.ok)
			{
				const newIters = await itersRes.json()
				if (isStale()) return
				if (Array.isArray(newIters))
				{
					iterations = newIters
				}
			}

			// Fetch live generation progress (in-memory on dashboard)
			try
			{
				const liveRes = await fetch(`/api/experiments/${idAtStart}/live-progress`)
				const newLiveProgress = liveRes.ok ? await liveRes.json() : null
				if (isStale()) return
				liveProgress = newLiveProgress
			}
			catch
			{
				if (isStale()) return
				liveProgress = null
			}

			// Also refresh flow data so duration stays in sync
			if (flow)
			{
				const flowRes = await fetch(`/api/flows/${flow.id}`)
				if (flowRes.ok)
				{
					const newFlow = await flowRes.json()
					if (isStale()) return
					flow.started_at = newFlow.started_at
					flow.completed_at = newFlow.completed_at
					flow.status = newFlow.status
					flow = flow
				}
			}
		}
		catch (e)
		{
			// Silently fail on refresh - don't disrupt the UI
			console.error('Refresh failed:', e)
		}
	}

	// Polling for active experiments - use light refresh
  // Poll any non-terminal status so we catch pending→running transitions
	$:
		{
			const isActive = experiment?.status === 'running' || experiment?.status === 'pending' || experiment?.status === 'queued'
			if (isActive)
			{
				if (!pollInterval)
				{
					pollInterval = setInterval(refreshRunningExperiment, 3000)
				}
			}
			else
			{
				if (pollInterval)
				{
					clearInterval(pollInterval)
					pollInterval = null
				}
				liveProgress = null
			}
		}

	// Refresh flow experiments (for Flow Progress bar)
	async function refreshFlowExperiments()
	{
		if (!experiment?.flow_id) return
		try
		{
			const res = await fetch(`/api/flows/${experiment.flow_id}/experiments`)
			if (res.ok)
			{
				const exps = await res.json()
				if (Array.isArray(exps))
				{
					flowExperiments = exps
				}
			}
		}
		catch (e)
		{
		// Silently fail
		}
	}

	// Poll flow experiments if any experiment in the flow is running/pending
	$: flowHasActiveExperiments = flowExperiments.some(e => e.status === 'running' || e.status === 'pending' || e.status === 'queued')
	$:
		{
			if (experiment?.flow_id && flowHasActiveExperiments)
			{
				if (!flowPollInterval)
				{
					flowPollInterval = setInterval(refreshFlowExperiments, 10000)
				}
			}
			else
			{
				if (flowPollInterval)
				{
					clearInterval(flowPollInterval)
					flowPollInterval = null
				}
			}
		}

	// Cleanup on destroy
	onDestroy(() =>
	{
		if (pollInterval)
		{
			clearInterval(pollInterval)
			pollInterval = null
		}
		if (flowPollInterval)
		{
			clearInterval(flowPollInterval)
			flowPollInterval = null
		}
	})

	async function openIterationDetails(iter: Iteration)
	{
		selectedIteration = iter
		showIterationModal = true
		loadingGenomes = true
		genomeEvaluations = []

		try
		{
			const res = await fetch(`/api/iterations/${iter.id}/genomes`)
			if (res.ok)
			{
				genomeEvaluations = await res.json()
			}
		}
		catch (e)
		{
			console.error('Failed to fetch genome evaluations:', e)
		}
		finally
		{
			loadingGenomes = false
		}
	}

	function closeIterationModal()
	{
		showIterationModal = false
		selectedIteration = null
		genomeEvaluations = []
	}

	async function runGating()
	{
		if (!experiment || gatingLoading) return

		gatingLoading = true
		try
		{
			const res = await fetch(`/api/experiments/${experimentId}/gating`, {
				method: 'POST'
			})

			if (res.ok)
			{
				const newRun: GatingRun = await res.json()
				// Add the new run to the list
				gatingRuns = [newRun, ...gatingRuns]
				// Update experiment status for display
				experiment.gating_status = newRun.status
				experiment = experiment
			}
			else
			{
				const data = await res.json()
				alert(data.error || 'Failed to start gating analysis')
			}
		}
		catch (e)
		{
			console.error('Failed to start gating:', e)
			alert('Failed to start gating analysis')
		}
		finally
		{
			gatingLoading = false
		}
	}

	// Flow steps directly from DB experiments (all exist with pending/running/completed status)
	$: flowSteps = flowExperiments
		.sort((a, b) => (a.sequence_order ?? 0) - (b.sequence_order ?? 0))
		.map((exp, i) => ({
			name: exp.name,
			status: exp.status,
			id: exp.id,
			index: i
		}))

	// Chart data - iterations directly from experiment
	$: displayIterations = iterations
	$: chartData = displayIterations.map(iter => ({
		iter: iter.iteration_num,
		ce: iter.best_ce,
		acc: iter.best_accuracy !== null ? iter.best_accuracy * 100 : null,
		avgCe: iter.avg_ce,
		avgAcc: iter.avg_accuracy !== null ? iter.avg_accuracy * 100 : null,
		f1: iter.best_f1 !== null && iter.best_f1 !== undefined ? iter.best_f1 * 100 : null,
		fpr: iter.best_fpr !== null && iter.best_fpr !== undefined ? iter.best_fpr * 100 : null
	}))

	// Metrics
	$: bestCE = iterations.length > 0 ? Math.min(...iterations.map(i => i.best_ce)) : Infinity
	$: bestAcc = iterations.length > 0
		? (() =>
		{
			const accVals = iterations.filter(i => i.best_accuracy !== null).map(i => i.best_accuracy!)
			return accVals.length > 0 ? Math.max(...accVals) : null
		})()
		: null
	$: bestF1 = iterations.length > 0 ? Math.max(...iterations.filter(i => i.best_f1 !== null && i.best_f1 !== undefined).map(i => i.best_f1!), 0) || null : null
	$: bestFpr = iterations.length > 0
		? (() =>
		{
			const fprVals = iterations.filter(i => i.best_fpr !== null && i.best_fpr !== undefined).map(i => i.best_fpr!)
			return fprVals.length > 0 ? Math.min(...fprVals) : null
		})()
		: null

	// Baseline values (first iteration)
	$: baselineCE = iterations.length > 0 ? iterations[0].best_ce : null
	$: baselineAcc = iterations.length > 0 ? iterations[0].best_accuracy : null
	$: baselineF1 = iterations.length > 0 ? iterations[0].best_f1 ?? null : null
	$: baselineFpr = iterations.length > 0 ? iterations[0].best_fpr ?? null : null

	// Improvement percentages
	$: ceImprovement = baselineCE !== null && bestCE !== Infinity && baselineCE > 0
		? ((baselineCE - bestCE) / baselineCE) * 100
		: null
	$: accImprovement = baselineAcc !== null && bestAcc !== null && baselineAcc > 0
		? ((bestAcc - baselineAcc) / baselineAcc) * 100
		: null
	$: f1Improvement = baselineF1 !== null && bestF1 !== null && baselineF1 > 0
		? ((bestF1 - baselineF1) / baselineF1) * 100
		: null
	$: fprImprovement = baselineFpr !== null && bestFpr !== null && baselineFpr > 0
		? ((baselineFpr - bestFpr) / baselineFpr) * 100
		: null

	// Max iterations from experiment config
	$: maxIterations = experiment?.max_iterations ?? null

	// Experiment type detection (architecture_type can be missing on older rows;
  // best_f1 on iterations is an IDS-only signal — never silently fall back to LM columns)
	$: isIDS = isIdsExperiment(experiment, iterations)
	// Flow param is the primary multiclass signal; metadata-shape fallback
  // covers rows loaded before (or without) the flow config.
	$: isMulticlass = isMulticlassFlow(flow)
		|| hasMulticlassMetadata(validationSummaries)
		|| hasMulticlassMetadata(flowValidationSummaries)
	$: isController = experiment?.architecture_type === 'controller'
	$: bestAttitudeDeg = (() =>
	{
		const vs = iterations.map((i) => i.mean_attitude_error_deg).filter((v): v is number => v != null)
		return vs.length ? Math.min(...vs) : null
	})()
	// Grid search: detect and auto-load results
	$: isGridSearch = experiment?.phase_type === 'grid_search'

	// Auto-load grid search genome evaluations when iterations arrive or update.
  // _lastGridIterCount is committed only on SUCCESS (so a failed load retries);
  // _gridRetryAfter throttles the retry so a persistent failure can't hot-loop.
	let _lastGridIterCount = 0
	let _gridRetryAfter = 0
	$: if (isGridSearch && iterations.length > 0 && !gridSearchLoading
		&& iterations.length !== _lastGridIterCount && Date.now() >= _gridRetryAfter)
	{
		loadGridSearchResults(iterations.length)
	}

	async function loadGridSearchResults(iterCount: number)
	{
		if (!iterations.length) return
		gridSearchLoading = true
		try
		{
			// Each per-config iteration has one genome evaluation with (neurons, bits)
  		// The final iteration (N+1) has the expanded population — skip it
			const configIters = iterations.filter(i => i.candidates_total && i.candidates_total > 1)
			const perConfigIters = configIters.length > 0
				? iterations.filter(i => i.iteration_num <= (configIters[0]?.candidates_total ?? iterations.length))
				: iterations

			const results: GridSearchRow[] = []

			// Fetch genome evaluations for each per-config iteration
			const fetches = perConfigIters.map(iter =>
				fetch(`/api/iterations/${iter.id}/genomes`).then(r => r.ok ? r.json() : [])
			)
			const allEvals = await Promise.all(fetches)

			for (let i = 0; i < perConfigIters.length; i++)
			{
				const evals: GenomeEvaluation[] = allEvals[i]
				const iter = perConfigIters[i]
				if (evals.length === 0) continue
				const ev = evals[0]
				let neurons = 0, bits = 0
				if (ev.tiers_json)
				{
					try
					{
						const tiers: GenomeTier[] = JSON.parse(ev.tiers_json)
						if (tiers.length > 0)
						{
							neurons = tiers[0].neurons; bits = tiers[0].bits
						}
					}
					catch
					{}
				}
				results.push({
					rank: 0, neurons, bits, ce: ev.ce, accuracy: ev.accuracy,
					fitness: ev.fitness_score, count: 1, elapsed: iter.elapsed_secs ?? 0,
					f1_macro: ev.f1_macro ?? iter.best_f1 ?? null, fpr: ev.fpr ?? iter.best_fpr ?? null,
				})
			}

			// Sort by fitness (lower = better), fall back to CE if fitness unavailable
			results.sort((a, b) =>
			{
				if (a.fitness != null && b.fitness != null) return a.fitness - b.fitness
				if (a.fitness != null) return -1
				if (b.fitness != null) return 1
				return a.ce - b.ce
			})
			gridSearchResults = results.map((r, i) => ({ ...r, rank: i + 1 }))

			// Load expanded population from seed iterations or final summary
  		// Three iteration phases: per-config (1..N), seed (N+1..N+K), final summary (N+K+1)
			const totalConfigs = configIters.length > 0 ? configIters[0].candidates_total ?? 0 : 0
			const afterConfigIters = iterations.filter(i => i.iteration_num > totalConfigs)
			if (afterConfigIters.length > 0)
			{
				// Find the max iteration_num to identify the final summary
				const maxIterNum = Math.max(...afterConfigIters.map(i => i.iteration_num))
				const maxIter = afterConfigIters.find(i => i.iteration_num === maxIterNum)!

				// Check if this is the final summary (has multiple genome evaluations)
  			// vs a seed iteration (has exactly 1). Fetch it to check.
				const maxRes = await fetch(`/api/iterations/${maxIter.id}/genomes`)
				const maxEvals: GenomeEvaluation[] = maxRes.ok ? await maxRes.json() : []

				if (maxEvals.length > 1)
				{
					// Final summary iteration — show full sorted population
					const expanded: typeof expandedPopulation = []
					for (const ev of maxEvals)
					{
						let neurons = 0, bits = 0
						if (ev.tiers_json)
						{
							try
							{
								const tiers: GenomeTier[] = JSON.parse(ev.tiers_json)
								if (tiers.length > 0)
								{
									neurons = tiers[0].neurons; bits = tiers[0].bits
								}
							}
							catch
							{}
						}
						expanded.push({
							rank: ev.position + 1, neurons, bits,
							ce: ev.ce, accuracy: ev.accuracy, fitness: ev.fitness_score,
							f1_macro: ev.f1_macro, fpr: ev.fpr,
						})
					}
					expandedPopulation = expanded
					seedEvalComplete = true
				}
				else
				{
					// Seed evaluation still in progress — gather individual seed genomes
					const seedIters = afterConfigIters.sort((a, b) => a.iteration_num - b.iteration_num)
					const seedFetches = seedIters.map(si =>
						fetch(`/api/iterations/${si.id}/genomes`).then(r => r.ok ? r.json() : [])
					)
					const seedEvals = await Promise.all(seedFetches)
					const expanded: typeof expandedPopulation = []
					for (let s = 0; s < seedIters.length; s++)
					{
						const evals: GenomeEvaluation[] = seedEvals[s]
						if (evals.length === 0) continue
						const ev = evals[0]
						let neurons = 0, bits = 0
						if (ev.tiers_json)
						{
							try
							{
								const tiers: GenomeTier[] = JSON.parse(ev.tiers_json)
								if (tiers.length > 0)
								{
									neurons = tiers[0].neurons; bits = tiers[0].bits
								}
							}
							catch
							{}
						}
						expanded.push({
							rank: s + 1, neurons, bits,
							ce: ev.ce, accuracy: ev.accuracy, fitness: ev.fitness_score,
							f1_macro: ev.f1_macro, fpr: ev.fpr,
						})
					}
					// Sort by fitness (lower = better), fall back to CE
					expanded.sort((a, b) =>
					{
						if (a.fitness != null && b.fitness != null) return a.fitness - b.fitness
						if (a.fitness != null) return -1
						if (b.fitness != null) return 1
						return a.ce - b.ce
					})
					expanded.forEach((g, i) => g.rank = i + 1)
					expandedPopulation = expanded
					seedEvalComplete = false
				}
			}
			else
			{
				expandedPopulation = []
				seedEvalComplete = false
			}

			// Success — commit the count so this iteration snapshot isn't re-fetched
			_lastGridIterCount = iterCount
			_gridRetryAfter = 0
		}
		catch (e)
		{
			console.error('Failed to load grid search results:', e)
			_gridRetryAfter = Date.now() + 5000 // retry on a later poll, not immediately
		}
		finally
		{
			gridSearchLoading = false
		}
	}

	// Average seconds per iteration
	$: avgSecsPerIter = iterations.length > 0
		? iterations.reduce((sum, i) => sum + (i.elapsed_secs ?? 0), 0) / iterations.length
		: null

	// Get tier_stats from the final checkpoint's genome_stats
  // If no checkpoint with tier_stats exists, fall back to parsing tier_config string
	$: finalCheckpoint = checkpoints.find(c => c.checkpoint_type === 'experiment_end' && c.genome_stats?.tier_stats)

	$: tierStats = finalCheckpoint?.genome_stats?.tier_stats ?? null

	// Bitwise cluster stats from checkpoint genome_stats (when cluster_type is 'bitwise')
	$: bitwiseClusterStats = (() =>
	{
		const cp = checkpoints.find(c => c.genome_stats?.cluster_stats)
		return cp?.genome_stats?.cluster_stats ?? null
	})()

	// Parse tier_config string for the optimize flag (not in computed tier_stats)
  // Format: "100,15,20;400,10,12;rest,5,8" or "100,15,20,true;400,10,12,false;rest,5,8,false"
	$: tierConfigOptimize = (() =>
	{
		if (!experiment?.tier_config) return []
		try
		{
			return experiment.tier_config.split(';').map(tierStr =>
			{
				const parts = tierStr.trim().split(',')
				// 4th part is optional optimize flag (defaults to true for backward compat)
				return parts.length >= 4 ? parts[3].trim().toLowerCase() === 'true' : true
			})
		}
		catch
		{
			return []
		}
	})()

	// Fallback: parse tier_config when no computed tier_stats available
	interface ParsedTier {
		clusters: string  // number or "rest"
		neurons: number
		bits: number
		optimize: boolean
	}
	// Latest gating run (most recent by created_at, which is already sorted DESC from API)
	$: latestGatingRun = gatingRuns.length > 0 ? gatingRuns[0] : null
	$: hasActiveGating = latestGatingRun && (latestGatingRun.status === 'pending' || latestGatingRun.status === 'running')
	$: hasCompletedGating = latestGatingRun && latestGatingRun.status === 'completed' && latestGatingRun.results

	$: parsedTiers = (() =>
	{
		if (!experiment?.tier_config) return []
		try
		{
			return experiment.tier_config.split(';').map(tierStr =>
			{
				const parts = tierStr.trim().split(',')
				if (parts.length < 3) return null
				const clusters = parts[0].trim()
				const neurons = parseInt(parts[1].trim())
				const bits = parseInt(parts[2].trim())
				const optimize = parts.length >= 4 ? parts[3].trim().toLowerCase() === 'true' : true
				return { clusters, neurons, bits, optimize }
			}).filter((t): t is ParsedTier => t !== null)
		}
		catch
		{
			return []
		}
	})()

	$: cumulativeValidationProgression = (() =>
	{
		if (!experiment || flowValidationSummaries.length === 0 || flowExperiments.length === 0)
		{
			// Fall back to current experiment's validations if no flow context
			if (!experiment || validationSummaries.length === 0) return []

			const points: ValidationProgressionPoint[] = []
			const initSummaries = validationSummaries.filter(s => s.validation_point === 'init')
			const finalSummaries = validationSummaries.filter(s => s.validation_point === 'final')

			if (initSummaries.length > 0)
			{
				points.push({
					label: 'Init',
					expId: experiment.id,
					sequenceOrder: experiment.sequence_order ?? 0,
					validationPoint: 'init',
					summaries: initSummaries.map(s => ({ genomeType: s.genome_type, ce: s.ce, accuracy: s.accuracy, f1_macro: s.f1_macro, fpr: s.fpr, threshold_metadata: s.threshold_metadata }))
				})
			}
			if (finalSummaries.length > 0)
			{
				points.push({
					label: experiment.name.replace(/^Phase \d+[ab]: /, ''),
					expId: experiment.id,
					sequenceOrder: experiment.sequence_order ?? 0,
					validationPoint: 'final',
					summaries: finalSummaries.map(s => ({ genomeType: s.genome_type, ce: s.ce, accuracy: s.accuracy, f1_macro: s.f1_macro, fpr: s.fpr, threshold_metadata: s.threshold_metadata }))
				})
			}
			return points
		}

		// Build cumulative progression from flow validations
		const currentSeqOrder = experiment.sequence_order ?? 0

		// Create a map of experiment_id -> experiment info
		const expMap = new Map(flowExperiments.map(e => [e.id, e]))

		// Filter validations to only include experiments up to and including current
		const relevantValidations = flowValidationSummaries.filter(v =>
		{
			const exp = expMap.get(v.experiment_id)
			if (!exp) return false
			return (exp.sequence_order ?? 0) <= currentSeqOrder
		})

		// Group by (experiment_id, validation_point)
		const grouped = new Map<string, ValidationSummary[]>()
		for (const v of relevantValidations)
		{
			const key = `${v.experiment_id}-${v.validation_point}`
			if (!grouped.has(key)) grouped.set(key, [])
			grouped.get(key)!.push(v)
		}

		// Convert to progression points
		const points: ValidationProgressionPoint[] = []
		for (const [key, validations] of grouped)
		{
			const [expIdStr, point] = key.split('-')
			const expId = parseInt(expIdStr)
			const exp = expMap.get(expId)
			if (!exp) continue

			const seqOrder = exp.sequence_order ?? 0

			// Label: "Init" for first init, otherwise "Phase 1a", "Phase 1b", etc.
			let label: string
			if (point === 'init' && seqOrder === 0)
			{
				label = 'Init'
			}
			else if (point === 'init')
			{
				// Skip non-first init points (they're the same as previous final)
				continue
			}
			else
			{
				label = exp.name.replace(/^Phase \d+[ab]: /, '')
			}

			points.push({
				label,
				expId,
				sequenceOrder: seqOrder,
				validationPoint: point as 'init' | 'final',
				summaries: validations.map(v => ({ genomeType: v.genome_type, ce: v.ce, accuracy: v.accuracy, f1_macro: v.f1_macro, fpr: v.fpr, threshold_metadata: v.threshold_metadata }))
			})
		}

		// Sort by sequence order, then init before final
		return points.sort((a, b) =>
		{
			if (a.sequenceOrder !== b.sequenceOrder) return a.sequenceOrder - b.sequenceOrder
			return a.validationPoint === 'init' ? -1 : 1
		})
	})()
</script>

<div class="container">
	{#if loading}
		<div class="loading">Loading experiment...</div>
	{:else if error}
		<div class="error">{error}</div>
	{:else if experiment}
		<!-- Header -->
		<ExperimentHeader
			{experiment}
			{flow}
			{latestGatingRun}
			hasActiveGating={!!hasActiveGating}
			hasCompletedGating={!!hasCompletedGating}
			{gatingLoading}
			on:runGating={runGating}
		/>

		<!-- Flow Progress Bar -->
		{#if flowSteps.length > 0}
			<FlowProgressBar {flowSteps} currentExperimentId={experiment.id} />
		{/if}

		<!-- Info Cards -->
		<ExperimentInfoCards
			{experiment}
			{flow}
			{isIDS}
			{isController}
			{isGridSearch}
			{bestCE}
			{bestAcc}
			{bestF1}
			{bestFpr}
			{bestAttitudeDeg}
			{ceImprovement}
			{accImprovement}
			{f1Improvement}
			{fprImprovement}
			iterationsCount={iterations.length}
			gridConfigsCount={gridSearchResults.length}
			{maxIterations}
			{avgSecsPerIter}
		/>

		{#if experiment.status === 'running' && experiment.status_message}
			<div class="status-message">{experiment.status_message}</div>
		{/if}

		<!-- Cumulative Validation Progression -->
		{#if cumulativeValidationProgression.length > 0}
			<ValidationProgressionTable
				points={cumulativeValidationProgression}
				{isIDS}
				{isMulticlass}
				currentExperimentId={experiment.id}
				bind:perClassPointChoice
				bind:perClassThresholdChoice
			/>
		{/if}

		<!-- Gating Results -->
		<GatingResultsPanel
			{latestGatingRun}
			gatingRunsCount={gatingRuns.length}
			hasCompletedGating={!!hasCompletedGating}
		/>

		<!-- Tier Stats (Best Genome) / fallback Tier Configuration -->
		<TierStatsPanel {tierStats} {tierConfigOptimize} {parsedTiers} />

		<!-- Bitwise Cluster Stats (per-cluster view for bitwise experiments) -->
		{#if experiment.architecture_type === 'bitwise' || experiment.cluster_type === 'bitwise' || bitwiseClusterStats}
			<BitwiseClusterStats clusterStats={bitwiseClusterStats} />
		{/if}

		<!-- Chart (hidden for grid search — results shown in Grid Search Results section) -->
		{#if chartData.length > 0 && !isGridSearch}
			<ProgressChart {isIDS} {chartData} />
		{/if}

		<!-- Grid Search Results -->
		{#if isGridSearch}
			<GridSearchResults
				{iterations}
				{gridSearchResults}
				{expandedPopulation}
				{seedEvalComplete}
				{gridSearchLoading}
				{isIDS}
				status={experiment.status}
			/>
		{/if}

		<!-- Live Generation Progress -->
		{#if liveProgress}
			<LiveProgressBar {liveProgress} {isIDS} />
		{/if}

		<!-- Iterations Table (hidden for grid search) -->
		{#if !isGridSearch}
			<IterationsTable
				{displayIterations}
				{maxIterations}
				{isIDS}
				{bestCE}
				{bestAcc}
				{bestF1}
				{bestFpr}
				on:openDetails={(e) => openIterationDetails(e.detail)}
			/>
		{/if}
	{/if}
</div>

<!-- Iteration Details Modal -->
{#if showIterationModal && selectedIteration}
	<IterationDetailModal
		{selectedIteration}
		{genomeEvaluations}
		{loadingGenomes}
		{isIDS}
		on:close={closeIterationModal}
	/>
{/if}

<style>
  .container {
    max-width: 1848px;  /* +10% to fit per-class table on the validation panel */
    margin: 0 auto;
    padding: 1rem;
  }

  .status-message {
    font-size: 1rem;
    color: var(--text-secondary);
    background: var(--bg-secondary);
    padding: 0.5rem 1rem;
    border-radius: 6px;
    margin-bottom: 1rem;
    font-family: var(--font-mono, monospace);
  }

  .loading, .error {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--text-secondary);
  }

  .error {
    color: var(--accent-red);
  }
</style>
