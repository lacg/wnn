<script lang="ts">
	import { onDestroy } from 'svelte'
	import { page } from '$app/stores'
	import type { Flow, Experiment, Checkpoint, ValidationSummary, CombinedValidation } from '$lib/types'
	import { makeLatestGuard } from '$lib/api'
	import { isIdsFlow, isMulticlassFlow } from '$lib/ids'
	import { currentFlow, flows } from '$lib/stores'
	import FlowHeader from '$lib/components/flow/FlowHeader.svelte'
	import FlowInfoCards from '$lib/components/flow/FlowInfoCards.svelte'
	import ValidationProgressionChart from '$lib/components/flow/ValidationProgressionChart.svelte'
	import FlowConfigEditor from '$lib/components/flow/FlowConfigEditor.svelte'
	import FlowParamsPanel from '$lib/components/flow/FlowParamsPanel.svelte'
	import AddExperimentForm from '$lib/components/flow/AddExperimentForm.svelte'
	import ExperimentsTable from '$lib/components/flow/ExperimentsTable.svelte'
	import IDSResultsSection from '$lib/components/flow/IDSResultsSection.svelte'
	import CombinedResultsTable from '$lib/components/flow/CombinedResultsTable.svelte'
	import FinalResultsCard from '$lib/components/flow/FinalResultsCard.svelte'

	// Note: With normalized design, experiments come from the experiments table (via DB),
  // NOT from flow.config.experiments. The `experiments` array is fetched separately
  // via /api/flows/${flowId}/experiments and used directly.

	let flow: Flow | null = null

	// Subscribe to flow updates from WebSocket
	const unsubscribeCurrentFlow = currentFlow.subscribe((wsFlow) =>
	{
		if (wsFlow && flow && wsFlow.id === flow.id)
		{
			// Update local flow with WebSocket data
			flow = wsFlow
			// Refetch experiments when flow updates (experiment status may have changed)
			refreshExperiments()
		}
	})

	// Also subscribe to flows list updates (for FlowQueued, FlowStarted, etc.)
	const unsubscribeFlows = flows.subscribe((flowList) =>
	{
		if (flow)
		{
			const updated = flowList.find((f) => f.id === flow!.id)
			if (updated && updated.status !== flow.status)
			{
				flow = updated
				// Refetch experiments when flow status changes
				refreshExperiments()
			}
		}
	})

	// Drops stale responses: any newer load/refresh invalidates older in-flight
  // ones, so a slow poll can't roll status backwards (re-arming Pause/Stop) or
  // splice another flow's data under this flow's UI after navigation.
	const requestGuard = makeLatestGuard()

	// Refresh flow, experiments and validations without full page reload
	async function refreshExperiments()
	{
		// Never race a full load (e.g. mid-navigation)
		if (loading) return
		const token = requestGuard.begin()
		const idAtStart = flowId
		const isStale = () => !requestGuard.isCurrent(token) || flowId !== idAtStart
		try
		{
			const [flowRes, expsRes, validationsRes] = await Promise.all([
				fetch(`/api/flows/${idAtStart}`),
				fetch(`/api/flows/${idAtStart}/experiments`),
				fetch(`/api/flows/${idAtStart}/validations`)
			])
			const flowData = flowRes.ok ? await flowRes.json() : null
			const expsData = expsRes.ok ? await expsRes.json() : null
			const validationsData = validationsRes.ok ? await validationsRes.json() : null
			if (isStale()) return
			if (flowData) flow = flowData
			if (expsRes.ok) experiments = Array.isArray(expsData) ? expsData : []
			if (validationsRes.ok) validationSummaries = Array.isArray(validationsData) ? validationsData : []
		}
		catch (e)
		{
			console.error('Failed to refresh experiments:', e)
		}
	}

	// Periodic refresh when flow is running
	let pollInterval: ReturnType<typeof setInterval> | null = null

	$:
		{
			// Start/stop polling based on flow status
			if (flow?.status === 'running' || flow?.status === 'queued')
			{
				if (!pollInterval)
				{
					pollInterval = setInterval(refreshExperiments, 3000) // Poll every 3 seconds
				}
			}
			else
			{
				if (pollInterval)
				{
					clearInterval(pollInterval)
					pollInterval = null
				}
			}
		}

	onDestroy(() =>
	{
		unsubscribeCurrentFlow()
		unsubscribeFlows()
		if (pollInterval)
		{
			clearInterval(pollInterval)
		}
	})
	let experiments: Experiment[] = []
	let checkpoints: Checkpoint[] = []
	let validationSummaries: ValidationSummary[] = []
	let combinedValidations: CombinedValidation[] = []
	let loading = true
	let error: string | null = null        // page-level (initial load) only
	let actionError: string | null = null  // per-action banner — must NOT replace the page
	let saving = false
	let editMode = false
	// Shared guard for queue/stop/pause/resume/restart — a double-click must not
  // send duplicate POSTs into the flow state machine.
	let actionInFlight = false

	let showAddPhase = false
	let newPhase = {
		name: '',
		experiment_type: 'ga' as string,
		optimize_bits: false,
		optimize_neurons: true,
		optimize_connections: false,
		// Lamarckian dimension (mirrors GA's optimize_* dimension picker)
		genesis_mode: 'neurogenesis' as string,
		// Lambda sweep fields
		s0_checkpoint_id: null as number | null,
		s1_checkpoint_id: null as number | null,
		lambda_values: '0.01,0.05,0.1,0.2,0.3,0.5,0.7,0.9',
		genome_type: 'best_ce' as string,
	}

	// Edit form state
	let editConfig = {
		patience: 10,
		ga_generations: 250,
		ts_iterations: 250,
		population_size: 50,
		neighbors_per_iter: 50,
		fitness_percentile: 0.75,
		fitness_calculator: 'normalized',
		fitness_weight_ce: 1.0,
		fitness_weight_acc: 1.0,
		min_accuracy_floor: 0,
		threshold_start: 0,
		threshold_step: 1,
		tier_config: '',
		phase_order: 'neurons_first',
		context_size: 4,
		cluster_crossover_ratio: 0.0,
		pool_shuffle_ratio: 0.0,
		assortative_mating_ratio: 0.85
	}

	// Flow rename state
	let editingName = false
	let editedName = ''

	// Duplicate state
	let duplicating = false

	$: flowId = $page.params.id
	$: isBitwise = flow?.config?.params?.architecture_type === 'bitwise'
	// IDS detection: flow config param OR any ids-typed experiment (older flows
  // may miss the param) — consistent with the experiment detail page.
	$: isIDS = isIdsFlow(flow, experiments)
	$: isMulticlass = isMulticlassFlow(flow)
	$: isController = flow?.config?.params?.architecture_type === 'controller'

	// Reactive display experiments - re-computed when flow or experiments change
	$: displayExperiments = getDisplayExperiments(flow, experiments)

	// Reload on flow-id change (in-page navigation). Runs once at init too, so
  // no onMount load is needed (it would double-load). Local state is reset so
  // the previous flow's data never shows under the new flow's URL.
	let _loadedFlowId: string | null = null
	$: if (flowId && flowId !== _loadedFlowId)
	{
		_loadedFlowId = flowId
		flow = null
		experiments = []
		checkpoints = []
		validationSummaries = []
		combinedValidations = []
		actionError = null
		editMode = false
		editingName = false
		loadFlow()
	}

	async function loadFlow()
	{
		const token = requestGuard.begin()
		const idAtStart = flowId
		const isStale = () => !requestGuard.isCurrent(token) || flowId !== idAtStart
		loading = true
		error = null  // a previous transient failure must not brick the page forever
		try
		{
			const [flowRes, expsRes, checkpointsRes, validationsRes, combinedRes] = await Promise.all([
				fetch(`/api/flows/${idAtStart}`),
				fetch(`/api/flows/${idAtStart}/experiments`),
				fetch(`/api/checkpoints`),
				fetch(`/api/flows/${idAtStart}/validations`),
				fetch(`/api/flows/${idAtStart}/combined-validations`)
			])

			if (!flowRes.ok) throw new Error('Flow not found')

			const flowData = await flowRes.json()
			// Ensure experiments is always an array (defensive against API returning {})
			const expsData = await expsRes.json()
			const checkpointsData = checkpointsRes.ok ? await checkpointsRes.json() : []
			const validationsData = validationsRes.ok ? await validationsRes.json() : []
			const combinedData = combinedRes.ok ? await combinedRes.json() : []

			// A newer load/refresh superseded this one — drop the stale response
			if (isStale()) return

			flow = flowData
			experiments = Array.isArray(expsData) ? expsData : []
			checkpoints = Array.isArray(checkpointsData) ? checkpointsData : []
			validationSummaries = Array.isArray(validationsData) ? validationsData : []
			// Combined validations (multi-stage end-to-end metrics)
			combinedValidations = Array.isArray(combinedData) ? combinedData : []

			// Populate edit form from config
			if (flow?.config?.params)
			{
				const p = flow.config.params
				editConfig.patience = p.patience ?? 10
				editConfig.ga_generations = p.ga_generations ?? 250
				editConfig.ts_iterations = p.ts_iterations ?? 250
				editConfig.population_size = p.population_size ?? 50
				editConfig.neighbors_per_iter = p.neighbors_per_iter ?? p.population_size ?? 50
				editConfig.fitness_percentile = p.fitness_percentile ?? 0.75
				editConfig.fitness_calculator = p.fitness_calculator ?? 'normalized'
				editConfig.fitness_weight_ce = p.fitness_weight_ce ?? 1.0
				editConfig.fitness_weight_acc = p.fitness_weight_acc ?? 1.0
				editConfig.min_accuracy_floor = p.min_accuracy_floor ?? 0
				editConfig.threshold_start = p.threshold_start ?? 0
				editConfig.threshold_step = p.threshold_step ?? 1
				editConfig.phase_order = p.phase_order ?? 'neurons_first'
				editConfig.context_size = p.context_size ?? 4
				editConfig.cluster_crossover_ratio = p.cluster_crossover_ratio ?? 0.0
				editConfig.pool_shuffle_ratio = p.pool_shuffle_ratio ?? 0.0
				editConfig.assortative_mating_ratio = p.assortative_mating_ratio ?? 0.85
				if (p.tier_config)
				{
					// Handle both string and array formats
					if (typeof p.tier_config === 'string')
					{
						editConfig.tier_config = p.tier_config
					}
					else
					{
						editConfig.tier_config = p.tier_config
							.map((t: (number|string|boolean)[]) =>
							{
								const base = `${t[0] ?? 'rest'},${t[1]},${t[2]}`
								return t.length > 3 ? `${base},${t[3]}` : base
							})
							.join(';')
					}
				}
			}
		}
		catch (e)
		{
			if (isStale()) return
			error = e instanceof Error ? e.message : 'Unknown error'
		}
		finally
		{
			if (!isStale()) loading = false
		}
	}

	async function saveChanges()
	{
		if (!flow) return
		saving = true

		try
		{
			// Parse tier_config string (supports 3 or 4 fields: clusters,neurons,bits[,optimize])
			let tier_config = null
			if (editConfig.tier_config.trim())
			{
				tier_config = editConfig.tier_config.split(';').map(tier =>
				{
					const parts = tier.trim().split(',').map(p => p.trim())
					const entry: (number | null | boolean)[] = [
						parts[0] === 'rest' ? null : parseInt(parts[0]),
						parseInt(parts[1]),
						parseInt(parts[2])
					]
					if (parts.length > 3)
					{
						entry.push(parts[3] === 'true')
					}
					return entry
				})
			}

			const updatedParams: Record<string, unknown> = {
				...flow.config.params,
				patience: editConfig.patience,
				ga_generations: editConfig.ga_generations,
				ts_iterations: editConfig.ts_iterations,
				population_size: editConfig.population_size,
				neighbors_per_iter: editConfig.neighbors_per_iter,
				fitness_percentile: editConfig.fitness_percentile,
				fitness_calculator: editConfig.fitness_calculator,
				fitness_weight_ce: editConfig.fitness_weight_ce,
				fitness_weight_acc: editConfig.fitness_weight_acc,
				min_accuracy_floor: editConfig.min_accuracy_floor,
				threshold_start: editConfig.threshold_start,
				threshold_step: editConfig.threshold_step,
				phase_order: editConfig.phase_order,
				context_size: editConfig.context_size,
				cluster_crossover_ratio: editConfig.cluster_crossover_ratio,
				pool_shuffle_ratio: editConfig.pool_shuffle_ratio,
				assortative_mating_ratio: editConfig.assortative_mating_ratio,
			}
			// Only include tier_config for tiered architectures
			if (!isBitwise)
			{
				updatedParams.tier_config = tier_config
			}
			const updatedConfig = {
				...flow.config,
				params: updatedParams
			}

			const res = await fetch(`/api/flows/${flowId}`, {
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ config: updatedConfig })
			})

			if (!res.ok) throw new Error('Failed to save')

			editMode = false
			await loadFlow()
		}
		catch (e)
		{
			actionError = e instanceof Error ? e.message : 'Failed to save'
		}
		finally
		{
			saving = false
		}
	}

	function cancelEdit()
	{
		editMode = false
		// Reload to reset form
		loadFlow()
	}

	// =========================================================================
  // Experiment Editing (for pending experiments)
  // Note: Experiments are now stored in the DB, not in flow.config
  // =========================================================================

	function canEditExperiment(index: number): boolean
	{
		if (!flow) return false
		// Can always edit if flow is pending, queued, or failed
		if (flow.status === 'pending' || flow.status === 'queued' || flow.status === 'failed') return true
		// Can't edit completed or cancelled flows
		if (flow.status !== 'running') return false

		// For running flows, check if this experiment has started
		const exp = experiments[index]
		if (!exp) return false
		return exp.status === 'pending'
	}

	async function deleteExperiment(index: number)
	{
		if (!flow || !canEditExperiment(index)) return
		const exp = displayExperiments[index]
		if (!exp) return
		if (!confirm(`Delete "${exp.name}"? This cannot be undone.`)) return

		saving = true
		try
		{
			const res = await fetch(`/api/experiments/${exp.id}`, { method: 'DELETE' })
			if (!res.ok)
			{
				const data = await res.json()
				throw new Error(data.error || 'Failed to delete experiment')
			}
			await loadFlow()
		}
		catch (e)
		{
			actionError = e instanceof Error ? e.message : 'Failed to delete'
		}
		finally
		{
			saving = false
		}
	}

	async function moveExperiment(index: number, direction: -1 | 1)
	{
		if (!flow) return
		const newIndex = index + direction
		if (newIndex < 0 || newIndex >= displayExperiments.length) return

		// Swap in local array to get new order
		const reordered = [...displayExperiments];
		[reordered[index], reordered[newIndex]] = [reordered[newIndex], reordered[index]]
		const experimentIds = reordered.map(e => e.id)

		saving = true
		try
		{
			const res = await fetch(`/api/flows/${flowId}/experiments/reorder`, {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ experiment_ids: experimentIds })
			})

			if (!res.ok)
			{
				const data = await res.json()
				throw new Error(data.error || 'Failed to reorder')
			}

			// Update from response
			const updated = await res.json()
			experiments = Array.isArray(updated) ? updated : []
		}
		catch (e)
		{
			actionError = e instanceof Error ? e.message : 'Failed to reorder'
		}
		finally
		{
			saving = false
		}
	}

	async function addExperiment()
	{
		if (!flow) return
		saving = true

		try
		{
			// Build params based on experiment type
			let expParams: Record<string, any> = {}
			if (newPhase.experiment_type === 'lambda_sweep')
			{
				expParams = {
					lambda_values: newPhase.lambda_values.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n)),
					s0_checkpoint_id: newPhase.s0_checkpoint_id,
					s1_checkpoint_id: newPhase.s1_checkpoint_id,
					genome_type: newPhase.genome_type,
				}
			}

			// Call the new dedicated endpoint to add experiment to the experiments table
			const res = await fetch(`/api/flows/${flowId}/experiments`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					experiment: {
						name: newPhase.name || `Experiment ${experiments.length + 1}`,
						// Lamarckian → send the chosen genesis_mode string; the worker maps
  					// it to the unified LAMARCKIAN strategy + genesis_mode.
						experiment_type: newPhase.experiment_type === 'lamarckian'
							? newPhase.genesis_mode : newPhase.experiment_type,
						optimize_bits: newPhase.optimize_bits,
						optimize_neurons: newPhase.optimize_neurons,
						optimize_connections: newPhase.optimize_connections,
						params: expParams
					}
				})
			})

			if (!res.ok)
			{
				const data = await res.json()
				throw new Error(data.error || 'Failed to add experiment')
			}

			showAddPhase = false
			newPhase = {
				name: '',
				experiment_type: 'ga',
				optimize_bits: false,
				optimize_neurons: true,
				optimize_connections: false,
				genesis_mode: 'neurogenesis',
				s0_checkpoint_id: null,
				s1_checkpoint_id: null,
				lambda_values: '0.01,0.05,0.1,0.2,0.3,0.5,0.7,0.9',
				genome_type: 'best_ce',
			}
			await loadFlow()
		}
		catch (e)
		{
			actionError = e instanceof Error ? e.message : 'Failed to add'
		}
		finally
		{
			saving = false
		}
	}

	// Update experiment's max_iterations
	async function updateExperimentIterations(expId: number, iterations: number)
	{
		if (iterations < 10 || iterations > 10000) return

		try
		{
			const response = await fetch(`/api/experiments/${expId}`, {
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ max_iterations: iterations })
			})

			if (!response.ok) throw new Error('Failed to update iterations')

			// Refresh experiments to show updated value
			await refreshExperiments()
		}
		catch (e)
		{
			actionError = e instanceof Error ? e.message : 'Failed to update iterations'
		}
	}

	// Save flow name
	async function saveFlowName()
	{
		if (!flow || !editedName.trim()) return
		saving = true

		try
		{
			const response = await fetch(`/api/flows/${flow.id}`, {
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ name: editedName.trim() })
			})

			if (!response.ok) throw new Error('Failed to rename flow')

			editingName = false
			editedName = ''
			await loadFlow()
		}
		catch (e)
		{
			actionError = e instanceof Error ? e.message : 'Failed to rename'
		}
		finally
		{
			saving = false
		}
	}

	// Duplicate flow
	async function duplicateFlow()
	{
		if (!flow) return
		const currentFlow = flow
		duplicating = true

		try
		{
			// Convert experiments to ExperimentSpec format for the new API.
  		// experiment_type must cover ALL phase families — the old ga/ts-only
  		// mapping silently duplicated adaptation/lambda phases as plain 'ts'.
			const expTypeFromPhase = (phaseType: string | null | undefined): string =>
			{
				if (!phaseType) return 'ga'
				if (phaseType === 'grid_search') return 'grid_search'
				if (phaseType === 'lambda_sweep') return 'lambda_sweep'
				if (['neurogenesis', 'synaptogenesis', 'axonogenesis'].includes(phaseType)) return phaseType
				if (phaseType.startsWith('ts')) return 'ts'
				return 'ga'
			}
			const experimentSpecs = experiments.map(exp =>
			{
				const expType = expTypeFromPhase(exp.phase_type)
				const isGridSearch = expType === 'grid_search'
				const isGa = expType === 'ga'
				const isTs = expType === 'ts'
				return {
					name: exp.name,
					experiment_type: expType,
					// Forward the original phase_type verbatim — create_flow uses it
  				// when present, so adaptation/lambda phases survive duplication.
					phase_type: exp.phase_type ?? undefined,
					optimize_bits: exp.phase_type?.includes('bits') ?? false,
					optimize_neurons: exp.phase_type?.includes('neurons') ?? false,
					optimize_connections: exp.phase_type?.includes('connections') ?? false,
					params: {
						// Only GA/TS carry generations/iterations defaults
						generations: isGa ? (currentFlow.config.params.ga_generations ?? 250) : undefined,
						iterations: isTs ? (currentFlow.config.params.ts_iterations ?? 250) : undefined,
						population_size: currentFlow.config.params.population_size ?? 50,
						neighbors_per_iter: currentFlow.config.params.neighbors_per_iter ?? 50,
						...(isGridSearch ? { phase_type: 'grid_search' } : {}),
					}
				}
			})

			const newName = `${currentFlow.name} (copy)`
			const response = await fetch('/api/flows', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					name: newName,
					description: currentFlow.description,
					config: currentFlow.config,  // Just params, no experiments
					experiments: experimentSpecs  // Experiments passed separately
				})
			})

			if (!response.ok) throw new Error('Failed to duplicate flow')

			const newFlow = await response.json()
			// Navigate to the new flow
			window.location.href = `/flows/${newFlow.id}`
		}
		catch (e)
		{
			actionError = e instanceof Error ? e.message : 'Failed to duplicate'
		}
		finally
		{
			duplicating = false
		}
	}

	// Delete flow state
	let deleting = false

	async function deleteFlow()
	{
		if (!flow) return
		if (!confirm(`Delete flow "${flow.name}"? This will delete all experiments, iterations, and checkpoints. This cannot be undone.`)) return

		deleting = true
		try
		{
			const response = await fetch(`/api/flows/${flow.id}`, {
				method: 'DELETE'
			})

			if (!response.ok)
			{
				const data = await response.json()
				throw new Error(data.error || 'Failed to delete flow')
			}

			// Navigate back to flows list
			window.location.href = '/flows'
		}
		catch (e)
		{
			actionError = e instanceof Error ? e.message : 'Failed to delete'
		}
		finally
		{
			deleting = false
		}
	}

	async function updateFitnessCalculator(value: string)
	{
		if (!flow) return
		saving = true

		try
		{
			const updatedConfig = {
				...flow.config,
				params: {
					...flow.config.params,
					fitness_calculator: value
				}
			}

			const response = await fetch(`/api/flows/${flow.id}`, {
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ config: updatedConfig })
			})

			if (!response.ok) throw new Error('Failed to update fitness calculator')
			await loadFlow()
		}
		catch (e)
		{
			actionError = e instanceof Error ? e.message : 'Failed to update'
		}
		finally
		{
			saving = false
		}
	}

	async function updateFitnessWeight(field: 'fitness_weight_ce' | 'fitness_weight_acc' | 'fitness_weight_f1' | 'fitness_weight_fpr' | 'min_accuracy_floor' | 'threshold_start' | 'threshold_step', value: number)
	{
		if (!flow) return
		saving = true

		try
		{
			const updatedConfig = {
				...flow.config,
				params: {
					...flow.config.params,
					[field]: value
				}
			}

			const response = await fetch(`/api/flows/${flow.id}`, {
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ config: updatedConfig })
			})

			if (!response.ok) throw new Error('Failed to update weight')
			await loadFlow()
		}
		catch (e)
		{
			actionError = e instanceof Error ? e.message : 'Failed to update'
		}
		finally
		{
			saving = false
		}
	}

	async function queueFlow()
	{
		if (!flow || actionInFlight) return
		actionInFlight = true
		try
		{
			const response = await fetch(`/api/flows/${flow.id}`, {
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ status: 'queued' })
			})
			if (response.ok)
			{
				await loadFlow()
			}
			else
			{
				actionError = 'Failed to queue flow'
			}
		}
		catch (e)
		{
			actionError = e instanceof Error ? e.message : 'Unknown error'
		}
		finally
		{
			actionInFlight = false
		}
	}

	async function stopFlow()
	{
		if (!flow || actionInFlight) return
		if (!confirm('Stop this flow? Current progress will be saved as a checkpoint.')) return

		actionInFlight = true
		try
		{
			const response = await fetch(`/api/flows/${flow.id}/stop`, {
				method: 'POST'
			})
			if (response.ok)
			{
				await loadFlow()
			}
			else
			{
				const data = await response.json()
				actionError = data.error || 'Failed to stop flow'
			}
		}
		catch (e)
		{
			actionError = e instanceof Error ? e.message : 'Unknown error'
		}
		finally
		{
			actionInFlight = false
		}
	}

	// Request the worker to pause this flow at the end of the current GA
  // generation. Sets flows.pause_requested=1; the worker polls the flag
  // between gens, writes the per-gen checkpoint, sets status='paused',
  // and moves on to the next queued flow.
	async function pauseFlow()
	{
		if (!flow || actionInFlight) return
		if (!confirm('Pause this flow at the end of the current generation? You can resume it later from where it left off.')) return
		actionInFlight = true
		try
		{
			const response = await fetch(`/api/flows/${flow.id}/pause`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: '{}'
			})
			if (response.ok)
			{
				await loadFlow()
			}
			else
			{
				const data = await response.json()
				actionError = data.error || 'Failed to pause flow'
			}
		}
		catch (e)
		{
			actionError = e instanceof Error ? e.message : 'Unknown error'
		}
		finally
		{
			actionInFlight = false
		}
	}

	// Resume a paused flow: clears pause_requested, flips status paused→queued.
  // Re-enters the queue normally (id-DESC, no front-jump).
	async function resumeFlow()
	{
		if (!flow || actionInFlight) return
		actionInFlight = true
		try
		{
			const response = await fetch(`/api/flows/${flow.id}/resume`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: '{}'
			})
			if (response.ok)
			{
				await loadFlow()
			}
			else
			{
				const data = await response.json()
				actionError = data.error || 'Failed to resume flow'
			}
		}
		catch (e)
		{
			actionError = e instanceof Error ? e.message : 'Unknown error'
		}
		finally
		{
			actionInFlight = false
		}
	}

	async function restartFlow(fromBeginning: boolean = false)
	{
		if (!flow || actionInFlight) return
		const msg = fromBeginning
			? 'Restart from the beginning? All progress will be lost.'
			: 'Restart from last checkpoint?'
		if (!confirm(msg)) return

		actionInFlight = true
		try
		{
			const response = await fetch(`/api/flows/${flow.id}/restart`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ from_beginning: fromBeginning })
			})
			if (response.ok)
			{
				await loadFlow()
			}
			else
			{
				const data = await response.json()
				actionError = data.error || 'Failed to restart flow'
			}
		}
		catch (e)
		{
			actionError = e instanceof Error ? e.message : 'Unknown error'
		}
		finally
		{
			actionInFlight = false
		}
	}

	async function restartFromExperiment(index: number)
	{
		if (!flow || actionInFlight) return
		const expName = experiments[index]?.name || `Experiment ${index + 1}`
		const msg = flow.status === 'running'
			? `Stop current experiment and restart from "${expName}"? The current experiment will be cancelled and earlier experiments will be skipped.`
			: `Restart flow from "${expName}"? Earlier experiments will be skipped.`
		if (!confirm(msg)) return

		actionInFlight = true
		try
		{
			const response = await fetch(`/api/flows/${flow.id}/restart`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ from_beginning: true, start_from_experiment: index })
			})
			if (response.ok)
			{
				await loadFlow()
			}
			else
			{
				const data = await response.json()
				actionError = data.error || 'Failed to restart flow'
			}
		}
		catch (e)
		{
			actionError = e instanceof Error ? e.message : 'Unknown error'
		}
		finally
		{
			actionInFlight = false
		}
	}

	// Helper to get display experiments - experiments from DB are the source of truth
  // Sorted by sequence_order
	function getDisplayExperiments(_f: Flow | null, exps: Experiment[]): Experiment[]
	{
		// Sort by sequence_order
		return [...exps].sort((a, b) => (a.sequence_order ?? 0) - (b.sequence_order ?? 0))
	}
</script>

<div class="container">
	{#if actionError}
		<div class="action-error" role="alert">
			<span>{actionError}</span>
			<button class="dismiss-btn" on:click={() => actionError = null}>✕</button>
		</div>
	{/if}

	{#if loading}
		<div class="loading">Loading flow...</div>
	{:else if error}
		<div class="error">{error}</div>
	{:else if flow}
		<FlowHeader
			{flow}
			{editMode}
			{saving}
			{duplicating}
			{deleting}
			{actionInFlight}
			bind:editingName
			bind:editedName
			on:saveName={saveFlowName}
			on:duplicate={duplicateFlow}
			on:queue={queueFlow}
			on:pause={pauseFlow}
			on:resume={resumeFlow}
			on:stop={stopFlow}
			on:restart={(e) => restartFlow(e.detail)}
			on:delete={deleteFlow}
			on:editConfig={() => editMode = true}
		/>

		{#if flow.description}
			<p class="description">{flow.description}</p>
		{/if}

		<FlowInfoCards {flow} {isIDS} {isController} />

		<!-- Validation Progression Chart -->
		<ValidationProgressionChart {validationSummaries} {displayExperiments} />

		{#if editMode}
			<FlowConfigEditor
				{editConfig}
				{saving}
				{isBitwise}
				on:save={saveChanges}
				on:cancel={cancelEdit}
			/>
		{:else if flow.config.params}
			<FlowParamsPanel
				{flow}
				{isBitwise}
				{saving}
				on:updateCalculator={(e) => updateFitnessCalculator(e.detail)}
				on:updateWeight={(e) => updateFitnessWeight(e.detail.field, e.detail.value)}
			/>
		{/if}

		<section class="section">
			<div class="section-header">
				<h2>Experiments ({experiments.length})</h2>
				{#if flow.status === 'pending' || flow.status === 'running' || flow.status === 'failed'}
					<button class="btn btn-sm btn-secondary" on:click={() => showAddPhase = true}>
						+ Add Experiment
					</button>
				{/if}
			</div>

			{#if showAddPhase}
				<AddExperimentForm
					{newPhase}
					{checkpoints}
					{saving}
					on:add={addExperiment}
					on:cancel={() => showAddPhase = false}
				/>
			{/if}

			<ExperimentsTable
				{flow}
				{displayExperiments}
				{experiments}
				{isIDS}
				{isMulticlass}
				{saving}
				{actionInFlight}
				on:move={(e) => moveExperiment(e.detail.index, e.detail.direction)}
				on:delete={(e) => deleteExperiment(e.detail)}
				on:updateIterations={(e) => updateExperimentIterations(e.detail.expId, e.detail.iterations)}
				on:stop={stopFlow}
				on:restartFrom={(e) => restartFromExperiment(e.detail)}
			/>
		</section>

		<!-- IDS Metrics (shown for IDS flows when experiments have extra_metrics) -->
		{#if isIDS}
			<IDSResultsSection {experiments} />
		{/if}

		<!-- Combined Results (for completed or running multi-stage flows) -->
		{#if combinedValidations.length > 0}
			<CombinedResultsTable {combinedValidations} />
			<!-- Final Results fallback (single-stage completed flows) -->
		{:else if flow.status === 'completed'}
			<FinalResultsCard {checkpoints} {experiments} {isIDS} />
		{/if}
	{/if}
</div>

<style>
  .action-error {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
    border: 1px solid var(--accent-red);
    border-radius: 8px;
    background: rgba(239, 68, 68, 0.12);
    color: var(--accent-red);
    font-size: 1rem;
  }

  .dismiss-btn {
    background: none;
    border: none;
    color: var(--accent-red);
    cursor: pointer;
    font-size: 1rem;
    line-height: 1;
  }

  .loading, .error {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--text-secondary);
  }

  .error {
    color: var(--accent-red);
  }

  .description {
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
  }

  .section {
    margin-bottom: 2rem;
  }

  h2 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
  }

  /* Section header with action button */
  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .section-header h2 {
    margin-bottom: 0;
  }

  .btn {
    padding: 0.5rem 1rem;
    border-radius: 4px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    border: none;
    transition: background 0.15s;
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn-secondary {
    background: rgba(51, 65, 85, 0.4);
    color: var(--text-primary);
    border: 1px solid var(--glass-border);
  }

  .btn-secondary:hover:not(:disabled) {
    background: var(--border);
  }

  .btn-sm {
    padding: 0.375rem 0.75rem;
    font-size: 1rem;
  }
</style>
