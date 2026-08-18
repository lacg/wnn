<script lang="ts">
	import { goto } from '$app/navigation'
	import TierConfigEditor from '$lib/components/TierConfigEditor.svelte'
	import FlowBasicsRow from '$lib/components/flowNew/FlowBasicsRow.svelte'
	import MultiStageConfig from '$lib/components/flowNew/MultiStageConfig.svelte'
	import SearchParamsSection from '$lib/components/flowNew/SearchParamsSection.svelte'
	import SeedPopulationSection from '$lib/components/flowNew/SeedPopulationSection.svelte'
	import PhaseListEditor from '$lib/components/flowNew/PhaseListEditor.svelte'
	import IDSConfigSection from '$lib/components/flowNew/IDSConfigSection.svelte'
	import BitwiseConfigSection from '$lib/components/flowNew/BitwiseConfigSection.svelte'
	import ControllerConfigSection from '$lib/components/flowNew/ControllerConfigSection.svelte'
	import type { StageConfig, PhaseSpec } from '$lib/flowTemplates'
	import { GRID_DEFAULTS, defaultStageConfig, generatePhasesForStage, generatePhases } from '$lib/flowTemplates'

	let name = ''
	let description = ''
	let template = 'bitwise-7-phase'
	let phaseOrder = 'neurons_first'

	// Bitwise-specific config (single-stage only)
	let bitwiseNumClusters = 16
	let bitwiseMinBits = 10
	let bitwiseMaxBits = 24
	let bitwiseMinNeurons = 10
	let bitwiseMaxNeurons = 300
	let bitwiseMemoryMode = 'QUAD_WEIGHTED'
	let bitwiseNeuronSampleRate = 0.25

	// IDS-specific config
	let idsDataset = 'unsw-nb15'
	let idsClassification = 'binary'
	let idsSingleCluster = true
	let idsNBits = 8
	let idsValFraction = 0.25
	let idsKFolds = 5
	let idsKFoldPerGen = 1
	let idsFitnessWeightF1 = 0.0
	let idsSplit = 'standard'
	let idsFeatureSelection = 'all'
	let idsRestBits = 8
	let idsMinBits = 4
	let idsMaxBits = 16
	let idsMinNeurons = 5
	let idsMaxNeurons = 500
	let idsMaxBitDelta = 0
	let idsNeuronSampleRate = 0.25
	let idsBalanceClasses = true
	// Single-genome eval mode: skip GA Neurons, force min=max for both bits/neurons.
  // Useful for ad-hoc evaluations like the 46M Pareto sweep.
	let idsSingleGenome = false
	let idsSingleNeurons = 200
	let idsSingleBits = 4

	// Controller-specific config (architecture_type='controller', drone attitude sim)
	let ctrlNumMotors = 4
	let ctrlLevelsPerMotor = 16
	let ctrlStateNeurons = 4
	let ctrlStateBits = 24
	let ctrlOutputBits = 24
	let ctrlInputWindowK = 4
	let ctrlBitsPerFeature = 8
	let ctrlEvalEpisodes = 20
	let ctrlSteps = 1500
	let ctrlTiltDeg = 15.0
	let ctrlDeltaControl = false
	let ctrlSeed = 0

	// Multi-stage config
	let numStages = 1
	let stageMode = 'input_concat'
	let selectedStage = 0

	let stageConfigs: StageConfig[] = [defaultStageConfig(), defaultStageConfig()]

	// Track previous stage for save/load on switch
	let _prevStage = 0

	function saveSearchParamsToStage(stage: number)
	{
		if (stage < 0 || stage >= stageConfigs.length) return
		stageConfigs[stage] = {
			...stageConfigs[stage],
			gaGenerations, tsIterations, adaptationIterations,
			populationSize, neighborsPerIter, patience,
			fitnessPercentile, fitnessCalculator,
			fitnessWeightCe, fitnessWeightAcc,
			minAccuracyFloor, thresholdStart, thresholdStep,
		}
	}

	function loadSearchParamsFromStage(stage: number)
	{
		if (stage < 0 || stage >= stageConfigs.length) return
		const c = stageConfigs[stage]
		gaGenerations = c.gaGenerations
		tsIterations = c.tsIterations
		adaptationIterations = c.adaptationIterations
		populationSize = c.populationSize
		neighborsPerIter = c.neighborsPerIter
		patience = c.patience
		fitnessPercentile = c.fitnessPercentile
		fitnessCalculator = c.fitnessCalculator
		fitnessWeightCe = c.fitnessWeightCe
		fitnessWeightAcc = c.fitnessWeightAcc
		minAccuracyFloor = c.minAccuracyFloor
		thresholdStart = c.thresholdStart
		thresholdStep = c.thresholdStep
	}

	// Shared multi-stage architecture params
	let msMemoryMode = 'QUAD_WEIGHTED'
	let msNeuronSampleRate = 0.25
	let msTemplate = 'full'
	let invalidMode = false
	let topM = 5

	$: isMultiStage = numStages >= 2
	$: isBitwise = !isMultiStage && template.startsWith('bitwise-')
	$: isIDS = !isMultiStage && template.startsWith('ids-')
	$: isController = !isMultiStage && template.startsWith('controller-')

	// Resize stageConfigs when numStages changes
	$:
		{
			const prevLen = stageConfigs.length
			while (stageConfigs.length < numStages)
			{
				const newIdx = stageConfigs.length
				const cfg = defaultStageConfig()
				// Apply mode-specific grid defaults for the new stage
				const mode = stageMode === 'selector' && newIdx > 0 ? 'selector'
					: (cfg.clusterType === 'tiered' || cfg.clusterType === 'semantic') ? cfg.clusterType
					: cfg.clusterType === 'semantic_bitwise' ? 'semantic_bitwise' : 'bitwise'
				cfg.neuronsGrid = GRID_DEFAULTS[mode].neurons
				cfg.bitsGrid = GRID_DEFAULTS[mode].bits
				stageConfigs = [...stageConfigs, cfg]
			}
			if (stageConfigs.length > numStages)
			{
				stageConfigs = stageConfigs.slice(0, numStages)
			}
			if (selectedStage >= numStages)
			{
				selectedStage = Math.max(0, numStages - 1)
			}
		}

	// Save/load search params when switching stages
	$: if (isMultiStage && selectedStage !== _prevStage)
	{
		saveSearchParamsToStage(_prevStage)
		loadSearchParamsFromStage(selectedStage)
		_prevStage = selectedStage
	}

	let gaGenerations = 250
	let tsIterations = 250
	let adaptationIterations = 50
	let populationSize = 50
	let neighborsPerIter = 50
	let patience = 10
	// How often (in generations) the early-stop patience check runs. IDS default
  // 10; controller flows default 5 (set by the controller queue script).
	let checkInterval = 10
	// CPU cores this flow's RAYON pool may use; the scheduler also reads it to
  // budget concurrency (ids default ~10, controller ~3).
	let wnnNumThreads = 10
	let fitnessPercentile = 0.75
	let fitnessCalculator = 'harmonic_rank'
	let fitnessWeightCe = 1.0
	let fitnessWeightAcc = 1.0
	let minAccuracyFloor = 0
	let thresholdStart = 0
	let thresholdStep = 1
	let contextSize = 4
	let clusterCrossoverRatio = 0.8
	let poolShuffleRatio = 0.0
	let assortativeMatingRatio = 0.85
	let reweightRounds = 0
	let reweightMaxBoost = 4
	let tierConfig = '100,15,20,true;400,10,12,false;rest,5,8,false'

	// Leaderboard seeding
	let seedFromLeaderboard = false
	let seedLeaderboardCount = 150

	// Apply template defaults (only in single-stage mode)
	function applyTemplateDefaults(templateName: string)
	{
		if (isMultiStage) return

		if (templateName === 'quick-4-phase')
		{
			gaGenerations = 50
			tsIterations = 50
			populationSize = 50
			neighborsPerIter = 50
			patience = 2
			fitnessPercentile = 0.75
			fitnessCalculator = 'normalized_harmonic'
			fitnessWeightCe = 1.0
			fitnessWeightAcc = 1.0
			contextSize = 4
			tierConfig = '100,15,16,true;400,10,12,false;rest,5,8,false'
			phaseOrder = 'neurons_first'
		}
		else if (templateName === 'standard-6-phase')
		{
			gaGenerations = 250
			tsIterations = 250
			populationSize = 50
			neighborsPerIter = 50
			patience = 10
			fitnessPercentile = 0.75
			fitnessCalculator = 'normalized_harmonic'
			fitnessWeightCe = 1.0
			fitnessWeightAcc = 1.0
			contextSize = 4
			tierConfig = '100,15,20,true;400,10,12,false;rest,5,8,false'
		}
		else if (templateName === 'bitwise-7-phase')
		{
			gaGenerations = 250
			tsIterations = 250
			populationSize = 50
			neighborsPerIter = 50
			patience = 10
			fitnessPercentile = 0.75
			fitnessCalculator = 'harmonic_rank'
			fitnessWeightCe = 1.0
			fitnessWeightAcc = 1.0
			contextSize = 4
			bitwiseNumClusters = 16
			bitwiseMinBits = 10
			bitwiseMaxBits = 24
			bitwiseMinNeurons = 10
			bitwiseMaxNeurons = 300
			bitwiseMemoryMode = 'QUAD_WEIGHTED'
			bitwiseNeuronSampleRate = 0.25
		}
		else if (templateName === 'bitwise-10-phase')
		{
			gaGenerations = 250
			tsIterations = 250
			adaptationIterations = 50
			populationSize = 50
			neighborsPerIter = 50
			patience = 10
			fitnessPercentile = 0.75
			fitnessCalculator = 'harmonic_rank'
			fitnessWeightCe = 1.0
			fitnessWeightAcc = 1.0
			contextSize = 4
			bitwiseNumClusters = 16
			bitwiseMinBits = 10
			bitwiseMaxBits = 24
			bitwiseMinNeurons = 10
			bitwiseMaxNeurons = 300
			bitwiseMemoryMode = 'QUAD_WEIGHTED'
			bitwiseNeuronSampleRate = 0.25
		}
		else if (templateName === 'ids-binary-2-phase' || templateName === 'ids-binary-5-phase')
		{
			gaGenerations = 250
			tsIterations = 250
			adaptationIterations = 50
			populationSize = 150
			neighborsPerIter = 150
			patience = 5
			fitnessPercentile = 0.75
			fitnessCalculator = 'ids_recall'
			fitnessWeightCe = 0.3
			fitnessWeightAcc = 1.0
			idsClassification = 'binary'
			idsNBits = 8
			idsValFraction = 0.25
			idsKFolds = 5
			idsKFoldPerGen = 1
			idsFitnessWeightF1 = 0.0
			idsSplit = 'standard'
			idsFeatureSelection = 'all'
		}
		else if (templateName === 'ids-binary-7-phase')
		{
			gaGenerations = 250
			tsIterations = 250
			populationSize = 150
			neighborsPerIter = 150
			patience = 5
			fitnessPercentile = 0.75
			fitnessCalculator = 'ids_recall'
			fitnessWeightCe = 0.3
			fitnessWeightAcc = 1.0
			idsClassification = 'binary'
			idsNBits = 8
			idsValFraction = 0.25
			idsKFolds = 5
			idsKFoldPerGen = 1
			idsFitnessWeightF1 = 0.0
			idsSplit = 'standard'
			idsFeatureSelection = 'all'
		}
		else if (templateName === 'ids-multi-7-phase')
		{
			gaGenerations = 250
			tsIterations = 250
			populationSize = 150
			neighborsPerIter = 150
			patience = 5
			fitnessPercentile = 0.75
			fitnessCalculator = 'ids_recall'
			fitnessWeightCe = 0.3
			fitnessWeightAcc = 1.0
			idsClassification = 'multi_tiered'
			idsNBits = 8
			idsValFraction = 0.25
			idsKFolds = 5
			idsKFoldPerGen = 1
			idsFitnessWeightF1 = 0.0
			idsSplit = 'standard'
			idsFeatureSelection = 'all'
		}
		else if (templateName === 'ids-binary-10-phase')
		{
			gaGenerations = 250
			tsIterations = 250
			adaptationIterations = 50
			populationSize = 150
			neighborsPerIter = 150
			patience = 5
			fitnessPercentile = 0.75
			fitnessCalculator = 'ids_recall'
			fitnessWeightCe = 0.3
			fitnessWeightAcc = 1.0
			idsClassification = 'binary'
			idsNBits = 8
			idsValFraction = 0.25
			idsKFolds = 5
			idsKFoldPerGen = 1
			idsFitnessWeightF1 = 0.0
			idsSplit = 'standard'
			idsFeatureSelection = 'all'
		}
		else if (templateName === 'ids-multi-10-phase')
		{
			gaGenerations = 250
			tsIterations = 250
			adaptationIterations = 50
			populationSize = 150
			neighborsPerIter = 150
			patience = 5
			fitnessPercentile = 0.75
			fitnessCalculator = 'ids_recall'
			fitnessWeightCe = 0.3
			fitnessWeightAcc = 1.0
			idsClassification = 'multi_tiered'
			idsNBits = 8
			idsValFraction = 0.25
			idsKFolds = 5
			idsKFoldPerGen = 1
			idsFitnessWeightF1 = 0.0
			idsSplit = 'standard'
			idsFeatureSelection = 'all'
		}
		else if (templateName === 'controller-ga-memory')
		{
			// Paradigm-B neuroevolution of QSR cells (no training). Matches run_ga_memory.py.
			gaGenerations = 3000
			populationSize = 150
			patience = 3000            // effectively off — the held-out comparison ran patience-off
			fitnessCalculator = 'ce'   // controller ce = −reward, so 'ce' ranking maximises reward
			fitnessWeightCe = 1.0
			fitnessWeightAcc = 1.0
		}
		else if (templateName === 'controller-full-matrix')
		{
			// GA across {neurons, bits, connections, memory} — the Phase B matrix.
			gaGenerations = 500
			populationSize = 100
			patience = 50
			fitnessCalculator = 'ce'
			fitnessWeightCe = 1.0
			fitnessWeightAcc = 1.0
		}
	}

	$: applyTemplateDefaults(template)
	let seedCheckpointId: number | null = null

	let loading = false
	let error: string | null = null

	// --- Per-stage phase storage (multi-stage only) ---
	let perStagePhases: PhaseSpec[][] = []

	// Regenerate per-stage phases when template changes
	let _prevMsTemplate = msTemplate
	$: if (isMultiStage && msTemplate !== _prevMsTemplate)
	{
		_prevMsTemplate = msTemplate
		perStagePhases = Array.from({ length: numStages }, (_, i) =>
			generatePhasesForStage(`S${i}`, msTemplate)
		)
	}

	// Resize per-stage phases when numStages changes
	$: if (isMultiStage && perStagePhases.length !== numStages)
	{
		const updated = [...perStagePhases]
		while (updated.length < numStages)
		{
			updated.push(generatePhasesForStage(`S${updated.length}`, msTemplate))
		}
		if (updated.length > numStages)
		{
			updated.length = numStages
		}
		perStagePhases = updated
	}

	// Single-stage phases from template
	let singleStagePhases: PhaseSpec[] = []
	$: if (!isMultiStage)
	{
		singleStagePhases = generatePhases(template, phaseOrder)
	}

	// What to display in the Phases panel
	$: displayPhases = isMultiStage
		? (perStagePhases[selectedStage] ?? [])
		: singleStagePhases

	// All experiments flattened for submit
	$: allExperiments = isMultiStage
		? perStagePhases.flat()
		: singleStagePhases

	function handleAddPhase(newPhase: PhaseSpec)
	{
		if (isMultiStage)
		{
			newPhase.name = `S${selectedStage}: ${newPhase.name}`
			perStagePhases[selectedStage] = [...perStagePhases[selectedStage], newPhase]
			perStagePhases = perStagePhases
		}
		else
		{
			singleStagePhases = [...singleStagePhases, newPhase]
		}
	}

	function removePhase(index: number)
	{
		if (isMultiStage)
		{
			perStagePhases[selectedStage] = perStagePhases[selectedStage].filter((_, i) => i !== index)
			perStagePhases = perStagePhases
		}
		else
		{
			singleStagePhases = singleStagePhases.filter((_, i) => i !== index)
		}
	}

	function movePhase(index: number, direction: -1 | 1)
	{
		const arr = isMultiStage ? perStagePhases[selectedStage] : singleStagePhases
		const newIndex = index + direction
		if (newIndex < 0 || newIndex >= arr.length) return
		const copy = [...arr];
		[copy[index], copy[newIndex]] = [copy[newIndex], copy[index]]
		if (isMultiStage)
		{
			perStagePhases[selectedStage] = copy
			perStagePhases = perStagePhases
		}
		else
		{
			singleStagePhases = copy
		}
	}

	async function handleSubmit()
	{
		if (!name.trim())
		{
			error = 'Name is required'
			return
		}
		if (allExperiments.length === 0)
		{
			// Rule 2: a flow with 0 experiments is marked completed instantly by
  		// the worker, doing zero work. The API rejects it too (400) since
  		// 12/06 — this guard just gives a friendlier message.
			error = 'Flow has no experiments — add at least one phase (the worker would complete an empty flow instantly, doing nothing).'
			return
		}

		loading = true
		error = null

		try
		{
			// Save current stage's search params before submit
			if (isMultiStage)
			{
				saveSearchParamsToStage(selectedStage)
			}

			const adaptationTypes = new Set(['neurogenesis', 'synaptogenesis', 'axonogenesis'])

			// Helper to get search params for a given experiment
			function getSearchParams(exp: PhaseSpec, stageIdx: number)
			{
				const cfg = isMultiStage ? stageConfigs[stageIdx] : null
				const gens = cfg ? cfg.gaGenerations : gaGenerations
				const tsIts = cfg ? cfg.tsIterations : tsIterations
				const adaptIts = cfg ? cfg.adaptationIterations : adaptationIterations
				const pop = cfg ? cfg.populationSize : populationSize
				const neighbors = cfg ? cfg.neighborsPerIter : neighborsPerIter

				const isAdaptation = adaptationTypes.has(exp.phase_type ?? '')
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
				}
			}

			// Enrich experiments with per-stage search params
			let enrichedExperiments
			if (isMultiStage)
			{
				enrichedExperiments = perStagePhases.flatMap((phases, stageIdx) =>
					phases.map((exp) => ({ ...exp, params: getSearchParams(exp, stageIdx) }))
				)
			}
			else
			{
				let phasesToUse = singleStagePhases
				// In IDS single-genome mode, drop GA Neurons (and any other GA/TS refinement)
  			// and keep only the grid_search phase (which will evaluate the single point).
				if (isIDS && idsSingleGenome)
				{
					phasesToUse = singleStagePhases.filter((p) => p.phase_type === 'grid_search')
					if (phasesToUse.length === 0)
					{
						// Fallback: synthesize a grid_search phase if none was generated
						phasesToUse = [{
							name: 'Grid Search (1 point)',
							experiment_type: 'grid_search',
							optimize_bits: false,
							optimize_neurons: false,
							optimize_connections: false,
							phase_type: 'grid_search',
						} as PhaseSpec]
					}
				}
				enrichedExperiments = phasesToUse.map((exp) => ({
					...exp, params: getSearchParams(exp, 0),
				}))
			}

			const params: Record<string, unknown> = {
				phase_order: phaseOrder,
				ga_generations: gaGenerations,
				ts_iterations: tsIterations,
				adaptation_iterations: adaptationIterations,
				population_size: populationSize,
				neighbors_per_iter: neighborsPerIter,
				patience,
				check_interval: checkInterval,
				wnn_num_threads: wnnNumThreads,
				fitness_percentile: fitnessPercentile,
				fitness_calculator: fitnessCalculator,
				fitness_weight_ce: fitnessWeightCe,
				fitness_weight_acc: fitnessWeightAcc,
				min_accuracy_floor: minAccuracyFloor,
				threshold_start: thresholdStart,
				threshold_step: thresholdStep,
				context_size: contextSize,
				cluster_crossover_ratio: clusterCrossoverRatio,
				pool_shuffle_ratio: poolShuffleRatio,
				assortative_mating_ratio: assortativeMatingRatio,
			}

			if (seedFromLeaderboard)
			{
				params.seed_from_leaderboard = true
				params.seed_leaderboard_count = seedLeaderboardCount
			}

			if (isMultiStage)
			{
				params.architecture_type = 'multi_stage'
				params.num_stages = numStages
				params.stage_k = stageConfigs.slice(0, numStages).map(s => s.k)
				params.stage_cluster_type = stageConfigs.slice(0, numStages).map(s => s.clusterType)
				params.stage_context_size = stageConfigs.slice(0, numStages).map(s => s.contextSize)
				params.context_size = Math.max(...stageConfigs.slice(0, numStages).map(s => s.contextSize))
				params.stage_mode = stageMode
				// Per-stage bounds
				params.stage_min_bits = stageConfigs.slice(0, numStages).map(s => s.minBits)
				params.stage_max_bits = stageConfigs.slice(0, numStages).map(s => s.maxBits)
				params.stage_min_neurons = stageConfigs.slice(0, numStages).map(s => s.minNeurons)
				params.stage_max_neurons = stageConfigs.slice(0, numStages).map(s => s.maxNeurons)
				// Global fallbacks (from S0 values)
				params.min_bits = stageConfigs[0].minBits
				params.max_bits = stageConfigs[0].maxBits
				params.min_neurons = stageConfigs[0].minNeurons
				params.max_neurons = stageConfigs[0].maxNeurons
				params.memory_mode = msMemoryMode
				params.neuron_sample_rate = msNeuronSampleRate
				params.invalid_mode = invalidMode
				params.top_m = topM
				// Per-stage search params
				params.stage_ga_generations = stageConfigs.slice(0, numStages).map(s => s.gaGenerations)
				params.stage_ts_iterations = stageConfigs.slice(0, numStages).map(s => s.tsIterations)
				params.stage_adaptation_iterations = stageConfigs.slice(0, numStages).map(s => s.adaptationIterations)
				params.stage_population_size = stageConfigs.slice(0, numStages).map(s => s.populationSize)
				params.stage_neighbors_per_iter = stageConfigs.slice(0, numStages).map(s => s.neighborsPerIter)
				params.stage_patience = stageConfigs.slice(0, numStages).map(s => s.patience)
				params.stage_fitness_percentile = stageConfigs.slice(0, numStages).map(s => s.fitnessPercentile)
				params.stage_fitness_calculator = stageConfigs.slice(0, numStages).map(s => s.fitnessCalculator)
				params.stage_fitness_weight_ce = stageConfigs.slice(0, numStages).map(s => s.fitnessWeightCe)
				params.stage_fitness_weight_acc = stageConfigs.slice(0, numStages).map(s => s.fitnessWeightAcc)
				// Per-stage grids (comma-separated strings → arrays of numbers)
				params.stage_neurons_grid = stageConfigs.slice(0, numStages).map(s =>
					s.neuronsGrid.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v))
				)
				params.stage_bits_grid = stageConfigs.slice(0, numStages).map(s =>
					s.bitsGrid.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v))
				)
				params.ms_template = msTemplate
				if (reweightRounds > 0)
				{
					params.reweight_rounds = reweightRounds
					params.reweight_max_boost = reweightMaxBoost
				}
			}
			else if (isIDS)
			{
				params.architecture_type = 'ids'
				params.ids_dataset = idsDataset
				// Split combined classification into classification + ids_arch_type
				const isBitwiseIds = idsClassification === 'multi_bitwise'
				const classMap: Record<string, string> = {
					'binary': 'binary',
					'hierarchical': 'hierarchical',
					'multi_tiered': 'multi',
					'multi_bitwise': 'multi',
				}
				params.ids_classification = classMap[idsClassification] || 'binary'
				params.ids_arch_type = isBitwiseIds ? 'bitwise' : 'tiered'
				params.ids_n_bits = idsNBits
				params.ids_val_fraction = idsValFraction
				params.ids_num_parts = idsKFolds > 1 ? idsKFolds : 3
				params.ids_k_folds = idsKFolds
				if (idsKFoldPerGen > 1)
				{
					params.ids_kfold_per_gen = idsKFoldPerGen
				}
				params.ids_fitness_weight_f1 = idsFitnessWeightF1
				params.ids_split = idsSplit
				params.ids_feature_selection = idsFeatureSelection
				if (idsFeatureSelection === 'top20_split')
				{
					params.ids_rest_bits = idsRestBits
				}
				if (idsSingleGenome)
				{
					// Single-genome mode: collapse the grid to one (n, b) point
					params.min_bits = idsSingleBits
					params.max_bits = idsSingleBits
					params.min_neurons = idsSingleNeurons
					params.max_neurons = idsSingleNeurons
				}
				else
				{
					params.min_bits = idsMinBits
					params.max_bits = idsMaxBits
					params.min_neurons = idsMinNeurons
					params.max_neurons = idsMaxNeurons
					if (idsMaxBitDelta > 0) params.max_bit_delta = idsMaxBitDelta
				}
				params.neuron_sample_rate = idsNeuronSampleRate
				params.balance_classes = idsBalanceClasses
				params.ids_single_cluster = idsSingleCluster
			}
			else if (isController)
			{
				params.architecture_type = 'controller'
				params.controller_num_motors = ctrlNumMotors
				params.controller_levels_per_motor = ctrlLevelsPerMotor
				params.controller_state_neurons = ctrlStateNeurons
				params.controller_state_bits = ctrlStateBits
				params.controller_output_bits = ctrlOutputBits
				params.controller_input_window_k = ctrlInputWindowK
				params.controller_bits_per_feature = ctrlBitsPerFeature
				params.controller_eval_episodes = ctrlEvalEpisodes
				params.controller_steps = ctrlSteps
				params.controller_tilt_deg = ctrlTiltDeg
				params.controller_delta_control = ctrlDeltaControl
				params.seed = ctrlSeed
			}
			else if (isBitwise)
			{
				params.architecture_type = 'bitwise'
				params.num_clusters = bitwiseNumClusters
				params.min_bits = bitwiseMinBits
				params.max_bits = bitwiseMaxBits
				params.min_neurons = bitwiseMinNeurons
				params.max_neurons = bitwiseMaxNeurons
				params.memory_mode = bitwiseMemoryMode
				params.neuron_sample_rate = bitwiseNeuronSampleRate
			}
			else
			{
				params.tier_config = tierConfig || null
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
					seed_checkpoint_id: seedFromLeaderboard ? null : seedCheckpointId
				})
			})

			if (!response.ok)
			{
				const errorData = await response.json().catch(() => ({}))
				throw new Error(errorData.error || `Failed to create flow (${response.status})`)
			}

			const flow = await response.json()
			goto(`/flows/${flow.id}`)
		}
		catch (e)
		{
			error = e instanceof Error ? e.message : 'Unknown error'
		}
		finally
		{
			loading = false
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
		<FlowBasicsRow bind:name bind:description bind:numStages />

		<!-- Multi-Stage Configuration (full width, only when stages > 1) -->
		{#if isMultiStage}
			<MultiStageConfig
				{numStages}
				bind:selectedStage
				bind:stageMode
				bind:msTemplate
				bind:stageConfigs
				bind:msMemoryMode
				bind:msNeuronSampleRate
				bind:invalidMode
				bind:topM
			/>
		{/if}

		<!-- Main two-column layout -->
		<div class="form-columns">
			<!-- Left column: Search parameters + Seed -->
			<div class="left-column">
				<SearchParamsSection
					{isMultiStage}
					{isBitwise}
					{isIDS}
					bind:template
					bind:phaseOrder
					bind:gaGenerations
					bind:tsIterations
					bind:adaptationIterations
					bind:populationSize
					bind:neighborsPerIter
					bind:clusterCrossoverRatio
					bind:poolShuffleRatio
					bind:assortativeMatingRatio
					bind:patience
					bind:checkInterval
					bind:wnnNumThreads
					bind:contextSize
					bind:fitnessCalculator
					bind:fitnessPercentile
					bind:fitnessWeightCe
					bind:fitnessWeightAcc
					bind:minAccuracyFloor
					bind:thresholdStart
					bind:thresholdStep
					bind:reweightRounds
					bind:reweightMaxBoost
				/>

				<SeedPopulationSection
					bind:seedFromLeaderboard
					bind:seedLeaderboardCount
					bind:seedCheckpointId
				/>
			</div>

			<!-- Right column: Phases + Architecture config -->
			<div class="right-column">
				<PhaseListEditor
					{displayPhases}
					{isMultiStage}
					{selectedStage}
					{isBitwise}
					{isIDS}
					on:add={(e) => handleAddPhase(e.detail)}
					on:remove={(e) => removePhase(e.detail)}
					on:move={(e) => movePhase(e.detail.index, e.detail.direction)}
				/>

				<!-- Architecture config (single-stage only — multi-stage config is above) -->
				{#if !isMultiStage}
					{#if isIDS}
						<IDSConfigSection
							bind:idsDataset
							bind:idsClassification
							bind:idsSingleCluster
							bind:idsNBits
							bind:idsValFraction
							bind:idsKFolds
							bind:idsKFoldPerGen
							bind:idsSplit
							bind:idsFeatureSelection
							bind:idsRestBits
							bind:idsMinBits
							bind:idsMaxBits
							bind:idsMinNeurons
							bind:idsMaxNeurons
							bind:idsMaxBitDelta
							bind:idsNeuronSampleRate
							bind:idsBalanceClasses
							bind:idsSingleGenome
							bind:idsSingleNeurons
							bind:idsSingleBits
						/>
					{:else if isBitwise}
						<BitwiseConfigSection
							bind:bitwiseNumClusters
							bind:bitwiseMinBits
							bind:bitwiseMaxBits
							bind:bitwiseMinNeurons
							bind:bitwiseMaxNeurons
							bind:bitwiseMemoryMode
							bind:bitwiseNeuronSampleRate
						/>
					{:else if isController}
						<ControllerConfigSection
							bind:ctrlNumMotors
							bind:ctrlLevelsPerMotor
							bind:ctrlStateNeurons
							bind:ctrlStateBits
							bind:ctrlOutputBits
							bind:ctrlInputWindowK
							bind:ctrlBitsPerFeature
							bind:ctrlEvalEpisodes
							bind:ctrlSteps
							bind:ctrlTiltDeg
							bind:ctrlDeltaControl
							bind:ctrlSeed
						/>
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
			<button
				type="submit"
				class="btn btn-primary"
				disabled={loading || allExperiments.length === 0}
				title={allExperiments.length === 0 ? 'Add at least one phase — an empty flow does nothing' : ''}
			>
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
</style>
