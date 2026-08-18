<script lang="ts">
	export let isMultiStage = false
	export let isBitwise = false
	export let isIDS = false

	// All search params are bound through to the page, which owns them
  // (applyTemplateDefaults / per-stage save-load / submit all live there).
	export let template = 'bitwise-7-phase'
	export let phaseOrder = 'neurons_first'
	export let gaGenerations = 250
	export let tsIterations = 250
	export let adaptationIterations = 50
	export let populationSize = 50
	export let neighborsPerIter = 50
	export let clusterCrossoverRatio = 0.8
	export let poolShuffleRatio = 0.0
	export let assortativeMatingRatio = 0.85
	export let patience = 10
	export let checkInterval = 10
	export let wnnNumThreads = 10
	export let contextSize = 4
	export let fitnessCalculator = 'harmonic_rank'
	export let fitnessPercentile = 0.75
	export let fitnessWeightCe = 1.0
	export let fitnessWeightAcc = 1.0
	export let minAccuracyFloor = 0
	export let thresholdStart = 0
	export let thresholdStep = 1
	export let reweightRounds = 0
	export let reweightMaxBoost = 4
</script>

<div class="form-section">
	<h2>Search Parameters</h2>

	{#if !isMultiStage}
		<div class="form-row">
			<div class="form-group">
				<label for="template">Template</label>
				<select id="template" bind:value={template}>
					<option value="quick-4-phase">Quick 4-Phase (Tiered)</option>
					<option value="standard-6-phase">Standard 6-Phase (Tiered)</option>
					<option value="bitwise-2-phase">Bitwise 2-Phase Explorer (Grid + GA Neurons)</option>
					<option value="bitwise-5-phase">Bitwise 5-Phase Deep (GA only + Synapt)</option>
					<option value="bitwise-7-phase">Bitwise 7-Phase</option>
					<option value="bitwise-10-phase">Bitwise 10-Phase (+ Adaptation)</option>
					<option value="ids-binary-2-phase">IDS Binary 2-Phase Explorer (UNSW-NB15)</option>
					<option value="ids-binary-5-phase">IDS Binary 5-Phase Deep (UNSW-NB15)</option>
					<option value="ids-binary-7-phase">IDS Binary 7-Phase (UNSW-NB15)</option>
					<option value="ids-binary-10-phase">IDS Binary 10-Phase + *genesis (UNSW-NB15)</option>
					<option value="ids-multi-7-phase">IDS Multi-class 7-Phase (UNSW-NB15)</option>
					<option value="ids-multi-10-phase">IDS Multi-class 10-Phase + *genesis (UNSW-NB15)</option>
					<option value="controller-ga-memory">Controller — GA Memory (drone attitude)</option>
					<option value="controller-full-matrix">Controller — Full Matrix (neurons/bits/conn/memory)</option>
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
					{:else if template === 'controller-ga-memory'}
						WNN drone attitude controller &mdash; GA neuroevolution of QSR cells (no training)
					{:else if template === 'controller-full-matrix'}
						Controller GA across neurons &rarr; bits &rarr; connections &rarr; memory
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
			<label for="clusterCrossoverRatio">Cluster Crossover Ratio</label>
			<input type="number" id="clusterCrossoverRatio" bind:value={clusterCrossoverRatio} min="0" max="1" step="0.1" />
			<span class="field-hint">0 = phase-specific only, 1 = cluster-level only</span>
		</div>
		<div class="form-group">
			<label for="poolShuffleRatio">Pool Shuffle Ratio</label>
			<input type="number" id="poolShuffleRatio" bind:value={poolShuffleRatio} min="0" max="1" step="0.1" />
			<span class="field-hint">0 = uniform (2→2), 1 = pool-and-shuffle (2→1)</span>
		</div>
	</div>
	<div class="form-row">
		<div class="form-group">
			<label for="assortativeMatingRatio">Assortative Mating Ratio</label>
			<input type="number" id="assortativeMatingRatio" bind:value={assortativeMatingRatio} min="0" max="1" step="0.05" />
			<span class="field-hint">0 = random p2, 0.85 = NEAT-style (similar mates)</span>
		</div>
	</div>

	<div class="form-row">
		<div class="form-group">
			<label for="patience">Patience</label>
			<input type="number" id="patience" bind:value={patience} min="1" />
			<span class="field-hint">Generations with no improvement before early stop</span>
		</div>

		<div class="form-group">
			<label for="checkInterval">Patience Check Interval</label>
			<input type="number" id="checkInterval" bind:value={checkInterval} min="1" />
			<span class="field-hint">Run the early-stop check every N generations (IDS 10 / controller 5)</span>
		</div>

		<div class="form-group">
			<label for="wnnNumThreads">CPU Threads</label>
			<input type="number" id="wnnNumThreads" bind:value={wnnNumThreads} min="1" />
			<span class="field-hint">RAYON cores; the scheduler budgets concurrency on this</span>
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

<style>
  .form-section {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 0; /* inside .form-columns the page zeroed this */
    box-shadow: var(--glass-shadow), var(--glass-inset);
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
  }

  .form-section:hover {
    box-shadow: var(--glass-shadow-hover), var(--glass-inset);
    border-color: var(--glass-border-highlight);
  }

  h2 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 0.75rem 0;
    text-transform: uppercase;
    letter-spacing: 0.03em;
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

  select:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
</style>
