<script lang="ts">
	export let idsDataset = 'unsw-nb15'
	export let idsClassification = 'binary'
	export let idsSingleCluster = true
	export let idsNBits = 8
	export let idsValFraction = 0.25
	export let idsKFolds = 5
	export let idsKFoldPerGen = 1
	export let idsSplit = 'standard'
	export let idsFeatureSelection = 'all'
	export let idsRestBits = 8
	export let idsMinBits = 4
	export let idsMaxBits = 16
	export let idsMinNeurons = 5
	export let idsMaxNeurons = 500
	export let idsMaxBitDelta = 0
	export let idsNeuronSampleRate = 0.25
	export let idsBalanceClasses = true
	export let idsSingleGenome = false
	export let idsSingleNeurons = 200
	export let idsSingleBits = 4
</script>

<div class="form-section">
	<h2>IDS Configuration</h2>
	<div class="form-row">
		<div class="form-group">
			<label for="idsDataset">Dataset</label>
			<select id="idsDataset" bind:value={idsDataset}>
				<option value="unsw-nb15">UNSW-NB15</option>
				<option value="cicids2017">CICIDS2017</option>
				<option value="ciciot2023">CIC-IoT-2023 (1.3M subsample)</option>
				<option value="ciciot2023_full">CIC-IoT-2023 (full 46M)</option>
			</select>
			<span class="field-hint">
				{#if idsDataset === 'unsw-nb15'}175K train / 82K test (temporal)
				{:else if idsDataset === 'cicids2017'}2.3M train / 566K test (random 80/20)
				{:else if idsDataset === 'ciciot2023'}1.07M train / 268K test (random 80/20)
				{:else}30.8M train / 7.7M test (random 80/20) — needs ~30 GB RAM
				{/if}
			</span>
		</div>
		<div class="form-group">
			<label for="idsSingleGenome">
				<input type="checkbox" id="idsSingleGenome" bind:checked={idsSingleGenome} />
				Single Genome Mode
			</label>
			<span class="field-hint">Skip GA, evaluate one (neurons, bits) point only — for ad-hoc evals like the 46M sweep</span>
		</div>
	</div>
	{#if idsSingleGenome}
		<div class="form-row">
			<div class="form-group">
				<label for="idsSingleNeurons">Neurons</label>
				<input type="number" id="idsSingleNeurons" bind:value={idsSingleNeurons} min="1" max="1000" />
				<span class="field-hint">Single neuron count for this eval</span>
			</div>
			<div class="form-group">
				<label for="idsSingleBits">Bits</label>
				<input type="number" id="idsSingleBits" bind:value={idsSingleBits} min="2" max="34" />
				<span class="field-hint">Single address width for this eval ({idsSingleNeurons * (2 ** idsSingleBits) * 2 / 8} bytes total)</span>
			</div>
		</div>
	{/if}
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
			<label for="idsFeatureSelection">Feature Selection</label>
			<select id="idsFeatureSelection" bind:value={idsFeatureSelection}>
				<option value="all">All features (uniform)</option>
				<option value="top20">Top-20 RF features only</option>
				<option value="top20_split">Top-20 high-res + rest standard</option>
			</select>
			<span class="field-hint">
				{#if idsFeatureSelection === 'all'}All 42 features at {idsNBits}b each
				{:else if idsFeatureSelection === 'top20'}20 features at 16b each (~288 bits)
				{:else}Top-20 at 16b + rest at {idsRestBits}b
				{/if}
			</span>
		</div>
		{#if idsFeatureSelection === 'top20_split'}
			<div class="form-group">
				<label for="idsRestBits">Rest Features Bits</label>
				<input type="number" id="idsRestBits" bind:value={idsRestBits} min="2" max="16" />
				<span class="field-hint">Bits for the 22 non-top-20 features</span>
			</div>
		{/if}
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
		<div class="form-group">
			<label for="idsKFoldPerGen">Folds per Gen</label>
			<input type="number" id="idsKFoldPerGen" bind:value={idsKFoldPerGen} min="1" max={idsKFolds} />
			<span class="field-hint">Folds evaluated per generation (1 = rotate, {idsKFolds} = all folds, {idsKFoldPerGen}x cost)</span>
		</div>
	</div>
	<div class="form-row">
		<div class="form-group">
			<label for="idsNeuronSampleRate">Neuron Sample Rate</label>
			<input type="number" id="idsNeuronSampleRate" bind:value={idsNeuronSampleRate} min="0.05" max="1.0" step="0.05" />
			<span class="field-hint">Fraction of neurons trained per example (0.25 = 25%)</span>
		</div>
	</div>
	<div class="form-row">
		<div class="form-group">
			<label for="idsBalanceClasses">
				<input type="checkbox" id="idsBalanceClasses" bind:checked={idsBalanceClasses} />
				Balance Classes
			</label>
			<span class="field-hint">Upweight minority class during training to prevent address saturation bias</span>
		</div>
		<div class="form-group">
			<label for="idsSingleCluster">
				<input type="checkbox" id="idsSingleCluster" bind:checked={idsSingleCluster} />
				Single-Cluster Mode
			</label>
			<span class="field-hint">1 cluster, threshold at 0.5 (unchecked = 2 clusters with softmax argmax)</span>
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
			<input type="number" id="idsMinNeurons" bind:value={idsMinNeurons} min="1" max="1000" />
			<span class="field-hint">Min neurons per class</span>
		</div>
		<div class="form-group">
			<label for="idsMaxNeurons">Max Neurons</label>
			<input type="number" id="idsMaxNeurons" bind:value={idsMaxNeurons} min="1" max="1000" />
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
</style>
