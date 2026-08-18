<script lang="ts">
	export let bitwiseNumClusters = 16
	export let bitwiseMinBits = 10
	export let bitwiseMaxBits = 24
	export let bitwiseMinNeurons = 10
	export let bitwiseMaxNeurons = 300
	export let bitwiseMemoryMode = 'QUAD_WEIGHTED'
	export let bitwiseNeuronSampleRate = 0.25
</script>

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
</style>
