<script lang="ts">
	import { createEventDispatcher } from 'svelte'
	import TierConfigEditor from '$lib/components/TierConfigEditor.svelte'

	/** Edit form state — owned by the page (loadFlow populates it); the form
   *  binds to its fields in place. */
	export let editConfig: {
		patience: number
		ga_generations: number
		ts_iterations: number
		population_size: number
		neighbors_per_iter: number
		fitness_percentile: number
		fitness_calculator: string
		fitness_weight_ce: number
		fitness_weight_acc: number
		min_accuracy_floor: number
		threshold_start: number
		threshold_step: number
		tier_config: string
		phase_order: string
		context_size: number
		cluster_crossover_ratio: number
		pool_shuffle_ratio: number
		assortative_mating_ratio: number
	}
	export let saving: boolean = false
	export let isBitwise: boolean = false

	const dispatch = createEventDispatcher<{ save: void, cancel: void }>()
</script>

<section class="section edit-section">
	<h2>Edit Configuration</h2>
	<div class="edit-form">
		<div class="form-row">
			<div class="form-group">
				<label for="patience">Patience</label>
				<input type="number" id="patience" bind:value={editConfig.patience} min="1" max="100" />
				<span class="form-hint">Early stopping patience</span>
			</div>
			<div class="form-group">
				<label for="phase_order">Phase Order</label>
				<select id="phase_order" bind:value={editConfig.phase_order}>
					<option value="neurons_first">Neurons First</option>
					<option value="bits_first">Bits First</option>
				</select>
			</div>
		</div>

		<div class="form-row">
			<div class="form-group">
				<label for="ga_generations">GA Generations</label>
				<input type="number" id="ga_generations" bind:value={editConfig.ga_generations} min="10" max="10000" />
			</div>
			<div class="form-group">
				<label for="ts_iterations">TS Iterations</label>
				<input type="number" id="ts_iterations" bind:value={editConfig.ts_iterations} min="10" max="10000" />
			</div>
		</div>

		<div class="form-row">
			<div class="form-group">
				<label for="population_size">Population Size</label>
				<input type="number" id="population_size" bind:value={editConfig.population_size} min="10" max="500" />
			</div>
			<div class="form-group">
				<label for="neighbors_per_iter">Neighbors/Iter</label>
				<input type="number" id="neighbors_per_iter" bind:value={editConfig.neighbors_per_iter} min="10" max="500" />
			</div>
		</div>

		<div class="form-row">
			<div class="form-group">
				<label for="cluster_crossover_ratio">Cluster Crossover Ratio</label>
				<input type="number" id="cluster_crossover_ratio" bind:value={editConfig.cluster_crossover_ratio} min="0" max="1" step="0.1" />
				<span class="form-hint">0 = phase-specific only, 1 = cluster-level only</span>
			</div>
			<div class="form-group">
				<label for="pool_shuffle_ratio">Pool Shuffle Ratio</label>
				<input type="number" id="pool_shuffle_ratio" bind:value={editConfig.pool_shuffle_ratio} min="0" max="1" step="0.1" />
				<span class="form-hint">0 = uniform (2→2), 1 = pool-and-shuffle (2→1)</span>
			</div>
		</div>
		<div class="form-row">
			<div class="form-group">
				<label for="assortative_mating_ratio">Assortative Mating Ratio</label>
				<input type="number" id="assortative_mating_ratio" bind:value={editConfig.assortative_mating_ratio} min="0" max="1" step="0.05" />
				<span class="form-hint">0 = random p2, 0.85 = NEAT-style (similar mates), 1 = always similar</span>
			</div>
		</div>

		<div class="form-row">
			<div class="form-group">
				<label for="fitness_percentile">Fitness Percentile</label>
				<input type="number" id="fitness_percentile" bind:value={editConfig.fitness_percentile} min="0" max="1" step="0.05" />
				<span class="form-hint">Keep top N% by fitness (0.75 = top 75%)</span>
			</div>
			<div class="form-group">
				<label for="fitness_calculator">Fitness Calculator</label>
				<select id="fitness_calculator" bind:value={editConfig.fitness_calculator}>
					<option value="normalized">Normalized</option>
					<option value="normalized_harmonic">Normalized Harmonic</option>
					<option value="harmonic_rank">Harmonic Rank</option>
					<option value="ce">CE Only</option>
				</select>
				<span class="form-hint">How to rank genomes by CE and accuracy</span>
			</div>
		</div>

		<div class="form-row">
			<div class="form-group">
				<label for="fitness_weight_ce">CE Weight</label>
				<input type="number" id="fitness_weight_ce" bind:value={editConfig.fitness_weight_ce} min="0" max="10" step="0.1" />
				<span class="form-hint">Weight for CE in fitness (higher = CE matters more)</span>
			</div>
			<div class="form-group">
				<label for="fitness_weight_acc">Accuracy Weight</label>
				<input type="number" id="fitness_weight_acc" bind:value={editConfig.fitness_weight_acc} min="0" max="10" step="0.1" />
				<span class="form-hint">Weight for accuracy (higher = acc matters more)</span>
			</div>
		</div>

		<div class="form-row">
			<div class="form-group">
				<label for="min_accuracy_floor">Accuracy Floor</label>
				<input type="number" id="min_accuracy_floor" bind:value={editConfig.min_accuracy_floor} min="0" max="0.1" step="0.001" />
				<span class="form-hint">Hard floor (0.003 = 0.3%). Below = fitness infinity</span>
			</div>
			<div class="form-group">
				<label for="context_size">Context Size (N-gram)</label>
				<input type="number" id="context_size" bind:value={editConfig.context_size} min="1" max="16" />
				<span class="form-hint">Number of context tokens (e.g., 4 = 4-gram)</span>
			</div>
		</div>

		<div class="form-row">
			<div class="form-group">
				<label for="threshold_start">Threshold Start (%)</label>
				<input type="number" id="threshold_start" bind:value={editConfig.threshold_start} min="0" max="50" step="0.1" />
				<span class="form-hint">Accuracy filter at phase 1 (0 = no filter)</span>
			</div>
			<div class="form-group">
				<label for="threshold_step">Threshold Increase / Phase (%)</label>
				<input type="number" id="threshold_step" bind:value={editConfig.threshold_step} min="0" max="50" step="0.1" />
				<span class="form-hint">How much accuracy filter grows each phase</span>
			</div>
		</div>

		{#if !isBitwise}
			<div class="form-group full-width">
				<!-- svelte-ignore a11y-label-has-associated-control -->
				<label>Tier Config</label>
				<TierConfigEditor bind:value={editConfig.tier_config} />
			</div>
		{/if}

		<div class="form-actions">
			<button class="btn btn-secondary" on:click={() => dispatch('cancel')} disabled={saving}>
				Cancel
			</button>
			<button class="btn btn-primary" on:click={() => dispatch('save')} disabled={saving}>
				{saving ? 'Saving...' : 'Save Changes'}
			</button>
		</div>
	</div>
</section>

<style>
  .section {
    margin-bottom: 2rem;
  }

  h2 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
  }

  /* Edit Form Styles */
  .edit-section {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--accent-blue);
    border-radius: 12px;
    padding: 1.5rem;
  }

  .edit-form {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .form-row {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
  }

  .form-group {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
  }

  .form-group.full-width {
    grid-column: 1 / -1;
  }

  label {
    font-size: 1rem;
    font-weight: 500;
    color: var(--text-primary);
  }

  input[type="number"],
  select {
    padding: 0.5rem 0.75rem;
    border: 1px solid var(--glass-border);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 1rem;
  }

  input[type="number"]:focus,
  select:focus {
    outline: none;
    border-color: var(--accent-blue);
  }

  .form-hint {
    font-size: 1rem;
    color: var(--text-tertiary);
  }

  .form-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--glass-border);
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

  .btn-primary {
    background: var(--accent-blue);
    color: white;
  }

  .btn-primary:hover:not(:disabled) {
    background: #2563eb;
  }

  .btn-secondary {
    background: rgba(51, 65, 85, 0.4);
    color: var(--text-primary);
    border: 1px solid var(--glass-border);
  }

  .btn-secondary:hover:not(:disabled) {
    background: var(--border);
  }

  @media (max-width: 768px) {
    .form-row {
      grid-template-columns: 1fr;
    }
  }
</style>
