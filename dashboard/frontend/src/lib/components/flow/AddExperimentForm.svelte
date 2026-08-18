<script lang="ts">
	import { createEventDispatcher } from 'svelte'
	import type { Checkpoint } from '$lib/types'

	/** New-experiment form state — owned by the page (addExperiment reads and
   *  resets it); the form binds to its fields in place. */
	export let newPhase: {
		name: string
		experiment_type: string
		optimize_bits: boolean
		optimize_neurons: boolean
		optimize_connections: boolean
		genesis_mode: string
		s0_checkpoint_id: number | null
		s1_checkpoint_id: number | null
		lambda_values: string
		genome_type: string
	}
	export let checkpoints: Checkpoint[] = []
	export let saving: boolean = false

	const dispatch = createEventDispatcher<{ add: void, cancel: void }>()

	// Types that don't need the Optimize radios (Lamarckian uses its own
  // genesis_mode picker instead — see below).
	$: newPhaseHidesOptimize = ['grid_search', 'lamarckian', 'lambda_sweep'].includes(newPhase.experiment_type)
</script>

<div class="add-phase-form">
	<h3>Add New Experiment</h3>
	<div class="edit-exp-form">
		<div class="form-row">
			<div class="form-group">
				<label for="new-exp-name">Name</label>
				<input type="text" id="new-exp-name" bind:value={newPhase.name} placeholder="Extra Experiment: GA Neurons" />
			</div>
			<div class="form-group">
				<label for="new-exp-type">Type</label>
				<select id="new-exp-type" bind:value={newPhase.experiment_type} on:change={() =>
				{
					if (newPhase.experiment_type === 'grid_search')
					{
						newPhase.optimize_neurons = true
						newPhase.optimize_bits = true
						newPhase.optimize_connections = false
					}
					else if (['lamarckian', 'lambda_sweep'].includes(newPhase.experiment_type))
					{
						newPhase.optimize_neurons = false
						newPhase.optimize_bits = false
						newPhase.optimize_connections = false
					}
				}}>
					<option value="ga">GA (Genetic Algorithm)</option>
					<option value="ts">TS (Tabu Search)</option>
					<option value="lamarckian">Lamarckian (stats-guided *genesis)</option>
					<option value="grid_search">Grid Search</option>
					<option value="lambda_sweep">Lambda Sweep (unigram interpolation)</option>
				</select>
			</div>
		</div>
		{#if newPhase.experiment_type === 'lamarckian'}
			<!-- Lamarckian dimension picker — mirrors GA/TS's Optimize control:
           one strategy, the *genesis operator chosen via a dropdown. -->
			<div class="form-row">
				<div class="form-group">
					<label for="new-exp-genesis">Genesis</label>
					<select id="new-exp-genesis" bind:value={newPhase.genesis_mode}>
						<option value="neurogenesis">Neurogenesis (neurons)</option>
						<option value="synaptogenesis">Synaptogenesis (connections)</option>
						<option value="axonogenesis">Axonogenesis (rewiring)</option>
					</select>
				</div>
			</div>
		{/if}
		{#if !newPhaseHidesOptimize}
			<div class="form-row">
				<div class="form-group">
					<!-- svelte-ignore a11y-label-has-associated-control -->
					<label>Optimize</label>
					<div class="checkbox-row">
						<label class="checkbox-label">
							<input type="radio" name="new-optimize" checked={newPhase.optimize_neurons}
								on:change={() =>
								{
									newPhase.optimize_neurons = true; newPhase.optimize_bits = false; newPhase.optimize_connections = false
								}} />
							Neurons
						</label>
						<label class="checkbox-label">
							<input type="radio" name="new-optimize" checked={newPhase.optimize_bits}
								on:change={() =>
								{
									newPhase.optimize_neurons = false; newPhase.optimize_bits = true; newPhase.optimize_connections = false
								}} />
							Bits
						</label>
						<label class="checkbox-label">
							<input type="radio" name="new-optimize" checked={newPhase.optimize_connections}
								on:change={() =>
								{
									newPhase.optimize_neurons = false; newPhase.optimize_bits = false; newPhase.optimize_connections = true
								}} />
							Connections
						</label>
					</div>
				</div>
			</div>
		{/if}
		{#if newPhase.experiment_type === 'lambda_sweep'}
			<div class="form-row">
				<div class="form-group">
					<label for="new-s0-ckpt">S0 Checkpoint ID</label>
					<div class="checkpoint-selector">
						<input type="number" id="new-s0-ckpt" bind:value={newPhase.s0_checkpoint_id} placeholder="Checkpoint ID" />
						<select on:change={(e) =>
						{
							const v = e.currentTarget.value; if (v) newPhase.s0_checkpoint_id = parseInt(v)
						}}>
							<option value="">Select from list...</option>
							{#each checkpoints.filter(c => c.flow_name) as ckpt}
								<option value={ckpt.id}>#{ckpt.id} — {ckpt.flow_name} — {ckpt.name} (CE: {ckpt.best_ce?.toFixed(4) ?? '?'})</option>
							{/each}
						</select>
					</div>
				</div>
			</div>
			<div class="form-row">
				<div class="form-group">
					<label for="new-s1-ckpt">S1 Checkpoint ID</label>
					<div class="checkpoint-selector">
						<input type="number" id="new-s1-ckpt" bind:value={newPhase.s1_checkpoint_id} placeholder="Checkpoint ID" />
						<select on:change={(e) =>
						{
							const v = e.currentTarget.value; if (v) newPhase.s1_checkpoint_id = parseInt(v)
						}}>
							<option value="">Select from list...</option>
							{#each checkpoints.filter(c => c.flow_name) as ckpt}
								<option value={ckpt.id}>#{ckpt.id} — {ckpt.flow_name} — {ckpt.name} (CE: {ckpt.best_ce?.toFixed(4) ?? '?'})</option>
							{/each}
						</select>
					</div>
				</div>
			</div>
			<div class="form-row">
				<div class="form-group">
					<label for="new-lambda-values">Lambda Values (comma-separated)</label>
					<input type="text" id="new-lambda-values" bind:value={newPhase.lambda_values} placeholder="0.01,0.05,0.1,0.2,0.3,0.5,0.7,0.9" />
				</div>
				<div class="form-group">
					<label for="new-genome-type">Genome Type</label>
					<select id="new-genome-type" bind:value={newPhase.genome_type}>
						<option value="best_ce">Best CE</option>
						<option value="best_acc">Best ACC</option>
						<option value="best_fitness">Best Fitness</option>
					</select>
				</div>
			</div>
		{/if}
		<div class="form-actions">
			<button class="btn btn-secondary" on:click={() => dispatch('cancel')}>Cancel</button>
			<button class="btn btn-primary" on:click={() => dispatch('add')} disabled={saving}>
				{saving ? 'Adding...' : 'Add Experiment'}
			</button>
		</div>
	</div>
</div>

<style>
  /* Add phase form */
  .add-phase-form {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--accent-blue);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
  }

  .add-phase-form h3 {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 1rem;
    color: var(--text-primary);
  }

  /* Experiment edit form */
  .edit-exp-form {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .edit-exp-form .form-row {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
  }

  .edit-exp-form .form-group {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .edit-exp-form label {
    font-size: 1rem;
    font-weight: 500;
    color: var(--text-secondary);
  }

  /* Page-level base rule (number inputs were styled by the page's generic
     input rule before extraction, not by the .edit-exp-form override). */
  input[type="number"] {
    padding: 0.5rem 0.75rem;
    border: 1px solid var(--glass-border);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 1rem;
  }

  input[type="number"]:focus {
    outline: none;
    border-color: var(--accent-blue);
  }

  .edit-exp-form input[type="text"],
  .edit-exp-form select {
    padding: 0.375rem 0.5rem;
    border: 1px solid var(--glass-border);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 1rem;
  }

  .edit-exp-form input:focus,
  .edit-exp-form select:focus {
    outline: none;
    border-color: var(--accent-blue);
  }

  .checkbox-row {
    display: flex;
    gap: 1rem;
  }

  .checkbox-label {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    font-size: 1rem;
    color: var(--text-primary);
    cursor: pointer;
  }

  .checkbox-label input {
    cursor: pointer;
  }

  .edit-exp-form .form-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    margin-top: 0.5rem;
    padding-top: 0.75rem;
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

  .checkpoint-selector {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }

  .checkpoint-selector input[type="number"] {
    width: 6rem;
    flex-shrink: 0;
  }

  .checkpoint-selector select {
    flex: 1;
    min-width: 0;
  }
</style>
