<script lang="ts">
	import SeedCheckpointSelector from '$lib/components/SeedCheckpointSelector.svelte'

	export let seedFromLeaderboard = false
	export let seedLeaderboardCount = 150
	export let seedCheckpointId: number | null = null
</script>

<div class="form-section">
	<h2>Seed Population</h2>
	<p class="section-hint">
		Seed from a checkpoint or the leaderboard. Remove Grid Search when using leaderboard seed.
	</p>
	<div class="form-group">
		<label for="seedFromLeaderboard">
			<input type="checkbox" id="seedFromLeaderboard" bind:checked={seedFromLeaderboard} />
			Seed from Leaderboard
		</label>
		<span class="field-hint">Use top genomes (with connections) as initial population — skip Grid Search</span>
	</div>
	{#if seedFromLeaderboard}
		<div class="form-row">
			<div class="form-group">
				<label for="seedLeaderboardCount">Top N Genomes</label>
				<input type="number" id="seedLeaderboardCount" bind:value={seedLeaderboardCount} min="10" max="500" step="10" />
				<span class="field-hint">Number of genomes to pull from leaderboard</span>
			</div>
		</div>
	{:else}
		<SeedCheckpointSelector bind:value={seedCheckpointId} />
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

  .section-hint {
    font-size: 1rem;
    color: var(--text-secondary);
    margin: -0.5rem 0 0.75rem 0;
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

  input {
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

  input:focus {
    outline: none;
    border-color: rgba(59, 130, 246, 0.6);
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.15), 0 0 0 3px rgba(59, 130, 246, 0.15);
  }
</style>
