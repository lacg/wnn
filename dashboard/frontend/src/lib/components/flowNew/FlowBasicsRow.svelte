<script lang="ts">
	export let name = ''
	export let description = ''
	export let numStages = 1
</script>

<!-- Top row: Name + Description + Stages -->
<div class="form-section">
	<div class="form-row-header">
		<div class="form-group">
			<label for="name">Name *</label>
			<input type="text" id="name" bind:value={name} placeholder="e.g., Pass 1 - Initial Search" />
		</div>
		<div class="form-group">
			<label for="description">Description</label>
			<input type="text" id="description" bind:value={description} placeholder="Optional description..." />
		</div>
		<div class="form-group">
			<label for="numStages">Stages</label>
			<input type="number" id="numStages" bind:value={numStages} min="1" max="4" />
			<span class="field-hint">
				{#if numStages === 1}
					Single-stage
				{:else}
					{numStages}-stage factorized
				{/if}
			</span>
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
    margin-bottom: 1.5rem;
    box-shadow: var(--glass-shadow), var(--glass-inset);
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
  }

  .form-section:hover {
    box-shadow: var(--glass-shadow-hover), var(--glass-inset);
    border-color: var(--glass-border-highlight);
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

  .form-row-header {
    display: grid;
    grid-template-columns: 2fr 2fr 1fr;
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
