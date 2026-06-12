<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { PhaseSpec } from '$lib/flowTemplates';
  import { generatePhaseName } from '$lib/flowTemplates';

  export let displayPhases: PhaseSpec[] = [];
  export let isMultiStage = false;
  export let selectedStage = 0;
  export let isBitwise = false;
  export let isIDS = false;

  const dispatch = createEventDispatcher<{
    add: PhaseSpec;
    remove: number;
    move: { index: number; direction: -1 | 1 };
  }>();

  // Add-phase form state
  let newPhaseType: 'ga' | 'ts' | 'lamarckian' = 'ga';
  let newPhaseGenesisMode: 'neurogenesis' | 'synaptogenesis' | 'axonogenesis' = 'neurogenesis';
  let newPhaseGrid = false;
  let newPhaseNeurons = true;
  let newPhaseBits = false;
  let newPhaseConnections = false;

  function addPhase() {
    let newPhase: PhaseSpec;

    if (newPhaseGrid) {
      newPhase = {
        name: 'Grid Search (neurons × bits)',
        experiment_type: 'ga' as const,
        optimize_bits: true,
        optimize_neurons: true,
        optimize_connections: false,
        phase_type: 'grid_search' as const,
      };
    } else if (newPhaseType === 'lamarckian') {
      // Lamarckian → the chosen genesis_mode string (worker maps it to the
      // unified LAMARCKIAN strategy + genesis_mode).
      const mode = newPhaseGenesisMode;
      const label = mode.charAt(0).toUpperCase() + mode.slice(1);
      newPhase = {
        name: label,
        experiment_type: mode as PhaseSpec['experiment_type'],
        optimize_bits: false,
        optimize_neurons: false,
        optimize_connections: false,
        phase_type: mode as PhaseSpec['phase_type'],
      };
    } else {
      if (!newPhaseNeurons && !newPhaseBits && !newPhaseConnections) return;
      const gaTs = newPhaseType as 'ga' | 'ts';  // grid + lamarckian handled above
      newPhase = {
        name: generatePhaseName(gaTs, newPhaseNeurons, newPhaseBits, newPhaseConnections),
        experiment_type: gaTs,
        optimize_bits: newPhaseBits,
        optimize_neurons: newPhaseNeurons,
        optimize_connections: newPhaseConnections,
      };
    }

    dispatch('add', newPhase);
  }
</script>

<div class="form-section">
  <h2>
    {#if isMultiStage}
      Stage {selectedStage} Phases ({displayPhases.length})
    {:else}
      Phases ({displayPhases.length})
    {/if}
  </h2>
  {#if displayPhases.length > 0}
    <div class="phase-list">
      {#each displayPhases as phase, i}
        <div class="phase-item">
          <div class="phase-move">
            <button type="button" class="move-btn" on:click={() => dispatch('move', { index: i, direction: -1 })} disabled={i === 0} title="Move up">&uarr;</button>
            <button type="button" class="move-btn" on:click={() => dispatch('move', { index: i, direction: 1 })} disabled={i === displayPhases.length - 1} title="Move down">&darr;</button>
          </div>
          <span class="phase-num">{i + 1}</span>
          <span class="phase-name">{phase.name}</span>
          <span class="phase-type"
            class:ga={phase.experiment_type === 'ga'}
            class:ts={phase.experiment_type === 'ts'}
            class:adapt={['neurogenesis', 'synaptogenesis', 'axonogenesis'].includes(phase.experiment_type)}>
            {phase.phase_type === 'grid_search' ? 'GRID' : phase.experiment_type.toUpperCase()}
          </span>
          <button type="button" class="remove-btn" on:click={() => dispatch('remove', i)} title="Remove">&times;</button>
        </div>
      {/each}
    </div>
  {:else}
    <p class="empty-phases">No phases. Use a template or add phases manually.</p>
  {/if}
  <div class="add-phase-row">
    {#if isBitwise || isMultiStage || isIDS}
      <label class="inline-check">
        <input type="checkbox" bind:checked={newPhaseGrid} /> Grid Search
      </label>
    {/if}
    {#if !newPhaseGrid}
      <select bind:value={newPhaseType} class="phase-type-select">
        <option value="ga">GA</option>
        <option value="ts">TS</option>
        {#if isBitwise || isMultiStage || isIDS}
          <option value="lamarckian">Lamarckian</option>
        {/if}
      </select>
      {#if newPhaseType === 'lamarckian'}
        <!-- Lamarckian dimension picker — mirrors GA/TS's Neurons/Bits/
             Connections checkboxes: one strategy, *genesis via dropdown. -->
        <select bind:value={newPhaseGenesisMode} class="phase-type-select">
          <option value="neurogenesis">Neurogenesis</option>
          <option value="synaptogenesis">Synaptogenesis</option>
          <option value="axonogenesis">Axonogenesis</option>
        </select>
      {:else}
        <label class="inline-check"><input type="checkbox" bind:checked={newPhaseNeurons} /> Neurons</label>
        <label class="inline-check"><input type="checkbox" bind:checked={newPhaseBits} /> Bits</label>
        <label class="inline-check"><input type="checkbox" bind:checked={newPhaseConnections} /> Connections</label>
      {/if}
    {/if}
    <button type="button" class="btn btn-add" on:click={addPhase}
      disabled={!newPhaseGrid && newPhaseType !== 'lamarckian' && !newPhaseNeurons && !newPhaseBits && !newPhaseConnections}>
      + Add Phase
    </button>
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

  input[type="checkbox"] {
    width: auto;
    margin-right: 0.5rem;
    vertical-align: middle;
  }

  select {
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

  select:focus {
    outline: none;
    border-color: rgba(59, 130, 246, 0.6);
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.15), 0 0 0 3px rgba(59, 130, 246, 0.15);
  }

  /* Phases Preview */
  .phase-list {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
  }

  .phase-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1rem;
    padding: 0.375rem 0.5rem;
    border-radius: 8px;
    background: rgba(30, 41, 59, 0.3);
    border: 1px solid rgba(148, 163, 184, 0.08);
    transition: background 0.15s ease, border-color 0.15s ease;
  }

  .phase-item:hover {
    background: rgba(30, 41, 59, 0.5);
    border-color: rgba(148, 163, 184, 0.15);
  }

  .phase-num {
    width: 1.5rem;
    height: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 6px;
    font-size: 1rem;
    color: var(--accent-blue);
    font-weight: 600;
    flex-shrink: 0;
  }

  .phase-name {
    flex: 1;
    color: var(--text-primary);
  }

  .phase-type {
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 1rem;
    font-weight: 600;
    text-transform: uppercase;
    flex-shrink: 0;
  }

  .phase-type.ga {
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.2);
    color: var(--accent-blue);
    box-shadow: 0 0 8px rgba(59, 130, 246, 0.1);
  }

  .phase-type.ts {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.2);
    color: var(--accent-green);
    box-shadow: 0 0 8px rgba(16, 185, 129, 0.1);
  }

  .phase-type.adapt {
    background: rgba(168, 85, 247, 0.15);
    border: 1px solid rgba(168, 85, 247, 0.2);
    color: #a855f7;
    box-shadow: 0 0 8px rgba(168, 85, 247, 0.1);
  }

  .phase-move {
    display: flex;
    flex-direction: column;
    gap: 1px;
  }

  .move-btn {
    background: rgba(51, 65, 85, 0.4);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 4px;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 1rem;
    line-height: 1;
    padding: 0 0.25rem;
    transition: all 0.15s;
  }

  .move-btn:hover:not(:disabled) {
    background: var(--border);
    color: var(--text-primary);
  }

  .move-btn:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }

  .remove-btn {
    background: none;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 1.25rem;
    line-height: 1;
    padding: 0 0.25rem;
    transition: color 0.15s;
    flex-shrink: 0;
  }

  .remove-btn:hover {
    color: var(--accent-red);
  }

  .empty-phases {
    color: var(--text-secondary);
    font-size: 1rem;
    font-style: italic;
    margin: 0.5rem 0;
  }

  .add-phase-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 1px solid var(--border);
    flex-wrap: wrap;
  }

  .phase-type-select {
    width: auto;
    padding: 0.25rem 0.5rem;
    font-size: 1rem;
  }

  .inline-check {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 1rem;
    color: var(--text-primary);
    cursor: pointer;
    margin-bottom: 0;
    font-weight: 400;
  }

  .inline-check input[type="checkbox"] {
    width: auto;
    margin: 0;
  }

  .btn-add {
    background: rgba(51, 65, 85, 0.4);
    border: 1px solid var(--glass-border);
    color: var(--text-primary);
    padding: 0.25rem 0.75rem;
    border-radius: 8px;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.2s ease;
    margin-left: auto;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
  }

  .btn-add:hover:not(:disabled) {
    background: rgba(71, 85, 105, 0.5);
    border-color: var(--glass-border-highlight);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  }

  .btn-add:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
</style>
