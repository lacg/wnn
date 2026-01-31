<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import type { Flow, Experiment, Checkpoint, Phase, PhaseResult } from '$lib/types';
  import { formatDate } from '$lib/dateFormat';

  let flow: Flow | null = null;
  let experiments: Experiment[] = [];
  let checkpoints: Checkpoint[] = [];
  let phases: Phase[] = [];
  let phaseResults: Map<number, PhaseResult> = new Map();
  let loading = true;
  let error: string | null = null;
  let saving = false;
  let editMode = false;

  // Experiment editing state
  let editingExpIndex: number | null = null;
  let editingExp: {
    name: string;
    experiment_type: 'ga' | 'ts';
    optimize_bits: boolean;
    optimize_neurons: boolean;
    optimize_connections: boolean;
  } | null = null;
  let showAddPhase = false;
  let newPhase = {
    name: '',
    experiment_type: 'ga' as 'ga' | 'ts',
    optimize_bits: false,
    optimize_neurons: true,
    optimize_connections: false
  };

  // Edit form state
  let editConfig = {
    patience: 10,
    ga_generations: 250,
    ts_iterations: 250,
    population_size: 50,
    neighbors_per_iter: 50,
    fitness_percentile: 0.75,
    tier_config: '',
    optimize_tier0_only: false,
    phase_order: 'neurons_first'
  };

  $: flowId = $page.params.id;

  onMount(async () => {
    await loadFlow();
  });

  async function loadFlow() {
    loading = true;
    try {
      const [flowRes, expsRes, checkpointsRes] = await Promise.all([
        fetch(`/api/flows/${flowId}`),
        fetch(`/api/flows/${flowId}/experiments`),
        fetch(`/api/checkpoints`)
      ]);

      if (!flowRes.ok) throw new Error('Flow not found');

      flow = await flowRes.json();
      experiments = await expsRes.json();
      checkpoints = checkpointsRes.ok ? await checkpointsRes.json() : [];

      // Fetch phases for each experiment
      phases = [];
      phaseResults = new Map();
      for (const exp of experiments) {
        const phasesRes = await fetch(`/api/experiments/${exp.id}/phases`);
        if (phasesRes.ok) {
          const expPhases: Phase[] = await phasesRes.json();
          phases = [...phases, ...expPhases];
        }
      }

      // Populate edit form from config
      if (flow?.config?.params) {
        const p = flow.config.params;
        editConfig.patience = p.patience ?? 10;
        editConfig.ga_generations = p.ga_generations ?? 250;
        editConfig.ts_iterations = p.ts_iterations ?? 250;
        editConfig.population_size = p.population_size ?? 50;
        editConfig.neighbors_per_iter = p.neighbors_per_iter ?? 50;
        editConfig.fitness_percentile = p.fitness_percentile ?? 0.75;
        editConfig.optimize_tier0_only = p.optimize_tier0_only ?? false;
        editConfig.phase_order = p.phase_order ?? 'neurons_first';
        if (p.tier_config) {
          // Handle both string and array formats
          if (typeof p.tier_config === 'string') {
            editConfig.tier_config = p.tier_config;
          } else {
            editConfig.tier_config = p.tier_config
              .map((t: number[]) => `${t[0] ?? 'rest'},${t[1]},${t[2]}`)
              .join(';');
          }
        }
      }
    } catch (e) {
      error = e instanceof Error ? e.message : 'Unknown error';
    } finally {
      loading = false;
    }
  }

  async function saveChanges() {
    if (!flow) return;
    saving = true;

    try {
      // Parse tier_config string
      let tier_config = null;
      if (editConfig.tier_config.trim()) {
        tier_config = editConfig.tier_config.split(';').map(tier => {
          const parts = tier.trim().split(',');
          return [
            parts[0] === 'rest' ? null : parseInt(parts[0]),
            parseInt(parts[1]),
            parseInt(parts[2])
          ];
        });
      }

      const updatedConfig = {
        ...flow.config,
        params: {
          ...flow.config.params,
          patience: editConfig.patience,
          ga_generations: editConfig.ga_generations,
          ts_iterations: editConfig.ts_iterations,
          population_size: editConfig.population_size,
          neighbors_per_iter: editConfig.neighbors_per_iter,
          fitness_percentile: editConfig.fitness_percentile,
          optimize_tier0_only: editConfig.optimize_tier0_only,
          phase_order: editConfig.phase_order,
          tier_config
        }
      };

      const res = await fetch(`/api/flows/${flowId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: updatedConfig })
      });

      if (!res.ok) throw new Error('Failed to save');

      editMode = false;
      await loadFlow();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to save';
    } finally {
      saving = false;
    }
  }

  function cancelEdit() {
    editMode = false;
    // Reload to reset form
    loadFlow();
  }

  // =========================================================================
  // Experiment Editing (for pending phases in running flows)
  // =========================================================================

  function canEditExperiment(index: number): boolean {
    if (!flow) return false;
    // Can always edit if flow is pending or failed
    if (flow.status === 'pending' || flow.status === 'failed') return true;
    // Can't edit completed or cancelled flows
    if (flow.status !== 'running') return false;

    // For running flows, check if this phase has started
    const experiments = flow.config.experiments || [];
    const exp = experiments[index];
    if (!exp) return false;
    const status = getExpStatus(exp, index);
    return status === 'pending';
  }

  function startEditExperiment(index: number) {
    if (!flow || !canEditExperiment(index)) return;
    const experiments = flow.config.experiments || [];
    const exp = experiments[index];
    if (!exp) return;
    editingExpIndex = index;
    editingExp = {
      name: exp.name,
      experiment_type: exp.experiment_type,
      optimize_bits: exp.optimize_bits,
      optimize_neurons: exp.optimize_neurons,
      optimize_connections: exp.optimize_connections
    };
  }

  function cancelEditExperiment() {
    editingExpIndex = null;
    editingExp = null;
  }

  async function saveExperiment() {
    if (!flow || editingExpIndex === null || !editingExp) return;
    saving = true;

    try {
      // Update the experiment in the config
      const currentExperiments = flow.config.experiments || [];
      const updatedExperiments = [...currentExperiments];
      updatedExperiments[editingExpIndex] = {
        ...updatedExperiments[editingExpIndex],
        ...editingExp
      };

      const updatedConfig = {
        ...flow.config,
        experiments: updatedExperiments
      };

      const res = await fetch(`/api/flows/${flowId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: updatedConfig })
      });

      if (!res.ok) throw new Error('Failed to save experiment');

      editingExpIndex = null;
      editingExp = null;
      await loadFlow();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to save';
    } finally {
      saving = false;
    }
  }

  async function deleteExperiment(index: number) {
    if (!flow || !canEditExperiment(index)) return;
    const experiments = flow.config.experiments || [];
    if (!confirm(`Delete "${experiments[index]?.name || 'Experiment'}"? This cannot be undone.`)) return;

    saving = true;
    try {
      const updatedExperiments = experiments.filter((_, i) => i !== index);

      const updatedConfig = {
        ...flow.config,
        experiments: updatedExperiments
      };

      const res = await fetch(`/api/flows/${flowId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: updatedConfig })
      });

      if (!res.ok) throw new Error('Failed to delete experiment');
      await loadFlow();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to delete';
    } finally {
      saving = false;
    }
  }

  async function addExperiment() {
    if (!flow) return;
    saving = true;

    try {
      const currentExperiments = flow.config.experiments || [];
      const updatedExperiments = [
        ...currentExperiments,
        {
          name: newPhase.name || `Phase ${currentExperiments.length + 1}`,
          experiment_type: newPhase.experiment_type,
          optimize_bits: newPhase.optimize_bits,
          optimize_neurons: newPhase.optimize_neurons,
          optimize_connections: newPhase.optimize_connections,
          params: {}
        }
      ];

      const updatedConfig = {
        ...flow.config,
        experiments: updatedExperiments
      };

      const res = await fetch(`/api/flows/${flowId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: updatedConfig })
      });

      if (!res.ok) throw new Error('Failed to add experiment');

      showAddPhase = false;
      newPhase = {
        name: '',
        experiment_type: 'ga',
        optimize_bits: false,
        optimize_neurons: true,
        optimize_connections: false
      };
      await loadFlow();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to add';
    } finally {
      saving = false;
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'queued': return 'var(--accent-yellow, #f59e0b)';
      case 'running': return 'var(--accent-blue)';
      case 'completed': return 'var(--accent-green)';
      case 'failed': return 'var(--accent-red)';
      case 'cancelled': return 'var(--text-tertiary)';
      default: return 'var(--text-secondary)';
    }
  }

  async function queueFlow() {
    if (!flow) return;
    try {
      const response = await fetch(`/api/flows/${flow.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'queued' })
      });
      if (response.ok) {
        await loadFlow();
      } else {
        error = 'Failed to queue flow';
      }
    } catch (e) {
      error = e instanceof Error ? e.message : 'Unknown error';
    }
  }

  // Build expected phase_type from expSpec
  function getExpectedPhaseType(expSpec: { experiment_type: string; optimize_neurons?: boolean; optimize_bits?: boolean; optimize_connections?: boolean }): string {
    const type = expSpec.experiment_type; // 'ga' or 'ts'
    let target = '';
    if (expSpec.optimize_neurons) target = 'neurons';
    else if (expSpec.optimize_bits) target = 'bits';
    else if (expSpec.optimize_connections) target = 'connections';
    return `${type}_${target}`;
  }

  // Find phase by matching phase_type
  function findPhase(expSpec: { name: string; experiment_type: string; optimize_neurons?: boolean; optimize_bits?: boolean; optimize_connections?: boolean }, index: number): Phase | null {
    const expectedPhaseType = getExpectedPhaseType(expSpec);

    // Find all phases with matching type
    const matchingPhases = phases.filter(p => p.phase_type === expectedPhaseType);

    // If we have multiple phases of same type (shouldn't happen normally), return first
    // If only one, return it
    if (matchingPhases.length > 0) {
      return matchingPhases[0];
    }

    // Fallback: try to match by index if phases are in order
    if (index < phases.length) {
      return phases[index];
    }

    return null;
  }

  // Get final checkpoint for an experiment
  function getFinalCheckpoint(experimentId: number): Checkpoint | null {
    return checkpoints.find(c => c.experiment_id === experimentId && c.is_final) ?? null;
  }

  // Get phase status: completed, running, or pending
  function getExpStatus(expSpec: { name: string; experiment_type: string; optimize_neurons?: boolean; optimize_bits?: boolean; optimize_connections?: boolean }, index: number): 'completed' | 'running' | 'pending' {
    const phase = findPhase(expSpec, index);
    if (phase) {
      if (phase.status === 'completed') return 'completed';
      if (phase.status === 'running') return 'running';
      return 'pending';
    }

    // Fallback: if flow is running but no phases exist yet, assume first experiment is running
    if (flow?.status === 'running' && phases.length === 0 && index === 0) {
      return 'running';
    }

    return 'pending';
  }

  // Check if this is the currently running phase
  function isRunningExperiment(expSpec: { name: string; experiment_type: string; optimize_neurons?: boolean; optimize_bits?: boolean; optimize_connections?: boolean }, index: number): boolean {
    const phase = findPhase(expSpec, index);
    if (phase?.status === 'running') return true;

    // Fallback: if flow is running but no phases exist yet, assume first experiment is running
    if (flow?.status === 'running' && phases.length === 0 && index === 0) {
      return true;
    }

    return false;
  }

  // Get metrics for completed phase
  // TODO: Add phase_results API endpoint for per-phase metrics
  function getExpMetrics(expSpec: { name: string; experiment_type: string; optimize_neurons?: boolean; optimize_bits?: boolean; optimize_connections?: boolean }, index: number): { ce: number | null; acc: number | null } | null {
    const phase = findPhase(expSpec, index);
    if (!phase || phase.status !== 'completed') return null;

    // For the last completed phase, we can show checkpoint metrics
    // This is an approximation until we have per-phase results
    const completedPhases = phases.filter(p => p.status === 'completed');
    const isLastCompleted = completedPhases.length > 0 &&
      completedPhases[completedPhases.length - 1].id === phase.id;

    if (isLastCompleted && experiments.length > 0) {
      const exp = experiments[0];
      const checkpoint = checkpoints.find(c => c.experiment_id === exp.id);
      if (checkpoint) {
        return {
          ce: checkpoint.final_fitness,
          acc: checkpoint.final_accuracy
        };
      }
    }

    // For other completed phases, show completion without metrics for now
    return null;
  }

  // Format CE value
  function formatCE(ce: number | null): string {
    if (ce === null) return '-';
    return ce.toFixed(4);
  }

  // Format accuracy percentage
  function formatAccuracy(acc: number | null): string {
    if (acc === null) return '-';
    return `${(acc * 100).toFixed(2)}%`;
  }
</script>

<div class="container">
  {#if loading}
    <div class="loading">Loading flow...</div>
  {:else if error}
    <div class="error">{error}</div>
  {:else if flow}
    <div class="flow-header">
      <div class="header-left">
        <a href="/flows" class="back-link">&larr; Flows</a>
        <h1>{flow.name}</h1>
        <span class="status-badge" style="background: {getStatusColor(flow.status)}">
          {flow.status}
        </span>
      </div>
      <div class="header-actions">
        {#if !editMode && flow.status === 'pending'}
          <button class="btn btn-primary" on:click={queueFlow}>
            Start
          </button>
          <button class="btn btn-secondary" on:click={() => editMode = true}>
            Edit Config
          </button>
        {/if}
        {#if flow.status === 'queued'}
          <span class="queued-hint">Waiting for worker to pick up...</span>
        {/if}
        {#if flow.status === 'failed'}
          <button class="btn btn-primary" on:click={queueFlow}>
            Restart
          </button>
        {/if}
      </div>
    </div>

    {#if flow.description}
      <p class="description">{flow.description}</p>
    {/if}

    <div class="info-cards">
      <div class="info-card">
        <span class="info-label">Created</span>
        <span class="info-value">{formatDate(flow.created_at)}</span>
      </div>
      {#if flow.started_at}
        <div class="info-card">
          <span class="info-label">Started</span>
          <span class="info-value">{formatDate(flow.started_at)}</span>
        </div>
      {/if}
      {#if flow.completed_at}
        <div class="info-card">
          <span class="info-label">Completed</span>
          <span class="info-value">{formatDate(flow.completed_at)}</span>
        </div>
      {/if}
      {#if flow.config.template}
        <div class="info-card">
          <span class="info-label">Template</span>
          <span class="info-value">{flow.config.template}</span>
        </div>
      {/if}
    </div>

    {#if editMode}
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
              <label for="fitness_percentile">Fitness Percentile</label>
              <input type="number" id="fitness_percentile" bind:value={editConfig.fitness_percentile} min="0" max="1" step="0.05" />
              <span class="form-hint">Keep top N% by fitness (0.75 = top 75%)</span>
            </div>
            <div class="form-group checkbox-group">
              <label>
                <input type="checkbox" bind:checked={editConfig.optimize_tier0_only} />
                Optimize Tier0 Only
              </label>
              <span class="form-hint">Only mutate most frequent tokens</span>
            </div>
          </div>

          <div class="form-group full-width">
            <label for="tier_config">Tier Config</label>
            <input type="text" id="tier_config" bind:value={editConfig.tier_config}
                   placeholder="100,15,20;400,10,12;rest,5,8" />
            <span class="form-hint">Format: clusters,neurons,bits per tier (semicolon separated)</span>
          </div>

          <div class="form-actions">
            <button class="btn btn-secondary" on:click={cancelEdit} disabled={saving}>
              Cancel
            </button>
            <button class="btn btn-primary" on:click={saveChanges} disabled={saving}>
              {saving ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </div>
      </section>
    {:else if flow.config.params}
      <section class="section">
        <h2>Parameters</h2>
        <div class="params-grid">
          <div class="param-item">
            <span class="param-label">Patience</span>
            <span class="param-value">{flow.config.params.patience ?? '-'}</span>
          </div>
          <div class="param-item">
            <span class="param-label">Phase Order</span>
            <span class="param-value">{flow.config.params.phase_order ?? 'neurons_first'}</span>
          </div>
          <div class="param-item">
            <span class="param-label">GA Generations</span>
            <span class="param-value">{flow.config.params.ga_generations ?? '-'}</span>
          </div>
          <div class="param-item">
            <span class="param-label">TS Iterations</span>
            <span class="param-value">{flow.config.params.ts_iterations ?? '-'}</span>
          </div>
          <div class="param-item">
            <span class="param-label">Population</span>
            <span class="param-value">{flow.config.params.population_size ?? '-'}</span>
          </div>
          <div class="param-item">
            <span class="param-label">Neighbors/Iter</span>
            <span class="param-value">{flow.config.params.neighbors_per_iter ?? '-'}</span>
          </div>
          <div class="param-item">
            <span class="param-label">Fitness %</span>
            <span class="param-value">{flow.config.params.fitness_percentile ?? '-'}</span>
          </div>
          <div class="param-item">
            <span class="param-label">Tier0 Only</span>
            <span class="param-value">{(flow.config.params.optimize_tier0_only || flow.config.params.tier0_only) ? 'Yes' : 'No'}</span>
          </div>
          {#if flow.config.params.tier_config}
            <div class="param-item full-width">
              <span class="param-label">Tier Config</span>
              <span class="param-value mono">
                {typeof flow.config.params.tier_config === 'string'
                  ? flow.config.params.tier_config
                  : flow.config.params.tier_config.map(t => `${t[0] ?? 'rest'},${t[1]},${t[2]}`).join('; ')}
              </span>
            </div>
          {/if}
          {#if flow.config.params.context_size}
            <div class="param-item">
              <span class="param-label">Context Size</span>
              <span class="param-value">{flow.config.params.context_size}-gram</span>
            </div>
          {/if}
        </div>
      </section>
    {/if}

    <section class="section">
      <div class="section-header">
        <h2>Experiments ({(flow.config.experiments || []).length})</h2>
        {#if flow.status === 'pending' || flow.status === 'running' || flow.status === 'failed'}
          <button class="btn btn-sm btn-secondary" on:click={() => showAddPhase = true}>
            + Add Phase
          </button>
        {/if}
      </div>

      {#if showAddPhase}
        <div class="add-phase-form">
          <h3>Add New Phase</h3>
          <div class="edit-exp-form">
            <div class="form-row">
              <div class="form-group">
                <label>Name</label>
                <input type="text" bind:value={newPhase.name} placeholder="Extra Phase 1: GA Neurons" />
              </div>
              <div class="form-group">
                <label>Type</label>
                <select bind:value={newPhase.experiment_type}>
                  <option value="ga">GA (Genetic Algorithm)</option>
                  <option value="ts">TS (Tabu Search)</option>
                </select>
              </div>
            </div>
            <div class="form-row">
              <div class="form-group">
                <label>Optimize</label>
                <div class="checkbox-row">
                  <label class="checkbox-label">
                    <input type="radio" name="new-optimize" checked={newPhase.optimize_neurons}
                           on:change={() => { newPhase.optimize_neurons = true; newPhase.optimize_bits = false; newPhase.optimize_connections = false; }} />
                    Neurons
                  </label>
                  <label class="checkbox-label">
                    <input type="radio" name="new-optimize" checked={newPhase.optimize_bits}
                           on:change={() => { newPhase.optimize_neurons = false; newPhase.optimize_bits = true; newPhase.optimize_connections = false; }} />
                    Bits
                  </label>
                  <label class="checkbox-label">
                    <input type="radio" name="new-optimize" checked={newPhase.optimize_connections}
                           on:change={() => { newPhase.optimize_neurons = false; newPhase.optimize_bits = false; newPhase.optimize_connections = true; }} />
                    Connections
                  </label>
                </div>
              </div>
            </div>
            <div class="form-actions">
              <button class="btn btn-secondary" on:click={() => showAddPhase = false}>Cancel</button>
              <button class="btn btn-primary" on:click={addExperiment} disabled={saving}>
                {saving ? 'Adding...' : 'Add Phase'}
              </button>
            </div>
          </div>
        </div>
      {/if}

      <div class="experiments-list">
        {#each (flow.config.experiments || []) as exp, i}
          {@const status = getExpStatus(exp, i)}
          {@const metrics = getExpMetrics(exp, i)}
          {@const isRunning = isRunningExperiment(exp, i)}
          {@const canEdit = canEditExperiment(i)}
          {@const isEditing = editingExpIndex === i}

          {#if isEditing && editingExp}
            <div class="experiment-item editing">
              <div class="exp-order">{i + 1}</div>
              <div class="exp-content">
                <div class="edit-exp-form">
                  <div class="form-row">
                    <div class="form-group">
                      <label>Name</label>
                      <input type="text" bind:value={editingExp.name} />
                    </div>
                    <div class="form-group">
                      <label>Type</label>
                      <select bind:value={editingExp.experiment_type}>
                        <option value="ga">GA (Genetic Algorithm)</option>
                        <option value="ts">TS (Tabu Search)</option>
                      </select>
                    </div>
                  </div>
                  <div class="form-row">
                    <div class="form-group">
                      <label>Optimize</label>
                      <div class="checkbox-row">
                        <label class="checkbox-label">
                          <input type="radio" name="edit-optimize" checked={editingExp.optimize_neurons}
                                 on:change={() => { if (editingExp) { editingExp.optimize_neurons = true; editingExp.optimize_bits = false; editingExp.optimize_connections = false; } }} />
                          Neurons
                        </label>
                        <label class="checkbox-label">
                          <input type="radio" name="edit-optimize" checked={editingExp.optimize_bits}
                                 on:change={() => { if (editingExp) { editingExp.optimize_neurons = false; editingExp.optimize_bits = true; editingExp.optimize_connections = false; } }} />
                          Bits
                        </label>
                        <label class="checkbox-label">
                          <input type="radio" name="edit-optimize" checked={editingExp.optimize_connections}
                                 on:change={() => { if (editingExp) { editingExp.optimize_neurons = false; editingExp.optimize_bits = false; editingExp.optimize_connections = true; } }} />
                          Connections
                        </label>
                      </div>
                    </div>
                  </div>
                  <div class="form-actions">
                    <button class="btn btn-secondary" on:click={cancelEditExperiment}>Cancel</button>
                    <button class="btn btn-primary" on:click={saveExperiment} disabled={saving}>
                      {saving ? 'Saving...' : 'Save'}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          {:else}
            <div class="experiment-item" class:running={isRunning} class:completed={status === 'completed'}>
              <div class="exp-order" class:order-completed={status === 'completed'} class:order-running={isRunning}>
                {#if status === 'completed'}
                  <span class="checkmark">✓</span>
                {:else}
                  {i + 1}
                {/if}
              </div>
              <div class="exp-content">
                <div class="exp-header">
                  <div class="exp-name">{exp.name}</div>
                  <span class="status-indicator" class:status-completed={status === 'completed'} class:status-running={isRunning} class:status-pending={status === 'pending'}>
                    {status}
                  </span>
                </div>
                <div class="exp-meta">
                  <span class="exp-type">{exp.experiment_type.toUpperCase()}</span>
                  {#if exp.optimize_bits}
                    <span class="exp-tag">Bits</span>
                  {/if}
                  {#if exp.optimize_neurons}
                    <span class="exp-tag">Neurons</span>
                  {/if}
                  {#if exp.optimize_connections}
                    <span class="exp-tag">Connections</span>
                  {/if}
                </div>
                {#if metrics}
                  <div class="exp-metrics">
                    <span class="metric">
                      <span class="metric-label">CE:</span>
                      <span class="metric-value">{formatCE(metrics.ce)}</span>
                    </span>
                    <span class="metric">
                      <span class="metric-label">Acc:</span>
                      <span class="metric-value">{formatAccuracy(metrics.acc)}</span>
                    </span>
                  </div>
                {/if}
              </div>
              <div class="exp-actions">
                {#if isRunning}
                  <a href="/" class="live-btn" title="View live dashboard">
                    <span class="pulse"></span>
                    Live
                  </a>
                {:else if canEdit}
                  <button class="btn-icon" title="Edit" on:click={() => startEditExperiment(i)}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                    </svg>
                  </button>
                  <button class="btn-icon btn-danger" title="Delete" on:click={() => deleteExperiment(i)}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <polyline points="3 6 5 6 21 6"></polyline>
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                  </button>
                {/if}
              </div>
            </div>
          {/if}
        {/each}
      </div>
    </section>

    {#if experiments.length > 0}
      <section class="section">
        <h2>Completed Runs</h2>
        <div class="runs-table">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Started</th>
                <th>Ended</th>
              </tr>
            </thead>
            <tbody>
              {#each experiments as exp}
                <tr>
                  <td>{exp.name}</td>
                  <td>
                    <span class="status-dot" style="background: {getStatusColor(exp.status)}"></span>
                    {exp.status}
                  </td>
                  <td>{formatDate(exp.started_at)}</td>
                  <td>{formatDate(exp.ended_at)}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </section>
    {/if}

    <!-- Final Results (for completed flows) -->
    {#if flow.status === 'completed'}
      {@const finalCheckpoint = checkpoints.find(c => c.is_final && experiments.some(e => e.id === c.experiment_id))}
      <section class="section">
        <h2>Final Results</h2>
        {#if finalCheckpoint}
          <div class="final-results-card">
            <div class="results-grid">
              <div class="result-item">
                <div class="result-label">Best CE</div>
                <div class="result-value">{finalCheckpoint.final_fitness?.toFixed(4) ?? '—'}</div>
              </div>
              <div class="result-item">
                <div class="result-label">Best Accuracy</div>
                <div class="result-value">{finalCheckpoint.final_accuracy ? (finalCheckpoint.final_accuracy * 100).toFixed(2) + '%' : '—'}</div>
              </div>
              <div class="result-item">
                <div class="result-label">Checkpoint</div>
                <div class="result-value result-path">{finalCheckpoint.name}</div>
              </div>
            </div>
            <div class="results-footer">
              <a href="/" class="btn btn-secondary">View Live Dashboard</a>
              <a href="/checkpoints" class="btn btn-secondary">View All Checkpoints</a>
            </div>
          </div>
        {:else}
          <div class="empty-state">
            <p>No final checkpoint recorded</p>
          </div>
        {/if}
      </section>
    {/if}
  {/if}
</div>

<style>
  .loading, .error {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--text-secondary);
  }

  .error {
    color: var(--accent-red);
  }

  .flow-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1.5rem;
    padding-top: 2rem;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .header-actions {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }

  .queued-hint {
    font-size: 0.875rem;
    color: var(--accent-yellow, #f59e0b);
    font-style: italic;
  }

  .back-link {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 0.875rem;
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

  .status-badge {
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    color: white;
    text-transform: capitalize;
  }

  .description {
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
  }

  .info-cards {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 2rem;
  }

  .info-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .info-label {
    font-size: 0.75rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
  }

  .info-value {
    font-size: 0.875rem;
    color: var(--text-primary);
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

  /* Edit Form Styles */
  .edit-section {
    background: var(--bg-secondary);
    border: 1px solid var(--accent-blue);
    border-radius: 8px;
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

  .form-group.checkbox-group {
    flex-direction: row;
    align-items: center;
    gap: 0.5rem;
  }

  .form-group.checkbox-group label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
  }

  label {
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--text-primary);
  }

  input[type="number"],
  input[type="text"],
  select {
    padding: 0.5rem 0.75rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 0.875rem;
  }

  input[type="number"]:focus,
  input[type="text"]:focus,
  select:focus {
    outline: none;
    border-color: var(--accent-blue);
  }

  .form-hint {
    font-size: 0.75rem;
    color: var(--text-tertiary);
  }

  .form-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
  }

  .btn {
    padding: 0.5rem 1rem;
    border-radius: 4px;
    font-size: 0.875rem;
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
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border);
  }

  .btn-secondary:hover:not(:disabled) {
    background: var(--border);
  }

  /* Parameters Grid */
  .params-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
  }

  .param-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .param-item.full-width {
    grid-column: 1 / -1;
  }

  .param-label {
    font-size: 0.75rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
  }

  .param-value {
    font-size: 0.875rem;
    color: var(--text-primary);
  }

  .param-value.mono {
    font-family: monospace;
    font-size: 0.8125rem;
  }

  /* Experiments List */
  .experiments-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .experiment-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
  }

  .exp-order {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--bg-tertiary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
  }

  .exp-content {
    flex: 1;
  }

  .exp-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .exp-name {
    font-weight: 500;
    color: var(--text-primary);
  }

  .status-indicator {
    font-size: 0.675rem;
    padding: 0.125rem 0.375rem;
    border-radius: 3px;
    text-transform: uppercase;
    font-weight: 600;
  }

  .status-completed {
    background: var(--accent-green);
    color: white;
  }

  .status-running {
    background: var(--accent-blue);
    color: white;
  }

  .status-pending {
    background: var(--bg-tertiary);
    color: var(--text-tertiary);
  }

  .exp-meta {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.25rem;
  }

  .exp-type {
    font-size: 0.75rem;
    color: var(--accent-blue);
    font-weight: 600;
  }

  .exp-tag {
    font-size: 0.75rem;
    color: var(--text-tertiary);
    background: var(--bg-tertiary);
    padding: 0.125rem 0.375rem;
    border-radius: 3px;
  }

  .exp-metrics {
    display: flex;
    gap: 1rem;
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid var(--border);
  }

  .metric {
    display: flex;
    gap: 0.375rem;
    align-items: baseline;
  }

  .metric-label {
    font-size: 0.75rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
  }

  .metric-value {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-primary);
    font-family: monospace;
  }

  .experiment-item.running {
    border-color: var(--accent-blue);
    background: rgba(59, 130, 246, 0.05);
  }

  .experiment-item.completed {
    border-color: var(--accent-green);
  }

  .exp-order.order-completed {
    background: var(--accent-green);
    color: white;
  }

  .exp-order.order-running {
    background: var(--accent-blue);
    color: white;
  }

  .checkmark {
    font-size: 0.875rem;
  }

  .live-btn {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.375rem 0.75rem;
    background: var(--accent-blue);
    color: white;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    text-decoration: none;
    border-radius: 4px;
    transition: background 0.15s;
  }

  .live-btn:hover {
    background: #2563eb;
  }

  .pulse {
    width: 8px;
    height: 8px;
    background: white;
    border-radius: 50%;
    animation: pulse 1.5s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.2); }
  }

  /* Runs Table */
  .runs-table {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }

  table {
    width: 100%;
    border-collapse: collapse;
  }

  th, td {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }

  th {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-tertiary);
    text-transform: uppercase;
    background: var(--bg-tertiary);
  }

  td {
    font-size: 0.875rem;
    color: var(--text-primary);
  }

  tr:last-child td {
    border-bottom: none;
  }

  .status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 0.5rem;
  }

  @media (max-width: 768px) {
    .form-row {
      grid-template-columns: 1fr;
    }

    .params-grid {
      grid-template-columns: repeat(2, 1fr);
    }
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

  .btn-sm {
    padding: 0.375rem 0.75rem;
    font-size: 0.75rem;
  }

  /* Add phase form */
  .add-phase-form {
    background: var(--bg-secondary);
    border: 1px solid var(--accent-blue);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
  }

  .add-phase-form h3 {
    font-size: 0.875rem;
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
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-secondary);
  }

  .edit-exp-form input[type="text"],
  .edit-exp-form select {
    padding: 0.375rem 0.5rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 0.875rem;
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
    font-size: 0.875rem;
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
    border-top: 1px solid var(--border);
  }

  /* Experiment actions */
  .exp-actions {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }

  .btn-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border: none;
    border-radius: 4px;
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn-icon:hover {
    background: var(--border);
    color: var(--text-primary);
  }

  .btn-icon.btn-danger:hover {
    background: var(--accent-red);
    color: white;
  }

  .experiment-item.editing {
    border-color: var(--accent-blue);
    background: rgba(59, 130, 246, 0.05);
  }

  /* Final Results */
  .final-results-card {
    background: var(--bg-secondary);
    border: 1px solid var(--accent-green);
    border-radius: 8px;
    padding: 1.5rem;
  }

  .results-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
    margin-bottom: 1.5rem;
  }

  .result-item {
    text-align: center;
  }

  .result-label {
    font-size: 0.75rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
  }

  .result-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-primary);
    font-family: monospace;
  }

  .result-path {
    font-size: 0.875rem;
    word-break: break-all;
  }

  .results-footer {
    display: flex;
    gap: 0.75rem;
    justify-content: center;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
  }

  .empty-state {
    text-align: center;
    padding: 2rem;
    color: var(--text-secondary);
  }

  @media (max-width: 640px) {
    .results-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
