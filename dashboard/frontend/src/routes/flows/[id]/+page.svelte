<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { page } from '$app/stores';
  import type { Flow, Experiment, Checkpoint } from '$lib/types';
  import { formatDate } from '$lib/dateFormat';
  import { currentFlow, flows } from '$lib/stores';

  // Note: With normalized design, experiments come from the experiments table (via DB),
  // NOT from flow.config.experiments. The `experiments` array is fetched separately
  // via /api/flows/${flowId}/experiments and used directly.

  let flow: Flow | null = null;

  // Subscribe to flow updates from WebSocket
  const unsubscribeCurrentFlow = currentFlow.subscribe((wsFlow) => {
    if (wsFlow && flow && wsFlow.id === flow.id) {
      // Update local flow with WebSocket data
      flow = wsFlow;
      // Refetch experiments when flow updates (experiment status may have changed)
      refreshExperiments();
    }
  });

  // Also subscribe to flows list updates (for FlowQueued, FlowStarted, etc.)
  const unsubscribeFlows = flows.subscribe((flowList) => {
    if (flow) {
      const updated = flowList.find((f) => f.id === flow!.id);
      if (updated && updated.status !== flow.status) {
        flow = updated;
        // Refetch experiments when flow status changes
        refreshExperiments();
      }
    }
  });

  // Refresh experiments without full page reload
  async function refreshExperiments() {
    try {
      const res = await fetch(`/api/flows/${flowId}/experiments`);
      if (res.ok) {
        const expsData = await res.json();
        experiments = Array.isArray(expsData) ? expsData : [];
      }
    } catch (e) {
      console.error('Failed to refresh experiments:', e);
    }
  }

  // Periodic refresh when flow is running
  let pollInterval: ReturnType<typeof setInterval> | null = null;

  $: {
    // Start/stop polling based on flow status
    if (flow?.status === 'running' || flow?.status === 'queued') {
      if (!pollInterval) {
        pollInterval = setInterval(refreshExperiments, 3000); // Poll every 3 seconds
      }
    } else {
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
    }
  }

  onDestroy(() => {
    unsubscribeCurrentFlow();
    unsubscribeFlows();
    if (pollInterval) {
      clearInterval(pollInterval);
    }
  });
  let experiments: Experiment[] = [];
  let checkpoints: Checkpoint[] = [];
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
    fitness_calculator: 'normalized',
    fitness_weight_ce: 1.0,
    fitness_weight_acc: 1.0,
    min_accuracy_floor: 0,
    tier_config: '',
    tier0_only: false,
    phase_order: 'neurons_first',
    context_size: 4
  };

  // Flow rename state
  let editingName = false;
  let editedName = '';

  // Duplicate state
  let duplicating = false;

  $: flowId = $page.params.id;

  // Reactive display experiments - re-computed when flow or experiments change
  $: displayExperiments = getDisplayExperiments(flow, experiments);

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
      // Ensure experiments is always an array (defensive against API returning {})
      const expsData = await expsRes.json();
      experiments = Array.isArray(expsData) ? expsData : [];
      // Ensure checkpoints is always an array
      const checkpointsData = checkpointsRes.ok ? await checkpointsRes.json() : [];
      checkpoints = Array.isArray(checkpointsData) ? checkpointsData : [];

      // Populate edit form from config
      if (flow?.config?.params) {
        const p = flow.config.params;
        editConfig.patience = p.patience ?? 10;
        editConfig.ga_generations = p.ga_generations ?? 250;
        editConfig.ts_iterations = p.ts_iterations ?? 250;
        editConfig.population_size = p.population_size ?? 50;
        editConfig.neighbors_per_iter = p.neighbors_per_iter ?? p.population_size ?? 50;
        editConfig.fitness_percentile = p.fitness_percentile ?? 0.75;
        editConfig.fitness_calculator = p.fitness_calculator ?? 'normalized';
        editConfig.fitness_weight_ce = p.fitness_weight_ce ?? 1.0;
        editConfig.fitness_weight_acc = p.fitness_weight_acc ?? 1.0;
        editConfig.min_accuracy_floor = p.min_accuracy_floor ?? 0;
        editConfig.tier0_only = p.tier0_only ?? p.optimize_tier0_only ?? false;
        editConfig.phase_order = p.phase_order ?? 'neurons_first';
        editConfig.context_size = p.context_size ?? 4;
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
          fitness_calculator: editConfig.fitness_calculator,
          fitness_weight_ce: editConfig.fitness_weight_ce,
          fitness_weight_acc: editConfig.fitness_weight_acc,
          min_accuracy_floor: editConfig.min_accuracy_floor,
          tier0_only: editConfig.tier0_only,
          phase_order: editConfig.phase_order,
          context_size: editConfig.context_size,
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
  // Experiment Editing (for pending experiments)
  // Note: Experiments are now stored in the DB, not in flow.config
  // =========================================================================

  function canEditExperiment(index: number): boolean {
    if (!flow) return false;
    // Can always edit if flow is pending, queued, or failed
    if (flow.status === 'pending' || flow.status === 'queued' || flow.status === 'failed') return true;
    // Can't edit completed or cancelled flows
    if (flow.status !== 'running') return false;

    // For running flows, check if this experiment has started
    const exp = experiments[index];
    if (!exp) return false;
    return exp.status === 'pending';
  }

  function startEditExperiment(index: number) {
    if (!flow || !canEditExperiment(index)) return;
    const exp = experiments[index];
    if (!exp) return;
    editingExpIndex = index;
    // Derive experiment_type and optimize_* from phase_type
    const isGa = exp.phase_type?.startsWith('ga') ?? true;
    editingExp = {
      name: exp.name,
      experiment_type: isGa ? 'ga' : 'ts',
      optimize_bits: exp.phase_type?.includes('bits') ?? false,
      optimize_neurons: exp.phase_type?.includes('neurons') ?? true,
      optimize_connections: exp.phase_type?.includes('connections') ?? false
    };
  }

  function cancelEditExperiment() {
    editingExpIndex = null;
    editingExp = null;
  }

  async function saveExperiment() {
    // TODO: Implement experiment update via PATCH /api/experiments/:id
    // For now, experiment editing is not supported after creation
    error = 'Experiment editing not yet implemented - delete and re-add instead';
    editingExpIndex = null;
    editingExp = null;
  }

  async function deleteExperiment(index: number) {
    if (!flow || !canEditExperiment(index)) return;
    const exp = experiments[index];
    if (!exp) return;
    if (!confirm(`Delete "${exp.name}"? This cannot be undone.`)) return;

    saving = true;
    try {
      // TODO: Add DELETE /api/experiments/:id endpoint
      // For now, just show a message
      error = 'Experiment deletion not yet implemented';
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
      // Call the new dedicated endpoint to add experiment to the experiments table
      const res = await fetch(`/api/flows/${flowId}/experiments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          experiment: {
            name: newPhase.name || `Experiment ${experiments.length + 1}`,
            experiment_type: newPhase.experiment_type,
            optimize_bits: newPhase.optimize_bits,
            optimize_neurons: newPhase.optimize_neurons,
            optimize_connections: newPhase.optimize_connections,
            params: {}
          }
        })
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.error || 'Failed to add experiment');
      }

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

  // Start editing flow name
  function startEditName() {
    if (!flow) return;
    editedName = flow.name;
    editingName = true;
  }

  // Cancel editing flow name
  function cancelEditName() {
    editingName = false;
    editedName = '';
  }

  // Save flow name
  async function saveFlowName() {
    if (!flow || !editedName.trim()) return;
    saving = true;

    try {
      const response = await fetch(`/api/flows/${flow.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: editedName.trim() })
      });

      if (!response.ok) throw new Error('Failed to rename flow');

      editingName = false;
      editedName = '';
      await loadFlow();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to rename';
    } finally {
      saving = false;
    }
  }

  // Duplicate flow
  async function duplicateFlow() {
    if (!flow) return;
    duplicating = true;

    try {
      // Convert experiments to ExperimentSpec format for the new API
      // Experiments are now passed separately, not in config
      const experimentSpecs = experiments.map(exp => ({
        name: exp.name,
        experiment_type: exp.phase_type?.startsWith('ga') ? 'ga' : 'ts',
        optimize_bits: exp.phase_type?.includes('bits') ?? false,
        optimize_neurons: exp.phase_type?.includes('neurons') ?? false,
        optimize_connections: exp.phase_type?.includes('connections') ?? false,
        params: {}
      }));

      const newName = `${flow.name} (copy)`;
      const response = await fetch('/api/flows', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newName,
          description: flow.description,
          config: flow.config,  // Just params, no experiments
          experiments: experimentSpecs  // Experiments passed separately
        })
      });

      if (!response.ok) throw new Error('Failed to duplicate flow');

      const newFlow = await response.json();
      // Navigate to the new flow
      window.location.href = `/flows/${newFlow.id}`;
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to duplicate';
    } finally {
      duplicating = false;
    }
  }

  async function updateFitnessCalculator(value: string) {
    if (!flow) return;
    saving = true;

    try {
      const updatedConfig = {
        ...flow.config,
        params: {
          ...flow.config.params,
          fitness_calculator: value
        }
      };

      const response = await fetch(`/api/flows/${flow.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: updatedConfig })
      });

      if (!response.ok) throw new Error('Failed to update fitness calculator');
      await loadFlow();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to update';
    } finally {
      saving = false;
    }
  }

  async function updateFitnessWeight(field: 'fitness_weight_ce' | 'fitness_weight_acc', value: number) {
    if (!flow) return;
    saving = true;

    try {
      const updatedConfig = {
        ...flow.config,
        params: {
          ...flow.config.params,
          [field]: value
        }
      };

      const response = await fetch(`/api/flows/${flow.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: updatedConfig })
      });

      if (!response.ok) throw new Error('Failed to update weight');
      await loadFlow();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to update';
    } finally {
      saving = false;
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

  async function stopFlow() {
    if (!flow) return;
    if (!confirm('Stop this flow? Current progress will be saved as a checkpoint.')) return;

    try {
      const response = await fetch(`/api/flows/${flow.id}/stop`, {
        method: 'POST'
      });
      if (response.ok) {
        await loadFlow();
      } else {
        const data = await response.json();
        error = data.error || 'Failed to stop flow';
      }
    } catch (e) {
      error = e instanceof Error ? e.message : 'Unknown error';
    }
  }

  async function restartFlow(fromBeginning: boolean = false) {
    if (!flow) return;
    const msg = fromBeginning
      ? 'Restart from the beginning? All progress will be lost.'
      : 'Restart from last checkpoint?';
    if (!confirm(msg)) return;

    try {
      const response = await fetch(`/api/flows/${flow.id}/restart`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ from_beginning: fromBeginning })
      });
      if (response.ok) {
        await loadFlow();
      } else {
        const data = await response.json();
        error = data.error || 'Failed to restart flow';
      }
    } catch (e) {
      error = e instanceof Error ? e.message : 'Unknown error';
    }
  }

  async function restartFromExperiment(index: number) {
    if (!flow) return;
    const expName = experiments[index]?.name || `Experiment ${index + 1}`;
    const msg = flow.status === 'running'
      ? `Stop current experiment and restart from "${expName}"? The current experiment will be cancelled and earlier experiments will be skipped.`
      : `Restart flow from "${expName}"? Earlier experiments will be skipped.`;
    if (!confirm(msg)) return;

    try {
      const response = await fetch(`/api/flows/${flow.id}/restart`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ from_beginning: true, start_from_experiment: index })
      });
      if (response.ok) {
        await loadFlow();
      } else {
        const data = await response.json();
        error = data.error || 'Failed to restart flow';
      }
    } catch (e) {
      error = e instanceof Error ? e.message : 'Unknown error';
    }
  }

  // Get final checkpoint for an experiment
  function getFinalCheckpoint(experimentId: number): Checkpoint | null {
    return checkpoints.find(c => c.experiment_id === experimentId && c.is_final) ?? null;
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

  // Get the link URL for an experiment based on its status
  function getExperimentLink(exp: Experiment): string | null {
    if (exp.status === 'running') {
      return '/';  // Running experiments link to live dashboard
    } else if (exp.status === 'completed') {
      return `/experiments/${exp.id}`;  // Completed experiments link to detail page
    }
    // Pending experiments are not clickable
    return null;
  }

  // Helper to get display experiments - experiments from DB are the source of truth
  // Sorted by sequence_order
  function getDisplayExperiments(_f: Flow | null, exps: Experiment[]): Experiment[] {
    // Sort by sequence_order
    return [...exps].sort((a, b) => (a.sequence_order ?? 0) - (b.sequence_order ?? 0));
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
        {#if editingName}
          <div class="name-edit">
            <input
              type="text"
              bind:value={editedName}
              class="name-input"
              on:keydown={(e) => e.key === 'Enter' && saveFlowName()}
              on:keydown={(e) => e.key === 'Escape' && cancelEditName()}
            />
            <button class="btn btn-sm btn-primary" on:click={saveFlowName} disabled={saving}>✓</button>
            <button class="btn btn-sm btn-secondary" on:click={cancelEditName}>✕</button>
          </div>
        {:else}
          <h1 class="flow-name-editable" on:click={startEditName} title="Click to rename">{flow.name}</h1>
        {/if}
        <span class="status-badge" style="background: {getStatusColor(flow.status)}">
          {flow.status}
        </span>
      </div>
      <div class="header-actions">
        <button class="btn btn-sm btn-secondary" on:click={duplicateFlow} disabled={duplicating} title="Duplicate flow">
          {duplicating ? '...' : '📋 Duplicate'}
        </button>
        {#if !editMode && flow.status === 'pending'}
          <button class="btn btn-primary" on:click={queueFlow}>
            Start
          </button>
        {/if}
        {#if !editMode && flow.status !== 'running' && flow.status !== 'queued'}
          <button class="btn btn-secondary" on:click={() => editMode = true}>
            Edit Config
          </button>
        {/if}
        {#if flow.status === 'queued'}
          <span class="queued-hint">Waiting for worker to pick up...</span>
        {/if}
        {#if flow.status === 'running'}
          <button class="btn btn-danger" on:click={stopFlow}>
            Stop
          </button>
        {/if}
        {#if flow.status === 'failed' || flow.status === 'cancelled'}
          <button class="btn btn-primary" on:click={() => restartFlow(false)}>
            Resume
          </button>
          <button class="btn btn-secondary" on:click={() => restartFlow(true)}>
            Restart from Beginning
          </button>
        {/if}
        {#if flow.status === 'completed'}
          <button class="btn btn-secondary" on:click={() => restartFlow(true)}>
            Run Again
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
              <span class="form-hint">Min accuracy threshold (0.003 = 0.3%). Below = rejected</span>
            </div>
            <div class="form-group">
              <label for="context_size">Context Size (N-gram)</label>
              <input type="number" id="context_size" bind:value={editConfig.context_size} min="1" max="16" />
              <span class="form-hint">Number of context tokens (e.g., 4 = 4-gram)</span>
            </div>
          </div>

          <div class="form-row">
            <div class="form-group checkbox-group">
              <label>
                <input type="checkbox" bind:checked={editConfig.tier0_only} />
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
            <span class="param-value">{flow.config.params.neighbors_per_iter ?? flow.config.params.population_size ?? 50}</span>
          </div>
          <div class="param-item">
            <span class="param-label">Fitness %</span>
            <span class="param-value">{flow.config.params.fitness_percentile ?? 0.75}</span>
          </div>
          <div class="param-item">
            <span class="param-label">Tier0 Only</span>
            <span class="param-value">{(flow.config.params.optimize_tier0_only || flow.config.params.tier0_only) ? 'Yes' : 'No'}</span>
          </div>
          <div class="param-group full-width">
            <span class="param-group-label">Fitness</span>
            <div class="param-group-items">
              <div class="param-group-item">
                <span class="param-label">Calculator</span>
                <select
                  class="inline-select"
                  value={flow.config.params.fitness_calculator ?? 'normalized'}
                  on:change={(e) => updateFitnessCalculator(e.currentTarget.value)}
                  disabled={saving}
                >
                  <option value="normalized">Normalized</option>
                  <option value="normalized_harmonic">Normalized Harmonic</option>
                  <option value="harmonic_rank">Harmonic Rank</option>
                  <option value="ce">CE Only</option>
                </select>
              </div>
              <div class="param-group-item">
                <span class="param-label">CE Weight</span>
                <input
                  type="number"
                  class="inline-input"
                  value={flow.config.params.fitness_weight_ce ?? 1.0}
                  min="0"
                  max="10"
                  step="0.1"
                  on:change={(e) => updateFitnessWeight('fitness_weight_ce', parseFloat(e.currentTarget.value))}
                  disabled={saving}
                />
              </div>
              <div class="param-group-item">
                <span class="param-label">Acc Weight</span>
                <input
                  type="number"
                  class="inline-input"
                  value={flow.config.params.fitness_weight_acc ?? 1.0}
                  min="0"
                  max="10"
                  step="0.1"
                  on:change={(e) => updateFitnessWeight('fitness_weight_acc', parseFloat(e.currentTarget.value))}
                  disabled={saving}
                />
              </div>
              <div class="param-group-item">
                <span class="param-label">Acc Floor</span>
                <input
                  type="number"
                  class="inline-input"
                  value={flow.config.params.min_accuracy_floor ?? 0}
                  min="0"
                  max="0.1"
                  step="0.001"
                  on:change={(e) => updateFitnessWeight('min_accuracy_floor', parseFloat(e.currentTarget.value))}
                  disabled={saving}
                />
              </div>
            </div>
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
        <h2>Experiments ({experiments.length})</h2>
        {#if flow.status === 'pending' || flow.status === 'running' || flow.status === 'failed'}
          <button class="btn btn-sm btn-secondary" on:click={() => showAddPhase = true}>
            + Add Experiment
          </button>
        {/if}
      </div>

      {#if showAddPhase}
        <div class="add-phase-form">
          <h3>Add New Experiment</h3>
          <div class="edit-exp-form">
            <div class="form-row">
              <div class="form-group">
                <label>Name</label>
                <input type="text" bind:value={newPhase.name} placeholder="Extra Experiment: GA Neurons" />
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
                {saving ? 'Adding...' : 'Add Experiment'}
              </button>
            </div>
          </div>
        </div>
      {/if}

      <div class="experiments-table">
        <table>
          <thead>
            <tr>
              <th class="col-order">#</th>
              <th class="col-name">Name</th>
              <th class="col-type">Type</th>
              <th class="col-status">Status</th>
              <th class="col-ce">Best CE</th>
              <th class="col-acc">Best Acc</th>
              <th class="col-actions">Actions</th>
            </tr>
          </thead>
          <tbody>
            {#each displayExperiments as exp, i}
              {@const isRunning = exp.status === 'running'}
              {@const isCompleted = exp.status === 'completed'}
              {@const isPending = exp.status === 'pending'}
              {@const canEdit = canEditExperiment(i)}
              {@const isEditingName = editingExpIndex === i}
              {@const expType = exp.phase_type?.startsWith('ga') ? 'GA' : exp.phase_type?.startsWith('ts') ? 'TS' : '—'}
              {@const optimizeTarget = exp.phase_type?.includes('bits') ? 'Bits' : exp.phase_type?.includes('neurons') ? 'Neurons' : exp.phase_type?.includes('connections') ? 'Conn' : '—'}
              {@const expLink = getExperimentLink(exp)}
              <tr class:row-running={isRunning} class:row-completed={isCompleted} class:row-pending={isPending}>
                <td class="col-order">
                  <span class="order-badge" class:order-completed={isCompleted} class:order-running={isRunning}>
                    {#if isCompleted}✓{:else}{i + 1}{/if}
                  </span>
                </td>
                <td class="col-name">
                  {#if isEditingName && editingExp}
                    <div class="inline-edit">
                      <input
                        type="text"
                        bind:value={editingExp.name}
                        class="inline-name-input"
                        on:keydown={(e) => e.key === 'Enter' && saveExperiment()}
                        on:keydown={(e) => e.key === 'Escape' && cancelEditExperiment()}
                      />
                      <button class="btn-icon btn-save" title="Save" on:click={saveExperiment}>✓</button>
                      <button class="btn-icon" title="Cancel" on:click={cancelEditExperiment}>✕</button>
                    </div>
                  {:else}
                    <span
                      class="exp-name-cell"
                      class:editable={canEdit}
                      on:click={() => canEdit && startEditExperiment(i)}
                      on:keydown={(e) => e.key === 'Enter' && canEdit && startEditExperiment(i)}
                      role={canEdit ? 'button' : 'cell'}
                      tabindex={canEdit ? 0 : -1}
                      title={canEdit ? 'Click to rename' : ''}
                    >
                      {exp.name}
                      {#if isRunning}
                        <span class="live-badge"><span class="pulse"></span>Live</span>
                      {/if}
                    </span>
                  {/if}
                </td>
                <td class="col-type">
                  <span class="type-badge" class:type-ga={expType === 'GA'} class:type-ts={expType === 'TS'}>{expType}</span>
                  <span class="target-badge">{optimizeTarget}</span>
                </td>
                <td class="col-status">
                  <span class="status-pill" style="background: {getStatusColor(exp.status)}">{exp.status}</span>
                </td>
                <td class="col-ce mono">{formatCE(exp.best_ce)}</td>
                <td class="col-acc mono">{formatAccuracy(exp.best_accuracy)}</td>
                <td class="col-actions">
                  <div class="action-buttons">
                    {#if expLink}
                      <a href={expLink} class="btn-icon" title="View details">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                          <circle cx="12" cy="12" r="3"></circle>
                        </svg>
                      </a>
                    {/if}
                    {#if canEdit}
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
                    {#if (isCompleted || isRunning) && (flow.status === 'running' || flow.status === 'failed' || flow.status === 'cancelled' || flow.status === 'completed')}
                      <button class="btn-icon" title="Restart from here" on:click={() => restartFromExperiment(i)}>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                          <polyline points="1 4 1 10 7 10"></polyline>
                          <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path>
                        </svg>
                      </button>
                    {/if}
                  </div>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    </section>

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
              <a href="/" class="btn btn-secondary">View Iterations</a>
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
    font-size: 1rem;
    color: var(--accent-yellow, #f59e0b);
    font-style: italic;
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

  /* Editable flow name */
  .flow-name-editable {
    cursor: pointer;
    padding: 0.25rem 0.5rem;
    margin: -0.25rem -0.5rem;
    border-radius: 4px;
    transition: background-color 0.15s;
  }

  .flow-name-editable:hover {
    background-color: var(--bg-tertiary);
  }

  .name-edit {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .name-input {
    font-size: 1.25rem;
    font-weight: 600;
    padding: 0.25rem 0.5rem;
    border: 1px solid var(--accent-blue);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-width: 200px;
  }

  .name-input:focus {
    outline: none;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
  }

  .status-badge {
    font-size: 1rem;
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
    font-size: 1rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
  }

  .info-value {
    font-size: 1rem;
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
    font-size: 1rem;
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
    font-size: 1rem;
  }

  input[type="number"]:focus,
  input[type="text"]:focus,
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
    border-top: 1px solid var(--border);
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

  .param-item.full-width,
  .param-group.full-width {
    grid-column: 1 / -1;
  }

  .param-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 0.75rem;
    background: var(--bg-tertiary);
    border-radius: 6px;
    border: 1px solid var(--border);
  }

  .param-group-label {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
  }

  .param-group-items {
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
  }

  .param-group-item {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .param-group-item .param-label {
    font-size: 1rem;
  }

  .param-label {
    font-size: 1rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
  }

  .param-value {
    font-size: 1rem;
    color: var(--text-primary);
  }

  .param-value.mono {
    font-family: monospace;
    font-size: 1rem;
  }

  /* Experiments List */
  .experiments-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .experiment-item,
  .experiment-item-link {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.75rem 1rem;
  }

  .experiment-item-link {
    text-decoration: none;
    color: inherit;
    cursor: pointer;
    transition: border-color 0.15s, background-color 0.15s;
  }

  .experiment-item-link:hover {
    border-color: var(--accent-blue);
    background: rgba(59, 130, 246, 0.05);
  }

  .experiment-item-link.completed:hover {
    border-color: var(--accent-green);
    background: rgba(34, 197, 94, 0.05);
  }

  .experiment-item-link.pending {
    opacity: 0.6;
    cursor: default;
  }

  .experiment-item-link.pending:hover {
    border-color: var(--border);
    background: var(--bg-secondary);
  }

  .experiment-item-link.pending .view-arrow {
    opacity: 0.3;
  }

  .live-indicator {
    display: flex;
    align-items: center;
    gap: 0.375rem;
    font-size: 1rem;
    font-weight: 600;
    color: var(--accent-blue);
    text-transform: uppercase;
  }

  .view-arrow {
    font-size: 1.25rem;
    color: var(--text-tertiary);
    transition: transform 0.15s, color 0.15s;
  }

  .experiment-item-link:hover .view-arrow {
    transform: translateX(3px);
    color: var(--accent-blue);
  }

  .experiment-item-link.completed:hover .view-arrow {
    color: var(--accent-green);
  }

  .exp-order {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--bg-tertiary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
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
    font-size: 1rem;
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
    font-size: 1rem;
    color: var(--accent-blue);
    font-weight: 600;
  }

  .exp-tag {
    font-size: 1rem;
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
    font-size: 1rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
  }

  .metric-value {
    font-size: 1rem;
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
    font-size: 1rem;
  }

  .live-btn {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.375rem 0.75rem;
    background: var(--accent-blue);
    color: white;
    font-size: 1rem;
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

  /* Experiments Table */
  .experiments-table {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }

  .experiments-table table {
    width: 100%;
    border-collapse: collapse;
  }

  .experiments-table th,
  .experiments-table td {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }

  .experiments-table th {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-tertiary);
    text-align: center;
    text-transform: uppercase;
    background: var(--bg-tertiary);
  }

  .experiments-table td {
    font-size: 0.875rem;
    color: var(--text-primary);
  }

  .experiments-table tr:last-child td {
    border-bottom: none;
  }

  .experiments-table .col-order { width: 40px; text-align: center; }
  .experiments-table .col-name { min-width: 200px; text-align: left; }
  .experiments-table .col-type { width: 140px; white-space: nowrap; text-align: center; }
  .experiments-table .col-status { width: 100px; text-align: center; }
  .experiments-table .col-ce { width: 100px; text-align: right; }
  .experiments-table .col-acc { width: 100px; text-align: right; }
  .experiments-table .col-actions { width: 120px; text-align: center; }

  .experiments-table .mono {
    font-family: monospace;
  }

  .order-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--bg-tertiary);
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
  }

  .order-badge.order-completed {
    background: var(--accent-green);
    color: white;
  }

  .order-badge.order-running {
    background: var(--accent-blue);
    color: white;
  }

  .exp-name-cell {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
  }

  .exp-name-cell.editable {
    cursor: pointer;
    padding: 0.25rem 0.5rem;
    margin: -0.25rem -0.5rem;
    border-radius: 4px;
    transition: background-color 0.15s;
  }

  .exp-name-cell.editable:hover {
    background: var(--bg-tertiary);
  }

  .inline-edit {
    display: flex;
    align-items: center;
    gap: 0.25rem;
  }

  .inline-name-input {
    padding: 0.25rem 0.5rem;
    border: 1px solid var(--accent-blue);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 0.875rem;
    min-width: 180px;
  }

  .inline-name-input:focus {
    outline: none;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
  }

  .btn-save {
    background: var(--accent-green) !important;
    color: white !important;
  }

  .live-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.625rem;
    font-weight: 600;
    color: var(--accent-blue);
    text-transform: uppercase;
    background: rgba(59, 130, 246, 0.1);
    padding: 0.125rem 0.375rem;
    border-radius: 3px;
  }

  .type-badge {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.125rem 0.375rem;
    border-radius: 3px;
  }

  .type-badge.type-ga {
    background: rgba(59, 130, 246, 0.15);
    color: var(--accent-blue);
  }

  .type-badge.type-ts {
    background: rgba(16, 185, 129, 0.15);
    color: var(--accent-green);
  }

  .target-badge {
    font-size: 0.75rem;
    color: var(--text-tertiary);
    margin-left: 0.25rem;
  }

  .status-pill {
    display: inline-block;
    font-size: 0.625rem;
    font-weight: 600;
    padding: 0.25rem 0.5rem;
    border-radius: 9999px;
    color: white;
    text-transform: uppercase;
  }

  .action-buttons {
    display: flex;
    gap: 0.25rem;
    justify-content: flex-end;
  }

  .action-buttons a.btn-icon {
    text-decoration: none;
  }

  .row-running {
    background: rgba(59, 130, 246, 0.05);
  }

  .row-completed {
    /* subtle green tint */
  }

  .row-pending {
    opacity: 0.7;
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
    font-size: 1rem;
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

  .edit-exp-form input[type="text"],
  .edit-exp-form select {
    padding: 0.375rem 0.5rem;
    border: 1px solid var(--border);
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
    font-size: 1rem;
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
    font-size: 1rem;
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

  /* Inline editable select in params */
  .param-editable {
    display: flex;
    align-items: center;
  }

  .inline-select {
    padding: 0.25rem 0.5rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 1rem;
    cursor: pointer;
    transition: border-color 0.15s;
  }

  .inline-select:hover:not(:disabled) {
    border-color: var(--accent-blue);
  }

  .inline-select:focus {
    outline: none;
    border-color: var(--accent-blue);
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
  }

  .inline-select:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .inline-input {
    width: 70px;
    padding: 0.25rem 0.5rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 1rem;
    text-align: right;
    transition: border-color 0.15s;
  }

  .inline-input:hover:not(:disabled) {
    border-color: var(--accent-blue);
  }

  .inline-input:focus {
    outline: none;
    border-color: var(--accent-blue);
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
  }

  .inline-input:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
</style>
