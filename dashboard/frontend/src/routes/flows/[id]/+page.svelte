<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { page } from '$app/stores';
  import type { Flow, Experiment, Checkpoint, ValidationSummary } from '$lib/types';
  import { formatDate } from '$lib/dateFormat';
  import { currentFlow, flows } from '$lib/stores';

  // Note: With normalized design, experiments come from the experiments table (via DB),
  // NOT from flow.config.experiments. The `experiments` array is fetched separately
  // via /api/flows/${flowId}/experiments and used directly.

  /** Format tier_config which may be a string or array of tuples */
  function formatTierConfig(tierConfig: unknown): string {
    if (typeof tierConfig === 'string') return tierConfig;
    if (Array.isArray(tierConfig)) {
      return tierConfig.map((t: (number|string|boolean)[]) => {
        const base = `${t[0] ?? 'rest'},${t[1]},${t[2]}`;
        return t.length > 3 ? `${base},${t[3]}` : base;
      }).join('; ');
    }
    return String(tierConfig);
  }

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

  // Refresh experiments and validations without full page reload
  async function refreshExperiments() {
    try {
      const [expsRes, validationsRes] = await Promise.all([
        fetch(`/api/flows/${flowId}/experiments`),
        fetch(`/api/flows/${flowId}/validations`)
      ]);
      if (expsRes.ok) {
        const expsData = await expsRes.json();
        experiments = Array.isArray(expsData) ? expsData : [];
      }
      if (validationsRes.ok) {
        const validationsData = await validationsRes.json();
        validationSummaries = Array.isArray(validationsData) ? validationsData : [];
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
  let validationSummaries: ValidationSummary[] = [];
  let loading = true;
  let error: string | null = null;
  let saving = false;
  let editMode = false;

  // Validation chart tooltip state
  let validationTooltip: {
    x: number;
    y: number;
    label: string;
    genomeType: string;
    ce: number;
    accuracy: number;
  } | null = null;

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
      const [flowRes, expsRes, checkpointsRes, validationsRes] = await Promise.all([
        fetch(`/api/flows/${flowId}`),
        fetch(`/api/flows/${flowId}/experiments`),
        fetch(`/api/checkpoints`),
        fetch(`/api/flows/${flowId}/validations`)
      ]);

      if (!flowRes.ok) throw new Error('Flow not found');

      flow = await flowRes.json();
      // Ensure experiments is always an array (defensive against API returning {})
      const expsData = await expsRes.json();
      experiments = Array.isArray(expsData) ? expsData : [];
      // Ensure checkpoints is always an array
      const checkpointsData = checkpointsRes.ok ? await checkpointsRes.json() : [];
      checkpoints = Array.isArray(checkpointsData) ? checkpointsData : [];
      // Ensure validationSummaries is always an array
      const validationsData = validationsRes.ok ? await validationsRes.json() : [];
      validationSummaries = Array.isArray(validationsData) ? validationsData : [];

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
        editConfig.phase_order = p.phase_order ?? 'neurons_first';
        editConfig.context_size = p.context_size ?? 4;
        if (p.tier_config) {
          // Handle both string and array formats
          if (typeof p.tier_config === 'string') {
            editConfig.tier_config = p.tier_config;
          } else {
            editConfig.tier_config = p.tier_config
              .map((t: (number|string|boolean)[]) => {
                const base = `${t[0] ?? 'rest'},${t[1]},${t[2]}`;
                return t.length > 3 ? `${base},${t[3]}` : base;
              })
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
      // Parse tier_config string (supports 3 or 4 fields: clusters,neurons,bits[,optimize])
      let tier_config = null;
      if (editConfig.tier_config.trim()) {
        tier_config = editConfig.tier_config.split(';').map(tier => {
          const parts = tier.trim().split(',').map(p => p.trim());
          const entry: (number | null | boolean)[] = [
            parts[0] === 'rest' ? null : parseInt(parts[0]),
            parseInt(parts[1]),
            parseInt(parts[2])
          ];
          if (parts.length > 3) {
            entry.push(parts[3] === 'true');
          }
          return entry;
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

  // Get default iterations based on experiment type (GA vs TS)
  function getDefaultIterations(expType: string): number {
    return expType === 'GA'
      ? (flow?.config.params.ga_generations ?? 250)
      : (flow?.config.params.ts_iterations ?? 250);
  }

  // Update experiment's max_iterations
  async function updateExperimentIterations(expId: number, iterations: number) {
    if (iterations < 10 || iterations > 10000) return;

    try {
      const response = await fetch(`/api/experiments/${expId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_iterations: iterations })
      });

      if (!response.ok) throw new Error('Failed to update iterations');

      // Refresh experiments to show updated value
      await refreshExperiments();
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to update iterations';
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

  // Delete flow state
  let deleting = false;

  async function deleteFlow() {
    if (!flow) return;
    if (!confirm(`Delete flow "${flow.name}"? This will delete all experiments, iterations, and checkpoints. This cannot be undone.`)) return;

    deleting = true;
    try {
      const response = await fetch(`/api/flows/${flow.id}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.error || 'Failed to delete flow');
      }

      // Navigate back to flows list
      window.location.href = '/flows';
    } catch (e) {
      error = e instanceof Error ? e.message : 'Failed to delete';
    } finally {
      deleting = false;
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

  async function updateFitnessWeight(field: 'fitness_weight_ce' | 'fitness_weight_acc' | 'min_accuracy_floor', value: number) {
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

  // Get the link URL for an experiment - all experiments are viewable
  function getExperimentLink(exp: Experiment): string {
    return `/experiments/${exp.id}`;
  }

  // Helper to get display experiments - experiments from DB are the source of truth
  // Sorted by sequence_order
  function getDisplayExperiments(_f: Flow | null, exps: Experiment[]): Experiment[] {
    // Sort by sequence_order
    return [...exps].sort((a, b) => (a.sequence_order ?? 0) - (b.sequence_order ?? 0));
  }

  // =========================================================================
  // Validation Progression Chart
  // =========================================================================

  // Build chart data points from validation summaries
  // Group by experiment and validation_point, creating x-axis labels like "Init", "Exp1", "Exp2", etc.
  $: validationChartData = (() => {
    if (validationSummaries.length === 0) return [];

    // Create a map of experiment_id -> sequence_order for labeling
    const expOrderMap = new Map<number, number>();
    displayExperiments.forEach((exp, idx) => {
      expOrderMap.set(exp.id, idx);
    });

    // Group validations by (experiment_id, validation_point)
    const points: {
      label: string;
      sortKey: number;
      validations: { genomeType: string; ce: number; accuracy: number }[];
    }[] = [];

    // Group by experiment and point
    const grouped = new Map<string, ValidationSummary[]>();
    for (const v of validationSummaries) {
      const key = `${v.experiment_id}-${v.validation_point}`;
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key)!.push(v);
    }

    // Convert to chart points
    for (const [key, validations] of grouped) {
      const [expIdStr, point] = key.split('-');
      const expId = parseInt(expIdStr);
      const order = expOrderMap.get(expId) ?? 0;

      // Create label: "Init" for first init, "Exp N" for finals
      let label: string;
      let sortKey: number;
      if (point === 'init') {
        label = order === 0 ? 'Init' : `Exp${order} Init`;
        sortKey = order * 2;  // init comes before final
      } else {
        label = `Exp${order + 1}`;
        sortKey = order * 2 + 1;
      }

      points.push({
        label,
        sortKey,
        validations: validations.map(v => ({
          genomeType: v.genome_type,
          ce: v.ce,
          accuracy: v.accuracy
        }))
      });
    }

    // Sort by sortKey
    return points.sort((a, b) => a.sortKey - b.sortKey);
  })();

  // Chart dimensions
  const valChartPadding = { top: 30, right: 40, bottom: 50, left: 60 };
  const valChartSvgWidth = 700;
  const valChartSvgHeight = 280;
  $: valChartWidth = valChartSvgWidth - valChartPadding.left - valChartPadding.right;
  $: valChartHeight = valChartSvgHeight - valChartPadding.top - valChartPadding.bottom;

  // Compute CE range for y-axis
  $: allCeValues = validationChartData.flatMap(p => p.validations.map(v => v.ce));
  $: valCeMin = allCeValues.length > 0 ? Math.min(...allCeValues) : 0;
  $: valCeMax = allCeValues.length > 0 ? Math.max(...allCeValues) : 1;
  $: valCeRange = valCeMax - valCeMin || 0.001;
  // Add 10% padding to range
  $: valCeMinPadded = valCeMin - valCeRange * 0.1;
  $: valCeMaxPadded = valCeMax + valCeRange * 0.1;
  $: valCeRangePadded = valCeMaxPadded - valCeMinPadded;

  // Genome type colors and markers
  function getGenomeTypeColor(genomeType: string): string {
    switch (genomeType) {
      case 'best_ce': return 'var(--accent-blue)';
      case 'best_acc': return 'var(--accent-green)';
      case 'best_fitness': return 'var(--accent-purple, #8b5cf6)';
      default: return 'var(--text-secondary)';
    }
  }

  function getGenomeTypeLabel(genomeType: string): string {
    switch (genomeType) {
      case 'best_ce': return 'Best CE';
      case 'best_acc': return 'Best Acc';
      case 'best_fitness': return 'Best Fitness';
      default: return genomeType;
    }
  }

  // Helper to compute polyline points for a genome type
  function getPolylinePoints(genomeType: string): string {
    const points: { x: number; y: number }[] = [];
    validationChartData.forEach((p, i) => {
      const v = p.validations.find(val => val.genomeType === genomeType);
      if (v) {
        const x = valChartPadding.left + (i / Math.max(validationChartData.length - 1, 1)) * valChartWidth;
        const y = valChartPadding.top + valChartHeight - ((v.ce - valCeMinPadded) / valCeRangePadded) * valChartHeight;
        points.push({ x, y });
      }
    });
    return points.map(p => `${p.x},${p.y}`).join(' ');
  }

  // Check if we have enough points for a polyline
  function hasMultiplePoints(genomeType: string): boolean {
    let count = 0;
    for (const p of validationChartData) {
      if (p.validations.some(v => v.genomeType === genomeType)) count++;
      if (count > 1) return true;
    }
    return false;
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
        {#if flow.status !== 'running' && flow.status !== 'queued'}
          <button class="btn btn-danger" on:click={deleteFlow} disabled={deleting} title="Delete flow">
            {deleting ? 'Deleting...' : 'Delete'}
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

    <!-- Validation Progression Chart -->
    {#if validationChartData.length > 0}
      <section class="section">
        <div class="card">
          <div class="card-header">
            <span class="card-title">Validation Progression</span>
            <div class="val-chart-legend">
              <span class="legend-item"><span class="legend-marker best-ce"></span> Best CE</span>
              <span class="legend-item"><span class="legend-marker best-acc"></span> Best Acc</span>
              <span class="legend-item"><span class="legend-marker best-fitness"></span> Best Fitness</span>
            </div>
          </div>
          <div class="val-chart-container">
            <svg viewBox="-20 0 {valChartSvgWidth + 20} {valChartSvgHeight}" class="val-chart">
              <!-- Y-axis labels (CE) -->
              <text x={valChartPadding.left - 8} y={valChartPadding.top + 4} text-anchor="end" class="axis-label">{valCeMaxPadded.toFixed(2)}</text>
              <text x={valChartPadding.left - 8} y={valChartPadding.top + valChartHeight} text-anchor="end" class="axis-label">{valCeMinPadded.toFixed(2)}</text>
              <text x={valChartPadding.left - 8} y={valChartPadding.top + valChartHeight / 2} text-anchor="end" class="axis-label">{((valCeMaxPadded + valCeMinPadded) / 2).toFixed(2)}</text>

              <!-- Y-axis title -->
              <text x={-8} y={valChartPadding.top + valChartHeight / 2} text-anchor="middle" class="axis-title" transform="rotate(-90, -8, {valChartPadding.top + valChartHeight / 2})">CE (Loss)</text>

              <!-- Grid lines -->
              <line x1={valChartPadding.left} y1={valChartPadding.top} x2={valChartPadding.left + valChartWidth} y2={valChartPadding.top} stroke="var(--border)" stroke-dasharray="4" />
              <line x1={valChartPadding.left} y1={valChartPadding.top + valChartHeight} x2={valChartPadding.left + valChartWidth} y2={valChartPadding.top + valChartHeight} stroke="var(--border)" stroke-dasharray="4" />
              <line x1={valChartPadding.left} y1={valChartPadding.top + valChartHeight / 2} x2={valChartPadding.left + valChartWidth} y2={valChartPadding.top + valChartHeight / 2} stroke="var(--border)" stroke-dasharray="2" opacity="0.5" />

              <!-- X-axis line -->
              <line x1={valChartPadding.left} y1={valChartPadding.top + valChartHeight} x2={valChartPadding.left + valChartWidth} y2={valChartPadding.top + valChartHeight} stroke="var(--text-tertiary)" />

              <!-- X-axis labels -->
              {#each validationChartData as point, i}
                {@const x = valChartPadding.left + (i / Math.max(validationChartData.length - 1, 1)) * valChartWidth}
                <line x1={x} y1={valChartPadding.top + valChartHeight} x2={x} y2={valChartPadding.top + valChartHeight + 5} stroke="var(--text-tertiary)" />
                <text x={x} y={valChartPadding.top + valChartHeight + 20} text-anchor="middle" class="axis-label x-label">{point.label}</text>
              {/each}

              <!-- Connect lines for each genome type -->
              {#each ['best_ce', 'best_acc', 'best_fitness'] as genomeType}
                {#if hasMultiplePoints(genomeType)}
                  <polyline
                    fill="none"
                    stroke={getGenomeTypeColor(genomeType)}
                    stroke-width="2"
                    stroke-opacity="0.5"
                    points={getPolylinePoints(genomeType)}
                  />
                {/if}
              {/each}

              <!-- Data points (offset horizontally when multiple markers at same point) -->
              {#each validationChartData as point, i}
                {@const baseX = valChartPadding.left + (i / Math.max(validationChartData.length - 1, 1)) * valChartWidth}
                {@const numValidations = point.validations.length}
                {@const offsetStep = numValidations > 1 ? 12 : 0}
                {#each point.validations as v, vi}
                  {@const xOffset = numValidations > 1 ? (vi - (numValidations - 1) / 2) * offsetStep : 0}
                  {@const x = baseX + xOffset}
                  {@const y = valChartPadding.top + valChartHeight - ((v.ce - valCeMinPadded) / valCeRangePadded) * valChartHeight}
                  <!-- Marker based on genome type -->
                  {#if v.genomeType === 'best_ce'}
                    <circle
                      cx={x} cy={y} r="6"
                      fill={getGenomeTypeColor(v.genomeType)}
                      stroke="white" stroke-width="2"
                      class="data-point"
                      role="button"
                      tabindex="-1"
                      on:mouseenter={() => validationTooltip = { x, y, label: point.label, genomeType: v.genomeType, ce: v.ce, accuracy: v.accuracy }}
                      on:mouseleave={() => validationTooltip = null}
                    />
                  {:else if v.genomeType === 'best_acc'}
                    <rect
                      x={x - 5} y={y - 5} width="10" height="10"
                      fill={getGenomeTypeColor(v.genomeType)}
                      stroke="white" stroke-width="2"
                      class="data-point"
                      role="button"
                      tabindex="-1"
                      on:mouseenter={() => validationTooltip = { x, y, label: point.label, genomeType: v.genomeType, ce: v.ce, accuracy: v.accuracy }}
                      on:mouseleave={() => validationTooltip = null}
                    />
                  {:else}
                    <polygon
                      points="{x},{y - 7} {x + 6},{y + 4} {x - 6},{y + 4}"
                      fill={getGenomeTypeColor(v.genomeType)}
                      stroke="white" stroke-width="2"
                      class="data-point"
                      role="button"
                      tabindex="-1"
                      on:mouseenter={() => validationTooltip = { x, y, label: point.label, genomeType: v.genomeType, ce: v.ce, accuracy: v.accuracy }}
                      on:mouseleave={() => validationTooltip = null}
                    />
                  {/if}
                {/each}
              {/each}

              <!-- Tooltip -->
              {#if validationTooltip}
                <g class="tooltip-group" transform="translate({validationTooltip.x}, {validationTooltip.y - 70})">
                  <rect x="-100" y="-10" width="200" height="70" rx="4" fill="var(--bg-secondary)" stroke="var(--border)" />
                  <text x="0" y="10" text-anchor="middle" class="tooltip-label">{validationTooltip.label}</text>
                  <text x="0" y="30" text-anchor="middle" class="tooltip-type" fill={getGenomeTypeColor(validationTooltip.genomeType)}>{getGenomeTypeLabel(validationTooltip.genomeType)}</text>
                  <text x="0" y="50" text-anchor="middle" class="tooltip-value">CE: {validationTooltip.ce.toFixed(4)} | Acc: {(validationTooltip.accuracy * 100).toFixed(2)}%</text>
                </g>
              {/if}
            </svg>
          </div>
        </div>
      </section>
    {/if}

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

          <div class="form-group full-width">
            <label for="tier_config">Tier Config</label>
            <input type="text" id="tier_config" bind:value={editConfig.tier_config}
                   placeholder="100,15,20,true;400,10,12,false;rest,5,8,false" />
            <span class="form-hint">Format: clusters,neurons,bits,optimize per tier (semicolon separated, optimize=true/false)</span>
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
                {formatTierConfig(flow.config.params.tier_config)}
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
              <th class="col-iters">Iterations</th>
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
                <td class="col-name clickable-cell">
                  <a href={expLink} class="cell-link">
                    {exp.name}
                    {#if isRunning}
                      <span class="live-badge"><span class="pulse"></span>Live</span>
                    {/if}
                  </a>
                </td>
                <td class="col-type">
                  <span class="type-badge" class:type-ga={expType === 'GA'} class:type-ts={expType === 'TS'}>{expType}</span>
                  <span class="target-badge">{optimizeTarget}</span>
                </td>
                <td class="col-iters">
                  {#if isPending && canEdit}
                    <input
                      type="number"
                      class="iters-input"
                      value={exp.max_iterations ?? getDefaultIterations(expType)}
                      min="10"
                      max="10000"
                      on:change={(e) => updateExperimentIterations(exp.id, parseInt(e.currentTarget.value))}
                    />
                  {:else if isRunning}
                    <span class="iters-progress">{exp.current_iteration ?? 0}/{exp.max_iterations ?? '?'}</span>
                  {:else}
                    <span class="mono">{exp.current_iteration ?? exp.max_iterations ?? '—'}</span>
                  {/if}
                </td>
                <td class="col-status">
                  <span class="status-pill" style="background: {getStatusColor(exp.status)}">{exp.status}</span>
                </td>
                <td class="col-ce mono">{formatCE(exp.best_ce)}</td>
                <td class="col-acc mono">{formatAccuracy(exp.best_accuracy)}</td>
                <td class="col-actions">
                  <div class="action-buttons">
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
                    {#if isRunning}
                      <button class="btn-icon btn-danger" title="Stop" on:click={stopFlow}>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                          <rect x="6" y="6" width="12" height="12" rx="2"></rect>
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
  .experiments-table .col-iters { width: 100px; text-align: center; }
  .experiments-table .col-status { width: 100px; text-align: center; }
  .experiments-table .col-ce { width: 100px; text-align: right; }
  .experiments-table .col-acc { width: 100px; text-align: right; }
  .experiments-table .col-actions { width: 120px; text-align: center; }

  .iters-input {
    width: 70px;
    padding: 0.25rem 0.5rem;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 0.875rem;
    text-align: center;
  }

  .iters-input:hover {
    border-color: var(--accent-blue);
  }

  .iters-input:focus {
    outline: none;
    border-color: var(--accent-blue);
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
  }

  .iters-progress {
    font-size: 0.875rem;
    font-family: monospace;
    color: var(--accent-blue);
  }

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

  .clickable-cell {
    padding: 0 !important;
  }

  .cell-link {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    width: 100%;
    height: 100%;
    padding: 0.75rem 1rem;
    color: var(--text-primary);
    text-decoration: none;
    transition: background-color 0.15s, color 0.15s;
  }

  .cell-link:hover {
    background: var(--bg-tertiary);
    color: var(--accent-blue);
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

  /* Validation Progression Chart */
  .card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
    background: var(--bg-tertiary);
  }

  .card-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-primary);
  }

  .val-chart-legend {
    display: flex;
    gap: 1rem;
    font-size: 1rem;
    color: var(--text-secondary);
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 0.375rem;
  }

  .legend-marker {
    width: 12px;
    height: 12px;
    border-radius: 2px;
  }

  .legend-marker.best-ce {
    background: var(--accent-blue);
    border-radius: 50%;
  }

  .legend-marker.best-acc {
    background: var(--accent-green);
  }

  .legend-marker.best-fitness {
    background: var(--accent-purple, #8b5cf6);
    clip-path: polygon(50% 0%, 100% 100%, 0% 100%);
  }

  .val-chart-container {
    padding: 1rem;
  }

  .val-chart {
    width: 100%;
    height: 260px;
  }

  .val-chart .axis-label {
    font-size: 1rem;
    fill: var(--text-tertiary);
  }

  .val-chart .axis-label.x-label {
    font-size: 1rem;
  }

  .val-chart .axis-title {
    font-size: 1rem;
    fill: var(--text-secondary);
    font-weight: 500;
  }

  .val-chart .data-point {
    cursor: pointer;
  }

  .val-chart .tooltip-group {
    pointer-events: none;
  }

  .val-chart .tooltip-label {
    font-size: 1rem;
    font-weight: 600;
    fill: var(--text-primary);
  }

  .val-chart .tooltip-type {
    font-size: 1rem;
    font-weight: 500;
  }

  .val-chart .tooltip-value {
    font-size: 1rem;
    fill: var(--text-secondary);
  }

  .val-chart .tooltip-group rect {
    filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
  }
</style>
