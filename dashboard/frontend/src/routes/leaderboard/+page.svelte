<script lang="ts">
	import { onMount } from 'svelte';
	import type { BestGenome } from '$lib/types';
	import { formatDate } from '$lib/dateFormat';

	let genomes: BestGenome[] = [];
	let loading = true;
	let error: string | null = null;

	// Filters
	let taskType = '';
	let stage = '';
	let metric = '';

	const taskTypes = [
		{ value: '', label: 'All Tasks' },
		{ value: 'lm', label: 'Language Model' },
		{ value: 'ids', label: 'IDS' },
	];

	const stages = [
		{ value: '', label: 'All Stages' },
		{ value: 'stage_0', label: 'Stage 0' },
		{ value: 'stage_1', label: 'Stage 1' },
		{ value: 'stage_2', label: 'Stage 2' },
		{ value: 'combined', label: 'Combined' },
	];

	const metrics = [
		{ value: '', label: 'All Metrics' },
		{ value: 'ce', label: 'Cross-Entropy' },
		{ value: 'accuracy', label: 'Accuracy' },
		{ value: 'f1_macro', label: 'F1-Macro' },
	];

	async function fetchGenomes() {
		loading = true;
		error = null;
		try {
			const params = new URLSearchParams();
			if (taskType) params.set('task_type', taskType);
			if (stage) params.set('stage', stage);
			if (metric) params.set('metric', metric);
			params.set('limit', '100');

			const response = await fetch(`/api/best-genomes?${params}`);
			if (!response.ok) throw new Error('Failed to fetch leaderboard');
			genomes = await response.json();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Unknown error';
		} finally {
			loading = false;
		}
	}

	onMount(fetchGenomes);

	function handleFilterChange() {
		fetchGenomes();
	}

	// Expanded row tracking
	let expandedId: number | null = null;

	function toggleExpand(id: number) {
		expandedId = expandedId === id ? null : id;
	}

	function formatMetric(value: number | null, decimals: number = 4): string {
		if (value === null || value === undefined) return '-';
		return value.toFixed(decimals);
	}

	function formatPercent(value: number | null): string {
		if (value === null || value === undefined) return '-';
		return (value * 100).toFixed(2) + '%';
	}

	function stageLabel(s: string): string {
		switch (s) {
			case 'stage_0': return 'S0';
			case 'stage_1': return 'S1';
			case 'stage_2': return 'S2';
			case 'combined': return 'Comb';
			default: return s;
		}
	}

	function metricLabel(m: string): string {
		switch (m) {
			case 'ce': return 'CE';
			case 'accuracy': return 'Acc';
			case 'f1_macro': return 'F1';
			default: return m;
		}
	}
</script>

<div class="container">
	<div class="page-header">
		<h1>Best Genomes Leaderboard</h1>
		<p class="subtitle">Ranked genomes across all experiments</p>
	</div>

	<div class="filters">
		<div class="filter-group">
			<label for="task-type">Task</label>
			<select id="task-type" bind:value={taskType} on:change={handleFilterChange}>
				{#each taskTypes as t}
					<option value={t.value}>{t.label}</option>
				{/each}
			</select>
		</div>
		<div class="filter-group">
			<label for="stage">Stage</label>
			<select id="stage" bind:value={stage} on:change={handleFilterChange}>
				{#each stages as s}
					<option value={s.value}>{s.label}</option>
				{/each}
			</select>
		</div>
		<div class="filter-group">
			<label for="metric">Metric</label>
			<select id="metric" bind:value={metric} on:change={handleFilterChange}>
				{#each metrics as m}
					<option value={m.value}>{m.label}</option>
				{/each}
			</select>
		</div>
	</div>

	{#if loading}
		<div class="loading">Loading leaderboard...</div>
	{:else if error}
		<div class="error">{error}</div>
	{:else if genomes.length === 0}
		<div class="empty">
			<p>No genomes in the leaderboard yet.</p>
			<p class="hint">Run experiments with validation to populate the leaderboard.</p>
		</div>
	{:else}
		<div class="leaderboard-table">
			<table>
				<thead>
					<tr>
						<th class="col-rank">Rank</th>
						<th class="col-task">Task</th>
						<th class="col-stage">Stage</th>
						<th class="col-metric">Metric</th>
						<th class="col-ce">CE</th>
						<th class="col-acc">Accuracy</th>
						<th class="col-f1">F1</th>
						<th class="col-fpr">FPR</th>
						<th class="col-arch">Neurons</th>
						<th class="col-arch">Clusters</th>
						<th class="col-flow">Flow</th>
						<th class="col-hf">HF</th>
					</tr>
				</thead>
				<tbody>
					{#each genomes as genome}
						<tr
							class="genome-row"
							class:expanded={expandedId === genome.id}
							on:click={() => toggleExpand(genome.id)}
						>
							<td class="col-rank">
								<span class="rank-badge" class:rank-1={genome.rank === 1} class:rank-2={genome.rank === 2} class:rank-3={genome.rank === 3}>
									{genome.rank ?? '-'}
								</span>
							</td>
							<td class="col-task">
								<span class="tag tag-{genome.task_type}">{genome.task_type.toUpperCase()}</span>
							</td>
							<td class="col-stage">{stageLabel(genome.stage)}</td>
							<td class="col-metric">{metricLabel(genome.metric)}</td>
							<td class="col-ce mono">{formatMetric(genome.ce)}</td>
							<td class="col-acc mono">{formatPercent(genome.accuracy)}</td>
							<td class="col-f1 mono">{genome.f1_macro !== null ? formatPercent(genome.f1_macro) : '-'}</td>
							<td class="col-fpr mono">{genome.fpr !== null ? formatPercent(genome.fpr) : '-'}</td>
							<td class="col-arch mono">{genome.total_neurons ?? '-'}</td>
							<td class="col-arch mono">{genome.total_clusters ?? '-'}</td>
							<td class="col-flow">
								{#if genome.flow_id}
									<a href="/flows/{genome.flow_id}" class="flow-link" on:click|stopPropagation>F{genome.flow_id}</a>
								{:else}
									-
								{/if}
							</td>
							<td class="col-hf">
								{#if genome.hf_repo_id}
									<span class="hf-badge">HF</span>
								{:else}
									-
								{/if}
							</td>
						</tr>
						{#if expandedId === genome.id}
							<tr class="detail-row">
								<td colspan="12">
									<div class="detail-content">
										<div class="detail-grid">
											<div class="detail-item">
												<span class="detail-label">Genome Hash</span>
												<span class="detail-value mono">{genome.genome_hash}</span>
											</div>
											<div class="detail-item">
												<span class="detail-label">Architecture</span>
												<span class="detail-value">{genome.architecture_type_str ?? 'unknown'}</span>
											</div>
											{#if genome.tiers_json}
												<div class="detail-item">
													<span class="detail-label">Tiers</span>
													<span class="detail-value mono">{genome.tiers_json}</span>
												</div>
											{/if}
											<div class="detail-item">
												<span class="detail-label">Experiment</span>
												<span class="detail-value">
													{#if genome.experiment_id}
														<a href="/experiments/{genome.experiment_id}" on:click|stopPropagation>E{genome.experiment_id}</a>
													{:else}
														-
													{/if}
												</span>
											</div>
											<div class="detail-item">
												<span class="detail-label">Created</span>
												<span class="detail-value">{formatDate(genome.created_at)}</span>
											</div>
											{#if genome.hf_repo_id}
												<div class="detail-item">
													<span class="detail-label">HuggingFace</span>
													<span class="detail-value">{genome.hf_repo_id}</span>
												</div>
											{/if}
										</div>
									</div>
								</td>
							</tr>
						{/if}
					{/each}
				</tbody>
			</table>
		</div>
		<div class="count">{genomes.length} genome{genomes.length !== 1 ? 's' : ''}</div>
	{/if}
</div>

<style>
	.page-header {
		margin-bottom: 1.5rem;
	}

	.page-header h1 {
		font-size: 1.5rem;
		font-weight: 600;
		color: var(--text-primary);
		margin-bottom: 0.25rem;
	}

	.subtitle {
		color: var(--text-secondary);
		font-size: 1rem;
	}

	.filters {
		display: flex;
		gap: 1rem;
		margin-bottom: 1.5rem;
		flex-wrap: wrap;
	}

	.filter-group {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.filter-group label {
		font-size: 1rem;
		color: var(--text-secondary);
	}

	.filter-group select {
		background: var(--bg-secondary);
		color: var(--text-primary);
		border: 1px solid var(--glass-border);
		border-radius: 6px;
		padding: 0.5rem 0.75rem;
		font-size: 1rem;
		cursor: pointer;
		min-width: 140px;
	}

	.filter-group select:focus {
		outline: none;
		border-color: var(--accent-blue);
	}

	.leaderboard-table {
		background: var(--glass-bg);
		backdrop-filter: blur(var(--glass-blur));
		-webkit-backdrop-filter: blur(var(--glass-blur));
		border: 1px solid var(--glass-border);
		border-radius: 12px;
		overflow: hidden;
	}

	table {
		width: 100%;
		border-collapse: collapse;
	}

	thead {
		background: rgba(15, 23, 42, 0.6);
	}

	th {
		font-size: 1rem;
		font-weight: 600;
		color: var(--text-secondary);
		text-align: left;
		padding: 0.75rem 1rem;
		border-bottom: 1px solid var(--glass-border);
		white-space: nowrap;
	}

	td {
		font-size: 1rem;
		padding: 0.625rem 1rem;
		border-bottom: 1px solid rgba(148, 163, 184, 0.08);
		color: var(--text-primary);
	}

	.mono {
		font-family: 'JetBrains Mono', monospace;
	}

	.genome-row {
		cursor: pointer;
		transition: background 0.15s;
	}

	.genome-row:hover {
		background: rgba(51, 65, 85, 0.3);
	}

	.genome-row.expanded {
		background: rgba(51, 65, 85, 0.2);
	}

	.rank-badge {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 2rem;
		height: 2rem;
		border-radius: 50%;
		font-weight: 700;
		font-size: 1rem;
		background: rgba(51, 65, 85, 0.5);
		color: var(--text-secondary);
	}

	.rank-1 {
		background: rgba(234, 179, 8, 0.2);
		color: var(--accent-yellow);
		box-shadow: 0 0 8px rgba(234, 179, 8, 0.3);
	}

	.rank-2 {
		background: rgba(148, 163, 184, 0.2);
		color: var(--text-primary);
	}

	.rank-3 {
		background: rgba(180, 120, 60, 0.2);
		color: #c8956c;
	}

	.tag {
		display: inline-block;
		padding: 0.125rem 0.5rem;
		border-radius: 4px;
		font-size: 1rem;
		font-weight: 600;
	}

	.tag-lm {
		background: rgba(59, 130, 246, 0.15);
		color: var(--accent-blue);
	}

	.tag-ids {
		background: rgba(34, 197, 94, 0.15);
		color: var(--accent-green);
	}

	.hf-badge {
		display: inline-block;
		padding: 0.125rem 0.375rem;
		border-radius: 4px;
		font-size: 1rem;
		font-weight: 600;
		background: rgba(255, 216, 0, 0.15);
		color: #ffd800;
	}

	.flow-link {
		color: var(--accent-blue);
		text-decoration: none;
	}

	.flow-link:hover {
		text-decoration: underline;
	}

	.detail-row td {
		padding: 0;
		border-bottom: 1px solid var(--glass-border);
	}

	.detail-content {
		padding: 1rem 1.5rem;
		background: rgba(15, 23, 42, 0.4);
	}

	.detail-grid {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
		gap: 1rem;
	}

	.detail-item {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.detail-label {
		font-size: 1rem;
		color: var(--text-secondary);
	}

	.detail-value {
		font-size: 1rem;
		color: var(--text-primary);
		word-break: break-all;
	}

	.detail-value a {
		color: var(--accent-blue);
		text-decoration: none;
	}

	.detail-value a:hover {
		text-decoration: underline;
	}

	.count {
		margin-top: 1rem;
		font-size: 1rem;
		color: var(--text-secondary);
		text-align: right;
	}

	.loading, .error, .empty {
		text-align: center;
		padding: 3rem;
		color: var(--text-secondary);
		font-size: 1rem;
	}

	.error {
		color: var(--accent-red);
	}

	.hint {
		font-size: 1rem;
		color: var(--text-tertiary);
		margin-top: 0.5rem;
	}

	.col-rank { width: 60px; text-align: center; }
	.col-task { width: 60px; }
	.col-stage { width: 50px; }
	.col-metric { width: 50px; }
	.col-ce, .col-acc, .col-f1, .col-fpr { width: 80px; text-align: right; }
	.col-arch { width: 70px; text-align: right; }
	.col-flow { width: 60px; }
	.col-hf { width: 40px; }
</style>
