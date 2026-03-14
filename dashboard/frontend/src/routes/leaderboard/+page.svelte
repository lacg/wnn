<script lang="ts">
	import { onMount } from 'svelte';
	import type { BestGenome } from '$lib/types';
	import { formatDate } from '$lib/dateFormat';

	let rawGenomes: BestGenome[] = [];
	let loading = true;
	let error: string | null = null;

	// Filters
	let taskType = 'ids';
	let stage = '';

	// Fitness weights (configurable, default: F1+FPR focused for IDS)
	let wCe = 0.3;
	let wAcc = 0.3;
	let wF1 = 1.0;
	let wFpr = 1.0;
	let showWeights = false;

	interface RankedGenome extends BestGenome {
		fitnessRank: number;
		fitnessScore: number;
		ceRank: number;
		accRank: number;
		f1Rank: number;
		fprRank: number;
		sources: string[];
	}

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

	async function fetchGenomes() {
		loading = true;
		error = null;
		try {
			const params = new URLSearchParams();
			if (taskType) params.set('task_type', taskType);
			if (stage) params.set('stage', stage);
			params.set('limit', '500');

			const response = await fetch(`/api/best-genomes?${params}`);
			if (!response.ok) throw new Error('Failed to fetch leaderboard');
			rawGenomes = await response.json();
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

	// Deduplicate by genome_hash, keeping the best metrics across duplicates
	function deduplicateGenomes(genomes: BestGenome[]): BestGenome[] {
		const map = new Map<string, BestGenome & { sources: string[] }>();
		for (const g of genomes) {
			const key = g.genome_hash;
			const existing = map.get(key);
			if (!existing) {
				map.set(key, { ...g, sources: [g.metric] });
			} else {
				// Keep best of each metric
				if (g.ce < existing.ce) existing.ce = g.ce;
				if (g.accuracy > existing.accuracy) existing.accuracy = g.accuracy;
				if (g.f1_macro !== null && (existing.f1_macro === null || g.f1_macro > existing.f1_macro)) {
					existing.f1_macro = g.f1_macro;
				}
				if (g.fpr !== null && (existing.fpr === null || g.fpr < existing.fpr)) {
					existing.fpr = g.fpr;
				}
				if (!existing.sources.includes(g.metric)) {
					existing.sources.push(g.metric);
				}
				// Keep the most recent flow_id
				if (g.flow_id && (!existing.flow_id || g.flow_id > existing.flow_id)) {
					existing.flow_id = g.flow_id;
					existing.experiment_id = g.experiment_id;
				}
			}
		}
		return Array.from(map.values());
	}

	// Compute fitness ranking using weighted harmonic mean of ranks
	function computeFitnessRanking(genomes: BestGenome[]): RankedGenome[] {
		const n = genomes.length;
		if (n === 0) return [];

		// Assign per-metric ranks
		// CE: lower is better → sort ascending
		const byCe = genomes.map((g, i) => ({ idx: i, val: g.ce }))
			.sort((a, b) => a.val - b.val);
		const ceRanks = new Array(n);
		byCe.forEach((item, rank) => { ceRanks[item.idx] = rank + 1; });

		// Accuracy: higher is better → sort descending
		const byAcc = genomes.map((g, i) => ({ idx: i, val: g.accuracy }))
			.sort((a, b) => b.val - a.val);
		const accRanks = new Array(n);
		byAcc.forEach((item, rank) => { accRanks[item.idx] = rank + 1; });

		// F1: higher is better → sort descending
		const byF1 = genomes.map((g, i) => ({ idx: i, val: g.f1_macro ?? 0 }))
			.sort((a, b) => b.val - a.val);
		const f1Ranks = new Array(n);
		byF1.forEach((item, rank) => { f1Ranks[item.idx] = rank + 1; });

		// FPR: lower is better → sort ascending
		const byFpr = genomes.map((g, i) => ({ idx: i, val: g.fpr ?? 1 }))
			.sort((a, b) => a.val - b.val);
		const fprRanks = new Array(n);
		byFpr.forEach((item, rank) => { fprRanks[item.idx] = rank + 1; });

		// Compute weighted harmonic mean of ranks
		const totalWeight = wCe + wAcc + wF1 + wFpr;
		const scored: RankedGenome[] = genomes.map((g, i) => {
			const whm = totalWeight > 0
				? totalWeight / (
					(wCe > 0 ? wCe / ceRanks[i] : 0) +
					(wAcc > 0 ? wAcc / accRanks[i] : 0) +
					(wF1 > 0 ? wF1 / f1Ranks[i] : 0) +
					(wFpr > 0 ? wFpr / fprRanks[i] : 0)
				)
				: n;
			return {
				...g,
				fitnessRank: 0,
				fitnessScore: whm,
				ceRank: ceRanks[i],
				accRank: accRanks[i],
				f1Rank: f1Ranks[i],
				fprRank: fprRanks[i],
				sources: (g as any).sources || [g.metric],
			};
		});

		// Sort by fitness score (lower = better) and assign ranks
		scored.sort((a, b) => a.fitnessScore - b.fitnessScore);
		scored.forEach((g, i) => { g.fitnessRank = i + 1; });

		return scored;
	}

	// Reactive: recompute whenever rawGenomes or weights change
	$: deduplicated = deduplicateGenomes(rawGenomes);
	$: rankedGenomes = computeFitnessRanking(deduplicated);

	// Expanded row tracking
	let expandedHash: string | null = null;

	function toggleExpand(hash: string) {
		expandedHash = expandedHash === hash ? null : hash;
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
</script>

<div class="container">
	<div class="page-header">
		<h1>Best Genomes Leaderboard</h1>
		<p class="subtitle">Ranked by fitness (weighted harmonic mean of metric ranks)</p>
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
		<div class="filter-group weight-toggle">
			<button class="weights-btn" on:click={() => showWeights = !showWeights}>
				{showWeights ? 'Hide' : 'Show'} Weights
			</button>
		</div>
	</div>

	{#if showWeights}
		<div class="weights-panel">
			<div class="weight-control">
				<label for="w-ce">CE: {wCe.toFixed(1)}</label>
				<input id="w-ce" type="range" min="0" max="2" step="0.1" bind:value={wCe}>
			</div>
			<div class="weight-control">
				<label for="w-acc">Acc: {wAcc.toFixed(1)}</label>
				<input id="w-acc" type="range" min="0" max="2" step="0.1" bind:value={wAcc}>
			</div>
			<div class="weight-control">
				<label for="w-f1">F1: {wF1.toFixed(1)}</label>
				<input id="w-f1" type="range" min="0" max="2" step="0.1" bind:value={wF1}>
			</div>
			<div class="weight-control">
				<label for="w-fpr">FPR: {wFpr.toFixed(1)}</label>
				<input id="w-fpr" type="range" min="0" max="2" step="0.1" bind:value={wFpr}>
			</div>
		</div>
	{/if}

	{#if loading}
		<div class="loading">Loading leaderboard...</div>
	{:else if error}
		<div class="error">{error}</div>
	{:else if rankedGenomes.length === 0}
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
						<th class="col-score">Score</th>
						<th class="col-task">Task</th>
						<th class="col-f1">F1</th>
						<th class="col-fpr">FPR</th>
						<th class="col-acc">Accuracy</th>
						<th class="col-ce">CE</th>
						<th class="col-arch">Neurons</th>
						<th class="col-flow">Flow</th>
					</tr>
				</thead>
				<tbody>
					{#each rankedGenomes as genome}
						<tr
							class="genome-row"
							class:expanded={expandedHash === genome.genome_hash}
							on:click={() => toggleExpand(genome.genome_hash)}
						>
							<td class="col-rank">
								<span class="rank-badge" class:rank-1={genome.fitnessRank === 1} class:rank-2={genome.fitnessRank === 2} class:rank-3={genome.fitnessRank === 3}>
									{genome.fitnessRank}
								</span>
							</td>
							<td class="col-score mono">{genome.fitnessScore.toFixed(1)}</td>
							<td class="col-task">
								<span class="tag tag-{genome.task_type}">{genome.task_type.toUpperCase()}</span>
							</td>
							<td class="col-f1 mono">{genome.f1_macro !== null ? formatPercent(genome.f1_macro) : '-'}</td>
							<td class="col-fpr mono">{genome.fpr !== null ? formatPercent(genome.fpr) : '-'}</td>
							<td class="col-acc mono">{formatPercent(genome.accuracy)}</td>
							<td class="col-ce mono">{formatMetric(genome.ce)}</td>
							<td class="col-arch mono">{genome.total_neurons ?? '-'}</td>
							<td class="col-flow">
								{#if genome.flow_id}
									<a href="/flows/{genome.flow_id}" class="flow-link" on:click|stopPropagation>F{genome.flow_id}</a>
								{:else}
									-
								{/if}
							</td>
						</tr>
						{#if expandedHash === genome.genome_hash}
							<tr class="detail-row">
								<td colspan="9">
									<div class="detail-content">
										<div class="detail-grid">
											<div class="detail-item">
												<span class="detail-label">Genome Hash</span>
												<span class="detail-value mono">{genome.genome_hash}</span>
											</div>
											<div class="detail-item">
												<span class="detail-label">Per-Metric Ranks</span>
												<span class="detail-value mono">CE: #{genome.ceRank} | Acc: #{genome.accRank} | F1: #{genome.f1Rank} | FPR: #{genome.fprRank}</span>
											</div>
											<div class="detail-item">
												<span class="detail-label">Stage</span>
												<span class="detail-value">{stageLabel(genome.stage)}</span>
											</div>
											<div class="detail-item">
												<span class="detail-label">Sources</span>
												<span class="detail-value">{genome.sources.join(', ')}</span>
											</div>
											<div class="detail-item">
												<span class="detail-label">Architecture</span>
												<span class="detail-value">{genome.architecture_type_str ?? 'unknown'}</span>
											</div>
											{#if genome.tiers_json}
												<div class="detail-item">
													<span class="detail-label">Config</span>
													<span class="detail-value mono">{genome.tiers_json}</span>
												</div>
											{/if}
											<div class="detail-item">
												<span class="detail-label">Clusters</span>
												<span class="detail-value">{genome.total_clusters ?? '-'}</span>
											</div>
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
		<div class="count">{rankedGenomes.length} unique genome{rankedGenomes.length !== 1 ? 's' : ''} (from {rawGenomes.length} entries)</div>
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
		margin-bottom: 1rem;
		flex-wrap: wrap;
		align-items: flex-end;
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

	.weights-btn {
		background: var(--bg-secondary);
		color: var(--text-primary);
		border: 1px solid var(--glass-border);
		border-radius: 6px;
		padding: 0.5rem 0.75rem;
		font-size: 1rem;
		cursor: pointer;
	}

	.weights-btn:hover {
		border-color: var(--accent-blue);
	}

	.weights-panel {
		display: flex;
		gap: 1.5rem;
		margin-bottom: 1.5rem;
		padding: 1rem;
		background: var(--glass-bg);
		border: 1px solid var(--glass-border);
		border-radius: 8px;
		flex-wrap: wrap;
	}

	.weight-control {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		min-width: 120px;
	}

	.weight-control label {
		font-size: 1rem;
		color: var(--text-secondary);
		font-family: 'JetBrains Mono', monospace;
	}

	.weight-control input[type="range"] {
		width: 100%;
		accent-color: var(--accent-blue);
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
	.col-score { width: 60px; text-align: right; }
	.col-task { width: 60px; }
	.col-ce, .col-acc, .col-f1, .col-fpr { width: 80px; text-align: right; }
	.col-arch { width: 70px; text-align: right; }
	.col-flow { width: 60px; }
</style>
