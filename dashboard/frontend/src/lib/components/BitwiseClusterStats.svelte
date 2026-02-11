<script lang="ts">
	export let clusterStats: {
		cluster: number;
		bits: number;
		neurons: number;
		connections: number;
		memory_words: number;
	}[] | null | undefined = undefined;

	const BYTES_PER_WORD = 8;
	const MB = 1024 * 1024;
	const KB = 1024;
	const THRESHOLD_LARGE = 10 * MB;
	const THRESHOLD_MEDIUM = 1 * MB;

	function formatMemory(words: number): string {
		const bytes = words * BYTES_PER_WORD;
		if (bytes >= MB) {
			return (bytes / MB).toFixed(1) + ' MB';
		} else if (bytes >= KB) {
			return (bytes / KB).toFixed(1) + ' KB';
		} else {
			return bytes + ' B';
		}
	}

	function memoryColor(words: number): string {
		const bytes = words * BYTES_PER_WORD;
		if (bytes >= THRESHOLD_LARGE) return 'var(--accent-red)';
		if (bytes >= THRESHOLD_MEDIUM) return 'var(--accent-yellow, #eab308)';
		return 'var(--accent-green)';
	}

	$: totals = (() => {
		if (!clusterStats || clusterStats.length === 0) return null;
		return {
			neurons: clusterStats.reduce((sum, s) => sum + s.neurons, 0),
			connections: clusterStats.reduce((sum, s) => sum + s.connections, 0),
			memory_words: clusterStats.reduce((sum, s) => sum + s.memory_words, 0),
		};
	})();
</script>

{#if clusterStats && clusterStats.length > 0}
	<div class="gating-section">
		<div class="gating-header">
			<span class="gating-title">Bitwise Cluster Stats</span>
			<span class="gating-meta">
				{clusterStats.length} clusters
			</span>
		</div>
		<div class="gating-table-container">
			<table class="gating-table">
				<thead>
					<tr>
						<th>Cluster</th>
						<th>Bits</th>
						<th>Neurons</th>
						<th>Connections</th>
						<th>Memory Size</th>
					</tr>
				</thead>
				<tbody>
					{#each clusterStats as stat}
						<tr>
							<td class="mono">{stat.cluster}</td>
							<td class="mono">{stat.bits}</td>
							<td class="mono">{stat.neurons.toLocaleString()}</td>
							<td class="mono">{stat.connections.toLocaleString()}</td>
							<td class="mono" style="color: {memoryColor(stat.memory_words)}">
								{formatMemory(stat.memory_words)}
							</td>
						</tr>
					{/each}
					{#if totals}
						<tr class="totals-row">
							<td class="totals-label">Total</td>
							<td class="mono">—</td>
							<td class="mono">{totals.neurons.toLocaleString()}</td>
							<td class="mono">{totals.connections.toLocaleString()}</td>
							<td class="mono" style="color: {memoryColor(totals.memory_words)}">
								{formatMemory(totals.memory_words)}
							</td>
						</tr>
					{/if}
				</tbody>
			</table>
		</div>
	</div>
{/if}

<style>
	.gating-section {
		background: var(--bg-secondary);
		border: 1px solid var(--border);
		border-radius: 8px;
		padding: 1rem;
		border-left: 4px solid var(--accent-purple, #9b59b6);
	}

	.gating-header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		margin-bottom: 0.75rem;
		padding-bottom: 0.5rem;
		border-bottom: 1px solid var(--border);
	}

	.gating-title {
		font-weight: 600;
		font-size: 1rem;
		color: var(--text-primary);
	}

	.gating-meta {
		font-size: 1rem;
		color: var(--muted, var(--text-secondary));
	}

	.gating-table-container {
		overflow-x: auto;
	}

	.gating-table {
		width: 100%;
		border-collapse: collapse;
		font-size: 1rem;
	}

	.gating-table th {
		text-align: left;
		padding: 0.5rem;
		border-bottom: 2px solid var(--border);
		color: var(--muted, var(--text-secondary));
		font-weight: 500;
		font-size: 1rem;
	}

	.gating-table td {
		padding: 0.5rem;
		border-bottom: 1px solid var(--border);
		font-size: 1rem;
	}

	.gating-table tr:last-child td {
		border-bottom: none;
	}

	.mono {
		font-family: 'Berkeley Mono', monospace;
	}

	.totals-row {
		border-top: 2px solid var(--border);
		font-weight: 600;
	}

	.totals-row td {
		border-top: 2px solid var(--border);
	}

	.totals-label {
		font-weight: 600;
		color: var(--text-primary);
	}
</style>
