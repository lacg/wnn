<script lang="ts">
	import type { LiveProgress } from './types'

	export let liveProgress: LiveProgress
	export let isIDS: boolean = false
</script>

<div class="live-progress-card">
	<div class="live-progress-bar">
		<span class="live-dot"></span>
		{#if liveProgress.total_generations > 0}
			<strong>Gen {liveProgress.generation}/{liveProgress.total_generations}</strong>
		{/if}
		<span class="live-phase">{liveProgress.phase === 'ga_offspring' ? 'GA Offspring' : liveProgress.phase === 'ts_neighbors' ? 'TS Neighbors' : 'Evaluating'}</span>
		<progress value={liveProgress.evaluated} max={liveProgress.target_count}></progress>
		<span>{liveProgress.evaluated}/{liveProgress.target_count}</span>
		{#if liveProgress.viable != null}
			<span>({liveProgress.viable} viable)</span>
		{/if}
		{#if liveProgress.best_ce > 0}
			{#if !isIDS}
				<span class="live-metric">CE: {liveProgress.best_ce.toFixed(4)}</span>
			{/if}
			<span class="live-metric">Acc: {(liveProgress.best_acc * 100).toFixed(2)}%</span>
		{/if}
		<span class="live-elapsed">{liveProgress.elapsed_secs.toFixed(0)}s</span>
	</div>
</div>

<style>
  /* Live generation progress */
  .live-progress-card {
    background: var(--card-bg, #1e1e2e);
    border: 1px solid var(--border, #333);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
  }
  .live-progress-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-size: 1rem;
    flex-wrap: wrap;
  }
  .live-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #4ade80;
    animation: pulse 1.5s ease-in-out infinite;
    flex-shrink: 0;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }
  .live-phase {
    color: var(--text-secondary, #aaa);
  }
  .live-progress-bar progress {
    flex: 1;
    min-width: 100px;
    max-width: 200px;
    height: 8px;
    border-radius: 4px;
    appearance: none;
  }
  .live-progress-bar progress::-webkit-progress-bar {
    background: var(--border, #333);
    border-radius: 4px;
  }
  .live-progress-bar progress::-webkit-progress-value {
    background: #4ade80;
    border-radius: 4px;
  }
  .live-metric {
    color: var(--accent, #60a5fa);
    font-variant-numeric: tabular-nums;
  }
  .live-elapsed {
    color: var(--text-secondary, #aaa);
    font-variant-numeric: tabular-nums;
  }
</style>
