<script lang="ts">
	import type { Experiment, Checkpoint } from '$lib/types'

	export let checkpoints: Checkpoint[] = []
	export let experiments: Experiment[] = []
	export let isIDS: boolean = false

	$: flowCheckpoints = checkpoints.filter(c => c.checkpoint_type === 'experiment_end' && experiments.some(e => e.id === c.experiment_id))
	$: bestCeCheckpoint = flowCheckpoints.filter(c => c.best_ce != null).sort((a, b) => (a.best_ce ?? 0) - (b.best_ce ?? 0))[0]
	$: bestAccCheckpoint = flowCheckpoints.filter(c => c.best_accuracy != null).sort((a, b) => (b.best_accuracy ?? 0) - (a.best_accuracy ?? 0))[0]
	$: completedExps = experiments.filter(e => e.status === 'completed')
	$: bestF1Exp = isIDS ? completedExps.filter(e => e.extra_metrics?.f1_macro != null).sort((a, b) => (b.extra_metrics?.f1_macro ?? 0) - (a.extra_metrics?.f1_macro ?? 0))[0] : null
	$: bestFprExp = isIDS ? completedExps.filter(e => e.extra_metrics?.fpr != null).sort((a, b) => (a.extra_metrics?.fpr ?? 1) - (b.extra_metrics?.fpr ?? 1))[0] : null
</script>

<section class="section">
	<h2>Final Results</h2>
	{#if isIDS && (bestF1Exp || bestAccCheckpoint)}
		<div class="final-results-card">
			<div class="results-grid">
				<div class="result-item">
					<div class="result-phase">{bestF1Exp?.name ?? '—'}</div>
					<div class="result-label">Best F1-Macro</div>
					<div class="result-value">{bestF1Exp?.extra_metrics?.f1_macro != null ? (bestF1Exp.extra_metrics.f1_macro * 100).toFixed(2) + '%' : '—'}</div>
				</div>
				<div class="result-item">
					<div class="result-phase">{bestFprExp?.name ?? '—'}</div>
					<div class="result-label">Best FPR</div>
					<div class="result-value">{bestFprExp?.extra_metrics?.fpr != null ? (bestFprExp.extra_metrics.fpr * 100).toFixed(2) + '%' : '—'}</div>
				</div>
				<div class="result-item">
					<div class="result-phase">{bestAccCheckpoint?.name ?? '—'}</div>
					<div class="result-label">Best Accuracy</div>
					<div class="result-value">{bestAccCheckpoint?.best_accuracy ? (bestAccCheckpoint.best_accuracy * 100).toFixed(2) + '%' : '—'}</div>
				</div>
			</div>
			<div class="results-footer">
				<a href="/" class="btn btn-secondary">View Iterations</a>
				<a href="/checkpoints" class="btn btn-secondary">View All Checkpoints</a>
			</div>
		</div>
	{:else if bestCeCheckpoint}
		<div class="final-results-card">
			<div class="results-grid">
				<div class="result-item">
					<div class="result-phase">{bestCeCheckpoint.name}</div>
					<div class="result-label">Best CE</div>
					<div class="result-value">{bestCeCheckpoint.best_ce?.toFixed(4) ?? '—'}</div>
				</div>
				<div class="result-item">
					<div class="result-phase">{bestAccCheckpoint?.name ?? '—'}</div>
					<div class="result-label">Best Accuracy</div>
					<div class="result-value">{bestAccCheckpoint?.best_accuracy ? (bestAccCheckpoint.best_accuracy * 100).toFixed(2) + '%' : '—'}</div>
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

<style>
  .section {
    margin-bottom: 2rem;
  }

  h2 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
  }

  /* Final Results */
  .final-results-card {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--accent-green);
    border-radius: 12px;
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

  .result-phase {
    font-size: 1rem;
    color: var(--text-secondary);
    margin-bottom: 0.25rem;
  }

  .result-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-primary);
    font-family: monospace;
  }

  .results-footer {
    display: flex;
    gap: 0.75rem;
    justify-content: center;
    padding-top: 1rem;
    border-top: 1px solid var(--glass-border);
  }

  .empty-state {
    text-align: center;
    padding: 2rem;
    color: var(--text-secondary);
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

  .btn-secondary {
    background: rgba(51, 65, 85, 0.4);
    color: var(--text-primary);
    border: 1px solid var(--glass-border);
  }

  .btn-secondary:hover:not(:disabled) {
    background: var(--border);
  }

  @media (max-width: 640px) {
    .results-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
