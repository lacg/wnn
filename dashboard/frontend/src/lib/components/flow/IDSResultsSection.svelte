<script lang="ts">
	import type { Experiment } from '$lib/types'
	import IDSMetrics from '$lib/components/IDSMetrics.svelte'

	export let experiments: Experiment[] = []

	$: idsExperiments = experiments.filter(e => e.extra_metrics && e.status === 'completed')
</script>

{#if idsExperiments.length > 0}
	<section class="section">
		<h2>IDS Results (Test Set)</h2>
		{#each idsExperiments as exp}
			{#if exp.extra_metrics}
				<IDSMetrics metrics={exp.extra_metrics} title="{exp.name}" />
			{/if}
		{/each}
	</section>
{/if}

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
</style>
