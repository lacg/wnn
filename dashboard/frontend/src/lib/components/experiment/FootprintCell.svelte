<script lang="ts">
	import type { ValidationFootprint } from '$lib/types'

	// One genome's footprint, LOGICAL and PHYSICAL — always both (IDS-37).
	// Physical < logical when the multiplicity layer trained duplicate
	// neurons once; equal for the plain layer.
	export let footprint: ValidationFootprint | null | undefined = null
	export let colspan: number = 3

	function cells(n: number): string
	{
		for (const [scale, unit] of [[1e9, 'Gcells'], [1e6, 'Mcells'], [1e3, 'Kcells']] as const)
		{
			if (n >= scale) return `${(n / scale).toFixed(2)} ${unit}`
		}
		return `${n} cells`
	}
</script>

<td class="mono footprint-cell" {colspan}>
	{#if footprint}
		<span class="footprint-line">{footprint.logical_neurons}n · {cells(footprint.logical_cells)} (logical)</span>
		<span class="footprint-line" class:footprint-saved={footprint.physical_neurons < footprint.logical_neurons}>{footprint.physical_neurons}n · {cells(footprint.physical_cells)} (physical)</span>
	{:else}
		—
	{/if}
</td>

<style>
  .footprint-cell {
    font-size: 1rem;
    text-align: center;
    color: var(--text-secondary);
  }

  .footprint-line {
    display: block;
    font-size: 1rem;
  }

  .footprint-saved {
    color: var(--accent-green);
  }
</style>
