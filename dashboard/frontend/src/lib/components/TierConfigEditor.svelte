<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	export let value: string = '';
	export let readonly: boolean = false;

	const dispatch = createEventDispatcher<{ change: string }>();

	interface TierConfigEntry {
		clusters: number | null; // null = "rest" tier
		neurons: number;
		bits: number;
		optimize: boolean;
	}

	let tiers: TierConfigEntry[] = [];
	let lastParsedValue = '';

	// Parse tier config string into structured entries
	function parse(str: string): TierConfigEntry[] {
		if (!str.trim()) return [{ clusters: null, neurons: 5, bits: 8, optimize: false }];
		return str.split(';').map(part => {
			const fields = part.trim().split(',').map(f => f.trim());
			return {
				clusters: fields[0] === 'rest' ? null : parseInt(fields[0]) || 100,
				neurons: parseInt(fields[1]) || 10,
				bits: parseInt(fields[2]) || 12,
				optimize: fields.length > 3 ? fields[3] === 'true' : false
			};
		});
	}

	// Serialize structured entries back to string
	function serialize(entries: TierConfigEntry[]): string {
		return entries.map(t => {
			const c = t.clusters === null ? 'rest' : String(t.clusters);
			return `${c},${t.neurons},${t.bits},${t.optimize}`;
		}).join(';');
	}

	// Reactive sync: external value changes re-parse, guarded by sentinel
	$: if (value !== lastParsedValue) {
		tiers = parse(value);
		lastParsedValue = value;
	}

	function emitChange() {
		const serialized = serialize(tiers);
		lastParsedValue = serialized;
		value = serialized;
		dispatch('change', serialized);
	}

	function addTier() {
		// Insert before the rest tier (last entry)
		const restIndex = tiers.findIndex(t => t.clusters === null);
		const newTier: TierConfigEntry = { clusters: 100, neurons: 10, bits: 12, optimize: false };
		if (restIndex >= 0) {
			tiers = [...tiers.slice(0, restIndex), newTier, ...tiers.slice(restIndex)];
		} else {
			tiers = [...tiers, newTier];
		}
		emitChange();
	}

	function removeTier(index: number) {
		if (tiers[index].clusters === null) return; // Can't remove rest tier
		tiers = tiers.filter((_, i) => i !== index);
		emitChange();
	}

	function handleFieldChange() {
		tiers = tiers; // trigger reactivity
		emitChange();
	}
</script>

{#if readonly}
	<div class="tier-grid readonly">
		<div class="tier-header">
			<span class="col-clusters">Clusters</span>
			<span class="col-neurons">Neurons</span>
			<span class="col-bits">Bits</span>
			<span class="col-optimize">Optimize</span>
		</div>
		{#each tiers as tier}
			<div class="tier-row">
				<span class="col-clusters mono">{tier.clusters === null ? 'rest' : tier.clusters}</span>
				<span class="col-neurons mono">{tier.neurons}</span>
				<span class="col-bits mono">{tier.bits}</span>
				<span class="col-optimize mono">{tier.optimize ? 'yes' : 'no'}</span>
			</div>
		{/each}
	</div>
{:else}
	<div class="tier-grid editable">
		<div class="tier-header">
			<span class="col-clusters">Clusters</span>
			<span class="col-neurons">Neurons</span>
			<span class="col-bits">Bits</span>
			<span class="col-optimize">Optimize</span>
			<span class="col-actions"></span>
		</div>
		{#each tiers as tier, i}
			<div class="tier-row">
				<span class="col-clusters">
					{#if tier.clusters === null}
						<span class="rest-label">rest</span>
					{:else}
						<input type="number" bind:value={tier.clusters} on:change={handleFieldChange} min="1" />
					{/if}
				</span>
				<span class="col-neurons">
					<input type="number" bind:value={tier.neurons} on:change={handleFieldChange} min="1" max="50" />
				</span>
				<span class="col-bits">
					<input type="number" bind:value={tier.bits} on:change={handleFieldChange} min="1" max="32" />
				</span>
				<span class="col-optimize">
					<input type="checkbox" bind:checked={tier.optimize} on:change={handleFieldChange} />
				</span>
				<span class="col-actions">
					{#if tier.clusters !== null}
						<button class="btn-icon" on:click={() => removeTier(i)} title="Remove tier">
							&times;
						</button>
					{/if}
				</span>
			</div>
		{/each}
		<div class="tier-add">
			<button class="btn-add" on:click={addTier}>+ Add Tier</button>
		</div>
	</div>
{/if}

<style>
	.tier-grid {
		border: 1px solid var(--border);
		border-radius: 6px;
		overflow: hidden;
		font-size: 1rem;
	}

	.tier-header {
		display: grid;
		gap: 0.5rem;
		padding: 0.5rem 0.75rem;
		background: var(--bg-tertiary);
		border-bottom: 1px solid var(--border);
		font-weight: 500;
		font-size: 1rem;
		text-transform: uppercase;
		letter-spacing: 0.05em;
		color: var(--text-secondary);
	}

	.editable .tier-header {
		grid-template-columns: 1fr 1fr 1fr 4rem 2rem;
	}

	.readonly .tier-header {
		grid-template-columns: 1fr 1fr 1fr 4rem;
	}

	.tier-row {
		display: grid;
		gap: 0.5rem;
		padding: 0.375rem 0.75rem;
		align-items: center;
		border-bottom: 1px solid var(--border);
	}

	.tier-row:last-child {
		border-bottom: none;
	}

	.editable .tier-row {
		grid-template-columns: 1fr 1fr 1fr 4rem 2rem;
	}

	.readonly .tier-row {
		grid-template-columns: 1fr 1fr 1fr 4rem;
	}

	.mono {
		font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
	}

	.rest-label {
		font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
		color: var(--text-secondary);
		font-style: italic;
	}

	.tier-grid input[type="number"] {
		width: 100%;
		padding: 0.25rem 0.5rem;
		border: 1px solid var(--border);
		border-radius: 4px;
		background: var(--bg-primary);
		color: var(--text-primary);
		font-size: 1rem;
		font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
	}

	.tier-grid input[type="number"]:focus {
		outline: none;
		border-color: var(--accent-blue);
	}

	.tier-grid input[type="checkbox"] {
		width: auto;
		margin: 0;
		cursor: pointer;
	}

	.col-optimize {
		text-align: center;
	}

	.col-actions {
		text-align: center;
	}

	.btn-icon {
		background: none;
		border: none;
		color: var(--text-secondary);
		cursor: pointer;
		font-size: 1.125rem;
		line-height: 1;
		padding: 0.125rem 0.25rem;
		border-radius: 4px;
	}

	.btn-icon:hover {
		color: var(--accent-red);
		background: rgba(239, 68, 68, 0.1);
	}

	.tier-add {
		padding: 0.5rem 0.75rem;
		border-top: 1px solid var(--border);
	}

	.btn-add {
		background: none;
		border: 1px dashed var(--border);
		color: var(--text-secondary);
		cursor: pointer;
		padding: 0.25rem 0.75rem;
		border-radius: 4px;
		font-size: 1rem;
		width: 100%;
	}

	.btn-add:hover {
		color: var(--accent-blue);
		border-color: var(--accent-blue);
		background: rgba(59, 130, 246, 0.05);
	}
</style>
