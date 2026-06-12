<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import type { Flow } from '$lib/types';

  export let flow: Flow;
  export let editMode: boolean = false;
  export let saving: boolean = false;
  export let duplicating: boolean = false;
  export let deleting: boolean = false;
  export let actionInFlight: boolean = false;
  // Bound to the page so navigation resets cancel an in-progress rename.
  export let editingName: boolean = false;
  export let editedName: string = '';

  const dispatch = createEventDispatcher<{
    saveName: void;
    duplicate: void;
    queue: void;
    pause: void;
    resume: void;
    stop: void;
    restart: boolean; // fromBeginning
    delete: void;
    editConfig: void;
  }>();

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
</script>

<div class="flow-header">
  <div class="header-left">
    <a href="/flows" class="back-link">&larr; Flows</a>
    {#if editingName}
      <div class="name-edit">
        <input
          type="text"
          bind:value={editedName}
          class="name-input"
          on:keydown={(e) => e.key === 'Enter' && dispatch('saveName')}
          on:keydown={(e) => e.key === 'Escape' && cancelEditName()}
        />
        <button class="btn btn-sm btn-primary" on:click={() => dispatch('saveName')} disabled={saving}>✓</button>
        <button class="btn btn-sm btn-secondary" on:click={cancelEditName}>✕</button>
      </div>
    {:else}
      <button class="flow-name-editable" on:click={startEditName} title="Click to rename">
        <h1>{flow.name}</h1>
      </button>
    {/if}
    <span class="status-badge" style="background: {getStatusColor(flow.status)}">
      {flow.status}
    </span>
    {#if flow.status_message && (flow.status === 'running' || flow.status === 'queued')}
      <span class="flow-status-message">{flow.status_message}</span>
    {/if}
  </div>
  <div class="header-actions">
    <button class="btn btn-sm btn-secondary" on:click={() => dispatch('duplicate')} disabled={duplicating} title="Duplicate flow">
      {duplicating ? '...' : '📋 Duplicate'}
    </button>
    {#if !editMode && flow.status === 'pending'}
      <button class="btn btn-primary" on:click={() => dispatch('queue')} disabled={actionInFlight}>
        Start
      </button>
    {/if}
    {#if !editMode && flow.status !== 'running' && flow.status !== 'queued'}
      <button class="btn btn-secondary" on:click={() => dispatch('editConfig')}>
        Edit Config
      </button>
    {/if}
    {#if flow.status === 'queued'}
      <span class="queued-hint">Waiting for worker to pick up...</span>
    {/if}
    {#if flow.status === 'running'}
      <button class="btn btn-secondary" on:click={() => dispatch('pause')} disabled={actionInFlight} title="Pause at end of current generation; resume later from checkpoint">
        ⏸ Pause
      </button>
    {/if}
    {#if flow.status === 'paused'}
      <button class="btn btn-primary" on:click={() => dispatch('resume')} disabled={actionInFlight} title="Resume from per-gen checkpoint (clears pause, re-queues)">
        ▶ Resume
      </button>
    {/if}
    {#if flow.status === 'running' || flow.status === 'queued'}
      <button class="btn btn-danger" on:click={() => dispatch('stop')} disabled={actionInFlight}>
        Stop
      </button>
    {/if}
    {#if flow.status === 'failed' || flow.status === 'cancelled'}
      <button class="btn btn-primary" on:click={() => dispatch('restart', false)} disabled={actionInFlight}>
        Resume
      </button>
      <button class="btn btn-secondary" on:click={() => dispatch('restart', true)} disabled={actionInFlight}>
        Restart from Beginning
      </button>
    {/if}
    {#if flow.status === 'paused'}
      <button class="btn btn-secondary" on:click={() => dispatch('restart', true)} disabled={actionInFlight}>
        Restart from Beginning
      </button>
    {/if}
    {#if flow.status === 'completed'}
      <button class="btn btn-secondary" on:click={() => dispatch('restart', true)} disabled={actionInFlight}>
        Run Again
      </button>
    {/if}
    {#if flow.status !== 'running' && flow.status !== 'queued'}
      <button class="btn btn-danger" on:click={() => dispatch('delete')} disabled={deleting} title="Delete flow">
        {deleting ? 'Deleting...' : 'Delete'}
      </button>
    {/if}
  </div>
</div>

<style>
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
    background: none;
    border: none;
    font: inherit;
    color: inherit;
    text-align: left;
  }

  .flow-name-editable h1 {
    margin: 0;
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

  .flow-status-message {
    font-size: 1rem;
    color: var(--text-secondary, #888);
    font-style: italic;
    margin-left: 0.5rem;
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
    background: rgba(51, 65, 85, 0.4);
    color: var(--text-primary);
    border: 1px solid var(--glass-border);
  }

  .btn-secondary:hover:not(:disabled) {
    background: var(--border);
  }

  .btn-sm {
    padding: 0.375rem 0.75rem;
    font-size: 1rem;
  }
</style>
