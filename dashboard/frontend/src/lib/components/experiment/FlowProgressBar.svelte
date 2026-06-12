<script lang="ts">
  import type { ExperimentStatus } from '$lib/types';

  export let flowSteps: { name: string; status: ExperimentStatus; id: number | null; index: number }[] = [];
  export let currentExperimentId: number;
</script>

<div class="flow-progress">
  <div class="flow-progress-label">Flow Progress</div>
  <div class="flow-progress-bar">
    {#each flowSteps as step, idx}
      {@const isCurrent = step.id === currentExperimentId}
      {@const hasId = step.id !== null}
      <div class="flow-step" class:current={isCurrent}>
        {#if hasId && !isCurrent}
          <a href="/experiments/{step.id}" class="step-link step-{step.status}">
            <span class="step-number">{idx + 1}</span>
            <span class="step-name">{step.name.replace(/^Phase \d+[ab]: /, '')}</span>
          </a>
        {:else}
          <div class="step-box step-{step.status}" class:step-current={isCurrent}>
            <span class="step-number">{idx + 1}</span>
            <span class="step-name">{step.name.replace(/^Phase \d+[ab]: /, '')}</span>
          </div>
        {/if}
      </div>
      {#if idx < flowSteps.length - 1}
        <div class="step-connector" class:connector-done={step.status === 'completed'}></div>
      {/if}
    {/each}
  </div>
</div>

<style>
  /* Flow Progress Bar */
  .flow-progress {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: 0.5rem;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
    overflow-x: auto;
  }

  .flow-progress-label {
    font-size: 1rem;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
  }

  .flow-progress-bar {
    display: flex;
    align-items: center;
    gap: 0;
    min-width: max-content;
  }

  .flow-step {
    flex-shrink: 0;
  }

  .step-link, .step-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 0.375rem 0.75rem;
    border-radius: 0.375rem;
    font-size: 1rem;
    text-decoration: none;
    min-width: 5rem;
    text-align: center;
    transition: all 0.15s;
  }

  .step-link {
    cursor: pointer;
  }

  .step-link:hover {
    transform: translateY(-1px);
  }

  .step-number {
    font-weight: 600;
    font-size: 1rem;
  }

  .step-name {
    font-size: 1rem;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 8rem;
  }

  .step-completed {
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid var(--accent-green);
    color: var(--accent-green);
  }

  .step-completed .step-name {
    color: var(--accent-green);
  }

  .step-running {
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid var(--accent-blue);
    color: var(--accent-blue);
  }

  .step-running .step-name {
    color: var(--accent-blue);
  }

  .step-pending {
    background: rgba(128, 128, 128, 0.2);
    border: 1px dashed var(--text-tertiary);
    color: var(--text-secondary);
  }

  .step-pending .step-name {
    color: var(--text-tertiary);
  }

  .step-current {
    box-shadow: 0 0 0 2px var(--accent-blue);
  }

  .step-connector {
    width: 1.5rem;
    height: 2px;
    background: var(--border);
    flex-shrink: 0;
  }

  .connector-done {
    background: var(--accent-green);
  }
</style>
