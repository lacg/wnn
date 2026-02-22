<script lang="ts">
  import { onMount } from 'svelte';
  import {
    type DateFormatPrefs,
    detectDateFormat,
    savePreferences,
    clearPreferences,
    hasCustomPreferences,
    getAutoDetectedPrefs,
    formatSampleDate
  } from '$lib/dateFormat';

  let prefs: DateFormatPrefs = {
    order: 'dmy',
    separator: '/',
    padded: true,
    is24h: true
  };

  let autoDetected: DateFormatPrefs | null = null;
  let hasCustom = false;
  let saved = false;

  onMount(() => {
    prefs = detectDateFormat();
    autoDetected = getAutoDetectedPrefs();
    hasCustom = hasCustomPreferences();
  });

  function save() {
    savePreferences(prefs);
    hasCustom = true;
    saved = true;
    setTimeout(() => saved = false, 2000);
  }

  function resetToAuto() {
    clearPreferences();
    if (autoDetected) {
      prefs = { ...autoDetected };
    }
    hasCustom = false;
  }

  $: preview = formatSampleDate(prefs);
</script>

<div class="container">
  <div class="page-header">
    <a href="/" class="back-link">&larr; Iterations</a>
    <h1>Settings</h1>
  </div>

  <section class="section">
    <h2>Date & Time Format</h2>

    {#if autoDetected}
      <div class="auto-detected">
        <span class="label">Auto-detected:</span>
        <span class="value">{formatSampleDate(autoDetected)}</span>
        {#if !hasCustom}
          <span class="badge">Active</span>
        {/if}
      </div>
    {/if}

    <div class="settings-card">
      <div class="preview">
        <span class="preview-label">Preview:</span>
        <span class="preview-value">{preview}</span>
        {#if hasCustom}
          <span class="badge custom">Custom</span>
        {/if}
      </div>

      <div class="form-grid">
        <div class="form-group">
          <label for="order">Date Order</label>
          <select id="order" bind:value={prefs.order}>
            <option value="dmy">DD/MM/YYYY (European)</option>
            <option value="mdy">MM/DD/YYYY (US)</option>
            <option value="ymd">YYYY-MM-DD (ISO 8601)</option>
          </select>
        </div>

        <div class="form-group">
          <label for="separator">Separator</label>
          <select id="separator" bind:value={prefs.separator}>
            <option value="/">/  (slash)</option>
            <option value="-">-  (dash)</option>
            <option value=".">. (dot)</option>
          </select>
        </div>

        <div class="form-group">
          <label for="padded">Zero Padding</label>
          <select id="padded" bind:value={prefs.padded}>
            <option value={true}>01, 02, ... 09 (padded)</option>
            <option value={false}>1, 2, ... 9 (no padding)</option>
          </select>
        </div>

        <div class="form-group">
          <label for="is24h">Time Format</label>
          <select id="is24h" bind:value={prefs.is24h}>
            <option value={true}>24-hour (14:30)</option>
            <option value={false}>12-hour (2:30 PM)</option>
          </select>
        </div>
      </div>

      <div class="form-actions">
        {#if hasCustom}
          <button class="btn btn-secondary" on:click={resetToAuto}>
            Reset to Auto-Detect
          </button>
        {/if}
        <button class="btn btn-primary" on:click={save}>
          {saved ? 'Saved!' : 'Save Preferences'}
        </button>
      </div>
    </div>
  </section>
</div>

<style>
  .page-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 2rem;
    padding-top: 2rem;
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

  .section {
    margin-bottom: 2rem;
  }

  h2 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
  }

  .auto-detected {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    background: rgba(51, 65, 85, 0.4);
    border: 1px solid var(--glass-border);
    border-radius: 10px;
    margin-bottom: 1rem;
    font-size: 1rem;
  }

  .auto-detected .label {
    color: var(--text-tertiary);
  }

  .auto-detected .value {
    color: var(--text-secondary);
    font-family: monospace;
  }

  .badge {
    font-size: 1rem;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    background: var(--accent-blue);
    color: white;
    text-transform: uppercase;
    font-weight: 600;
  }

  .badge.custom {
    background: var(--accent-green);
  }

  .settings-card {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    box-shadow: var(--glass-shadow), var(--glass-inset);
    padding: 1.5rem;
  }

  .preview {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 1rem;
    background: var(--glass-input-bg);
    border: 1px solid var(--glass-border);
    border-radius: 8px;
    margin-bottom: 1.5rem;
  }

  .preview-label {
    font-size: 1rem;
    color: var(--text-tertiary);
  }

  .preview-value {
    font-size: 1.125rem;
    font-weight: 600;
    font-family: monospace;
    color: var(--text-primary);
  }

  .form-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
  }

  .form-group {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
  }

  label {
    font-size: 1rem;
    font-weight: 500;
    color: var(--text-primary);
  }

  select {
    padding: 0.5rem 0.75rem;
    background: var(--glass-input-bg);
    border: 1px solid var(--glass-border);
    border-radius: 8px;
    backdrop-filter: blur(8px);
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.15);
    color: var(--text-primary);
    font-size: 1rem;
    cursor: pointer;
  }

  select:focus {
    outline: none;
    border-color: var(--accent-blue);
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.15), 0 0 0 3px rgba(59, 130, 246, 0.15);
  }

  .form-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(71, 85, 105, 0.4);
  }

  .btn {
    padding: 0.5rem 1rem;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    border: none;
    transition: all 0.15s;
  }

  .btn-primary {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.85), rgba(99, 102, 241, 0.85));
    border: 1px solid rgba(59, 130, 246, 0.4);
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    color: white;
    min-width: 140px;
  }

  .btn-primary:hover {
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.2);
    transform: translateY(-1px);
  }

  .btn-secondary {
    background: rgba(51, 65, 85, 0.5);
    backdrop-filter: blur(8px);
    color: var(--text-primary);
    border: 1px solid var(--glass-border);
  }

  .btn-secondary:hover {
    background: rgba(51, 65, 85, 0.7);
    border-color: var(--glass-border-highlight);
  }

  @media (max-width: 640px) {
    .form-grid {
      grid-template-columns: 1fr;
    }
  }
</style>
