<script lang="ts">
  import '../app.css';
  import { onMount, onDestroy } from 'svelte';
  import { connectWebSocket, disconnectWebSocket, wsConnected } from '$lib/stores';

  onMount(() => {
    connectWebSocket();
  });

  onDestroy(() => {
    disconnectWebSocket();
  });
</script>

<header class="header">
  <div class="container header-content">
    <div class="header-left">
      <a href="/" class="logo">WNN Dashboard</a>
      <nav class="nav">
        <a href="/" class="nav-link">Iterations</a>
        <a href="/flows" class="nav-link">Flows</a>
        <a href="/checkpoints" class="nav-link">Checkpoints</a>
        <a href="/experiments" class="nav-link">History</a>
      </nav>
    </div>
    <div class="header-right">
      <div class="connection-status">
        {#if $wsConnected}
          <span class="status-dot connected"></span> Connected
        {:else}
          <span class="status-dot disconnected"></span> Disconnected
        {/if}
      </div>
      <a href="/settings" class="settings-link" title="Settings">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="3"></circle>
          <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
        </svg>
      </a>
    </div>
  </div>
</header>

<main>
  <slot />
</main>

<style>
  .header {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    padding: 1rem 0;
    position: sticky;
    top: 0;
    z-index: 100;
  }

  .header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 2rem;
  }

  .logo {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--accent-blue);
    text-decoration: none;
  }

  .logo:hover {
    opacity: 0.8;
  }

  .nav {
    display: flex;
    gap: 1rem;
  }

  .nav-link {
    color: var(--text-secondary);
    text-decoration: none;
    font-size: 0.875rem;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    transition: all 0.2s;
  }

  .nav-link:hover {
    color: var(--text-primary);
    background: var(--bg-tertiary);
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .connection-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
    color: var(--text-secondary);
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .status-dot.connected {
    background: var(--accent-green);
  }

  .status-dot.disconnected {
    background: var(--accent-red);
  }

  .settings-link {
    color: var(--text-tertiary);
    padding: 0.375rem;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
  }

  .settings-link:hover {
    color: var(--text-primary);
    background: var(--bg-tertiary);
  }

  main {
    min-height: calc(100vh - 60px);
  }
</style>
