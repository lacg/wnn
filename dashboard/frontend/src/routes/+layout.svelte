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
        <a href="/" class="nav-link">Live</a>
        <a href="/flows" class="nav-link">Flows</a>
        <a href="/checkpoints" class="nav-link">Checkpoints</a>
        <a href="/experiments" class="nav-link">History</a>
      </nav>
    </div>
    <div class="connection-status">
      {#if $wsConnected}
        <span class="status-dot connected"></span> Connected
      {:else}
        <span class="status-dot disconnected"></span> Disconnected
      {/if}
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

  main {
    min-height: calc(100vh - 60px);
  }
</style>
