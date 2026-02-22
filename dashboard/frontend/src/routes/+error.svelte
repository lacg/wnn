<script lang="ts">
  import { page } from '$app/stores';
</script>

<div class="container">
  <div class="error-page">
    <h1>Error {$page.status}</h1>
    <p class="error-message">{$page.error?.message || 'Something went wrong'}</p>

    <div class="error-actions">
      <button on:click={() => window.location.reload()}>Reload Page</button>
      <a href="/">Go Home</a>
    </div>

    <p class="error-hint">
      If this keeps happening, try restarting the dashboard backend and frontend.
    </p>
  </div>
</div>

<style>
  .error-page {
    text-align: center;
    padding: 4rem 2rem;
  }

  h1 {
    font-size: 3rem;
    color: var(--accent-red);
    margin-bottom: 1rem;
  }

  .error-message {
    font-size: 1.25rem;
    color: var(--text-secondary);
    margin-bottom: 2rem;
  }

  .error-actions {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin-bottom: 2rem;
  }

  button, a {
    padding: 0.75rem 1.5rem;
    border-radius: 0.5rem;
    font-size: 1rem;
    cursor: pointer;
    text-decoration: none;
  }

  button {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.85), rgba(99, 102, 241, 0.85));
    color: white;
    border: 1px solid rgba(59, 130, 246, 0.4);
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.15);
  }

  button:hover {
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.35);
    transform: translateY(-1px);
  }

  a {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    color: var(--text-primary);
    border: 1px solid var(--glass-border);
  }

  a:hover {
    border-color: var(--glass-border-highlight);
  }

  .error-hint {
    color: var(--text-tertiary);
    font-size: 1rem;
  }
</style>
