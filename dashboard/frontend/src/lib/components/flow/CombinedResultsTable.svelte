<script lang="ts">
  import type { CombinedValidation } from '$lib/types';

  export let combinedValidations: CombinedValidation[] = [];
</script>

<section class="section">
  <h2>Combined Results</h2>
  <div class="final-results-card">
    <div class="table-container">
      <table class="data-table">
        <thead>
          <tr>
            <th>Genome Type</th>
            {#if combinedValidations.some(cv => cv.unigram_lambda != null)}
              <th>Lambda</th>
            {/if}
            <th>Combined CE</th>
            <th>Combined ACC</th>
            {#if combinedValidations[0]?.per_stage_ce}
              {#each combinedValidations[0].per_stage_ce as _, i}
                <th>S{i} CE</th>
                <th>S{i} ACC</th>
              {/each}
            {/if}
          </tr>
        </thead>
        <tbody>
          {#each combinedValidations as cv}
            <tr>
              <td>
                <span class="genome-type-badge" class:best-ce={cv.genome_type === 'best_ce' || cv.genome_type === 'best_overall_ce'} class:best-acc={cv.genome_type === 'best_acc' || cv.genome_type === 'best_overall_acc'} class:best-fitness={cv.genome_type === 'best_fitness'} class:best-overall={cv.genome_type === 'best_overall_ce' || cv.genome_type === 'best_overall_acc'} class:lambda-sweep={cv.genome_type.startsWith('unigram_l')}>
                  {cv.genome_type === 'best_ce' ? 'Best CE' : cv.genome_type === 'best_acc' ? 'Best ACC' : cv.genome_type === 'best_fitness' ? 'Best Fitness' : cv.genome_type === 'best_overall_ce' ? 'Best Overall CE' : cv.genome_type === 'best_overall_acc' ? 'Best Overall ACC' : cv.genome_type.startsWith('unigram_l') ? `λ=${cv.genome_type.replace('unigram_l', '')}` : cv.genome_type}
                </span>
              </td>
              {#if combinedValidations.some(cv => cv.unigram_lambda != null)}
                <td class="mono">{cv.unigram_lambda != null ? cv.unigram_lambda.toFixed(3) : '—'}</td>
              {/if}
              <td class="mono">{cv.combined_ce.toFixed(4)}</td>
              <td class="mono">{(cv.combined_accuracy * 100).toFixed(2)}%</td>
              {#if cv.per_stage_ce}
                {#each cv.per_stage_ce as stageCe, i}
                  <td class="mono">{stageCe.toFixed(4)}</td>
                  <td class="mono">{cv.per_stage_acc ? (cv.per_stage_acc[i] * 100).toFixed(2) + '%' : '—'}</td>
                {/each}
              {/if}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
    <div class="results-footer">
      <a href="/" class="btn btn-secondary">View Iterations</a>
      <a href="/checkpoints" class="btn btn-secondary">View All Checkpoints</a>
    </div>
  </div>
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

  .final-results-card {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--accent-green);
    border-radius: 12px;
    padding: 1.5rem;
  }

  .results-footer {
    display: flex;
    gap: 0.75rem;
    justify-content: center;
    padding-top: 1rem;
    border-top: 1px solid var(--glass-border);
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

  /* Combined Results table */
  .genome-type-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 1rem;
    font-weight: 500;
  }

  .genome-type-badge.best-ce {
    background: color-mix(in srgb, var(--accent-blue) 20%, transparent);
    color: var(--accent-blue);
  }

  .genome-type-badge.best-acc {
    background: color-mix(in srgb, var(--accent-green) 20%, transparent);
    color: var(--accent-green);
  }

  .genome-type-badge.best-fitness {
    background: color-mix(in srgb, var(--accent-purple, #8b5cf6) 20%, transparent);
    color: var(--accent-purple, #8b5cf6);
  }

  .genome-type-badge.best-overall {
    border: 1.5px solid currentColor;
    font-weight: 600;
  }

  .genome-type-badge.lambda-sweep {
    background: color-mix(in srgb, var(--accent-orange, #f59e0b) 20%, transparent);
    color: var(--accent-orange, #f59e0b);
    font-family: var(--font-mono, monospace);
  }
</style>
