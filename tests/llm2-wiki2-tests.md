  Quick test results with WIKITEXT_WORD tokenizer:
  - Vocab: ~6,500 tokens (standard word-level)
  - Initial PPL: ~1482 (high - room for improvement)
  - Optimization running with perplexity as the goal

  Reference perplexity benchmarks on WikiText-2:
  | Model       | PPL      |
  |-------------|----------|
  | Trigram     | ~150-200 |
  | LSTM        | ~65-100  |
  | AWD-LSTM    | ~57      |
  | GPT-2 Small | ~29      |
  | GPT-2 Large | ~22      |

Key ideas:

  1. More neurons (256-512): 128 neurons may be saturating. More neurons = more patterns captured.
  2. Add exact RAMs for n=5, n=6: Currently only n=2,3,4 have exact matching. High-frequency 5-grams and 6-grams would improve the "covered" percentage significantly.
  3. Probability distribution output: Instead of predicting one word, have each RAM output a distribution. When voting, aggregate distributions rather than discrete votes. This gives much better perplexity since we're measuring probability, not just correctness.
  4. Higher cascade threshold: 0.05 is very low - RAMs are outputting low-confidence guesses. Try 0.15-0.2 to force only confident predictions to be used, falling back to voting more often but with better calibration.
  5. Learned voting weights in perplexity path: Your meta-classifier got +2.9% over cascade. Use those learned weights in the actual perplexity calculation, not just evaluation.
  6. n=7 context: Longer range patterns could help with the 26% of cases going to voting.