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
