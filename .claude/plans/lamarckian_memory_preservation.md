# Lamarckian Memory Preservation — Implementation Plan (blueprint 05/06/2026)

Goal: preserve learned cells through arch mutations (N/B/C) instead of re-training from
scratch. Fixes re-train variance (82→55) + sample efficiency. ALL PYTHON, no Rust rebuild
(Rust getters last_state/output_layer_input already live).

VERIFIED already-implemented: _remap_grow (bits↑ replicate a<<d same value),
_remap_shrink (bits↓ majority-collapse), _mutate_connections→_drop_changed_neurons
(connectivity invalidation in mutation), evaluate_for_adaptation(write_back) + warm-start
from genome.cells.to_triples().

Build order (each pure-Python, test at each):
- STEP 2 (bits): cell-math done; ADD max_cells budget-trim in set_state_suffix/set_output_suffix
  (add config param). Tests: test_bits_grow_replicates_cells, test_bits_shrink_collapses_cells.
- STEP 3 (connections): make _filter_inherited_cells CONNECTIVITY-AWARE (kwargs changed_state/
  changed_output → skip cells of rewired neurons; it's currently bounds-only/blind). Compute
  deltas in ControllerArchGAStrategy.crossover_genomes. Test: test_crossover_invalidates_rewired_neurons.
- STEP 1 (wire): add lamarckian flag + _lamarckian_evaluate_batch (→evaluate_for_adaptation
  write_back=True) to ControllerArchGAStrategy; crossover_genomes inherits+filters cells; promote
  _filter_inherited_cells to module-level; --lamarckian in run_phased_ga _run_arch_phase; remove
  dup _filter_cells. Test: test_lamarckian_arch_ga_carries_cells + smoke run.
- STEP 4 (axonogenesis): remove stale hasattr guard in arch_adaptation.record_input_entropy
  (getters live); update test_axonogenesis_wired_pending_rebuild → entropy+rewire.

Files: src/wnn/control/arch_strategy.py, recurrent_genome.py, arch_adaptation.py;
tests/run_phased_ga.py, test_controller_arch_ga.py, test_controller_lamarckian.py.

Risks: keep lamarckian ON for a whole stage (1-seed write-back vs K-seed evaluate_batch not
mixable); state-neuron grow replicates cells 2^k (cap via max_cells); QSR majority ties→FALSE
(conservative, OK). Final-report should use evaluate_for_adaptation(write_back=False).
Full blueprint: this file. Architect agent aedffdd891c0e4f6a (SendMessage to continue).
