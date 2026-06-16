//! Unit tests for the adaptive module.
//!
//! Split out of adaptive.rs (D3, 11/06/2026).

#[cfg(test)]
mod tests {
    use crate::adaptive::*;

    #[test]
    fn test_build_config_groups() {
        // 5 clusters with 3 different configs
        let bits = vec![8, 8, 10, 10, 8];
        let neurons = vec![5, 5, 3, 3, 5];

        let groups = build_config_groups(&bits, &neurons);

        assert_eq!(groups.len(), 2); // (5,8) and (3,10)

        // Find the (5,8) group
        let group_5_8 = groups.iter().find(|g| g.neurons == 5 && g.bits == 8).unwrap();
        assert_eq!(group_5_8.cluster_ids, vec![0, 1, 4]);

        // Find the (3,10) group
        let group_3_10 = groups.iter().find(|g| g.neurons == 3 && g.bits == 10).unwrap();
        assert_eq!(group_3_10.cluster_ids, vec![2, 3]);
    }

    // =========================================================================
    // OI (Order-Independent) training — dense backend
    // =========================================================================
    //
    // These tests assert that the new OI path produces cell states determined
    // by net vote counts alone, regardless of the order in which (positive,
    // negative) nudges are applied. The current `nudge` path would fail the
    // permutation-invariance test by construction (that's the bug we're fixing).

    fn dense_train_oi(
        nudges: &[(usize, usize, bool, u32)], // (neuron, addr, target_true, weight)
        num_neurons: usize,
        bits: usize,
    ) -> Vec<i64> {
        let mut mem = GroupDenseMemory::new(num_neurons, bits, crate::neuron_memory::MODE_QUAD_WEIGHTED);
        mem.init_oi_counters();
        for &(n, a, t, w) in nudges {
            mem.nudge_oi(n, a, t, w);
        }
        mem.commit_oi();
        // Snapshot cell values for every (neuron, address).
        let n_addrs = 1usize << bits;
        let mut snap = Vec::with_capacity(num_neurons * n_addrs);
        for n in 0..num_neurons {
            for a in 0..n_addrs {
                snap.push(mem.read(n, a));
            }
        }
        snap
    }

    #[test]
    fn oi_dense_permutation_invariance() {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        // 1 neuron, 3-bit address (8 addrs). Train a sequence with both signs.
        let mut nudges: Vec<(usize, usize, bool, u32)> = Vec::new();
        for a in 0..8 {
            // Address `a` gets `a+1` positives and `(7-a)` negatives,
            // mostly with weight=1 but a few with weight=3.
            for i in 0..(a + 1) {
                nudges.push((0, a, true, if i % 3 == 0 { 3 } else { 1 }));
            }
            for i in 0..(7 - a) {
                nudges.push((0, a, false, if i % 4 == 0 { 2 } else { 1 }));
            }
        }

        let baseline = dense_train_oi(&nudges, 1, 3);

        // 10 random permutations: all must produce identical snapshots.
        for seed in 0..10u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut shuffled = nudges.clone();
            shuffled.shuffle(&mut rng);
            let snap = dense_train_oi(&shuffled, 1, 3);
            assert_eq!(snap, baseline, "permutation {} produced a different snapshot", seed);
        }
    }

    #[test]
    fn oi_dense_bin_oracle() {
        // 1 neuron, 2-bit (4 addresses). Hand-construct nudges per address
        // and verify the binned cell matches `oi_bin_to_cell`.
        let nudges = vec![
            // addr 0: untouched → expect WEAK_FALSE
            // addr 1: single positive (weight=1) → expect WEAK_TRUE
            (0, 1, true, 1),
            // addr 2: single negative (weight=5, class-weighted) → expect WEAK_FALSE (hybrid)
            (0, 2, false, 5),
            // addr 3: 5 positives, 3 negatives → net=+2, obs=8 → expect TRUE
            (0, 3, true, 1), (0, 3, true, 1), (0, 3, true, 1), (0, 3, true, 1), (0, 3, true, 1),
            (0, 3, false, 1), (0, 3, false, 1), (0, 3, false, 1),
        ];
        let snap = dense_train_oi(&nudges, 1, 2);

        use crate::neuron_memory::{QUAD_FALSE, QUAD_WEAK_FALSE, QUAD_WEAK_TRUE, QUAD_TRUE};
        let _ = (QUAD_FALSE, QUAD_TRUE); // silence unused if both pass
        assert_eq!(snap[0], QUAD_WEAK_FALSE);
        assert_eq!(snap[1], QUAD_WEAK_TRUE);
        assert_eq!(snap[2], QUAD_WEAK_FALSE);
        assert_eq!(snap[3], QUAD_TRUE);
    }

    #[test]
    fn oi_dense_concurrent_nudges_match_serial() {
        use std::sync::Arc;
        use std::thread;
        use crate::neuron_memory::MODE_QUAD_WEIGHTED;

        // Train the same nudge multiset in serial vs parallel and verify
        // cell snapshots match exactly.
        let bits = 4;
        let num_neurons = 4;
        let n_addrs = 1usize << bits;

        // Generate ~1000 nudges deterministically.
        let mut nudges: Vec<(usize, usize, bool, u32)> = Vec::new();
        for i in 0..1000 {
            let n = i % num_neurons;
            let a = (i * 7) % n_addrs;
            let t = (i % 3) != 0;
            let w = 1 + (i % 4) as u32;
            nudges.push((n, a, t, w));
        }

        let serial = dense_train_oi(&nudges, num_neurons, bits);

        // Parallel: spawn threads each doing a slice of nudges into the same memory.
        let mem = Arc::new({
            let mut m = GroupDenseMemory::new(num_neurons, bits, MODE_QUAD_WEIGHTED);
            m.init_oi_counters();
            m
        });

        let num_threads = 4;
        let chunk = nudges.len() / num_threads;
        let handles: Vec<_> = (0..num_threads).map(|t| {
            let mem = mem.clone();
            let start = t * chunk;
            let end = if t == num_threads - 1 { nudges.len() } else { (t + 1) * chunk };
            let slice = nudges[start..end].to_vec();
            thread::spawn(move || {
                for (n, a, tt, w) in slice {
                    mem.nudge_oi(n, a, tt, w);
                }
            })
        }).collect();
        for h in handles { h.join().unwrap(); }

        // Commit and snapshot.
        let mut mem = Arc::try_unwrap(mem).map_err(|_| "Arc still has refs").unwrap();
        mem.commit_oi();
        let mut parallel = Vec::with_capacity(num_neurons * n_addrs);
        for n in 0..num_neurons {
            for a in 0..n_addrs {
                parallel.push(mem.read(n, a));
            }
        }

        assert_eq!(serial, parallel, "concurrent OI nudges produced different cell states than serial");
    }

    // =========================================================================
    // OI training — sparse DashMap backend
    // =========================================================================

    fn sparse_train_oi(
        nudges: &[(usize, u64, bool, u32)], // (neuron, addr, target_true, weight)
        num_neurons: usize,
    ) -> Vec<(usize, u64, u8)> {
        let mut mem = GroupSparseMemory::new(num_neurons, crate::neuron_memory::MODE_QUAD_WEIGHTED);
        mem.init_oi_counters();
        for &(n, a, t, w) in nudges {
            mem.nudge_oi(n, a, t, w);
        }
        mem.commit_oi();
        // Snapshot eval-visible state: filter out default_empty values so
        // results are comparable across sparse backends (DashMap removes
        // default_empty entries; atomic-HT keeps them as default_empty in
        // claimed slots — eval treats both as "absent").
        let default_empty = mem.default_empty;
        let mut snap: Vec<(usize, u64, u8)> = Vec::new();
        for (n, map) in mem.neurons.iter().enumerate() {
            for entry in map.iter() {
                if *entry.value() != default_empty {
                    snap.push((n, *entry.key(), *entry.value()));
                }
            }
        }
        snap.sort_unstable();
        snap
    }

    #[test]
    fn oi_sparse_permutation_invariance() {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        // 2 neurons, addresses spanning the u64 space (sparse regime).
        let mut nudges: Vec<(usize, u64, bool, u32)> = Vec::new();
        for n in 0..2 {
            for i in 0..50 {
                // High-bit addresses to confirm sparse path.
                let addr = (i as u64) * 0x10_000 + (n as u64) * 0x100;
                // Varied vote patterns.
                for _ in 0..(i % 5 + 1) {
                    nudges.push((n, addr, true, 1));
                }
                for _ in 0..(i % 3 + 1) {
                    nudges.push((n, addr, false, if i % 2 == 0 { 2 } else { 1 }));
                }
            }
        }

        let baseline = sparse_train_oi(&nudges, 2);

        for seed in 0..10u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut shuffled = nudges.clone();
            shuffled.shuffle(&mut rng);
            let snap = sparse_train_oi(&shuffled, 2);
            assert_eq!(snap, baseline, "permutation {} differed", seed);
        }
    }

    #[test]
    fn oi_sparse_bin_oracle() {
        use crate::neuron_memory::{QUAD_WEAK_FALSE, QUAD_WEAK_TRUE, QUAD_TRUE};
        // Neuron 0: addr 100 (untouched, should not appear in snapshot since it
        // bins to default_empty=WEAK_FALSE for QUAD mode)
        // Neuron 0: addr 200 single negative weight=5 → WEAK_FALSE → not stored
        // Neuron 0: addr 300 single positive weight=1 → WEAK_TRUE
        // Neuron 0: addr 400: 3 positives, 1 negative → net=+2 obs>=2 → TRUE
        let nudges = vec![
            (0usize, 200u64, false, 5u32),
            (0, 300, true, 1),
            (0, 400, true, 1), (0, 400, true, 1), (0, 400, true, 1),
            (0, 400, false, 1),
        ];
        let snap = sparse_train_oi(&nudges, 1);

        // Expected: addrs that bin to WEAK_FALSE (default_empty for quad) are NOT
        // inserted; we expect only addr 300 (WEAK_TRUE=2) and 400 (TRUE=3).
        let expected: Vec<(usize, u64, u8)> = vec![
            (0, 300, QUAD_WEAK_TRUE as u8),
            (0, 400, QUAD_TRUE as u8),
        ];
        assert_eq!(snap, expected);
        let _ = QUAD_WEAK_FALSE; // silence unused
    }

    #[test]
    fn oi_sparse_concurrent_match_serial() {
        use std::sync::Arc;
        use std::thread;
        use crate::neuron_memory::MODE_QUAD_WEIGHTED;

        // Deterministic ~1000-nudge multiset.
        let mut nudges: Vec<(usize, u64, bool, u32)> = Vec::new();
        for i in 0..1000 {
            let n = i % 3;
            let a = ((i as u64) * 0x100) ^ ((i as u64) >> 1);
            let t = (i % 4) != 0;
            let w = 1 + (i % 3) as u32;
            nudges.push((n, a, t, w));
        }

        let serial = sparse_train_oi(&nudges, 3);

        let mem = Arc::new({
            let mut m = GroupSparseMemory::new(3, MODE_QUAD_WEIGHTED);
            m.init_oi_counters();
            m
        });

        let num_threads = 4;
        let chunk = nudges.len() / num_threads;
        let handles: Vec<_> = (0..num_threads).map(|t| {
            let mem = mem.clone();
            let start = t * chunk;
            let end = if t == num_threads - 1 { nudges.len() } else { (t + 1) * chunk };
            let slice = nudges[start..end].to_vec();
            thread::spawn(move || {
                for (n, a, tt, w) in slice {
                    mem.nudge_oi(n, a, tt, w);
                }
            })
        }).collect();
        for h in handles { h.join().unwrap(); }

        let mut mem = Arc::try_unwrap(mem).map_err(|_| "Arc still has refs").unwrap();
        mem.commit_oi();
        let mut parallel: Vec<(usize, u64, u8)> = Vec::new();
        for (n, map) in mem.neurons.iter().enumerate() {
            for entry in map.iter() {
                parallel.push((n, *entry.key(), *entry.value()));
            }
        }
        parallel.sort_unstable();

        assert_eq!(serial, parallel, "concurrent OI sparse nudges diverged from serial");
    }

    // =========================================================================
    // OI training — sparse AtomicHashTable backend
    // =========================================================================

    fn sparse_atomic_train_oi(
        nudges: &[(usize, u64, bool, u32)],
        num_neurons: usize,
    ) -> Vec<(usize, u64, u8)> {
        let initial_cap = crate::atomic_hashtable::estimate_capacity(10_000);
        let mut mem = GroupSparseMemoryAtomic::new(
            num_neurons,
            crate::neuron_memory::MODE_QUAD_WEIGHTED,
            initial_cap,
        );
        mem.init_oi_counters();
        for &(n, a, t, w) in nudges {
            mem.nudge_oi(n, a, t, w);
        }
        mem.commit_oi();
        // Same eval-visible-state filter as sparse_train_oi: skip default_empty.
        let default_empty = mem.default_empty;
        let mut snap: Vec<(usize, u64, u8)> = Vec::new();
        for (n, table) in mem.neurons.iter().enumerate() {
            for (k, v) in table.snapshot_sorted() {
                if v != default_empty {
                    snap.push((n, k, v));
                }
            }
        }
        snap.sort_unstable();
        snap
    }

    #[test]
    fn oi_atomic_permutation_invariance() {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut nudges: Vec<(usize, u64, bool, u32)> = Vec::new();
        for n in 0..2 {
            for i in 0..40 {
                let addr = (i as u64) * 0x10_000 + (n as u64);
                for _ in 0..(i % 4 + 1) { nudges.push((n, addr, true, 1)); }
                for _ in 0..(i % 3 + 1) { nudges.push((n, addr, false, if i % 2 == 0 { 2 } else { 1 })); }
            }
        }

        let baseline = sparse_atomic_train_oi(&nudges, 2);
        for seed in 0..6u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut shuffled = nudges.clone();
            shuffled.shuffle(&mut rng);
            let snap = sparse_atomic_train_oi(&shuffled, 2);
            assert_eq!(snap, baseline, "atomic-HT permutation {} differed", seed);
        }
    }

    #[test]
    fn oi_atomic_matches_dashmap_backend() {
        // Same nudges through both sparse backends should produce identical
        // (neuron, addr, cell) snapshots — proving the two backends share
        // OI semantics.
        let mut nudges: Vec<(usize, u64, bool, u32)> = Vec::new();
        for i in 0..500 {
            let n = i % 3;
            let a = ((i as u64) * 0x100) ^ ((i as u64) >> 1);
            let t = (i % 4) != 0;
            let w = 1 + (i % 3) as u32;
            nudges.push((n, a, t, w));
        }

        let dashmap_snap = sparse_train_oi(&nudges, 3);
        let atomic_snap = sparse_atomic_train_oi(&nudges, 3);
        assert_eq!(dashmap_snap, atomic_snap,
            "DashMap and AtomicHT sparse backends diverged on OI commit output");
    }
}

#[cfg(test)]
mod flat_genome_validation_tests {
    use crate::adaptive::validate_flat_genomes;

    // 2 genomes × 2 clusters, 2 neurons/cluster, 3 bits/neuron.
    fn valid_triple() -> (Vec<usize>, Vec<usize>, Vec<i64>) {
        let neurons = vec![2usize, 2, 2, 2];          // 4 = num_genomes * num_clusters
        let bits = vec![3usize; 8];                   // 8 = Σ neurons (per-NEURON)
        let conns = vec![0i64; 24];                   // 24 = Σ bits
        (bits, neurons, conns)
    }

    #[test]
    fn accepts_valid_and_empty_connections() {
        let (bits, neurons, conns) = valid_triple();
        assert!(validate_flat_genomes(&bits, &neurons, &conns, 2, 2).is_ok());
        // Empty connections = documented random-fallback, allowed.
        assert!(validate_flat_genomes(&bits, &neurons, &[], 2, 2).is_ok());
    }

    #[test]
    fn rejects_wrong_neurons_len() {
        let (bits, _, conns) = valid_triple();
        let err = validate_flat_genomes(&bits, &[2, 2, 2], &conns, 2, 2).unwrap_err();
        assert!(err.contains("genomes_neurons_flat"), "{}", err);
    }

    #[test]
    fn rejects_per_cluster_bits_layout() {
        // The classic protocol mistake: one bits entry per CLUSTER (4) instead
        // of per NEURON (8).
        let (_, neurons, _) = valid_triple();
        let err = validate_flat_genomes(&[3, 3, 3, 3], &neurons, &[], 2, 2).unwrap_err();
        assert!(err.contains("per-NEURON"), "{}", err);
    }

    #[test]
    fn rejects_partial_connections() {
        // Mixed batch where only some genomes carried connections: flattened
        // length lands between 0 and Σ bits — previously read misaligned.
        let (bits, neurons, _) = valid_triple();
        let err = validate_flat_genomes(&bits, &neurons, &vec![0i64; 12], 2, 2).unwrap_err();
        assert!(err.contains("genomes_connections_flat"), "{}", err);
    }

    // =========================================================================
    // GOLDEN TEST — evaluate_genomes_parallel_hybrid end-to-end (the eval_hybrid
    // decomposition safety net, 16/06/2026). Tiny DENSE input (b=8 ≤ SPARSE_THRESHOLD
    // ⇒ CPU path, no Metal) run in a 1-thread rayon pool for determinism (non-OI
    // training is order-dependent under multi-thread). Snapshots the first 5 tuple
    // fields (ce, acc, f1, fpr, threshold); the 6th (per-genome ms) is wall-clock =
    // non-deterministic and deliberately excluded. ANY change to these values across
    // the seam extraction means the refactor altered behavior — it MUST stay green.
    // =========================================================================

    /// 2 genomes × 2 clusters × 2 neurons, 8 bits/neuron, 8 input bits.
    /// Both genomes wire every neuron to bits [0..8]; train/eval on the 8
    /// binary patterns 0..8 with alternating class targets.
    fn golden_hybrid_inputs() -> (
        Vec<usize>, Vec<usize>, Vec<i64>,
        crate::packed_bits::PackedBits, Vec<i64>, Vec<i64>,
        crate::packed_bits::PackedBits, Vec<i64>,
    ) {
        let num_genomes = 2usize;
        let num_clusters = 2usize;
        let neurons_per = 2usize;
        let bits = 8usize;
        let total_input_bits = 8usize;

        let neurons_flat = vec![neurons_per; num_genomes * num_clusters]; // [2,2,2,2]
        let total_neurons = num_genomes * num_clusters * neurons_per;     // 8
        let bits_flat = vec![bits; total_neurons];                        // [8;8]
        // conns: Σ bits = total_neurons*bits = 64; every neuron → bits [0..8].
        let mut conns: Vec<i64> = Vec::with_capacity(total_neurons * bits);
        for _ in 0..total_neurons {
            for b in 0..bits {
                conns.push(b as i64);
            }
        }

        let num_train = 8usize;
        let mut train_bytes: Vec<u8> = Vec::with_capacity(num_train * total_input_bits);
        for i in 0..num_train {
            for b in 0..total_input_bits {
                train_bytes.push(((i >> b) & 1) as u8);
            }
        }
        let train_input = crate::packed_bits::PackedBits::from_bool_bytes(&train_bytes, total_input_bits);
        let train_targets: Vec<i64> = (0..num_train).map(|i| (i % 2) as i64).collect();
        // num_negatives = 1: each example's negative is the OTHER class.
        let train_negatives: Vec<i64> = (0..num_train).map(|i| (1 - (i % 2)) as i64).collect();

        let eval_input = train_input.clone();
        let eval_targets = train_targets.clone();

        (bits_flat, neurons_flat, conns, train_input, train_targets, train_negatives, eval_input, eval_targets)
    }

    fn run_golden_hybrid() -> Vec<(f64, f64, f64, f64, f64)> {
        let (bits_flat, neurons_flat, conns, train_input, train_targets, train_negatives, eval_input, eval_targets) =
            golden_hybrid_inputs();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let res = pool.install(|| {
            crate::adaptive::evaluate_genomes_parallel_hybrid(
                &bits_flat, &neurons_flat, &conns,
                2, 2,
                &train_input, &train_targets, &train_negatives,
                8, 1,
                &eval_input, &eval_targets, 8,
                8,
                crate::neuron_memory::EvalSettings::default(),
                1.0, 0,
                None,
            )
        });
        // Drop the non-deterministic per-genome ms (6th field).
        res.into_iter().map(|(ce, acc, f1, fpr, th, _ms)| (ce, acc, f1, fpr, th)).collect()
    }

    #[test]
    fn golden_hybrid_dense_cpu_is_deterministic() {
        // Two runs with identical inputs in a 1-thread pool MUST be bit-identical.
        let a = run_golden_hybrid();
        let b = run_golden_hybrid();
        assert_eq!(a.len(), 2, "expected 2 genome results");
        assert_eq!(a, b, "hybrid eval is non-deterministic under single-thread");
    }

    #[test]
    fn golden_hybrid_dense_cpu_snapshot() {
        let got = run_golden_hybrid();
        assert_eq!(got.len(), 2);
        // Snapshot baked from the PRE-refactor build (bit-exact). Both genomes
        // are wired identically ⇒ identical results; the 8 alternating patterns
        // are perfectly separable ⇒ acc=f1=1.0, fpr=0.0. If this fails after a
        // seam extraction, the refactor changed behavior — investigate, do NOT
        // re-bake.
        const EXPECTED: (f64, f64, f64, f64, f64) =
            (0.38687095046043396, 1.0, 1.0, 0.0, 0.5);
        for (i, g) in got.iter().enumerate() {
            // Bit-exact comparison (a single-ULP drift is a behavior change).
            assert_eq!(g.0.to_bits(), EXPECTED.0.to_bits(), "genome {i} ce drift: {g:?}");
            assert_eq!(g.1.to_bits(), EXPECTED.1.to_bits(), "genome {i} acc drift: {g:?}");
            assert_eq!(g.2.to_bits(), EXPECTED.2.to_bits(), "genome {i} f1 drift: {g:?}");
            assert_eq!(g.3.to_bits(), EXPECTED.3.to_bits(), "genome {i} fpr drift: {g:?}");
            assert_eq!(g.4.to_bits(), EXPECTED.4.to_bits(), "genome {i} threshold drift: {g:?}");
        }
    }

    // =========================================================================
    // materialized_cells — the convention-free sparse-footprint primitive
    // (docs/sparse_footprint_fix.md). Dense groups count the full array
    // (true_neurons × 2^bits); sparse groups count ONLY distinct trained
    // addresses (keys.len()) — NOT the 2^bits dense fiction.
    // =========================================================================
    #[test]
    fn materialized_cells_counts_dense_array_and_sparse_keys() {
        use crate::adaptive::{ConfigGroup, GenomeExport, SparseGpuExport};
        // Group 0: DENSE — 2 neurons × 4 bits → 2 × 2^4 = 32 materialized cells.
        let dense_group = ConfigGroup::new(2, 4, vec![0]);
        // Group 1: SPARSE — 5 distinct trained addresses (its 14-bit space is NOT
        // materialized; only the 5 keys are).
        let sparse_group = ConfigGroup::new(3, 14, vec![1]);
        let sparse = SparseGpuExport {
            keys: vec![10, 20, 30, 40, 50],
            values: vec![1, 1, 1, 1, 1],
            offsets: vec![0],
            counts: vec![5],
            num_neurons: 1,
        };
        let export = GenomeExport {
            connections: vec![],
            group_info: vec![(false, 0, vec![0]), (true, 0, vec![1])],
            dense_exports: vec![vec![0i64; 1]], // contents irrelevant to the count
            sparse_exports: vec![sparse],
            groups: vec![dense_group, sparse_group],
        };
        // 32 (dense full array) + 5 (sparse keys) = 37 — NOT 2 + 3×2^14.
        assert_eq!(export.materialized_cells(), 37);
        // Sentinel empty export → 0 cells.
        assert_eq!(GenomeExport::empty().materialized_cells(), 0);
    }
}
