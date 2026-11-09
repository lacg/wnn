# IDS receptive fields (ids_receptive_fields, worker ABI 27)

Status: BUILT 08/11/2026 in a worktree, not deployed, nothing queued. This is the IDS port
of the controller's CTRL-45 (docs/ctrl45_axis_receptive_fields_design.md,
docs/ctrl45_validity_audit.md).

## What it is

A constraint on the connectivity map, not a new neuron type. Every encoded input bit
belongs to one feature FAMILY (one mechanism: UNSW ttl_state, handshake, volume, rate,
timing, context, protocol, ...). A neuron of width `w` may hold at most
`budget(w) = floor(ids_rf_cross * w)` connections outside its family. Memory, cells,
decode, kernels and exporters are untouched: they only see connection indices.

Flow params: `ids_receptive_fields` = `off` (default, bit-identical, no RNG draw) | `family`;
`ids_rf_cross` in [0, 0.5), default 0.2 (refused when set with RF off).

## The IDS difference: no output bank

The controller keys a neuron's family on its output bank (the roll bank gets roll
features). IDS has one decision, so the family is per NEURON.

**Storage: derived, never stored, never positional.** A neuron's family is the family
holding a strict MAJORITY of its connections. After projection the family holds
`w - budget(w) > w/2` (that is why cross < 0.5), so the family survives the operators
that move whole neurons (neuron crossover, pool-shuffle, dedupe), a few connection drifts
and BITS grow/shrink. It needs no genome / checkpoint / DB change and no flat-position
key (the IDS-50 lesson). A stored per-neuron tag was rejected: it would have to ride
through every crossover, the dedupe NeuronKey, ClusterGenome serialization and the
checkpoints, for no gain over a rule the connections already encode.

**Pick rule (a neuron with no majority family: fresh, or scrambled by a per-connection
crossover).** Draw one eligible family weighted by the neuron's own connections in each.
For a fresh uniform neuron that is Luiz's "my first connection picks the family":
P(family) = the family's share of the encoded bits. So the cluster's aggregate wiring per
family matches `spread`; only its concentration changes, which isolates the RF effect
from a reallocation effect. (Uniform-over-families was rejected: a 1-feature family like
CICIDS `port` would get as many neurons as `timing`.) A family that already holds more
than half of a neuron's connections keeps it, so a family above ~40% of all bits gets
somewhat more than its share (those fresh neurons often start with a majority).

**Fallback (small families).** Family f is ELIGIBLE as a home for a w-wide neuron iff it
has at least `w - budget(w)` distinct encoded bits. A smaller family is reachable at
that width only through the cross budget. A neuron no family can host is EXEMPT (left as
drawn, no draw). Nothing is ever placed out-of-family to cover a deficit (the
controller's corner), so the majority rule — and therefore idempotence — always holds.

**Declared merge (IDS-55 follow-up, Luiz "merge ttl_state").** A family may declare
`"merge_into": "<family>"` in the map. When its encoded-bit count (after dedupe) is below
`host_bits = max_bits - floor(ids_rf_cross * max_bits)` — what the run's widest neuron
needs, the eligibility rule above — its features join the partner (chained through the
partner's own `merge_into`). Families that can host are untouched. UNSW `ttl_state` declares
`merge_into: context`: with dedupe on it keeps 12–16 bits (unswr/unswt, both rankings) and
merges; off it has 196 (unswr) / 48–52 (unswt) and does not. The `[RF]` line names each
merge. **v2 (Luiz 08/11):** UNSW `handshake` declares `merge_into: timing` (tcprtt/synack/ackdat
are the setup round-trip timings; 10 bits with dedupe on, 32 off on unswt legacy → merges only
with dedupe on at max_bits 34) and `protocol` declares `merge_into: context` (12 bits on unswt
legacy even with dedupe OFF → merges there too). Because a dedupe-off map changes, the map
version moved to `families_v2` (the validation cache re-keys); the IDS-55 unswt A/B
(flows 7451–7460) ran on `families_v1`, where protocol stayed cross-budget-only.

## The projection

ram_core `receptive_fields.rs`: per neuron, if it is satisfied (or exempt) nothing
happens and NO RNG is drawn; else pick the family, keep a uniform `budget` of its
out-of-family connections and redraw the rest, in the same slots, from family bits the
neuron does not hold (eligibility guarantees enough). Idempotent; deterministic in the
operator's RNG.

**Where it is applied (every path that creates or moves a connection):**

| site | covers |
|---|---|
| Rust `neighbor_search::mutate_genome` (after the neuron budget) | every GA child — uniform crossover2, pool-shuffle, cluster crossover, crossover_bits blends, dedupe_fill — and every TS neighbour: search_offspring mutates every child (clones too), search_neighbors every neighbour |
| Rust `adaptive_eval.rs`, after the genesis passes, before the retrain | synaptogenesis growth, neurogenesis fresh neurons, axonogenesis rewiring |
| Python `ArchitectureGAStrategy` `_build_genome` / `mutate_genome` / `crossover_genomes` + the seed population | random init, immigrants, seed mutants, the Python breeding fallback, grid / leaderboard / resumed seeds |
| Python TS / SA `mutate_genome`, TS start genome | Python neighbour fallbacks |
| Python `GridSearchStrategy._create_genome*` + leaderboard seeds | every grid genome and expansion variant |
| Python `AdaptationStrategy` padding | fresh-connection clones |

Python calls the SAME Rust function (`rf_project_connections`); there is no Python copy.
No GPU path creates or edits connections, so there is no GPU twin; CPU/GPU parity of
training and scoring is untouched by construction.

## Family maps

Versioned package data: `src/wnn/ids/feature_families/v1/{unsw-nb15,cicids2017,ciciot2023}.json`
(`FAMILY_MAP_VERSION = families_v2`; v1 kept as the A/B provenance), each family with its mechanism. They cover the
legacy top-20 lists, the CIC-IoT MI lists and the FULL train_only_v1 rankings (so top15 /
top25 / all work too). A feature without a family, or in two, is refused. The bit map is
built from the encoder's actual layout (`feature_bit_ranges`, after dedupe; the
is_invalid flag bit belongs to its feature) and must tile the dataset's bits
(`top20_split`, two joined encoders, is refused).

Refinement from the first proposal: CIC-IoT `Duration` is the packet TTL in the dataset's
feature table, so it sits in context, not timing. CICIDS per-second rates join timing
(on the legacy list a rate family would be one feature).

## Provenance

The factory logs `[RF] ids_receptive_fields=family map=families_v2/<file> ids_rf_cross=...`
with each family's encoded-bit count; the experiment logs the version per phase; the
validation-cache key carries `rf=families_v2:<cross>` when on (conservative: RF steers
the search, which genome_hash already pins).
