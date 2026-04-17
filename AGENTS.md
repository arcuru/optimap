# OptiMap — Agent Context

## What This Is

A Rust library (`optimap`) providing multiple SIMD-accelerated hash map implementations
with different performance trade-offs, benchmarked against hashbrown (Rust's std HashMap).

## Designs

| Design | Key Idea | Best At |
|--------|----------|---------|
| **UnorderedFlatMap** | 15-slot groups, overflow byte | High-load miss, churn |
| **Splitsies** | 16-slot, separate overflow array | Balanced (miss + insert), tombstone-free |
| **InPlaceOverflow** | 16-slot Swiss-table style | Lookup hit, insert |
| **IPO64** | 64-slot cache-line, AVX-512 | Specialty: high-load resilience |
| **Gaps** | 15-slot + power-of-2 buckets | Iteration |
| **FlatBTree** | 256-byte B+ tree nodes | Sorted iteration, range queries |

## Build & Test

```bash
# Uses flake.nix devShell (direnv auto-activates)
cargo test
cargo bench
```

Requires Rust nightly (for SIMD intrinsics). The flake provides it.

## Project Structure

- `src/` — All implementations behind `Map`/`SortedMap`/`Set`/`SortedSet` traits
  - `raw/` — Shared SIMD group ops, bitmask, hash mixing
  - `map.rs` — UnorderedFlatMap
  - `set.rs` — UnorderedFlatSet (hand-tuned set with SIMD fast-path)
  - `split_overflow/` — Splitsies
  - `in_place_overflow/` — InPlaceOverflow (IPO)
  - `ipo64/` — IPO64
  - `gaps/` — Gaps
  - `flat_btree/` — FlatBTree (B+ tree)
  - `traits.rs` — `Map`/`Set`/`SortedMap`/`SortedSet` traits + impls for hashbrown/std
  - `generic_set.rs` — `GenericSet<T, M>` wrapper (set from any Map via `Map<T, ()>`)
  - `optimap.rs` — `OptiMap<K, V>` smart wrapper with dynamic backend selection
- `benches/` — Criterion benchmarks (throughput, construction, distributions, workloads, load_factor, sets)
- `tests/` — Integration tests
- `docs/` — mdbook: designs, benchmarks, optimization logs, roadmap

## Key Design Decisions

- All designs use **foldhash** by default (avalanching, fast)
- Overflow-bit designs (UFM, Splitsies, Gaps) are **tombstone-free** with O(1) miss termination
- IPO/IPO64 use **tombstones** like hashbrown but with 254 hash values (vs hashbrown's 128)
- 70% default load factor across all designs
- Generic `Map` trait allows benchmarking all implementations + hashbrown uniformly
- `Map` trait covers full std::HashMap interface: get/insert/remove/entry + get_key_value, remove_entry, retain, drain, reserve, shrink_to_fit, iter_mut, keys, values, values_mut
- `Set` trait mirrors Map for sets: insert/contains/get/remove/take/retain/drain/reserve/shrink_to_fit/iter
- `SortedMap` trait covers ordered ops: first/last_key_value, pop_first/pop_last, range, iter_sorted
- `SortedSet` trait mirrors SortedMap for sets: first/last, pop_first/pop_last, iter_sorted, range
- Entry API matches std: or_insert, or_insert_with, or_insert_with_key, or_default, and_modify, into_key
- `OptiMap<K, V>` wraps all backends behind an enum with policy-driven backend selection (by capacity, KV size, workload hint) and optional auto-transition on resize

## Known Gaps / TODO

- **OptiMap Entry API**: All concrete map types have `entry()` returning their own `Entry`/`OccupiedEntry`/`VacantEntry` types, but `OptiMap` does not expose an entry API yet. This is non-trivial because each backend's Entry type borrows the map with different concrete types — would need an enum entry type or boxed approach with careful lifetime management.

## Optimization Status

All designs have been through extensive optimization passes. See `docs/` (mdbook) for
detailed logs. Key finding: optimizations that fail on one design (bucket prefetch,
cold continuation) tend to fail on all for the same fundamental CPU reasons.
