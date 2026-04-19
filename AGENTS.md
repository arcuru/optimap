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

# Sweep benchmarks (continuous N-curve, CSV + PNG plots)
./scripts/sweep-bench.sh

# Miri (UB detection) — uses scalar SIMD fallbacks via cfg(miri)
RUSTFLAGS="" MIRIFLAGS="-Zmiri-disable-isolation" cargo miri test
```

Requires Rust nightly (for SIMD intrinsics) + miri component. The flake provides both.

## Architecture: Two Design Families

Hash map designs split into two families based on deletion strategy:

**Overflow-bit family** (tombstone-free): UFM, Splitsies, Gaps, matrix variants
- Overflow bits track displaced entries → O(1) miss termination
- Deletion clears the slot and adjusts max_load — no tombstones
- Generic `RawTable<K,V,L: GroupLayout>` in `src/raw/overflow_table.rs`

**Tombstone family**: InPlaceOverflow (IPO), Hi128_Tomb, Top128_Tomb
- Tombstones for deletion, EMPTY-based probe termination (like hashbrown)
- IPO's `RawTable<K,V,T: TombstoneTag>` in `src/in_place_overflow/raw/mod.rs`

Both families use the **mid-pointer memory layout**: a single `ctrl` pointer
sits between buckets (backward) and metadata (forward). This eliminates a
struct field and an address computation from the hot path — the same trick
hashbrown uses. For overflow-bit designs, the overflow region sits after
metadata (also forward from ctrl).

### Parameterization axes

The design space is parameterized by composable traits:

| Axis | Trait | Implementations |
|------|-------|-----------------|
| **Tag extraction** | `TagStrategy` / `TombstoneTag` | LowByte255, HighByte255, LowByte128, TopTag128, TopTag255, LowByte254, HighByte128, TopByte128 |
| **Overflow storage** | `OverflowStrategy` | ByteSeparate (8-channel), BitSeparate (1-bit), UfmEmbedded (byte 15) |
| **Group indexing** | `GroupLayout::AND_INDEX` | Shift-based (`h >> shift`, default) or AND-based (`h & mask`) |
| **Group ops** | `GroupOps` / `Group<SLOT_MASK>` | 15-slot (0x7FFF) or 16-slot (0xFFFF) |

New design variants are ~30 lines: a type alias composing these traits.
The `matrix_types` module in `src/lib.rs` has experimental combinations.

**AND-based indexing constraint**: uses low hash bits for group index, so tags
must come from top bits (57+) and 8-bit overflow channels (also low bits)
would correlate. Only safe with 1-bit overflow (BitSeparate).

## Project Structure

- `src/` — All implementations behind `Map`/`SortedMap`/`Set`/`SortedSet` traits
  - `raw/` — Shared infrastructure:
    - `table_api.rs` — `RawTableApi<K,V>` trait (internal contract for all raw table backends)
    - `group_layout.rs` — `GroupLayout` trait + `Layout16`/`Layout16And` + named layouts
    - `tag_strategy.rs` — `TagStrategy` trait (tag + overflow channel extraction)
    - `overflow_strategy.rs` — `OverflowStrategy` trait (ByteSeparate, BitSeparate)
    - `generic_group.rs` — Shared SIMD group ops parameterized by slot mask
    - `overflow_table.rs` — Generic overflow-bit `RawTable<K,V,L>` (mid-pointer layout)
    - `bitmask.rs`, `hash.rs`, `group.rs` — Bitmask, hash mixing, legacy UFM group ops
    - `mod.rs` — Legacy UFM RawTable (still used by `set.rs`)
  - `generic_map.rs` — `GenericMap<K,V,S,R>` (single map wrapper over any RawTableApi backend)
  - `map.rs` — `UnorderedFlatMap` type alias (= `GenericMap` + `UfmLayout`)
  - `set.rs` — UnorderedFlatSet (hand-tuned set with SIMD fast-path)
  - `split_overflow/` — `Splitsies` type alias (= `GenericMap` + `SplitsiesLayout`)
  - `in_place_overflow/` — InPlaceOverflow (own RawTable + TombstoneTag, mid-pointer layout)
  - `ipo64/` — IPO64 (own RawTable + RawTableApi impl, GenericMap alias)
  - `gaps/` — `Gaps` type alias (= `GenericMap` + `GapsLayout`)
  - `flat_btree/` — FlatBTree (B+ tree, independent architecture)
  - `traits.rs` — `Map`/`Set`/`SortedMap`/`SortedSet` traits + impls for hashbrown/std
  - `generic_set.rs` — `GenericSet<T, M>` wrapper (set from any Map via `Map<T, ()>`)
  - `optimap.rs` — `OptiMap<K, V>` smart wrapper with dynamic backend selection
  - `opti_set.rs` — `OptiSet<T>` smart set wrapper (wraps `OptiMap<T, ()>`)
  - `opti_sorted.rs` — `OptiSortedMap<K, V>` and `OptiSortedSet<T>` smart sorted wrappers (wraps `FlatBTree`)
- `benches/` — Criterion benchmarks (throughput, construction, distributions, workloads, load_factor, sets)
- `tests/` — Integration tests
- `docs/` — mdbook: designs, benchmarks, optimization logs, roadmap

## Key Design Decisions

- All designs use **foldhash** by default (avalanching, fast)
- Overflow-bit designs (UFM, Splitsies, Gaps) are **tombstone-free** with O(1) miss termination
- IPO/IPO64 use **tombstones** like hashbrown but with 254 hash values (vs hashbrown's 128)
- 70% default load factor across all designs
- Generic `Map` trait allows benchmarking all implementations + hashbrown uniformly
- `Map` trait covers full std::HashMap interface: get/insert/remove/entry + get_key_value, remove_entry, retain, drain, reserve, shrink_to_fit, iter_mut, keys, values, values_mut, try_insert, into_keys, into_values
- `Set` trait mirrors Map for sets: insert/contains/get/remove/take/retain/drain/reserve/shrink_to_fit/iter
- `SortedMap` trait covers ordered ops: first/last_key_value, pop_first/pop_last, range, iter_sorted
- `SortedSet` trait mirrors SortedMap for sets: first/last, pop_first/pop_last, iter_sorted, range
- Entry API matches std: or_insert, or_insert_with, or_insert_with_key, or_default, and_modify, into_key
- OptiMap has full entry API via enum `Entry`/`OccupiedEntry`/`VacantEntry` types with `entry_match!` macro dispatch
- `OptiMap<K, V>` wraps all backends behind an enum with policy-driven backend selection (by capacity, KV size, workload hint) and optional auto-transition on resize
- `OptiSet<T>` wraps `OptiMap<T, ()>` with set-specific API, inheriting all Hint/MapType/Backend selection
- `OptiSortedMap<K, V>` and `OptiSortedSet<T>` wrap `FlatBTree` for sorted containers (single backend for now, extensible)

## Known Gaps / TODO

- `docs/src/architecture.md` doesn't yet cover AND-based indexing or the mid-pointer layout
- No mdbook page for the design matrix (tag × overflow × indexing combinations)
- Sweep plots need regenerating after the latest optimizations

## Optimization Status

All designs have been through extensive optimization passes. See `docs/` (mdbook) for
detailed logs and `.claude/plans/abstract-churning-tome.md` for the latest investigation.

Key results (107K entries, Hi128_Tomb vs hashbrown):
- Lookup hit: 4.07 vs 4.25 ns (4% faster)
- Insert: 503 vs 603 µs (17% faster)
- Remove: 763 vs 1079 µs (29% faster)

Key optimizations applied:
- **Mid-pointer memory layout** — single `ctrl` pointer between buckets (backward)
  and metadata (forward), eliminating a struct field and address computation
- **AND-based group indexing** — `h & mask` (1 instruction) vs `h >> shift` (2 instructions),
  applied to IPO and 1-bit overflow designs
- **Tag-group decorrelation** — tags must use hash bits disjoint from group index bits;
  AND indexing requires tags from top bits (57+)

Future ideas (see plan file): AND indexing for 8-bit overflow with shifted channels,
mid-pointer for 15-slot designs (UFM/Gaps embed overflow → only 2 regions).
