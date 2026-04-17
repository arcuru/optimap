# Future Work

Ordered roughly by expected impact. Items in the "Closed" section have been
thoroughly investigated and proven unproductive — see
[Closed Investigations](optimization/closed.md) for details.

## Recently Completed

### Hash Map + FlatBTree API (April 2025)

Full `Map` trait expansion matching std::HashMap's interface:

| Item | Scope |
|------|-------|
| `get_key_value()` / `remove_entry()` | All maps, `Map` trait |
| `iter_mut()` / `keys()` / `values()` / `values_mut()` | All maps, `Map` trait (defaults for `keys`/`values`/`values_mut`) |
| `reserve()` / `shrink_to_fit()` | All hash maps + FlatBTree, `Map` trait |
| `drain()` iterator | All hash maps + FlatBTree, `Map` trait |
| `retain(&mut self, f)` | All hash maps + FlatBTree, `Map` trait |
| Entry: `and_modify()` / `or_insert_with_key()` / `into_key()` | All 6 map types |
| `pop_first()` / `pop_last()` | FlatBTree, `SortedMap` trait |
| `SortedMap` for `std::BTreeMap` | `pop_first` / `pop_last` added |

## Open — Hash Maps

### API Completeness

| Item | Difficulty | Notes |
|------|-----------|-------|
| ~~`try_insert()`~~ | ~~Low~~ | ✅ Done — all 6 designs, OptiMap, OptiSortedMap, Map trait (with default impl). Returns `Result<(), OccupiedError<K, V>>`. |
| ~~`into_keys()` / `into_values()`~~ | ~~Low~~ | ✅ Done — all 6 designs, OptiMap, OptiSortedMap, Map trait. |
| `raw_entry()` API | Medium | Custom key lookup by hash + eq. Niche. |

### Performance

| Item | Difficulty | Notes |
|------|-----------|-------|
| Eliminate Borrow indirection in insert/entry | Medium | `find_by_hash_eq(&K)` that compares directly. |
| Large-value insert regression (Splitsies 128B+) | Medium | 1.48-1.65x slower than hashbrown. Needs investigation. |

### Testing / Quality

| Item | Difficulty | Notes |
|------|-----------|-------|
| Miri testing | Low-Medium | Verify no UB. Needs scalar fallback for SIMD intrinsics. |
| ~~Fuzzing harness~~ | ~~Low~~ | ✅ Done — proptest differential tests + cargo-fuzz targets for all 6 designs. See [Testing & Fuzzing](testing.md). |
| Allocator stress testing | Low | Custom allocator for misalignment and leak tracking. |

### Structural (Speculative)

| Item | Difficulty | Risk | Notes |
|------|-----------|------|-------|
| Interleaved memory layout | High | High | Better spatial locality, but large bucket types push groups apart. |
| Generic group size | High | Unclear | `GROUP_SIZE` as const generic. |
| Concurrent / lock-free variant | Very High | Research | Overflow bits are suited to lock-free reads. |

## Open — FlatBTree

### Performance

| Item | Difficulty | Notes |
|------|-----------|-------|
| Remove rebalancing (steal/merge) | Medium | Currently lazy (no rebalancing on remove). Tree stays valid but wastes memory under heavy churn. Low-watermark nodes are never reclaimed. |
| VacantEntry re-search elimination | Medium | `VacantEntry::insert` currently re-searches after inserting to find the value reference. Should return the position directly from the insert path. Affects entry API / counting workload perf (~1.08x vs BTreeMap). |
| Child node prefetching | Low | Prefetch next child's cache lines during internal node scan. Already faster than BTreeMap — diminishing returns. |

### API Completeness

| Item | Difficulty | Notes |
|------|-----------|-------|
| `range_mut()` | Low-Medium | Mutable range iteration. |
| ~~`into_keys()` / `into_values()`~~ | ~~Low~~ | ✅ Done — inherent methods + Map trait. |
| Arena `shrink_to_fit()` | Medium | Current impl is a no-op. Compact the arena requires rebuilding the tree to eliminate free-list gaps. Bulk-load from drain could work. |

### Testing / Quality

| Item | Difficulty | Notes |
|------|-----------|-------|
| Miri testing | High | FlatBTree has extensive unsafe pointer arithmetic in node.rs and raw.rs. Miri validation is critical. |
| ~~Fuzz against BTreeMap~~ | ~~Low-Medium~~ | ✅ Done — proptest + cargo-fuzz differential tests vs std::BTreeMap. See [Testing & Fuzzing](testing.md). |

## Closed

These have been extensively tested and proven structural. See
[Closed Investigations](optimization/closed.md) for full documentation.

| Item | Why Closed |
|------|-----------|
| Lookup hit gap (1.11-1.25x) | Per-probe instruction count is inherent to overflow-bit design. 7 attempts across 2 designs, all failed or traded hit for miss. |
| Selective prefetch policy | No universal policy exists. Design selection (IPO vs Splitsies vs UFM) is the prefetch policy. |
| AVX2/AVX-512 for 16-slot groups | 93%+ of probes resolve in home group (one SSE2 load). AVX2 targets the wrong bottleneck. Implemented for IPO64 only. |
| Dense iteration fast path | `tzcnt` + `blsr` is already ~2 cycles/element. Extra branch per `next()` caused +33% regression. |
| Custom Iterator::fold | Nested closure chain generates worse code than default `next()`-based fold. +5-18% regression. |
| #[inline] on entry API | Helps hit-heavy (-7%), hurts insert-heavy (+31%). Compiler heuristics are correct. |
| Inline find_by_hash + cold continuation | Register pressure at `#[inline(never)]` boundary. +10-14% regression on 2 designs. |
