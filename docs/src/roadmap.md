# Future Work

Ordered roughly by expected impact. Items in the "Closed" section have been
thoroughly investigated and proven unproductive — see
[Closed Investigations](optimization/closed.md) for details.

## Recently Completed

### API Completeness (April 2025)

| Item | Scope |
|------|-------|
| `try_insert()` | All 6 designs, OptiMap, OptiSortedMap, `Map` trait (default impl). Returns `Result<(), OccupiedError<K, V>>`. |
| `into_keys()` / `into_values()` | All 6 designs, OptiMap, OptiSortedMap, FlatBTree, `Map` trait |
| `get_key_value()` / `remove_entry()` | All maps, `Map` trait |
| `iter_mut()` / `keys()` / `values()` / `values_mut()` | All maps, `Map` trait (defaults for `keys`/`values`/`values_mut`) |
| `reserve()` / `shrink_to_fit()` | All hash maps + FlatBTree, `Map` trait |
| `drain()` iterator | All hash maps + FlatBTree, `Map` trait |
| `retain(&mut self, f)` | All hash maps + FlatBTree, `Map` trait |
| Entry: `and_modify()` / `or_insert_with_key()` / `into_key()` | All 6 map types |
| `pop_first()` / `pop_last()` | FlatBTree, `SortedMap` trait |
| `SortedMap` for `std::BTreeMap` | `pop_first` / `pop_last` added |
| Enum iterators for OptiMap | Replaced `Box<dyn Iterator>` — zero-cost dispatch for `Iter`, `IterMut`, `IntoIter` |
| OptiSet / OptiSortedMap / OptiSortedSet | Smart wrappers with dynamic backend selection and sorted ops |
| Set benchmarks | Insert, contains, remove, iter, churn across all 8 set types |
| OptiMap Entry API | Enum `Entry`/`OccupiedEntry`/`VacantEntry` wrapping all 5 backends with `entry_match!` macro dispatch. Also added `OccupiedEntry::key()` to all backends. |
| FlatBTree VacantEntry direct return | `insert_at_vacant()` returns `(leaf_idx, slot_idx)` directly — no re-search needed. Entry counting workload now within ~2% of BTreeMap. |

## Open — Hash Maps

### API Completeness

| Item | Difficulty | Notes |
|------|-----------|-------|
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
| Child node prefetching | Low | Prefetch next child's cache lines during internal node scan. Already faster than BTreeMap — diminishing returns. |

### API Completeness

| Item | Difficulty | Notes |
|------|-----------|-------|
| `range_mut()` | Low-Medium | Mutable range iteration. |
| Arena `shrink_to_fit()` | Medium | Current impl is a no-op. Compaction requires rebuilding the tree to eliminate free-list gaps. Bulk-load from drain could work. |

### Testing / Quality

| Item | Difficulty | Notes |
|------|-----------|-------|
| Miri testing | High | FlatBTree has extensive unsafe pointer arithmetic in node.rs and raw.rs. Miri validation is critical. |

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
