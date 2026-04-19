# Code Deduplication: GenericMap & RawTableApi

## Problem

The five hash map designs shared 95-100% of their code at two levels:

1. **Map wrapper** (~1000 lines each): constructors, entry API, iterators,
   `FromIterator`, `Clone`, `Debug`, `PartialEq`, `Index`, etc. — identical
   across all 5 designs except the struct name.

2. **Overflow-bit raw table** (~900 lines each): UFM, Splitsies, and Gaps had
   nearly identical probe loops, insert logic, allocation, rehash, and iteration.
   The only differences were parameterizable: overflow storage location, bucket
   stride, and SIMD bitmask width.

Total duplication: ~8,500 lines across 15 files.

## Solution: Two Generic Abstractions

### GenericMap<K, V, S, R: RawTableApi>

A single map wrapper that replaces all 5 `map.rs` files. Contains:

- Constructors (`new`, `with_capacity`, `with_hasher`, `with_capacity_and_hasher`)
- Core ops (`get`, `insert`, `remove`, `contains_key`, `get_key_value`, `get_mut`)
- Entry API (`Entry`, `OccupiedEntry`, `VacantEntry`)
- Iterators (`Iter`, `IterMut`, `IntoIter`, `Keys`, `Values`, `ValuesMut`)
- Trait impls (`Default`, `IntoIterator`, `FromIterator`, `Extend`, `Index`,
  `Debug`, `Clone`, `PartialEq`, `Eq`)
- Bulk ops (`retain`, `drain`, `reserve`, `shrink_to_fit`)

Each concrete map type is a type alias:

```rust
pub type UnorderedFlatMap<K, V, S> = GenericMap<K, V, S, RawTable<K, V, UfmLayout>>;
pub type Splitsies<K, V, S>       = GenericMap<K, V, S, RawTable<K, V, SplitsiesLayout>>;
pub type Gaps<K, V, S>            = GenericMap<K, V, S, RawTable<K, V, GapsLayout>>;
pub type InPlaceOverflow<K, V, S> = GenericMap<K, V, S, ipo::RawTable<K, V>>;
pub type IPO64<K, V, S>           = GenericMap<K, V, S, ipo64::RawTable<K, V>>;
```

### RawTableApi<K, V> — Internal Trait

The contract between GenericMap and each raw table backend. Key methods:

| Category | Methods |
|----------|---------|
| Construction | `new()`, `with_capacity()` |
| Queries | `len()`, `capacity()`, `is_allocated()`, `num_groups()` |
| Lookups | `find_bucket()`, `find_by_hash()` |
| Insert | `insert_or_replace()` (fused fast path), `insert_at()`, `insert_no_check()` |
| Entry | `find_for_entry()` (fused fast path), `ensure_capacity()` |
| Remove | `remove_by_hash()`, `erase_slot()` (design-specific cleanup) |
| Iteration | `iter_slots()`, `into_iter_impl()`, `drain_impl()` |
| Capacity | `reserve()`, `shrink_to_fit()`, `rehash_with()` |

Performance-critical methods (`insert_or_replace`, `find_for_entry`) include the
fused home-group SIMD fast path inside the raw table, not in GenericMap. Each
design's fast path is fully specialized via monomorphization.

### GroupLayout + overflow_table::RawTable<K, V, L>

A single generic overflow-bit raw table replaces three separate implementations.
The `GroupLayout` trait parameterizes:

| Axis | UFM | Splitsies | Gaps |
|------|-----|-----------|------|
| Usable slots | 15 | 16 | 15 |
| Bucket stride | 15 (`gi*15+si`) | 16 (`(gi<<4)\|si`) | 16 (`(gi<<4)\|si`) |
| SIMD mask | `0x7FFF` | `0xFFFF` | `0x7FFF` |
| Overflow location | Byte 15 of group | Separate array | Byte 15 of group |
| Extra allocation | 0 | `num_groups` bytes | 0 |
| Prefetch strategy | 2 prefetches/probe | 3 (extra for overflow) | 2 |

All constants and pointer arithmetic are resolved at compile time. The `GroupOps`
associated type on `GroupLayout` carries the SIMD operations parameterized by
slot mask, avoiding unstable `generic_const_exprs`.

## What Stays Separate

- **IPO and IPO64** keep their own `RawTable` implementations. Their probe
  strategy (tombstone-based, EMPTY termination) is fundamentally different from
  the overflow-bit family. They implement `RawTableApi` and use GenericMap for
  the wrapper layer.

- **FlatBTree** is a B+ tree, not a hash table. No overlap.

- **UnorderedFlatSet** (`set.rs`) is hand-written with direct UFM raw table
  access for SIMD fast paths. Uses the legacy UFM raw table in `raw/mod.rs`.

## Impact

| Metric | Before | After |
|--------|--------|-------|
| Map wrapper code | 5 x ~1000 lines | 1 x 740 lines |
| Overflow-bit raw tables | 3 x ~900 lines | 1 x 936 lines |
| Overflow-bit group ops | 3 x ~250 lines | 1 x 173 lines |
| New shared infrastructure | — | 371 lines (traits + layout) |
| **Total** | **~8,500 lines** | **~2,400 lines** |
| **Net reduction** | — | **~4,500 lines deleted (-72%)** |

Performance: zero-cost. All generics monomorphize to identical machine code.
Benchmarks show no systematic regressions (17 improvements vs 3 regressions
in full throughput suite, regressions attributable to measurement noise).

## Adding a New Overflow-Bit Design

To add a new overflow-bit variant (e.g., Splitsies-1bit):

1. Define a new layout struct implementing `GroupLayout` (~30 lines)
2. Add a type alias: `pub type Splitsies1Bit<K,V,S> = GenericMap<K,V,S, RawTable<K,V, Splitsies1BitLayout>>;`
3. Add `impl_map_trait!(Splitsies1Bit);` for the `Map` trait

That's it. No new probe loops, no new entry API, no new iterators.
