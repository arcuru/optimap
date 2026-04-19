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
The `GroupLayout` trait composes three strategy traits:

```rust
pub trait GroupLayout: 'static + Copy {
    type Grp: GroupOps;           // SIMD operations (slot mask)
    type Tag: TagStrategy;        // Hash tag + overflow channel extraction
    type Overflow: OverflowStrategy; // Overflow storage format

    const GROUP_SIZE: usize;      // 15 or 16
    const BUCKET_STRIDE: usize;   // 15 or 16
    const SEPARATE_OVERFLOW: bool; // Controls extra prefetch
    const AND_INDEX: bool;         // Group index method (see below)
}
```

Named layouts for existing designs:

| Axis | UFM | Splitsies | Gaps |
|------|-----|-----------|------|
| Usable slots | 15 | 16 | 15 |
| Bucket stride | 15 (`gi*15+si`) | 16 (`(gi<<4)\|si`) | 16 (`(gi<<4)\|si`) |
| SIMD mask | `0x7FFF` | `0xFFFF` | `0x7FFF` |
| Overflow location | Byte 15 of group | Separate array | Byte 15 of group |
| Extra allocation | 0 | `num_groups` bytes | 0 |
| Prefetch strategy | 2 prefetches/probe | 3 (extra for overflow) | 2 |

All constants and pointer arithmetic are resolved at compile time. The `GroupOps`
associated type carries the SIMD operations parameterized by slot mask, avoiding
unstable `generic_const_exprs`.

### Design matrix: tag × overflow × indexing

Beyond the three named designs, `Layout16<T, O>` and `Layout16And<T, O>` compose
any `TagStrategy` with any `OverflowStrategy`:

| Tag \ Overflow | 8-bit (ByteSeparate) | 1-bit (BitSeparate) | Tombstone (IPO) |
|---|---|---|---|
| LowByte255 | Splitsies (baseline) | Lo8_1bit | IPO (baseline) |
| HighByte255 | Hi8_8bit | Hi8_1bit | — |
| LowByte128 | Lo128_8bit | Lo128_1bit | — |
| LowByte254 | — | — | IPO (baseline) |
| HighByte128 | — | — | Hi128_Tomb |
| TopByte128 | — | — | Top128_Tomb |
| TopTag128 (AND) | — | Top128_1bitAnd | — |
| TopTag255 (AND) | — | Top255_1bitAnd | — |

AND-indexed variants use `Layout16And` which sets `AND_INDEX = true`. See
the "Group indexing" section below.

### Group indexing: shift vs AND

Hash tables map a hash value to a group index. Two strategies:

**Shift-based** (default): `gi = (h >> shift) & mask` — uses high hash bits.
Tags can safely use low bits (LowByte255, etc.) since they're decorrelated.
Costs 2 instructions (variable shift + AND).

**AND-based**: `gi = h & mask` — uses low hash bits. Saves 1 instruction
(just AND), but tags must come from top hash bits (57+) to avoid correlation.
Additionally, 8-bit overflow channels use `1 << (h & 7)` which also uses
low bits — every key in the same group would get the same channel, making
8-channel overflow useless. **AND indexing is only safe with 1-bit overflow
(BitSeparate) or tombstone designs (no overflow channels).**

### Memory layout: mid-pointer design

Both RawTable implementations use a mid-pointer allocation layout inspired
by hashbrown. A single `ctrl` pointer sits at the boundary between buckets
(backward) and metadata (forward):

```text
  Overflow-bit designs:
  ┌──────────────────────┬────────────────────┬───────────────┐
  │ Buckets (KV pairs)   │ Metadata (16B/grp) │ Overflow bytes│
  │ ◄── backward         │ forward ──►        │ forward ──►   │
  └──────────────────────┴────────────────────┴───────────────┘
  ↑ alloc_ptr (computed)  ↑ ctrl (stored)

  Tombstone designs (IPO):
  ┌──────────────────────┬────────────────────┐
  │ Buckets (KV pairs)   │ Metadata (16B/grp) │
  │ ◄── backward         │ forward ──►        │
  └──────────────────────┴────────────────────┘
  ↑ alloc_ptr (computed)  ↑ ctrl (stored)
```

- **Metadata**: `ctrl + gi * 16` (forward from ctrl)
- **Buckets**: `ctrl.cast::<(K,V)>().sub(slot_index + 1)` (backward from ctrl)
- **Overflow** (overflow-bit only): `ctrl + num_groups * 16 + offset` (forward, after metadata)

This eliminates a separate `buckets` pointer field, reducing the struct
from 7 fields to 5 (overflow-bit) or 5 (tombstone). Both metadata and
bucket access derive from `ctrl` in opposite directions, saving a register
and an address computation in the hot path. hashbrown uses the same trick.

Overflow-bit designs have 3 memory regions but only need 2 pointers worth
of addressing: the hot path (metadata + bucket) uses `ctrl`, and overflow
is computed as a forward offset from `ctrl` (only accessed on miss/insert).

## What Stays Separate

- **IPO and IPO64** keep their own `RawTable` implementations. Their probe
  strategy (tombstone-based, EMPTY termination) is fundamentally different from
  the overflow-bit family. IPO's `RawTable<K,V,T: TombstoneTag>` is parameterized
  by tag strategy and also uses the mid-pointer layout. They implement
  `RawTableApi` and use GenericMap for the wrapper layer.

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
