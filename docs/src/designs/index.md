# Design Overview

OptiMap provides five hash map implementations and one sorted map (B+ tree).

## Hash Maps

The five hash map implementations share these properties:

- SIMD-accelerated metadata probing (SSE2/NEON/scalar fallback)
- Quadratic probe sequence over power-of-two group counts
- foldhash as default hasher (avalanching, fast)
- 70% default load factor
- Generic `Map` trait for uniform benchmarking and generic code

The designs split into two families:

## Overflow-Bit Family (tombstone-free)

These store an overflow byte per group that tracks which hash classes were displaced.
Misses terminate in O(1) — if the overflow bit for your hash class isn't set,
no element with that hash was ever displaced from this group.

Deletion is tombstone-free: set the slot to EMPTY and (if displaced) decrement
the anti-drift counter. No performance degradation under churn.

| Design | Group Size | Bucket Addressing | Trade-off |
|--------|-----------|-------------------|-----------|
| [UnorderedFlatMap](unordered_flat_map.md) | 15 slots | `gi * 15 + si` (multiply) | Original design, proven |
| [Splitsies](splitsies.md) | 16 slots | `(gi << 4) \| si` (shift) | Faster arithmetic, separate overflow array |
| [Gaps](gaps.md) | 15 slots | `(gi << 4) \| si` (shift) | UFM + power-of-2 buckets (wastes 1/16 slots) |

## Tombstone Family (Swiss-table style)

These use EMPTY/TOMBSTONE sentinels like hashbrown. Misses scan until EMPTY.
Key advantage over hashbrown: 254 hash values (8-bit, reserving only 0x00 and 0x01)
vs hashbrown's 128 (7-bit h2), giving ~2x fewer false-positive SIMD matches.

| Design | Group Size | Key Idea | Trade-off |
|--------|-----------|----------|-----------|
| [InPlaceOverflow](in_place_overflow.md) | 16 slots | Swiss-table + 8-bit hash | Best lookup hit, needs periodic rehash |
| [IPO64](ipo64.md) | 64 slots | Cache-line groups, AVX-512 | Flat degradation at extreme load, slower per-probe |

## Performance Comparison

| Property | UFM | Splitsies | IPO | hashbrown |
|----------|:---:|:---------:|:---:|:---------:|
| Hash values | 255 | 255 | **254** | 128 |
| Tombstone-free | yes | yes | no | no |
| O(1) miss termination | yes | yes | no | no |
| Power-of-2 addressing | no | yes | yes | yes |
| Full 16-bit SIMD mask | no | yes | yes | yes |
| Single-instruction empty check | no | no | no | yes |
| Miss at high load (85%) | **0.52x** | **0.28x** | ~1x | 1x |
| Churn performance | **0.62x** | **0.62x** | ~1x | 1x |
| Insert (fresh) | 0.87x | 0.95x | **0.85x** | 1x |
| Lookup hit (medium) | 1.20x | 1.11x | **1.01x** | 1x |

Ratios are vs hashbrown; <1.0 = faster than hashbrown.

## Sorted Map

[**FlatBTree**](flat_btree.md) is a B+ tree with 256-byte nodes (4 cache lines),
arena-allocated with a doubly-linked leaf chain. It provides sorted iteration,
range queries, and O(log n) lookup. Not a hash map — uses `K: Ord` instead of
hashing. Faster than `std::BTreeMap` on most operations (iteration 1.6-2x,
range queries 1.3-1.5x, remove 1.5-2.3x, clone 2-5x).

## Trait Hierarchy

### `Map<K: Hash + Eq, V>` — Core interface (all 6 designs + hashbrown + std)

| Category | Methods |
|----------|---------|
| Construction | `new()`, `with_capacity()` |
| Lookup | `get()`, `get_key_value()`, `get_mut()`, `contains_key()` |
| Mutation | `insert()`, `remove()`, `remove_entry()` |
| Capacity | `len()`, `is_empty()`, `capacity()`, `reserve()`, `shrink_to_fit()` |
| Bulk | `clear()`, `retain()`, `drain()` |
| Iteration | `iter()`, `iter_mut()`, `keys()`, `values()`, `values_mut()` |

### `SortedMap<K, V>` — Ordered operations (FlatBTree + std::BTreeMap)

| Method | Description |
|--------|-------------|
| `first_key_value()` / `last_key_value()` | Peek at min/max |
| `pop_first()` / `pop_last()` | Remove and return min/max |
| `iter_sorted()` | Sorted iteration |
| `range(bounds)` | Range queries |

### Entry API (all 6 designs)

`or_insert()`, `or_insert_with()`, `or_insert_with_key()`, `or_default()`,
`key()`, `and_modify()` on `Entry`. `get()`, `get_mut()`, `insert()`,
`into_mut()`, `key()` on `OccupiedEntry`. `insert()`, `key()`, `into_key()`
on `VacantEntry`.
