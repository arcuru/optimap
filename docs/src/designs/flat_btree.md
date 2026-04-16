# FlatBTree

Cache-line-optimized B+ tree. Unlike the other designs (which are hash maps),
FlatBTree is a sorted data structure providing ordered iteration and range queries.

## Memory Layout

```text
Arena: contiguous slab of 256-byte-aligned node blocks
Nodes referenced by u32 index (NodeIdx), not pointers

Leaf Node (256 bytes):
[Header 8B][keys: K × LEAF_CAP][values: V × LEAF_CAP][prev: u32][next: u32]

Internal Node (256 bytes):
[Header 8B][keys: K × INTERNAL_CAP][children: u32 × (INTERNAL_CAP + 1)]
```

## Key Properties

- **Node size**: 256 bytes (4 cache lines on x86-64)
- **Structure**: B+ tree (values only in leaves, internal nodes have keys + child pointers)
- **Leaf chain**: Doubly-linked for O(n) sorted iteration without visiting internal nodes
- **Search**: Linear scan within nodes (no SIMD, no binary search)
- **Arena allocated**: All nodes in a contiguous slab, referenced by u32 indices
- **Generic capacity**: Computed at compile time from `size_of::<K>()` and `size_of::<V>()`

## Why Linear Scan

At typical fan-outs (15-30 keys per node), linear scan of 1-2 cache lines is
competitive with binary search. The CPU prefetcher loads cache lines 2-4 while
scanning line 1. By the time you find the target child pointer, the remaining
data is already in L1. The first key is at byte 8 (after the header) — within
the first cache line.

## Node Capacities

| K | V | Leaf Cap | Internal Cap |
|---|---|----------|-------------|
| u64 (8B) | u64 (8B) | 15 | 20 |
| u32 (4B) | u32 (4B) | 30 | 30 |
| String (24B) | String (24B) | 5 | 8 |
| u64 (8B) | u128 (16B) | 10 | 20 |

Minimum: LEAF_CAP >= 1, INTERNAL_CAP >= 2. Compile-time assertion fires
for oversized types.

## Map Trait Compatibility

FlatBTree implements the `Map` trait, but with a caveat: the trait's lookup
methods (`get`, `remove`) require `Q: Hash + Eq`, not `Q: Ord`. Since the
B-tree needs Ord to navigate, the trait methods fall back to an O(n) leaf
chain scan using Eq.

For O(log n) performance, use FlatBTree's inherent methods directly (which
require `Q: Ord`). The Map trait impl exists for generic code and benchmark
infrastructure.

## SortedMap Trait

FlatBTree also implements the `SortedMap` trait, which provides:
- `first_key_value()` / `last_key_value()` — O(1)
- `pop_first()` / `pop_last()` — O(log n), remove and return min/max
- `range(bounds)` — O(log n + k) where k is the number of results
- `iter_sorted()` — O(n) sorted iteration

## Design Trade-offs vs Hash Maps

| Property | FlatBTree | Hash Maps (Splitsies, etc.) |
|----------|:---------:|:--------------------------:|
| Lookup | O(log n) | O(1) amortized |
| Insert | O(log n) | O(1) amortized |
| Sorted iteration | yes | no |
| Range queries | yes | no |
| Cache misses per lookup | O(log_B n) | 1-2 |
| Delete | O(log n), lazy | O(1) |

FlatBTree is not a replacement for hash maps. Use it when you need sorted
iteration or range queries. For unordered key-value storage, the hash map
designs are faster.

## Benchmark Results vs std::BTreeMap

All benchmarks use `u64` keys and values. Ratios <1.0 = FlatBTree is faster.

### Core Operations

| Operation | 1K | 10K | 100K |
|-----------|:--:|:---:|:----:|
| Insert (random) | 1.10x | **0.97x** | **0.92x** |
| Insert (sorted) | **0.34x** | **0.25x** | **0.31x** |
| Lookup hit | **0.95x** | **0.76x** | **0.78x** |
| Lookup miss | 1.04x | **0.80x** | **0.78x** |
| Remove | **0.44x** | **0.66x** | **0.67x** |
| Iteration | **0.48x** | **0.62x** | **0.64x** |
| Clone | **0.22x** | **0.30x** | **0.44x** |

### Sorted Operations

| Operation | Size | FlatBTree | BTreeMap |
|-----------|-----:|:---------:|:--------:|
| Range query (10% of keyspace) | 10K | **0.76x** | 1x |
| Range query (10% of keyspace) | 100K | **0.66x** | 1x |
| first_key_value | any | **~0.3x** | 1x |
| last_key_value | any | **~0.3x** | 1x |

### Mixed Workloads

| Workload | Ratio |
|----------|:-----:|
| Read-heavy (80/15/5) | **0.83x** |
| Counting (entry API) | 1.08x |

### Key Advantages

- **Sorted insert 3-4x faster**: append fast-path skips tree search when
  key > all existing keys. Symmetric prepend fast-path for reverse order.
- **Iteration 1.6-2x faster**: leaf chain traversal, no tree navigation.
- **Clone 2-5x faster**: bulk arena copy + in-place value cloning.
- **Remove 1.5-2.3x faster**: lazy removal (shift within leaf, no rebalancing).

### Known Gaps

- **Counting/entry API ~1.08x slower**: VacantEntry re-searches after insert.
- **Random insert at 1K ~1.10x slower**: tree overhead for small sizes.
