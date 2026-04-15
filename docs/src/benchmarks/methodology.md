# Benchmark Methodology

## Design Principles

1. **No allocation conflation**: Throughput benchmarks use pre-allocated, page-faulted
   tables. `clear()` + re-insert between iterations. Construction benchmarks
   intentionally include allocation overhead.
2. **Controlled load factors**: Fixed table geometry, filled to target percentage.
   Default: 70%. Fixed-N benchmarks land at arbitrary load factors and are unreliable
   for miss comparisons.
3. **Multiple key distributions**: Random, sequential, byte-swapped. Not just random u64.
4. **Multiple value sizes**: 8B, 64B, 128B, 256B.
5. **hashbrown-only comparison**: std::HashMap and indexmap are not relevant competitors.

## Table Sizes

Two standard sizes, chosen for cache hierarchy alignment:

| Size | Slots | Groups | Metadata | Buckets (u64/u64) |
|------|------:|-------:|---------:|------------------:|
| Medium | 15,360 | 1,024 | 16KB (fits L1) | 240KB (fits L2) |
| Large | 122,880 | 8,192 | 128KB (fits L2) | 1.9MB (exceeds L2) |

At 70% load: medium = ~10,752 entries, large = ~86,016 entries.

## Benchmark Suite

```
benches/
  throughput.rs      — Core single-operation throughput (pre-allocated, controlled LF)
  construction.rs    — Allocation, growth, clone, drop costs
  distributions.rs   — Key distribution and value size sensitivity
  workloads.rs       — Realistic mixed-operation scenarios
  load_factor.rs     — Load factor sensitivity sweeps
```

### throughput.rs

Pre-warmed tables. Measures pure operation cost.

| Benchmark | What it measures |
|-----------|-----------------|
| insert | Insert into pre-warmed table |
| lookup_hit | Successful lookup |
| lookup_miss | Failed lookup |
| remove | Remove all keys from full table |
| insert_existing | Overwrite existing keys |
| iteration | Full table scan |
| entry_or_insert | `entry().or_insert()` on occupied keys |

### construction.rs

Includes allocation overhead. Measures costs that happen once per table lifetime.

| Benchmark | What it measures |
|-----------|-----------------|
| grow_from_empty | Insert N into `::new()` |
| insert_with_capacity | Insert N into `with_capacity(N)` |
| clone | Clone a full table |
| from_iter | `collect()` from iterator |

### distributions.rs

All pre-allocated at 70% load. Tests key distribution and value size sensitivity.

### workloads.rs

Realistic mixed scenarios: equilibrium churn, read-heavy (95/5), write-heavy (50/30/20),
counting/aggregation, post-delete lookup, miss ratio sweep.

## Allocation Overhead at 1M

At 1M elements, `with_capacity` allocates ~32MB via `mmap`. The kernel lazily
zero-fills ~7,680 pages on first write, at ~1.5µs per fault = **~11ms of page
fault overhead per iteration**.

This affects both us and hashbrown equally. The `insert_prealloc` benchmark
isolates true insert throughput:

| Config | ours | hashbrown | ratio |
|--------|-----:|----------:|:-----:|
| insert_u64 1M (alloc per iter) | 20.8 ms | 25.3 ms | 0.82x |
| insert_prealloc 1M (no alloc) | **9.5 ms** | 12.0 ms | **0.79x** |

This also explains why "single allocation regression" tests showed +40-108% at 1M:
the benchmark was measuring allocation strategy differences (glibc arena caching for
smaller allocs vs mmap/munmap for one large alloc), not insert performance.

## RNG

All benchmarks use SFC64 RNG with checksummed outputs (Ankerl methodology).

## Load Factor Sensitivity

Our table uses 15-slot groups. With power-of-two group counts, the actual load
factor at any given size depends on where we sit between rehashes:

- Right after a rehash (capacity doubles): ~44% load
- Right before next rehash: ~87.5% load

The load_factor.rs benchmark isolates this variable by pre-allocating a fixed
capacity and filling to controlled percentages. Key finding:

- **Hit performance is flat** across load factors for both implementations
- **Miss crossover** at ~70-75% load (100K scale) and ~45% load (1M scale)
- At 85% load, overflow-bit designs are 1.8-3.6x faster on misses
