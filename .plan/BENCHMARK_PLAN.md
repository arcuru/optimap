# Benchmark Restructure Plan

## Motivation

The existing benchmarks (hashmap_comparison.rs, comprehensive.rs) have several issues:
1. **Allocation conflation**: Most benchmarks create+drop the map per criterion iteration. At 1M entries, ~11ms of OS page fault overhead dominates the measurement.
2. **Arbitrary load factors**: Fixed sizes (1K/10K/100K/1M) land at different load factors. Miss performance varies 2x across load factors, making fixed-N comparisons unreliable.
3. **Missing key distributions**: Only random u64 tested. No sequential, byte-swapped, or zipf keys.
4. **Missing value sizes**: Only 8-byte values. Real workloads have larger values.
5. **Unnecessary comparisons**: std::HashMap and indexmap are not relevant competitors.

## New Structure

```
benches/
  throughput.rs      — Core single-operation throughput (pre-allocated, controlled LF)
  construction.rs    — Allocation, growth, clone, drop costs
  distributions.rs   — Key distribution and value size sensitivity
  workloads.rs       — Realistic mixed-operation scenarios
  load_factor.rs     — Keep existing (already good)
```

## Shared Infrastructure

All files use the same `Sfc64` RNG (copied per file — criterion bench files are independent crates).

`table_geometry(capacity)` helper replicates the group-count calculation to compute exact entry counts for target load factors.

Two standard table sizes:
- **Medium**: 15,360 slots (1024 groups). Metadata = 16KB (fits L1). Buckets at u64/u64 = 240KB (fits L2).
- **Large**: 122,880 slots (8192 groups). Metadata = 128KB (fits L2). Buckets at u64/u64 = 1.9MB (exceeds L2).

Default target load factor: **70%** (representative mid-point, near our miss crossover with hashbrown).

## File 1: throughput.rs

Core operations, pre-allocated, page-faulted tables. Every benchmark uses `clear()` + re-insert. Maps are pre-warmed before the benchmark loop.

| Benchmark | What it measures | Sizes | Key/Value |
|-----------|-----------------|-------|-----------|
| insert | Insert into pre-warmed table | medium, large | u64/u64, u64/[u8;128] |
| lookup_hit | Successful lookup | medium, large | u64/u64, u64/[u8;128] |
| lookup_miss | Failed lookup | medium, large | u64/u64 |
| remove | Remove all keys from full table | medium, large | u64/u64 |
| insert_existing | Overwrite existing keys | medium, large | u64/u64 |
| iteration | Full table scan | medium, large | u64/u64 |
| entry_or_insert | entry().or_insert() on occupied keys | medium | u64/u64 |

All compare against hashbrown only. All at 70% load factor.

## File 2: construction.rs

Intentionally includes allocation overhead. Measures costs that happen once per table lifetime.

| Benchmark | What it measures | Sizes |
|-----------|-----------------|-------|
| grow_from_empty | Insert N into ::new() | 1K, 10K, 100K, 1M |
| insert_with_capacity | Insert N into with_capacity(N) | 1K, 10K, 100K, 1M |
| clone | Clone a full table | 1K, 100K, 1M |
| from_iter | collect() from iterator | 10K, 100K |

## File 3: distributions.rs

Key distribution and value size sensitivity. All pre-allocated at 70% load.

| Benchmark | Variable | Fixed at |
|-----------|----------|---------|
| lookup_hit_by_distribution | random, sequential, byte-swapped | large, u64/u64 |
| lookup_miss_by_distribution | random, sequential, byte-swapped | large, u64/u64 |
| insert_by_distribution | random, sequential, byte-swapped | large, u64/u64 |
| string_key_sizes | 7b, 8b, 13b, 24b, 100b | medium, String/u64 |
| value_size_sensitivity | u64, [u8;64], [u8;128], [u8;256] | medium, insert+lookup |

## File 4: workloads.rs

Realistic mixed scenarios.

| Benchmark | Pattern | Sizes |
|-----------|---------|-------|
| equilibrium_churn | Insert+remove at steady state | 4K, 64K, 1M equilibrium |
| read_heavy | 95% read (80% hit, 15% miss), 5% write | large, 500K ops |
| write_heavy | 50% read, 30% insert, 20% remove | large, 500K ops |
| counting | entry().or_insert(0) += 1 | 5%, 50%, 100% distinct, 5M ops |
| post_delete_lookup | Remove half, then lookup all | medium, large |
| miss_ratio_sweep | 0%, 25%, 50%, 75%, 100% miss | large |

## Dropped Benchmarks

- std::HashMap and indexmap comparisons (not useful competitors)
- ahash benchmarks (foldhash is the standard)
- insert_erase_phases (conflates too many things)
- growing_lookup (unusual pattern; read_heavy is more representative)
- iterate_grow_shrink (pathological pattern)

## Implementation Order

1. throughput.rs (establishes patterns)
2. distributions.rs
3. workloads.rs
4. construction.rs
5. Update Cargo.toml
6. Delete old benchmark files
7. Verify with cargo bench
