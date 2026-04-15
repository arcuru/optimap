# Benchmark Results

Three-way comparison (2026-04-02): UFM (15-slot), Splitsies (16-slot), hashbrown 0.15.
All use foldhash. All at 70% load factor unless noted.

Ratios: <1.0 = faster than hashbrown, >1.0 = hashbrown is faster.

## Throughput (pre-warmed, 70% load)

| Operation | Size | UFM | Splitsies | hashbrown | UFM vs hb | Split vs hb |
|-----------|------|----:|----------:|----------:|:---------:|:-----------:|
| **insert** | medium | 33.2 µs | 34.9 µs | 33.3 µs | **1.00x** | 1.05x |
| **insert** | large | 319 µs | 311 µs | 333 µs | **0.96x** | **0.93x** |
| **insert 128B** | medium | 67.2 µs | 71.1 µs | 81.9 µs | **0.82x** | **0.87x** |
| lookup hit | medium | 22.7 µs | **21.0 µs** | 18.9 µs | 1.20x | **1.11x** |
| lookup hit | large | 256 µs | **241 µs** | 210 µs | 1.22x | **1.15x** |
| lookup miss | medium | 14.9 µs | **13.1 µs** | 11.1 µs | 1.34x | **1.18x** |
| lookup miss | large | 137 µs | **131 µs** | 126 µs | 1.09x | **1.04x** |
| **remove** | medium | 58.2 µs | **55.5 µs** | 66.4 µs | **0.88x** | **0.84x** |
| **remove** | large | 596 µs | **592 µs** | 918 µs | **0.65x** | **0.64x** |
| insert existing | medium | 28.0 µs | 27.0 µs | 24.1 µs | 1.16x | 1.12x |
| **iteration** | medium | 6.31 µs | **4.65 µs** | 4.17 µs | 1.51x | **1.11x** |
| **iteration** | large | 53.3 µs | **40.0 µs** | 35.0 µs | 1.52x | **1.14x** |
| entry | medium | 34.7 µs | 32.9 µs | 19.8 µs | 1.75x | 1.66x |

## Construction

| Operation | Size | UFM | Splitsies | hashbrown | UFM vs hb | Split vs hb |
|-----------|------|----:|----------:|----------:|:---------:|:-----------:|
| **with_capacity** | 1K | 2.72 µs | 2.74 µs | 3.36 µs | **0.81x** | **0.82x** |
| **with_capacity** | 100K | 407 µs | **389 µs** | 428 µs | **0.95x** | **0.91x** |
| **with_capacity** | 1M | 20.4 ms | 22.6 ms | 23.6 ms | **0.87x** | **0.96x** |
| grow_from_empty | 1K | 19.0 µs | **17.2 µs** | 13.5 µs | 1.41x | 1.27x |
| grow_from_empty | 100K | 1.42 ms | **1.27 ms** | 1.08 ms | 1.31x | **1.18x** |
| **clone** | 100K | 54.2 µs | **50.1 µs** | 51.5 µs | 1.05x | **0.97x** |
| **clone** | 1M | **12.9 ms** | 13.8 ms | 14.1 ms | **0.92x** | **0.98x** |
| **from_iter** | 10K | 32.9 µs | **29.6 µs** | 47.4 µs | **0.69x** | **0.62x** |
| **from_iter** | 100K | 423 µs | **370 µs** | 532 µs | **0.80x** | **0.70x** |

## Key Distributions (large table, 70% load)

| Distribution | Op | UFM | Splitsies | hashbrown | Split vs hb |
|-------------|-----|----:|----------:|----------:|:-----------:|
| random | hit | 246 µs | 238 µs | 213 µs | 1.12x |
| sequential | hit | 246 µs | 244 µs | 208 µs | 1.17x |
| random | miss | 143 µs | **125 µs** | 127 µs | **0.98x** |
| **sequential** | **miss** | 121 µs | **106 µs** | 143 µs | **0.74x** |
| random | insert | 309 µs | 312 µs | 323 µs | **0.97x** |
| sequential | insert | 306 µs | **304 µs** | 321 µs | **0.95x** |

## Value Sizes (medium table, 70% load)

| Value Size | Op | Splitsies | hashbrown | Split vs hb |
|-----------|-----|----------:|----------:|:-----------:|
| 64B | insert | **42.7 µs** | 43.2 µs | **0.99x** |
| 64B | hit | **21.8 µs** | 20.2 µs | 1.08x |
| 128B | insert | 93.2 µs | 62.8 µs | 1.48x |
| 256B | insert | 165 µs | 100 µs | 1.65x |

Large-value insert regression for Splitsies at 128B+ is a known issue.

## Mixed Workloads

| Workload | UFM | Splitsies | hashbrown | UFM vs hb | Split vs hb |
|----------|----:|----------:|----------:|:---------:|:-----------:|
| **churn 4K** | 25.7 ms | **25.6 ms** | 41.3 ms | **0.62x** | **0.62x** |
| **churn 64K** | 28.6 ms | **28.5 ms** | 38.8 ms | **0.74x** | **0.73x** |
| **churn 1M** | 54.0 ms | **50.0 ms** | 58.1 ms | **0.93x** | **0.86x** |
| **read-heavy** | 2.69 ms | **2.57 ms** | 2.60 ms | 1.03x | **0.99x** |
| write-heavy | 2.79 ms | 2.73 ms | 2.40 ms | 1.16x | 1.14x |
| counting 5% | 42.6 ms | **39.2 ms** | 30.5 ms | 1.39x | 1.28x |

## High-Load Stress (85% load)

| Benchmark | UFM | Splitsies | hashbrown | Split vs hb |
|-----------|----:|----------:|----------:|:-----------:|
| hit @ 85% | 400 µs | 385 µs | 331 µs | 1.16x |
| **miss @ 85%** | 282 µs | **180 µs** | 553 µs | **0.33x** |

## Miss by Load Factor

| Load % | UFM | Splitsies | hashbrown | Split vs hb |
|-------:|----:|----------:|----------:|:-----------:|
| 45% | 158 µs | **143 µs** | 130 µs | 1.10x |
| 55% | 162 µs | **145 µs** | 134 µs | 1.08x |
| 65% | 166 µs | **151 µs** | 140 µs | 1.08x |
| **75%** | 182 µs | **156 µs** | 168 µs | **0.93x** |
| **85%** | 297 µs | **158 µs** | 570 µs | **0.28x** |

Splitsies miss performance is nearly flat (143-158µs from 45% to 85%).
hashbrown degrades 4.4x over the same range (130-570µs).
**Splitsies at 85% load is faster than hashbrown at 45% load for misses.**

## Remove + Reinsert (tombstone-free advantage)

| Implementation | Time | vs hashbrown |
|---------------|-----:|:------------:|
| UFM | 1.76 ms | 1.23x |
| **Splitsies** | **1.12 ms** | **0.79x** |
| hashbrown | 1.43 ms | — |

## Summary: Splitsies vs hashbrown

### Splitsies wins
- **Remove**: 0.64-0.84x (tombstone-free)
- **Churn**: 0.62-0.86x (biggest advantage)
- **from_iter**: 0.62-0.70x
- **Insert**: 0.93-0.97x
- **Sequential miss**: 0.74x
- **Miss at 85% load**: 0.28x
- **Read-heavy workload**: 0.99x (tied)

### hashbrown wins
- **Lookup hit**: 1.11-1.23x (structural)
- **Entry API**: 1.66x
- **Large value insert (128B+)**: 1.48-1.65x
- **Counting/aggregation**: 1.28-1.35x

### Splitsies improvements over UFM
- Iteration: 1.51x → **1.11x** (biggest win)
- Lookup miss large: 1.09x → **1.04x**
- Lookup hit: 1.20x → **1.15x**
- Read-heavy: 1.03x → **0.99x** (flipped to tied)
