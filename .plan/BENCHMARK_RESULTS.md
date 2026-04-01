# Benchmark Results (2026-04-01)

All benchmarks at 70% load factor unless noted. Pre-warmed tables for
throughput tests. Compared against hashbrown 0.15 with foldhash.

Ratios: <1.0 means we're faster, >1.0 means hashbrown is faster.

## Throughput (pre-warmed, 70% load)

Medium = 10,752 entries in 15,360 slots (1,024 groups).
Large = 86,016 entries in 122,880 slots (8,192 groups).

| Operation | Size | ours | hashbrown | ratio |
|-----------|------|-----:|----------:|:-----:|
| **insert** | medium | 30.9 µs | 33.9 µs | **0.91x** |
| **insert** | large | 299 µs | 338 µs | **0.88x** |
| insert (128B val) | medium | 85.0 µs | 65.2 µs | 1.30x |
| insert (128B val) | large | 991 µs | 946 µs | 1.05x |
| lookup hit | medium | 23.9 µs | 18.8 µs | 1.27x |
| lookup hit | large | 258 µs | 209 µs | 1.24x |
| lookup miss | medium | 16.2 µs | 11.5 µs | 1.42x |
| lookup miss | large | 149 µs | 128 µs | 1.16x |
| **remove** | medium | 63.4 µs | 67.0 µs | **0.95x** |
| **remove** | large | 635 µs | 930 µs | **0.68x** |
| insert existing | medium | 27.7 µs | 25.3 µs | 1.09x |
| insert existing | large | 301 µs | 256 µs | 1.18x |
| iteration | medium | 6.5 µs | 4.2 µs | 1.54x |
| iteration | large | 61.3 µs | 34.6 µs | 1.77x |
| entry (occupied) | medium | 36.4 µs | 20.6 µs | 1.77x |

## Construction (includes allocation overhead)

| Operation | Size | ours | hashbrown | ratio |
|-----------|------|-----:|----------:|:-----:|
| **with_capacity** | 1K | 2.88 µs | 3.26 µs | **0.88x** |
| **with_capacity** | 10K | 31.8 µs | 33.9 µs | **0.94x** |
| **with_capacity** | 100K | 423 µs | 432 µs | **0.98x** |
| **with_capacity** | 1M | 20.9 ms | 23.8 ms | **0.88x** |
| grow_from_empty | 1K | 20.0 µs | 13.8 µs | 1.45x |
| grow_from_empty | 10K | 160 µs | 115 µs | 1.40x |
| grow_from_empty | 100K | 2.49 ms | 2.22 ms | 1.12x |
| grow_from_empty | 1M | 48.1 ms | 45.4 ms | 1.06x |
| **clone** | 1K | 594 ns | 553 ns | 1.07x |
| **clone** | 100K | 62.0 µs | 52.3 µs | 1.19x |
| **clone** | 1M | **13.3 ms** | 15.3 ms | **0.87x** |
| **from_iter** | 10K | 33.5 µs | 36.0 µs | **0.93x** |
| **from_iter** | 100K | 436 µs | 436 µs | 1.00x |

## Key Distributions (large table, 70% load)

| Distribution | Operation | ours | hashbrown | ratio |
|-------------|-----------|-----:|----------:|:-----:|
| **random** | **insert** | 300 µs | 329 µs | **0.91x** |
| **sequential** | **insert** | 289 µs | 327 µs | **0.88x** |
| **byteswapped** | **insert** | 302 µs | 336 µs | **0.90x** |
| random | hit | 263 µs | 212 µs | 1.24x |
| sequential | hit | 261 µs | 214 µs | 1.22x |
| byteswapped | hit | 256 µs | 206 µs | 1.24x |
| random | miss | 147 µs | 128 µs | 1.15x |
| sequential | miss | 140 µs | 122 µs | 1.14x |
| byteswapped | miss | 146 µs | 133 µs | 1.10x |

## Value Sizes (medium table, 70% load)

| Value Size | insert (ours) | insert (hb) | ratio | hit (ours) | hit (hb) | ratio |
|-----------|-------------:|------------:|:-----:|----------:|---------:|:-----:|
| 8B (u64) | 30.9 µs | 33.9 µs | **0.91x** | 23.9 µs | 18.8 µs | 1.27x |
| **64B** | **45.6 µs** | 61.9 µs | **0.74x** | 26.7 µs | 20.6 µs | 1.30x |
| **128B** | **69.9 µs** | 80.2 µs | **0.87x** | 31.8 µs | 25.9 µs | 1.23x |
| **256B** | **99.1 µs** | 119 µs | **0.83x** | 32.3 µs | 26.6 µs | 1.21x |

## String Key Sizes (medium table, lookup hit)

| Key Length | ours | hashbrown | ratio |
|-----------|-----:|----------:|:-----:|
| 7b | 60.0 µs | 56.1 µs | 1.07x |
| 8b | 59.1 µs | 54.1 µs | 1.09x |
| 13b | 60.0 µs | 54.6 µs | 1.10x |
| 24b | 73.1 µs | 72.8 µs | 1.00x |
| 100b | 130 µs | 130 µs | 1.00x |

## Mixed Workloads

| Workload | ours | hashbrown | ratio |
|----------|-----:|----------:|:-----:|
| **churn 4K** | **27.2 ms** | 45.6 ms | **0.60x** |
| **churn 64K** | **30.4 ms** | 40.8 ms | **0.74x** |
| **churn 1M** | **63.3 ms** | 68.4 ms | **0.93x** |
| read-heavy (95/5) | 2.75 ms | 2.67 ms | 1.03x |
| write-heavy (50/30/20) | 2.99 ms | 2.53 ms | 1.18x |
| counting 5% distinct | 42.2 ms | 30.3 ms | 1.39x |
| counting 50% distinct | 241 ms | 168 ms | 1.43x |
| counting 100% distinct | 244 ms | 193 ms | 1.26x |

## Post-Delete Lookup (50% removed, then lookup all)

| Size | ours | hashbrown | ratio |
|------|-----:|----------:|:-----:|
| medium | 19.0 µs | 12.6 µs | 1.51x |
| large | 187 µs | 149 µs | 1.25x |

## Miss Ratio Sweep (large table, 70% load)

| Miss % | ours | hashbrown | ratio |
|-------:|-----:|----------:|:-----:|
| 0% | 240 µs | 182 µs | 1.32x |
| 25% | 208 µs | 169 µs | 1.23x |
| 50% | 189 µs | 156 µs | 1.21x |
| 75% | 166 µs | 138 µs | 1.20x |
| 100% | 141 µs | 123 µs | 1.15x |

## Summary: Where We Win

- **Insert** (small values): 0.88-0.91x across all distributions
- **Insert** (large values 64-256B): 0.74-0.87x — fused insert avoids double data movement
- **Remove**: 0.68-0.95x — tombstone-free deletion
- **Churn** (insert+remove equilibrium): 0.60-0.93x — our biggest advantage
- **with_capacity + fill**: 0.88-0.98x at all sizes
- **Clone at 1M**: 0.87x
- **from_iter**: 0.93-1.00x
- **Read-heavy workload**: ~tied (1.03x)

## Summary: Where hashbrown Wins

- **Lookup hit**: 1.22-1.27x (structural, 15-slot group overhead)
- **Lookup miss** (at 70% load): 1.10-1.42x (below our crossover point)
- **Iteration**: 1.54-1.77x (sub-cycle per-element overhead)
- **Entry API** (occupied): 1.77x (hit overhead + enum construction)
- **Insert existing** (overwrite): 1.09-1.18x (similar to lookup hit)
- **Counting/aggregation**: 1.26-1.43x (entry API dominated)
- **Write-heavy mixed**: 1.18x
- **Grow from empty**: 1.06-1.45x (rehash cost)
