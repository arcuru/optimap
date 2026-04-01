# Benchmark Results

Both us and hashbrown use foldhash as the default hasher.
Benchmarks use SFC64 RNG and checksummed outputs (Ankerl methodology).

## Load Factor Analysis

Our table uses 15-slot groups with a fixed 87.5% max load factor.
With power-of-two group counts, the actual load factor at any given
size depends on where we sit between rehashes:

- Right after a rehash (capacity doubles): ~44% load
- Right before next rehash: ~87.5% load

Fixed-N benchmarks (e.g. "insert 10K") land at arbitrary load factors
depending on the exact N chosen. The load-factor sweep below isolates
this variable by pre-allocating a fixed capacity, then filling to a
controlled percentage.

### Lookup Hit by Load Factor (100K-slot table, 100K ops)
| Load % | ours | hashbrown | ratio |
|-------:|-----:|----------:|:-----:|
| 45% | 416 µs | 327 µs | 1.27x |
| 55% | 420 µs | 325 µs | 1.29x |
| 65% | 413 µs | 328 µs | 1.26x |
| 75% | 418 µs | 321 µs | 1.30x |
| 85% | 423 µs | 333 µs | 1.27x |

**Hit performance is flat across load factors** for both implementations.
The ~1.27x gap is structural (per-probe overhead) and doesn't change with load.

### Lookup Miss by Load Factor (100K-slot table, 100K ops)
| Load % | ours | hashbrown | ratio | winner |
|-------:|-----:|----------:|:-----:|:------:|
| 45% | 169 µs | 130 µs | 1.30x | hb |
| 55% | 175 µs | 139 µs | 1.26x | hb |
| 65% | 180 µs | 145 µs | 1.24x | hb |
| **75%** | **204 µs** | 252 µs | **0.81x** | **ours** |
| **85%** | **321 µs** | 564 µs | **0.57x** | **ours** |

**Crossover at ~70% load factor.** Below 70%, hashbrown's tighter probe
loop wins. Above 70%, our overflow bits terminate misses in O(1) while
hashbrown must probe until it finds an empty control byte — and at high
load, empty bytes are scarce.

At 85% load we're **1.76x faster** on misses.

### Mixed Workload by Load Factor (100K-slot table, 50% insert/30% lookup/20% remove)
| Load % | ours | hashbrown | ratio |
|-------:|-----:|----------:|:-----:|
| 45% | 531 µs | 444 µs | 1.20x |
| 55% | 537 µs | 443 µs | 1.21x |
| 65% | 549 µs | 456 µs | 1.20x |
| 75% | 580 µs | 482 µs | 1.20x |
| 85% | 629 µs | 548 µs | 1.15x |

Mixed workload: consistently ~1.2x slower, gap narrows slightly at high load.

### 1M Scale by Load Factor (500K ops)
| Load % | ours hit | hb hit | hit ratio | ours miss | hb miss | miss ratio |
|-------:|---------:|-------:|:---------:|----------:|--------:|:----------:|
| 45% | 9.4 ms | 8.6 ms | 1.09x | **1.37 ms** | 1.52 ms | **0.90x** |
| 65% | 8.9 ms | 8.5 ms | 1.05x | **1.59 ms** | 1.91 ms | **0.83x** |
| **85%** | **8.8 ms** | 8.9 ms | **0.99x** | **2.73 ms** | 4.11 ms | **0.66x** |

At 1M scale, cache effects dominate. We **tie on hits at 85% load** and
are **1.5x faster on misses**. The crossover point for misses shifts
lower at 1M (we win even at 45%) because cache misses amplify the cost
of hashbrown's longer probe chains.

### Key Insight: The Design Trade-off

| Property | ours (Boost design) | hashbrown (Swiss table) |
|----------|:-------------------:|:-----------------------:|
| Per-probe overhead | higher (15-slot groups, overflow bookkeeping) | lower (16-byte aligned, minimal metadata) |
| Miss termination | O(1) via overflow bit | O(chain length) — must find empty byte |
| Miss cost at high load | grows slowly | grows rapidly |
| Hit cost vs load | flat | flat |
| Crossover (100K) | ~70% load factor | — |
| Crossover (1M) | ~45% load factor (cache effects) | — |

Since our table operates at 44-87.5% load (averaging ~65%), we're right
on the cusp at 100K scale. At 1M+ scale, we win on misses across the
entire load range.

---

## Summary: Where We Win / Lose vs hashbrown

### We Win (overflow-bit design strengths)
| Workload | Speedup | Why |
|----------|--------:|-----|
| Lookup miss 100K (high load) | **1.76x** | Overflow bit terminates without bucket read |
| Lookup miss 1M (all loads) | **1.1-1.5x** | Same, amplified by cache effects |
| Insert 1M | **1.39x** | Compact metadata fits L2/L3 |
| Clone 1M | **7.1x** | SIMD match_non_empty + bulk copy |
| Equilibrium churn 4K | **1.28x** | Tombstone-free deletion |
| Equilibrium churn 65K | **1.10x** | Same |
| Growing lookup 2K | **1.28x** | Miss-heavy read workload |
| Growing lookup 100K | **1.19x** | Same |
| String insert (all sizes) | **1.06-1.27x** | Faster hashing path |
| String miss (all sizes) | **1.16-1.30x** | Overflow bit early termination |
| Miss-heavy (75%+ miss) 100K | **1.55-1.64x** | Overflow bits dominate |

### hashbrown Wins (Swiss table strengths)
| Workload | hashbrown speedup | Why |
|----------|------------------:|-----|
| Lookup hit (all sizes, all loads) | 1.25-1.30x | Tighter probe loop, 16-byte alignment |
| Insert 1K-100K | 1.21-1.82x | More optimized small-table codegen |
| Entry API (or_insert) | 1.4-1.6x | Very optimized entry path |
| Iteration (small) | 1.4-1.5x | 16-byte aligned metadata groups |
| Lookup miss (low load, <1M) | 1.24-1.30x | Faster per-probe at low load |
| Insert/erase phases 5M | 1.26x | Better rehash path |

---

## Detailed Results (fixed-N benchmarks)

### Insert (u64, pre-allocated)
| Size | ours | hashbrown | ratio |
|-----:|-----:|----------:|:-----:|
| 1K | 4.60 µs | 3.80 µs | 1.21x |
| 10K | 68.8 µs | 37.8 µs | 1.82x |
| 100K | 747 µs | 446 µs | 1.67x |
| **1M** | **17.6 ms** | 24.4 ms | **0.72x** |

### Lookup Hit (u64, pre-allocated)
| Size | ours | hashbrown | ratio |
|-----:|-----:|----------:|:-----:|
| 1K | 2.04 µs | 1.61 µs | 1.27x |
| 10K | 21.8 µs | 17.3 µs | 1.26x |
| 100K | 308 µs | 247 µs | 1.25x |
| 1M | 14.8 ms | 14.3 ms | 1.04x |

### Lookup Miss (u64, pre-allocated)
| Size | ours | hashbrown | ratio |
|-----:|-----:|----------:|:-----:|
| 1K | 1.45 µs | 902 ns | 1.61x |
| 10K | 14.7 µs | 10.2 µs | 1.44x |
| **100K** | **255 µs** | 413 µs | **0.62x** |
| **1M** | **3.00 ms** | 3.70 ms | **0.81x** |

### Equilibrium Churn (2M insert+erase ops)
| Size | ours | hashbrown | ratio |
|-----:|-----:|----------:|:-----:|
| **4K** | **32.4 ms** | 41.6 ms | **0.78x** |
| **65K** | **36.6 ms** | 40.3 ms | **0.91x** |
| 1M | 73.0 ms | 70.5 ms | 1.04x |

### Random Distinct (entry API, 5M ops)
| Distinct | ours | hashbrown | ratio |
|---------:|-----:|----------:|:-----:|
| 5% | 44.6 ms | 29.5 ms | 1.51x |
| 50% | 274 ms | 172 ms | 1.59x |
| 100% | 257 ms | 187 ms | 1.38x |

### Growing Lookup (insert 4, lookup many, ~50% miss)
| Size | ours | hashbrown | ratio |
|-----:|-----:|----------:|:-----:|
| **2K** | **6.27 ms** | 8.03 ms | **0.78x** |
| **100K** | **67.6 ms** | 80.4 ms | **0.84x** |

### String Sizes (200K entries for ≤13b, 50K for 100b)
| Len | ours insert | hb insert | ours miss | hb miss |
|----:|-----------:|----------:|----------:|--------:|
| **7b** | **12.7 ms** | 13.4 ms | **1.49 ms** | 1.93 ms |
| **8b** | **9.77 ms** | 10.2 ms | **1.50 ms** | 1.88 ms |
| **13b** | **12.6 ms** | 16.1 ms | **1.48 ms** | 1.93 ms |
| 100b | 2.80 ms | 2.85 ms | **548 µs** | 635 µs |

### Clone
| Size | ours | hashbrown | ratio |
|-----:|-----:|----------:|:-----:|
| 1K | 637 ns | 612 ns | 1.04x |
| 100K | 57.5 µs | 53.5 µs | 1.07x |
| **1M** | **2.15 ms** | 15.1 ms | **0.14x** |

### Iteration During Growth
| | ours | hashbrown | ratio |
|-|-----:|----------:|:-----:|
| grow+iterate | 1.45 s | 1.01 s | 1.43x |

### Mixed Workload (50% insert, 30% lookup, 20% remove)
| Size | ours | hashbrown | ratio |
|-----:|-----:|----------:|:-----:|
| 10K | 35.4 µs | 28.6 µs | 1.24x |
| 100K | 883 µs | 838 µs | 1.05x |

---

## Design Trade-offs Revealed by Benchmarks

1. **15-slot groups + overflow byte** vs hashbrown's 16-byte control groups:
   - Win: Fast miss termination (overflow bit = 1 byte read)
   - Win: Tombstone-free deletion (no performance degradation under churn)
   - Win: 7x faster clone (SIMD bulk copy without tombstone handling)
   - Lose: 1 wasted SIMD lane (slot 15 is overflow byte)
   - Lose: Slightly larger metadata per element

2. **Two separate allocations** vs hashbrown's single allocation:
   - Win: Better insert at 1M (compact metadata fits L2/L3)
   - Lose: Extra pointer indirection on every access

3. **No post-mixer** (foldhash is avalanching):
   - Win: Zero hash overhead beyond foldhash itself
   - Neutral: Same hash quality as hashbrown (both use foldhash)

4. **Load factor sensitivity**:
   - Our table is most competitive at high load (75%+) and large scale (1M+)
   - hashbrown has more consistent performance across the load spectrum
   - Our average operating load (~65%) is just below the miss crossover point at medium scale
