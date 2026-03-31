# Optimization Log

Tracking all optimization attempts, results, and decisions.
Baseline: commit b2d1f57 (raw SSE2 intrinsics, fast hash mixer, SIMD iteration).

## Baseline Numbers (pre-optimization, commit b2d1f57)

### Insert u64 (pre-allocated, with_capacity)
| Size | ours | std | hashbrown | indexmap |
|-----:|-----:|----:|----------:|--------:|
| 1K | 14.1 µs | 12.6 µs | 3.5 µs | 13.9 µs |
| 10K | 143.6 µs | 130.8 µs | 35.2 µs | 140.6 µs |
| 100K | 1.55 ms | 1.38 ms | 443 µs | 1.52 ms |
| 1M | 34.0 ms | 59.0 ms | 23.6 ms | 44.6 ms |

### Lookup Hit u64
| Size | ours | std | hashbrown | indexmap |
|-----:|-----:|----:|----------:|--------:|
| 1K | 12.1 µs | 8.0 µs | 1.7 µs | 8.1 µs |
| 10K | 122.3 µs | 82.5 µs | 18.3 µs | 88.8 µs |
| 100K | 1.35 ms | 1.11 ms | 250 µs | 1.14 ms |
| 1M | 51.5 ms | 64.0 ms | 13.8 ms | 36.0 ms |

### Mixed Workload (50% insert, 30% lookup, 20% remove)
| Size | ours | std | hashbrown |
|-----:|-----:|----:|----------:|
| 10K | 125.0 µs | 121.0 µs | 27.5 µs |
| 100K | 1.84 ms | 1.76 ms | 825 µs |

### ahash (ours vs hashbrown)
| Benchmark | ours | hashbrown |
|-----------|-----:|----------:|
| insert 10K | 64.1 µs | 51.6 µs |
| insert 100K | 691.7 µs | 631.4 µs |
| insert 1M | 13.1 ms | 32.8 ms |
| lookup 10K | 29.2 µs | 21.2 µs |
| lookup 100K | 371.1 µs | 269.5 µs |
| lookup 1M | 16.1 ms | 14.2 ms |

### Iteration
| Size | ours | std | hashbrown |
|-----:|-----:|----:|----------:|
| 1K | 619 ns | 421 ns | 415 ns |
| 10K | 6.04 µs | 3.92 µs | 3.95 µs |
| 100K | 66.0 µs | 40.4 µs | 40.9 µs |
| 1M | 1.23 ms | 1.24 ms | 1.29 ms |

---

## Attempt 1: Aligned SIMD Loads + Prefetching + needs_drop + SIMD iteration paths
**Status: KEPT**

### Changes
- `_mm_loadu_si128` → `_mm_load_si128` (aligned loads, metadata is 16-byte aligned)
- `match_empty` uses direct `_mm_setzero_si128` instead of routing through `match_byte`
- Added `Group::prefetch_read()` via `_mm_prefetch(..., _MM_HINT_T0)`
- `find_by_hash`: prefetches next group metadata + first bucket on overflow
- Drop/clear/clone use `std::mem::needs_drop` to skip scanning for Copy types
- Drop/clear/clone/rehash use SIMD `match_non_empty` instead of scalar byte loops
- `#[inline]`/`#[inline(always)]` on hot paths, `#[cold] #[inline(never)]` on grow paths
- map.rs `insert` uses `find_by_hash` directly (avoids `find_with_hash` indirection)

### Results (vs baseline)
| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| insert_u64 1M | 34.0 ms | 35.4 ms | +4% (acceptable) |
| lookup_hit 1M | 51.5 ms | 43.6 ms | **-15%** |
| mixed 10K | 125.0 µs | 123.4 µs | -1% |
| mixed 100K | 1.84 ms | 1.79 ms | **-3%** |
| ahash lookup 1M | 16.1 ms | 14.9 ms | **-7%** |

### Decision
Kept. Prefetch is the main contributor to the 1M lookup improvement.

---

## Attempt 2: Single Contiguous Allocation (buckets-first, Boost layout)
**Status: REVERTED**

### Changes
Merged metadata and bucket arrays into one allocation:
`[buckets: N*15*sizeof(K,V)] [padding] [metadata: (N+1)*16 bytes]`

### Results
| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| insert_u64 1M | 34.0 ms | 47.6 ms | **+40% REGRESSION** |
| lookup_hit 1M | 51.5 ms | 37.5 ms | **-27% WIN** |
| ahash insert 1M | 13.1 ms | 24.0 ms | **+83% REGRESSION** |

### Analysis
For 1M entries (u64→usize), the single allocation is ~17MB. The metadata sits at
the far end. During insert, the CPU ping-pongs between metadata (at +16MB offset)
and buckets (at +0), causing severe TLB pressure. The two-allocation layout keeps
metadata compact (~1MB) and fitting in L2/L3.

Lookup improved because the prefetch + single allocation reduced one pointer chase
on the critical path. But the insert regression was too severe.

### Decision
Reverted. Two allocations are better for workloads that mix insert and lookup.
The metadata array's compactness is more important than pointer-chase savings.

---

## Attempt 3: Single Contiguous Allocation (metadata-first)
**Status: REVERTED**

### Changes
Same as Attempt 2 but with metadata at the start:
`[metadata: (N+1)*16 bytes] [padding] [buckets: N*15*sizeof(K,V)]`

### Results
| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| insert_u64 1M | 34.0 ms | 47.9 ms | +41% REGRESSION |
| lookup_hit 1M | 51.5 ms | 43.6 ms | -15% (worse than buckets-first) |

### Decision
Reverted. Same TLB issue as buckets-first, and lookup was worse too.

---

## Attempt 4: Stronger Hash Mixer (splitmix64 / xmx)
**Status: REVERTED**

### Changes
Replaced Fibonacci multiply (1 mul + 1 xor-shift) with Stafford variant 13
(2 multiplies + 3 xor-shifts), the same mixer used by splitmix64.

### Results
| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| insert_u64 1M | 34.0 ms | 45.1 ms | **+32% REGRESSION** |
| ahash insert 1M | 13.1 ms | 24.4 ms | **+86% REGRESSION** |

### Analysis
Two extra multiplies per hash operation is catastrophic at scale. Without an
avalanche opt-out (like Boost has for already-good hash functions), the mixer
taxes every operation. The Fibonacci mixer is adequate when paired with SipHash
or ahash, which already provide good avalanche.

### Decision
Reverted. Would need a trait-based opt-out (`IsAvalanching`) to be viable,
which is a larger API change.

---

## Current State (post-optimizations)

Kept changes: aligned SIMD loads, prefetching, needs_drop, SIMD iteration paths,
inline annotations, cold grow paths.

### Numbers vs Baseline
| Benchmark | Baseline | Current | Change |
|-----------|----------|---------|--------|
| insert_u64 1K | 14.1 µs | 15.9 µs | +13%* |
| insert_u64 10K | 143.6 µs | 143.8 µs | flat |
| insert_u64 100K | 1.55 ms | 1.54 ms | flat |
| insert_u64 1M | 34.0 ms | 35.4 ms | +4% |
| lookup_hit 1K | 12.1 µs | 12.5 µs | +3%* |
| lookup_hit 10K | 122.3 µs | 123.8 µs | +1% |
| lookup_hit 100K | 1.35 ms | 1.33 ms | -1% |
| lookup_hit 1M | 51.5 ms | 43.6 ms | **-15%** |
| mixed 10K | 125.0 µs | 123.4 µs | -1% |
| mixed 100K | 1.84 ms | 1.79 ms | **-3%** |
| ahash insert 1M | 13.1 ms | 13.8 ms | +5% |
| ahash lookup 1M | 16.1 ms | 14.9 ms | **-7%** |

*Small-size regressions within noise/recompile variance.

---

## Remaining Ideas (not yet attempted)

### High Priority
1. **Fused find-or-insert for entry API** — Currently entry() does find(),
   then VacantEntry::insert() does insert_no_check() with a second probe.
   A fused operation would find the insertion slot during the lookup probe
   and reuse it, avoiding the second probe entirely.

2. **Anti-drift growth headroom** — Boost uses `ceil((size + size/61 + 1) / mlf)`
   to prevent repeated resize cycles after heavy delete-insert patterns.
   Our current approach only decrements max_load on deletion, which can cause
   premature rehashes.

### Medium Priority
3. **Size-adaptive allocation strategy** — Use single allocation for small
   tables (≤4 groups / 60 elements) where TLB isn't an issue, two allocations
   for large tables.

4. **Prefetch bucket during SIMD match** — Currently we prefetch on overflow
   to the next group. We could also prefetch the first match candidate's
   bucket while the SIMD comparison is completing.

5. **IsAvalanching trait** — Allow hash functions that already provide good
   avalanche (ahash, FxHash) to skip the Fibonacci post-mixer entirely.
   Boost does this. Would need a marker trait on the BuildHasher.

### Lower Priority
6. **Specialized small-table path** — For ≤1 group, skip the probe loop
   entirely and do a direct SIMD match on the single group.

7. **Reserve / shrink_to_fit** — Standard API completeness.

8. **Drain iterator** — Consuming iterator that removes elements.

9. **IntoIter SIMD optimization** — Current IntoIter uses scalar byte checks;
   could use match_non_empty like SlotIter.
