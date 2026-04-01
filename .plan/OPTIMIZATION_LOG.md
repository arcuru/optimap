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
**NOTE: The "TLB pressure" diagnosis below was later found to be wrong.**
The insert regression was actually caused by OS page fault overhead: the
`insert_u64` benchmark allocates a fresh map on every criterion iteration.
A single large mmap (~32MB) incurs ~11ms of page faults as the kernel
lazily zero-fills ~7,680 pages. With two smaller allocations, glibc's
arena caching may reuse previously freed pages across iterations, avoiding
re-faulting. See attempt 24/26 and BENCHMARKS.md methodology note for
the full analysis.

Original (incorrect) analysis: "CPU ping-pongs between metadata and
buckets causing TLB pressure." The actual insert operations are the same
speed regardless of allocation strategy.

### Decision
Reverted at the time. Later re-adopted in attempt 26 after understanding
the page fault overhead was a benchmark artifact.

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

## Attempt 5: Fused find-or-locate Probe
**Status: KEPT**

### Changes
Added `find_or_locate()` that tracks the first empty slot during the lookup
probe. Returns `Found(gi,si)`, `InsertSlot(gi,si)`, or `NotFound`.
- `map.insert()` uses fused probe when not at capacity
- `map.entry()` pre-locates the insertion slot in `VacantEntry`
- `set.insert()` uses fused probe similarly
- Falls back to `insert_no_check` when `NotFound` (all probed groups full)

### Results
| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| insert_u64 1K | 14.5 µs | 14.0 µs | **-3%** |
| insert_u64 10K | 143 µs | 146 µs | +2% |
| insert_u64 100K | 1.54 ms | 1.57 ms | +2% |
| insert_u64 1M | 35.4 ms | 35.1 ms | flat |

### Analysis
Mixed results on raw insert (small improvement at 1K, slight overhead at
10K-100K from tracking `first_empty` Option in the hot loop). Main benefit
is architectural: entry().or_insert() avoids a duplicate probe walk.

### Decision
Kept. The entry API improvement is valuable even if raw insert doesn't
benefit at all sizes.

---

## Attempt 6: SIMD-accelerated IntoIter
**Status: KEPT**

### Changes
Replaced scalar byte-checking IntoIter (`meta >= 2` per slot) with SIMD
`Group::match_non_empty()` bitmask iteration, matching the SlotIter design.

### Decision
Kept. Consistent with the rest of the codebase, no regression expected.

---

## Attempt 7: Initial Bucket Prefetch in find_by_hash
**Status: KEPT (with trade-off noted)**

### Changes
Added `Group::prefetch_read(bucket_ptr(gi, 0))` at the start of `find_by_hash`,
before the first SIMD metadata comparison. This overlaps the bucket memory
fetch with the metadata load.

### Results
| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| lookup_hit 1K | 12.5 µs | 12.3 µs | -2% |
| lookup_hit 100K | 1.33 ms | 1.30 ms | **-2.4%** |
| lookup_hit 1M | 43.6 ms | 38.6 ms | **-11%** |
| lookup_miss 1K | 11.2 µs | 11.4 µs | +2% |
| lookup_miss 100K | 1.31 ms | 1.35 ms | +3% |
| lookup_miss 1M | 13.1 ms | 15.8 ms | **+21%** |

### Analysis
Trades miss performance for hit performance. On a hit, the prefetched bucket
data is ready when we dereference it. On a miss, the prefetch is wasted and
pollutes the cache/prefetch queue. Hit-dominated workloads benefit significantly.

### Decision
Kept. Most real workloads are hit-dominated. The 11% hit improvement at 1M
outweighs the miss regression for typical usage patterns.

---

## Final State (all optimizations applied)

### Final Numbers (absolute, full benchmark suite)
| Benchmark | ours | std | hashbrown | indexmap |
|-----------|-----:|----:|----------:|--------:|
| **insert_u64 1K** | 14.1 µs | 12.6 µs | 3.5 µs | 13.7 µs |
| **insert_u64 10K** | 150.7 µs | 125.3 µs | 35.3 µs | 141.5 µs |
| **insert_u64 100K** | 1.54 ms | 1.36 ms | 448 µs | 1.56 ms |
| **insert_u64 1M** | 39.0 ms | 63.1 ms | 24.8 ms | 41.4 ms |
| **lookup_hit 1K** | 12.3 µs | 8.0 µs | 1.7 µs | 8.3 µs |
| **lookup_hit 10K** | 124.1 µs | 80.2 µs | 18.1 µs | 89.6 µs |
| **lookup_hit 100K** | 1.32 ms | 1.09 ms | 257 µs | 1.19 ms |
| **lookup_hit 1M** | 41.2 ms | 59.3 ms | 15.5 ms | 44.2 ms |
| **lookup_miss 1K** | 11.4 µs | 6.9 µs | 930 ns | — |
| **lookup_miss 100K** | 1.36 ms | 1.17 ms | 404 µs | — |
| **lookup_miss 1M** | 16.3 ms | 15.2 ms | 4.6 ms | — |
| **mixed 10K** | 133 µs | 124 µs | 30.3 µs | — |
| **mixed 100K** | 1.86 ms | 1.79 ms | 831 µs | — |
| **string insert 1K** | 38.7 µs | 41.6 µs | 35.3 µs | — |
| **string lookup 1K** | 10.7 µs | 13.2 µs | 4.3 µs | — |
| **string insert 10K** | 458 µs | 479 µs | 383 µs | — |
| **string lookup 10K** | 126 µs | 170 µs | 62.0 µs | — |
| **iteration 1K** | 620 ns | 423 ns | 420 ns | — |
| **iteration 100K** | 63.3 µs | 41.1 µs | 41.3 µs | — |
| **iteration 1M** | 1.43 ms | 1.37 ms | 1.39 ms | — |
| **grow_from_empty 1K** | 27.9 µs | 32.6 µs | 13.7 µs | — |
| **grow_from_empty 100K** | 2.47 ms | 2.65 ms | 1.06 ms | — |
| **ahash insert 10K** | 50.3 µs | 48.1 µs | — | — |
| **ahash insert 1M** | 15.5 ms | 38.3 ms | — | — |
| **ahash lookup 10K** | 32.1 µs | 21.3 µs | — | — |
| **ahash lookup 1M** | 18.0 ms | 17.1 ms | — | — |

### Where We Win vs std::HashMap
- **Insert at 1M**: 39ms vs 63ms (**38% faster**)
- **Lookup hit at 1M**: 41ms vs 59ms (**30% faster**)
- **String insert** at all sizes (1K-10K)
- **String lookup** at all sizes (**25-35% faster**)
- **Grow from empty** at all sizes (~7% faster)

### Where We Lose vs hashbrown
- **Everything at small-medium sizes**: 3-8x slower (hashbrown's Swiss table
  is extremely mature with years of optimization)
- **Lookup at all sizes**: even at 1M we're 2.7x slower
- **Iteration**: 1.5x slower except at 1M where we're close

### Summary of Changes vs Original Baseline
| Change | Status | Impact |
|--------|--------|--------|
| Aligned SIMD loads (`_mm_load_si128`) | Kept | Negligible alone |
| Prefetch on overflow to next group | Kept | ~5% lookup improvement at 1M |
| Initial bucket prefetch in find_by_hash | Kept | ~11% lookup_hit improvement, ~21% miss regression |
| `needs_drop` skip in Drop/clear | Kept | Faster for Copy types |
| SIMD `match_non_empty` in Drop/clear/clone/rehash | Kept | Cleaner, potentially faster |
| `#[cold]`/`#[inline(never)]` grow paths | Kept | Better code layout |
| Fused find-or-locate probe | Kept | ~3% insert_1K improvement, entry API benefit |
| SIMD IntoIter | Kept | Consistency, skips empty groups |
| Single allocation (buckets-first) | **Reverted** | +40% insert regression at 1M |
| Single allocation (metadata-first) | **Reverted** | Worse than buckets-first |
| splitmix64/xmx hash mixer | **Reverted** | +32-86% regression without avalanche opt-out |

---

## Attempt 8: IsAvalanching Trait with Specialization
**Status: PARTIAL — trait exported, auto-dispatch reverted**

### Changes
Added `IsAvalanching` marker trait for hash builders with good avalanche.
Implemented for `ahash::RandomState` behind `ahash` feature flag.
Attempted `#![feature(specialization)]` for automatic dispatch.

### Results
The specialization-based `default fn compute_hash()` caused a ~10% regression
on the default (std RandomState) path. The `default fn` keyword prevents the
compiler from fully inlining the trait method — it must go through a vtable-like
dispatch since specialization could override it.

### Decision
Kept the trait and `hash_no_mix()` helper as public API for manual opt-in.
Reverted the automatic specialization dispatch. Users who construct maps with
`with_hasher(ahash::RandomState::new())` can benefit, but must build the map
to opt in — the map's internal hash_key still applies the mixer unconditionally.

---

## Attempt 9: Size-Adaptive Allocation
**Status: SKIPPED**

### Analysis
Would use single allocation for small tables (≤4 groups) and two allocations
for large tables. However:
- Single allocation only improved 1M lookup by 9% (41ms→37.5ms) beyond what
  prefetch already provides
- Adds significant complexity (branching on alloc mode in allocate/deallocate/
  rehash, tracking allocation strategy)
- Small tables already fit in L1 cache with two allocations

Complexity not justified for marginal gains.

---

## Attempt 10: Single-Group Fast Path
**Status: KEPT**

### Changes
Added `if self.num_groups == 1` fast path at the top of `find_by_hash`.
Skips the probe loop, overflow bit check, and prefetch entirely. Just does
one SIMD match on the single group.

### Results
Affects only tiny tables (≤13 elements). Not visible in standard benchmarks
(1K+). The branch is always not-taken for larger tables and costs ~0 after
branch predictor warmup.

### Decision
Kept. Zero cost for large tables, small benefit for tiny tables.

---

## Attempt 11: Conditional Prefetch (only on match)
**Status: REVERTED**

### Changes
Only issue bucket prefetch when `match_byte` returns a non-empty bitmask
(i.e., we have candidates to check). Avoids wasted prefetches on miss path.

### Results
No improvement on miss path. The `if matches.any_set()` branch adds latency
before the prefetch can issue, delaying the memory fetch on the hit path.
The cost of a wasted prefetch (~1 cycle) is much less than the cost of
delaying a needed prefetch by a branch (~5-10 cycles of branch resolution).

### Decision
Reverted. Unconditional prefetch is better — fire-and-forget.

---

## Attempt 12: Switch Default Hasher to foldhash
**Status: KEPT**

### Changes
Replace `std::hash::RandomState` (SipHash) with `foldhash::fast::RandomState`
as the default hasher — the same fast hasher used by hashbrown. Since foldhash
is avalanching, the Fibonacci post-mixer is skipped entirely (hash_no_mix).

### Results
By far the largest single improvement across all attempts:
| Benchmark | Before (SipHash+mixer) | After (foldhash, no mixer) | Change |
|-----------|----------------------:|---------------------------:|-------:|
| insert 1K | 15.0 µs | 4.4 µs | **3.4x faster** |
| insert 1M | 38.9 ms | 19.1 ms | **2x faster** |
| lookup_hit 1K | 12.7 µs | 2.2 µs | **5.8x faster** |
| lookup_hit 1M | 36.1 ms | 11.3 ms | **3.2x faster** |
| lookup_miss 1K | 11.1 µs | 1.5 µs | **7.4x faster** |
| mixed 10K | 132 µs | 34.5 µs | **3.8x faster** |

### Decision
Kept. The hasher was the bottleneck all along.

---

## Attempt 13: Single Allocation Re-test (with foldhash)
**Status: REVERTED (same pattern as before)**

### Rationale for Re-test
The original single-allocation test (Attempt 2) was done with SipHash+mixer,
where hashing dominated the cost. With foldhash, hashing is ~10x cheaper,
so memory layout effects should be proportionally larger. Re-tested to see
if the trade-off changed.

### Results (single alloc vs two-alloc, both with foldhash)
| Benchmark | Two-alloc | Single-alloc | Change |
|-----------|----------:|-------------:|-------:|
| insert 1K | 7.2 µs | 8.0 µs | +12% |
| insert 10K | 68.8 µs | 54.9 µs | **-20%** |
| insert 100K | 727 µs | 722 µs | flat |
| **insert 1M** | 20.5 ms | 34.1 ms | **+66%** |
| lookup_hit 100K | 324 µs | 312 µs | **-4%** |
| **lookup_hit 1M** | 11.9 ms | 11.0 ms | **-7%** |
| lookup_miss 100K | 255 µs | 266 µs | +4% |
| lookup_miss 1M | 3.93 ms | 4.08 ms | +4% |
| high_load hit 1M | 13.3 ms | 12.1 ms | **-9%** |
| high_load miss 1M | 3.94 ms | 3.67 ms | **-7%** |
| miss_ratio 50% 100K | 296 µs | 272 µs | **-8%** |
| miss_ratio 100% 100K | 302 µs | 289 µs | **-4%** |

### Analysis
Same fundamental trade-off as before:
- **1M insert: +66% regression** — rehashing allocates old+new (both huge),
  TLB thrashing during element migration
- **1M lookup: -7 to -9% improvement** — single allocation reduces TLB
  misses on the access path
- **10K insert: -20% improvement** — one alloc call vs two helps at medium scale
- **100K misses: +4%** — slight regression from larger working set

The 1M insert regression is still a dealbreaker. The rehash path dominates:
with single alloc, we need ~17MB old + ~34MB new simultaneously, vs
~1MB metadata + ~16MB buckets + ~2MB metadata + ~32MB buckets with two allocs.
The two-alloc approach keeps the metadata compact and hot.

### Decision
Reverted. Two allocations remain better for insert-heavy workloads.

---

## All Attempts Summary

| # | Technique | Status | Key Finding |
|---|-----------|--------|-------------|
| 1 | Aligned SIMD loads + prefetch + cold paths | **Kept** | 15% lookup improvement |
| 2 | Single allocation (buckets-first, SipHash) | Reverted | +40% insert regression |
| 3 | Single allocation (metadata-first, SipHash) | Reverted | Worse than #2 |
| 4 | splitmix64 hash mixer | Reverted | +32-86% regression |
| 5 | Fused find-or-locate | **Kept** | Entry API avoids double probe |
| 6 | SIMD IntoIter | **Kept** | Consistency |
| 7 | Initial bucket prefetch | **Kept** | 11% hit improvement, 21% miss regression |
| 8 | IsAvalanching auto-dispatch | **Partial** | Specialization hurts default path |
| 9 | Size-adaptive allocation | Skipped | Not worth complexity |
| 10 | Single-group fast path | **Kept** | Zero-cost for large tables |
| 11 | Conditional prefetch | Reverted | Branch overhead > wasted prefetch cost |
| 12 | foldhash default hasher | **Kept** | 3-7x faster across all operations |
| 13 | Single allocation re-test (foldhash) | Reverted | +66% insert 1M still dealbreaker |

## Attempt 14: Remove Manual Prefetch (with foldhash)
**Status: KEPT**

### Rationale
With foldhash (10x faster than SipHash), the hash computation is so short
that there's less latency to hide. The hardware prefetcher may handle the
bucket access pattern on its own.

### Results (no prefetch vs with prefetch, foldhash baseline)
| Benchmark | With prefetch | No prefetch | Change |
|-----------|-------------:|------------:|-------:|
| lookup_hit 1K | 2.17 µs | 2.07 µs | **-5%** |
| lookup_hit 10K | 23.4 µs | 22.5 µs | **-4%** |
| lookup_hit 1M | 11.9 ms | 14.0 ms | +18% |
| lookup_miss 1K | 1.50 µs | 1.47 µs | -2% |
| lookup_miss 10K | 15.5 µs | 14.9 µs | **-4%** |
| lookup_miss 100K | 255 µs | 245 µs | **-4%** |
| lookup_miss 1M | 3.93 ms | 2.89 ms | **-27%** |
| high_load miss 1M | 3.94 ms | 2.77 ms | **-30%** |

### Decision
Kept. The miss improvement (-27% to -30% at 1M) massively outweighs the
hit regression (+18% at 1M). Small-medium sizes improve across the board.
With the prefetch removed, our miss performance is now 2x faster than
hashbrown at 100K.

---

## All Attempts Summary

| # | Technique | Status | Key Finding |
|---|-----------|--------|-------------|
| 1 | Aligned SIMD loads + prefetch + cold paths | **Kept** | 15% lookup improvement |
| 2 | Single allocation (buckets-first, SipHash) | Reverted | +40% insert regression |
| 3 | Single allocation (metadata-first, SipHash) | Reverted | Worse than #2 |
| 4 | splitmix64 hash mixer | Reverted | +32-86% regression |
| 5 | Fused find-or-locate | **Kept** | Entry API avoids double probe |
| 6 | SIMD IntoIter | **Kept** | Consistency |
| 7 | Initial bucket prefetch | **Superseded by #14** | Was 11% hit improvement |
| 8 | IsAvalanching auto-dispatch | **Partial** | Specialization hurts default path |
| 9 | Size-adaptive allocation | Skipped | Not worth complexity |
| 10 | Single-group fast path | **Kept** | Zero-cost for large tables |
| 11 | Conditional prefetch | Reverted | Branch overhead > wasted prefetch cost |
| 12 | foldhash default hasher | **Kept** | 3-7x faster across all operations |
| 13 | Single allocation re-test (foldhash) | Reverted | +66% insert 1M still dealbreaker |
| 14 | Remove manual prefetch (foldhash) | **Kept** | -27% miss 1M, -4% small sizes |

---

## Phase 2: Structural Optimizations

### Attempt P7+P8 (Batch 1): Overflow-only prefetch + fused SIMD match
**Status: KEPT**

Prefetch next group only after overflow-bit check (not on miss fast path).
Fused `match_byte_and_empty` does one SIMD load, two compares, two movemasks.

### Attempt P6+P2 (Batch 2): Home-group fast path + overflow bitmask in ProbeResult
**Status: KEPT**

Inline home-group check before probe loop. `ProbeResult::InsertSlot` carries
`full_mask: u8` bitmask of overflow groups, avoiding re-walking on insert.

### Attempt: find_or_locate in insert() path
**Status: REVERTED**

The fused find_or_locate was slower than simple find + insert_no_check for
bulk insert. At 1K: 7.85 → 5.22 µs (33% faster with two-pass). The tracking
overhead (first_empty, overflow bitmask) outweighs the saved second probe
since insert() mostly inserts new keys into sparse tables. find_or_locate is
kept only for the entry API.

### Phase 2 Final Numbers
| Benchmark | Pre-Phase-2 | Post-Phase-2 | Change |
|-----------|------------:|-------------:|-------:|
| insert 1K | 4.3 µs | 4.60 µs | +7% (different load point) |
| insert 1M | 22.1 ms | 17.6 ms | **-20%** |
| lookup_hit 1M | 17.4 ms | 14.8 ms | **-15%** |
| lookup_miss 100K | 197 µs | 255 µs | +29% (different load point) |
| lookup_miss 1M | 2.92 ms | 3.00 ms | +3% (within noise) |

Note: Fixed-N comparisons are unreliable due to load-factor sensitivity.
See BENCHMARKS.md for load-factor-controlled measurements.

---

## Phase 3: Fused Home-Group Operations

### Attempt 15: Fused home-group insert
**Status: KEPT — largest single improvement since foldhash**

### Changes
Replaced the two-pass insert (find_by_hash + insert_no_check) with a fused
home-group path: one `match_byte_and_empty` SIMD load produces both the
key-match and empty-slot bitmasks. When the home group has space and no
overflow (the common case), the entire insert — duplicate check, slot
location, metadata write, bucket write — completes without a second
metadata load.

Cold paths (`#[cold] #[inline(never)]`) handle overflow and at-capacity
cases, keeping the inlined fast path compact.

Applied to `map.insert()`, `set.insert()`, and `map.entry()`.

### Results
| Benchmark | Before | After | Change |
|-----------|-------:|------:|-------:|
| insert 1K | 4.60 µs | 2.99 µs | **-35%** |
| insert 10K | 68.8 µs | 36.1 µs | **-48%** |
| insert 100K | 747 µs | 457 µs | **-39%** |
| insert 1M | 17.6 ms | 10.4 ms | **-41%** |
| mixed 10K | 35.4 µs | 28.6 µs | **-19%** |
| mixed 100K | 883 µs | 808 µs | **-8%** |
| lookup_hit (all) | unchanged | unchanged | 0% |
| lookup_miss (all) | unchanged | unchanged | 0% |

Ratios vs hashbrown:
- insert 10K: 1.82x → **1.03x** (tied)
- insert 100K: 1.67x → **1.02x** (tied)
- insert 1M: 0.72x → **0.43x** (2.3x faster)
- mixed 10K: 1.24x → **0.88x** (we now win)
- mixed 100K: 1.05x → **0.96x** (we now win)

### Analysis
The previous two-pass approach loaded the home group metadata twice: once
during `find_by_hash` (to check for the key) and once during `insert_no_check`
(to find an empty slot). Both loads went to the same L1-cached address, but
the surrounding computation (reduced_hash, overflow_bit, group_index, meta_ptr)
was also duplicated.

The fused path eliminates all this redundancy. The extra cost is one additional
`_mm_cmpeq_epi8` + `_mm_movemask_epi8` (comparing against zero to find empties)
which runs in parallel with the match comparison on a superscalar CPU — effectively
free.

The key insight vs the earlier find_or_locate attempt (Attempt 5 / Phase 2 revert):
find_or_locate tried to track the first empty slot across the ENTIRE probe chain,
adding overhead to every probe step. The fused home-group path only optimizes the
home group (one probe step), and delegates the rare overflow case to a cold path.
This keeps the hot path minimal while still eliminating the second SIMD load for
>85% of inserts.

### Decision
Kept. The largest improvement since switching to foldhash (Attempt 12).

---

### Attempt 16: Remove single-group fast path from find_by_hash
**Status: KEPT**

### Changes
Removed the `if self.num_groups == 1` branch from `find_by_hash`. The general
loop handles single-group tables correctly: overflow bits are never set on a
single group, so the probe terminates after one iteration.

### Results
Marginal improvement (~0.02x) on lookup hit ratios across load factors.
Removing the branch simplifies the code and frees one branch prediction slot.

### Decision
Kept. Zero downside, small upside.

---

### Attempt 17: Dense iteration fast path
**Status: REVERTED**

### Changes
Added a `dense_remaining` counter to `SlotIter`. When `match_non_empty` returns
`0x7FFF` (all 15 slots full), yielded slots 0..14 sequentially via a counter
instead of bitmask iteration.

### Results
| Benchmark | Before | After | Change |
|-----------|-------:|------:|-------:|
| iteration 100K | 63.3 µs | 84.4 µs | **+33% regression** |
| iteration 1M | 1.43 ms | 1.79 ms | **+25% regression** |

### Analysis
The extra `if dense_remaining > 0` check at the top of every `next()` call
dominated the savings. `tzcnt` + `blsr` (trailing zeros + clear lowest bit)
is already effectively 2 cycles per element — near-optimal. Adding a branch
+ subtract for the counter path plus the entry-check branch made every call
slower, even for non-full groups.

### Decision
Reverted. Bitmask iteration is already near-optimal on x86_64.

---

### Attempt 18: Inline home-group in find_by_hash with cold continuation
**Status: REVERTED**

### Changes
Restructured `find_by_hash` to inline the home-group check, with the overflow
probe loop in a separate `#[inline(never)]` function. Also deferred `overflow_bit`
computation to after the home-group miss (first attempt) or kept it early for
parallel execution (second attempt).

### Results (deferred overflow_bit)
| Benchmark | Before | After | Change |
|-----------|-------:|------:|-------:|
| lookup_hit 1K | 2.04 µs | 1.90 µs | **-7%** |
| lookup_miss 1K | 1.45 µs | 1.68 µs | **+16% regression** |

### Results (early overflow_bit + cold continuation)
| Benchmark | Before | After | Change |
|-----------|-------:|------:|-------:|
| load_factor hit (all) | ~416 µs | ~470 µs | **+10-14% regression** |
| load_factor miss (all) | unchanged | unchanged | flat |

### Analysis
The `#[inline(never)]` continuation function forced the compiler to save/restore
registers at the call boundary, even though the cold path is rarely taken. This
register pressure degraded the hot path (home-group hit) by 10-14%.

Deferring `overflow_bit` past the SIMD match moved it onto the serial critical
path for misses, causing a 16% regression. Computing it early (in parallel with
SIMD) avoided this but didn't recover the cold-continuation regression.

### Decision
Reverted both variants. The simple loop structure generates better code than
inline + cold continuation, despite the loop having "unnecessary" setup for the
home-group-only case.

---

### Attempt 19: Custom Iterator::fold for internal iteration
**Status: REVERTED**

### Changes
Implemented custom `fold` on `SlotIter`, `Iter`, `IterMut`, `IntoIter`, `Keys`,
`Values`, and `ValuesMut`. The idea: `fold` processes all remaining elements in a
tight group-by-group loop without the per-element state-machine overhead of
`next()`. Methods like `for_each`, `sum`, `count`, and `collect` all delegate to
`fold`, so they would benefit.

The `SlotIter::fold` iterated groups directly:
```rust
fn fold<B, F>(self, init: B, mut f: F) -> B {
    let mut acc = init;
    let mut mask = self.current_mask;
    let mut group = self.group;
    loop {
        for si in mask { acc = f(acc, (group, si)); }
        group += 1;
        if group >= self.table.num_groups { return acc; }
        mask = Group::match_non_empty(self.table.meta_ptr(group));
    }
}
```

Each wrapper iterator's fold delegated down: `Values::fold` → `Iter::fold` →
`SlotIter::fold`, each adding a closure layer.

### Results
| Benchmark | for loop (next) | .values().fold() | Change |
|-----------|----------------:|-----------------:|-------:|
| 1K | 616 ns | 646 ns | +5% |
| 10K | 5.95 µs | 6.31 µs | +6% |
| 100K | 65.2 µs | 67.0 µs | +3% |
| 1M | 1.30 ms | 1.53 ms | **+18%** |

hashbrown showed a similar pattern: their `.values().fold()` was also slower
than the for loop at 100K (68.7 µs vs 40.5 µs).

### Analysis
The nested closure chain (`Values::fold` → `Iter::fold` → `SlotIter::fold`)
creates 3 levels of generic closures that LLVM cannot fully inline and optimize.
The default `fold` implementation (which calls `next()` in a simple loop) produces
cleaner IR that LLVM optimizes into the same tight loop our custom fold was trying
to achieve.

This is a known Rust pattern: custom `fold` through wrapper iterators is often
slower than the default `next()`-based fold because the simpler control flow
gives LLVM more optimization headroom. The approach might work if applied at the
outermost level (e.g., a `for_each` method on the map itself that bypasses the
Iterator trait entirely), but that would be a non-standard API.

### Decision
Reverted. The default fold (using `next()`) already generates near-optimal code.

---

## All Attempts Summary

| # | Technique | Status | Key Finding |
|---|-----------|--------|-------------|
| 1 | Aligned SIMD loads + prefetch + cold paths | **Kept** | 15% lookup improvement |
| 2 | Single allocation (buckets-first, SipHash) | Reverted | +40% insert regression |
| 3 | Single allocation (metadata-first, SipHash) | Reverted | Worse than #2 |
| 4 | splitmix64 hash mixer | Reverted | +32-86% regression |
| 5 | Fused find-or-locate | **Kept** | Entry API avoids double probe |
| 6 | SIMD IntoIter | **Kept** | Consistency |
| 7 | Initial bucket prefetch | **Superseded by #14** | Was 11% hit improvement |
| 8 | IsAvalanching auto-dispatch | **Partial** | Specialization hurts default path |
| 9 | Size-adaptive allocation | Skipped | Not worth complexity |
| 10 | Single-group fast path | **Superseded by #16** | Zero-cost for large tables |
| 11 | Conditional prefetch | Reverted | Branch overhead > wasted prefetch cost |
| 12 | foldhash default hasher | **Kept** | 3-7x faster across all operations |
| 13 | Single allocation re-test (foldhash) | Reverted | +66% insert 1M still dealbreaker |
| 14 | Remove manual prefetch (foldhash) | **Kept** | -27% miss 1M, -4% small sizes |
| 15 | Fused home-group insert | **Kept** | **-35 to -48% insert, mixed now wins** |
| 16 | Remove single-group branch | **Kept** | Marginal hit improvement |
| 17 | Dense iteration fast path | Reverted | +33% iteration regression |
| 18 | Inline find_by_hash + cold continuation | Reverted | +10-14% hit regression |
| 19 | Custom Iterator::fold | Reverted | +5-18% regression from closure nesting |
| 20 | #[inline] on entry API | Reverted | Helps hit-heavy (-7%), hurts insert-heavy (+31%) |
| 21 | AVX2 multi-group probing/iteration | Skipped | Analysis shows <8% of probes overflow; iteration bottleneck is bucket access, not SIMD loads |
| 22 | Derive group_mask from num_groups | **Kept** | -8 bytes struct, no perf regression |
| 23 | Store mask instead of num_groups | **Kept** | Hot-path reads mask directly, cold paths derive num_groups = mask+1 |
| 24 | Single allocation (metadata+buckets) | Tested | -4% insert 1K-100K, +108% insert 1M (rehash TLB), not committed |
| 25 | Home-group bucket prefetch | Tested | -5-8% hits, +6-11% misses at low load, not committed |

---

### Attempt 20: #[inline] on entry API methods
**Status: REVERTED**

Added `#[inline]` to `entry()`, `or_insert()`, `or_insert_with()`, and
`or_default()`. For hit-heavy workloads (5% distinct), inlining entry()
removed function call overhead and improved by 7%. But for insert-heavy
workloads (100% distinct), the inlined entry body caused code bloat and
instruction cache pressure: +31% regression. The compiler's default
heuristics (not inlining entry()) are correct — the function is too large
and the hit/insert trade-off makes any single inline decision suboptimal.

---

### Attempt 21: AVX2 multi-group probing and iteration
**Status: SKIPPED after analysis**

Measured probe chain statistics at various load factors using Poisson model:

| Load % | λ (elems/group) | Full groups | Home-group hit rate | Avg probes/hit |
|-------:|----------------:|------------:|--------------------:|---------------:|
| 45% | 6.8 | 0.4% | 99.6% | 1.00 |
| 55% | 8.2 | 2.2% | 97.8% | 1.02 |
| 65% | 9.8 | 7.1% | 92.9% | 1.08 |
| 75% | 11.2 | 16.5% | 83.5% | 1.20 |
| 85% | 12.8 | 30.0% | 70.0% | 1.43 |

**For probing**: AVX2 could combine two SIMD comparisons into one 256-bit operation,
but the quadratic probe sequence visits non-adjacent groups, so a single 32-byte load
cannot cover two probe steps. Would need two separate 16-byte loads combined via
`_mm256_set_m128i`, saving one cmpeq+movemask but adding a combine. At 65% load,
only 7% of operations need >1 probe — the home group resolves 93%+ of operations.
AVX2 cannot help the home-group fast path (it's already a single SIMD load).

**For iteration**: AVX2 would halve SIMD loads (one 32-byte load → two groups of
metadata). But iteration at small-medium sizes is bottlenecked by bucket access
(bucket_ptr arithmetic + memory loads), not metadata SIMD loads. At 1M+ where
iteration matters, it's already memory-bound (we tie hashbrown). Halving SIMD loads
for metadata wouldn't move the needle.

**Additional concerns**:
- Runtime feature detection (`is_x86_feature_detected!`) adds a branch on every call
- AVX2 can cause frequency throttling on some Intel CPUs
- Code complexity doubles (SSE2 + AVX2 paths)

**Decision**: Skipped. The analysis shows AVX2 targets the wrong bottleneck for both
probing and iteration.

---

### Attempt 22: Derive group_mask from num_groups
**Status: KEPT**

Removed the stored `group_mask: usize` field. Replaced with an inline
`group_mask()` method returning `num_groups.wrapping_sub(1)`. The
subtraction is single-cycle and executes in parallel with surrounding
work. No measurable regression. Struct size: 64 → 56 bytes.

---

### Attempt 23: Store mask instead of num_groups
**Status: KEPT**

Replaced `num_groups: usize` with `mask: usize` (= num_groups - 1).
The mask is used directly on the hot path (8 uses: group_index, all
probe wraparound sites). num_groups is derived as `mask + 1` only on
cold paths (allocation, growth, deallocation). Empty table detection
switched from `num_groups == 0` to `metadata.is_null()` since mask = 0
is ambiguous (empty or 1 group).

No measurable regression. Same struct size (mask replaces num_groups).

---

### Attempt 24: Single allocation (re-test with fused insert)
**Status: TESTED, NOT COMMITTED**

Re-tested single allocation (metadata-first: `[metadata][padding][buckets]`
in one malloc) with the current fused home-group insert path.

| Benchmark | two-alloc | single-alloc | Change |
|-----------|----------:|-------------:|-------:|
| insert 1K | 2.99 µs | 2.86 µs | **-4%** |
| insert 10K | 36.1 µs | 33.0 µs | **-9%** |
| insert 100K | 457 µs | 441 µs | **-4%** |
| insert 1M | 10.4 ms | 21.6 ms | **+108%** |
| mixed 100K | 808 µs | 750 µs | **-7%** |

Insert 1M doubled due to the same TLB/rehash issue as attempts 2/3/13:
during rehash, old + new allocations are both huge (~17MB + ~34MB), and
the single-allocation layout puts metadata at the start of these blocks,
causing TLB thrashing. Lookup and miss performance were unchanged.

The single allocation could be viable if combined with a better rehash
strategy (e.g., in-place growth), but as-is the 1M regression is a
dealbreaker.

---

### Attempt 25: Home-group bucket prefetch (re-test)
**Status: TESTED, NOT COMMITTED**

Re-tested prefetching the home group's buckets before the SIMD metadata
match in `find_by_hash`. Tested with both allocation strategies.

Load-factor-controlled results (100K-slot table, two-alloc + prefetch):

| Load % | Hit (before) | Hit (prefetch) | Miss (before) | Miss (prefetch) |
|-------:|-------------:|---------------:|--------------:|----------------:|
| 45% | 404 µs | **378 µs** (-6%) | 169 µs | 184 µs (+9%) |
| 55% | 405 µs | **382 µs** (-6%) | 174 µs | 184 µs (+6%) |
| 65% | 412 µs | **381 µs** (-8%) | 178 µs | 189 µs (+6%) |
| 75% | 420 µs | **395 µs** (-6%) | 199 µs | 196 µs (flat) |
| 85% | 420 µs | **404 µs** (-4%) | 311 µs | 305 µs (flat) |

Confirmed the predicted trade-off: prefetch helps hits at low-medium load
(5-8% improvement) but hurts misses at low load (6-9% regression from
cache pollution by unused bucket data). Neutral at high load for both.
Not committed because the miss regression and hit improvement are
workload-dependent with no universal win.

---

## Struct Size Evolution

| State | RawTable | UnorderedFlatMap | hashbrown |
|-------|--------:|-----------------:|----------:|
| Original (with group_mask field) | 56 | 64 | 40 |
| After removing group_mask (#22) | 48 | 56 | 40 |
| After mask-instead-of-num_groups (#23) | 48 | 56 | 40 |

Current struct (48 bytes RawTable, 56 bytes UnorderedFlatMap):
```
mask:       usize   (8)  — num_groups - 1, hot-path masking
metadata:   *mut u8 (8)  — pointer to group metadata array
buckets:    *mut u8 (8)  — pointer to bucket array
len:        usize   (8)  — number of elements
max_load:   usize   (8)  — threshold for rehash
shift:      u32     (4)  — hash >> shift gives group index
padding:            (4)
```

The 16-byte gap to hashbrown (40 bytes) comes from:
- `buckets` pointer (8 bytes) — hashbrown uses single allocation, derives
  bucket pointer from ctrl pointer. We use two allocations (necessary to
  avoid 1M insert regression).
- `max_load` (8 bytes) — hashbrown stores `growth_left` (similar purpose)
  in their `RawTable`, so this is equivalent.
- `shift` + padding (8 bytes) — hashbrown stores `bucket_mask` instead,
  also 8 bytes. Equivalent.

The remaining gap is essentially the `buckets` pointer, which is the cost
of our two-allocation design. Eliminating it requires single allocation
(attempt 24) which regresses 1M insert by 108%.

Further shrinking options evaluated and rejected:
- `shift: u32 → u8`: saves 0 bytes due to alignment padding (u8 + 7 pad = u32 + 4 pad = 8 bytes)
- Remove num_groups/mask entirely, derive from shift: saves 8 bytes but adds branch + shift to every iteration bounds check (~5% regression)
- Always-allocate (eliminate null checks): removes 3-4 hot-path branches but forces 256+ bytes allocation on `new()`, breaking zero-cost empty maps convention

---

## Remaining Ideas

See FUTURE.md for a comprehensive list of further improvements.
