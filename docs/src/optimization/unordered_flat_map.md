# UnorderedFlatMap Optimization Log

26 optimization attempts from baseline to final state. Baseline: commit b2d1f57
(raw SSE2 intrinsics, fast hash mixer, SIMD iteration).

## Baseline Numbers (SipHash + Fibonacci mixer)

| Benchmark | ours | hashbrown |
|-----------|-----:|----------:|
| insert 1K | 14.1 µs | 3.5 µs |
| insert 1M | 34.0 ms | 23.6 ms |
| lookup_hit 1K | 12.1 µs | 1.7 µs |
| lookup_hit 1M | 51.5 ms | 13.8 ms |
| iteration 1K | 619 ns | 415 ns |
| iteration 1M | 1.23 ms | 1.29 ms |

We were 4-7x slower than hashbrown on small tables.

---

## Phase 1: SIMD Fundamentals

### #1: Aligned SIMD loads + prefetch + cold paths — KEPT

- `_mm_loadu_si128` → `_mm_load_si128` (metadata is 16-byte aligned)
- `match_empty` uses direct `_mm_setzero_si128`
- Prefetch next group metadata + first bucket on overflow
- `#[cold] #[inline(never)]` on grow paths
- SIMD `match_non_empty` in Drop/clear/clone/rehash

Result: -15% lookup_hit at 1M. Prefetch is the main contributor.

### #2, #3: Single contiguous allocation — REVERTED

Merged metadata + buckets into one allocation. Two layouts tested
(buckets-first, metadata-first).

Result: +40% insert regression at 1M. Originally diagnosed as "TLB pressure"
but later found to be OS page fault overhead — `mmap` lazily zero-fills ~7,680
pages at ~1.5µs each = ~11ms per benchmark iteration. Two smaller allocations
benefit from glibc arena caching of previously freed pages.

### #4: Stronger hash mixer (splitmix64) — REVERTED

Replaced Fibonacci multiply with Stafford variant 13 (2 multiplies + 3 xor-shifts).

Result: +32-86% regression. Two extra multiplies per hash are catastrophic at scale
without an avalanche opt-out for already-good hash functions.

### #5: Fused find-or-locate probe — KEPT

`find_or_locate()` tracks the first empty slot during the lookup probe. Used by
entry API to avoid double probing.

Result: Mixed on raw insert (±2%), but entry API benefits architecturally.

### #6: SIMD-accelerated IntoIter — KEPT

Replaced scalar byte-checking with SIMD `match_non_empty()` bitmask iteration.

### #7: Initial bucket prefetch — KEPT (later superseded by #14)

Prefetch bucket data at start of `find_by_hash`, before SIMD metadata match.

Result: -11% lookup_hit at 1M, but +21% lookup_miss at 1M. Trades miss
performance for hit performance via cache pollution.

### #8: IsAvalanching trait — PARTIAL

Added marker trait for avalanching hash builders. `#![feature(specialization)]`
for auto-dispatch caused ~10% regression on the default path (vtable-like dispatch).
Kept as manual opt-in API only.

### #9: Size-adaptive allocation — SKIPPED

Would use single allocation for small tables, two for large. Complexity not
justified: single allocation only improved 1M lookup by 9% beyond prefetch.

### #10: Single-group fast path — KEPT (later superseded by #16)

`if self.num_groups == 1` branch in `find_by_hash`. Skips probe loop for tiny
tables. Zero cost for large tables (branch predictor).

### #11: Conditional prefetch (only on match) — REVERTED

Only issue bucket prefetch when `match_byte` returns a non-empty bitmask.

Result: No improvement. The `if` branch adds latency before the prefetch issues.
Cost of a wasted prefetch (~1 cycle) << cost of delaying a needed prefetch (~5-10 cycles).

---

## The Foldhash Turning Point

### #12: Switch to foldhash — KEPT

Replaced SipHash with foldhash (same hasher used by hashbrown). Since foldhash is
avalanching, the post-mixer is skipped entirely.

**By far the largest improvement:**

| Benchmark | Before (SipHash) | After (foldhash) | Change |
|-----------|------------------:|------------------:|-------:|
| insert 1K | 15.0 µs | 4.4 µs | **3.4x** |
| insert 1M | 38.9 ms | 19.1 ms | **2.0x** |
| lookup_hit 1K | 12.7 µs | 2.2 µs | **5.8x** |
| lookup_hit 1M | 36.1 ms | 11.3 ms | **3.2x** |
| lookup_miss 1K | 11.1 µs | 1.5 µs | **7.4x** |
| mixed 10K | 132 µs | 34.5 µs | **3.8x** |

### #13: Single allocation re-test (with foldhash) — REVERTED

Re-tested with foldhash since hashing is no longer dominant. Same result:
+66% insert 1M regression. The page fault / rehash overhead is fundamental.

### #14: Remove manual prefetch — KEPT

With foldhash (10x faster), hash computation is too short to hide latency.
Hardware prefetcher handles the bucket access pattern.

| Benchmark | With prefetch | No prefetch | Change |
|-----------|-------------:|------------:|-------:|
| lookup_miss 1M | 3.93 ms | 2.89 ms | **-27%** |
| high_load miss 1M | 3.94 ms | 2.77 ms | **-30%** |
| lookup_hit 1M | 11.9 ms | 14.0 ms | +18% |

The miss improvement massively outweighs the hit regression.

---

## Phase 2: Structural Optimizations

### P7+P8: Overflow-only prefetch + fused SIMD match — KEPT

Prefetch next group only after overflow-bit check (not on miss fast path).
`match_byte_and_empty` does one SIMD load, two compares, two movemasks.

### P6+P2: Home-group fast path + overflow bitmask carry — KEPT

Inline home-group check before probe loop. `InsertSlot` carries overflow
group bitmask, avoiding re-walking on insert.

---

## Phase 3: Fused Home-Group Operations

### #15: Fused home-group insert — KEPT

**Largest improvement since foldhash.** One `match_byte_and_empty` SIMD load
produces both key-match and empty-slot bitmasks. When the home group has space
and no overflow, the entire insert completes without a second metadata load.

| Benchmark | Before | After | Change |
|-----------|-------:|------:|-------:|
| insert 1K | 4.60 µs | 2.99 µs | **-35%** |
| insert 10K | 68.8 µs | 36.1 µs | **-48%** |
| insert 100K | 747 µs | 457 µs | **-39%** |
| insert 1M | 17.6 ms | 10.4 ms | **-41%** |

Ratios vs hashbrown: insert 10K went from 1.82x → **1.03x** (tied).

Key insight vs the earlier find_or_locate attempt (#5): that tracked the first
empty slot across the entire probe chain, adding overhead to every step. The
fused approach only optimizes the home group (one step), delegating overflow
to a cold path.

### #16: Remove single-group branch — KEPT

Removed `if num_groups == 1` fast path. The general loop handles it correctly
(overflow bits are never set on a single group). One fewer branch.

---

## Failed Attempts (Phase 3)

### #17: Dense iteration fast path — REVERTED

Added counter for all-full groups to avoid bitmask iteration.

Result: +33% iteration regression. The `if dense_remaining > 0` check on every
`next()` dominated the savings. `tzcnt` + `blsr` is already ~2 cycles per
element — near-optimal.

### #18: Inline find_by_hash + cold continuation — REVERTED

`#[inline(never)]` on the overflow continuation forced register save/restore
at the call boundary, degrading the hot path by 10-14%.

Deferring `overflow_bit` past the SIMD match moved it onto the serial critical
path for misses (+16%). Computing it early avoided this but didn't recover the
cold-continuation regression.

### #19: Custom Iterator::fold — REVERTED

Nested closure chain (`Values::fold` → `Iter::fold` → `SlotIter::fold`)
generated worse code than the default `next()`-based fold. +5-18% regression.
LLVM optimizes simpler control flow better than deeply nested generic closures.

### #20: #[inline] on entry API — REVERTED

Helps hit-heavy (-7%) but hurts insert-heavy (+31%) due to code bloat.
Compiler's default heuristics are correct.

### #21: AVX2 multi-group probing — SKIPPED

Probe chain statistics at various load factors:

| Load % | Home-group hit rate | Avg probes/hit |
|-------:|--------------------:|---------------:|
| 45% | 99.6% | 1.00 |
| 65% | 92.9% | 1.08 |
| 85% | 70.0% | 1.43 |

At 65% load, only 7% of operations need >1 probe. AVX2 can't help the
home-group fast path. For iteration, the bottleneck is bucket memory access,
not metadata SIMD loads.

---

## Struct Optimizations

### #22: Derive group_mask from num_groups — KEPT

Replaced stored `group_mask: usize` with inline `num_groups.wrapping_sub(1)`.
Single-cycle subtraction. Struct: 64 → 56 bytes.

### #23: Store mask instead of num_groups — KEPT

`mask` is used on the hot path (8 uses). `num_groups` derived as `mask + 1`
only on cold paths.

### Struct size evolution

| State | RawTable | UnorderedFlatMap | hashbrown |
|-------|--------:|-----------------:|----------:|
| Original | 56 | 64 | 40 |
| After #22-23 | 48 | 56 | 40 |

The 16-byte gap to hashbrown is the `buckets` pointer (8 bytes, needed for
two-allocation strategy) plus equivalent fields.

---

## Late Investigations

### #24: Single allocation re-test (with fused insert) — NOT COMMITTED

Same pattern: -4 to -9% at small-medium, +108% at 1M from page faults.

### #25: Home-group bucket prefetch — NOT COMMITTED

| Load % | Hit change | Miss change |
|-------:|:----------:|:-----------:|
| 45% | -6% | +9% |
| 65% | -8% | +6% |
| 85% | -4% | flat |

Workload-dependent with no universal win.

### #26: grow_from_empty overhead — ANALYZED

1.1-1.15x overhead at scale from 15-slot group arithmetic in the rehash loop.
`gi * 15 + si` costs ~2-3 extra instructions per element vs hashbrown's flat
index. Same structural overhead as lookup/iteration. Not fixable without
changing the group design.

---

## Summary Table

| # | Technique | Status | Key Finding |
|---|-----------|--------|-------------|
| 1 | Aligned loads + prefetch + cold | **Kept** | 15% lookup improvement |
| 2 | Single alloc (buckets-first) | Reverted | +40% insert (page faults) |
| 3 | Single alloc (meta-first) | Reverted | Worse than #2 |
| 4 | splitmix64 mixer | Reverted | +32-86% without avalanche opt-out |
| 5 | Fused find-or-locate | **Kept** | Entry API double-probe fix |
| 6 | SIMD IntoIter | **Kept** | Consistency |
| 7 | Initial bucket prefetch | Superseded | → #14 |
| 8 | IsAvalanching dispatch | **Partial** | Specialization hurts default |
| 9 | Size-adaptive alloc | Skipped | Not worth complexity |
| 10 | Single-group fast path | Superseded | → #16 |
| 11 | Conditional prefetch | Reverted | Branch > wasted prefetch |
| 12 | **foldhash** | **Kept** | **3-7x faster** |
| 13 | Single alloc + foldhash | Reverted | +66% insert 1M |
| 14 | Remove prefetch | **Kept** | -27% miss 1M |
| 15 | **Fused home-group insert** | **Kept** | **-35 to -48% insert** |
| 16 | Remove single-group branch | **Kept** | Marginal |
| 17 | Dense iteration | Reverted | +33% regression |
| 18 | Inline + cold continuation | Reverted | +10-14% hit regression |
| 19 | Custom fold | Reverted | +5-18% closure nesting |
| 20 | #[inline] entry | Reverted | +31% insert-heavy |
| 21 | AVX2 multi-group | Skipped | Targets wrong bottleneck |
| 22 | Derive group_mask | **Kept** | -8 bytes |
| 23 | Store mask | **Kept** | Hot path reads mask |
| 24 | Single alloc + fused | Not committed | +108% at 1M |
| 25 | Bucket prefetch | Not committed | No universal win |
| 26 | grow_from_empty | Analyzed | Structural, 15-slot arithmetic |
