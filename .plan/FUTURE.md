# Future Improvements

Ordered roughly by expected impact. Each entry notes what benchmark
gap it targets and the estimated difficulty.

## Completed

- **Fused home-group insert**: Single SIMD load for find + insert. Closed
  the 1.7x insert gap at 10K-100K to parity. Now faster at 1K and 1M.
- **Fused home-group entry**: Same pattern for entry API. ~5-8% improvement.
- **Remove single-group fast path**: Simplified find_by_hash; marginal improvement.
- **Dense iteration fast path**: REVERTED — extra branch per next() hurt more
  than sequential scan helped. tzcnt + blsr is already near-optimal.
- **Inline home group in find_by_hash with cold continuation**: REVERTED —
  #[inline(never)] continuation caused 10-14% hit regression from register
  pressure at the call boundary.

---

## Performance

### 1. Lookup hit: reduce per-probe overhead
**Gap**: 1.25x slower than hashbrown on hits, constant across all load factors.
**Root cause**: Our probe loop does more work per step — 15-slot groups (vs 16-byte
aligned), overflow-bit bookkeeping, and slightly wider metadata reads. hashbrown's
inner loop is extremely tight: one aligned `_mm_load_si128`, one `_mm_cmpeq_epi8`,
one `_mm_movemask_epi8`, mask to 14 bits, done.

**Ideas**:
- **Branchless key comparison**: If the SIMD match yields exactly one candidate
  (common at low-mid load), skip the bitmask iteration and directly compare that
  single slot. The bitmask → `trailing_zeros` → index calculation can be fused.
- **Prefetch bucket on SIMD match**: When `match_byte` returns non-zero, issue a
  prefetch for the first matching bucket slot before entering the comparison loop.
  This overlaps memory latency with the bitmask iteration. (Previously tested as
  unconditional prefetch — conditional on match may avoid the miss regression.)
- **Profile-guided group sizing**: Consider whether a compile-time option for
  7-slot groups (8-byte metadata) could help on workloads where cache line
  utilization matters more than SIMD width.

**Difficulty**: Medium. Mostly micro-optimization of the hot loop.

### 2. Insert 10K-100K — DONE
**Solved by fused home-group insert.** Single SIMD load for both find and insert
in the home group. Closed 1.7x gap to parity; now faster at 1K and 1M.

Further ideas for marginal gains:
- **Deferred overflow-bit setting**: Batch overflow bit updates in extend/from_iter.
- **Rehash without re-hashing**: Store full hash to avoid re-hashing on grow.

### 3. Entry API — DONE (partially)
**Improved by ~5-8%** via fused home-group entry. Remaining gap (~1.2-1.7x)
is dominated by lookup hit overhead (structural) and Entry enum construction.

Further ideas:
- Avoid moving key into OccupiedEntry (reference-based API)
- Specialize for small key types that are cheap to copy

### 4. Iteration: close the 1.4x gap at small sizes
**Gap**: 1.4-1.5x slower at 1K-100K, nearly tied at 1M.
**Root cause**: hashbrown's 16-byte aligned metadata means its iterator can
use aligned loads and the full 16-bit movemask. Our 15-slot groups waste one
SIMD lane on the overflow byte, requiring a mask-off step.

**Ideas**:
- **Dense-group fast path**: When `match_non_empty` returns 0x7FFF (all 15 full),
  yield all 15 elements sequentially without bitmask iteration. ~50-60% of groups
  are full at typical load. This turns bitmask iteration (15 branch+shift ops)
  into a single comparison + linear scan.
- **Prefetch next group**: Issue a prefetch for the next group's buckets while
  iterating the current group. This is especially effective for `sum()`-style
  traversals.

**Difficulty**: Low-Medium.

### 5. Selective prefetch for hit-heavy workloads
**Gap**: We removed all manual prefetching to favor misses. This costs ~18%
on hits at 1M scale.
**Root cause**: Without prefetch, every hit requires a cache miss on the bucket
array. At 1M, bucket data is not in L2/L3.

**Ideas**:
- **User-selectable prefetch policy**: A compile-time or runtime flag that
  enables bucket prefetch on the first probe step. Users who know their workload
  is hit-dominated can opt in. Default remains no-prefetch (miss-optimized).
- **Adaptive prefetch**: Track hit/miss ratio at runtime (simple counter, checked
  every N ops) and toggle prefetch behavior. High complexity, unclear benefit.

**Difficulty**: Low (flag), High (adaptive).

### 6. SIMD backend: AVX2 / AVX-512
**Ideas**:
- **AVX2 (256-bit)**: Process two groups simultaneously in find_by_hash. The
  probe sequence visits groups in a known order — load groups 0 and 1 into a
  single 256-bit register and compare. Halves the number of loop iterations.
- **AVX-512**: Potentially process 4 groups at once, or use mask registers
  for more efficient bitmask handling.
- Must be behind runtime feature detection (`is_x86_feature_detected!`).

**Difficulty**: High. Needs careful benchmarking — wider SIMD can hurt if it
causes downclocking or cache pressure.

---

## API Completeness

### 7. `reserve()` / `shrink_to_fit()`
Standard HashMap API. `reserve(n)` pre-allocates for at least `n` additional
elements. `shrink_to_fit()` reallocates to the minimum table size.

**Difficulty**: Low.

### 8. `drain()` iterator
Returns an iterator that removes and yields all elements. The table is empty
after `drain()` completes. Needed for `HashMap` API parity.

**Difficulty**: Low-Medium.

### 9. `retain(&mut self, f: FnMut(&K, &mut V) -> bool)`
Remove all elements for which `f` returns false. More efficient than iterating
+ collecting keys + removing individually.

**Difficulty**: Low.

### 10. `raw_entry()` API
Advanced API for custom key lookup (e.g., looking up by hash + custom eq
without constructing the key type). Used by advanced consumers like
compiler symbol tables.

**Difficulty**: Medium.

### 11. `try_insert()` → `Result<&mut V, OccupiedError>`
Returns an error with the existing entry instead of silently replacing.
Stabilized in Rust std as of 1.82.

**Difficulty**: Low.

---

## Structural / Architectural

### 12. Interleaved memory layout
`[group0_meta][group0_buckets][group1_meta][group1_buckets]...`

Each group's metadata and buckets are adjacent in memory. This should improve
spatial locality — after the SIMD metadata match, the bucket data is in the
same or adjacent cache line.

**Risk**: Large bucket types (e.g., `HashMap<String, Vec<u8>>`) would push
the metadata of adjacent groups far apart, potentially hurting the probe
loop. Would need a threshold (e.g., only interleave when
`size_of::<(K,V)>() * 15 <= 960` so meta + buckets fit in ~1KB).

**Difficulty**: High. Major refactor of memory layout, allocation, and
pointer arithmetic.

### 13. Generic group size
Make `GROUP_SIZE` a const generic parameter. Smaller groups (7-slot, 8-byte
metadata) could be better for small tables where 15-slot groups waste space.
Larger groups (31-slot, 32-byte metadata with AVX2) could be better for
large tables.

**Difficulty**: High. Touches every part of the codebase. Unclear benefit
without extensive benchmarking.

### 14. Concurrent / lock-free variant
A read-optimized concurrent hashmap using the overflow-bit design. The
overflow bits are particularly suited to lock-free reads: a miss can be
determined without any atomic operations beyond a single byte read.

**Difficulty**: Very High. Research-level.

---

## Testing / Quality

### 15. Miri testing
Run the test suite under Miri to verify absence of undefined behavior,
especially around raw pointer manipulation in the bucket array and SIMD
intrinsics.

**Difficulty**: Low-Medium. May need to disable SIMD paths under Miri
(Miri doesn't support most intrinsics) and test the scalar fallback.

### 16. Fuzzing harness
Property-based fuzzing (cargo-fuzz / AFL) that generates random operation
sequences and verifies our map matches `std::HashMap` behavior exactly.

**Difficulty**: Low.

### 17. Allocator stress testing
Test with a custom allocator that returns misaligned or high-address memory
to catch alignment assumptions. Test with an allocator that tracks all
allocations to verify no leaks.

**Difficulty**: Low.
