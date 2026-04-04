# Splitsies Optimization Plan

## Current State (Splitsies vs hashbrown at 70% load)

Wins: insert (0.93x), remove (0.64x), churn (0.62x), read-heavy (0.99x),
from_iter (0.62x), sequential miss (0.74x), miss@85% (0.28x)

Gaps: lookup hit (1.15x), iteration (1.14x), entry API (1.66x),
miss at low load (1.08-1.10x), counting (1.28-1.35x)

Struct size: 64 bytes (vs UFM 56, hashbrown 40)

## Improvements to Try

### S1: Drop overflow pointer from struct (derive from metadata)
**Gap**: 64 bytes → 56 bytes struct size
**Idea**: The overflow array starts at `metadata + num_groups * 16`.
Compute `overflow_ptr(gi)` as `self.metadata.add(self.num_groups() * 16 + gi)`
instead of storing a separate pointer. This saves 8 bytes.
**Risk**: Extra arithmetic on every overflow access. But `num_groups() * 16`
is just `(mask + 1) << 4`, which is a shift + add. Already on the cold path
(overflow only checked after SIMD miss).
**Expected**: Neutral performance (overflow access is prefetched), -8 bytes.

### S2: Home-group bucket prefetch
**Gap**: lookup hit 1.15x
**Idea**: Prefetch bucket data for the home group before the SIMD match,
same as UFM attempt #7. In UFM this helped hits 11% but hurt misses 21%.
In Splitsies, the overflow prefetch is already issued before the SIMD match,
so the prefetch port is already warmed. Adding a bucket prefetch might
compete for prefetch bandwidth.
**Why it might work differently**: Splitsies' overflow prefetch gives us
a "free" prefetch slot that UFM didn't have. The CPU may handle two
prefetches better than one.
**Risk**: Miss regression from cache pollution, same as UFM #7.

### S3: Inline find_by_hash home group with cold continuation
**Gap**: lookup hit 1.15x
**Idea**: UFM attempt #18 failed because #[inline(never)] on the
continuation caused register pressure in the inlined hot path.
Splitsies might behave differently because:
- No `& 0x7FFF` mask means fewer registers used
- `bucket_ptr` is simpler (`gi << 4 | si` vs `gi * 15 + si`)
- The overflow prefetch is separate (different register pattern)
**Risk**: Same register pressure issue as UFM. But worth testing since
the register landscape is different.

### S4: Entry API `#[inline]`
**Gap**: entry API 1.66x
**Idea**: UFM attempt #20 showed #[inline] on entry() helps hit-heavy
(-7%) but hurts insert-heavy (+31%). Splitsies has faster insert, so the
insert-heavy penalty might be smaller.
**Risk**: Same code bloat trade-off. But the absolute entry gap (1.66x)
is large enough that even a partial improvement is valuable.

### S5: Remove needs_rehash anti-drift optimization
**Gap**: remove+reinsert pattern (0.79x, but UFM was 1.23x — anti-drift
hurts the 15-slot design more)
**Idea**: The anti-drift check reads the home group's overflow byte on
every remove. For Splitsies this is a separate array access. Could we
batch the anti-drift check or defer it?
Alternative: precompute the home group during find_by_hash and pass it
to remove_by_hash, avoiding the redundant group_index + overflow_ptr
computation.
**Expected**: Small improvement on remove-heavy workloads.

### S6: Explore removing overflow_bit from find_by_hash hot path
**Gap**: lookup hit 1.15x (overflow_bit is 1 shift, computed but unused on hits)
**Idea**: Compute overflow_bit lazily after SIMD match fails (same as
we did for insert_no_check). On the hit path, overflow_bit is never used.
**Why it failed on UFM**: Deferring overflow_bit in UFM attempt #18 moved
it to the serial critical path for misses. But if we combine it with the
prefetch (which already hides latency), the serial dependency might not matter.
**Risk**: Miss path becomes slower if the computation can't overlap with
the prefetch.

### S7: Combined metadata+overflow memset in allocate
**Gap**: Construction speed (minor)
**Idea**: In allocate(), we zero metadata and overflow separately. Since
they're contiguous, zero both in one call (like we already do in clear()).
**Expected**: Minor, only affects construction.

### S8: Struct field reordering for cache alignment
**Gap**: 64-byte struct (exactly one cache line)
**Idea**: Reorder struct fields so the hottest fields (mask, metadata,
shift) are in the first 32 bytes, and cold fields (overflow, buckets,
len, max_load) are in the second half. On L1 cache hits, only the first
half-line might be needed for group_index.
**Expected**: Marginal. Modern CPUs fetch full cache lines.

## Implementation Order

1. **S1** (drop overflow pointer) — free 8 bytes, test performance
2. **S7** (combined memset in allocate) — trivial fix
3. **S6** (lazy overflow_bit in find_by_hash) — might close hit gap
4. **S5** (anti-drift optimization in remove) — improve remove path
5. **S2** (home-group bucket prefetch) — test hit/miss trade-off
6. **S3** (inline find_by_hash home group) — test if register pressure is different
7. **S4** (entry API #[inline]) — test with Splitsies' different codegen
8. **S8** (struct field reorder) — marginal, try last
