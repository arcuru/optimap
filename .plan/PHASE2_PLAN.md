# Phase 2 Optimization Plan

Based on comprehensive benchmarks showing gaps vs hashbrown.
Ordered by expected impact × feasibility.

## Batch 1: Low-risk hot-path improvements

### P1: Precompute bucket stride in RawTable — REVERTED
- Storing runtime `group_stride` field replaced compile-time constant multiply
- Compiler already optimizes `(gi * 15 + si) * size_of` via lea+shift
- Runtime field load from struct was slower than the constant-folded math
- **Result**: Slight regression, reverted

### P3: Eliminate Borrow indirection in insert/entry — TODO
- Add `find_by_hash_eq(&K)` that compares directly without Borrow trait
- Use from insert() and entry() where we already have `&K`
- Keep Borrow path for get()/remove() where Q may differ
- **Target**: lookup hit, entry API

### P7: Overflow-only prefetch in find_by_hash — DONE
- Prefetch next group metadata + buckets only after overflow-bit check
- Doesn't fire on miss fast path (which terminates at overflow check)
- **Target**: lookup hit at large sizes

### P8: Fused match_byte + match_empty SIMD op — DONE
- `Group::match_byte_and_empty(ptr, value) -> (BitMask, BitMask)`
- One load, two compares, two movemask
- Used in find_or_locate to avoid double SIMD load per probe step
- **Target**: insert, entry API

## Batch 2: find_or_locate restructuring

### P6: Home-group fast path in find_or_locate — DONE
- Inline home-group check before entering probe loop
- #[inline(never)] overflow slow path for cold code
- Single SIMD match_byte_and_empty on home group resolves most operations
- **Target**: insert 1K-100K

### P2: Carry overflow bitmask through ProbeResult — DONE
- InsertSlot(gi, si, full_mask) carries u8 bitmask of full groups
- insert_at iterates bitmask to set overflow bits (no re-walking)
- Eliminates redundant SIMD match_empty loads on insert path
- **Target**: entry API

## Batch 3: Specialized paths

### P4: Dense iterator fast path
- When match_non_empty returns 0x7FFF (all 15 full), yield sequentially
- Avoids bitmask iteration (15 branch+shift ops → 1 comparison + counter)
- ~50-60% of groups are full at 87.5% load
- **Target**: iteration at small sizes

### P5: Rehash without re-hashing (deferred)
- Store full hash or reconstruct from metadata on 2x growth
- High complexity, only matters for grow-heavy workloads
- **Target**: insert/erase phases
