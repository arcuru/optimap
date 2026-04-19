# Future Work

Ordered roughly by expected impact. Items in the "Closed" section have been
thoroughly investigated and proven unproductive — see
[Closed Investigations](optimization/closed.md) for details.

## Recently Completed

### API Completeness (April 2025)

| Item | Scope |
|------|-------|
| `try_insert()` | All 6 designs, OptiMap, OptiSortedMap, `Map` trait (default impl). Returns `Result<(), OccupiedError<K, V>>`. |
| `into_keys()` / `into_values()` | All 6 designs, OptiMap, OptiSortedMap, FlatBTree, `Map` trait |
| `get_key_value()` / `remove_entry()` | All maps, `Map` trait |
| `iter_mut()` / `keys()` / `values()` / `values_mut()` | All maps, `Map` trait (defaults for `keys`/`values`/`values_mut`) |
| `reserve()` / `shrink_to_fit()` | All hash maps + FlatBTree, `Map` trait |
| `drain()` iterator | All hash maps + FlatBTree, `Map` trait |
| `retain(&mut self, f)` | All hash maps + FlatBTree, `Map` trait |
| Entry: `and_modify()` / `or_insert_with_key()` / `into_key()` | All 6 map types |
| `pop_first()` / `pop_last()` | FlatBTree, `SortedMap` trait |
| `SortedMap` for `std::BTreeMap` | `pop_first` / `pop_last` added |
| Enum iterators for OptiMap | Replaced `Box<dyn Iterator>` — zero-cost dispatch for `Iter`, `IterMut`, `IntoIter` |
| OptiSet / OptiSortedMap / OptiSortedSet | Smart wrappers with dynamic backend selection and sorted ops |
| Set benchmarks | Insert, contains, remove, iter, churn across all 8 set types |
| OptiMap Entry API | Enum `Entry`/`OccupiedEntry`/`VacantEntry` wrapping all 5 backends with `entry_match!` macro dispatch. Also added `OccupiedEntry::key()` to all backends. |
| FlatBTree VacantEntry direct return | `insert_at_vacant()` returns `(leaf_idx, slot_idx)` directly — no re-search needed. Entry counting workload now within ~2% of BTreeMap. |
| Miri testing (all designs) | Scalar SIMD fallbacks gated on `cfg(miri)`. 291 unit + 12 stress + 66 set_trait tests pass under Miri. Fixed 1 UB: group test helpers deallocating with wrong alignment. Zero UB in production code (841 unsafe blocks across 19 files). |
| Sweep benchmark harness | Ankerl-style N-sweep (100–10M, 362 points, median-of-5 trials) with CSV output + gnuplot visualization. Captures rehash sawtooth, cache boundary transitions, and load factor cycling. `./scripts/sweep-bench.sh` |
| Static empty sentinel | All 5 raw tables use a static SIMD-loadable sentinel instead of null metadata pointer, removing a branch from the find hot path. Measured ~0% impact (branch was already predicted). |
| find_bucket (direct pointer return) | All 5 raw tables expose `find_bucket()` returning `*mut (K,V)` directly, eliminating double `bucket_ptr` computation in `get/get_mut/get_key_value`. Measured ~0% impact (LLVM CSE already optimizing). |
| Large-value insert regression | Investigated and found non-reproducible — Splitsies beats hashbrown at all value sizes (0.84-0.93x). Original numbers were from a different machine. |
| Hash tag optimization (`hash_tag`) | Inline asm `cmp 0xFF; adc 0` (2 instructions, 255 values) replaces 3-instruction pure Rust. Feature-gated: `reduced-hash-asm` (default), `reduced-hash-128`, or pure Rust fallback. UFM sees -26% hit / -41% miss due to codegen scheduling effect. |

## Open — Hash Maps

### API Completeness

| Item | Difficulty | Notes |
|------|-----------|-------|
| `raw_entry()` API | Medium | Custom key lookup by hash + eq. Niche. |

### Performance

| Item | Difficulty | Notes |
|------|-----------|-------|
| Eliminate Borrow indirection in insert/entry | Medium | `find_by_hash_eq(&K)` that compares directly. |

### Testing / Quality

| Item | Difficulty | Notes |
|------|-----------|-------|
| Allocator stress testing | Low | Custom allocator for misalignment and leak tracking. |

### Structural (Speculative)

| Item | Difficulty | Risk | Notes |
|------|-----------|------|-------|
| **Splitsies-1bit** (new design) | Medium | Low | See below. |
| Interleaved memory layout | High | High | Better spatial locality, but large bucket types push groups apart. |
| Generic group size | High | Unclear | `GROUP_SIZE` as const generic. |
| Concurrent / lock-free variant | Very High | Research | Overflow bits are suited to lock-free reads. |

#### Splitsies-1bit: single-bit overflow

A variant of Splitsies where the per-group overflow byte is replaced by a single
overflow **bit**. The overflow array becomes a compact bitfield instead of a byte
array.

**Layout (same as Splitsies except overflow):**
- 16-slot groups: full 16 bytes for hash tags (all SIMD lanes used)
- Separate bucket array: `(K, V)` pairs
- Overflow bitfield: 1 bit per group (vs Splitsies' 1 byte per group)

**How it works:**
- On insert overflow (home group full → probe to next group): set the home
  group's overflow bit to 1
- On miss: if the home group has no SIMD tag match AND its overflow bit is 0,
  the key is definitely absent → O(1) miss termination (same as Splitsies)
- On miss with overflow bit = 1: must continue probing (same as Splitsies)
- Tombstone-free: same deletion strategy as Splitsies (backward-shift or
  equivalent), clearing the overflow bit when no more overflows exist

**Memory savings:**
| Table size | Groups | Splitsies overflow | 1-bit overflow |
|-----------|-------:|-------------------:|---------------:|
| 10K elements | ~640 | 640 bytes | 80 bytes |
| 100K | ~6.4K | 6.4 KB | 800 bytes |
| 1M | ~64K | 64 KB | 8 KB |
| 10M | ~640K | 640 KB | 80 KB |

At 1M elements the overflow array drops from 64 KB (L1-sized) to 8 KB — easily
fitting in L1 with room to spare. At 10M the savings are 560 KB.

**Tradeoff — miss path false positives:**

Splitsies' byte-level overflow stores `1 << (h & 7)`: 8 independent channels.
A miss probe continues only if the *specific* bit for that hash is set. With
1-bit overflow, a miss probe continues whenever *anything* overflowed from that
group — more false continuation.

At 70% load with 16-slot groups:
- ~7% of groups have any overflow at all (most lookups still terminate in O(1))
- Splitsies byte: of that 7%, only 1/8 of misses match the specific overflow bit
- 1-bit: of that 7%, all misses must continue probing

So the miss false-continuation rate rises from ~0.9% (7% × 1/8) to ~7%. This
matters most at high load (85%+) where overflow rates climb to ~30%.

**Why it might win anyway:**
1. The overflow bitfield is so small it's always hot in L1 — no cache miss to
   check it, ever. Splitsies' byte array at 1M+ can spill out of L1.
2. Checking a single bit (`bitmap[gi / 8] & (1 << (gi % 8))`) is simpler than
   loading a byte and testing a specific bit pattern (`overflow[gi] & (1 << (h & 7))`).
3. The extra probing on the 7% false-positive misses costs ~15-20ns per extra
   group checked. If the cache savings on the other 93% of operations save even
   1-2ns each, it could break even or win.
4. At low-medium load (<70%), overflow is rare enough that 1-bit vs 8-bit
   makes almost no difference — and the memory savings are pure upside.

**Implementation notes:**
- Could start as a fork of Splitsies with the overflow array replaced by a
  `Vec<u8>` bitfield (1 byte per 8 groups)
- The `overflow_bit(h)` function is no longer needed — the hash doesn't index
  into the overflow; it's just a boolean per group
- Deletion needs to check whether any elements in the group still have overflows
  before clearing the bit (scan the group's probe chains)
- insert/remove of the overflow bit is simpler: `bitmap[gi >> 3] |= 1 << (gi & 7)`
  to set, more complex to clear (must verify no remaining overflows)

## Open — FlatBTree

### Performance

| Item | Difficulty | Notes |
|------|-----------|-------|
| Remove rebalancing (steal/merge) | Medium | Currently lazy (no rebalancing on remove). Tree stays valid but wastes memory under heavy churn. Low-watermark nodes are never reclaimed. |
| Child node prefetching | Low | Prefetch next child's cache lines during internal node scan. Already faster than BTreeMap — diminishing returns. |

### API Completeness

| Item | Difficulty | Notes |
|------|-----------|-------|
| `range_mut()` | Low-Medium | Mutable range iteration. |
| Arena `shrink_to_fit()` | Medium | Current impl is a no-op. Compaction requires rebuilding the tree to eliminate free-list gaps. Bulk-load from drain could work. |

## Closed

These have been extensively tested and proven structural. See
[Closed Investigations](optimization/closed.md) for full documentation.

| Item | Why Closed |
|------|-----------|
| Lookup hit gap (1.11-1.25x) | Per-probe instruction count is inherent to overflow-bit design. 7 attempts across 2 designs, all failed or traded hit for miss. |
| Selective prefetch policy | No universal policy exists. Design selection (IPO vs Splitsies vs UFM) is the prefetch policy. |
| AVX2/AVX-512 for 16-slot groups | 93%+ of probes resolve in home group (one SSE2 load). AVX2 targets the wrong bottleneck. Implemented for IPO64 only. |
| Dense iteration fast path | `tzcnt` + `blsr` is already ~2 cycles/element. Extra branch per `next()` caused +33% regression. |
| Custom Iterator::fold | Nested closure chain generates worse code than default `next()`-based fold. +5-18% regression. |
| #[inline] on entry API | Helps hit-heavy (-7%), hurts insert-heavy (+31%). Compiler heuristics are correct. |
| Inline find_by_hash + cold continuation | Register pressure at `#[inline(never)]` boundary. +10-14% regression on 2 designs. |
