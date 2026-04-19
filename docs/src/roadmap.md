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
| Code deduplication (`GenericMap` + `RawTableApi`) | Unified 5 identical map.rs files into `GenericMap<K,V,S,R>` and 3 overflow-bit raw tables into generic `RawTable<K,V,L: GroupLayout>`. -4,500 lines (-72%). Zero performance cost (monomorphized). See [Architecture](architecture.md). |
| Design space matrix | Parameterized tag extraction (`TagStrategy`, `TombstoneTag`), overflow storage (`OverflowStrategy`), and group indexing (`AND_INDEX`). 16 design variants benchmarked. See [Architecture](architecture.md). |
| Mid-pointer memory layout | Both RawTable impls use hashbrown's mid-pointer trick: single `ctrl` pointer between buckets (backward) and metadata (forward). Eliminates a struct field and address computation. Hi128_Tomb beats hashbrown: lookup hit 4.07 vs 4.25 ns, insert 503 vs 603 µs, remove 763 vs 1079 µs. |
| AND-based group indexing | `h & mask` (1 instruction) vs `h >> shift` (2 instructions). Applied to IPO tombstone and 1-bit overflow designs. Requires tags from top hash bits (57+) to avoid correlation. |
| Splitsies-1bit (BitSeparate) | Implemented as `OverflowStrategy` + `Layout16` composition. 1 bit per group instead of 1 byte. See Splitsies-1bit section below for design rationale. |

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

### Design Space Exploration

These explore new axes in the parameterized design matrix. Each is a new
composition of existing traits or a small trait extension.

#### 8-bit overflow + AND indexing (shifted channels)

**Difficulty**: Low — new TagStrategy only \
**Expected impact**: ~5% lookup for Splitsies-style designs

AND-based group indexing saves 1 instruction per probe but is currently
blocked for 8-bit overflow designs because `overflow_channel = 1 << (h & 7)`
uses low bits that correlate with the AND group index. Fix: shift the
channel source to top bits: `1 << ((h >> 57) & 7)`. The channel only
needs 3 bits of entropy decorrelated from the group index.

This would let all overflow-bit designs (including Splitsies) benefit
from AND indexing, not just 1-bit variants.

#### Key-value separation (SoA layout)

**Difficulty**: Medium — new RawTable variant or GroupLayout axis \
**Expected impact**: Potentially large for big-value workloads

Store keys and values in separate arrays instead of interleaved `(K, V)`
tuples. On the hit path, the key comparison only touches the key array —
value bytes don't pollute the cache line.

For `HashMap<u64, [u8; 256]>`, interleaved layout pulls 264 bytes per
slot into cache just to compare an 8-byte key. SoA layout only pulls
8 bytes for the key comparison, then fetches the 256-byte value only
on match.

Trade-off: two memory regions to address on hit (key then value) vs one.
Wins when `sizeof(V) > cache_line - sizeof(K)` (~56 bytes for u64 keys).
Needs investigation for the common small-KV case where interleaved
already fits in one cache line.

#### 32-slot AVX2 groups

**Difficulty**: High — new Group implementation \
**Expected impact**: Fewer probes for large tables

IPO64 already does 64-slot groups with AVX-512. A 32-slot AVX2 variant
would be a middle ground available on more hardware (AVX2 is ubiquitous,
AVX-512 is not). 32 bytes of metadata per group, `vpcmpeqb ymm` for
matching. Doubles the chance of a home-group hit vs 16-slot.

Trade-off: larger groups = more wasted space at low load, more false
matches to iterate through per group. Metadata is 32-byte aligned
instead of 16-byte.

#### Load factor as a type parameter

**Difficulty**: Low — const on GroupLayout \
**Expected impact**: Tuning knob for memory/speed trade-off

Currently hardcoded at 7/8 (87.5%) for tombstone designs. Making this
a const on GroupLayout would let users tune per design. Lower load
factor = fewer collisions + faster probing, but more memory waste.

#### Mid-pointer for 15-slot embedded designs (UFM, Gaps)

**Difficulty**: Medium \
**Expected impact**: ~5% lookup

UFM and Gaps embed overflow at byte 15 of each 16-byte metadata group —
no separate overflow region. They have exactly 2 memory regions (metadata
+ buckets), same as tombstone designs. The mid-pointer trick applies
cleanly. Would also benefit from AND indexing if combined with shifted
overflow channels.

### Structural (Speculative)

| Item | Difficulty | Risk | Notes |
|------|-----------|------|-------|
| Concurrent / lock-free variant | Very High | Research | Overflow bits are suited to lock-free reads. |

#### Splitsies-1bit: design rationale (implemented)

Implemented as `BitSeparate` overflow strategy composed via `Layout16`.
Replaces per-group overflow byte with a single overflow bit. The overflow
array becomes a compact bitfield: 1 byte per 8 groups instead of 1 byte
per group.

**Memory savings** (1-bit vs 8-bit overflow):

| Table size | Groups | 8-bit | 1-bit |
|-----------|-------:|------:|------:|
| 100K | ~6.4K | 6.4 KB | 800 B |
| 1M | ~64K | 64 KB | 8 KB |
| 10M | ~640K | 640 KB | 80 KB |

**Trade-off**: miss false-continuation rate rises from ~0.9% (8-channel)
to ~7% (binary). But the bitfield is always L1-hot, and at typical load
(<70%) overflow is rare enough that 1-bit vs 8-bit makes almost no
difference — the memory savings are pure upside.

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
