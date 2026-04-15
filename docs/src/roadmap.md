# Future Work

Ordered roughly by expected impact. Items in the "Closed" section have been
thoroughly investigated and proven unproductive — see
[Closed Investigations](optimization/closed.md) for details.

## Open

### API Completeness

These are straightforward to implement and needed for HashMap API parity.

| Item | Difficulty | Notes |
|------|-----------|-------|
| `reserve()` / `shrink_to_fit()` | Low | Standard pre-allocation / compaction API |
| `drain()` iterator | Low-Medium | Remove + yield all elements |
| `retain(&mut self, f)` | Low | Filter in-place, more efficient than collect + remove |
| `try_insert()` → `Result<&mut V, OccupiedError>` | Low | Stabilized in std as of Rust 1.82 |
| `raw_entry()` API | Medium | Custom key lookup by hash + eq. Niche but used by compilers |

### Performance

| Item | Difficulty | Notes |
|------|-----------|-------|
| Eliminate Borrow indirection in insert/entry | Medium | Add `find_by_hash_eq(&K)` that compares directly without Borrow trait. Use from `insert()` and `entry()` where we already have `&K`. Keep Borrow path for `get()`/`remove()` where Q may differ. |
| Large-value insert regression (Splitsies 128B+) | Medium | Splitsies is 1.48-1.65x slower than hashbrown for 128B+ values. Needs investigation. |

### Testing / Quality

| Item | Difficulty | Notes |
|------|-----------|-------|
| Miri testing | Low-Medium | Verify no UB in raw pointer / SIMD code. Needs scalar fallback (Miri doesn't support SIMD intrinsics). |
| Fuzzing harness | Low | Property-based fuzzing: random op sequences, verify against std::HashMap. |
| Allocator stress testing | Low | Custom allocator for misalignment, high addresses, leak tracking. |

### Structural (Speculative)

| Item | Difficulty | Risk | Notes |
|------|-----------|------|-------|
| Interleaved memory layout | High | High | `[group0_meta][group0_buckets][group1_meta]...` — better spatial locality, but large bucket types push groups apart. |
| Generic group size | High | Unclear | `GROUP_SIZE` as const generic. Smaller groups for small tables, larger for big ones. Touches everything. |
| Concurrent / lock-free variant | Very High | Research | Read-optimized concurrent map. Overflow bits are suited to lock-free reads (miss = one byte read, no atomics). |

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
