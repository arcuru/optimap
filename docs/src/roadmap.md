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
| Load factor as type parameter | `LOAD_FACTOR_NUM`/`LOAD_FACTOR_DEN` constants on `GroupLayout` (default 7/8). Overflow-bit designs derive growth thresholds from the layout. Custom layouts can override to tune memory/speed trade-off. |
| Mid-pointer for 15-slot designs | Already implemented — UFM and Gaps share `overflow_table::RawTable<K,V,L>` which uses mid-pointer layout. Embedded overflow at byte 15 means exactly 2 memory regions, same as tombstone designs. |
| Borrow indirection in insert/entry | Investigated: already eliminated. Insert hot path uses `bucket.0 == key` directly. Cold fallback closures produce identical codegen via `#[inline(always)]` monomorphization. Added `find_by_hash_eq` wrapper for clarity, no perf impact. |
| Key-value separation (SoA layout) | `SoaRawTable<K,V,L>` + `SoaGenericMap` with separate key/value arrays. 7 matrix variants. Mid-pointer for keys, values after metadata+overflow. At 10K entries: competitive with Splitsies (32µs vs 31µs hit, 142µs vs 133µs insert for 256B values). Key-only probing may show more benefit at larger table sizes. |
| 32-slot (AVX2) and 64-slot (AVX-512) overflow-bit groups | `Group32<u32>` (1× 256-bit cmpeq+movemask) and `Group64<u64>` (1× 512-bit cmpeq_mask) added with compile-time `cfg(target_feature)` tier selection (AVX-512 → AVX2 → SSE2 → scalar Miri). New named layouts: `Splitsies32/64`, `Splitsies{32,64}_1bit`, `Hi8_1bit{32,64}`, `Top{128,255}_{1bit,8bit}And{32,64}`. Required `META_STRIDE`/`META_ALIGN` parameterization on `GroupLayout` and `meta_stride` parameter on `OverflowStrategy::overflow_ptr`. Initial benches at 9.4K entries / 70% load: 32-slot variants match 16-slot on hit/insert and slightly improve miss (`Top128_1bitAnd32`: 698 Mel/s miss vs 629 baseline, +11%); 64-slot underperforms 16-slot on hit/remove. No clear win at this size — wider groups may shine at higher load factors or for high-collision workloads, where home-group hit rate dominates. UFM/Gaps stay at 15-slot (embedded-overflow byte-15 trick is intrinsic to 16-byte metadata). |

## Open — Hash Maps

### API Completeness

| Item | Difficulty | Notes |
|------|-----------|-------|
| `raw_entry()` API | Medium | Custom key lookup by hash + eq. Niche. |

### Testing / Quality

| Item | Difficulty | Notes |
|------|-----------|-------|
| Allocator stress testing | Low | Custom allocator for misalignment and leak tracking. |

### Design Space Exploration

These explore new axes in the parameterized design matrix. Each is a new
composition of existing traits or a small trait extension.

#### ~~Sweep benchmarks for 32/64-slot variants~~ **CLOSED — full investigation completed**

See [32/64-Slot Investigation](optimization/32-64-slot-investigation.md) for
the complete analysis. Headline findings:

- **32-slot is not a net win for general use.** Slight insert advantage (5-15%
  at 1M+ entries) is offset by a permanent ~10% lookup_miss penalty from
  wider false-positive group checks. Lookup_hit is statistically tied (±3%).
- **64-slot is structurally slower** across all ops up to 2M entries (10-20%
  hit/insert, 40-60% miss). AVX-512 pays for itself only with pathological
  load factors outside realistic ranges.
- **Top255 is critical for wider groups**, confirming the hypothesis: base
  false-match rate of 1/255 vs 1/127 cuts spurious cache misses on miss
  paths by ~50% when probing 32-64 slots per step.
- **Embedded-overflow works well** at 32-slot: `Ufm32` (embedded, compact
  stride, low-byte 255-tag) is the best insert engine across all designs.
- **Full sweep harness updated**: the `for_each_design!` macro in
  `benches/sweep.rs` now includes all 32/64-slot separate, embedded,
  and AND-indexed variants (27 new designs). These will be available for
  future benchmark sweeps.
- **Data preserved**: raw CSV and gnuplot PNGs in `bench-results/`.
- **The shadow SIMD load (#6 below) remains open** but likely not worth
  fixing — both loads hit L1 (same cache line), saving one µOP would be
  invisible outside microbenchmarks.

#### Hot-path optimizations for 32/64-slot designs

**Difficulty**: Low-Medium \
**Expected impact**: Unknown per item — needs targeted benches

Candidates identified during the Group32/Group64 landing:

1. ~~**`bucket_index` shortcuts for 32/64 stride**~~ **Applied, inverse direction.**
   The existing 16-slot shortcut `(gi << 4) | si` was **worse** than the
   naïve `gi * 16 + si`: LEA can fuse `shift + add` into a single µop
   (`shlq`+`leaq` = 2 µops) but not `shift + or` (`mov`+`shlq`+`orq` =
   3 µops). Simplified `bucket_index` to just `gi * BUCKET_STRIDE + si`
   and trusted LLVM to fold the multiply; same change applied to
   `ipo64::bucket_ptr`. Bench signal was lost in machine noise at this
   granularity, but codegen is strictly better (1 µop saved per call).

5. ~~**Non-pow2 stride cost (Ufm32/Ufm64 compact stride)**~~ **Tiny,
   keep the design.** LLVM compiles Ufm32's `gi * 31` as `gi * 32 - gi`
   but reuses the `gi * 32` already computed for the meta_ptr, so the
   only actual cost is a single `sub %gi, %r9` (1 µop) per bucket access
   vs Gaps32's pow-2 stride. Saves 1/32 bucket of memory per group. Net
   trade is worth keeping both compact-stride (Ufm) and pow-2-stride
   (Gaps) variants at each width.
2. ~~**AVX-512 mask-register fusion**~~ **Verified optimal, no action.**
   Inspection of the matrix bench disassembly shows LLVM emits:
   - `vpcmpeqb (mem), %zmm0, %k0` — load fused with compare
   - `vptestmb` / `vptestnmb` for match_non_empty / match_empty (direct
     test against zero, no need for a broadcast zero register)
   - `kortestq` on k-registers for "any match" tests (no kmovq round-trip)
   - Single `kmovq` to GP only when iteration (`tzcnt`/`blsr`) is needed
   - `& SLOT_MASK` elided for full-width (all-ones) SLOT_MASK
   - **`match_byte_and_empty` reuses the load**: one `vmovdqa64 %zmm0`,
     then `vpcmpeqb` + `vptestnmb` both on `%zmm0` producing `%k0` and
     `%k1`. Zero spurious reloads.
   LLVM was already smarter than our source. Nothing to hand-optimize.
3. ~~**Inline propagation audit**~~ **Verified clean.** `objdump` of the
   matrix bench binary shows zero `call` instructions targeting
   `match_byte`/`match_empty`/`Group32`/`Group64` symbols. The raw
   SIMD ops appear directly at call sites: 68 × `vpcmpeqb %zmm` (AVX-512
   Group64), 83 × `vpcmpeqb %ymm` (AVX2 Group32), 359 × `vpcmpeqb %xmm`
   (SSE2 Group<>, VEX-encoded). Trait dispatch through `GroupOps` fully
   monomorphizes and inlines.
6. ~~**Embedded-overflow byte read adds a shadow SIMD load**~~ \
   **CLOSED — not worth fixing.** The 32/64-slot sweep investigation confirmed
   that even if we saved this one load-port µop, 32-slot designs still carry
   a structural ~10% lookup_miss penalty over 16-slot. The load-duplication
   is a symptom, not the cause. See [32/64-Slot Investigation](optimization/32-64-slot-investigation.md).

4. ~~**Top255 insert regression at 32/64-slot**~~ **Closed — not reproducible.**
   The initial `--quick` numbers showed Top255_1bitAnd{32,64} regressing
   vs Top128 on insert (−6%/−15%). A full-sample (100 samples) rerun
   flipped the sign: Top255 was +5%/+7% *faster*. Codegen analysis does
   show Top255 uses 1 more µop (`shr+cmp+adc` vs `shr+or`) and the
   inline-asm acts as a scheduling barrier — but the actual perf
   difference sits well inside measurement noise at medium size. Tag
   choice should be driven by false-match rate (255 strictly better at
   1/254 vs 128's 1/127), not per-op cost.

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
