# Future Work

Ordered roughly by expected impact. Items in the "Closed" section have been
thoroughly investigated and proven unproductive — see
[Closed Investigations](optimization/closed.md) for details.

## Recently Completed

### May 2026

| Item | Scope |
|------|-------|
| `FlatBTree::split_off` cutoff retuned 0.6 → 0.70 | The cost asymmetry between `surgical_left` and `surgical_right` (2-3× constant-factor on random-insert builds — `surgical_left` does per-spine-level array compaction shifts, an extra `mem::swap`, and "temp chain prepend" passes that `surgical_right` skips) means we should prefer `surgical_right` even when it copies somewhat more entries. Combined with the existing +0.10 estimator bias, the optimal biased-r cutoff sits at ~0.75; 0.70 leaves margin for noise. **At p045 / 1M random-build the dispatcher now picks `surgical_right` (5.7ms) instead of `surgical_left` (12.1ms) — 2.1× speedup**. p030 / 1M still routes correctly to `surgical_left` (estimator says ≈ 0.80). p050 and asymmetric pivots unchanged. New `DEFAULT_RIGHT_FRAC_LEFT_CUTOFF` const + `split_off_with_cutoff(at, cutoff)` benchmark hook exposed `#[doc(hidden)]`. Sequential-insert builds (the criterion bench) show the change is benign — at p050 the variants tie regardless of routing. |
| `FlatBTree::append` tree-surgery graft (disjoint adjacent) | Added `append_graft` (alongside `append_drain`, `append_concat`, `append_extend`) and routed the public `append` dispatcher through a O(1) disjointness check on `self.last_key` vs `other.first_key`. When ranges are disjoint adjacent (the common "split then re-merge" case), the graft path byte-copies other's nodes into self.arena, splices the leaf chain at self.last_leaf, and bridges the two roots — equal heights add a single new root, otherwise descend self's right spine to attach `other.root` as the new last child of an internal node at level `other.height + 1` (cap-cascade splits handled via the existing `propagate_split`). When `self.height < other.height`, falls back internally to drain. Bench wins on disjoint random keys (each in [0, 2^33) and [2^33, 2^34) so disjointness is guaranteed): **100K symmetric: 3.50ms vs drain 6.89ms (2.0×), vs std 8.16ms (2.3×)**; **1M symmetric: 79ms vs drain 142ms (1.8×), vs std 139ms (1.8×)**; **1M asymmetric (other=100): dispatch 17.5ms vs drain 69ms (3.9×), vs std 33ms (1.9×)**. The `append_extend` variant (drain other only, then per-key tail-insert) is competitive at small `m` but loses to graft at symmetric large-N. `append_concat` (chain drained sequences, skip merge step) saves ~10-30% over drain by avoiding the order-comparison merge but still rebuilds self's leaves. 13 new tests + cross-validation across all four variants. Bench helper bug fixed: prior `make_random_keys(...).wrapping_add(1<<33)` did NOT bound the key range — random u64 input wraps freely, so the "disjoint adjacent" benches were actually overlapping. Now bounded to half-u64 ranges. |
| `FlatBTree::split_off` bidirectional surgery (copy smaller side) | Added `split_off_surgical_left` mirror of `split_off_surgical_right`. The kept side stays in `self.arena` (mutated in-place); the smaller side is deep-copied to a fresh arena, then `mem::swap` flips them so `self` retains `< at` and the returned tree holds `>= at`. The dispatcher now picks direction by which side the estimator says is smaller (cutoff 0.6 to absorb the +0.10 estimator bias) and **drain is no longer routed to** — with both directions available, the side-of-min surgical always beats drain. Bench wins vs the prior drain-fallback dispatcher: **1K-p001 4.9×** (3.1µs → 622ns), **100K-p001 7.2×** (315µs → 44µs), **100K-p010 5.5×** (319µs → 58µs), **1M-p001/p010 2.3-2.4×** (13ms → 5.6ms). Mid/high pivots unchanged (already routed to surgical). 13 new copy-left tests + 3-way cross-validation against drain & surgical_right pass; copy-left small-N suite passes miri. `split_off_drain` retained as `#[doc(hidden)] pub` for benchmark comparison. |
| `FlatBTree::split_off` tree-surgery + spine-aware dispatch | Added `split_off_surgical_right` (O(log n + right_subtree_nodes)) alongside the v1 drain+bulk_load path. Public `split_off` dispatches based on a cheap O(log n) right-fraction estimate from the spine. **Wins**: at 100K, 1.8-2.2× faster across p050-p099 (266µs vs 485µs at half-split); at 1M-p090, 7.5ms vs drain's 15.6ms (2.1×, beats std). **Limit**: dispatch fell back to drain at very-low pivots on huge trees (1M-p001) where deep-copying ~99% of the tree was slower than O(n) drain — closed by the bidirectional follow-up above. All 12 surgical-path tests pass under miri (zero UB). |
| OptiMap policy: data-driven backend bands | Switch to Splitsies above the tombstone-DRAM cliff (~1M entries). IPO/IPO64 suffer 5-13× lookup_miss regression at ≥1M due to tombstone accumulation. New thresholds: `< 1024` Splitsies, `1024..1M` IPO, `≥1M` Splitsies. ReadHeavy / WriteHeavy hints fall back to Splitsies above the cliff. Boundary tests added. |
| `OptiSet` / `OptiSortedSet` operator parity | Added `BitOr`/`BitAnd`/`BitXor`/`Sub` operators on `&Set` (matching std::HashSet / std::BTreeSet). |
| `GenericSet::intersection` swap fix | Prior impl iterated `self` in *both* branches — when `self` was larger we still iterated the larger set. Fixed to iterate the smaller. |
| `FlatBTree::range_mut()` | Mutable variant of `range()`. New `RawBTree::resolve_range_bounds` shared between `range` and `range_mut`. Iterator stores tree as `*mut` for non-aliasing 'a-tied yields. Closes the `range_mut()` roadmap item. |
| `FlatBTree::from_sorted_iter()` | Bulk-load for already-sorted input. Skips the sort+dedup pass `FromIterator` does. |
| `bulk_load` separator-key bug fix | For trees with height ≥ 2, the separator key promoted into an internal parent at position j was read from the right child's `internal_key_ptr(0)` (the child's *own* first internal separator), not the leftmost leaf-key in the right subtree. Effect: `get` returned `None` for some keys on any `FromIterator` input ≳ 300 entries (u64/u64). The 100-entry `from_iterator` test produced only a single-level tree and couldn't catch it. Fix tracks parallel `min_keys: Vec<K>` across build levels. New 100K-entry `FromIterator` + `from_sorted_iter` multi-level tests pin down the regression. |
| `FlatBTree::shrink_to_fit` (was no-op) | Drains the tree in sorted order into a `Vec`, bulk-loads a fresh tree, swaps. Releases unused arena slots *and* compacts leaves from ~50% utilization (split-on-insert legacy) up to `LEAF_CAP`. |
| Roadmap accuracy: rebalance entry was stale | "Remove rebalancing (steal/merge) — Currently lazy" was incorrect. `RawBTree::remove` already calls `rebalance_leaf` on underflow (`raw.rs:1217`), which steals from right/left siblings or merges; the cascade up through `rebalance_internal` / `merge_internals` keeps internal nodes within `INTERNAL_CAP/2..INTERNAL_CAP`. Entry moved out of "Open" — only arena compaction *across* heavy churn remains, and `shrink_to_fit` already covers that on demand. |
| `Hash` impl + `IntoIterator for &mut FlatBTree` | std::BTreeMap parity; matches std hash semantics (len + sorted (k,v) pairs). Enables `for (k, v) in &mut tree`. |
| `FlatBTree::split_off(at)` + `append(&mut other)` | std::BTreeMap parity. v1 is drain → bulk_load (O(n)). At 100K, beats std `split_off` (800µs vs 1.38ms); at 1M loses to std's structural split (38ms vs 17ms). `append` ties std at 100K and 1M (both ~5% faster). Tree-surgery split_off filed as a follow-up. |
| `SortedMap` / `SortedSet` trait `split_off` + `append` | Lifted into trait so generic code can call them; default impls delegate to std on `BTreeMap`/`BTreeSet`; `OptiSortedMap`, `OptiSortedSet`, `GenericSet` mirror inherent + trait impls. |
| Hybrid binary/linear node search | `RawBTree::search` and `lower_bound` now switch on tree height: linear scan for `height < 3` (≤ ~1K entries, branch-prediction friendly), binary search for taller trees (cache-miss dominated). At 10K/100K: lookup_hit −14 to −22%, lookup_miss −15 to −24%. At 1K: small ~5–7% regression from the dispatch branch. Net win at the scale FlatBTree was designed for. |

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
| OptiMap/OptiSet dispatch inlining | `#[inline(always)]` on hot inherent methods (get/insert/remove/contains_key/len/…) and `#[inline]` on `Map`/`Set` trait impls. Disassembly showed `<wrapper as Map>::get` was a 1.1 KB function called once per loop iteration: the 5-arm enum dispatch + inlined backend probe was too large for LLVM's heuristic to inline through a wrapping function-call boundary, so any layer above OptiMap (a user struct, the bench's `OptiMapBench`) forced a real `call` per lookup. Closes the gap vs raw IPO at 107 K entries: lookup_hit +70% → −1%, lookup_miss +116% → −6%, remove +19% → ~0%. Cost: +3 KB (+0.04%) in the throughput bench binary; `liboptimap.rlib` unchanged (inline is downstream-only metadata). Same caveat as hashbrown — callers wrapping OptiMap in their own struct should mark their accessors `#[inline]` too. |
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
| Mid-pointer memory layout | Both RawTable impls use hashbrown's mid-pointer trick: single `ctrl` pointer between buckets (backward) and metadata (forward). Eliminates a struct field and address computation. `Byte7_128_Tomb` (then named `Hi128_Tomb`) beats hashbrown: lookup hit 4.07 vs 4.25 ns, insert 503 vs 603 µs, remove 763 vs 1079 µs. |
| AND-based group indexing | `h & mask` (1 instruction) vs `h >> shift` (2 instructions). Applied to IPO tombstone and 1-bit overflow designs. Requires tags from top hash bits (57+) to avoid correlation. |
| Splitsies-1bit (BitSeparate) | Implemented as `OverflowStrategy` + `Layout16` composition. 1 bit per group instead of 1 byte. See Splitsies-1bit section below for design rationale. |
| Load factor as type parameter | `LOAD_FACTOR_NUM`/`LOAD_FACTOR_DEN` constants on `GroupLayout` (default 7/8). Overflow-bit designs derive growth thresholds from the layout. Custom layouts can override to tune memory/speed trade-off. |
| Mid-pointer for 15-slot designs | Already implemented — UFM and Gaps share `overflow_table::RawTable<K,V,L>` which uses mid-pointer layout. Embedded overflow at byte 15 means exactly 2 memory regions, same as tombstone designs. |
| Borrow indirection in insert/entry | Investigated: already eliminated. Insert hot path uses `bucket.0 == key` directly. Cold fallback closures produce identical codegen via `#[inline(always)]` monomorphization. Added `find_by_hash_eq` wrapper for clarity, no perf impact. |
| Key-value separation (SoA layout) | `SoaRawTable<K,V,L>` + `SoaGenericMap` with separate key/value arrays. 7 matrix variants. Mid-pointer for keys, values after metadata+overflow. At 10K entries: competitive with Splitsies (32µs vs 31µs hit, 142µs vs 133µs insert for 256B values). Key-only probing may show more benefit at larger table sizes. |
| 32-slot (AVX2) and 64-slot (AVX-512) overflow-bit groups | `Group32<u32>` (1× 256-bit cmpeq+movemask) and `Group64<u64>` (1× 512-bit cmpeq_mask) added with compile-time `cfg(target_feature)` tier selection (AVX-512 → AVX2 → SSE2 → scalar Miri). New named layouts: `Splitsies32/64`, `Splitsies{32,64}_1bit`, `Byte1_1bit{32,64}` (formerly `Hi8_1bit{32,64}`), `Byte7_{128,255}_{1bit,8bit}And{32,64}` (formerly `Top{128,255}_*And{32,64}`). Required `META_STRIDE`/`META_ALIGN` parameterization on `GroupLayout` and `meta_stride` parameter on `OverflowStrategy::overflow_ptr`. Initial benches at 9.4K entries / 70% load: 32-slot variants match 16-slot on hit/insert and slightly improve miss (`Byte7_128_1bitAnd32`: 698 Mel/s miss vs 629 baseline, +11%); 64-slot underperforms 16-slot on hit/remove. No clear win at this size — wider groups may shine at higher load factors or for high-collision workloads, where home-group hit rate dominates. UFM/Gaps stay at 15-slot (embedded-overflow byte-15 trick is intrinsic to 16-byte metadata). |
| IPO tag/group-index collision fix + `ByteN_VVV` rename | IPO's default tombstone tag was `LowByte254` (bits 16-23). Once `num_groups > 2¹⁶`, the AND mask reaches into bits 16+, correlating tag bits with the group index and degrading SIMD discrimination — by 2²⁰ groups only ~16 distinct tags remain per group. Fix: switch IPO default to `Byte7_254` (bits 56-63, always decorrelated from AND mask). IPO64 keeps the bits-16-23 strategy under the new name `Byte2_254` (its shift indexing makes the middle of the hash safe). Strategy structs renamed to honest `ByteN_VVV` form; `HighByte128` and `TopByte128` consolidated into `Byte7_128`. |

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
| ~~Tree-surgery `append` (graft instead of drain+rebuild)~~ | ~~Medium~~ | **Done (May 2026).** See "FlatBTree::append tree-surgery graft" in Recently Completed. Outstanding: when `self.height < other.height` the graft falls back to drain — symmetric work, deferred until a workload demonstrates it's needed. |
| ~~Estimator bias correction~~ | ~~Low-Medium~~ | **Done (May 2026)** — but in a different shape than originally proposed. Investigation revealed two compounding biases: (1) the +0.10 estimator over-count from uniform-subtree-weighting; (2) a previously-unnoticed 2-3× cost asymmetry between `surgical_left` and `surgical_right` for the same workload (random-insert builds). Together they shift the optimal biased-r cutoff from ~0.6 to ~0.70. Retuned the cutoff in place of structural metadata tracking (no extra per-internal-node field). Win: 2.1× speedup at p045/1M random-build. Tracking actual rightmost-subtree count on insert/remove was deferred — the constant-factor asymmetry is the dominant signal, not the +0.10 count bias. |
| ~~Investigate std::BTreeMap split_off win at 1M-p050~~ | ~~Low~~ | **Closed (May 2026).** Hypothesis confirmed: std's structural split (Box pointer relinking) avoids deep-copy entirely. Our arena layout requires copy-on-split. Skip-`free_node` micro-optimization saves only 2–6%. Closing the remaining gap requires refcounted/shared arena or pointer-based node refs — both would tax every other operation. See [Closed Investigations](optimization/closed.md). |
| ~~Adaptive split direction (copy smaller side)~~ | ~~Medium~~ | **Done (May 2026).** See "FlatBTree::split_off bidirectional surgery" in Recently Completed. |
| ~~Remove rebalancing (steal/merge)~~ | ~~Medium~~ | **Already implemented** — `RawBTree::remove` triggers `rebalance_leaf` on underflow (`< LEAF_CAP/2`), which steals from a sibling or merges and cascades through `rebalance_internal`/`merge_internals`. Roadmap entry was stale. |
| ~~Child node prefetching~~ | ~~Low~~ | **Closed (May 2026).** No useful work between reading the child pointer and reading the next node — nothing to overlap. Speculative prefetch during the linear scan would mis-predict roughly half its prefetches for u64 keys (random access). HW prefetcher already covers the 4-cache-line node walk. |

### API Completeness

| Item | Difficulty | Notes |
|------|-----------|-------|
| ~~`range_mut()`~~ | ~~Low-Medium~~ | **Done (May 2026).** |
| ~~Arena `shrink_to_fit()`~~ | ~~Medium~~ | **Done (May 2026)** via drain + bulk_load rebuild. |

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
| FlatBTree split_off symmetric-pivot gap vs std | Arena memory layout requires copy-on-split (NodeIdx is arena-scoped); std's Box-based design relinks pointers without copy. At p050/1M dispatcher beats drain (5ms vs 13ms) and is within 1.1× of std's 4.7ms. Closing the gap would require refcounted shared arena or pointer-based node refs — taxes every other operation. |
