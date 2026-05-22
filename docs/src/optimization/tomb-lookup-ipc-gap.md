# Investigation: Tomb's lookup_hit IPC gap vs hashbrown (May 2026)

## Status

**Diagnosis: complete.** **Implementation: shipped (partial fix).** The K/V-prefetch-drop change is now the default for the IPO (Tomb) family; the byte-offset probe representation is available behind an opt-in flag but doesn't deliver wins on its own.

This document is the single source of truth on the lookup_hit gap. The supporting [roadmap entry](../roadmap.md#hashbrown-wins-at-large-n-on-lookup_hit--investigation-in-progress-methodology-issue-uncovered) summarizes the experiments; this document captures the *mechanistic understanding* that resulted, and below the original diagnosis is the **post-implementation update** showing which parts of the doc's theory held up under measurement.

## What triggered the investigation

The sweep CSV's `lookup_hit` graph shows hashbrown as a relatively flat line and every OptiMap design (including `Tomb`, the hashbrown-equivalent default backend) as a wild sawtooth. The headline single-sample comparison was "hashbrown 40% faster than Tomb at N=5M, with CV 10% vs Tomb's 55% in the N=1M–10M band."

After four falsified hypotheses and one round of perf-counter measurement, the gap turned out to be much narrower than the sweep CSV suggested — but real, mechanistic, and rooted in instruction-level pipeline efficiency rather than memory.

## The actual diagnosis (one paragraph)

`Tomb`'s probe loop emits **11 instructions per probe step**; hashbrown's emits **3**. Total binary instruction count is the same for the same workload (~170M instructions), but Tomb's per-iter density consumes **3 load-port slots per probe step** (two software prefetches + the SIMD meta load) versus hashbrown's **1**, and duplicates the `gi → gi * 16` transform across two separate registers (one for the prefetch math, one for the SIMD load). The result is **IPC 0.88 for Tomb vs 1.23 for hashbrown** at the resize transient (N≈920K), and 4–5× slower recovery from a resize event because each cache-warming pass costs more cycles. The sweep harness's single-batch timing window then catches Tomb mid-recovery, producing the sawtooth shape on the graph.

## Evidence

### 1. Hardware counters (`perf stat`) at the spike point

At N=920K, 5 passes of 50K lookups, identical key set, same machine state, back-to-back runs (`examples/perf_isolate.rs`):

| metric | Tomb | hashbrown | UFM | IPO | Splitsies |
|---|---:|---:|---:|---:|---:|
| Cycles | 187.8M | 144.7M | 187.3M | 208.6M | 200.5M |
| Instructions | 135.7M | 159.0M | 180.6M | 188.1M | 180.7M |
| **IPC** | **0.72**¹ | **1.10**¹ | **0.96** | **0.90** | **0.90** |
| L1d misses | 5.5M | 5.8M | 4.9M | 5.2M | 5.9M |
| L1i misses | 2.2K | 3.7K | — | — | — |
| dTLB misses | 514K | 501K | 402K | 414K | 399K |
| iTLB misses | 279 | 349 | — | — | — |
| Branch mispredicts | 435K | 352K | 384K | 367K | 387K |

¹ Two independent runs measured Tomb at IPC 0.72 and 0.88 (run-to-run variance from compilation/inlining differences when the binary's match arms changed); both are well below hashbrown's 1.10–1.23.

**Read the numbers carefully:** `L1d`, `L1i`, `dTLB`, `iTLB` are all approximately equal across designs. Memory pressure is not the differentiator. **Pipeline efficiency is.** Tomb is *executing fewer instructions* per lookup but taking *more cycles* to do it.

### 2. Per-pass timing (multi-pass recovery shape)

At N=920K (just past Tomb's resize boundary), 5 consecutive 50K-lookup passes (`examples/spike_repro.rs`):

| design | P1 | P2 | P3 | P4 | P5 |
|---|---:|---:|---:|---:|---:|
| **Tomb** | **25 ns** | **20 ns** | **19 ns** | **16 ns** | **15 ns** |
| **hashbrown** | **24 ns** | **4.3 ns** | **4.1 ns** | **4.0 ns** | **4.4 ns** |

Both maps start at the same P1 ≈ 24 ns spike — that's the shared post-resize transient (allocator/TLB/scheduler cost from a freshly-allocated 33 MB region). Hashbrown drops out of it in one pass. Tomb decays slowly: even at P5 it's still elevated. Extending the spike_repro to N=1100K shows Tomb eventually reaches ~4 ns/op at fully-warmed steady state, matching hashbrown — so the "gap" is the recovery duration, not a steady-state floor.

### 3. The disassembly (the mechanistic root)

Side-by-side, both compiled to `target/release/examples/perf_isolate` with `RUSTFLAGS="-C debuginfo=2"`, extracted by symbol address via `objdump --demangle --disassemble -M intel`.

**Hashbrown's probe-step iteration (3 instructions):**

```asm
df8c: lea r15, [r15+r12+0x10]    ; pos += probe + 16  (byte offset)
df91: add r12, 0x10               ; probe += 16
df95: jmp <back to load>
```

Then the next probe's load + SIMD compare:

```asm
df30: and r15, rdi                ; r15 = pos & bucket_mask
df33: vmovdqu xmm2, [r10+r15]     ; UNALIGNED 16-byte load of ctrl bytes
df39: vpcmpeqb k0, xmm2, xmm1     ; SIMD compare against tag
df3f: kortestw k0, k0             ; any match?
```

Total per probe (assuming miss-and-continue, the common case at the spike): **7 instructions, 1 load-port slot.**

**Tomb's probe-step iteration (11 instructions):**

```asm
e3bc: lea  rdx, [rdx+r12+1]       ; next gi (triangular: gi += probe + 1)
e3c1: inc  r12                    ; probe += 1
e3c4: and  rdx, r11               ; gi &= mask (in groups, not bytes)
e3c7: mov  rbx, rdx               ; copy gi for the prefetch math
e3ca: shl  rbx, 0x4               ; rbx = gi * 16 (bytes)
e3ce: prefetcht0 [rdi+rbx]        ; prefetch next group's meta   ← load port
e3d2: not  rbx                    ; mid-pointer trick: ~(gi*16)
e3d5: shl  rbx, 0x4               ; rbx = ~(gi*16) * 16 (= -16*(gi*16) - 16)
e3d9: prefetcht0 [rdi+rbx]        ; prefetch next group's K/V    ← load port
e3dd: jmp <back to load>
```

Then the next probe's load + SIMD compare:

```asm
e374: mov r13, rdx                ; ANOTHER copy of gi
e377: shl r13, 0x4                ; r13 = gi * 16 (AGAIN, separate from rbx)
e37b: vmovdqa xmm1, [rdi+r13]     ; ALIGNED 16-byte load of meta  ← load port
e381: vpcmpeqb k0, xmm1, xmm0     ; SIMD compare against tag
e387: kmovd ebx, k0               ; extract mask bits
```

Total per probe: **16 instructions, 3 load-port slots.**

### 4. What each instruction is paying for

| Tomb instruction | Cost | Hashbrown's equivalent |
|---|---|---|
| `mov rbx, rdx; shl rbx, 0x4` | 2 µops to materialize `gi*16` for prefetch | (none — hashbrown's `r15` is already in bytes) |
| `prefetcht0 [rdi+rbx]` (meta) | 1 load-port µop | (none — hashbrown emits no software prefetch in the probe loop) |
| `not rbx; shl rbx, 0x4` | 2 µops for the K/V address mid-pointer math | (none — Tomb prefetches K/V too; hashbrown doesn't) |
| `prefetcht0 [rdi+rbx]` (K/V) | 1 load-port µop | (none) |
| `mov r13, rdx; shl r13, 0x4` | 2 µops to materialize `gi*16` AGAIN for the SIMD load (because `rbx` was clobbered by the K/V prefetch math) | (none — hashbrown reuses `r15` directly) |

That's **6 extra integer µops and 2 extra load-port slots per probe step**, all of which exist because Tomb stores the probe position in *groups* (`gi`) and recomputes the byte offset (`gi * 16`) on every iteration — and does it twice per iteration because the prefetch math destroys the intermediate value.

### 5. Why earlier hypotheses failed

| Experiment | Hypothesis | Result | Why it didn't move the needle |
|---|---|---|---|
| `hasher-ahash` | Hash distribution problem | Rejected | Tomb CV unchanged (54.4% → 54.4%); overflow-bit family slower by +21% to +103% mean. Hashing isn't in the probe step. |
| `no-probe-prefetch` | Prefetches hurt | Rejected, wrong-signed | Removing prefetches makes Tomb CV go from 54.6% to **101.2%** (worse). The prefetches DO help steady-state cache warming — they just also cost load-port pressure. |
| `tomb-branch-hints` (steady state) | `likely()` hints needed | Rejected | Branch predictor is already trained at steady state. `likely()` made LLVM emit 25% more instructions (layout bloat). |
| `tomb-branch-hints` (spike) | `likely()` hints help during cold predictor | Rejected | P1 actually got worse (30 ns vs 25 ns). Instruction count went up; mispredict count down only marginally. |
| `I-cache warmup` (2000-key warmup pass before timing) | I-cache cold from inserts | Rejected | Warmup didn't accelerate Tomb's recovery rate. The bottleneck is per-iter pipeline pressure, not code presence in I-cache. |

The pattern in all five negatives: they were trying to fix something *outside* the probe step. The actual cost is *inside* the probe step's µop scheduling.

### 6. The full causal chain that produces the sawtooth

1. Bench inserts `keys[prev_n..n]` one at a time. At N=917,504 (= 0.875 × 2^16 × 16) Tomb's max_load fires and the table doubles from 2^16 to 2^17 groups (~33 MB freshly allocated).
2. The rehash walks the old 910K entries and re-inserts each into the new table, first-touching most of the new region's pages.
3. The bench then runs a calibration pass + the timed lookup pass. Both maps see P1 ≈ 24 ns (shared transient — TLB-cold metadata pages, allocator state, etc.).
4. Hashbrown's lookup loop dispatches at ~3 instructions per probe step and fills issue width with high parallelism (IPC 1.23) → one 50K-lookup pass is enough to fully warm the cache and the predictor.
5. Tomb's lookup loop dispatches at ~11 instructions per probe step and saturates the load port (IPC 0.88) → each cache-warming pass costs more cycles, so recovery takes 5+ passes to reach steady state.
6. The sweep's `calibrate_repeats` gives slow operations **fewer repeats** (one heavy timed window instead of two light ones), so the sweep CSV samples Tomb's curve at random points along the recovery — wild CV. Hashbrown's pass-2-recovery means the sample is always in steady state → flat curve.

## What this means going forward

- **Tomb's "consistency gap" is not a memory or load-factor problem.** It's an instruction-scheduling problem in the probe loop, exposed only at resize transients by the sweep methodology.
- **At fully-warmed steady state**, Tomb reaches ~4 ns/op — competitive with hashbrown.
- **At pinned 1M-cap, 85% load, three-run median** (`bench_load_factor_1m`), Tomb is 14.6 ns/op vs hashbrown's 13.0 ns/op — only **+12%** behind, not the +40% the sweep snapshot suggested.
- **UFM is 30% faster than hashbrown** at that same pinned point (9.4 ns), and its IPC at the spike is 0.96 — a partial existence proof that an OptiMap design can do better than Tomb. UFM achieves this with `vpcmpeqb` against a memory operand (one of the redundant `shl` is elided) and an 8-channel overflow-bit terminator (fewer false continuations under load).
- **Methodology note for any future "consistency" claim**: single-sample CV across N from one sweep run is *highly noisy* run-to-run (hashbrown CV measured at 10%, 33%, 34% across three back-to-back foldhash runs). Multi-run aggregation is required, or use `bench_load_factor` for pinned-capacity work.

## Post-implementation update (May 2026)

The fix was tried, but the result didn't follow the doc's predictions exactly. Two changes were prepared independently and measured separately:

1. **Byte-offset probe representation** (gated by `tomb-byte-offset-probe`) — the doc-proposed `pos = byte_offset` representation in place of `gi = group_index`.
2. **Drop the K/V prefetch** (controlled by the absence of `tomb-prefetch-kv`) — keep only the meta prefetch in the probe loop.

### What the doc got right

- The mechanistic story about load-port pressure and LFB saturation is correct in shape. Dropping the K/V prefetch *does* relieve load-port pressure exactly as predicted.
- The K/V prefetch's blast radius is real: it's a software prefetch that consumes an LFB and competes with the critical-path SIMD meta load on subsequent iters.

### What the doc got wrong

- **The byte-offset probe representation alone is essentially neutral.** The duplicate `gi*16` materialization the doc identified *does* go away in the disassembly (the rewritten probe loop emits `lea r11, [r11+r15+0x10]; add r15, 0x10; and r11, byte_mask; prefetcht0` — exactly hashbrown-shaped). Instruction count drops by ~2M out of ~170M (~1.2%). But total cycles barely change: the bottleneck at the resize spike is memory latency on the SIMD meta load and the K/V access, not per-iter dispatch density. Saving a few µops per iter doesn't help when most iters are waiting on the L2/L3.
- **The K/V prefetch drop is the actual win, and it's independent of the representation change.** At N=900K (88% load), median-of-5 perf_isolate shows the drop alone moves P5 from 7.49 → 5.56 ns (-26%) with IPC 1.07 → 1.10. At N=1.67M (85% load on a bigger map): 9.19 → 8.51 ns (-7%) with IPC 0.89 → 0.95.
- **The win does not transfer to other table families.** The same K/V-drop experiment was applied to the overflow-bit table family (`raw/overflow_table.rs`, used by UFM/Splitsies/Gaps) and to IPO64 (`ipo64/raw/mod.rs`). Both regressed — Splitsies dropped 36% at N=900K, UFM dropped 3%, IPO64 was a wash. The overflow-bit families terminate probes via overflow bits rather than EMPTY tags, which appears to change probe-chain dynamics enough that the next-group K/V prefetch is genuinely useful (presumably because probe lengths and the home-group hit rate differ). IPO64's 64-slot groups mean probe chains are mostly length-0, so neither prefetch matters much.

### Shipped configuration

- `Tomb` (IPO, 16-slot, EMPTY-terminated): K/V prefetch dropped by default. `tomb-prefetch-kv` feature flag restores legacy behavior.
- `InPlaceOverflow`, `TombSoa`, `Byte7_254_TombMap` and the other matrix `Byte*_TombMap` variants all share `in_place_overflow::raw::RawTable`, so they all pick up the change automatically.
- UFM, Splitsies, Gaps, all `overflow_table`-backed designs: K/V prefetch dropped by default. `overflow-table-prefetch-kv` flag restores. (See methodology revision below — the initial perf_isolate measurement that suggested overflow_table regressed was contaminated by single-N noise; the proper sweep showed it's a clear win.)
- IPO64: K/V prefetch dropped by default. `ipo64-prefetch-kv` flag restores. The sweep evidence for IPO64 specifically was inconclusive (system noise during the all-families-dropped batches dominated), but the change was applied for consistency with the rest of the family.
- `tomb-byte-offset-probe`: available as an opt-in flag for future codegen experiments (e.g., paired with disabling K/V prefetch the saved `gi*16` materialization could compound), but not net-positive on its own.

### Methodology revision — why an early perf_isolate measurement was wrong

The first attempt to measure the K/V drop on overflow_table used `examples/perf_isolate` at a single N (N=900K) over 50K lookups, reporting median P5 over 5 runs. That measurement showed Splitsies *regressing* 36% with K/V dropped, and was used to conclude the change was Tomb-specific.

A subsequent **3-run sweep at N=100..10M** showed the opposite: Splitsies improves ~15% on mean, Gaps ~15%, UFM CV drops 21%→13%. The perf_isolate measurement was wrong because it was a single sample in a band where individual N values swing ±40% run-to-run (sweep evidence: same code, different runs, OvSplit per-N delta histogram spans -41% to +17% with no code change). A 50K-lookup batch is a low-signal sample even with median-of-5, because the median-of-5 collapses *within-batch* variance but not *between-N* variance.

Lesson: for any probe-loop change going forward, use the proper sweep methodology (3+ full sweep runs, median per (op, design, N), compare in N-band) and use hashbrown's delta as a sentinel (if hashbrown moves >5% between two sweep medians, the comparison is contaminated by system state and should be re-run).

### Measured deltas (median-of-3 sweep, lookup_hit, N=1M-10M)

Sentinel check (hashbrown delta between the two binaries that informed defaults): +2.8% — within noise.

| design | mean baseline | mean new default | Δmean |
|---|---:|---:|---:|
| Tomb (IPO Byte7_128) | 10.41 ns | 8.99 ns | **-13.6%** ¹ |
| TombWide (IPO Byte7_254) | 10.58 ns | 9.83 ns | **-7.1%** |
| TombSoa | 13.52 ns | 12.09 ns | **-10.5%** |
| OptiMap (auto policy = Tomb) | 10.89 ns | 10.23 ns | **-6.1%** |
| Splitsies (overflow_table, separate overflow) | 12.58 ns | 10.63 ns | **-15.5%** ² |
| Gaps (overflow_table, 15-slot embedded) | 11.17 ns | 9.49 ns | **-15.1%** ² |
| UFM (overflow_table, 15-slot embedded) | 7.93 ns | 7.80 ns | -1.6% mean, but **CV 21% → 13%** ² |
| IPO64 | (inconclusive, contaminated batches) | | |

¹ The headline -13.6% on Tomb is real but partially inflated by one anomalous slow legacy run. Honest reading: -7 to -10% on mean, *plus* a noticeable reduction in run-to-run variance — Tomb on new-default had mean stable at 9.35-9.44 ns across three runs while legacy varied 9.36-12.69 ns.

² For the overflow_table family, this delta comes from the `ovdropkv` 3-run sweep (Tomb K/V + overflow K/V dropped) compared against the original `legacy` (everything kept). Hashbrown sentinel delta was -2.8% across that pair.

### Clean-environment paired sweeps (final measurements)

The earlier sweeps were run on a GUI-active machine with the CPU governor at
`powersave`. Run-to-run variance was ~±20% even with alternating paired runs,
so several of the headline numbers in this doc above are noise-inflated.

A subsequent **clean-environment** run was set up:

- GUI exited (multi-user.target)
- CPU governor `performance` (sub-second frequency steady state at ~5GHz)
- `taskset -c 8,20` pinning to physical core 8 (both SMT threads claimed)
- 3 paired experiments (Tomb-only / overflow-only / ipo64-only), each with
  6 alternating runs (KDKDKD), median of 3 per binary

The hashbrown sentinel delta was <2% across all 3 pairs, confirming the
environment was genuinely quiet. The real per-family effects came out
**much smaller than the dirty-environment measurements suggested**:

| family | dirty sweep claimed | clean sweep measured |
|---|---:|---:|
| Tomb K/V drop | -13.6% mean | **-2.9% mean** |
| TombWide K/V drop | -7.1% mean | -2.9% mean |
| TombSoa K/V drop | -10.5% mean | -0.6% mean (within noise) |
| Overflow_table K/V drop (UFM) | (mean +1.5%, CV -8pp) | +3.6% mean, CV +2.5pp **(neutral-negative)** |
| Overflow_table K/V drop (Splitsies) | -15.5% mean | +0.5% mean (within noise) |
| Overflow_table K/V drop (Gaps) | -15.1% mean | -1.0% mean (within noise) |
| IPO64 K/V drop (Tomb64) | inconclusive | -2.8% mean |

**Decision based on clean numbers:** drop K/V prefetch as default for Tomb
family and IPO64 (small but consistent ~-3% wins); revert the overflow_table
change (no win shown, slight regression on UFM).

### What the per-N profile says about *why* K/V drop wins on Tomb

The clean per-N data exposed a clean pattern hiding inside the mean:

```text
     N       kept   dropped    Δ%
  937803    13.25    14.30   +7.9%  ← post-resize, cold cache, LF~44%
  994915     7.32     8.00   +9.3%  ← still recovering
 1024762     6.33     6.68   +5.5%
 1087169     5.75     5.68   -1.2%  ← cache warmed up
 1223617     5.57     5.43   -2.5%  ← steady state, low load
 1796917     5.63     5.42   -3.7%  ← pre-resize, high load (88%)
 1850824    26.66    26.86   +0.8%  ← RESIZE → repeat cold cycle
```

The doc's original diagnosis ("Tomb's IPC gap is caused by LFB saturation
during the cold-cache transient — drop K/V prefetch to relieve it") was
**directionally wrong**. K/V prefetch is *useful* during the cold-cache
transient (covers the cold-K-miss latency) and only marginally harmful in
steady state (consumes one load-port slot redundantly). The net win from
dropping comes from steady-state N values outnumbering transient N values in
the sweep — but **the sawtooth peaks themselves got slightly worse** with
the K/V drop (post-resize Ns are +5-9% slower when K/V is dropped).

An adaptive policy (keep K/V when load_factor < 50%, drop otherwise) would
capture both wins, but the headline mean delta is small enough (~3%) that
the implementation complexity isn't worth it. Logged as a possible future
investigation; not blocking the current ship.

### Why the doc's "Phase 2 design" target metrics aren't met yet

The doc's target table predicted IPC ≥ 1.10 at N=920K with the byte-offset probe. In measurement, IPC at the spike moves modestly:

| metric | Tomb baseline | Tomb (new default) | doc target |
|---|---:|---:|---:|
| IPC at N=920K (median-5 runs) | 0.66 | 0.79 | ≥1.10 |
| IPC at N=900K (median-5 runs) | 0.64 | 0.67 | — |

The new default IPC is up but not at hashbrown levels. The remaining gap is presumably from differences in branch prediction, micro-op cache behavior, and the K-access cold-miss penalty that the K/V prefetch was masking. Closing it further would require either:

- Restructuring the K-access path so the K cache line is requested earlier in the pipeline (e.g., issue the K load speculatively before the tag-match decision).
- Restructuring the meta-load path so the SIMD load isn't on the critical path (hashbrown effectively does this by overlapping group N's compare with group N+1's address computation; the byte-offset representation enables this but LLVM doesn't actually exploit it).

The sweep-CV target (54% → ≤30%) was NOT met by this fix. The 3-run median sweep at N=1M-10M shows Tomb's CV stayed around 53% (vs 51.8% baseline) and TombWide's CV went *up* to 57.6%. The mean improvement is real (-7 to -14% on Tomb family, -15% on overflow_table family) but the wild swing pattern across N persists — suggesting it's a sweep-methodology artifact (single-batch timing windows catching mid-recovery operations at random Ns) rather than something the probe-loop fix can address. Further CV work belongs in the sweep harness, not the table implementations.

## The original fix proposal (Phase 2, preserved for reference)

### Design: byte-offset probe in `in_place_overflow::raw::RawTable`

Hashbrown stores the probe position as a *byte offset* into the ctrl array (`pos`), starts at `h1(hash) & bucket_mask`, and advances by `Group::WIDTH` (16) per probe step. Address computation is then a single `lea` or implicit base+index addressing mode.

Tomb stores the probe position as a *group index* (`gi`), starts at `h & (num_groups - 1)`, advances triangularly by `gi += probe`, and must recompute `gi * 16` (twice, in separate registers) on every probe step.

Move Tomb to hashbrown's byte-offset representation:

- **Field change:** `mask: usize` (= `num_groups - 1`) → `byte_mask: usize` (= `num_groups * GROUP_SIZE - 1`)
- **`group_index()` return type semantics:** returns a byte offset (always 16-aligned because `(h & byte_mask) >> 4 << 4`), not a group count.
- **Probe step:** `probe += GROUP_SIZE; pos = (pos + probe) & byte_mask;` — same as hashbrown.
- **Bucket address:** `ctrl + pos` for meta; mid-pointer math becomes `ctrl - (bucket_index + 1) * BUCKET_SIZE` where `bucket_index = pos | si` (no multiplication needed; `pos` is already pre-shifted).
- **Single prefetch:** with one address kept in registers, the two `prefetcht0` calls can collapse to one (or zero — see below).

### Expected impact

| metric | current | target |
|---|---:|---:|
| Per-probe instruction count | 11 | 3–5 |
| Per-probe load-port µops | 3 | 1–2 |
| IPC at N=920K | 0.72–0.88 | ≥1.10 |
| Multi-pass recovery at N=920K | 5+ passes | ≤2 passes |
| Sweep CV (lookup_hit, N=1M–10M, median of 3 runs) | 54% | ≤30% |
| Steady-state lookup_hit at pinned 1M-cap, 85% load | 15 ns | ≤14 ns |

### Implementation plan

1. **Gate the change behind a cargo feature `tomb-byte-offset-probe`.** This lets the A/B comparison run cleanly via `examples/perf_isolate.rs` + `examples/spike_repro.rs` without disturbing default behavior.

2. **Files to edit:**
    - `src/in_place_overflow/raw/mod.rs` — `mask` field, `group_index()`, `meta_ptr()`, `bucket_index()`, and every probe loop:
        - `find_by_hash` (lines ~272–305)
        - `find_or_locate` (around line 425)
        - `insert_no_check` (around line 372)
        - `insert_after_resize` (around line 342)
        - `find_with_hash` (around line 264)
        - `find_bucket` and `find_or_locate_overflow` if present
    - `src/in_place_overflow/raw/storage.rs` — `bucket_index` consumers may need a representation adjustment if they assume `gi * 16 + si`.
    - Nothing in `src/in_place_overflow/raw/group.rs` needs to change — the SIMD intrinsics are unaffected.

3. **Prefetch decision:** start by keeping a *single* prefetch (for the next meta load) and dropping the K/V prefetch. The K/V access only happens on tag match (~93% of lookups at home group); prefetching K/V for a slot we may never read is mostly wasted load-port slot. Measure; if dropping the K/V prefetch costs >5% on the steady-state hit at >1M, add it back conditionally.

4. **Validate against:**
    - `cargo test --lib` (515 lib tests should pass)
    - `cargo test --tests` (alloc_stress + integration suite — ahash-specific failure is expected/documented, not a regression)
    - `cargo test --release --features tomb-byte-offset-probe` for the new flag
    - `examples/perf_isolate.rs tomb 920000 50000 5` — confirm IPC ≥ 1.10
    - `examples/spike_repro.rs` — confirm Tomb recovers in ≤ 2 passes at N=920K
    - 3-run sweep with the new flag, `scripts/cv-compare.py` — confirm CV ≤ 30% in 1M–10M band
    - `cargo bench --bench load_factor` — confirm no regression at pinned 1M-cap

5. **Cluster effect:** the `in_place_overflow` engine is shared by `Tomb` (Byte7_128), `TombWide` (Byte7_254 / IPO), and `TombSoa` via the trait parameterization. All three pick up the change automatically. The `overflow_table` engine (UFM, Splitsies, Gaps and friends) is structurally similar but separate — applying the same refactor there is a follow-up if numbers are good on the tomb family.

6. **If numbers are clean, ship it** as the new default (remove the feature gate). If not, document why and fall back to one of:
    - A `Policy::auto()` rework that pins lookup-hit-heavy workloads at large N to UFM instead of Tomb.
    - Accept the gap as structural and document it.

### Risk

Medium. The refactor is deep in the hot path of three production designs. Test coverage is decent (515 lib tests + allocator stress + 30 integration tests in the in_place_overflow module). A bug in mid-pointer math would corrupt bucket reads silently — must be validated against the existing differential proptest harness (`tests/proptest_hashmap.rs`).

## Reproducing the experiment

### Tools (all on `main`, working)

- **`examples/spike_repro.rs`** — 5-pass per-N timing repro. `cargo run --release --example spike_repro`.
- **`examples/perf_isolate.rs`** — minimal binary for `perf stat ...` wrapping. Each design's `do_lookups<M>` is a separate symbol (function-pointer routed through `black_box`) suitable for `perf annotate` / `objdump --disassemble`. Args: `{tomb|hb|ufm|ipo|split} <N> <lookups> <passes>`.
- **`scripts/cv-compare.py`** — sweep CSV CV/delta analyzer.

### perf stat at the spike

```bash
RUSTFLAGS="-C debuginfo=2 -C force-frame-pointers=yes" \
  direnv exec . cargo build --release --example perf_isolate

nix shell nixpkgs#linuxKernel.packages.linux_6_18.perf -c perf stat \
  -e cycles,instructions,branch-misses,L1-dcache-load-misses,dTLB-load-misses \
  -- ./target/release/examples/perf_isolate tomb 920000 50000 5
```

### Disassembling a specific do_lookups<M> variant

```bash
BINUTILS=$(ls -d /nix/store/*-binutils-*/bin | head -1)

# Find symbol addresses:
$BINUTILS/nm --demangle target/release/examples/perf_isolate | grep do_lookups

# Dump a specific one (addresses below shift between builds; re-run nm each time):
$BINUTILS/objdump --demangle --disassemble \
  --start-address=0x000e300 --stop-address=0x000e430 -M intel \
  target/release/examples/perf_isolate
```

### perf record + report

```bash
mkdir -p /tmp/perf-data
nix shell nixpkgs#linuxKernel.packages.linux_6_18.perf -c \
  perf record -F 5000 --call-graph=no -o /tmp/perf-data/perf-tomb.data -- \
  ./target/release/examples/perf_isolate tomb 920000 100000 200

nix shell nixpkgs#linuxKernel.packages.linux_6_18.perf -c \
  perf report -i /tmp/perf-data/perf-tomb.data --stdio --no-children --sort symbol
```

### Feature flags (off by default)

The following are wired up and tested but proved NOT to close the gap:
- `--features hasher-ahash` — swaps default `BuildHasher` to ahash
- `--features no-probe-prefetch` — neutralizes every `Group::prefetch_read` impl body
- `--features tomb-branch-hints` — wraps `find_by_hash`'s hit + match_empty branches in `std::hint::likely()`

Future flag for the fix: `--features tomb-byte-offset-probe`.

## References

- TODO-0210344b — implementation tracking todo
- Roadmap entry: `docs/src/roadmap.md` → "Hashbrown wins at large N on lookup_hit"
- Closed investigations: `docs/src/optimization/closed.md`

## Investigation commits (local `main`, not pushed)

```
1a97cb4 bench(spike): add ufm/ipo/split kinds + force do_lookups as separate symbol
d6e2fc6 docs(roadmap): record perf-stat IPC diagnosis of the lookup_hit resize transient
3d1d7a7 bench(perf): perf_isolate binary for cycle/cache-miss counters
33ca426 bench(spike): I-cache warmup A/B (warmup doesn't accelerate Tomb's recovery)
5e7ae34 bench(spike): direct N=920K post-resize warmup repro
08ba22b perf(investigation): tomb-branch-hints flag; record UFM>hashbrown finding
194a074 bench(load_factor): include Tomb (Byte7_128_TombMap) in 1M-capacity row
0e7f964 perf(investigation): hasher-ahash + no-probe-prefetch flags; rule out hasher and prefetch
9afc441 tools(bench): cv-compare.py for sweep CSV analysis
```
