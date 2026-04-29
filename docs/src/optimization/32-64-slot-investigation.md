# Investigation: 32/64-slot and Embedded Variants (April 2026)

## Background
The previous optimization pass introduced `Group32` (AVX2) and `Group64` (AVX-512) SIMD metadata groups, allowing maps to probe 32 or 64 slots per loop iteration instead of 16. It also introduced embedded-overflow combinations (where the overflow byte is stored at the end of the SIMD metadata rather than in a separate flat array).

Initial `--quick` benchmarks (at a single medium size) showed promising signs, particularly for 32-slot and 64-slot embedded variants (`Gaps32`, `Gaps64`, `Ufm32`, `Ufm64`). This investigation ran a full `100 - 2M` structural sweep and Matrix Criterion runs (at 9.4K and 75K entries) to definitively evaluate these architectures against the highly optimized 16-slot baselines.

## Headline Finding
**Wider is not purely better. 32-slot variants occasionally win on insert/remove (up to +15% at large scales), but carry a 6-11% penalty on `lookup_miss`. 64-slot variants remain strictly worse (+40-60% miss, +10-20% hit/insert) than 16-slot designs up to 2 million entries.**

## Detailed Metrics

### 1. The Cache-Resident Regime (N < 30,000)
When the hash map fits entirely or mostly in L1/L2 cache, 16-slot designs reign supreme:

- **insert**: Tied. 16-slot `UFM` (6.2 ns), 32-slot `Hi8_1bit32` (6.3 ns).
- **lookup_hit**: Tied. 16-slot `Top128_1bitAnd` (2.1 ns), 32-slot `Top128_1bitAnd32` (2.1 ns).
- **lookup_miss**: 16-slot wins. `hashbrown` (1.2 ns), `Lo128_8bit` (1.4 ns) vs best 32-slot `Splitsies32` (1.5 ns).
- **remove**: 16-slot wins slightly. `Top255_EmbP2And` (12.6 ns) vs `Top255_1bitAnd32` (11.3 ns) vs `Hi128_Tomb` (11.2 ns).

At this size, the extra SIMD width (AVX2/AVX-512) does not pay for itself. Over 98% of probes resolve in the home group anyway, so checking 32/64 slots simultaneously provides zero reduction in probe steps, while costing more instruction bytes and register pressure.

### 2. The DRAM Scale Regime (N = 1,000,000 to 2,000,000+ entries)
As the table grows and load factors increase, probe chains lengthen. Here, probing more slots per step *should* help. The continuous sweep data from 100K to 2M reveals:

#### Insert
**32-slot clearly wins.** The best 32-slot designs (`Ufm32`) consistently edge out the best 16-slot designs by 5-15% after 500K entries.
- At 1.7M N: `Ufm32` (21.0 ns) vs `UFM` (24.2 ns) → **15% faster**

#### Lookup Hit
**Statistical Tie.** The gap is within 3-5% margin of error across the entire N = 100 to 2,000,000 sweep.
- At 1.0M N: `Top128_1bitAnd32` (6.4 ns) vs `Top128_EmbP2And` (6.3 ns)
- `Ufm32` perfectly overlaps `UFM`.

#### Lookup Miss
**32-slot is structurally slower (-6% to -11%).**
- The penalty for wider groups appears permanently structural on the miss path. 
- At 1.0M N: `Splitsies32` (3.2 ns) vs `Hi8_EmbP2` / `Gaps` (3.0 ns)
- Why? 32-slot designs check twice as many unoccupied slots per probe step. This doubles the false-positive rate from the tag hash collision byte. Finding a false-positive match requires reading the actual key from memory (a cache miss). The 255-tag variants fare better than 128-tag variants because their base false-positive rate is half as large (1/255 vs 1/127).

#### Remove
**32-slot and 16-slot trade blows** depending on the exact load factor (sawtooth alignment).
- `Ufm32` vs `UFM` shows the two lines crossing repeatedly between 30 and 45 ns/op at large scale. Overall, there is no clear victor, though Tombstone designs (`Hi128_Tomb`, 24.5 ns) significantly outperform all overflow-bit designs on this workload.

### 3. The 64-Slot (AVX-512) Reality
At N ≤ 2,000,000, `Group64` is simply too wide for this architecture.
- **Hit/Insert**: 10-20% slower than 16-slot.
- **Miss**: 40-60% slower, heavily punished by false-positive tag collisions when checking 64 slots per home probe.
- High constant instruction overhead. AVX-512 `cmpeq + mask` is powerful, but hashbrown's tight SSE2 inner loop is strictly faster for the 98% of queries that don't overflow.

### The Embedded-Overflow Trade-off
The sweep included full testing of designs incorporating `UfmEmbeddedOverflow` (where the overflow byte is appended to the 15/31/63 metadata bytes).
- At 16-slot: Embedded overflow provides slightly better `lookup_miss` characteristics than separate overflow arrays, but slightly trails on `insert`.
- At 32-slot: `Ufm32` (embedded, compact stride) emerged as the single fastest `insert` engine among all designs tested.

## Conclusion and Actions

1. **Keep 16-slot designs as the default.** The slight (5-15%) insert advantage of 32-slot designs at multi-million scales is not worth a permanent ~10% lookup_miss penalty across all sizes.
2. **Tag choice is critical for wider groups.** The sweep confirmed the hypothesis: `Top255` variants definitively outperform `Top128` variants on wider (32/64) groups. When probing 64 slots, a 1/127 false-match rate triggers a spurious cache miss roughly 50% of the time, destroying performance.
3. **No immediate architectural changes required.** `Hi128_Tomb` remains the best general-purpose map, and 16-slot `Splitsies` / `UFM` remain the best tombstone-free variants.

*Plots supporting this investigation have been generated and will be preserved in `bench-results/32-64-*.png`.*
