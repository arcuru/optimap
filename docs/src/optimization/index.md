# Optimization Overview

All five designs have been through extensive optimization passes. This section
documents what was tried, what worked, what failed, and why.

## Key Lessons

1. **The hasher was the bottleneck all along.** Switching from SipHash to foldhash
   gave a 3-7x speedup — by far the largest single improvement. Every other
   optimization is marginal by comparison.

2. **Fused home-group operations are the second biggest win.** Combining the
   duplicate check and empty-slot search into one SIMD load eliminated redundant
   work for >85% of inserts. -35 to -48% insert improvement.

3. **Removing prefetch beat adding prefetch.** With foldhash (10x faster than SipHash),
   hash computation is too short to hide memory latency. Removing manual prefetch
   improved misses by 27-30% at 1M, and small sizes improved across the board.

4. **Optimizations that fail on one design fail on all.** Bucket prefetch, cold
   continuation, and lazy overflow-bit computation were tested on UFM, Splitsies,
   and IPO64 — all failed for the same fundamental CPU reasons (register pressure,
   cache pollution, serial dependencies).

5. **The lookup hit gap is structural.** The ~1.1-1.25x hit overhead vs hashbrown
   comes from 15-slot group arithmetic, overflow-bit bookkeeping, and wider metadata
   reads. Multiple attempts to close it (prefetch, cold continuation, branchless
   comparison) all failed or traded hit improvement for miss regression.

6. **Single allocation is a benchmark trap.** It showed +40-108% insert regression at
   1M, which turned out to be OS page fault overhead (mmap zero-fill) rather than
   hash table behavior. The two-allocation strategy benefits from glibc arena caching.

## Optimization Timeline

| Phase | Focus | Key Result |
|-------|-------|------------|
| Phase 1 | SIMD fundamentals | Aligned loads, prefetch on overflow, cold paths |
| foldhash | Hasher switch | **3-7x speedup** |
| Phase 2 | Structural | Overflow-only prefetch, fused SIMD match, home-group fast path |
| Phase 3 | Fused operations | **-35 to -48% insert** via fused home-group insert |
| Struct opts | Size reduction | 64 → 56 bytes (derive mask, drop group_mask) |
| Splitsies | 16-slot design | Iteration 1.51x → 1.11x, miss 1.09x → 1.04x |
| IPO64 | AVX-512 | 14 → 3 SIMD ops per probe, miss -34% at 20M |

## Pages

- [UnorderedFlatMap optimization log](unordered_flat_map.md) — 26 attempts, from baseline to final state
- [Splitsies optimization log](splitsies.md) — 8 techniques tested
- [IPO64 optimization log](ipo64.md) — AVX-512 dispatch, load factor analysis
- [Closed investigations](closed.md) — Paths proven unproductive, with full rationale
