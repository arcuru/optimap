# Design Overview

OptiMap provides five hash map implementations, all sharing these properties:

- SIMD-accelerated metadata probing (SSE2/NEON/scalar fallback)
- Quadratic probe sequence over power-of-two group counts
- foldhash as default hasher (avalanching, fast)
- 70% default load factor
- Generic `Map` trait for uniform benchmarking and generic code

The designs split into two families:

## Overflow-Bit Family (tombstone-free)

These store an overflow byte per group that tracks which hash classes were displaced.
Misses terminate in O(1) — if the overflow bit for your hash class isn't set,
no element with that hash was ever displaced from this group.

Deletion is tombstone-free: set the slot to EMPTY and (if displaced) decrement
the anti-drift counter. No performance degradation under churn.

| Design | Group Size | Bucket Addressing | Trade-off |
|--------|-----------|-------------------|-----------|
| [UnorderedFlatMap](unordered_flat_map.md) | 15 slots | `gi * 15 + si` (multiply) | Original design, proven |
| [Splitsies](splitsies.md) | 16 slots | `(gi << 4) \| si` (shift) | Faster arithmetic, separate overflow array |
| [Gaps](gaps.md) | 15 slots | `(gi << 4) \| si` (shift) | UFM + power-of-2 buckets (wastes 1/16 slots) |

## Tombstone Family (Swiss-table style)

These use EMPTY/TOMBSTONE sentinels like hashbrown. Misses scan until EMPTY.
Key advantage over hashbrown: 254 hash values (8-bit, reserving only 0x00 and 0x01)
vs hashbrown's 128 (7-bit h2), giving ~2x fewer false-positive SIMD matches.

| Design | Group Size | Key Idea | Trade-off |
|--------|-----------|----------|-----------|
| [InPlaceOverflow](in_place_overflow.md) | 16 slots | Swiss-table + 8-bit hash | Best lookup hit, needs periodic rehash |
| [IPO64](ipo64.md) | 64 slots | Cache-line groups, AVX-512 | Flat degradation at extreme load, slower per-probe |

## Performance Comparison

| Property | UFM | Splitsies | IPO | hashbrown |
|----------|:---:|:---------:|:---:|:---------:|
| Hash values | 255 | 255 | **254** | 128 |
| Tombstone-free | yes | yes | no | no |
| O(1) miss termination | yes | yes | no | no |
| Power-of-2 addressing | no | yes | yes | yes |
| Full 16-bit SIMD mask | no | yes | yes | yes |
| Single-instruction empty check | no | no | no | yes |
| Miss at high load (85%) | **0.52x** | **0.28x** | ~1x | 1x |
| Churn performance | **0.62x** | **0.62x** | ~1x | 1x |
| Insert (fresh) | 0.87x | 0.95x | **0.85x** | 1x |
| Lookup hit (medium) | 1.20x | 1.11x | **1.01x** | 1x |

Ratios are vs hashbrown; <1.0 = faster than hashbrown.
