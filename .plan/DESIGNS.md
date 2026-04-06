# Hash Map Designs

Three hash map implementations with different architectural trade-offs.
All use foldhash, SIMD-accelerated probing, and quadratic probe sequences.

## UnorderedFlatMap (UFM) — Boost-style, 15-slot groups

```
Metadata: [h0 h1 ... h14 OVF] × num_groups  (16 bytes per group, 15 slots + 1 overflow byte)
Buckets:  [slot0 slot1 ... slot14] × num_groups
```

- **Group size**: 15 slots (1 SIMD lane wasted on overflow byte)
- **Hash values**: 255 (range [1, 255], only 0x00=EMPTY reserved)
- **Deletion**: Tombstone-free — sets slot to EMPTY, uses overflow bits for correctness
- **Miss termination**: O(1) via overflow bit — if bit not set, key was never displaced
- **Bucket addressing**: `gi * 15 + si` (multiply by non-power-of-2)
- **SIMD mask**: `& 0x7FFF` needed to discard overflow byte in position 15
- **Struct size**: 56 bytes

## Splitsies — 16-slot groups with separate overflow array

```
Metadata: [h0 h1 ... h15] × num_groups  (16 bytes per group, all 16 valid)
Overflow: [ovf0 ovf1 ... ovf_{n-1}]     (1 byte per group, separate contiguous array)
Buckets:  [slot0 slot1 ... slot15] × num_groups
```

- **Group size**: 16 slots (all SIMD lanes valid)
- **Hash values**: 255 (range [1, 255], only 0x00=EMPTY reserved)
- **Deletion**: Tombstone-free — same as UFM, uses overflow bits
- **Miss termination**: O(1) via overflow bit (separate array, prefetched)
- **Bucket addressing**: `(gi << 4) | si` (power-of-2 shift)
- **SIMD mask**: Full 16-bit, no masking needed
- **Struct size**: 56 bytes (overflow pointer derived from metadata)

## InPlaceOverflow (IPO) — Swiss-table-style with 8-bit hash

```
Metadata: [h0 h1 ... h15] × num_groups  (16 bytes per group, all 16 valid)
Buckets:  [slot0 slot1 ... slot15] × num_groups
(No overflow array)
```

- **Group size**: 16 slots (all SIMD lanes valid)
- **Hash values**: 254 (range [2, 255], 0x00=EMPTY and 0x01=TOMBSTONE reserved)
- **Deletion**: Tombstone-based — writes TOMBSTONE (0x01), requires periodic rehash
- **Miss termination**: Scan until EMPTY found (same as hashbrown)
- **Bucket addressing**: `(gi << 4) | si` (power-of-2 shift)
- **SIMD mask**: Full 16-bit, no masking needed
- **Struct size**: 56 bytes
- **Key advantage over hashbrown**: 254 hash values vs hashbrown's 128 (7-bit h2),
  giving ~2x fewer false-positive SIMD matches and faster large-scale misses

## Comparison: hashbrown (Swiss table)

```
Control: [c0 c1 ... c15] × num_groups  (1 byte per slot, 16 per group)
Buckets: [slot0 slot1 ... slot15] × num_groups
```

- **Group size**: 16 slots
- **Hash values**: 128 (7-bit h2, range [0x00-0x7F], high bit = special flag)
- **Deletion**: Tombstone-based (DELETED = 0x80)
- **Miss termination**: Scan until EMPTY (0xFF) found
- **Empty detection**: `movemask(data)` — high bit extraction, ONE instruction
  (vs IPO's `cmpeq(data, 0) + movemask` — TWO instructions)
- **Struct size**: 40 bytes

## Design Trade-off Summary

| Property | UFM | Splitsies | IPO | hashbrown |
|----------|:---:|:---------:|:---:|:---------:|
| Hash values | 255 | 255 | **254** | 128 |
| Tombstone-free | ✓ | ✓ | ✗ | ✗ |
| O(1) miss termination | ✓ | ✓ | ✗ | ✗ |
| Power-of-2 addressing | ✗ | ✓ | ✓ | ✓ |
| Full 16-bit SIMD mask | ✗ | ✓ | ✓ | ✓ |
| No overflow prefetch | ✗ | ✗ | ✓ | ✓ |
| Single-instruction empty check | ✗ | ✗ | ✗ | ✓ |
| Miss at high load (85%) | **0.52x** | **0.28x** | ~1x | 1x |
| Churn performance | **0.62x** | **0.62x** | ~1x | 1x |
| Insert (fresh) | 0.87x | 0.95x | **0.85x** | 1x |
| Lookup hit (medium) | 1.20x | 1.11x | **1.01x** | 1x |
