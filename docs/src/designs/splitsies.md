# Splitsies

16-slot groups with a separate contiguous overflow array. Combines the
tombstone-free overflow-bit design with power-of-2 bucket addressing.

## Memory Layout

```text
Metadata: [h0 h1 ... h15] × num_groups  (16 bytes per group, all 16 valid)
Overflow: [ovf0 ovf1 ... ovf_{n-1}]     (1 byte per group, separate contiguous array)
Buckets:  [slot0 slot1 ... slot15] × num_groups
```

## Key Properties

- **Group size**: 16 slots (all SIMD lanes valid — no masking needed)
- **Hash values**: 255 (range [1, 255], only 0x00=EMPTY reserved)
- **Deletion**: Tombstone-free — same as UFM, uses overflow bits
- **Miss termination**: O(1) via overflow bit (separate array, prefetched)
- **Bucket addressing**: `(gi << 4) | si` (power-of-2 shift, not multiply)
- **SIMD mask**: Full 16-bit, no masking needed
- **Struct size**: 56 bytes (overflow pointer derived from metadata)

## Advantages over UFM

- Full 16-bit SIMD mask (no `& 0x7FFF`)
- Power-of-2 bucket addressing (shift instead of multiply-by-15)
- All 16 SIMD lanes are valid hash slots

## Advantages over hashbrown

- Tombstone-free deletion: no performance degradation under churn
- O(1) miss termination via overflow bits
- 255 hash values vs 128 (fewer false-positive SIMD matches)
- Nearly flat miss performance across load factors (143-158µs from 45% to 85%)

## Performance vs hashbrown (70% load)

### Wins
- **Remove**: 0.64-0.84x (tombstone-free deletion)
- **Churn**: 0.62-0.86x (biggest advantage)
- **Insert**: 0.93-0.97x
- **from_iter**: 0.62-0.70x
- **Sequential miss**: 0.74x
- **Miss at 85% load**: 0.28x (3.6x faster)

### Losses
- **Lookup hit**: 1.11-1.23x (structural per-probe overhead)
- **Entry API**: 1.66x (hit overhead + enum construction)
- **Counting/aggregation**: 1.28-1.35x
- **Large value insert (128B+)**: 1.48-1.65x

## Design Note: Overflow Array

The overflow array is contiguous and positioned immediately after the metadata array.
The overflow pointer is derived from `metadata + num_groups * 16` rather than stored
separately, saving 8 bytes in the struct. Since overflow access is on the cold path
(only after a home-group miss), the arithmetic cost is negligible.
