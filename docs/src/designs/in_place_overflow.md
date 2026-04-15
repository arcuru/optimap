# InPlaceOverflow (IPO)

Swiss-table-style design with 8-bit hash values. No overflow array — uses
EMPTY/TOMBSTONE sentinels like hashbrown, but with 254 hash values instead
of hashbrown's 128.

## Memory Layout

```
Metadata: [h0 h1 ... h15] × num_groups  (16 bytes per group, all 16 valid)
Buckets:  [slot0 slot1 ... slot15] × num_groups
(No overflow array)
```

## Key Properties

- **Group size**: 16 slots (all SIMD lanes valid)
- **Hash values**: 254 (range [2, 255], 0x00=EMPTY and 0x01=TOMBSTONE reserved)
- **Deletion**: Tombstone-based — writes TOMBSTONE (0x01), requires periodic rehash
- **Miss termination**: Scan until EMPTY found (same as hashbrown)
- **Bucket addressing**: `(gi << 4) | si` (power-of-2 shift)
- **SIMD mask**: Full 16-bit, no masking needed
- **Struct size**: 56 bytes

## Key Advantage over hashbrown

254 hash values vs hashbrown's 128 (7-bit h2). This gives ~2x fewer
false-positive SIMD matches, improving large-scale miss performance.

hashbrown's design uses the high bit of each control byte as a flag
(set = EMPTY or DELETED), limiting hash values to 7 bits (0x00-0x7F).
IPO uses the full byte range minus two sentinels.

## Key Disadvantage

hashbrown can detect empty slots with a single `movemask` instruction
(high bit extraction). IPO needs `cmpeq(data, 0) + movemask` — two
instructions. This matters on the hot path.

## Performance

IPO has the best lookup hit performance of all OptiMap designs (~1.01x
vs hashbrown at medium scale), but lacks the tombstone-free deletion
and O(1) miss termination of the overflow-bit family.
