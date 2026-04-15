# Gaps

Boost-style 15-slot groups with power-of-2 bucket addressing. Identical to
UnorderedFlatMap except for a gap (unused 16th slot) in the bucket array.

## Memory Layout

```text
Metadata: [h0 h1 ... h14 OVF] × num_groups  (same as UFM)
Buckets:  [slot0 ... slot14 GAP] × num_groups  (16 slots allocated, 15 used)
```

## Key Properties

Same as UnorderedFlatMap (15-slot groups, overflow byte, tombstone-free deletion)
with one change:

- **Bucket addressing**: `(gi << 4) | si` instead of `gi * 15 + si`

This eliminates the multiply-by-15 on every operation at the cost of ~6.25%
wasted memory in the bucket array (1/16 slots unused).

## Trade-off

The same SIMD operations as UFM (including `& 0x7FFF` mask for 15-slot groups),
but with the simpler addressing arithmetic of the 16-slot designs. Best suited
for workloads where iteration performance matters — the power-of-2 stride
is more prefetcher-friendly.
