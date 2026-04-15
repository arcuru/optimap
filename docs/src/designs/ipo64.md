# IPO64

64-slot cache-line-aligned groups with AVX-512 acceleration. A specialty
design for applications that need predictable high-load performance.

## Memory Layout

```
Metadata: [h0 h1 ... h63] × num_groups  (64 bytes per group, cache-line aligned)
Buckets:  [slot0 slot1 ... slot63] × num_groups
```

## Key Properties

- **Group size**: 64 slots (one full cache line of metadata)
- **Hash values**: 254 (same as IPO16)
- **Deletion**: Tombstone-based
- **SIMD**: AVX-512 (single 64-byte load), AVX2 fallback, SSE2 fallback (4×16-byte loads)
- **Runtime dispatch**: Feature detection at `find_by_hash` entry point, not per-iteration

## SIMD Dispatch

AVX-512 reduces SIMD operations from 14 (SSE2) to 3 per probe step. Dispatch
is done once at the `find_by_hash` entry point — the entire probe loop runs
with the best available SIMD tier.

Per-call dispatch (checked inside each Group method) was tested and rejected:
the atomic load overhead of `is_x86_feature_detected!` inside the probe loop
caused a +3-9% hit regression.

## Performance Characteristics

IPO64's defining property is **flat miss degradation under load**:

| Load | IPO16 miss | IPO64 miss | hashbrown miss |
|-----:|-----------:|-----------:|---------------:|
| 50%  | 1.95 ns    | 3.40 ns    | 1.56 ns        |
| 90%  | 2.95 ns    | 4.56 ns    | 2.19 ns        |
| 95%  | 3.99 ns    | 5.26 ns    | 3.63 ns        |
| 97%  | 4.38 ns    | 4.68 ns    | 4.17 ns        |
| 99%  | 5.33 ns    | **5.13 ns**| 4.79 ns        |

Miss degradation 50% to 99%: IPO16 2.73x, **IPO64 1.51x**, hashbrown 3.07x.

The crossover with IPO16 occurs at 99% load — too extreme for general use.

## Why It's Slower at Typical Load

At 70% load, 64-slot groups resolve >99% of probes in a single step — same
as 16-slot groups. So IPO64 does more SIMD work per probe step (even with
AVX-512, 3 ops vs IPO16's 2 ops) for the same number of probe steps. The
cache-line advantage only helps when multiple probe steps are needed, which
is rare below 95% load.

## Conclusion

IPO64 is a research/specialty design. For general use, IPO16 or Splitsies
provide better performance. IPO64's value is in demonstrating the
cache-line-aligned approach and its predictable high-load behavior.
