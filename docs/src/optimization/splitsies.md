# Splitsies Optimization Log

Tested 8 techniques. Key finding: optimizations that fail on UFM also fail
on Splitsies for the same fundamental CPU reasons.

## Starting Point

Splitsies vs hashbrown at 70% load:

**Wins**: insert (0.93x), remove (0.64x), churn (0.62x), read-heavy (0.99x),
from_iter (0.62x), sequential miss (0.74x), miss@85% (0.28x)

**Gaps**: lookup hit (1.15x), iteration (1.14x), entry API (1.66x),
miss at low load (1.08-1.10x), counting (1.28-1.35x)

Struct size: 64 bytes (vs UFM 56, hashbrown 40)

## Results

### S1: Drop overflow pointer from struct — KEPT

Derive overflow array position from `metadata + num_groups * 16` instead of
storing a separate pointer. 64 → 56 bytes. Performance neutral (overflow
access is on the cold path, already prefetched).

### S2: Home-group bucket prefetch — REJECTED

Same as UFM attempt #25. +8-17% miss regression from cache pollution by
unused bucket data. The overflow prefetch already uses the prefetch port;
adding a bucket prefetch competes for bandwidth.

### S3: Inline find_by_hash + cold continuation — REJECTED

Same as UFM attempt #18. +12-26% hit regression from register pressure at
the `#[inline(never)]` boundary. The 16-slot design doesn't change the CPU's
register allocation behavior.

### S4: Entry API #[inline] — NOT TESTED

Expected same trade-off as UFM #20 (helps hit-heavy, hurts insert-heavy).
Skipped.

### S5: Anti-drift optimization — SKIPPED

Only 2 cheap instructions in the remove path. Not worth the refactor.

### S6: Lazy overflow_bit computation — REJECTED

Defer `overflow_bit` past SIMD match so it's not computed on the hit path.

Result: +1-3% hit improvement but +2% miss regression (moved to serial
critical path). Combined with the prefetch (which hides latency), the
serial dependency was exposed.

### S7: Combined metadata+overflow memset — KEPT

Zero metadata and overflow in one call during `allocate()` and `clone()`
since they're contiguous. Minor construction improvement.

### S8: Struct field reorder — NOT TESTED

Modern CPUs fetch full cache lines. Marginal expected impact.

## Key Lesson

The gains from Splitsies come from the power-of-2 arithmetic being cheaper,
not from enabling new optimization strategies. Every micro-optimization that
failed on UFM (bucket prefetch, cold continuation, lazy overflow_bit) also
fails on Splitsies for the same fundamental reasons:

- **Bucket prefetch**: Cache pollution from unused data on miss path
- **Cold continuation**: Register pressure at `#[inline(never)]` boundary
- **Lazy overflow_bit**: Serial dependency on miss path

The 16-slot design doesn't change CPU behavior around these constraints.
