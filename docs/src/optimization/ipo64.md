# IPO64 Optimization Log

## Baseline (SSE2 only, 4×16-byte loads)

IPO64 was 1.3-2.3x slower than IPO16 at all sizes. Root cause: 14 SIMD
operations per probe step vs IPO16's ~4.

## Attempt 1: AVX-512 + AVX2 with per-call dispatch — PARTIAL

Added `is_x86_feature_detected!` dispatch inside each Group method.

- Miss improved: -14% at 32K, -36% at 20M (fewer SIMD ops per step)
- Hit regressed: +3-9% (atomic load overhead of dispatch inside probe loop)

## Attempt 2: Dispatch at find_by_hash entry point — KEPT

One check at entry, entire probe loop runs with best SIMD tier. No
per-iteration overhead.

| Size | Hit change | Miss change |
|-----:|:----------:|:-----------:|
| 256 | **-11%** | -3% |
| 4K | **-8%** | -7% |
| 32K | -4% | **-19%** |
| 2M | +3% (noise) | **-23%** |
| 20M | **-2%** | **-34%** |

## Current State: IPO64 vs IPO16 vs hashbrown

| Size | IPO64 hit | IPO16 hit | hb hit | IPO64 miss | IPO16 miss | hb miss |
|-----:|----------:|----------:|-------:|-----------:|-----------:|--------:|
| 256 | 2.33 ns | 1.67 ns | 1.59 ns | 2.03 ns | 1.16 ns | 0.91 ns |
| 4K | 2.48 ns | 1.70 ns | 1.66 ns | 2.03 ns | 1.16 ns | 0.97 ns |
| 32K | 2.91 ns | 2.31 ns | 2.21 ns | 2.11 ns | 1.21 ns | 1.14 ns |
| 256K | 4.95 ns | 3.24 ns | 3.16 ns | 3.69 ns | 1.81 ns | 2.09 ns |
| 2M | 20.5 ns | 18.7 ns | 17.7 ns | 9.29 ns | 3.58 ns | 5.78 ns |
| 20M | 42.7 ns | 31.9 ns | 29.8 ns | 24.4 ns | 14.4 ns | 17.0 ns |

IPO64 remains 1.3-1.5x slower than IPO16 on hits. AVX-512 helped but
didn't close the gap.

## Attempt 3: Load factor sweep — CONFIRMED

IPO64's defining property: flat miss degradation under load.

| Load | IPO16 miss | IPO64 miss | hb miss |
|-----:|-----------:|-----------:|--------:|
| 50% | 1.95 ns | 3.40 ns | 1.56 ns |
| 90% | 2.95 ns | 4.56 ns | 2.19 ns |
| 95% | 3.99 ns | 5.26 ns | 3.63 ns |
| 97% | 4.38 ns | **4.68 ns** | 4.17 ns |
| **99%** | **5.33 ns** | **5.13 ns** | 4.79 ns |

Miss degradation 50→99%: IPO16 2.73x, **IPO64 1.51x**, hashbrown 3.07x.

Crossover with IPO16 at 99% load — too extreme for general use.

## Why IPO64 Remains Slower

At 70% load, 64-slot groups resolve >99% of probes in a single step — same
as 16-slot groups. IPO64 does more SIMD work per step (3 ops with AVX-512
vs IPO16's 2 ops) for the same number of steps. The cache-line advantage
only helps when multiple steps are needed, which is rare below 95% load.

## Conclusion

Optimized as far as practical:
- AVX-512 reduced SIMD ops from 14 to 3 per step
- Entry-point dispatch eliminated per-iteration overhead
- Load factor sweep confirms flat degradation curve
- Crossover with IPO16 only at 99% load

Further improvements (insert_no_check dispatch, iteration AVX-512) give
diminishing returns. IPO64 is a research/specialty design for applications
needing predictable high-load performance.
