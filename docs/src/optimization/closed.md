# Closed Investigations

These performance gaps have been thoroughly investigated across multiple designs
and proven to be structural. Further optimization attempts are unlikely to yield
improvements without fundamental design changes.

---

## Lookup Hit Gap (1.11-1.25x vs hashbrown)

**Status: CLOSED — structural overhead, not fixable within current designs**

The lookup hit gap is constant across all load factors and table sizes. It comes
from per-probe overhead that is inherent to the overflow-bit group design:

1. **15-slot groups** (UFM, Gaps): Waste one SIMD lane on the overflow byte,
   requiring `& 0x7FFF` after movemask. hashbrown uses all 16 bytes for hash values.
2. **Overflow-bit bookkeeping**: Computing `overflow_bit = 1 << (h % 8)` on every
   probe step, even though it's unused on the hit path.
3. **Bucket addressing**: `gi * 15 + si` (UFM) requires multiply-by-15 vs
   hashbrown's flat index. Splitsies fixes this with `(gi << 4) | si`.
4. **Wider metadata reads**: hashbrown's inner loop is extremely tight: one aligned
   `_mm_load_si128`, one `_mm_cmpeq_epi8`, one `_mm_movemask_epi8`, mask to 14 bits.

### Attempts to close this gap

| Attempt | Design | Result | Why it failed |
|---------|--------|--------|---------------|
| **#7: Initial bucket prefetch** | UFM | -11% hit, +21% miss | Cache pollution on miss path. Wasted prefetch for elements not in the table. |
| **#11: Conditional prefetch** (only on SIMD match) | UFM | No improvement | Branch overhead (~5-10 cycles to resolve) > cost of wasted prefetch (~1 cycle). Adding a conditional made the hit path slower, not faster. |
| **#18: Inline home-group + cold continuation** | UFM | +10-14% hit regression | `#[inline(never)]` on overflow continuation forced register save/restore at the call boundary. Register pressure degraded the hot path even though the cold path is rarely taken. |
| **S2: Home-group bucket prefetch** | Splitsies | -5-8% hit, +8-17% miss | Same cache pollution as UFM #7. Splitsies' overflow prefetch already uses the prefetch port. |
| **S3: Inline + cold continuation** | Splitsies | +12-26% hit regression | Same register pressure as UFM #18. 16-slot design doesn't change CPU register allocation. |
| **S6: Lazy overflow_bit** | Splitsies | +1-3% hit, +2% miss | Deferring computation to after SIMD match moved it onto the serial critical path for misses. |
| **#25: Bucket prefetch re-test** | UFM | -5-8% hit, +6-11% miss | Load-factor-controlled re-test confirmed: helps hits at low-medium load, hurts misses. No universal win. |
| **Static empty sentinel** | All 5 | ~0% change | Replaced `is_allocated()` null-pointer branch with static sentinel metadata. The branch was already perfectly predicted — removing it didn't help. |
| **find_bucket (direct pointer return)** | All 5 | ~0% change | Added `find_bucket()` returning `*mut (K,V)` directly, eliminating double `bucket_ptr` recomputation in `get()`. LLVM was already doing CSE on the inlined code — no measurable improvement. |

### Sweep benchmark analysis (April 2025)

A continuous N-sweep benchmark (100 to 10M, 362 points, 5 trials/point, median)
confirmed the gap structure with high resolution:

| N range | IPO vs hashbrown (hit) | IPO vs hashbrown (miss) |
|---------|:----------------------:|:-----------------------:|
| <10k | 1.03-1.05x | 1.29-1.39x |
| 10k-100k | 1.03-1.12x | 0.95-1.08x (load-dependent) |
| 100k-1M | 1.11-1.13x | 0.97-1.05x |
| >1M | 1.11x | ~1.05x |

The miss gap at small N (1.3-1.4x) is due to hashbrown's tighter miss hot path:
`Tag::full()` = pure shift+mask vs `reduced_hash()` = mask + conditional cmov.
At higher load (50k+), overflow-bit designs recover and sometimes win.

### Tag extraction: why we can't match hashbrown's 2-instruction path

hashbrown's `Tag::full(hash)` = `(hash >> 57) & 0x7F` — 2 instructions (shift + and).
This works because hashbrown reserves the MSB: full tags are 0x00-0x7F (128 values),
EMPTY=0xFF and DELETED=0x80 both have bit 7 set.

Our IPO/IPO64 `reduced_hash(h)` reserves 0x00 (EMPTY) and 0x01 (TOMBSTONE), giving
254 usable values (2-255). Avoiding those two values costs 5 instructions
(`mov; or; cmp; movzbl; cmov`). Alternatives considered:

| Approach | Instructions | Distinct values | Problem |
|----------|:-----------:|:---------------:|---------|
| `(h >> 57) & 0x7F` (hashbrown) | 2 | 128 | Halves our hash discrimination |
| `(h & 0xFF) \| 2` | 2 | 128 | Sets bit 1, collapsing half the inputs — same 128 values as hashbrown |
| `(h & 0xFF).saturating_add(2)` | 4 | 254 | Still needs a cmov (clamp to 255) |
| `if low < 2 { low + 2 }` (current) | 5 | 254 | cmov, but preserves all 254 values |

The 254 vs 128 value tradeoff matters: more distinct hash values = fewer false-positive
SIMD matches = fewer wasted key comparisons. At 254 values the false-match probability
per slot is 1/254 (~0.39%); at 128 it's 1/128 (~0.78%) — double the collision rate.
This is why our insert is faster than hashbrown despite the slower tag extraction.

The overflow-bit designs (UFM, Gaps, Splitsies) only reserve 0x00 (EMPTY), giving 255
values. Their `reduced_hash` uses `low | ((low == 0) as u8 * 8)` — 3 instructions,
no cmov, branchless arithmetic.

### Why further attempts are unlikely to help

The hit gap is fundamentally about **per-probe instruction count**. hashbrown's probe
loop is ~6 instructions (load, compare, movemask, mask, branch, trailing_zeros).
Ours is ~9-11 instructions (same, plus overflow bit computation, wider mask, and
non-power-of-2 bucket arithmetic).

Every attempt to reduce this count has either:
- **Moved work off the hot path** → it ended up on the miss path (serial dependency)
- **Added speculative work** (prefetch) → cache pollution on misses
- **Split the function** (inline + cold) → register pressure at the boundary
- **Removed branches** (sentinel, find_bucket) → branches were already predicted, LLVM already optimizing

The only way to close this gap is to eliminate the overflow byte entirely (which
loses O(1) miss termination) or switch to 16-slot groups (which Splitsies does,
getting the gap down to 1.11x — the remaining overhead is the separate overflow
array access).

**InPlaceOverflow (IPO) achieves ~1.03x** on hits at small N by dropping the overflow
design entirely and using tombstones like hashbrown, but with 254 hash values vs 128.

---

## Selective Prefetch for Hit-Heavy Workloads

**Status: CLOSED — no universal policy exists**

We removed all manual prefetching (#14) because miss improvement (-27% at 1M)
outweighed hit regression (+18% at 1M). This trades ~18% on hits at 1M scale
for a large miss improvement.

### What was explored

| Approach | Result |
|----------|--------|
| **Unconditional prefetch** (#7) | -11% hit, +21% miss |
| **No prefetch** (#14) | +18% hit at 1M, -27% miss at 1M |
| **Conditional prefetch** (#11) | Worse than unconditional (branch overhead) |
| **Overflow-only prefetch** (P7) | Best compromise: prefetch only after overflow check |
| **Bucket prefetch re-test** (#25) | -5-8% hit, +6-11% miss at low load |

### Why adaptive prefetch was not pursued

The idea of tracking hit/miss ratio at runtime and toggling prefetch behavior was
considered (FUTURE.md item #5). It was not pursued because:

1. **Counter overhead**: Even a simple counter checked every N ops adds a branch and
   memory write to the hot path
2. **Hysteresis**: The optimal prefetch policy depends on the *upcoming* access pattern,
   not the historical one. A workload that transitions from hit-heavy to miss-heavy
   would suffer during the detection lag.
3. **Existing solution**: Users who know their workload is hit-dominated can use
   InPlaceOverflow (IPO), which has ~1.01x hit performance. The design choice *is*
   the prefetch policy.

### Why user-selectable prefetch policy was not pursued

A compile-time or runtime flag could let users opt into bucket prefetch. Not
implemented because:

1. **API complexity**: Adds a non-obvious knob that requires benchmarking to tune
2. **The right answer is design selection**: IPO for hit-heavy, Splitsies for
   balanced, UFM for miss/churn-heavy. The Map trait makes switching easy.

---

## SIMD Backend: AVX2 / AVX-512

**Status: CLOSED for UFM/Splitsies — implemented for IPO64 only**

### AVX2 multi-group probing (UFM #21)

Measured probe chain statistics:

| Load % | Full groups | Home-group hit rate |
|-------:|------------:|--------------------:|
| 45% | 0.4% | 99.6% |
| 65% | 7.1% | 92.9% |
| 85% | 30.0% | 70.0% |

**For probing**: AVX2 could combine two SIMD comparisons into one 256-bit operation,
but the quadratic probe sequence visits non-adjacent groups. Would need two separate
16-byte loads combined via `_mm256_set_m128i`, saving one cmpeq+movemask but adding
a combine. At 65% load, 93% of operations resolve in the home group — AVX2 can't
help there.

**For iteration**: AVX2 would halve SIMD metadata loads (one 32-byte load → two groups).
But iteration at small-medium sizes is bottlenecked by bucket access (pointer arithmetic
+ memory loads), not metadata SIMD loads. At 1M+ it's already memory-bound (we tie
hashbrown).

**Additional concerns**:
- Runtime feature detection adds a branch on every call
- AVX2 can cause frequency throttling on some Intel CPUs
- Code complexity doubles (SSE2 + AVX2 paths)

### AVX-512 for IPO64

AVX-512 *was* implemented for IPO64 (#2 in IPO64 log) because 64-slot groups
naturally map to a single 512-bit load. This reduced SIMD ops from 14 to 3 per
probe step, with dispatch at the find_by_hash entry point.

For 16-slot groups (UFM, Splitsies, IPO), AVX-512 offers no benefit: a single
SSE2 128-bit load already covers the full 16-byte metadata group.
