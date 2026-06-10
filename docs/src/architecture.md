# Code Deduplication: GenericMap & RawTableApi

## Problem

The five hash map designs shared 95-100% of their code at two levels:

1. **Map wrapper** (~1000 lines each): constructors, entry API, iterators, `FromIterator`, `Clone`, `Debug`, `PartialEq`, `Index`, etc. — identical across all 5 designs except the struct name.

2. **Overflow-bit raw table** (~900 lines each): UFM, Splitsies, and Gaps had nearly identical probe loops, insert logic, allocation, rehash, and iteration. The only differences were parameterizable: overflow storage location, bucket stride, and SIMD bitmask width.

Total duplication: ~8,500 lines across 15 files.

## Solution: Two Generic Abstractions

The design matrix is parameterized via several composable trait layers:

### GenericMap<K, V, S, R: RawTableApi>

A single map wrapper that replaces all 5 `map.rs` files. Contains:

- Constructors (`new`, `with_capacity`, `with_hasher`, `with_capacity_and_hasher`)
- Core ops (`get`, `insert`, `remove`, `contains_key`, `get_key_value`, `get_mut`)
- Entry API (`Entry`, `OccupiedEntry`, `VacantEntry`)
- Iterators (`Iter`, `IterMut`, `IntoIter`, `Keys`, `Values`, `ValuesMut`)
- Trait impls (`Default`, `IntoIterator`, `FromIterator`, `Extend`, `Index`, `Debug`, `Clone`, `PartialEq`, `Eq`)
- Bulk ops (`retain`, `drain`, `reserve`, `shrink_to_fit`)

Each concrete map type is a type alias:

```rust
pub type UnorderedFlatMap<K, V, S> = GenericMap<K, V, S, RawTable<K, V, UfmLayout>>;
pub type Splitsies<K, V, S>       = GenericMap<K, V, S, RawTable<K, V, SplitsiesLayout>>;
pub type Gaps<K, V, S>            = GenericMap<K, V, S, RawTable<K, V, GapsLayout>>;
pub type InPlaceOverflow<K, V, S> = GenericMap<K, V, S, ipo::RawTable<K, V>>;
pub type IPO64<K, V, S>           = GenericMap<K, V, S, ipo64::RawTable<K, V>>;
```

### RawTableApi<K, V> — Internal Trait

The contract between GenericMap and each raw table backend. Key methods:

| Category | Methods |
| --- | --- |
| Construction | `new()`, `with_capacity()` |
| Queries | `len()`, `capacity()`, `is_allocated()`, `num_groups()` |
| Lookups | `find_bucket()`, `find_by_hash()` |
| Insert | `insert_or_replace()` (fused fast path), `insert_at()`, `insert_no_check()` |
| Entry | `find_for_entry()` (fused fast path), `ensure_capacity()` |
| Remove | `remove_by_hash()`, `erase_slot()` (design-specific cleanup) |
| Iteration | `iter_slots()`, `into_iter_impl()`, `drain_impl()` |
| Capacity | `reserve()`, `shrink_to_fit()`, `rehash_with()` |

Performance-critical methods (`insert_or_replace`, `find_for_entry`) include the fused home-group SIMD fast path inside the raw table, not in GenericMap. Each design's fast path is fully specialized via monomorphization.

### GroupLayout + overflow_table::RawTable<K, V, L>

A single generic overflow-bit raw table replaces three separate implementations. The `GroupLayout` trait composes three strategy traits:

```rust
pub trait GroupLayout: 'static + Copy {
    type Grp: GroupOps;           // SIMD operations (slot mask)
    type Tag: TagStrategy;        // Hash tag + overflow channel extraction
    type Overflow: OverflowStrategy; // Overflow storage format

    const GROUP_SIZE: usize;      // 15 or 16
    const BUCKET_STRIDE: usize;   // 15 or 16
    const SEPARATE_OVERFLOW: bool; // Controls extra prefetch
    const GROUP_INDEX_REGION: HashRegion; // High = shift, Low = AND (see below)
}
```

Named layouts for existing designs:

| Axis | UFM | Splitsies | Gaps |
| --- | --- | --- | --- |
| Usable slots | 15 | 16 | 15 |
| Bucket stride | 15 (`gi*15+si`) | 16 (`(gi<<4)\|si`) | 16 (`(gi<<4)\|si`) |
| SIMD mask | `0x7FFF` | `0xFFFF` | `0x7FFF` |
| Overflow location | Byte 15 of group | Separate array | Byte 15 of group |
| Extra allocation | 0 | `num_groups` bytes | 0 |
| Prefetch strategy | 2 prefetches/probe | 3 (extra for overflow) | 2 |

All constants and pointer arithmetic are resolved at compile time. The `GroupOps` associated type carries the SIMD operations parameterized by slot mask, avoiding unstable `generic_const_exprs`.

### Design matrix: tag × overflow × indexing

Beyond the three named designs, `Layout16<T, O>` and `Layout16And<T, O>` compose any `TagStrategy` with any `OverflowStrategy`:

| Tag (bits) \ Overflow | 8-bit (ByteSeparate) | 1-bit (BitSeparate) | Tombstone (IPO) |
| --- | --- | --- | --- |
| `LowTag255` (0-7) | Splitsies (baseline) | `LowTag255_1bit` | — |
| `LowTag128` (0-7) | `LowTag128_8bit` | `LowTag128_1bit` | — |
| `LowTag255ChSafe` (8-15) | `LowTag255ChSafe_8bit` | `LowTag255ChSafe_1bit` | — |
| `LowTomb` (0-7) | — | — | IPO64 (default) / `LowTomb_TombMap` |
| `HighTag128` (56-63) (AND) | — | `HighTag128_1bitAnd` | `HighTag128_TombMap` |
| `HighTag255` (56-63) (AND) | — | `HighTag255_1bitAnd` | — |
| `HighTomb` (56-63) (AND) | — | — | IPO (default) / `HighTomb_Tomb64Map` (collision-prone on IPO64) |

Strategy names follow `<Region><Kind><Width>[ChSafe][Pure]`: `Region` is `Low` (bottom of the hash) or `High` (top), `Kind` is `Tag` or `Tomb`, `Width` is the count of distinct tag values. The region word is the legibility cue — a safe pairing reads as opposite words (a `High` group index with a `LowTag`), a broken one as matching words.

AND-indexed variants use `Layout16And` which sets `GROUP_INDEX_REGION = HashRegion::Low`. See the "Group indexing" section below.

### Group indexing: shift vs AND

Hash tables map a hash value to a group index. Two strategies:

**Shift-based** (default): `gi = (h >> shift) & mask` — uses high hash bits. Tags can safely use low or middle bits (`LowTag*`) since they're decorrelated. Costs 2 instructions (variable shift + AND).

**AND-based**: `gi = h & mask` — uses low hash bits. Saves 1 instruction (just AND), but tags must come from top hash bits (`HighTag*`) to avoid correlation. Low-region tags (`LowTag*`, bits 0-7) are _not_ safe under AND indexing — the mask reaches into byte 0 at any non-trivial size, correlating tag bits with the group index. IPO uses `HighTomb` (bits 56-63) as its default to escape this; IPO64 (shift-indexed → top bits are the group index) uses `LowTomb` (bits 0-7), where the bottom of the hash is the safe region.

Additionally, 8-bit overflow channels use `1 << (h & 7)` which also uses low bits — every key in the same group would get the same channel, making 8-channel overflow useless. **AND indexing is only safe with 1-bit overflow (BitSeparate) or tombstone designs (no overflow channels).**

### Memory layout: mid-pointer design

Both RawTable implementations use a mid-pointer allocation layout inspired by hashbrown. A single `ctrl` pointer sits at the boundary between buckets (backward) and metadata (forward):

```text
  Overflow-bit designs:
  ┌──────────────────────┬────────────────────┬───────────────┐
  │ Buckets (KV pairs)   │ Metadata (16B/grp) │ Overflow bytes│
  │ ◄── backward         │ forward ──►        │ forward ──►   │
  └──────────────────────┴────────────────────┴───────────────┘
  ↑ alloc_ptr (computed)  ↑ ctrl (stored)

  Tombstone designs (IPO):
  ┌──────────────────────┬────────────────────┐
  │ Buckets (KV pairs)   │ Metadata (16B/grp) │
  │ ◄── backward         │ forward ──►        │
  └──────────────────────┴────────────────────┘
  ↑ alloc_ptr (computed)  ↑ ctrl (stored)
```

- **Metadata**: `ctrl + gi * 16` (forward from ctrl)
- **Buckets**: `ctrl.cast::<(K,V)>().sub(slot_index + 1)` (backward from ctrl)
- **Overflow** (overflow-bit only): `ctrl + num_groups * 16 + offset` (forward, after metadata)

This eliminates a separate `buckets` pointer field, reducing the struct from 7 fields to 5 (overflow-bit) or 5 (tombstone). Both metadata and bucket access derive from `ctrl` in opposite directions, saving a register and an address computation in the hot path. hashbrown uses the same trick.

Overflow-bit designs have 3 memory regions but only need 2 pointers worth of addressing: the hot path (metadata + bucket) uses `ctrl`, and overflow is computed as a forward offset from `ctrl` (only accessed on miss/insert).

## The design axes, visually

The text above describes the matrix in terms of trait composition. This section adds the visual scaffolding for each axis — what one SIMD probe step looks like, how the tag byte is encoded, how group width interacts with SIMD registers, and how the three overflow tracking strategies differ in memory.

### A probe step

A *probe step* loads one group's worth of metadata into a SIMD register and compares it against the broadcast tag in a single instruction:

```text
ctrl + gi*16 →   [m0 ][m1 ][m2 ][m3 ][m4 ][m5 ][m6 ][m7 ][m8 ][m9 ][mA ][mB ][mC ][mD ][mE ][mF ]
                  0x4A 0x73 0x4A TOMB EMPT 0x4A 0xB1 0x12 EMPT 0x4A 0x77 0x09 0x55 EMPT 0xC4 0x4A

  broadcast      [0x4A][0x4A][0x4A][0x4A][0x4A][0x4A][0x4A][0x4A][0x4A][0x4A][0x4A][0x4A][0x4A][0x4A][0x4A][0x4A]
  tag (target)

  _mm_cmpeq_epi8 → 16-bit mask of tag matches:

                  bit:  F E D C B A 9 8 7 6 5 4 3 2 1 0
                  val:  0 0 0 0 1 0 0 0 0 1 0 1 0 1 0 1   ← slots 0,2,5,9,B match
```

A bit set in the mask means "the tag matches — go check `K == key` at that bucket".
The probe terminates either when a key matches, or when the group contains an EMPTY slot (for the tombstone family) or when the overflow bit for the query's hash class is clear (for the overflow-bit family).

### Tag byte encoding (the three schemes)

The metadata byte is one of three encodings depending on which `TagStrategy` the layout uses:

```text
128 values (hashbrown, Tomb — HighTag128, LowTag128):
  FILLED:    0xxxxxxx        ← top bit = 0, low 7 bits = tag
  TOMBSTONE: 0x80            ← top bit = 1
  EMPTY:     0xFF            ← top bit = 1
  tag extraction:  shr h, 57; and reg, 0x7F            → 2 instructions

254 values (TombWide IPO, IPO64 — HighTomb, LowTomb):
  FILLED:    0x01..0xFE      ← 254 distinct tags
  EMPTY:     0x00
  TOMBSTONE: 0xFF
  tag extraction:  shr h, N; cmp 0xFF; adc reg, 0      → 3 instructions

255 values (overflow-bit family — UFM, Splitsies, Gaps — LowTag255 etc.):
  FILLED:    0x01..0xFF      ← 255 distinct tags
  EMPTY:     0x00            ← only sentinel (no tombstone needed)
  tag extraction:  inline asm: cmp 0xFF; adc reg, 0    → 2 instructions
```

False-match rate at a SIMD compare is `WIDTH / values_kept`. For a 16-slot group: ~12% for 128-value tags, ~6% for 254/255-value tags. Wider tags = fewer wasted bucket dereferences on miss, paid for in one extra instruction at hash time. The crate's `reduced-hash-asm` feature toggles between the `cmp; adc` x86_64 idiom and a pure-Rust fallback (`b | (b == 0) as u8`).

### Group width and SIMD register width

Each probe step loads `GROUP_SIZE` metadata bytes into a SIMD register and runs a single compare:

```text
 16-slot (SSE2):    [m]×16   →   xmm register   (128 bits, _mm_cmpeq_epi8)
 32-slot (AVX2):    [m]×32   →   ymm register   (256 bits, vpcmpeqb ymm)
 64-slot (AVX-512): [m]×64   →   zmm register   (512 bits, vpcmpeqb zmm)
```

Wider groups → fewer probe steps before terminating. At 64 slots most probes resolve in step 0 even at high load factor.

Hidden cost: a wider SIMD compare is also a wider net for false-positive tag matches. At 128 tag values and 64 slots, the false-match rate hits ~50% — half of all misses pay a wasted bucket dereference inside the group. This is why all wide-group designs in the matrix pair with 254- or 255-value tags (`*Tomb`, `*Tag255`) rather than 128-value tags (`*Tag128`). See [32/64-slot investigation](optimization/32-64-slot-investigation.md) for the measured tradeoff.

### Overflow tracking — three layouts compared

A tombstone-free design needs to know whether some entry once spilled past this group's home position. Three places to store that bit:

```text
8-bit (ByteSeparate) — one byte per group, 8 channels indexed by tag bits 0..2:

  metadata region:   [meta × 16] [meta × 16] [meta × 16] [meta × 16] …
  overflow region:   [ ov byte ] [ ov byte ] [ ov byte ] [ ov byte ] … (1B/group)

1-bit (BitSeparate) — one packed bit per group, bitfield:

  metadata region:   [meta × 16] [meta × 16] [meta × 16] [meta × 16] …
  overflow region:   [bbbbbbbb] [bbbbbbbb] …      ← 8 groups packed into one byte

embedded (UfmEmbedded) — overflow bits live inside the metadata itself:

  metadata region:   [meta × 15][ov]  [meta × 15][ov]  [meta × 15][ov]  …
                                 ▲
                       byte 15 of each group holds the overflow channels.
                       Only 15 usable slots per group, but zero extra loads.
```

| Strategy | Used by | Bytes / group | False-continuation | Slots / group |
|---|---|---:|---:|---:|
| `ByteSeparate` (8-bit, 8 channels) | Splitsies (legacy) | 1.0 | ~0.9% | 16 |
| `BitSeparate` (1-bit) | Splitsies (current), `*_1bitAnd*` matrix variants | 0.125 | ~7% | 16 |
| `UfmEmbedded` (in metadata byte 15) | UFM, Gaps | 0 (in-band) | varies by load | 15 |

`ByteSeparate` has the lowest false-continuation rate but the largest separate allocation. `BitSeparate` is 8× smaller and still always L1-hot. `UfmEmbedded` puts the bits inside the metadata cache line itself (zero extra memory loads), but burns one slot per group and is structurally tied to 16-byte SIMD width.

## What Stays Separate

- **IPO and IPO64** keep their own `RawTable` implementations. Their probe strategy (tombstone-based, EMPTY termination) is fundamentally different from the overflow-bit family. IPO's `RawTable<K,V,T: TombstoneTag>` is parameterized by tag strategy and also uses the mid-pointer layout. They implement `RawTableApi` and use GenericMap for the wrapper layer.

- **FlatBTree** is a B+ tree, not a hash table. No overlap.

- **UnorderedFlatSet** (`set.rs`) is hand-written with direct UFM raw table access for SIMD fast paths. Uses the legacy UFM raw table in `raw/mod.rs`.

## Impact

| Metric | Before | After |
| --- | --- | --- |
| Map wrapper code | 5 x ~1000 lines | 1 x 740 lines |
| Overflow-bit raw tables | 3 x ~900 lines | 1 x 936 lines |
| Overflow-bit group ops | 3 x ~250 lines | 1 x 173 lines |
| New shared infrastructure | — | 371 lines (traits + layout) |
| **Total** | **~8,500 lines** | **~2,400 lines** |
| **Net reduction** | — | **~4,500 lines deleted (-72%)** |

Performance: zero-cost. All generics monomorphize to identical machine code. Benchmarks show no systematic regressions (17 improvements vs 3 regressions in full throughput suite, regressions attributable to measurement noise).

## Adding a New Overflow-Bit Design

To add a new overflow-bit variant (e.g., Splitsies-1bit):

1. Define a new layout struct implementing `GroupLayout` (~30 lines)
2. Add a type alias: `pub type Splitsies1Bit<K,V,S> = GenericMap<K,V,S, RawTable<K,V, Splitsies1BitLayout>>;`
3. Add `impl_map_trait!(Splitsies1Bit);` for the `Map` trait

That's it. No new probe loops, no new entry API, no new iterators.

### TagStrategy & Hash Reduction

Each `GroupLayout` has an associated `TagStrategy` that determines:

- How the hash tag byte is extracted from the 64-bit hash
- How the overflow channel bits are selected

The tag byte for a given key is computed once during insertion and stored as the metadata byte. During probing, the SIMD group match compares stored tags against the probe key's tag.

Tag strategies come in two families:

| Strategy        | Byte     | Values | Reduction            | Indexing |
| --------------- | -------- | ------ | -------------------- | -------- |
| `LowTag255`     | 0 (low)  | 255    | crate `hash_tag()`   | Shift    |
| `LowTag128`     | 0 (low)  | 128    | `b \| 1` (inline)    | Shift    |
| `LowTag255Pure` | 0 (low)  | 255    | `b \| (b==0) as u8`  | Shift    |
| `LowTag255ChSafe`     | 1        | 255    | crate `hash_tag()`   | Shift    |
| `HighTag128`     | 7 (high) | 128    | `b \| 0x80` (inline) | AND      |
| `HighTag255`     | 7 (high) | 255    | crate `hash_tag()`   | AND      |
| `HighTag255Pure` | 7 (high) | 255    | `b \| (b==0) as u8`  | AND      |

The `Pure` variants always use a pure-Rust hash reduction (3 instructions: `test; sete; or`), independent of the crate's `reduced-hash-asm` feature. The non-Pure variants route through `crate::hash_tag()` which selects the asm idiom (`cmp; adc`) on x86_64 or the pure-Rust fallback elsewhere. `LowTag128` and `HighTag128` inline their own 1-instruction reduction and never call `crate::hash_tag()`.

This per-strategy control lets different backends choose different hash reduction strategies, decoupling from the global feature flag.
