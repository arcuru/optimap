# UnorderedFlatMap

Based on the design described in
[Inside boost::unordered_flat_map](https://bannalia.blogspot.com/2022/11/inside-boostunorderedflatmap.html).

An open-addressing hash table storing elements contiguously in a flat bucket array,
with a companion SIMD metadata array that accelerates lookup, insertion, and deletion.

## Memory Layout

```text
Metadata array:  2^n  ×  16-byte "group words"
┌─────────────────────────────────┬─────┐
│ hi0 hi1 hi2 … hi13 hi14        │ ofw │  ← 1 group = 15 metadata bytes + 1 overflow byte
└─────────────────────────────────┴─────┘

Bucket array:  2^n  ×  15  slots  (key,value)
┌───┬───┬───┬─────┬────┐
│ 0 │ 1 │ 2 │ ... │ 14 │  ← 1 group = 15 buckets
└───┴───┴───┴─────┴────┘
```

### Constants

| Name | Value | Meaning |
|------|-------|---------|
| `GROUP_SIZE` | 15 | Buckets per group |
| `META_GROUP_SIZE` | 16 | 15 hash bytes + 1 overflow byte |
| `EMPTY` | 0x00 | Slot is vacant |
| `SENTINEL` | 0x01 | Iteration terminator (placed after last group) |
| `MIN_HASH` | 0x02 | Lowest valid reduced-hash value |

### Metadata Byte Encoding

Each occupied bucket stores a reduced hash in `[1, 255]`, derived from the LSB
of the full hash via a saturating increment (`cmp 0xFF; adc 0` on x86_64).
Only 0x00 is reserved (EMPTY sentinel). The overflow bit is computed from the
raw hash (`1 << (h & 7)`), not from the reduced hash — the two are independent.

### Overflow Byte

Each group has a single overflow byte. Bit `i` (0..7) is set when an element whose
`hash % 8 == i` was displaced from this group to a later group during insertion.
During lookup, if the overflow bit for the query's `hash % 8` is **not** set,
probing stops immediately.

## Algorithms

### Lookup

```text
1. h = hash(key)
2. group_index = h >> (W - n)          // initial (home) group
3. reduced = reduced_hash(h)
4. ofw_bit = 1 << (h % 8)
5. loop:
     a. Load 16-byte metadata word for group_index
     b. SIMD compare: mask = (metadata[0..15] == reduced)
     c. For each set bit in mask:
          - Compare full key in bucket; return if match
     d. If overflow bit (ofw_bit) is NOT set → return NOT FOUND
     e. Advance probe sequence (quadratic: group_index += probe_delta)
     f. Prefetch next group metadata + buckets (overflow-only)
```

### Insertion (Fused Home-Group Path)

Combines the duplicate check and empty-slot search into a single SIMD load:

```text
1. h = hash(key)
2. If len >= max_load → cold path: find + grow + insert
3. Single SIMD load on home group metadata:
     match_mask = (metadata[0..15] == reduced)    // key candidates
     empty_mask = (metadata[0..15] == EMPTY)      // available slots
4. For each set bit in match_mask:
     - Compare full key; if match → replace value, return old
5. If empty_mask has a set bit AND overflow bit is NOT set:
     - Key is absent. Pick first empty slot.
     - Write (key, value), set metadata, increment count, return.
6. Cold: overflow → full probe, then insert_no_check with overflow bit setting
```

The fast path handles >85% of inserts at typical load factors with one SIMD load.

### Deletion (Tombstone-Free)

```text
1. Find element via lookup
2. Set metadata byte to EMPTY (0x00)
3. Decrement count
4. If the element was displaced from its home group:
     - Decrement max_load by 1 (anti-drift)
     - This triggers earlier rehashing to clear stale overflow bits
```

## SIMD Strategy

### x86_64 (SSE2)

- `_mm_load_si128` — aligned load of 16-byte metadata word
- `_mm_cmpeq_epi8` — compare all 16 bytes at once
- `_mm_movemask_epi8` — extract comparison result as bitmask

The fused insert path uses `match_byte_and_empty`: one aligned load, two compares,
two movemasks — yielding both key-match and empty-slot bitmasks from one memory access.

### aarch64 (NEON)

- `vld1q_u8` — load 16 bytes
- `vceqq_u8` — compare
- Bitmask extraction via shift+narrow sequence

### Fallback

Portable scalar fallback: iterate over 15 bytes with a simple loop.

## Struct Layout (56 bytes)

```text
mask:       usize   (8)  — num_groups - 1, hot-path masking
metadata:   *mut u8 (8)  — pointer to group metadata array
buckets:    *mut u8 (8)  — pointer to bucket array
len:        usize   (8)  — number of elements
max_load:   usize   (8)  — threshold for rehash
shift:      u32     (4)  — hash >> shift gives group index
padding:            (4)
```

## Key Trade-offs

- **15-slot groups** waste one SIMD lane on the overflow byte, requiring a
  `& 0x7FFF` mask after movemask
- **Bucket addressing** uses `gi * 15 + si` (multiply by non-power-of-2)
  instead of a shift — costs ~2-3 extra instructions per element
- **Two separate allocations** (metadata + buckets) avoid the 1M insert
  regression seen with single allocation, at the cost of an extra pointer
