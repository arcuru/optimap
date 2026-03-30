# UnorderedFlatMap — Design Document

## Overview

This is a Rust implementation of the data structure described in
[Inside boost::unordered_flat_map](https://bannalia.blogspot.com/2022/11/inside-boostunorderedflatmap.html).

It is an **open-addressing hash table** that stores elements contiguously in a
flat bucket array (no indirection / no linked lists), with a companion SIMD
metadata array that accelerates lookup, insertion, and deletion.

## Core Data Structure Layout

```
┌──────────────────────────────────────────────────┐
│  Metadata array:  2^n  ×  16-byte "group words"  │
│  ┌─────────────────────────────────┬─────┐        │
│  │ hi0 hi1 hi2 … hi13 hi14        │ ofw │  ← 1 group = 15 metadata bytes + 1 overflow byte
│  └─────────────────────────────────┴─────┘        │
│                                                    │
│  Bucket array:  2^n  ×  15  slots  (key,value)    │
│  ┌───┬───┬───┬─────┬────┐                         │
│  │ 0 │ 1 │ 2 │ ... │ 14 │  ← 1 group = 15 buckets│
│  └───┴───┴───┴─────┴────┘                         │
└──────────────────────────────────────────────────┘
```

### Constants

| Name | Value | Meaning |
|------|-------|---------|
| `GROUP_SIZE` | 15 | Buckets per group |
| `META_GROUP_SIZE` | 16 | 15 hash bytes + 1 overflow byte |
| `EMPTY` | 0x00 | Slot is vacant |
| `SENTINEL` | 0x01 | Iteration terminator (placed after last group) |
| `MIN_HASH` | 0x02 | Lowest valid reduced-hash value |
| `MAX_LOAD_FACTOR` | 0.875 | Fixed; triggers rehash |

### Metadata byte encoding

For each occupied bucket, the metadata byte stores a **reduced hash**
in the range `[2, 255]`. The reduced hash is derived from the **least
significant byte** of the full hash, mapped so that:

```
reduced_hash(h) = (h_low & 0xFE) | 0x02   // conceptually; keeps mod-8 alignment
```

More precisely, the mapping is: take `h & 0xFF`, if it's < 2 then add 2,
yielding a value in `[2, 255]`. The critical invariant is:

> `reduced_hash(h) % 8 == h % 8`

This ensures the overflow bit (indexed by `h % 8`) is consistent regardless
of whether we look at the full hash or the reduced hash.

### Overflow byte

Each group has a single overflow byte `ofw`. Bit `i` (0..7) is set when an
element whose `hash % 8 == i` was **displaced** from this group to a later
group during insertion. During lookup, if the overflow bit for the query's
`hash % 8` is **not** set, probing can stop immediately — no element with
that hash was ever displaced from this group.

## Hash Post-Mixing

Poor hash functions are automatically improved with a bit mixer:

- **64-bit**: `xmx` mixer — multiply-xor-multiply with fixed constants
- **32-bit**: Hash Function Prospector mixer

This is applied to every hash before use unless the hasher is explicitly
marked as "avalanching".

## Algorithms

### Lookup

```
1. h = mix(hash(key))
2. group_index = h >> (W - n)          // initial group
3. reduced = reduced_hash(h)
4. loop:
     a. Load 16-byte metadata word for group_index
     b. SIMD compare: mask = (metadata[0..15] == reduced)
     c. For each set bit in mask:
          - Compare full key in bucket; return if match
     d. If group not full OR overflow bit (h%8) is 0 → return NOT FOUND
     e. group_index = (group_index + probe_delta) % num_groups   // quadratic
     f. Advance probe sequence
```

### Insertion

```
1. Lookup key first; if found, return existing
2. h = mix(hash(key))
3. group_index = h >> (W - n)
4. reduced = reduced_hash(h)
5. loop:
     a. Load metadata for group_index
     b. SIMD compare: mask = (metadata[0..15] == EMPTY)
     c. If any empty slot:
          - Pick first empty slot
          - Write (key, value) into bucket
          - Set metadata byte to `reduced`
          - Increment count
          - If count > max_load → rehash
          - Return
     d. Set overflow bit: ofw |= (1 << (h % 8))
     e. Advance to next group (quadratic probing)
```

### Deletion (tombstone-free)

```
1. Find element via lookup
2. Set metadata byte to EMPTY (0x00)
3. Decrement count
4. If the overflow bit for (h%8) in the element's *initial* group is set:
     - Decrement max_load by 1 (anti-drift)
     - This triggers earlier rehashing to clear stale overflow bits
```

### Rehash / Growth

- Allocate new arrays with `2 × num_groups`
- Re-insert all elements (re-hash with new group count)
- Reset max_load = floor(new_capacity × 0.875)

## Quadratic Probing

Group probing uses a triangular-number sequence:

```
probe(i) = (group_index + i*(i+1)/2) % num_groups
```

This visits every group when `num_groups` is a power of 2.

## SIMD Strategy

### x86_64 (SSE2 — available on all x86_64)

- `_mm_loadu_si128` — load 16-byte metadata word
- `_mm_cmpeq_epi8` — compare all 16 bytes at once
- `_mm_movemask_epi8` — extract comparison result as bitmask

### aarch64 (NEON)

- `vld1q_u8` — load 16 bytes
- `vceqq_u8` — compare
- Bitmask extraction via shift+narrow sequence

### Fallback

Portable scalar fallback: iterate over 15 bytes with a simple loop.

## Public API

### `UnorderedFlatMap<K, V, S>`

- `new()`, `with_capacity()`, `with_hasher()`, `with_capacity_and_hasher()`
- `insert(k, v) -> Option<V>`
- `get(&k) -> Option<&V>`, `get_mut(&k) -> Option<&mut V>`
- `remove(&k) -> Option<V>`
- `contains_key(&k) -> bool`
- `len()`, `is_empty()`, `capacity()`
- `clear()`
- `iter()`, `iter_mut()`, `into_iter()`
- `keys()`, `values()`, `values_mut()`
- `entry(k) -> Entry<K, V>`
- `Index` trait for `map[&key]`
- `FromIterator`, `Extend`, `Debug`, `Clone`, `PartialEq`, `Eq`

### `UnorderedFlatSet<T, S>`

Thin wrapper around `UnorderedFlatMap<T, (), S>`:

- `new()`, `with_capacity()`, `with_hasher()`, `with_capacity_and_hasher()`
- `insert(v) -> bool`
- `contains(&v) -> bool`
- `remove(&v) -> bool`
- `len()`, `is_empty()`, `capacity()`
- `clear()`
- `iter()`, `into_iter()`
- `FromIterator`, `Extend`, `Debug`, `Clone`, `PartialEq`, `Eq`
- Set operations: `union`, `intersection`, `difference`, `symmetric_difference`
- `is_subset`, `is_superset`, `is_disjoint`

## File Structure

```
src/
  lib.rs              — public re-exports
  raw/
    mod.rs            — RawTable (core hash table engine)
    group.rs          — Group metadata + SIMD operations
    hash.rs           — Hash mixing
    bitmask.rs        — Bitmask iterator utilities
  map.rs              — UnorderedFlatMap<K,V,S>
  set.rs              — UnorderedFlatSet<T,S>
```

## Implementation Phases

1. **Phase 0** — Project scaffold (flake.nix, Cargo.toml, lib.rs)
2. **Phase 1** — SIMD group operations (group.rs, bitmask.rs)
3. **Phase 2** — Hash mixing (hash.rs)
4. **Phase 3** — RawTable core (insert, find, remove, rehash)
5. **Phase 4** — UnorderedFlatMap public API
6. **Phase 5** — UnorderedFlatSet wrapper
7. **Phase 6** — Iterator implementations
8. **Phase 7** — Entry API
9. **Phase 8** — Trait implementations (FromIterator, Extend, etc.)
10. **Phase 9** — Testing & benchmarks
