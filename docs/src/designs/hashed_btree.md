# HashedBTree _(proposal — not yet implemented)_

> **Status:** Design sketch. Not on the build path. Read this to decide whether the data structure is worth building; nothing in `src/` references it yet.

A B-tree whose nodes are sorted by `hash(K)` rather than by `K`. Implements [`HashedMap`](../../../src/traits.rs), **not** `SortedMap` — the user-visible key order is the hash order, which is meaningless to a caller, so range queries and sorted iteration are deliberately _not_ offered.

## Why

Optimap's hard invariant is:

> No type implements both `HashedMap` and `SortedMap`. A data structure stores keys in one order and is efficient at one dispatch.

The trade-off space today has only two points: hash-table designs that implement `HashedMap` (UFM, Splitsies, IPO, IPO64, Gaps, SoA variants), and the B+ tree (`FlatBTree`) that implements `SortedMap`. Real workloads sometimes want hash-map semantics in a tree-shaped data structure — for **bounded worst case per op** (hash tables have O(rehash) cliffs and high load tails) and **predictable memory growth** (no doubling-and-rehashing).

`HashedBTree` fills that gap without violating the invariant: it stores keys in _hash_ order, so it has a tree's complexity guarantees but a hash map's dispatch flavour.

## High-level structure

Identical to `FlatBTree`, with two changes:

1. **Comparison key is `hash(K)`, not `K`.** Each leaf slot holds the precomputed `u64` hash alongside the `(K, V)` pair: `[hash: u64][key: K][value: V]`.
2. **Equal-hash siblings**: when two keys hash to the same value, they live adjacent in the leaf. Lookup walks the equal-hash run with `K: Eq`.

Internal nodes hold separator hashes (`u64`, fixed size, no `K` dependence on internal layout) plus child pointers. Leaf chain remains doubly-linked, but the chain order is meaningless to callers — it's only there to make leaf splits cheap.

## API surface (`HashedMap`)

All hash-dispatch ops, identical signatures to today's hash maps:

```rust
fn get<Q>(&self, q: &Q) -> Option<&V> where K: Borrow<Q>, Q: Hash + Eq;
fn insert(&mut self, k: K, v: V) -> Option<V>;
fn remove<Q>(&mut self, q: &Q) -> Option<V>;
fn entry(&mut self, k: K) -> Entry<'_, K, V>;
fn iter(&self) -> /* unspecified order */;
// ... etc.
```

Notably absent (would violate the invariant): `first_key_value`, `pop_first/last`, `range`, `iter_sorted`, `split_off`, `append` — anything that implies a meaningful key order. `iter()` is unordered, like a hash map.

Raw entry API ([`raw_entry`]) ports cleanly — the lookup primitive is "hash + equality closure", which is exactly what `HashedBTree` does anyway.

## Complexity

| Op                     | Avg      | Worst                               |
| ---------------------- | -------- | ----------------------------------- |
| `get` / `contains_key` | O(log n) | O(log n + run-length-of-equal-hash) |
| `insert`               | O(log n) | O(log n + split cascade)            |
| `remove`               | O(log n) | O(log n + rebalance cascade)        |
| `iter` (full)          | O(n)     | O(n)                                |

No O(rehash) doubling spike. Memory grows in arena chunks (one alloc per new node), so the live-bytes curve is smooth rather than sawtooth.

Equal-hash run length is bounded by `n / 2^64` × birthday-paradox slop, so in practice the worst case is "a handful of slots" for any realistic table size. With a good hasher (foldhash by default), equal hashes are rare even at extreme N; under an adversarial hasher the run length degrades to O(n) just like any other hash structure.

## Node layout

Same arena machinery as `FlatBTree` — `[u8; 256]` nodes, `u32` NodeIdx, parent pointers in the header. Diffs from `FlatBTree`:

**Leaf node (256 B):**

```text
[Header 8B][hashes: u64 × LEAF_CAP][keys: K × LEAF_CAP][values: V × LEAF_CAP][prev: u32][next: u32]
```

The hashes array gets its own region so the search loop touches one cache line of hashes before pulling in keys/values. At `size_of::<K>() = 8` and `size_of::<V>() = 8`, `LEAF_CAP` shrinks vs `FlatBTree` because the 8 B hash per slot is overhead — call it `LEAF_CAP_HB ≈ LEAF_CAP / 1.5` for u64/u64. For larger K (e.g. String, 24 B), the hash is a much smaller fraction and the gap narrows.

**Internal node (256 B):**

```text
[Header 8B][hashes: u64 × INTERNAL_CAP][children: u32 × (INTERNAL_CAP + 1)]
```

Hash-only separators are _strictly cheaper_ than `FlatBTree`'s `K` separators when `size_of::<K>() > 8` — internal fan-out for `K: String` jumps from 8 to ~20.

## Search

```text
1. h = hash(k)
2. Walk from root, binary-search hashes at each internal level (or linear
   for height < 3, mirroring FlatBTree's hybrid policy).
3. In the leaf, find the first slot with hash == h (binary search).
4. Linearly scan adjacent slots while their hash == h, comparing keys via Eq.
5. Return the matching slot or None.
```

Insert is the same descent + B-tree split-on-full. Equal-hash runs that overflow a leaf split arbitrarily — there's no need to keep them in the same leaf, since `get` walks the chain across leaf boundaries when consecutive slots share a hash.

## Why this is not a hash map

It deliberately doesn't have:

- **Tags / SIMD probing** — the structure is a tree, not a probe loop.
- **Load factor / rehash** — no fixed array to fill or grow.
- **Cache-line group ops** — node size is set by tree fan-out, not SIMD.

It deliberately doesn't have, vs `FlatBTree`:

- **Range queries / sorted iteration** — hash order is meaningless.

The only thing it shares with `FlatBTree` is the arena + node layout machinery; the only thing it shares with the hash maps is the `HashedMap` trait dispatch.

## When you'd reach for it

- You need bounded worst-case `insert` cost (no rehash spike).
- You want predictable memory: each insert costs O(1) arena slots amortised, no O(n) reallocate-and-rebuild.
- You don't need iteration order.
- Your keys are large enough that the per-slot u64 hash overhead is a small fraction (`K: String`, `Box<[u8]>`, large structs).

For u64/u64 workloads, the existing hash maps will be faster on the hot path — linear probe in a SIMD group is ~4 ns; HashedBTree's tree descent is ~log₂(n) cache-miss-bounded ns. The win shows up for large-K workloads and for workloads sensitive to latency tails.

## Implementation plan (when / if built)

This factoring lets `HashedBTree` reuse most of `FlatBTree`'s code:

1. **Generalise `flat_btree::raw::RawBTree` to a comparator trait.** Today the inherent methods take `&self` and call `<K as Ord>::cmp` inline; lift that to `C: Comparator<K>` (where `Comparator` is private to the crate) and pass a `KeyCmp` (current behaviour) or `HashCmp` (new) at the type level.
2. **Add `HashedBTreeLeaf` / `HashedBTreeInternal` node layouts** with the hash array, parameterised by `C`.
3. **Expose `HashedBTree<K, V, S>`** as a fresh type alias / wrapper identical to `FlatBTree` minus the sorted-only API. Implements only `HashedMap`.
4. **Bench against IPO64 / Splitsies** at the workloads where hash-map tail latency matters: large K/V, high churn, p99 lookup hit.

Step 1 is the big one — it's a generalisation of the entire tree codebase. Estimate: 1-2 weeks of focused work, mostly in `flat_btree/raw.rs`. Subsequent steps are largely composition.

## Open questions

- Does the per-slot 8B hash overhead disqualify this for small-K workloads? At u64/u64, the existing hash maps will likely win on lookup hit by ~3×. Worth measuring on the workloads that motivate the design.
- Does the `OptiMap` policy engine want to route to `HashedBTree` automatically? Probably not by default — it'd require new `Hint` variants (`Hint::TailLatency`?) and a load-factor-independent threshold.
- Is `&raw const`/`&raw mut` discipline in `flat_btree::raw` already strict enough that `HashedBTree`'s `raw_entry` would land for free? (Yes, modulo the comparator refactor.)
