# OptiMap

Status: **Experiment**

A Rust library providing a **matrix of SIMD-accelerated hash map implementations** with different performance trade-offs, all benchmarked against hashbrown (Rust's std HashMap) and std::BTreeMap.

Individual designs are composed from a small set of orthogonal parameters — tag strategy, overflow strategy, group width, indexing method, and deletion model. Together they explore a design space: tombstone-free O(1) miss termination vs wider tags for fewer false matches; 8-bit overflow channels vs 1-bit vs embedded; 15-slot vs 16-slot vs 32-slot vs 64-slot groups; shift-based vs AND-based group indexing.

The library provides a single entrypoint `OptiMap` wrapper type as well. It is a smart wrapper around the individual hash map implementations, providing a consistent interface and policy engine for backend selection. There is some measured overhead from the dispatch logic due to the enum variant dispatch overhead, but it's small and usually easy for branch prediction to mitigate.

## On AI

This library is heavily AI-generated. The vast majority of container code is pretty formulaic so it should be fine. **The exception is the hot path for the core operations.**

High-level design and access strategies were all human-written. The core SIMD-accelerated paths were also reviewed/tweaked/optimized by hand as well, though I'm sure I've missed some opportunities for further SIMD optimization.

Correctness is ensured through fuzzing, benchmarks, and extensive testing.

## OptiMap: the smart wrapper

`OptiMap<K, V>` wraps all hash backends (plus FlatBTree) behind a 7-variant enum:

```
Ufm | Splitsies | Ipo | Tomb | Gaps | Ipo64 | FlatBTree
```

Every operation dispatches through a `match` — this is the only overhead vs using a raw type directly. The dispatch is ~1–2 ns (a single predicted branch after the inner method inlines). The `OptiMapBench*` wrappers in the benchmark harness measure pinned backends through this dispatch so the overhead is visible in benchmarks.

**Policy engine**: `OptiMap::new()` uses `Policy::auto()`, which picks `MapType::Tomb` (Byte7_128_TombMap) as a single-band default — currently the best all-rounder per sweep-2026-05-13 data. Users can configure multi-band policies that transition backends on resize:

```rust
use optimap::{OptiMap, Policy, MapType};

let policy = Policy::new()
    .band(0, MapType::Splitsies)
    .band(1024, MapType::Tomb);
let mut map = OptiMap::<u64, u64>::with_policy(policy);
```

Pinned constructors (`OptiMap::ipo()`, `OptiMap::tomb()`, etc.) never transition. Sorted containers go through `OptiMap::flat_btree()` or `Hint::Sorted`.

The `HashedMap` trait covers the full `std::HashMap` interface for all hash backends, and `SortedMap` covers ordered operations (first/last, range, pop_first/pop_last). `OptiSet<T>` wraps `OptiMap<T, ()>` with set-specific API.

## Usage

```rust
use optimap::{OptiMap, HashedMap};

// Let the policy choose (defaults to Byte7_128_TombMap)
let mut map = OptiMap::new();
map.insert("key", 42);
assert_eq!(map.get(&"key"), Some(&42));

// Pin a specific backend
let mut map = OptiMap::ipo();       // InPlaceOverflow (254-value tag)
let mut map = OptiMap::tomb();      // Byte7_128_TombMap (128-value tag)
let mut map = OptiMap::splitsies(); // Splitsies
let mut map = OptiMap::flat_btree();// FlatBTree (sorted)

// Workload hints
use optimap::Hint;
let mut map = OptiMap::<u64, u64>::with_hint(Hint::Churn);

// Raw types (no dispatch overhead)
use optimap::{UnorderedFlatMap, Splitsies, InPlaceOverflow, Gaps, SoaMap, FlatBTree};
let mut map = Splitsies::new();
map.insert(1, "one");
```

## Named designs

These are the "brand" types exported at the crate root. They are representative of each design family and used in the default benchmark tables.

Yes the naming is a mess, I'm sorry.

**Byte7_128_TombMap** tends to be the best overall performing and is used by default under the name "Tomb". It is the exact same design as hashbrown/swisstables.

| Design | Family | Groups | Deletion | Best at |
| --- | --- | --- | --- | --- |
| **Byte7_128_TombMap** | Tombstone | 16-slot | Tombstone (128-value tag) | Hit + remove (default) |
| **InPlaceOverflow** | Tombstone | 16-slot | Tombstone (254-value tag) | Lookup hit, insert |
| **UnorderedFlatMap** | Overflow-bit | 15-slot | Tombstone-free (embedded) | High-load miss, churn |
| **Splitsies** | Overflow-bit | 16-slot | Tombstone-free (separate) | Balanced miss+insert |
| **IPO64** | Tombstone | 64-slot (AVX-512) | Tombstone (254-value tag) | High-load resilience |
| **Gaps** | Overflow-bit | 15-slot + power-of-2 stride | Tombstone-free (embedded) | Iteration |
| **SoaMap** | Overflow-bit | 16-slot SoA | Tombstone-free | Large-value workloads |
| **FlatBTree** | Tree | 256-byte B+ tree nodes | — | Sorted iteration, range queries |

### What the names mean

Names encode the parameterization:

- **ByteN_VVV** — tag sourced from byte `N` (0 = low byte, 7 = high byte) with `VVV` distinct values. `Byte7_254` = top byte, 254 values (1/254 false-match rate). `Byte7_128` = top byte, 128 values (hashbrown-equivalent; 1/128 rate but a one-instruction tag extraction). `Byte0_254` = bottom byte, 254 values (used by IPO64 with shift indexing).

- **Overflow strategy** — how displaced entries are tracked:
  - `_8bit` = 8 independent overflow channels (`ByteSeparate`)
  - `_1bit` = single shared overflow bit (`BitSeparate`)
  - `_Emb` = embedded overflow byte (UFM/Gaps style, byte 15 of the group)
  - `_EmbP2` = embedded overflow with power-of-2 bucket stride (Gaps style)

- **Indexing** — how the hash maps to a group:
  - _(no suffix)_ = shift-based (`h >> shift`), uses top hash bits
  - `And` suffix = AND-based (`h & mask`), uses bottom hash bits — 1 instruction vs 2; requires tags from top bits (`Byte7_*`)

- **Width** — SIMD group slot count:
  - _(no suffix)_ = 16-slot (SSE2)
  - `32` suffix = 32-slot (AVX2), `64` suffix = 64-slot (AVX-512)

- **Ch** suffix (e.g. `Byte7_128Ch`) = shifted channel strategy — overflow channels sourced from bits disjoint from the AND index bits, needed for 8-bit overflow under AND indexing.

- **Tomb** suffix = tombstone-family table (IPO/IPO64 family, `TombstoneTag` trait instead of the overflow-bit `TagStrategy`).

## The design matrix (parameter space)

Designs are not hand-coded — they're composed from traits:

| Axis | Trait | Implementations |
| --- | --- | --- |
| Tag extraction | `TagStrategy` / `TombstoneTag` | `Byte0_255`, `Byte0_128`, `Byte0_254`, `Byte1_255`, `Byte7_128`, `Byte7_255`, `Byte7_254`, `Byte7_128Ch`, `Byte7_255Ch` |
| Overflow storage | `OverflowStrategy` | `ByteSeparate` (8-channel), `BitSeparate` (1-bit), `UfmEmbedded` (byte 15) |
| Group indexing | `GroupLayout::AND_INDEX` | Shift-based (`h >> shift`, default) or AND-based (`h & mask`) |
| Group width | `GroupOps` | 15-slot, 16-slot, 32-slot, 64-slot |
| Load factor | `GroupLayout::LOAD_FACTOR_NUM/DEN` | Default 7/8 (87.5%), customizable |

New variants are ~30 lines: a type alias composing these traits. The `matrix_types` module in `src/lib.rs` holds experimental combinations (~70 type aliases). The sweep benchmark's `--design-set all` exercises every one of them.

## Default benchmark tables

The project uses two tiers of benchmark tables:

**Sweep benchmarks** (`cargo bench --bench sweep`) — continuous N-curves from 100 to 10M entries, measuring per-operation latency as N grows. The default `--design-set curated` picks 12 representative designs: one or two best-of-cluster per design family, plus hashbrown and OptiMap-Auto. This is the authoritative data for performance comparisons across the full size range.

**Throughput benchmarks** (`cargo bench --bench throughput`) — fixed-size (13K and 107K entries, 70% load) single-operation latency using Criterion. Runs all named designs plus matrix variants (~20 rows), giving precise microbenchmarks at two sizes.

Performance claims in this README reference sweep data at representative N values, not a single cherry-picked size.

## Key architectural properties

- **foldhash** by default (avalanching, fast)
- Overflow-bit designs (UFM, Splitsies, Gaps) are **tombstone-free** — deletion clears the slot and adjusts max_load, giving O(1) miss termination
- Tombstone designs (IPO, IPO64, Tomb) use tombstones like hashbrown, with EMPTY-based probe termination — faster hit/insert when cache-resident, but lookup_miss degrades at DRAM scale (>1M entries) as tombstones accumulate
- Two memory layouts: **mid-pointer** (ctrl between buckets and metadata, eliminates one address computation from the hot path) used by tombstone and 16-slot overflow-bit designs; traditional layout for embedded-overflow designs (UFM, Gaps) where overflow lives inside the group
- 70% default load factor (87.5% of slots filled before resize)
- Raw entry API for interning and custom-equality lookups

## Building

```bash
# Requires Rust nightly (SIMD intrinsics) + Nix flake (direnv auto-activates)
cargo test
cargo bench

# Sweep benchmarks (N-curves → CSV + PNG plots)
./scripts/sweep-bench.sh
```

## License

[AGPL-3.0-or-later](https://www.gnu.org/licenses/agpl-3.0.html), as discussed in [Moral AI Licensing](https://jackson.dev/post/moral-ai-licensing/).
