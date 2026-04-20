# Testing & Fuzzing

OptiMap has three layers of testing: unit/integration tests, property-based
differential tests (proptest), and coverage-guided fuzzing (cargo-fuzz).

## Quick Reference

```bash
cargo test                              # Run everything (unit + proptest)
cargo test --test proptest_hashmap      # Just hash map differential tests
cargo test --test proptest_btree        # Just FlatBTree differential tests
cargo test --test stress                # Just the original stress tests
cargo fuzz list                         # List all fuzz targets
cargo fuzz run fuzz_btree               # Open-ended fuzz run (Ctrl-C to stop)
cargo fuzz run fuzz_btree -- -max_total_time=3600  # Time-bounded (1 hour)

# Miri (undefined behavior detection)
RUSTFLAGS="" MIRIFLAGS="-Zmiri-disable-isolation" cargo miri test
```

## Unit & Integration Tests

Standard Rust tests in `src/` (291 unit tests) and `tests/` (12 stress +
66 set trait + proptest integration tests). These are deterministic, fast,
and run on every `cargo test`.

`tests/stress.rs` compares `UnorderedFlatMap` against `std::HashMap` over
10,000 random operations with a fixed seed. `tests/set_trait.rs` validates
all 8 set types through the `Set` trait interface.

## Property-Based Differential Tests (proptest)

**Files:** `tests/proptest_hashmap.rs`, `tests/proptest_btree.rs`

**Idea:** Generate random sequences of map operations, apply them to both
our implementation and the standard library reference (`HashMap` or
`BTreeMap`), and assert identical results after every operation.

### How It Works

Each test defines an `Op` enum representing map operations:

- **Hash maps** (13 ops): `Insert`, `Remove`, `Get`, `GetKeyValue`, `GetMut`,
  `RemoveEntry`, `ContainsKey`, `Clear`, `Reserve`, `ShrinkToFit`, `Retain`,
  `Drain`, `IterCollect`
- **FlatBTree** (19 ops): all of the above plus `IterSorted`, `FirstKeyValue`,
  `LastKeyValue`, `PopFirst`, `PopLast`, `Range`, `EntryOrInsert`,
  `EntryOrDefault`, `EntryAndModify`

Key types are intentionally small (`u16` for hash maps, `u8` for FlatBTree)
to maximize collisions, overflow probing, tombstone reuse, and node
splits/merges.

### What Gets Checked

After every operation:
- Return values match the reference implementation
- `len()` matches
- At the end of the sequence: full iteration contents match (sorted for hash
  maps, sorted-order for FlatBTree)

### Configuration

Each test runs **500 cases** (configurable in the `proptest_config`). Each
case generates up to 500 operations. Insert/remove/get are weighted heavily;
structural ops (clear, drain, retain) are less frequent.

### Failure Behavior

When proptest finds a failing case, it:

1. **Shrinks** the input automatically — a 300-op failure might reduce to 3 ops
2. **Persists** the minimal seed to `tests/proptest-regressions/<test>/<fn>.txt`
3. **Replays** persisted regressions on every subsequent `cargo test`

The regression files should be committed to git so failures are reproduced
in CI. The directory only appears after the first failure — if you don't see
it, no bugs have been found.

### Running

```bash
# All proptest tests (included in normal cargo test)
cargo test --test proptest_hashmap --test proptest_btree

# Increase case count for deeper local testing
PROPTEST_CASES=5000 cargo test --test proptest_hashmap
```

## Coverage-Guided Fuzzing (cargo-fuzz)

**Directory:** `fuzz/`

**Idea:** Same differential approach as proptest, but driven by libFuzzer's
coverage-guided engine. The fuzzer observes which code paths each input
exercises and mutates inputs to explore new paths. This is much more
effective at finding bugs than random testing, but requires longer runs.

### Fuzz Targets

| Target | Tests | Key Type |
|--------|-------|----------|
| `fuzz_hashmap_ufm` | UnorderedFlatMap vs HashMap | `u16` |
| `fuzz_hashmap_splitsies` | Splitsies vs HashMap | `u16` |
| `fuzz_hashmap_ipo` | InPlaceOverflow vs HashMap | `u16` |
| `fuzz_hashmap_ipo64` | IPO64 (LowByte254) vs HashMap | `u16` |
| `fuzz_hashmap_ipo64_hi128` | IPO64 (HighByte128) vs HashMap | `u16` |
| `fuzz_hashmap_ipo64_top128` | IPO64 (TopByte128) vs HashMap | `u16` |
| `fuzz_hashmap_gaps` | Gaps vs HashMap | `u16` |
| `fuzz_btree` | FlatBTree vs BTreeMap | `u8` |

The op enums use `#[derive(Arbitrary)]` so libFuzzer generates structured
operation sequences directly from raw bytes.

### Running

```bash
# Open-ended (runs until crash or Ctrl-C)
cargo fuzz run fuzz_btree

# Time-bounded
cargo fuzz run fuzz_btree -- -max_total_time=3600    # 1 hour
cargo fuzz run fuzz_btree -- -max_total_time=86400   # 24 hours

# Iteration-bounded
cargo fuzz run fuzz_btree -- -runs=1000000

# Limit input size (default 4096 bytes ≈ ~800 ops)
cargo fuzz run fuzz_btree -- -max_len=8192
```

### Corpus

Each target accumulates a **corpus** of interesting inputs in
`fuzz/corpus/<target>/`. The corpus persists across runs — each new run
starts by replaying the existing corpus, then explores from there. Coverage
only grows over time.

The corpus is gitignored (machine-specific, large). If you want to share
corpus across machines, copy the directory manually.

### When a Crash Is Found

cargo-fuzz writes the failing input to `fuzz/artifacts/<target>/crash-<hash>`.

**Workflow:**

1. **Reproduce** the crash:
   ```bash
   cargo fuzz run fuzz_btree fuzz/artifacts/fuzz_btree/crash-abc123
   ```

2. **Minimize** the input:
   ```bash
   cargo fuzz tmin fuzz_btree fuzz/artifacts/fuzz_btree/crash-abc123
   ```

3. **Debug** — the crash output includes a stack trace with the assertion
   that failed (e.g., `remove mismatch` or `iter_sorted order differs`).
   The minimized input is a raw byte blob; to see the op sequence, you can
   add a `eprintln!("{ops:?}")` to the harness and re-run.

4. **Fix** the bug and write a human-readable regression test (either in
   `tests/stress.rs` or as a standalone unit test with the minimal
   reproducing op sequence).

5. The crash artifact in `fuzz/artifacts/` is gitignored — the permanent
   record is the regression test you write.

### Architecture

```
fuzz/
├── Cargo.toml                          # Separate crate (libfuzzer-sys dep)
└── fuzz_targets/
    ├── hashmap_harness.rs              # Shared: Op enum + differential runner
    ├── btree_harness.rs                # Shared: sorted-map Op enum + runner
    ├── fuzz_hashmap_ufm.rs             # One-liner targets that call the harness
    ├── fuzz_hashmap_splitsies.rs
    ├── fuzz_hashmap_ipo.rs
    ├── fuzz_hashmap_ipo64.rs
    ├── fuzz_hashmap_gaps.rs
    └── fuzz_btree.rs
```

## Miri (Undefined Behavior Detection)

[Miri](https://github.com/rust-lang/miri) is an interpreter for Rust's
MIR that detects undefined behavior: out-of-bounds access, use-after-free,
misaligned pointers, violation of aliasing rules, incorrect deallocation,
and more. It is the gold standard for validating `unsafe` code.

OptiMap has **841 `unsafe` blocks** across 19 files — SIMD group operations,
raw pointer arithmetic in FlatBTree nodes, manual memory layout, and aligned
allocation. Miri validates all of it.

### How It Works

All SIMD intrinsics (`_mm_*`, `_mm256_*`, `_mm512_*`) are gated on
`#[cfg(all(target_arch = "x86_64", not(miri)))]`. Under Miri, scalar
fallback implementations activate via
`#[cfg(any(not(target_arch = "x86_64"), miri))]`. These scalar fallbacks
are functionally identical — they produce the same bitmasks via byte loops
instead of SSE2/AVX2/AVX-512 instructions.

This means Miri tests the same algorithms and memory access patterns as
production, just with scalar group matching instead of SIMD.

### Coverage

All tests pass under Miri with **zero UB in production code**:

| Test suite | Tests | Miri time |
|-----------|-------|-----------|
| Unit tests (`src/`) | 291 | ~31 min |
| Stress tests (`tests/stress.rs`) | 12 | ~46 min |
| Set trait tests (`tests/set_trait.rs`) | 66 | ~24 min |
| Proptest (`tests/proptest_btree.rs`) | 1 (500 cases) | ~hours |

**Designs covered:** all 5 hash maps (UFM, Splitsies, IPO, IPO64, Gaps),
FlatBTree, all smart wrappers (OptiMap, OptiSet, OptiSortedMap,
OptiSortedSet), GenericSet, and the `Map`/`Set`/`SortedMap`/`SortedSet`
trait implementations.

### Bugs Found

One UB was found and fixed: group test helpers allocated memory with
alignment 16 via `std::alloc::alloc_zeroed`, then wrapped it in
`Vec::from_raw_parts` (alignment 1 for `u8`). On drop, the Vec
deallocated with the wrong alignment — undefined behavior per the
allocator contract. Fixed by replacing with a `#[repr(C, align(16))]`
stack-allocated wrapper.

### Running

```bash
# Must clear RUSTFLAGS (target-cpu=native conflicts with Miri interpreter)
# Must disable isolation (proptest needs getcwd)
RUSTFLAGS="" MIRIFLAGS="-Zmiri-disable-isolation" cargo miri test

# Run a specific test suite under Miri
RUSTFLAGS="" cargo miri test --lib              # Unit tests only (~31 min)
RUSTFLAGS="" cargo miri test --test stress      # Stress tests only (~46 min)
```

Note: `RUSTFLAGS=""` is needed because the flake sets `-C target-cpu=native`,
which Miri (as an interpreter) cannot use.

## Strategy

The four testing layers complement each other:

| Layer | Finds | Speed | Reproducer |
|-------|-------|-------|------------|
| Unit/stress tests | Regressions, known edge cases | Milliseconds | Hardcoded |
| Proptest | Logic bugs, with minimal repro | Seconds | Auto-shrunk, persisted |
| cargo-fuzz | Deep bugs in rare code paths | Hours-days | Raw bytes, needs manual tmin |
| Miri | Undefined behavior in `unsafe` code | ~30-60 min | Stack trace to exact UB |

**Recommended workflow:**
- `cargo test` on every change (includes proptest)
- `cargo miri test --lib` after touching any `unsafe` code
- Periodic long fuzz runs (overnight/weekend) on a beefy machine
- When adding a new operation to the `Map` or `SortedMap` trait, add it to
  all three: the Op enums in proptest, the Op enums in fuzz harnesses, and
  a basic unit test
