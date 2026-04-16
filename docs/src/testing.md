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
```

## Unit & Integration Tests

Standard Rust tests in `src/` (208 unit tests) and `tests/stress.rs` (12
integration tests). These are deterministic, fast, and run on every
`cargo test`.

`tests/stress.rs` compares `UnorderedFlatMap` against `std::HashMap` over
10,000 random operations with a fixed seed.

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
| `fuzz_hashmap_ipo64` | IPO64 vs HashMap | `u16` |
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

## Strategy

The three testing layers complement each other:

| Layer | Finds | Speed | Reproducer |
|-------|-------|-------|------------|
| Unit/stress tests | Regressions, known edge cases | Milliseconds | Hardcoded |
| Proptest | Logic bugs, with minimal repro | Seconds | Auto-shrunk, persisted |
| cargo-fuzz | Deep bugs in rare code paths | Hours-days | Raw bytes, needs manual tmin |

**Recommended workflow:**
- `cargo test` on every change (includes proptest)
- Periodic long fuzz runs (overnight/weekend) on a beefy machine
- When adding a new operation to the `Map` or `SortedMap` trait, add it to
  all three: the Op enums in proptest, the Op enums in fuzz harnesses, and
  a basic unit test
