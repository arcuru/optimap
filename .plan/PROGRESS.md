# Progress Tracker

## Phase 0 — Project Scaffold
- [x] flake.nix with devshell (Rust nightly for SIMD intrinsics)
- [x] Cargo.toml
- [x] src/lib.rs skeleton
- [x] Initial commit

## Phase 1 — SIMD Group Operations
- [x] src/raw/group.rs — Group struct, SIMD match_byte, match_empty, is_full
- [x] src/raw/bitmask.rs — BitMask iterator over SIMD comparison results
- [x] Tests for group operations

## Phase 2 — Hash Mixing
- [x] src/raw/hash.rs — 64-bit xmx mixer
- [x] Tests for hash mixing

## Phase 3 — RawTable Core
- [x] src/raw/mod.rs — RawTable struct, memory layout
- [x] find() — SIMD-accelerated lookup with overflow-bit-only termination
- [x] insert() — with overflow bit setting
- [x] remove() — tombstone-free with anti-drift
- [x] rehash/resize via rehash_with()
- [x] Tests for core operations
- [x] Bug fix: lookup termination uses overflow bit only (not fullness check)
       to handle post-deletion groups correctly

## Phase 4 — UnorderedFlatMap
- [x] src/map.rs — full public API with proper replacement semantics
- [x] Tests (basic ops, insert/replace, remove, entry API, iterators, traits)

## Phase 5 — UnorderedFlatSet
- [x] src/set.rs — wrapper API + set operations
- [x] Tests (basic ops, set operations, iterators, traits)

## Phase 6 — Iterators
- [x] Iter, IterMut, IntoIter for map
- [x] Keys, Values, ValuesMut
- [x] SetIter, SetIntoIter for set

## Phase 7 — Entry API
- [x] Entry, OccupiedEntry, VacantEntry
- [x] Tests

## Phase 8 — Trait Implementations
- [x] FromIterator, Extend, IntoIterator
- [x] Index, Debug, Clone, PartialEq, Eq, Default
- [x] Tests

## Phase 9 — Additional Testing
- [ ] Property-based / randomized stress tests
- [ ] Edge cases & adversarial inputs
