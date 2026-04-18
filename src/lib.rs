//! # OptiMap — Multiple SIMD-accelerated hash map designs
//!
//! OptiMap provides several hash map implementations with different
//! performance trade-offs, all sharing a common [`Map`] trait interface.
//!
//! ## Designs
//!
//! | Design | Groups | Deletion | Best at |
//! |--------|:------:|:--------:|---------|
//! | [`UnorderedFlatMap`] | 15-slot, overflow byte | Tombstone-free | High-load miss, churn |
//! | [`Gaps`] | 15-slot, overflow byte, power-of-2 buckets | Tombstone-free | Iteration |
//! | [`Splitsies`] | 16-slot, separate overflow array | Tombstone-free | Balanced (miss + insert) |
//! | [`InPlaceOverflow`] | 16-slot, no overflow (tombstones) | Tombstone | Lookup hit, insert |
//! | [`IPO64`] | 64-slot cache-line, AVX-512 | Tombstone | High-load resilience |
//!
//! ## Quick start
//!
//! ```
//! use optimap::Splitsies;
//!
//! let mut map = Splitsies::new();
//! map.insert("hello", 42);
//! assert_eq!(map.get("hello"), Some(&42));
//! ```
//!
//! ## Generic code via the Map trait
//!
//! ```
//! use optimap::{Map, InPlaceOverflow};
//!
//! fn count_words<M: Map<String, usize>>(map: &mut M, words: &[&str]) {
//!     for &word in words {
//!         let key = word.to_string();
//!         let count = map.get(&key).copied().unwrap_or(0);
//!         map.insert(key, count + 1);
//!     }
//! }
//!
//! let mut map = InPlaceOverflow::new();
//! count_words(&mut map, &["the", "cat", "sat", "on", "the", "mat"]);
//! assert_eq!(map.get("the"), Some(&2));
//! ```
//!
//! ## Sets
//!
//! Each map design has a corresponding set type, and all implement the [`Set`] trait:
//!
//! ```
//! use optimap::SplitsiesSet;
//!
//! let mut set = SplitsiesSet::new();
//! set.insert("hello");
//! set.insert("world");
//! assert!(set.contains("hello"));
//! assert_eq!(set.len(), 2);
//! ```
//!
//! Generic code over sets works just like maps:
//!
//! ```
//! use optimap::Set;
//!
//! fn has_duplicates<S: Set<i32>>(items: &[i32]) -> bool {
//!     let mut seen = S::new();
//!     items.iter().any(|&x| !seen.insert(x))
//! }
//! ```
//!
//! ## Smart wrappers
//!
//! [`OptiMap`] dynamically selects a hash map backend based on capacity,
//! key/value size, and optional workload [`Hint`]s. [`OptiSet`] does the
//! same for sets. Both can transition backends at resize boundaries:
//!
//! ```
//! use optimap::{OptiMap, OptiSet, Hint};
//!
//! // Let the policy engine choose:
//! let mut map = OptiMap::<String, i32>::new();
//! map.insert("hello".into(), 42);
//!
//! let mut set = OptiSet::<u64>::new();
//! set.insert(42);
//!
//! // Or hint at your workload:
//! let mut map = OptiMap::<u64, u64>::with_hint(Hint::Churn);
//! ```
//!
//! For sorted containers, [`OptiSortedMap`] and [`OptiSortedSet`] wrap
//! [`FlatBTree`] with sorted iteration, range queries, and first/last access:
//!
//! ```
//! use optimap::{OptiSortedMap, OptiSortedSet};
//!
//! let mut map = OptiSortedMap::new();
//! map.insert(3, "three");
//! map.insert(1, "one");
//! let keys: Vec<_> = map.iter_sorted().map(|(k, _)| *k).collect();
//! assert_eq!(keys, vec![1, 3]);
//!
//! let mut set: OptiSortedSet<i32> = [3, 1, 2].into_iter().collect();
//! assert_eq!(set.first(), Some(&1));
//! ```
//!
//! ## Choosing a design
//!
//! - **Let OptiMap decide**: [`OptiMap`] / [`OptiSet`] — auto-selects backend, good default
//! - **Sorted**: [`OptiSortedMap`] / [`OptiSortedSet`] — sorted iteration, range queries
//! - **General purpose**: [`InPlaceOverflow`] — closest to hashbrown, best
//!   lookup hit, fastest insert
//! - **Delete-heavy / churn**: [`Splitsies`] — tombstone-free deletion,
//!   O(1) miss termination, flat performance at high load
//! - **Maximum compatibility**: [`UnorderedFlatMap`] — original Boost-inspired design

#![allow(clippy::manual_div_ceil)]
#![allow(dead_code)]

// ── reduced_hash implementation (feature-gated) ───────────────────────────

/// Shared `reduced_hash` implementation for overflow-bit designs (UFM, Gaps, Splitsies).
/// Maps the low byte of a hash to [1, 255], avoiding 0x00 (EMPTY sentinel).
///
/// Three variants selectable via crate features:
/// - **default**: `low | (low == 0) as u8` — 3 instructions (`test; sete; or`), 255 distinct values
/// - **`reduced-hash-asm`**: `cmp 0xFF; adc 0` — 2 instructions, 255 values (x86_64 only, falls back to default)
/// - **`reduced-hash-128`**: `low | 1` — 1 instruction, 128 values (higher false-match rate)
#[inline(always)]
pub(crate) fn reduced_hash_impl(h: u64) -> u8 {
    #[cfg(feature = "reduced-hash-128")]
    {
        // 1 instruction, 128 distinct values. Forces bit 0, collapsing even/odd pairs.
        // False-match rate per slot: 1/128 (0.78%) vs 1/255 (0.39%) for 255-value variants.
        (h as u8) | 1
    }
    #[cfg(all(
        feature = "reduced-hash-asm",
        not(feature = "reduced-hash-128"),
        target_arch = "x86_64",
        not(miri),
    ))]
    {
        // 2 instructions (`cmp; adc`), 255 distinct values, no cmov.
        // Saturating add: 0→1, 1→2, ..., 254→255, 255→255.
        // LLVM won't emit this from safe Rust (generates 4-instruction cmov sequence instead).
        let result: u8;
        unsafe {
            core::arch::asm!(
                "cmp {h}, 0xFF",
                "adc {h}, 0",
                h = inout(reg_byte) (h as u8) => result,
            );
        }
        result
    }
    #[cfg(not(any(
        feature = "reduced-hash-128",
        all(
            feature = "reduced-hash-asm",
            target_arch = "x86_64",
            not(miri),
        ),
    )))]
    {
        // 3 instructions (`test; sete; or`), 255 distinct values, no cmov.
        // Maps 0→1, everything else unchanged. Collision pair: {0,1}→1.
        let low = (h & 0xFF) as u8;
        low | (low == 0) as u8
    }
}

pub mod flat_btree;
pub mod gaps;
mod generic_set;
pub mod in_place_overflow;
pub mod ipo64;
pub(crate) mod map;
mod opti_set;
mod opti_sorted;
pub mod optimap;
mod raw;
mod set;
pub mod split_overflow;
mod traits;

// ── Map types ───────────────────────────────────────────────────────────────

pub use flat_btree::FlatBTree;
pub use gaps::Gaps;
pub use in_place_overflow::InPlaceOverflow;
pub use ipo64::IPO64;
pub use map::UnorderedFlatMap;
pub use split_overflow::Splitsies;

// ── Smart wrapper ──────────────────────────────────────────────────────────

pub use optimap::OptiMap;
pub use optimap::Hint;
pub use optimap::MapType;
pub use optimap::Entry;
pub use optimap::OccupiedEntry;
pub use optimap::VacantEntry;
pub use opti_set::OptiSet;
pub use opti_sorted::OptiSortedMap;
pub use opti_sorted::OptiSortedSet;

// ── Set types ───────────────────────────────────────────────────────────────

/// The original UFM set (tightly coupled to UnorderedFlatMap internals).
pub use set::UnorderedFlatSet;

/// Generic set wrapper — works with any Map implementation.
pub use generic_set::{FlatBTreeSet, GapsSet, GenericSet, Ipo64Set, IpoSet, SplitsiesSet, UfmSet};

// ── Traits ──────────────────────────────────────────────────────────────────

pub use raw::hash::IsAvalanching;
pub use traits::Map;
pub use traits::OccupiedError;
pub use traits::Set;
pub use traits::SortedMap;
pub use traits::SortedSet;
