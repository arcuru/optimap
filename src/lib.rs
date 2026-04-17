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
//! ## Smart wrapper
//!
//! [`OptiMap`] dynamically selects a backend based on capacity, key/value
//! size, and optional workload [`Hint`]s. It implements [`Map`] and can
//! transition backends at resize boundaries:
//!
//! ```
//! use optimap::{OptiMap, Hint};
//!
//! // Let the policy engine choose:
//! let mut map = OptiMap::<String, i32>::new();
//! map.insert("hello".into(), 42);
//!
//! // Or hint at your workload:
//! let mut map = OptiMap::<u64, u64>::with_hint(Hint::Churn);
//! ```
//!
//! ## Choosing a design
//!
//! - **Let OptiMap decide**: [`OptiMap`] — auto-selects backend, good default
//! - **General purpose**: [`InPlaceOverflow`] — closest to hashbrown, best
//!   lookup hit, fastest insert
//! - **Delete-heavy / churn**: [`Splitsies`] — tombstone-free deletion,
//!   O(1) miss termination, flat performance at high load
//! - **Maximum compatibility**: [`UnorderedFlatMap`] — original Boost-inspired design

#![allow(clippy::manual_div_ceil)]
#![allow(dead_code)]

pub mod flat_btree;
pub mod gaps;
mod generic_set;
pub mod in_place_overflow;
pub mod ipo64;
mod map;
mod raw;
mod set;
pub mod split_overflow;
pub mod optimap;
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

// ── Set types ───────────────────────────────────────────────────────────────

/// The original UFM set (tightly coupled to UnorderedFlatMap internals).
pub use set::UnorderedFlatSet;

/// Generic set wrapper — works with any Map implementation.
pub use generic_set::{FlatBTreeSet, GapsSet, GenericSet, Ipo64Set, IpoSet, SplitsiesSet, UfmSet};

// ── Traits ──────────────────────────────────────────────────────────────────

pub use raw::hash::IsAvalanching;
pub use traits::Map;
pub use traits::Set;
pub use traits::SortedMap;
pub use traits::SortedSet;
