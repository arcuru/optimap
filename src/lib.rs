//! # OptiMap — Multiple SIMD-accelerated hash map designs
//!
//! OptiMap provides several hash map implementations with different
//! performance trade-offs, all sharing a common [`HashedMap`] trait interface.
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
//! ## Generic code via the HashedMap trait
//!
//! ```
//! use optimap::{HashedMap, InPlaceOverflow};
//!
//! fn count_words<M: HashedMap<String, usize>>(map: &mut M, words: &[&str]) {
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
//! [`OptiMap`] dynamically selects a hash map backend per its [`Policy`].
//! [`OptiSet`] does the same for sets. Both transition backends on resize
//! (organic growth via `insert` or explicit `reserve`) if the new capacity
//! falls into a different policy band.
//!
//! ```
//! use optimap::{OptiMap, OptiSet, Hint, Policy, MapType};
//!
//! // Default policy (single-band Tomb — see Policy::auto docs):
//! let mut map = OptiMap::<String, i32>::new();
//! map.insert("hello".into(), 42);
//!
//! let mut set = OptiSet::<u64>::new();
//! set.insert(42);
//!
//! // Or hint at your workload:
//! let mut map = OptiMap::<u64, u64>::with_hint(Hint::Churn);
//!
//! // Or define a custom policy with capacity bands:
//! let policy = Policy::new()
//!     .band(0, MapType::Splitsies)
//!     .band(1024, MapType::Tomb);
//! let mut map = OptiMap::<u64, u64>::with_policy(policy);
//! ```
//!
//! For sorted containers, pin [`OptiMap`] / [`OptiSet`] to the [`FlatBTree`]
//! backend via [`OptiMap::flat_btree()`] / [`OptiSet::flat_btree()`] (or
//! [`Hint::Sorted`] through the auto-policy):
//!
//! ```
//! use optimap::{OptiMap, OptiSet};
//!
//! let mut map = OptiMap::flat_btree();
//! map.insert(3, "three");
//! map.insert(1, "one");
//! let keys: Vec<_> = map.iter_sorted().map(|(k, _)| *k).collect();
//! assert_eq!(keys, vec![1, 3]);
//!
//! let set: OptiSet<i32> = OptiSet::from_sorted_iter([1, 2, 3]);
//! assert_eq!(set.first(), Some(&1));
//! ```
//!
//! The full sorted API — `first_key_value`, `pop_first`, `range`,
//! `range_mut`, `split_off`, `append`, etc. — is available on
//! [`OptiMap`] / [`OptiSet`] when the backend is `FlatBTree`. Calling
//! these on a hash backend panics; pin the backend (or use
//! [`Hint::Sorted`]) to guarantee them.
//!
//! ## Raw entry API (interning, custom equality)
//!
//! Every hash-backed map exposes [`raw_entry`] / [`raw_entry_mut`] for cases
//! the regular `Entry` API can't express — look up by hash + custom equality,
//! supply a pre-computed hash, intern a borrowed key without allocating until
//! you need to:
//!
//! ```
//! use optimap::Splitsies;
//! use optimap::raw_entry::RawEntryMut;
//! use std::hash::BuildHasher;
//!
//! let mut interner: Splitsies<String, u32> = Splitsies::new();
//! let query: &str = "needle";
//! let h = interner.hasher().hash_one(query);
//!
//! // Look up by &str; only allocate a String when the key is absent.
//! let id = match interner.raw_entry_mut().from_key_hashed_nocheck(h, query) {
//!     RawEntryMut::Occupied(e) => *e.get(),
//!     RawEntryMut::Vacant(e) => {
//!         let next_id = 42;
//!         e.insert_hashed_nocheck(h, query.to_string(), next_id);
//!         next_id
//!     }
//! };
//! assert_eq!(id, 42);
//! ```
//!
//! See [`raw_entry`] for the full builder/entry surface.
//!
//! [`raw_entry`]: crate::raw_entry
//! [`raw_entry_mut`]: crate::raw_entry
//!
//! ## Choosing a design
//!
//! - **Let OptiMap decide**: [`OptiMap`] / [`OptiSet`] — auto-selects backend, good default
//! - **Sorted**: [`OptiMap::flat_btree()`] / [`OptiSet::flat_btree()`] — sorted iteration, range queries
//! - **General purpose**: [`InPlaceOverflow`] — closest to hashbrown, best
//!   lookup hit, fastest insert
//! - **Delete-heavy / churn**: [`Splitsies`] — tombstone-free deletion,
//!   O(1) miss termination, flat performance at high load
//! - **Maximum compatibility**: [`UnorderedFlatMap`] — original Boost-inspired design

#![allow(clippy::manual_div_ceil)]
#![allow(dead_code)]
// The SIMD group ops (GroupOps, Group{,32,64}::match_*, etc.) share a single
// precondition (16/32/64-byte-aligned metadata pointer). Safety docs live at
// the trait / module level rather than per-method.
#![allow(clippy::missing_safety_doc)]

// ── Hash tag extraction (feature-gated) ───────────────────────────────────

/// Extract a non-zero tag byte from a hash value.
///
/// Hash tables that use 0x00 as an EMPTY sentinel need tag values in [1, 255].
/// This function extracts the low byte of a hash and maps it into that range.
///
/// Three implementations are available via crate features, trading off between
/// instruction count and hash discrimination (distinct output values):
///
/// | Feature | Instructions | Distinct values | Notes |
/// |---------|:-----------:|:---------------:|-------|
/// | **`reduced-hash-asm`** (default) | 2 | 255 | Inline asm, x86_64 only |
/// | `reduced-hash-128` | 1 | 128 | Fastest, but doubles false-match rate |
/// | *(neither)* | 3 | 255 | Pure Rust fallback |
///
/// More distinct values = fewer false-positive SIMD matches = fewer wasted key
/// comparisons. At 255 values the false-match rate is 1/255 (0.39% per slot);
/// at 128 values it's 1/128 (0.78%).
///
/// The `reduced-hash-asm` variant also acts as an LLVM optimization barrier that
/// improves instruction scheduling in some probe loops (notably UFM: -26% hit,
/// -41% miss).
#[inline(always)]
pub(crate) fn hash_tag(h: u64) -> u8 {
    #[cfg(feature = "reduced-hash-128")]
    {
        // Force bit 0 high: output is always odd, giving 128 distinct values
        // (1, 3, 5, ..., 255). Collapses even/odd pairs (e.g. 0x10 and 0x11
        // both produce 0x11).
        //
        // x86 assembly: `or al, 1` — 1 instruction.
        (h as u8) | 1
    }
    #[cfg(all(
        feature = "reduced-hash-asm",
        not(feature = "reduced-hash-128"),
        target_arch = "x86_64",
        not(miri),
    ))]
    {
        // Saturating increment via carry flag: 0→1, 1→2, ..., 254→255, 255→255.
        // 255 distinct values; only collision is {254, 255} → 255.
        //
        // x86 assembly (2 instructions, no branch, no cmov):
        //   cmp al, 0xFF   ; sets CF=1 if al < 255 (unsigned comparison)
        //   adc al, 0      ; al = al + 0 + CF
        //                  ;   if al < 255: al = al + 1  (CF was 1)
        //                  ;   if al == 255: al = 255    (CF was 0, no change)
        //
        // Why inline asm: LLVM lowers `u8::saturating_add(1)` to a 4-instruction
        // sequence with cmov (`inc; movzbl; mov $0xFF; cmovne`). It doesn't know
        // the `cmp; adc` idiom for single-byte saturation.
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
        all(feature = "reduced-hash-asm", target_arch = "x86_64", not(miri),),
    )))]
    {
        // Conditional fix-up: 0→1, everything else unchanged.
        // 255 distinct values; only collision is {0, 1} → 1.
        //
        // x86 assembly (3 instructions, no cmov):
        //   test al, al    ; set ZF if al == 0
        //   sete cl        ; cl = 1 if al was 0, else 0
        //   or al, cl      ; al |= cl — sets bit 0 only when al was 0
        //
        // This is the pure Rust fallback, used on non-x86_64 and under Miri.
        let low = (h & 0xFF) as u8;
        low | (low == 0) as u8
    }
}

pub mod flat_btree;
pub mod gaps;
pub mod generic_map;
mod generic_set;
pub mod in_place_overflow;
pub mod ipo64;
pub(crate) mod map;
mod opti_set;
pub mod optimap;
pub mod raw;
pub mod raw_entry;
mod set;
pub mod soa;
pub mod split_overflow;
mod traits;

// ── Map types ───────────────────────────────────────────────────────────────

pub use flat_btree::FlatBTree;
pub use gaps::Gaps;
pub use in_place_overflow::InPlaceOverflow;
pub use ipo64::IPO64;
pub use map::UnorderedFlatMap;
pub use split_overflow::Splitsies;

// ── SoA (Structure-of-Arrays) map types ───────────────────────────────────

pub use soa::SoaMap;

// ── Smart wrapper ──────────────────────────────────────────────────────────

pub use opti_set::OptiSet;
pub use optimap::Entry;
pub use optimap::Hint;
pub use optimap::MapType;
pub use optimap::OccupiedEntry;
pub use optimap::OptiMap;
pub use optimap::Policy;
pub use optimap::VacantEntry;

// ── Set types ───────────────────────────────────────────────────────────────

/// The original UFM set (tightly coupled to UnorderedFlatMap internals).
pub use set::UnorderedFlatSet;

/// Generic set wrapper — works with any HashedMap implementation.
pub use generic_set::{FlatBTreeSet, GapsSet, GenericSet, Ipo64Set, IpoSet, SplitsiesSet, UfmSet};

// ── Design matrix types (experimental) ─────────────────────────────────────

/// Matrix variants for benchmarking different tag × overflow combinations.
#[allow(non_camel_case_types)]
pub mod matrix_types {
    use crate::generic_map::{DefaultHashBuilder, GenericMap};
    use crate::raw::group_layout::{
        Byte0_1bit, Byte0_128_1bit, Byte0_128_1bit32, Byte0_128_1bit64, Byte0_128_8bit,
        Byte0_128_8bit32, Byte0_128_8bit64, Byte0_128_Emb, Byte0_128_Emb32, Byte0_128_Emb64,
        Byte0_128_EmbP2, Byte0_128_EmbP232, Byte0_128_EmbP264, Byte0_255PureLayout,
        Byte1_255PureLayout, Byte1_8bit, Byte1_8bit32, Byte1_8bit64, Byte1_Emb, Byte1_Emb32,
        Byte1_Emb64, Byte1_EmbP2, Byte1_EmbP232, Byte1_EmbP264, Byte7_128_1bitAnd,
        Byte7_128_1bitAnd32, Byte7_128_1bitAnd64, Byte7_128_8bitAnd, Byte7_128_8bitAnd32,
        Byte7_128_8bitAnd64, Byte7_128Ch_EmbAnd, Byte7_128Ch_EmbAnd32, Byte7_128Ch_EmbAnd64,
        Byte7_128Ch_EmbP2And, Byte7_128Ch_EmbP2And32, Byte7_128Ch_EmbP2And64,
        Byte7_255_1bitAnd, Byte7_255_1bitAnd32, Byte7_255_1bitAnd64, Byte7_255_8bitAnd,
        Byte7_255_8bitAnd32, Byte7_255_8bitAnd64, Byte7_255Ch_EmbAnd, Byte7_255Ch_EmbAnd32,
        Byte7_255Ch_EmbAnd64, Byte7_255Ch_EmbP2And, Byte7_255Ch_EmbP2And32,
        Byte7_255Ch_EmbP2And64, Byte7_255Pure_1bitAnd, Byte7_255Pure_1bitAnd32,
        Byte7_255Pure_1bitAnd64, Byte7_255Pure_8bitAnd, Byte7_255Pure_8bitAnd32,
        Byte7_255Pure_8bitAnd64, Gaps32Layout, Gaps64Layout, Splitsies32_1bit,
        Splitsies32Layout, Splitsies64_1bit, Splitsies64Layout, Ufm32Layout, Ufm64Layout,
    };
    use crate::raw::overflow_table::RawTable;
    use crate::raw::tag_strategy::{Byte0_254, Byte2_254, Byte7_128, Byte7_254};

    /// Define a map type alias over `overflow_table::RawTable<K, V, $layout>` and
    /// impl the `HashedMap` trait for it in one go. Keeps matrix entries to one line.
    macro_rules! matrix_map {
        ($map:ident, $layout:ty) => {
            pub type $map<K, V, S = DefaultHashBuilder> =
                GenericMap<K, V, S, RawTable<K, V, $layout>>;
            crate::traits::impl_map_trait!($map);
        };
    }

    /// Same but for the tombstone RawTable variants (IPO + IPO64) which have
    /// their own RawTable type and take a `TombstoneTag` instead of a layout.
    macro_rules! ipo_map {
        ($map:ident, $tag:ty) => {
            pub type $map<K, V, S = DefaultHashBuilder> =
                GenericMap<K, V, S, crate::in_place_overflow::raw::RawTable<K, V, $tag>>;
            crate::traits::impl_map_trait!($map);
        };
    }

    macro_rules! ipo64_map {
        ($map:ident, $tag:ty) => {
            pub type $map<K, V, S = DefaultHashBuilder> =
                GenericMap<K, V, S, crate::ipo64::raw::RawTable<K, V, $tag>>;
            crate::traits::impl_map_trait!($map);
        };
    }

    // Separate-overflow at 16-slot (shift indexed)
    matrix_map!(Byte1_8bitMap, Byte1_8bit);
    matrix_map!(Byte0_128_8bitMap, Byte0_128_8bit);
    matrix_map!(Byte0_1bitMap, Byte0_1bit);
    matrix_map!(Byte0_128_1bitMap, Byte0_128_1bit);

    // Separate-overflow at 16-slot (AND indexed)
    matrix_map!(Byte7_128_1bitAndMap, Byte7_128_1bitAnd);
    matrix_map!(Byte7_255_1bitAndMap, Byte7_255_1bitAnd);
    matrix_map!(Byte7_128_8bitAndMap, Byte7_128_8bitAnd);
    matrix_map!(Byte7_255_8bitAndMap, Byte7_255_8bitAnd);

    // Separate-overflow at 32-slot (AVX2)
    matrix_map!(Splitsies32Map, Splitsies32Layout);
    matrix_map!(Splitsies32_1bitMap, Splitsies32_1bit);
    matrix_map!(Byte1_8bit32Map, Byte1_8bit32);
    matrix_map!(Byte0_128_8bit32Map, Byte0_128_8bit32);
    matrix_map!(Byte0_128_1bit32Map, Byte0_128_1bit32);
    matrix_map!(Byte7_128_1bitAnd32Map, Byte7_128_1bitAnd32);
    matrix_map!(Byte7_255_1bitAnd32Map, Byte7_255_1bitAnd32);
    matrix_map!(Byte7_128_8bitAnd32Map, Byte7_128_8bitAnd32);
    matrix_map!(Byte7_255_8bitAnd32Map, Byte7_255_8bitAnd32);

    // Separate-overflow at 64-slot (AVX-512 / tiered fallback)
    matrix_map!(Splitsies64Map, Splitsies64Layout);
    matrix_map!(Splitsies64_1bitMap, Splitsies64_1bit);
    matrix_map!(Byte1_8bit64Map, Byte1_8bit64);
    matrix_map!(Byte0_128_8bit64Map, Byte0_128_8bit64);
    matrix_map!(Byte0_128_1bit64Map, Byte0_128_1bit64);
    matrix_map!(Byte7_128_1bitAnd64Map, Byte7_128_1bitAnd64);
    matrix_map!(Byte7_255_1bitAnd64Map, Byte7_255_1bitAnd64);
    matrix_map!(Byte7_128_8bitAnd64Map, Byte7_128_8bitAnd64);
    matrix_map!(Byte7_255_8bitAnd64Map, Byte7_255_8bitAnd64);

    // Pure Rust tag variants (always pure, independent of cfg features)
    matrix_map!(Byte7_255Pure_1bitAndMap, Byte7_255Pure_1bitAnd);
    matrix_map!(Byte7_255Pure_8bitAndMap, Byte7_255Pure_8bitAnd);
    matrix_map!(Byte7_255Pure_1bitAnd32Map, Byte7_255Pure_1bitAnd32);
    matrix_map!(Byte7_255Pure_8bitAnd32Map, Byte7_255Pure_8bitAnd32);
    matrix_map!(Byte7_255Pure_1bitAnd64Map, Byte7_255Pure_1bitAnd64);
    matrix_map!(Byte7_255Pure_8bitAnd64Map, Byte7_255Pure_8bitAnd64);
    matrix_map!(Byte0_255PureLayoutMap, Byte0_255PureLayout);
    matrix_map!(Byte1_255PureLayoutMap, Byte1_255PureLayout);

    // Embedded-overflow (UFM/Gaps-style) — byte-0 tag, all three widths
    matrix_map!(Ufm32Map, Ufm32Layout);
    matrix_map!(Gaps32Map, Gaps32Layout);
    matrix_map!(Ufm64Map, Ufm64Layout);
    matrix_map!(Gaps64Map, Gaps64Layout);

    // Embedded-overflow — Byte1 (decorrelated 255 tag, shift indexing)
    matrix_map!(Byte1_EmbMap, Byte1_Emb);
    matrix_map!(Byte1_EmbP2Map, Byte1_EmbP2);
    matrix_map!(Byte1_Emb32Map, Byte1_Emb32);
    matrix_map!(Byte1_EmbP232Map, Byte1_EmbP232);
    matrix_map!(Byte1_Emb64Map, Byte1_Emb64);
    matrix_map!(Byte1_EmbP264Map, Byte1_EmbP264);

    // Embedded-overflow — Byte0_128 (128-value low tag, faster hash_tag, shift)
    matrix_map!(Byte0_128_EmbMap, Byte0_128_Emb);
    matrix_map!(Byte0_128_EmbP2Map, Byte0_128_EmbP2);
    matrix_map!(Byte0_128_Emb32Map, Byte0_128_Emb32);
    matrix_map!(Byte0_128_EmbP232Map, Byte0_128_EmbP232);
    matrix_map!(Byte0_128_Emb64Map, Byte0_128_Emb64);
    matrix_map!(Byte0_128_EmbP264Map, Byte0_128_EmbP264);

    // Embedded-overflow — Byte7_128Ch + AND indexing (first AND-indexed embedded)
    matrix_map!(Byte7_128Ch_EmbAndMap, Byte7_128Ch_EmbAnd);
    matrix_map!(Byte7_128Ch_EmbP2AndMap, Byte7_128Ch_EmbP2And);
    matrix_map!(Byte7_128Ch_EmbAnd32Map, Byte7_128Ch_EmbAnd32);
    matrix_map!(Byte7_128Ch_EmbP2And32Map, Byte7_128Ch_EmbP2And32);
    matrix_map!(Byte7_128Ch_EmbAnd64Map, Byte7_128Ch_EmbAnd64);
    matrix_map!(Byte7_128Ch_EmbP2And64Map, Byte7_128Ch_EmbP2And64);

    // Embedded-overflow — Byte7_255Ch + AND indexing
    matrix_map!(Byte7_255Ch_EmbAndMap, Byte7_255Ch_EmbAnd);
    matrix_map!(Byte7_255Ch_EmbP2AndMap, Byte7_255Ch_EmbP2And);
    matrix_map!(Byte7_255Ch_EmbAnd32Map, Byte7_255Ch_EmbAnd32);
    matrix_map!(Byte7_255Ch_EmbP2And32Map, Byte7_255Ch_EmbP2And32);
    matrix_map!(Byte7_255Ch_EmbAnd64Map, Byte7_255Ch_EmbAnd64);
    matrix_map!(Byte7_255Ch_EmbP2And64Map, Byte7_255Ch_EmbP2And64);

    // Tombstone variants — IPO/IPO64 take TombstoneTag instead of a layout.
    //
    // The IPO entries form an A/B test for the tag/group-index collision fix:
    //   - InPlaceOverflow = current default (Byte7_254, top byte; safe with AND
    //     indexing at any size). Already exported above; not re-aliased here.
    //   - Byte0_254_TombMap = bottom byte; correlates with the AND mask at
    //     any non-trivial size → maximally collision-prone (full-byte overlap
    //     once num_groups ≥ 2^8). Used as the "always-collides" regression case.
    //   - Byte2_254_TombMap = pre-fix default (bits 16-23; correlates with the
    //     AND mask above 2^16 groups → degraded SIMD discrimination).
    //   - Byte7_128_TombMap = consolidated 128-value top-byte alternative.
    //
    // Byte7_254_Tomb64Map is intentionally unsafe — IPO64 uses shift indexing,
    // so byte 7 IS the group index. Used to verify the symmetric collision claim.
    ipo_map!(Byte0_254_TombMap, Byte0_254);
    ipo_map!(Byte2_254_TombMap, Byte2_254);
    ipo_map!(Byte7_128_TombMap, Byte7_128);
    ipo64_map!(Byte7_254_Tomb64Map, Byte7_254);
}

// ── Traits ──────────────────────────────────────────────────────────────────

pub use raw::hash::IsAvalanching;
pub use traits::HashedMap;
pub use traits::Map;
pub use traits::OccupiedError;
pub use traits::Set;
pub use traits::SortedMap;
pub use traits::SortedSet;
