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
        all(
            feature = "reduced-hash-asm",
            target_arch = "x86_64",
            not(miri),
        ),
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
mod opti_sorted;
pub mod optimap;
pub mod raw;
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

// ── Design matrix types (experimental) ─────────────────────────────────────

/// Matrix variants for benchmarking different tag × overflow combinations.
#[allow(non_camel_case_types)]
pub mod matrix_types {
    use crate::generic_map::{DefaultHashBuilder, GenericMap};
    use crate::raw::group_layout::{
        Gaps32Layout, Gaps64Layout,
        Hi8_1bit, Hi8_1bit32, Hi8_1bit64, Hi8_8bit, Hi8_8bit32, Hi8_8bit64,
        Hi8_Emb, Hi8_Emb32, Hi8_Emb64, Hi8_EmbP2, Hi8_EmbP232, Hi8_EmbP264,
        Lo128_1bit, Lo128_1bit32, Lo128_1bit64, Lo128_8bit, Lo128_8bit32, Lo128_8bit64,
        Lo128_Emb, Lo128_Emb32, Lo128_Emb64, Lo128_EmbP2, Lo128_EmbP232, Lo128_EmbP264,
        Lo8_1bit,
        Splitsies32Layout, Splitsies32_1bit, Splitsies64Layout, Splitsies64_1bit,
        Top128_1bitAnd, Top128_1bitAnd32, Top128_1bitAnd64,
        Top128_8bitAnd, Top128_8bitAnd32, Top128_8bitAnd64,
        Top128_EmbAnd, Top128_EmbAnd32, Top128_EmbAnd64,
        Top128_EmbP2And, Top128_EmbP2And32, Top128_EmbP2And64,
        Top255_1bitAnd, Top255_1bitAnd32, Top255_1bitAnd64,
        Top255_8bitAnd, Top255_8bitAnd32, Top255_8bitAnd64,
        Top255_EmbAnd, Top255_EmbAnd32, Top255_EmbAnd64,
        Top255_EmbP2And, Top255_EmbP2And32, Top255_EmbP2And64,
        Ufm32Layout, Ufm64Layout,
    };
    use crate::raw::overflow_table::RawTable;
    use crate::raw::tag_strategy::{HighByte128, TopByte128};

    // Overflow-bit variants
    pub type Hi8_8bitMap<K, V, S = DefaultHashBuilder> = GenericMap<K, V, S, RawTable<K, V, Hi8_8bit>>;
    pub type Lo128_8bitMap<K, V, S = DefaultHashBuilder> = GenericMap<K, V, S, RawTable<K, V, Lo128_8bit>>;
    pub type Lo8_1bitMap<K, V, S = DefaultHashBuilder> = GenericMap<K, V, S, RawTable<K, V, Lo8_1bit>>;
    pub type Hi8_1bitMap<K, V, S = DefaultHashBuilder> = GenericMap<K, V, S, RawTable<K, V, Hi8_1bit>>;
    pub type Lo128_1bitMap<K, V, S = DefaultHashBuilder> = GenericMap<K, V, S, RawTable<K, V, Lo128_1bit>>;

    // AND-indexed overflow variants
    pub type Top128_1bitAndMap<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top128_1bitAnd>>;
    pub type Top255_1bitAndMap<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top255_1bitAnd>>;
    pub type Top128_8bitAndMap<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top128_8bitAnd>>;
    pub type Top255_8bitAndMap<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top255_8bitAnd>>;

    // 32-slot (AVX2) overflow-bit variants
    pub type Splitsies32Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Splitsies32Layout>>;
    pub type Splitsies32_1bitMap<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Splitsies32_1bit>>;
    pub type Hi8_1bit32Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Hi8_1bit32>>;
    pub type Top128_1bitAnd32Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top128_1bitAnd32>>;
    pub type Top255_1bitAnd32Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top255_1bitAnd32>>;
    pub type Top128_8bitAnd32Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top128_8bitAnd32>>;
    pub type Top255_8bitAnd32Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top255_8bitAnd32>>;
    pub type Hi8_8bit32Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Hi8_8bit32>>;
    pub type Lo128_8bit32Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Lo128_8bit32>>;
    pub type Lo128_1bit32Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Lo128_1bit32>>;

    // 32-slot embedded-overflow variants (Ufm32 / Gaps32)
    pub type Ufm32Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Ufm32Layout>>;
    pub type Gaps32Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Gaps32Layout>>;

    // 64-slot (AVX-512 / tiered) overflow-bit variants
    pub type Splitsies64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Splitsies64Layout>>;
    pub type Splitsies64_1bitMap<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Splitsies64_1bit>>;
    pub type Hi8_1bit64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Hi8_1bit64>>;
    pub type Top128_1bitAnd64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top128_1bitAnd64>>;
    pub type Top255_1bitAnd64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top255_1bitAnd64>>;
    pub type Top128_8bitAnd64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top128_8bitAnd64>>;
    pub type Top255_8bitAnd64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top255_8bitAnd64>>;
    pub type Hi8_8bit64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Hi8_8bit64>>;
    pub type Lo128_8bit64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Lo128_8bit64>>;
    pub type Lo128_1bit64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Lo128_1bit64>>;

    // 64-slot embedded-overflow variants (Ufm64 / Gaps64)
    pub type Ufm64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Ufm64Layout>>;
    pub type Gaps64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Gaps64Layout>>;

    // Embedded-overflow matrix (other tags, 3 widths × 2 strides × 2 index modes)
    // — Hi8 (decorrelated 255 tag, shift indexing)
    pub type Hi8_EmbMap<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Hi8_Emb>>;
    pub type Hi8_EmbP2Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Hi8_EmbP2>>;
    pub type Hi8_Emb32Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Hi8_Emb32>>;
    pub type Hi8_EmbP232Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Hi8_EmbP232>>;
    pub type Hi8_Emb64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Hi8_Emb64>>;
    pub type Hi8_EmbP264Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Hi8_EmbP264>>;

    // — Lo128 (128-value low tag, faster hash_tag, shift indexing)
    pub type Lo128_EmbMap<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Lo128_Emb>>;
    pub type Lo128_EmbP2Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Lo128_EmbP2>>;
    pub type Lo128_Emb32Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Lo128_Emb32>>;
    pub type Lo128_EmbP232Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Lo128_EmbP232>>;
    pub type Lo128_Emb64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Lo128_Emb64>>;
    pub type Lo128_EmbP264Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Lo128_EmbP264>>;

    // — Top128Ch + AND-indexed embedded (first AND-indexed embedded variants)
    pub type Top128_EmbAndMap<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top128_EmbAnd>>;
    pub type Top128_EmbP2AndMap<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top128_EmbP2And>>;
    pub type Top128_EmbAnd32Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top128_EmbAnd32>>;
    pub type Top128_EmbP2And32Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top128_EmbP2And32>>;
    pub type Top128_EmbAnd64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top128_EmbAnd64>>;
    pub type Top128_EmbP2And64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top128_EmbP2And64>>;

    // — Top255Ch + AND-indexed embedded
    pub type Top255_EmbAndMap<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top255_EmbAnd>>;
    pub type Top255_EmbP2AndMap<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top255_EmbP2And>>;
    pub type Top255_EmbAnd32Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top255_EmbAnd32>>;
    pub type Top255_EmbP2And32Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top255_EmbP2And32>>;
    pub type Top255_EmbAnd64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top255_EmbAnd64>>;
    pub type Top255_EmbP2And64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, RawTable<K, V, Top255_EmbP2And64>>;

    // Tombstone variants — different tag strategies on IPO infrastructure
    pub type Hi128_TombMap<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, crate::in_place_overflow::raw::RawTable<K, V, HighByte128>>;
    pub type Top128_TombMap<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, crate::in_place_overflow::raw::RawTable<K, V, TopByte128>>;

    // IPO64 tombstone variants — different tag strategies on 64-slot groups
    pub type Hi128_Tomb64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, crate::ipo64::raw::RawTable<K, V, HighByte128>>;
    pub type Top128_Tomb64Map<K, V, S = DefaultHashBuilder> =
        GenericMap<K, V, S, crate::ipo64::raw::RawTable<K, V, TopByte128>>;

    crate::traits::impl_map_trait!(Hi8_8bitMap);
    crate::traits::impl_map_trait!(Lo128_8bitMap);
    crate::traits::impl_map_trait!(Lo8_1bitMap);
    crate::traits::impl_map_trait!(Hi8_1bitMap);
    crate::traits::impl_map_trait!(Lo128_1bitMap);
    crate::traits::impl_map_trait!(Top128_1bitAndMap);
    crate::traits::impl_map_trait!(Top255_1bitAndMap);
    crate::traits::impl_map_trait!(Top128_8bitAndMap);
    crate::traits::impl_map_trait!(Top255_8bitAndMap);
    crate::traits::impl_map_trait!(Splitsies32Map);
    crate::traits::impl_map_trait!(Splitsies32_1bitMap);
    crate::traits::impl_map_trait!(Hi8_1bit32Map);
    crate::traits::impl_map_trait!(Top128_1bitAnd32Map);
    crate::traits::impl_map_trait!(Top255_1bitAnd32Map);
    crate::traits::impl_map_trait!(Top128_8bitAnd32Map);
    crate::traits::impl_map_trait!(Top255_8bitAnd32Map);
    crate::traits::impl_map_trait!(Hi8_8bit32Map);
    crate::traits::impl_map_trait!(Lo128_8bit32Map);
    crate::traits::impl_map_trait!(Lo128_1bit32Map);
    crate::traits::impl_map_trait!(Ufm32Map);
    crate::traits::impl_map_trait!(Gaps32Map);
    crate::traits::impl_map_trait!(Splitsies64Map);
    crate::traits::impl_map_trait!(Splitsies64_1bitMap);
    crate::traits::impl_map_trait!(Hi8_1bit64Map);
    crate::traits::impl_map_trait!(Top128_1bitAnd64Map);
    crate::traits::impl_map_trait!(Top255_1bitAnd64Map);
    crate::traits::impl_map_trait!(Top128_8bitAnd64Map);
    crate::traits::impl_map_trait!(Top255_8bitAnd64Map);
    crate::traits::impl_map_trait!(Hi8_8bit64Map);
    crate::traits::impl_map_trait!(Lo128_8bit64Map);
    crate::traits::impl_map_trait!(Lo128_1bit64Map);
    crate::traits::impl_map_trait!(Ufm64Map);
    crate::traits::impl_map_trait!(Gaps64Map);
    // Embedded matrix — Hi8
    crate::traits::impl_map_trait!(Hi8_EmbMap);
    crate::traits::impl_map_trait!(Hi8_EmbP2Map);
    crate::traits::impl_map_trait!(Hi8_Emb32Map);
    crate::traits::impl_map_trait!(Hi8_EmbP232Map);
    crate::traits::impl_map_trait!(Hi8_Emb64Map);
    crate::traits::impl_map_trait!(Hi8_EmbP264Map);
    // Embedded matrix — Lo128
    crate::traits::impl_map_trait!(Lo128_EmbMap);
    crate::traits::impl_map_trait!(Lo128_EmbP2Map);
    crate::traits::impl_map_trait!(Lo128_Emb32Map);
    crate::traits::impl_map_trait!(Lo128_EmbP232Map);
    crate::traits::impl_map_trait!(Lo128_Emb64Map);
    crate::traits::impl_map_trait!(Lo128_EmbP264Map);
    // Embedded matrix — Top128 AND
    crate::traits::impl_map_trait!(Top128_EmbAndMap);
    crate::traits::impl_map_trait!(Top128_EmbP2AndMap);
    crate::traits::impl_map_trait!(Top128_EmbAnd32Map);
    crate::traits::impl_map_trait!(Top128_EmbP2And32Map);
    crate::traits::impl_map_trait!(Top128_EmbAnd64Map);
    crate::traits::impl_map_trait!(Top128_EmbP2And64Map);
    // Embedded matrix — Top255 AND
    crate::traits::impl_map_trait!(Top255_EmbAndMap);
    crate::traits::impl_map_trait!(Top255_EmbP2AndMap);
    crate::traits::impl_map_trait!(Top255_EmbAnd32Map);
    crate::traits::impl_map_trait!(Top255_EmbP2And32Map);
    crate::traits::impl_map_trait!(Top255_EmbAnd64Map);
    crate::traits::impl_map_trait!(Top255_EmbP2And64Map);
    crate::traits::impl_map_trait!(Hi128_TombMap);
    crate::traits::impl_map_trait!(Top128_TombMap);
    crate::traits::impl_map_trait!(Hi128_Tomb64Map);
    crate::traits::impl_map_trait!(Top128_Tomb64Map);
}

// ── Traits ──────────────────────────────────────────────────────────────────

pub use raw::hash::IsAvalanching;
pub use traits::Map;
pub use traits::OccupiedError;
pub use traits::Set;
pub use traits::SortedMap;
pub use traits::SortedSet;
