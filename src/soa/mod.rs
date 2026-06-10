#![allow(non_camel_case_types)] // Matrix types follow the shared <Region><Kind><Width> naming convention

//! SoA (Structure-of-Arrays) hash map designs.
//!
//! Stores keys and values in separate arrays instead of interleaved `(K, V)` tuples.
//! The probe loop only touches the key array, reducing cache pollution for large values.
//!
//! SoA is a storage dimension (`KvStorage` trait), not a separate implementation.
//! All existing designs get SoA variants by composing `RawTable<K,V,L,SoA>` with
//! `GenericMap`. The probe logic, SIMD matching, and overflow handling are identical
//! to the AoS versions — only the bucket memory layout differs.

use crate::generic_map::{DefaultHashBuilder, GenericMap};
use crate::raw::group_layout::*;
use crate::raw::kv_storage::SoA;
use crate::raw::overflow_table::RawTable;

// ── Overflow-bit SoA variants ─────────────────────────────────────────────

/// SoA hash map: Splitsies layout (16-slot, separate byte overflow, low-byte tag).
pub type SoaMap<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, SplitsiesLayout, SoA>>;

/// SoA + 128-value fast tag + 8-channel byte overflow.
pub type SoaLowTag128<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, LowTag128_8bit, SoA>>;

/// SoA + decorrelated tag (byte 1) + 8-channel byte overflow.
pub type SoaLowTag255ChSafe<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, LowTag255ChSafe_8bit, SoA>>;

/// SoA + low-byte 255 tag + 1-bit overflow.
pub type SoaLowTag255_1bit<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, LowTag255_1bit, SoA>>;

/// SoA + 128-value fast tag + 1-bit overflow.
pub type SoaLowTag128_1bit<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, LowTag128_1bit, SoA>>;

/// SoA + 128-value top-bit tag + 1-bit overflow + AND indexing.
pub type SoaByte7_128And<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, HighTag128_1bitAnd, SoA>>;

/// SoA + 255-value top-bit tag + 1-bit overflow + AND indexing.
pub type SoaByte7_255And<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, HighTag255_1bitAnd, SoA>>;

/// SoA + 128-value top-bit tag + 8-bit overflow + AND indexing (shifted channels).
pub type SoaHighTag128_8bitAnd<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, HighTag128_8bitAnd, SoA>>;

/// SoA + 255-value top-bit tag + 8-bit overflow + AND indexing (shifted channels).
pub type SoaHighTag255_8bitAnd<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, HighTag255_8bitAnd, SoA>>;

// ── Tombstone SoA variants (IPO family) ───────────────────────────────────

use crate::in_place_overflow::raw::RawTable as IpoRawTable;
use crate::raw::tag_strategy::HighTag128;

/// SoA + IPO tombstone (default `HighTomb` tag — top byte, decorrelated
/// from AND group index at any size).
pub type SoaIpo<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, IpoRawTable<K, V, crate::raw::tag_strategy::HighTomb, SoA>>;

/// SoA + IPO tombstone, `HighTag128` tag (consolidated 128-value top-byte
/// strategy — replaces the old `SoaHi128_Tomb`/`SoaTop128_Tomb` pair).
pub type SoaHighTag128_Tomb<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, IpoRawTable<K, V, HighTag128, SoA>>;

// ── HashedMap trait impls ─────────────────────────────────────────────────
// GenericMap already implements HashedMap via impl_map_trait!, so SoA type
// aliases automatically get HashedMap trait support — no extra code needed.

crate::traits::impl_map_trait!(SoaMap);
crate::traits::impl_map_trait!(SoaLowTag128);
crate::traits::impl_map_trait!(SoaLowTag255ChSafe);
crate::traits::impl_map_trait!(SoaLowTag255_1bit);
crate::traits::impl_map_trait!(SoaLowTag128_1bit);
crate::traits::impl_map_trait!(SoaByte7_128And);
crate::traits::impl_map_trait!(SoaByte7_255And);
crate::traits::impl_map_trait!(SoaHighTag128_8bitAnd);
crate::traits::impl_map_trait!(SoaHighTag255_8bitAnd);
crate::traits::impl_map_trait!(SoaIpo);
crate::traits::impl_map_trait!(SoaHighTag128_Tomb);
