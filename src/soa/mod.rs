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
pub type SoaLo128<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, Lo128_8bit, SoA>>;

/// SoA + decorrelated tag (byte 1) + 8-channel byte overflow.
pub type SoaHi8<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, Hi8_8bit, SoA>>;

/// SoA + low-byte 255 tag + 1-bit overflow.
pub type SoaLo8_1bit<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, Lo8_1bit, SoA>>;

/// SoA + decorrelated tag (byte 1) + 1-bit overflow.
pub type SoaHi8_1bit<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, Hi8_1bit, SoA>>;

/// SoA + 128-value fast tag + 1-bit overflow.
pub type SoaLo128_1bit<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, Lo128_1bit, SoA>>;

/// SoA + 128-value top-bit tag + 1-bit overflow + AND indexing.
pub type SoaTop128And<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, Top128_1bitAnd, SoA>>;

/// SoA + 255-value top-bit tag + 1-bit overflow + AND indexing.
pub type SoaTop255And<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, Top255_1bitAnd, SoA>>;

/// SoA + 128-value top-bit tag + 8-bit overflow + AND indexing (shifted channels).
pub type SoaTop128_8bitAnd<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, Top128_8bitAnd, SoA>>;

/// SoA + 255-value top-bit tag + 8-bit overflow + AND indexing (shifted channels).
pub type SoaTop255_8bitAnd<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, Top255_8bitAnd, SoA>>;

// ── Tombstone SoA variants (IPO family) ───────────────────────────────────

use crate::in_place_overflow::raw::RawTable as IpoRawTable;
use crate::raw::tag_strategy::{LowByte254, HighByte128, TopByte128};

/// SoA + IPO tombstone (default LowByte254 tag).
pub type SoaIpo<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, IpoRawTable<K, V, LowByte254, SoA>>;

/// SoA + IPO tombstone, HighByte128 tag.
pub type SoaHi128_Tomb<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, IpoRawTable<K, V, HighByte128, SoA>>;

/// SoA + IPO tombstone, TopByte128 tag.
pub type SoaTop128_Tomb<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, IpoRawTable<K, V, TopByte128, SoA>>;

// ── Map trait impls ───────────────────────────────────────────────────────
// GenericMap already implements Map via impl_map_trait!, so SoA type aliases
// automatically get Map trait support — no extra code needed.

crate::traits::impl_map_trait!(SoaMap);
crate::traits::impl_map_trait!(SoaLo128);
crate::traits::impl_map_trait!(SoaHi8);
crate::traits::impl_map_trait!(SoaLo8_1bit);
crate::traits::impl_map_trait!(SoaHi8_1bit);
crate::traits::impl_map_trait!(SoaLo128_1bit);
crate::traits::impl_map_trait!(SoaTop128And);
crate::traits::impl_map_trait!(SoaTop255And);
crate::traits::impl_map_trait!(SoaTop128_8bitAnd);
crate::traits::impl_map_trait!(SoaTop255_8bitAnd);
crate::traits::impl_map_trait!(SoaIpo);
crate::traits::impl_map_trait!(SoaHi128_Tomb);
crate::traits::impl_map_trait!(SoaTop128_Tomb);
