//! Layout configurations for overflow-bit hash table designs.
//!
//! `GroupLayout` composes three orthogonal axes:
//!   1. SIMD group ops (`GroupOps`) — parameterized by slot mask
//!   2. Tag strategy (`TagStrategy`) — how hash tags and overflow channels are extracted
//!   3. Overflow strategy (`OverflowStrategy`) — how overflow info is stored
//!
//! The `Layout16<T, O>` generic struct makes new matrix entries trivial:
//! just pick a tag strategy and overflow strategy.

// Matrix entry type aliases use mixed-case conventions (e.g. Hi8_Emb,
// Top128_EmbAnd) that combine tag/overflow/index shorthand. Suppress the
// camel-case lint for this file.
#![allow(non_camel_case_types)]

use std::marker::PhantomData;

use super::bitmask::{BitMask, BitMask32, BitMask64, BitMaskOps};
use super::generic_group::Group;
use super::group32::Group32;
use super::group64::Group64;
use super::overflow_strategy::OverflowStrategy;
use super::tag_strategy::TagStrategy;

/// Unified interface for SIMD group operations.
///
/// The `Mask` associated type carries the bitmask width (u16 for 16-slot,
/// u32 for 32-slot, u64 for 64-slot) without spreading generics through
/// the table code.
pub trait GroupOps {
    /// Bitmask type — width matches the group's slot count.
    type Mask: BitMaskOps;

    unsafe fn match_byte(ptr: *const u8, value: u8) -> Self::Mask;
    unsafe fn match_empty(ptr: *const u8) -> Self::Mask;
    unsafe fn match_non_empty(ptr: *const u8) -> Self::Mask;
    unsafe fn match_byte_and_empty(ptr: *const u8, value: u8) -> (Self::Mask, Self::Mask);
    fn empty_mask() -> Self::Mask;
    unsafe fn prefetch_read(ptr: *const u8);
    unsafe fn get_meta(ptr: *const u8, idx: usize) -> u8;
    unsafe fn set_meta(ptr: *mut u8, idx: usize, value: u8);
}

impl<const M: u16> GroupOps for Group<M> {
    type Mask = BitMask;

    #[inline(always)]
    unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask {
        unsafe { Group::<M>::match_byte(ptr, value) }
    }
    #[inline(always)]
    unsafe fn match_empty(ptr: *const u8) -> BitMask {
        unsafe { Group::<M>::match_empty(ptr) }
    }
    #[inline(always)]
    unsafe fn match_non_empty(ptr: *const u8) -> BitMask {
        unsafe { Group::<M>::match_non_empty(ptr) }
    }
    #[inline(always)]
    unsafe fn match_byte_and_empty(ptr: *const u8, value: u8) -> (BitMask, BitMask) {
        unsafe { Group::<M>::match_byte_and_empty(ptr, value) }
    }
    #[inline(always)]
    fn empty_mask() -> BitMask { BitMask(0) }
    #[inline(always)]
    unsafe fn prefetch_read(ptr: *const u8) {
        unsafe { Group::<M>::prefetch_read(ptr) }
    }
    #[inline(always)]
    unsafe fn get_meta(ptr: *const u8, idx: usize) -> u8 {
        unsafe { Group::<M>::get_meta(ptr, idx) }
    }
    #[inline(always)]
    unsafe fn set_meta(ptr: *mut u8, idx: usize, value: u8) {
        unsafe { Group::<M>::set_meta(ptr, idx, value) }
    }
}

impl<const M: u32> GroupOps for Group32<M> {
    type Mask = BitMask32;

    #[inline(always)]
    unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask32 {
        unsafe { Group32::<M>::match_byte(ptr, value) }
    }
    #[inline(always)]
    unsafe fn match_empty(ptr: *const u8) -> BitMask32 {
        unsafe { Group32::<M>::match_empty(ptr) }
    }
    #[inline(always)]
    unsafe fn match_non_empty(ptr: *const u8) -> BitMask32 {
        unsafe { Group32::<M>::match_non_empty(ptr) }
    }
    #[inline(always)]
    unsafe fn match_byte_and_empty(ptr: *const u8, value: u8) -> (BitMask32, BitMask32) {
        unsafe { Group32::<M>::match_byte_and_empty(ptr, value) }
    }
    #[inline(always)]
    fn empty_mask() -> BitMask32 { BitMask32(0) }
    #[inline(always)]
    unsafe fn prefetch_read(ptr: *const u8) {
        unsafe { Group32::<M>::prefetch_read(ptr) }
    }
    #[inline(always)]
    unsafe fn get_meta(ptr: *const u8, idx: usize) -> u8 {
        unsafe { Group32::<M>::get_meta(ptr, idx) }
    }
    #[inline(always)]
    unsafe fn set_meta(ptr: *mut u8, idx: usize, value: u8) {
        unsafe { Group32::<M>::set_meta(ptr, idx, value) }
    }
}

impl<const M: u64> GroupOps for Group64<M> {
    type Mask = BitMask64;

    #[inline(always)]
    unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask64 {
        unsafe { Group64::<M>::match_byte(ptr, value) }
    }
    #[inline(always)]
    unsafe fn match_empty(ptr: *const u8) -> BitMask64 {
        unsafe { Group64::<M>::match_empty(ptr) }
    }
    #[inline(always)]
    unsafe fn match_non_empty(ptr: *const u8) -> BitMask64 {
        unsafe { Group64::<M>::match_non_empty(ptr) }
    }
    #[inline(always)]
    unsafe fn match_byte_and_empty(ptr: *const u8, value: u8) -> (BitMask64, BitMask64) {
        unsafe { Group64::<M>::match_byte_and_empty(ptr, value) }
    }
    #[inline(always)]
    fn empty_mask() -> BitMask64 { BitMask64(0) }
    #[inline(always)]
    unsafe fn prefetch_read(ptr: *const u8) {
        unsafe { Group64::<M>::prefetch_read(ptr) }
    }
    #[inline(always)]
    unsafe fn get_meta(ptr: *const u8, idx: usize) -> u8 {
        unsafe { Group64::<M>::get_meta(ptr, idx) }
    }
    #[inline(always)]
    unsafe fn set_meta(ptr: *mut u8, idx: usize, value: u8) {
        unsafe { Group64::<M>::set_meta(ptr, idx, value) }
    }
}

/// Layout configuration for overflow-bit hash table designs.
pub trait GroupLayout: 'static + Copy {
    /// The SIMD group type for this layout.
    type Grp: GroupOps;
    /// How hash tags and overflow channels are extracted from hash values.
    type Tag: TagStrategy;
    /// How overflow information is stored and accessed.
    type Overflow: OverflowStrategy;

    /// Number of usable element slots per group.
    const GROUP_SIZE: usize;
    /// Bucket array stride per group.
    const BUCKET_STRIDE: usize;
    /// Metadata bytes per group (the SIMD load width).
    /// Defaults to 16 for backward compat with 15/16-slot SSE2 designs;
    /// 32-slot (AVX2) uses 32, 64-slot uses 64.
    const META_STRIDE: usize = 16;
    /// Required alignment for metadata loads. Matches META_STRIDE so
    /// `ctrl + gi * META_STRIDE` is naturally aligned for the SIMD load
    /// width (avoids cache-line-split penalties on unaligned wide loads).
    const META_ALIGN: usize = Self::META_STRIDE;
    /// Whether overflow is in a separate array (controls extra prefetch).
    const SEPARATE_OVERFLOW: bool;

    /// Use AND-based group indexing (`h & mask`) instead of shift-based (`h >> shift`).
    ///
    /// AND-based is 1 instruction faster (eliminates variable shift) but requires
    /// tags from the top hash bits (57+) to avoid correlation with the group index.
    /// Only safe with non-channeled overflow (BitSeparate) — 8-bit overflow channels
    /// use low bits which would correlate with the AND group index.
    const AND_INDEX: bool = false;

    /// Load factor numerator. Table grows when `len >= capacity * NUM / DEN`.
    /// Default: 7/8 = 87.5%. Lower values waste memory but reduce collisions.
    const LOAD_FACTOR_NUM: usize = 7;
    /// Load factor denominator.
    const LOAD_FACTOR_DEN: usize = 8;

    /// Compute bucket index from (group_index, slot_index).
    ///
    /// Using `+` rather than `|` even for power-of-2 strides is intentional:
    /// LLVM folds `gi * const_pow2 + si` into a single `lea` (2 µops total:
    /// shift-in-place + LEA), while `(gi << N) | si` forces a `mov` + `shl` +
    /// `or` (3 µops) because LEA cannot fuse bitwise OR. Leave the multiply
    /// form and trust the backend.
    #[inline(always)]
    fn bucket_index(gi: usize, si: usize) -> usize {
        gi * Self::BUCKET_STRIDE + si
    }
}

// ── Layout16: generic 16-slot layout ───────────────────────────────────────

/// Generic 16-slot layout with separate overflow. Compose any tag + overflow strategy.
#[derive(Clone, Copy)]
pub struct Layout16<T: TagStrategy, O: OverflowStrategy>(PhantomData<(T, O)>);

impl<T: TagStrategy, O: OverflowStrategy> GroupLayout for Layout16<T, O> {
    type Grp = Group<0xFFFF>;
    type Tag = T;
    type Overflow = O;
    const GROUP_SIZE: usize = 16;
    const BUCKET_STRIDE: usize = 16;
    const SEPARATE_OVERFLOW: bool = true;
}

// ── Layout16And: 16-slot layout with AND-based group indexing ──────────────

/// Like Layout16 but uses AND-based group indexing (`h & mask`).
///
/// Saves 1 instruction per probe (AND vs variable shift). Requires:
/// - Tags from top hash bits (57+) to avoid correlation with group index
/// - Non-channeled overflow (BitSeparate) — 8-bit channels use low bits
///   which would correlate with the AND group index
#[derive(Clone, Copy)]
pub struct Layout16And<T: TagStrategy, O: OverflowStrategy>(PhantomData<(T, O)>);

impl<T: TagStrategy, O: OverflowStrategy> GroupLayout for Layout16And<T, O> {
    type Grp = Group<0xFFFF>;
    type Tag = T;
    type Overflow = O;
    const GROUP_SIZE: usize = 16;
    const BUCKET_STRIDE: usize = 16;
    const SEPARATE_OVERFLOW: bool = true;
    const AND_INDEX: bool = true;
}

// ── Layout32: generic 32-slot layout (AVX2) ────────────────────────────────

/// Generic 32-slot layout with separate overflow. Requires AVX2 for the
/// single-load fast path; falls back to two SSE2 loads when AVX2 is not
/// enabled at compile time. Metadata is 32 bytes per group, 32-byte aligned.
#[derive(Clone, Copy)]
pub struct Layout32<T: TagStrategy, O: OverflowStrategy>(PhantomData<(T, O)>);

impl<T: TagStrategy, O: OverflowStrategy> GroupLayout for Layout32<T, O> {
    type Grp = Group32<0xFFFF_FFFF>;
    type Tag = T;
    type Overflow = O;
    const GROUP_SIZE: usize = 32;
    const BUCKET_STRIDE: usize = 32;
    const META_STRIDE: usize = 32;
    const SEPARATE_OVERFLOW: bool = true;
}

/// Layout32 with AND-based group indexing. Same constraints as Layout16And:
/// tags must come from top hash bits (57+) and overflow must not use low-bit
/// channels correlated with the group index.
#[derive(Clone, Copy)]
pub struct Layout32And<T: TagStrategy, O: OverflowStrategy>(PhantomData<(T, O)>);

impl<T: TagStrategy, O: OverflowStrategy> GroupLayout for Layout32And<T, O> {
    type Grp = Group32<0xFFFF_FFFF>;
    type Tag = T;
    type Overflow = O;
    const GROUP_SIZE: usize = 32;
    const BUCKET_STRIDE: usize = 32;
    const META_STRIDE: usize = 32;
    const SEPARATE_OVERFLOW: bool = true;
    const AND_INDEX: bool = true;
}

// ── Layout64: generic 64-slot layout (AVX-512 / tiered fallback) ───────────

/// Generic 64-slot layout with separate overflow. Best on AVX-512BW (single
/// 512-bit load), with AVX2 (2 × 256-bit) and SSE2 (4 × 128-bit) fallbacks.
/// Metadata is 64 bytes per group (one cache line), 64-byte aligned.
#[derive(Clone, Copy)]
pub struct Layout64<T: TagStrategy, O: OverflowStrategy>(PhantomData<(T, O)>);

impl<T: TagStrategy, O: OverflowStrategy> GroupLayout for Layout64<T, O> {
    type Grp = Group64<0xFFFF_FFFF_FFFF_FFFF>;
    type Tag = T;
    type Overflow = O;
    const GROUP_SIZE: usize = 64;
    const BUCKET_STRIDE: usize = 64;
    const META_STRIDE: usize = 64;
    const SEPARATE_OVERFLOW: bool = true;
}

/// Layout64 with AND-based group indexing.
#[derive(Clone, Copy)]
pub struct Layout64And<T: TagStrategy, O: OverflowStrategy>(PhantomData<(T, O)>);

impl<T: TagStrategy, O: OverflowStrategy> GroupLayout for Layout64And<T, O> {
    type Grp = Group64<0xFFFF_FFFF_FFFF_FFFF>;
    type Tag = T;
    type Overflow = O;
    const GROUP_SIZE: usize = 64;
    const BUCKET_STRIDE: usize = 64;
    const META_STRIDE: usize = 64;
    const SEPARATE_OVERFLOW: bool = true;
    const AND_INDEX: bool = true;
}

// ── Named layouts for existing designs ─────────────────────────────────────

use super::overflow_strategy::ByteSeparate;
use super::tag_strategy::LowByte255;

/// Splitsies: 16-slot, separate byte overflow, low-byte tag.
pub type SplitsiesLayout = Layout16<LowByte255, ByteSeparate>;

/// UFM: 15-slot, embedded overflow at byte 15, low-byte tag, compact stride.
#[derive(Clone, Copy)]
pub struct UfmLayout;

impl GroupLayout for UfmLayout {
    type Grp = Group<0x7FFF>;
    type Tag = LowByte255;
    type Overflow = UfmEmbeddedOverflow;
    const GROUP_SIZE: usize = 15;
    const BUCKET_STRIDE: usize = 15;
    const SEPARATE_OVERFLOW: bool = false;
}

/// Gaps: 15-slot, embedded overflow at byte 15, low-byte tag, power-of-2 stride.
#[derive(Clone, Copy)]
pub struct GapsLayout;

impl GroupLayout for GapsLayout {
    type Grp = Group<0x7FFF>;
    type Tag = LowByte255;
    type Overflow = GapsEmbeddedOverflow;
    const GROUP_SIZE: usize = 15;
    const BUCKET_STRIDE: usize = 16;
    const SEPARATE_OVERFLOW: bool = false;
}

// ── Embedded overflow at 32/64-slot widths ────────────────────────────────
// Same trick: last byte of each metadata group is the overflow byte.
// Needs a SLOT_MASK that masks off the top bit (bit 31 at 32-slot,
// bit 63 at 64-slot) so the overflow byte is never matched as a hash tag.

/// UFM-32: 31-slot, embedded overflow at byte 31, low-byte tag, compact stride.
#[derive(Clone, Copy)]
pub struct Ufm32Layout;

impl GroupLayout for Ufm32Layout {
    type Grp = Group32<0x7FFF_FFFF>;
    type Tag = LowByte255;
    type Overflow = EmbeddedOverflow;
    const GROUP_SIZE: usize = 31;
    const BUCKET_STRIDE: usize = 31;
    const META_STRIDE: usize = 32;
    const SEPARATE_OVERFLOW: bool = false;
}

/// Gaps-32: 31-slot, embedded overflow at byte 31, low-byte tag, power-of-2 stride.
#[derive(Clone, Copy)]
pub struct Gaps32Layout;

impl GroupLayout for Gaps32Layout {
    type Grp = Group32<0x7FFF_FFFF>;
    type Tag = LowByte255;
    type Overflow = EmbeddedOverflow;
    const GROUP_SIZE: usize = 31;
    const BUCKET_STRIDE: usize = 32;
    const META_STRIDE: usize = 32;
    const SEPARATE_OVERFLOW: bool = false;
}

/// UFM-64: 63-slot, embedded overflow at byte 63, low-byte tag, compact stride.
#[derive(Clone, Copy)]
pub struct Ufm64Layout;

impl GroupLayout for Ufm64Layout {
    type Grp = Group64<0x7FFF_FFFF_FFFF_FFFF>;
    type Tag = LowByte255;
    type Overflow = EmbeddedOverflow;
    const GROUP_SIZE: usize = 63;
    const BUCKET_STRIDE: usize = 63;
    const META_STRIDE: usize = 64;
    const SEPARATE_OVERFLOW: bool = false;
}

/// Gaps-64: 63-slot, embedded overflow at byte 63, low-byte tag, power-of-2 stride.
#[derive(Clone, Copy)]
pub struct Gaps64Layout;

impl GroupLayout for Gaps64Layout {
    type Grp = Group64<0x7FFF_FFFF_FFFF_FFFF>;
    type Tag = LowByte255;
    type Overflow = EmbeddedOverflow;
    const GROUP_SIZE: usize = 63;
    const BUCKET_STRIDE: usize = 64;
    const META_STRIDE: usize = 64;
    const SEPARATE_OVERFLOW: bool = false;
}

// ── Embedded overflow for UFM/Gaps ─────────────────────────────────────────
// These can't use the generic OverflowStrategy because the overflow byte
// is at a fixed offset within the metadata group (byte 15), not in a
// separate array. The pointer arithmetic differs.

/// Embedded overflow: one overflow byte at the LAST byte of each metadata group.
///
/// Works for any `META_STRIDE` — the overflow byte sits at
/// `gi * meta_stride + (meta_stride - 1)`. The layout's SLOT_MASK must mask
/// off the top bit of the match results so the overflow byte is never
/// mistaken for an EMPTY-or-matching hash slot.
///
/// Zero extra allocation — the overflow byte steals slot `meta_stride - 1`
/// from the SIMD group, costing one usable slot per group. 16→15, 32→31, 64→63.
#[derive(Clone, Copy)]
pub struct EmbeddedOverflow;

impl OverflowStrategy for EmbeddedOverflow {
    const CHANNELED: bool = true;

    #[inline(always)]
    fn extra_alloc_bytes(_num_groups: usize) -> usize { 0 }
    #[inline(always)]
    fn overflow_bytes_to_zero(_num_groups: usize) -> usize { 0 }
    #[inline(always)]
    fn overflow_bytes_to_copy(_num_groups: usize) -> usize { 0 }

    #[inline(always)]
    unsafe fn overflow_ptr(metadata: *mut u8, _mask: usize, gi: usize, meta_stride: usize) -> *mut u8 {
        unsafe { metadata.add(gi * meta_stride + meta_stride - 1) }
    }

    #[inline(always)]
    unsafe fn has_overflow(ptr: *mut u8, _gi: usize, channel: u8) -> bool {
        unsafe { (*ptr & channel) != 0 }
    }

    #[inline(always)]
    unsafe fn set_overflow(ptr: *mut u8, _gi: usize, channel: u8) {
        unsafe { *ptr |= channel; }
    }
}

/// Back-compat aliases.
pub type UfmEmbeddedOverflow = EmbeddedOverflow;
pub type GapsEmbeddedOverflow = EmbeddedOverflow;

// ── Generic embedded-overflow layouts ──────────────────────────────────────
// Parametric over tag strategy. `EmbCompact` uses compact bucket stride
// (GROUP_SIZE), `EmbP2` uses power-of-2 stride (META_STRIDE, wasting 1 bucket
// per group). `*And` variants use AND-based group indexing — requires a tag
// whose `overflow_channel` also comes from top bits (TopTag128Ch /
// TopTag255Ch) to avoid correlation with the low-bit group index.

macro_rules! define_embedded_layout {
    ($name:ident, $grp:ty, $gs:expr, $bs:expr, $ms:expr, $and:expr) => {
        #[derive(Clone, Copy)]
        pub struct $name<T: TagStrategy>(PhantomData<T>);
        impl<T: TagStrategy> GroupLayout for $name<T> {
            type Grp = $grp;
            type Tag = T;
            type Overflow = EmbeddedOverflow;
            const GROUP_SIZE: usize = $gs;
            const BUCKET_STRIDE: usize = $bs;
            const META_STRIDE: usize = $ms;
            const SEPARATE_OVERFLOW: bool = false;
            const AND_INDEX: bool = $and;
        }
    };
}

// 16-byte metadata, 15 usable slots (byte 15 is overflow)
define_embedded_layout!(Layout16EmbCompact,    Group<0x7FFF>, 15, 15, 16, false);
define_embedded_layout!(Layout16EmbP2,         Group<0x7FFF>, 15, 16, 16, false);
define_embedded_layout!(Layout16EmbCompactAnd, Group<0x7FFF>, 15, 15, 16, true);
define_embedded_layout!(Layout16EmbP2And,      Group<0x7FFF>, 15, 16, 16, true);

// 32-byte metadata, 31 usable slots (byte 31 is overflow)
define_embedded_layout!(Layout32EmbCompact,    Group32<0x7FFF_FFFF>, 31, 31, 32, false);
define_embedded_layout!(Layout32EmbP2,         Group32<0x7FFF_FFFF>, 31, 32, 32, false);
define_embedded_layout!(Layout32EmbCompactAnd, Group32<0x7FFF_FFFF>, 31, 31, 32, true);
define_embedded_layout!(Layout32EmbP2And,      Group32<0x7FFF_FFFF>, 31, 32, 32, true);

// 64-byte metadata, 63 usable slots (byte 63 is overflow)
define_embedded_layout!(Layout64EmbCompact,    Group64<0x7FFF_FFFF_FFFF_FFFF>, 63, 63, 64, false);
define_embedded_layout!(Layout64EmbP2,         Group64<0x7FFF_FFFF_FFFF_FFFF>, 63, 64, 64, false);
define_embedded_layout!(Layout64EmbCompactAnd, Group64<0x7FFF_FFFF_FFFF_FFFF>, 63, 63, 64, true);
define_embedded_layout!(Layout64EmbP2And,      Group64<0x7FFF_FFFF_FFFF_FFFF>, 63, 64, 64, true);

// ── Matrix entries ─────────────────────────────────────────────────────────

use super::overflow_strategy::BitSeparate;
use super::tag_strategy::{HighByte255, LowByte128, TopTag128, TopTag128Ch, TopTag255, TopTag255Ch};

/// Hi8_8bit: decorrelated tag (byte 1) + 8-channel byte overflow.
pub type Hi8_8bit = Layout16<HighByte255, ByteSeparate>;

/// Lo128_8bit: 128-value fast tag + 8-channel byte overflow.
pub type Lo128_8bit = Layout16<LowByte128, ByteSeparate>;

/// Lo8_1bit: low-byte 255 tag + 1-bit binary overflow.
pub type Lo8_1bit = Layout16<LowByte255, BitSeparate>;

/// Hi8_1bit: decorrelated tag (byte 1) + 1-bit binary overflow.
pub type Hi8_1bit = Layout16<HighByte255, BitSeparate>;

/// Lo128_1bit: 128-value fast tag + 1-bit binary overflow.
pub type Lo128_1bit = Layout16<LowByte128, BitSeparate>;

// ── AND-indexed matrix entries ────────────────────────────────────────────

/// Top128_1bitAnd: 128-value top-bit tag + 1-bit overflow + AND group indexing.
pub type Top128_1bitAnd = Layout16And<TopTag128, BitSeparate>;

/// Top255_1bitAnd: 255-value top-bit tag + 1-bit overflow + AND group indexing.
pub type Top255_1bitAnd = Layout16And<TopTag255, BitSeparate>;

/// Top128_8bitAnd: 128-value top-bit tag + 8-channel byte overflow + AND indexing.
/// First 8-bit overflow design compatible with AND indexing (shifted channels).
pub type Top128_8bitAnd = Layout16And<TopTag128Ch, ByteSeparate>;

/// Top255_8bitAnd: 255-value top-bit tag + 8-channel byte overflow + AND indexing.
pub type Top255_8bitAnd = Layout16And<TopTag255Ch, ByteSeparate>;

// ── 32-slot (AVX2) matrix entries ─────────────────────────────────────────

/// Splitsies32: 32-slot, separate byte overflow, low-byte tag.
pub type Splitsies32Layout = Layout32<LowByte255, ByteSeparate>;

/// Splitsies32-1bit: 32-slot, 1-bit binary overflow, low-byte tag.
pub type Splitsies32_1bit = Layout32<LowByte255, BitSeparate>;

/// Hi8_1bit32: 32-slot, decorrelated tag + 1-bit overflow.
pub type Hi8_1bit32 = Layout32<HighByte255, BitSeparate>;

/// Hi8_8bit32: 32-slot, decorrelated tag + 8-channel byte overflow.
pub type Hi8_8bit32 = Layout32<HighByte255, ByteSeparate>;

/// Lo128_8bit32: 32-slot, 128-value low-byte tag + 8-channel overflow.
pub type Lo128_8bit32 = Layout32<LowByte128, ByteSeparate>;

/// Lo128_1bit32: 32-slot, 128-value low-byte tag + 1-bit overflow.
pub type Lo128_1bit32 = Layout32<LowByte128, BitSeparate>;

/// Top128_1bitAnd32: 32-slot AND-indexed, top-bit tag + 1-bit overflow.
pub type Top128_1bitAnd32 = Layout32And<TopTag128, BitSeparate>;

/// Top255_1bitAnd32: 32-slot AND-indexed, 255-value top-bit tag + 1-bit overflow.
pub type Top255_1bitAnd32 = Layout32And<TopTag255, BitSeparate>;

/// Top128_8bitAnd32: 32-slot AND-indexed, top-bit tag + 8-channel overflow.
pub type Top128_8bitAnd32 = Layout32And<TopTag128Ch, ByteSeparate>;

/// Top255_8bitAnd32: 32-slot AND-indexed, 255-value top tag + 8-channel overflow.
pub type Top255_8bitAnd32 = Layout32And<TopTag255Ch, ByteSeparate>;

// ── 64-slot (AVX-512) matrix entries ──────────────────────────────────────

/// Splitsies64: 64-slot, separate byte overflow, low-byte tag.
pub type Splitsies64Layout = Layout64<LowByte255, ByteSeparate>;

/// Splitsies64-1bit: 64-slot, 1-bit binary overflow, low-byte tag.
pub type Splitsies64_1bit = Layout64<LowByte255, BitSeparate>;

/// Hi8_1bit64: 64-slot, decorrelated tag + 1-bit overflow.
pub type Hi8_1bit64 = Layout64<HighByte255, BitSeparate>;

/// Hi8_8bit64: 64-slot, decorrelated tag + 8-channel byte overflow.
pub type Hi8_8bit64 = Layout64<HighByte255, ByteSeparate>;

/// Lo128_8bit64: 64-slot, 128-value low-byte tag + 8-channel overflow.
pub type Lo128_8bit64 = Layout64<LowByte128, ByteSeparate>;

/// Lo128_1bit64: 64-slot, 128-value low-byte tag + 1-bit overflow.
pub type Lo128_1bit64 = Layout64<LowByte128, BitSeparate>;

/// Top128_1bitAnd64: 64-slot AND-indexed, top-bit tag + 1-bit overflow.
pub type Top128_1bitAnd64 = Layout64And<TopTag128, BitSeparate>;

/// Top255_1bitAnd64: 64-slot AND-indexed, 255-value top-bit tag + 1-bit overflow.
pub type Top255_1bitAnd64 = Layout64And<TopTag255, BitSeparate>;

/// Top128_8bitAnd64: 64-slot AND-indexed, top-bit tag + 8-channel overflow.
pub type Top128_8bitAnd64 = Layout64And<TopTag128Ch, ByteSeparate>;

/// Top255_8bitAnd64: 64-slot AND-indexed, 255-value top tag + 8-channel overflow.
pub type Top255_8bitAnd64 = Layout64And<TopTag255Ch, ByteSeparate>;

// ── Embedded-overflow matrix entries ──────────────────────────────────────
// Covers all tag × stride × indexing combinations at 15/31/63-slot widths
// (the "embedded" family — one overflow byte at position meta_stride-1).
// LowByte255 variants (UfmLayout/GapsLayout/Ufm32Layout/Gaps32Layout/
// Ufm64Layout/Gaps64Layout) already exist above.

// Hi8 (decorrelated 255-tag, shift indexing)
pub type Hi8_Emb    = Layout16EmbCompact<HighByte255>;
pub type Hi8_EmbP2  = Layout16EmbP2<HighByte255>;
pub type Hi8_Emb32   = Layout32EmbCompact<HighByte255>;
pub type Hi8_EmbP232 = Layout32EmbP2<HighByte255>;
pub type Hi8_Emb64   = Layout64EmbCompact<HighByte255>;
pub type Hi8_EmbP264 = Layout64EmbP2<HighByte255>;

// Lo128 (128-value low-byte tag, shift indexing; faster hash_tag)
pub type Lo128_Emb    = Layout16EmbCompact<LowByte128>;
pub type Lo128_EmbP2  = Layout16EmbP2<LowByte128>;
pub type Lo128_Emb32   = Layout32EmbCompact<LowByte128>;
pub type Lo128_EmbP232 = Layout32EmbP2<LowByte128>;
pub type Lo128_Emb64   = Layout64EmbCompact<LowByte128>;
pub type Lo128_EmbP264 = Layout64EmbP2<LowByte128>;

// Top128Ch + AND indexing (tag AND channel from top bits — decorrelated from AND group index)
pub type Top128_EmbAnd    = Layout16EmbCompactAnd<TopTag128Ch>;
pub type Top128_EmbP2And  = Layout16EmbP2And<TopTag128Ch>;
pub type Top128_EmbAnd32   = Layout32EmbCompactAnd<TopTag128Ch>;
pub type Top128_EmbP2And32 = Layout32EmbP2And<TopTag128Ch>;
pub type Top128_EmbAnd64   = Layout64EmbCompactAnd<TopTag128Ch>;
pub type Top128_EmbP2And64 = Layout64EmbP2And<TopTag128Ch>;

// Top255Ch + AND indexing (255-value top-bit tag, shifted channels)
pub type Top255_EmbAnd    = Layout16EmbCompactAnd<TopTag255Ch>;
pub type Top255_EmbP2And  = Layout16EmbP2And<TopTag255Ch>;
pub type Top255_EmbAnd32   = Layout32EmbCompactAnd<TopTag255Ch>;
pub type Top255_EmbP2And32 = Layout32EmbP2And<TopTag255Ch>;
pub type Top255_EmbAnd64   = Layout64EmbCompactAnd<TopTag255Ch>;
pub type Top255_EmbP2And64 = Layout64EmbP2And<TopTag255Ch>;
