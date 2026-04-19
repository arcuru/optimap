//! Layout configurations for overflow-bit hash table designs.
//!
//! `GroupLayout` composes three orthogonal axes:
//!   1. SIMD group ops (`GroupOps`) — parameterized by slot mask
//!   2. Tag strategy (`TagStrategy`) — how hash tags and overflow channels are extracted
//!   3. Overflow strategy (`OverflowStrategy`) — how overflow info is stored
//!
//! The `Layout16<T, O>` generic struct makes new matrix entries trivial:
//! just pick a tag strategy and overflow strategy.

use std::marker::PhantomData;

use super::bitmask::BitMask;
use super::generic_group::Group;
use super::overflow_strategy::OverflowStrategy;
use super::tag_strategy::TagStrategy;

/// Unified interface for SIMD group operations.
///
/// Hides the const-generic `SLOT_MASK` parameter behind a trait so that
/// `GroupLayout` can carry it as an associated type without requiring
/// `generic_const_exprs` (unstable).
pub trait GroupOps {
    unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask;
    unsafe fn match_empty(ptr: *const u8) -> BitMask;
    unsafe fn match_non_empty(ptr: *const u8) -> BitMask;
    unsafe fn match_byte_and_empty(ptr: *const u8, value: u8) -> (BitMask, BitMask);
    unsafe fn prefetch_read(ptr: *const u8);
    unsafe fn get_meta(ptr: *const u8, idx: usize) -> u8;
    unsafe fn set_meta(ptr: *mut u8, idx: usize, value: u8);
}

impl<const M: u16> GroupOps for Group<M> {
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
    /// Whether overflow is in a separate array (controls extra prefetch).
    const SEPARATE_OVERFLOW: bool;

    /// Use AND-based group indexing (`h & mask`) instead of shift-based (`h >> shift`).
    ///
    /// AND-based is 1 instruction faster (eliminates variable shift) but requires
    /// tags from the top hash bits (57+) to avoid correlation with the group index.
    /// Only safe with non-channeled overflow (BitSeparate) — 8-bit overflow channels
    /// use low bits which would correlate with the AND group index.
    const AND_INDEX: bool = false;

    /// Compute bucket index from (group_index, slot_index).
    #[inline(always)]
    fn bucket_index(gi: usize, si: usize) -> usize {
        if Self::BUCKET_STRIDE == 16 {
            (gi << 4) | si
        } else {
            gi * Self::BUCKET_STRIDE + si
        }
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

// ── Embedded overflow for UFM/Gaps ─────────────────────────────────────────
// These can't use the generic OverflowStrategy because the overflow byte
// is at a fixed offset within the metadata group (byte 15), not in a
// separate array. The pointer arithmetic differs.

/// Embedded overflow at byte 15 of each 16-byte metadata group (UFM).
#[derive(Clone, Copy)]
pub struct UfmEmbeddedOverflow;

impl OverflowStrategy for UfmEmbeddedOverflow {
    const CHANNELED: bool = true;

    #[inline(always)]
    fn extra_alloc_bytes(_num_groups: usize) -> usize { 0 }
    #[inline(always)]
    fn overflow_bytes_to_zero(_num_groups: usize) -> usize { 0 }
    #[inline(always)]
    fn overflow_bytes_to_copy(_num_groups: usize) -> usize { 0 }

    #[inline(always)]
    unsafe fn overflow_ptr(metadata: *mut u8, _mask: usize, gi: usize) -> *mut u8 {
        unsafe { metadata.add(gi * 16 + 15) }
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

/// Embedded overflow at byte 15 (Gaps — same logic as UFM).
pub type GapsEmbeddedOverflow = UfmEmbeddedOverflow;

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
