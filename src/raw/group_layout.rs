//! Layout configurations for overflow-bit hash table designs.
//!
//! The `GroupLayout` trait parameterizes the generic overflow-bit `RawTable`
//! over the two independent axes:
//!   1. Overflow storage: embedded (byte 15) vs separate array
//!   2. Bucket stride: 15 (compact) vs 16 (power-of-2)
//!
//! Three zero-size types implement this trait, each producing identical
//! machine code to the original hand-written implementations via monomorphization.

use super::bitmask::BitMask;
use super::generic_group::Group;

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
///
/// Implementors are zero-size types with only `const` members and trivial
/// `#[inline(always)]` methods. After monomorphization the compiler produces
/// the same code as the original hand-written implementations.
pub trait GroupLayout: 'static + Copy {
    /// The SIMD group type for this layout, parameterized by SLOT_MASK.
    type Grp: GroupOps;

    /// Number of usable element slots per group.
    const GROUP_SIZE: usize;

    /// Bucket array stride per group (may differ from GROUP_SIZE for Gaps).
    const BUCKET_STRIDE: usize;

    /// Whether overflow bytes are in a separate array (vs embedded at byte 15).
    /// Controls extra prefetch instructions in probe loops.
    const SEPARATE_OVERFLOW: bool;

    /// Pointer to the overflow byte for group `gi`.
    ///
    /// # Safety
    /// `metadata` must point to a valid allocation with enough space.
    unsafe fn overflow_ptr(metadata: *mut u8, mask: usize, gi: usize) -> *mut u8;

    /// Extra bytes beyond metadata needed in the allocation (0 or num_groups).
    fn extra_alloc_bytes(num_groups: usize) -> usize;

    /// Bytes to zero on clear/init (metadata only, or metadata + overflow array).
    fn bytes_to_zero(num_groups: usize) -> usize;

    /// Bytes to copy on clone (metadata only, or metadata + overflow array).
    fn bytes_to_copy(num_groups: usize) -> usize;

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

// ── UFM Layout ─────────────────────────────────────────────────────────────

/// 15-slot groups with embedded overflow byte at position 15.
/// Compact bucket stride (multiply-by-15). Original UnorderedFlatMap design.
#[derive(Clone, Copy)]
pub struct UfmLayout;

impl GroupLayout for UfmLayout {
    type Grp = Group<0x7FFF>;
    const GROUP_SIZE: usize = 15;
    const BUCKET_STRIDE: usize = 15;
    const SEPARATE_OVERFLOW: bool = false;

    #[inline(always)]
    unsafe fn overflow_ptr(metadata: *mut u8, _mask: usize, gi: usize) -> *mut u8 {
        unsafe { metadata.add(gi * 16 + 15) }
    }

    #[inline(always)]
    fn extra_alloc_bytes(_num_groups: usize) -> usize {
        0
    }

    #[inline(always)]
    fn bytes_to_zero(num_groups: usize) -> usize {
        num_groups * 16
    }

    #[inline(always)]
    fn bytes_to_copy(num_groups: usize) -> usize {
        num_groups * 16
    }
}

// ── Gaps Layout ────────────────────────────────────────────────────────────

/// 15-slot groups with embedded overflow byte at position 15.
/// Power-of-2 bucket stride (shift-by-4). Wastes 1 slot per group for faster indexing.
#[derive(Clone, Copy)]
pub struct GapsLayout;

impl GroupLayout for GapsLayout {
    type Grp = Group<0x7FFF>;
    const GROUP_SIZE: usize = 15;
    const BUCKET_STRIDE: usize = 16;
    const SEPARATE_OVERFLOW: bool = false;

    #[inline(always)]
    unsafe fn overflow_ptr(metadata: *mut u8, _mask: usize, gi: usize) -> *mut u8 {
        unsafe { metadata.add(gi * 16 + 15) }
    }

    #[inline(always)]
    fn extra_alloc_bytes(_num_groups: usize) -> usize {
        0
    }

    #[inline(always)]
    fn bytes_to_zero(num_groups: usize) -> usize {
        num_groups * 16
    }

    #[inline(always)]
    fn bytes_to_copy(num_groups: usize) -> usize {
        num_groups * 16
    }
}

// ── Splitsies Layout ───────────────────────────────────────────────────────

/// 16-slot groups with separate overflow array after metadata.
/// Power-of-2 bucket stride. All 16 SIMD bits are usable.
#[derive(Clone, Copy)]
pub struct SplitsiesLayout;

impl GroupLayout for SplitsiesLayout {
    type Grp = Group<0xFFFF>;
    const GROUP_SIZE: usize = 16;
    const BUCKET_STRIDE: usize = 16;
    const SEPARATE_OVERFLOW: bool = true;

    #[inline(always)]
    unsafe fn overflow_ptr(metadata: *mut u8, mask: usize, gi: usize) -> *mut u8 {
        unsafe { metadata.add(((mask + 1) << 4) + gi) }
    }

    #[inline(always)]
    fn extra_alloc_bytes(num_groups: usize) -> usize {
        num_groups
    }

    #[inline(always)]
    fn bytes_to_zero(num_groups: usize) -> usize {
        num_groups * 16 + num_groups
    }

    #[inline(always)]
    fn bytes_to_copy(num_groups: usize) -> usize {
        num_groups * 16 + num_groups
    }
}
