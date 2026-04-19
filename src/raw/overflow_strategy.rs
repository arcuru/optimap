//! Overflow storage strategies for overflow-bit hash table designs.
//!
//! An `OverflowStrategy` determines how overflow information is stored
//! and accessed. Two strategies:
//!
//! - `ByteSeparate`: 1 byte per group, 8 independent channels via `1 << (h & 7)`
//! - `BitSeparate`: 1 bit per group, binary flag (no per-hash discrimination)

/// Strategy for overflow storage and access.
pub trait OverflowStrategy: 'static + Copy {
    /// Whether this uses per-hash channels (8-bit) or binary (1-bit).
    /// Controls whether `overflow_channel(h)` is meaningful.
    const CHANNELED: bool;

    /// Extra bytes beyond metadata needed in the allocation.
    fn extra_alloc_bytes(num_groups: usize) -> usize;

    /// Bytes to zero on clear/init (overflow portion only — metadata zeroed separately).
    fn overflow_bytes_to_zero(num_groups: usize) -> usize;

    /// Bytes to copy on clone (overflow portion only).
    fn overflow_bytes_to_copy(num_groups: usize) -> usize;

    /// Pointer to the overflow data for group `gi`.
    ///
    /// For ByteSeparate: pointer to the overflow byte for this group.
    /// For BitSeparate: pointer to the byte containing this group's bit.
    ///
    /// # Safety
    /// `metadata` must point to a valid allocation.
    unsafe fn overflow_ptr(metadata: *mut u8, mask: usize, gi: usize) -> *mut u8;

    /// Check if overflow is set for the given channel.
    /// For 1-bit: `channel` is ignored; checks the group's single bit.
    ///
    /// # Safety
    /// `ptr` must be valid (from `overflow_ptr`), `gi` must be in range.
    unsafe fn has_overflow(ptr: *mut u8, gi: usize, channel: u8) -> bool;

    /// Set overflow for the given channel.
    /// For 1-bit: `channel` is ignored; sets the group's single bit.
    ///
    /// # Safety
    /// `ptr` must be valid (from `overflow_ptr`), `gi` must be in range.
    unsafe fn set_overflow(ptr: *mut u8, gi: usize, channel: u8);
}

// ── ByteSeparate ───────────────────────────────────────────────────────────

/// 1 byte per group in a contiguous array after metadata.
/// 8 independent overflow channels via `1 << (h & 7)`.
///
/// This is the current Splitsies overflow scheme.
#[derive(Clone, Copy)]
pub struct ByteSeparate;

impl OverflowStrategy for ByteSeparate {
    const CHANNELED: bool = true;

    #[inline(always)]
    fn extra_alloc_bytes(num_groups: usize) -> usize {
        num_groups
    }

    #[inline(always)]
    fn overflow_bytes_to_zero(num_groups: usize) -> usize {
        num_groups
    }

    #[inline(always)]
    fn overflow_bytes_to_copy(num_groups: usize) -> usize {
        num_groups
    }

    #[inline(always)]
    unsafe fn overflow_ptr(metadata: *mut u8, mask: usize, gi: usize) -> *mut u8 {
        // Overflow array starts at metadata + num_groups * 16
        // num_groups = mask + 1
        unsafe { metadata.add(((mask + 1) << 4) + gi) }
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

// ── BitSeparate ────────────────────────────────────────────────────────────

/// 1 bit per group in a compact bitfield after metadata.
/// Binary overflow flag — no per-hash channel discrimination.
///
/// The bitfield is `ceil(num_groups / 8)` bytes. Group `gi`'s bit is at
/// byte `gi >> 3`, bit `gi & 7`.
///
/// Trade-off vs ByteSeparate:
/// - Pro: ~8x smaller overflow storage, always fits in L1 even at 10M+ elements
/// - Con: higher false-continuation rate on misses (~7% vs ~0.9% at 70% load)
///   because there's no per-hash channel to filter against
#[derive(Clone, Copy)]
pub struct BitSeparate;

impl OverflowStrategy for BitSeparate {
    const CHANNELED: bool = false;

    #[inline(always)]
    fn extra_alloc_bytes(num_groups: usize) -> usize {
        (num_groups + 7) / 8
    }

    #[inline(always)]
    fn overflow_bytes_to_zero(num_groups: usize) -> usize {
        (num_groups + 7) / 8
    }

    #[inline(always)]
    fn overflow_bytes_to_copy(num_groups: usize) -> usize {
        (num_groups + 7) / 8
    }

    #[inline(always)]
    unsafe fn overflow_ptr(metadata: *mut u8, mask: usize, gi: usize) -> *mut u8 {
        // Bitfield starts at metadata + num_groups * 16
        // Group gi's byte is at offset gi >> 3
        unsafe { metadata.add(((mask + 1) << 4) + (gi >> 3)) }
    }

    #[inline(always)]
    unsafe fn has_overflow(ptr: *mut u8, gi: usize, _channel: u8) -> bool {
        unsafe { (*ptr & (1 << (gi & 7))) != 0 }
    }

    #[inline(always)]
    unsafe fn set_overflow(ptr: *mut u8, gi: usize, _channel: u8) {
        unsafe { *ptr |= 1 << (gi & 7); }
    }
}
