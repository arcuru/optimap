//! Shared SIMD group operations parameterized by slot mask.
//!
//! All overflow-bit designs share the same SSE2 intrinsics — the only
//! difference is whether the 16th bit is masked off (15-slot embedded
//! overflow) or kept (16-slot separate overflow). The `SLOT_MASK` const
//! generic eliminates the mask instruction entirely when it's 0xFFFF.

#[cfg(all(target_arch = "x86_64", not(miri)))]
use std::arch::x86_64::*;

use super::bitmask::BitMask;

/// Metadata byte: slot is empty.
pub const EMPTY: u8 = 0x00;

/// Overflow bit index for a given hash value.
#[inline(always)]
pub fn overflow_bit(h: u64) -> u8 {
    1u8 << (h & 7)
}

/// Extract a non-zero hash tag from a hash value.
/// Delegates to the crate-level feature-gated implementation.
#[inline(always)]
pub fn reduced_hash(h: u64) -> u8 {
    crate::hash_tag(h)
}

// ── x86_64 SSE2 implementation ────────────────────────────────────────────

/// SIMD group operations for overflow-bit designs.
///
/// `SLOT_MASK` is applied to all movemask results:
/// - `0x7FFF` for 15-slot groups (masks off byte 15 = embedded overflow)
/// - `0xFFFF` for 16-slot groups (all bits valid)
///
/// When `SLOT_MASK == 0xFFFF`, the `& SLOT_MASK` compiles away to nothing.
#[cfg(all(target_arch = "x86_64", not(miri)))]
pub struct Group<const SLOT_MASK: u16>;

#[cfg(all(target_arch = "x86_64", not(miri)))]
impl<const SLOT_MASK: u16> Group<SLOT_MASK> {
    /// Return a bitmask of slots matching `value`.
    /// SAFETY: `ptr` must be 16-byte aligned.
    #[inline(always)]
    pub unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask {
        unsafe {
            let data = _mm_load_si128(ptr as *const __m128i);
            let needle = _mm_set1_epi8(value as i8);
            let cmp = _mm_cmpeq_epi8(data, needle);
            BitMask(_mm_movemask_epi8(cmp) as u16 & SLOT_MASK)
        }
    }

    /// Return a bitmask of empty slots.
    #[inline(always)]
    pub unsafe fn match_empty(ptr: *const u8) -> BitMask {
        unsafe {
            let data = _mm_load_si128(ptr as *const __m128i);
            let zero = _mm_setzero_si128();
            let cmp = _mm_cmpeq_epi8(data, zero);
            BitMask(_mm_movemask_epi8(cmp) as u16 & SLOT_MASK)
        }
    }

    /// Return a bitmask of non-empty slots.
    #[inline(always)]
    pub unsafe fn match_non_empty(ptr: *const u8) -> BitMask {
        unsafe {
            let data = _mm_load_si128(ptr as *const __m128i);
            let zero = _mm_setzero_si128();
            let cmp = _mm_cmpeq_epi8(data, zero);
            let mask = _mm_movemask_epi8(cmp) as u16;
            BitMask((!mask) & SLOT_MASK)
        }
    }

    /// Return both match and empty bitmasks with a single SIMD load.
    #[inline(always)]
    pub unsafe fn match_byte_and_empty(ptr: *const u8, value: u8) -> (BitMask, BitMask) {
        unsafe {
            let data = _mm_load_si128(ptr as *const __m128i);
            let needle = _mm_set1_epi8(value as i8);
            let zero = _mm_setzero_si128();
            let match_cmp = _mm_cmpeq_epi8(data, needle);
            let empty_cmp = _mm_cmpeq_epi8(data, zero);
            (
                BitMask(_mm_movemask_epi8(match_cmp) as u16 & SLOT_MASK),
                BitMask(_mm_movemask_epi8(empty_cmp) as u16 & SLOT_MASK),
            )
        }
    }

    /// Prefetch a cache line for temporal read access.
    #[inline(always)]
    pub unsafe fn prefetch_read(ptr: *const u8) {
        unsafe {
            _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
        }
    }

    /// Get the metadata byte for slot `idx`.
    #[inline(always)]
    pub unsafe fn get_meta(ptr: *const u8, idx: usize) -> u8 {
        unsafe { *ptr.add(idx) }
    }

    /// Set the metadata byte for slot `idx`.
    #[inline(always)]
    pub unsafe fn set_meta(ptr: *mut u8, idx: usize, value: u8) {
        unsafe {
            *ptr.add(idx) = value;
        }
    }
}

// ── Fallback for non-x86_64 / Miri ────────────────────────────────────────

#[cfg(any(not(target_arch = "x86_64"), miri))]
pub struct Group<const SLOT_MASK: u16>;

#[cfg(any(not(target_arch = "x86_64"), miri))]
impl<const SLOT_MASK: u16> Group<SLOT_MASK> {
    #[inline(always)]
    pub unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask {
        let mut mask = 0u16;
        // GROUP_SIZE is derived from SLOT_MASK: 15 for 0x7FFF, 16 for 0xFFFF
        let count = if SLOT_MASK == 0x7FFF { 15 } else { 16 };
        for i in 0..count {
            if unsafe { *ptr.add(i) } == value {
                mask |= 1 << i;
            }
        }
        BitMask(mask)
    }

    #[inline(always)]
    pub unsafe fn match_empty(ptr: *const u8) -> BitMask {
        unsafe { Self::match_byte(ptr, EMPTY) }
    }

    #[inline(always)]
    pub unsafe fn match_non_empty(ptr: *const u8) -> BitMask {
        let mut mask = 0u16;
        let count = if SLOT_MASK == 0x7FFF { 15 } else { 16 };
        for i in 0..count {
            if unsafe { *ptr.add(i) } != EMPTY {
                mask |= 1 << i;
            }
        }
        BitMask(mask)
    }

    #[inline(always)]
    pub unsafe fn match_byte_and_empty(ptr: *const u8, value: u8) -> (BitMask, BitMask) {
        unsafe { (Self::match_byte(ptr, value), Self::match_empty(ptr)) }
    }

    #[inline(always)]
    pub unsafe fn prefetch_read(_ptr: *const u8) {}

    #[inline(always)]
    pub unsafe fn get_meta(ptr: *const u8, idx: usize) -> u8 {
        unsafe { *ptr.add(idx) }
    }

    #[inline(always)]
    pub unsafe fn set_meta(ptr: *mut u8, idx: usize, value: u8) {
        unsafe {
            *ptr.add(idx) = value;
        }
    }
}
