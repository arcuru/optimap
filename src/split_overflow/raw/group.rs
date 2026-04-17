//! 16-slot group operations with separate overflow array.
//!
//! Unlike the original 15-slot design, all 16 bytes of the SIMD word are
//! valid slot metadata. Overflow bytes live in a separate contiguous array.
//! This eliminates the `& 0x7FFF` mask and gives power-of-2 bucket addressing.

#[cfg(all(target_arch = "x86_64", not(miri)))]
use std::arch::x86_64::*;

use crate::raw::bitmask::BitMask;

/// Number of element slots per group.
pub const GROUP_SIZE: usize = 16;

/// Total metadata bytes per group (all 16 are slot metadata).
pub const META_GROUP_BYTES: usize = 16;

/// Metadata byte: slot is empty.
pub const EMPTY: u8 = 0x00;

/// Compute the reduced hash value from the low byte of a hash.
/// Maps to range [1, 255] while preserving `result % 8 == h % 8`.
/// Only 0x00 is reserved (EMPTY).
#[inline(always)]
pub fn reduced_hash(h: u64) -> u8 {
    // Branchless: map 0 → 8, everything else unchanged.
    // Preserves h % 8 (since 8 % 8 == 0 % 8).
    let low = (h & 0xFF) as u8;
    low | ((low == 0) as u8 * 8)
}

/// Overflow bit index for a given hash value.
#[inline(always)]
pub fn overflow_bit(h: u64) -> u8 {
    1u8 << (h & 7)
}

// ── x86_64 SSE2 implementation ──────────────────────────────────────────────

#[cfg(all(target_arch = "x86_64", not(miri)))]
pub struct Group;

#[cfg(all(target_arch = "x86_64", not(miri)))]
impl Group {
    /// Return a bitmask of slots matching `value`. All 16 bits are valid.
    #[inline(always)]
    pub unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask {
        unsafe {
            let data = _mm_load_si128(ptr as *const __m128i);
            let needle = _mm_set1_epi8(value as i8);
            let cmp = _mm_cmpeq_epi8(data, needle);
            BitMask(_mm_movemask_epi8(cmp) as u16)
        }
    }

    /// Return a bitmask of empty slots.
    #[inline(always)]
    pub unsafe fn match_empty(ptr: *const u8) -> BitMask {
        unsafe {
            let data = _mm_load_si128(ptr as *const __m128i);
            let zero = _mm_setzero_si128();
            let cmp = _mm_cmpeq_epi8(data, zero);
            BitMask(_mm_movemask_epi8(cmp) as u16)
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
            BitMask(!mask)
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
                BitMask(_mm_movemask_epi8(match_cmp) as u16),
                BitMask(_mm_movemask_epi8(empty_cmp) as u16),
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

    /// Check if a specific overflow bit is set.
    /// `ptr` points directly to the overflow byte for this group.
    #[inline(always)]
    pub unsafe fn has_overflow_bit(ptr: *const u8, bit: u8) -> bool {
        unsafe { (*ptr & bit) != 0 }
    }

    /// Set a bit in the overflow byte.
    /// `ptr` points directly to the overflow byte for this group.
    #[inline(always)]
    pub unsafe fn set_overflow_bit(ptr: *mut u8, bit: u8) {
        unsafe {
            *ptr |= bit;
        }
    }

    /// Get the metadata byte for slot `idx`.
    #[inline(always)]
    pub unsafe fn get_meta(ptr: *const u8, idx: usize) -> u8 {
        debug_assert!(idx < GROUP_SIZE);
        unsafe { *ptr.add(idx) }
    }

    /// Set the metadata byte for slot `idx`.
    #[inline(always)]
    pub unsafe fn set_meta(ptr: *mut u8, idx: usize, value: u8) {
        debug_assert!(idx < GROUP_SIZE);
        unsafe {
            *ptr.add(idx) = value;
        }
    }
}

// ── Fallback implementation ─────────────────────────────────────────────────

#[cfg(any(not(target_arch = "x86_64"), miri))]
pub struct Group;

#[cfg(any(not(target_arch = "x86_64"), miri))]
impl Group {
    #[inline(always)]
    pub unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask {
        let mut mask = 0u16;
        for i in 0..GROUP_SIZE {
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
        for i in 0..GROUP_SIZE {
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
    pub unsafe fn has_overflow_bit(ptr: *const u8, bit: u8) -> bool {
        unsafe { (*ptr & bit) != 0 }
    }

    #[inline(always)]
    pub unsafe fn set_overflow_bit(ptr: *mut u8, bit: u8) {
        unsafe {
            *ptr |= bit;
        }
    }

    #[inline(always)]
    pub unsafe fn get_meta(ptr: *const u8, idx: usize) -> u8 {
        debug_assert!(idx < GROUP_SIZE);
        unsafe { *ptr.add(idx) }
    }

    #[inline(always)]
    pub unsafe fn set_meta(ptr: *mut u8, idx: usize, value: u8) {
        debug_assert!(idx < GROUP_SIZE);
        unsafe {
            *ptr.add(idx) = value;
        }
    }
}
