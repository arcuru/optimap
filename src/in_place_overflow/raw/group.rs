//! 16-slot group operations with tombstone-based deletion (no overflow bytes).
//!
//! Like Splitsies but without the separate overflow array. Uses tombstones
//! for deletion and EMPTY slots for probe termination, similar to hashbrown's
//! Swiss table but with 8-bit hash values (254 values) instead of 7-bit (128).
//!
//! Metadata encoding:
//!   0x00 = EMPTY (slot never used or cleared by rehash)
//!   0x01 = TOMBSTONE (slot was occupied, now deleted)
//!   0x02-0xFF = reduced hash (slot is occupied)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::raw::bitmask::BitMask;

/// Number of element slots per group.
pub const GROUP_SIZE: usize = 16;

/// Total metadata bytes per group.
pub const META_GROUP_BYTES: usize = 16;

/// Metadata byte: slot is empty (never used or cleared by rehash).
pub const EMPTY: u8 = 0x00;

/// Metadata byte: slot was occupied but has been deleted.
pub const TOMBSTONE: u8 = 0x01;

/// Compute the reduced hash value from the low byte of a hash.
/// Maps to range [2, 255]. Values 0 (EMPTY) and 1 (TOMBSTONE) are reserved.
/// No need to preserve h%8 since there are no overflow bits.
/// Uses low byte (uncorrelated with group_index which uses high bits).
#[inline(always)]
pub fn reduced_hash(h: u64) -> u8 {
    let low = (h & 0xFF) as u8;
    // Branchless: OR with 2, then mask off bit 0 if it was already >= 2
    // Actually: `if low < 2 { low + 2 }` is already branchless via cmov.
    // The branch predictor handles the 0.78% case perfectly.
    if low < 2 { low + 2 } else { low }
}

// ── x86_64 SSE2 implementation ──────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub struct Group;

#[cfg(target_arch = "x86_64")]
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

    /// Return a bitmask of EMPTY slots (0x00 only, not tombstones).
    #[inline(always)]
    pub unsafe fn match_empty(ptr: *const u8) -> BitMask {
        unsafe {
            let data = _mm_load_si128(ptr as *const __m128i);
            let zero = _mm_setzero_si128();
            let cmp = _mm_cmpeq_epi8(data, zero);
            BitMask(_mm_movemask_epi8(cmp) as u16)
        }
    }

    /// Load group metadata into an SSE2 register for reuse.
    /// Avoids reloading for subsequent match_byte/match_empty calls.
    #[inline(always)]
    pub unsafe fn load(ptr: *const u8) -> __m128i {
        unsafe { _mm_load_si128(ptr as *const __m128i) }
    }

    /// Match byte against pre-loaded group data.
    #[inline(always)]
    pub unsafe fn loaded_match_byte(data: __m128i, value: u8) -> BitMask {
        unsafe {
            let needle = _mm_set1_epi8(value as i8);
            let cmp = _mm_cmpeq_epi8(data, needle);
            BitMask(_mm_movemask_epi8(cmp) as u16)
        }
    }

    /// Check for empty slots in pre-loaded group data.
    #[inline(always)]
    pub unsafe fn loaded_match_empty(data: __m128i) -> BitMask {
        unsafe {
            let zero = _mm_setzero_si128();
            let cmp = _mm_cmpeq_epi8(data, zero);
            BitMask(_mm_movemask_epi8(cmp) as u16)
        }
    }

    /// Return a bitmask of slots that are EMPTY or TOMBSTONE (available for insertion).
    #[inline(always)]
    pub unsafe fn match_empty_or_tombstone(ptr: *const u8) -> BitMask {
        unsafe {
            let data = _mm_load_si128(ptr as *const __m128i);
            let zero = _mm_setzero_si128();
            let one = _mm_set1_epi8(1);
            let empty_cmp = _mm_cmpeq_epi8(data, zero);
            let tomb_cmp = _mm_cmpeq_epi8(data, one);
            let combined = _mm_or_si128(empty_cmp, tomb_cmp);
            BitMask(_mm_movemask_epi8(combined) as u16)
        }
    }

    /// Return a bitmask of live (occupied) slots — excludes EMPTY and TOMBSTONE.
    #[inline(always)]
    pub unsafe fn match_occupied(ptr: *const u8) -> BitMask {
        unsafe {
            let data = _mm_load_si128(ptr as *const __m128i);
            let zero = _mm_setzero_si128();
            let one = _mm_set1_epi8(1);
            let empty_cmp = _mm_cmpeq_epi8(data, zero);
            let tomb_cmp = _mm_cmpeq_epi8(data, one);
            let dead = _mm_or_si128(empty_cmp, tomb_cmp);
            let mask = _mm_movemask_epi8(dead) as u16;
            BitMask(!mask) // invert: occupied = not (empty or tombstone)
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
        unsafe { *ptr.add(idx) = value; }
    }
}

// ── Fallback implementation ─────────────────────────────────────────────────

#[cfg(not(target_arch = "x86_64"))]
pub struct Group;

#[cfg(not(target_arch = "x86_64"))]
impl Group {
    #[inline(always)]
    pub unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask {
        let mut mask = 0u16;
        for i in 0..GROUP_SIZE {
            if unsafe { *ptr.add(i) } == value { mask |= 1 << i; }
        }
        BitMask(mask)
    }

    #[inline(always)]
    pub unsafe fn match_empty(ptr: *const u8) -> BitMask {
        unsafe { Self::match_byte(ptr, EMPTY) }
    }

    #[inline(always)]
    pub unsafe fn match_empty_or_tombstone(ptr: *const u8) -> BitMask {
        let mut mask = 0u16;
        for i in 0..GROUP_SIZE {
            let b = unsafe { *ptr.add(i) };
            if b == EMPTY || b == TOMBSTONE { mask |= 1 << i; }
        }
        BitMask(mask)
    }

    #[inline(always)]
    pub unsafe fn match_occupied(ptr: *const u8) -> BitMask {
        let mut mask = 0u16;
        for i in 0..GROUP_SIZE {
            let b = unsafe { *ptr.add(i) };
            if b >= 2 { mask |= 1 << i; }
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
        debug_assert!(idx < GROUP_SIZE);
        unsafe { *ptr.add(idx) }
    }

    #[inline(always)]
    pub unsafe fn set_meta(ptr: *mut u8, idx: usize, value: u8) {
        debug_assert!(idx < GROUP_SIZE);
        unsafe { *ptr.add(idx) = value; }
    }
}
