//! 64-slot group operations — one full cache line of metadata per group.
//!
//! Same encoding as IPO (0x00=EMPTY, 0x01=TOMBSTONE, 0x02-0xFF=hash) but
//! with 64 slots per group instead of 16. Each group's metadata occupies
//! one 64-byte cache line. SIMD matching requires 4 SSE2 loads per group.
//!
//! The advantage: one cache line fetch gives all metadata for 64 slots,
//! eliminating multi-probe cache misses at large scale.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::raw::bitmask::BitMask;

/// Number of element slots per group (one cache line of metadata).
pub const GROUP_SIZE: usize = 64;

/// Total metadata bytes per group.
pub const META_GROUP_BYTES: usize = 64;

/// Metadata byte: slot is empty.
pub const EMPTY: u8 = 0x00;

/// Metadata byte: slot was occupied but has been deleted.
pub const TOMBSTONE: u8 = 0x01;

/// Compute the reduced hash value from the low byte of a hash.
/// Maps to range [2, 255]. Values 0 (EMPTY) and 1 (TOMBSTONE) are reserved.
#[inline(always)]
pub fn reduced_hash(h: u64) -> u8 {
    let low = (h & 0xFF) as u8;
    if low < 2 { low + 2 } else { low }
}

/// 64-bit bitmask for 64-slot groups.
#[derive(Clone, Copy, Debug)]
pub struct BitMask64(pub u64);

impl BitMask64 {
    #[inline]
    pub fn any_set(self) -> bool {
        self.0 != 0
    }

    #[inline]
    pub fn lowest_set_bit(self) -> Option<usize> {
        if self.0 == 0 { None } else { Some(self.0.trailing_zeros() as usize) }
    }
}

impl Iterator for BitMask64 {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            let idx = self.0.trailing_zeros() as usize;
            self.0 &= self.0 - 1;
            Some(idx)
        }
    }
}

// ── x86_64 SSE2 implementation ──────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub struct Group;

#[cfg(target_arch = "x86_64")]
impl Group {
    /// Match byte across all 64 slots (4 SIMD loads).
    #[inline(always)]
    pub unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask64 {
        unsafe {
            let needle = _mm_set1_epi8(value as i8);
            let d0 = _mm_load_si128(ptr as *const __m128i);
            let d1 = _mm_load_si128(ptr.add(16) as *const __m128i);
            let d2 = _mm_load_si128(ptr.add(32) as *const __m128i);
            let d3 = _mm_load_si128(ptr.add(48) as *const __m128i);
            let m0 = _mm_movemask_epi8(_mm_cmpeq_epi8(d0, needle)) as u64;
            let m1 = _mm_movemask_epi8(_mm_cmpeq_epi8(d1, needle)) as u64;
            let m2 = _mm_movemask_epi8(_mm_cmpeq_epi8(d2, needle)) as u64;
            let m3 = _mm_movemask_epi8(_mm_cmpeq_epi8(d3, needle)) as u64;
            BitMask64(m0 | (m1 << 16) | (m2 << 32) | (m3 << 48))
        }
    }

    /// Match EMPTY slots across all 64 slots.
    #[inline(always)]
    pub unsafe fn match_empty(ptr: *const u8) -> BitMask64 {
        unsafe {
            let zero = _mm_setzero_si128();
            let d0 = _mm_load_si128(ptr as *const __m128i);
            let d1 = _mm_load_si128(ptr.add(16) as *const __m128i);
            let d2 = _mm_load_si128(ptr.add(32) as *const __m128i);
            let d3 = _mm_load_si128(ptr.add(48) as *const __m128i);
            let m0 = _mm_movemask_epi8(_mm_cmpeq_epi8(d0, zero)) as u64;
            let m1 = _mm_movemask_epi8(_mm_cmpeq_epi8(d1, zero)) as u64;
            let m2 = _mm_movemask_epi8(_mm_cmpeq_epi8(d2, zero)) as u64;
            let m3 = _mm_movemask_epi8(_mm_cmpeq_epi8(d3, zero)) as u64;
            BitMask64(m0 | (m1 << 16) | (m2 << 32) | (m3 << 48))
        }
    }

    /// Match EMPTY or TOMBSTONE slots (available for insertion).
    #[inline(always)]
    pub unsafe fn match_empty_or_tombstone(ptr: *const u8) -> BitMask64 {
        unsafe {
            let zero = _mm_setzero_si128();
            let one = _mm_set1_epi8(1);
            let d0 = _mm_load_si128(ptr as *const __m128i);
            let d1 = _mm_load_si128(ptr.add(16) as *const __m128i);
            let d2 = _mm_load_si128(ptr.add(32) as *const __m128i);
            let d3 = _mm_load_si128(ptr.add(48) as *const __m128i);
            let c = |d: __m128i| -> u64 {
                let e = _mm_cmpeq_epi8(d, zero);
                let t = _mm_cmpeq_epi8(d, one);
                _mm_movemask_epi8(_mm_or_si128(e, t)) as u64
            };
            BitMask64(c(d0) | (c(d1) << 16) | (c(d2) << 32) | (c(d3) << 48))
        }
    }

    /// Match occupied slots (not EMPTY, not TOMBSTONE).
    #[inline(always)]
    pub unsafe fn match_occupied(ptr: *const u8) -> BitMask64 {
        unsafe {
            let zero = _mm_setzero_si128();
            let one = _mm_set1_epi8(1);
            let c = |d: __m128i| -> u64 {
                let e = _mm_cmpeq_epi8(d, zero);
                let t = _mm_cmpeq_epi8(d, one);
                let dead = _mm_or_si128(e, t);
                (!_mm_movemask_epi8(dead) as u16) as u64
            };
            let d0 = _mm_load_si128(ptr as *const __m128i);
            let d1 = _mm_load_si128(ptr.add(16) as *const __m128i);
            let d2 = _mm_load_si128(ptr.add(32) as *const __m128i);
            let d3 = _mm_load_si128(ptr.add(48) as *const __m128i);
            BitMask64(c(d0) | (c(d1) << 16) | (c(d2) << 32) | (c(d3) << 48))
        }
    }

    /// Match byte and empty in one pass (shares SIMD loads).
    #[inline(always)]
    pub unsafe fn match_byte_and_empty(ptr: *const u8, value: u8) -> (BitMask64, BitMask64) {
        unsafe {
            let needle = _mm_set1_epi8(value as i8);
            let zero = _mm_setzero_si128();
            let mut match_bits = 0u64;
            let mut empty_bits = 0u64;
            for i in 0..4u64 {
                let d = _mm_load_si128(ptr.add((i as usize) * 16) as *const __m128i);
                match_bits |= (_mm_movemask_epi8(_mm_cmpeq_epi8(d, needle)) as u64) << (i * 16);
                empty_bits |= (_mm_movemask_epi8(_mm_cmpeq_epi8(d, zero)) as u64) << (i * 16);
            }
            (BitMask64(match_bits), BitMask64(empty_bits))
        }
    }

    /// Prefetch a cache line for temporal read access.
    #[inline(always)]
    pub unsafe fn prefetch_read(ptr: *const u8) {
        unsafe { _mm_prefetch(ptr as *const i8, _MM_HINT_T0); }
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
    pub unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask64 {
        let mut mask = 0u64;
        for i in 0..GROUP_SIZE {
            if unsafe { *ptr.add(i) } == value { mask |= 1 << i; }
        }
        BitMask64(mask)
    }

    #[inline(always)]
    pub unsafe fn match_empty(ptr: *const u8) -> BitMask64 {
        let mut mask = 0u64;
        for i in 0..GROUP_SIZE {
            if unsafe { *ptr.add(i) } == EMPTY { mask |= 1 << i; }
        }
        BitMask64(mask)
    }

    #[inline(always)]
    pub unsafe fn match_empty_or_tombstone(ptr: *const u8) -> BitMask64 {
        let mut mask = 0u64;
        for i in 0..GROUP_SIZE {
            let b = unsafe { *ptr.add(i) };
            if b <= TOMBSTONE { mask |= 1 << i; }
        }
        BitMask64(mask)
    }

    #[inline(always)]
    pub unsafe fn match_occupied(ptr: *const u8) -> BitMask64 {
        let mut mask = 0u64;
        for i in 0..GROUP_SIZE {
            if unsafe { *ptr.add(i) } >= 2 { mask |= 1 << i; }
        }
        BitMask64(mask)
    }

    #[inline(always)]
    pub unsafe fn match_byte_and_empty(ptr: *const u8, value: u8) -> (BitMask64, BitMask64) {
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
