//! 64-slot group operations (AVX-512BW preferred, AVX2 / SSE2 fallbacks).
//!
//! Compile-time tier selection based on `target_feature`. The flake's
//! devShell sets `RUSTFLAGS="-C target-cpu=native"` so the build picks the
//! richest tier the host supports.
//!
//! Tiers:
//! - **AVX-512BW**: 1 × 512-bit load + cmpeq → `__mmask64` directly
//! - **AVX2** (no AVX-512): 2 × 256-bit loads, combine two u32 movemasks → u64
//! - **SSE2** (no AVX2): 4 × 128-bit loads, combine four u16 movemasks → u64
//! - **non-x86_64 / Miri**: scalar fallback
//!
//! Metadata pointer must be 64-byte aligned for the AVX-512 aligned load.
//! `Layout64` sets `META_ALIGN = 64` so the table allocator guarantees this.

#[cfg(all(target_arch = "x86_64", not(miri)))]
use std::arch::x86_64::*;

use super::bitmask::BitMask64;

/// SIMD group operations for 64-slot designs.
///
/// `SLOT_MASK` is applied to all match results. For full 64-slot designs
/// pass `0xFFFF_FFFF_FFFF_FFFF` (the `& SLOT_MASK` compiles away).
pub struct Group64<const SLOT_MASK: u64>;

// ── x86_64 + AVX-512BW ────────────────────────────────────────────────────

#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw", not(miri)))]
impl<const SLOT_MASK: u64> Group64<SLOT_MASK> {
    /// SAFETY: `ptr` must be 64-byte aligned.
    #[inline(always)]
    pub unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask64 {
        unsafe {
            let data = _mm512_load_si512(ptr as *const __m512i);
            let needle = _mm512_set1_epi8(value as i8);
            BitMask64(_mm512_cmpeq_epi8_mask(data, needle) & SLOT_MASK)
        }
    }

    #[inline(always)]
    pub unsafe fn match_empty(ptr: *const u8) -> BitMask64 {
        unsafe {
            let data = _mm512_load_si512(ptr as *const __m512i);
            let zero = _mm512_setzero_si512();
            BitMask64(_mm512_cmpeq_epi8_mask(data, zero) & SLOT_MASK)
        }
    }

    #[inline(always)]
    pub unsafe fn match_non_empty(ptr: *const u8) -> BitMask64 {
        unsafe {
            let data = _mm512_load_si512(ptr as *const __m512i);
            let zero = _mm512_setzero_si512();
            let empty = _mm512_cmpeq_epi8_mask(data, zero);
            BitMask64((!empty) & SLOT_MASK)
        }
    }

    #[inline(always)]
    pub unsafe fn match_byte_and_empty(ptr: *const u8, value: u8) -> (BitMask64, BitMask64) {
        unsafe {
            let data = _mm512_load_si512(ptr as *const __m512i);
            let needle = _mm512_set1_epi8(value as i8);
            let zero = _mm512_setzero_si512();
            (
                BitMask64(_mm512_cmpeq_epi8_mask(data, needle) & SLOT_MASK),
                BitMask64(_mm512_cmpeq_epi8_mask(data, zero) & SLOT_MASK),
            )
        }
    }

    #[inline(always)]
    pub unsafe fn prefetch_read(ptr: *const u8) {
        unsafe { _mm_prefetch(ptr as *const i8, _MM_HINT_T0); }
    }

    #[inline(always)] pub unsafe fn get_meta(ptr: *const u8, idx: usize) -> u8 { unsafe { *ptr.add(idx) } }
    #[inline(always)] pub unsafe fn set_meta(ptr: *mut u8, idx: usize, value: u8) { unsafe { *ptr.add(idx) = value; } }
}

// ── x86_64 + AVX2 (no AVX-512) ────────────────────────────────────────────

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512bw"), not(miri)))]
impl<const SLOT_MASK: u64> Group64<SLOT_MASK> {
    #[inline(always)]
    pub unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask64 {
        unsafe {
            let lo = _mm256_load_si256(ptr as *const __m256i);
            let hi = _mm256_load_si256(ptr.add(32) as *const __m256i);
            let needle = _mm256_set1_epi8(value as i8);
            let lo_cmp = _mm256_cmpeq_epi8(lo, needle);
            let hi_cmp = _mm256_cmpeq_epi8(hi, needle);
            let lo_m = _mm256_movemask_epi8(lo_cmp) as u32 as u64;
            let hi_m = _mm256_movemask_epi8(hi_cmp) as u32 as u64;
            BitMask64(((hi_m << 32) | lo_m) & SLOT_MASK)
        }
    }

    #[inline(always)]
    pub unsafe fn match_empty(ptr: *const u8) -> BitMask64 {
        unsafe { Self::match_byte(ptr, 0) }
    }

    #[inline(always)]
    pub unsafe fn match_non_empty(ptr: *const u8) -> BitMask64 {
        unsafe {
            let lo = _mm256_load_si256(ptr as *const __m256i);
            let hi = _mm256_load_si256(ptr.add(32) as *const __m256i);
            let zero = _mm256_setzero_si256();
            let lo_cmp = _mm256_cmpeq_epi8(lo, zero);
            let hi_cmp = _mm256_cmpeq_epi8(hi, zero);
            let lo_m = _mm256_movemask_epi8(lo_cmp) as u32 as u64;
            let hi_m = _mm256_movemask_epi8(hi_cmp) as u32 as u64;
            BitMask64((!((hi_m << 32) | lo_m)) & SLOT_MASK)
        }
    }

    #[inline(always)]
    pub unsafe fn match_byte_and_empty(ptr: *const u8, value: u8) -> (BitMask64, BitMask64) {
        unsafe { (Self::match_byte(ptr, value), Self::match_empty(ptr)) }
    }

    #[inline(always)] pub unsafe fn prefetch_read(ptr: *const u8) { unsafe { _mm_prefetch(ptr as *const i8, _MM_HINT_T0); } }
    #[inline(always)] pub unsafe fn get_meta(ptr: *const u8, idx: usize) -> u8 { unsafe { *ptr.add(idx) } }
    #[inline(always)] pub unsafe fn set_meta(ptr: *mut u8, idx: usize, value: u8) { unsafe { *ptr.add(idx) = value; } }
}

// ── x86_64 SSE2 only (no AVX2 / AVX-512) ──────────────────────────────────

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2"), not(miri)))]
impl<const SLOT_MASK: u64> Group64<SLOT_MASK> {
    #[inline(always)]
    pub unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask64 {
        unsafe {
            let needle = _mm_set1_epi8(value as i8);
            let mut mask: u64 = 0;
            for i in 0..4 {
                let chunk = _mm_load_si128(ptr.add(i * 16) as *const __m128i);
                let cmp = _mm_cmpeq_epi8(chunk, needle);
                let m = _mm_movemask_epi8(cmp) as u32 as u64 & 0xFFFF;
                mask |= m << (i * 16);
            }
            BitMask64(mask & SLOT_MASK)
        }
    }

    #[inline(always)]
    pub unsafe fn match_empty(ptr: *const u8) -> BitMask64 {
        unsafe { Self::match_byte(ptr, 0) }
    }

    #[inline(always)]
    pub unsafe fn match_non_empty(ptr: *const u8) -> BitMask64 {
        unsafe {
            let zero = _mm_setzero_si128();
            let mut empty_mask: u64 = 0;
            for i in 0..4 {
                let chunk = _mm_load_si128(ptr.add(i * 16) as *const __m128i);
                let cmp = _mm_cmpeq_epi8(chunk, zero);
                let m = _mm_movemask_epi8(cmp) as u32 as u64 & 0xFFFF;
                empty_mask |= m << (i * 16);
            }
            BitMask64((!empty_mask) & SLOT_MASK)
        }
    }

    #[inline(always)]
    pub unsafe fn match_byte_and_empty(ptr: *const u8, value: u8) -> (BitMask64, BitMask64) {
        unsafe { (Self::match_byte(ptr, value), Self::match_empty(ptr)) }
    }

    #[inline(always)] pub unsafe fn prefetch_read(ptr: *const u8) { unsafe { _mm_prefetch(ptr as *const i8, _MM_HINT_T0); } }
    #[inline(always)] pub unsafe fn get_meta(ptr: *const u8, idx: usize) -> u8 { unsafe { *ptr.add(idx) } }
    #[inline(always)] pub unsafe fn set_meta(ptr: *mut u8, idx: usize, value: u8) { unsafe { *ptr.add(idx) = value; } }
}

// ── Scalar fallback ───────────────────────────────────────────────────────

#[cfg(any(not(target_arch = "x86_64"), miri))]
impl<const SLOT_MASK: u64> Group64<SLOT_MASK> {
    #[inline(always)]
    pub unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask64 {
        let mut mask = 0u64;
        for i in 0..64 {
            if unsafe { *ptr.add(i) } == value { mask |= 1u64 << i; }
        }
        BitMask64(mask & SLOT_MASK)
    }

    #[inline(always)]
    pub unsafe fn match_empty(ptr: *const u8) -> BitMask64 { unsafe { Self::match_byte(ptr, 0) } }

    #[inline(always)]
    pub unsafe fn match_non_empty(ptr: *const u8) -> BitMask64 {
        let mut mask = 0u64;
        for i in 0..64 {
            if unsafe { *ptr.add(i) } != 0 { mask |= 1u64 << i; }
        }
        BitMask64(mask & SLOT_MASK)
    }

    #[inline(always)]
    pub unsafe fn match_byte_and_empty(ptr: *const u8, value: u8) -> (BitMask64, BitMask64) {
        unsafe { (Self::match_byte(ptr, value), Self::match_empty(ptr)) }
    }

    #[inline(always)] pub unsafe fn prefetch_read(_ptr: *const u8) {}
    #[inline(always)] pub unsafe fn get_meta(ptr: *const u8, idx: usize) -> u8 { unsafe { *ptr.add(idx) } }
    #[inline(always)] pub unsafe fn set_meta(ptr: *mut u8, idx: usize, value: u8) { unsafe { *ptr.add(idx) = value; } }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C, align(64))]
    struct Aligned64([u8; 64]);

    fn make() -> Aligned64 { Aligned64([0; 64]) }

    #[test]
    fn empty_buffer_all_match() {
        let buf = make();
        unsafe {
            let m = Group64::<0xFFFF_FFFF_FFFF_FFFF>::match_empty(buf.0.as_ptr());
            assert_eq!(m.0, 0xFFFF_FFFF_FFFF_FFFF);
            let n = Group64::<0xFFFF_FFFF_FFFF_FFFF>::match_non_empty(buf.0.as_ptr());
            assert_eq!(n.0, 0);
        }
    }

    #[test]
    fn match_byte_finds_hits() {
        let mut buf = make();
        buf.0[0] = 7;
        buf.0[33] = 7;
        buf.0[63] = 7;
        unsafe {
            let m = Group64::<0xFFFF_FFFF_FFFF_FFFF>::match_byte(buf.0.as_ptr(), 7);
            let hits: Vec<usize> = m.collect();
            assert_eq!(hits, vec![0, 33, 63]);
        }
    }

    #[test]
    fn match_non_empty_skips_zero() {
        let mut buf = make();
        buf.0[1] = 1;
        buf.0[40] = 99;
        buf.0[55] = 200;
        unsafe {
            let m = Group64::<0xFFFF_FFFF_FFFF_FFFF>::match_non_empty(buf.0.as_ptr());
            let hits: Vec<usize> = m.collect();
            assert_eq!(hits, vec![1, 40, 55]);
        }
    }

    #[test]
    fn match_byte_and_empty_split() {
        let mut buf = make();
        buf.0[0] = 5;
        buf.0[10] = 5;
        buf.0[42] = 99;
        unsafe {
            let (matches, empties) =
                Group64::<0xFFFF_FFFF_FFFF_FFFF>::match_byte_and_empty(buf.0.as_ptr(), 5);
            assert_eq!(matches.collect::<Vec<_>>(), vec![0, 10]);
            let empty_set: Vec<usize> = empties.collect();
            assert_eq!(empty_set.len(), 61);
            assert!(!empty_set.contains(&0));
            assert!(!empty_set.contains(&10));
            assert!(!empty_set.contains(&42));
        }
    }
}
