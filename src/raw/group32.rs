//! 32-slot group operations (AVX2).
//!
//! Compile-time tier selection based on `target_feature`:
//! - `avx2` enabled → single `_mm256_load_si256` + `_mm256_cmpeq_epi8` + `_mm256_movemask_epi8`
//! - `avx2` disabled → SSE2 fallback: two 128-bit loads, combine u16 movemasks into u32
//! - non-x86_64 / Miri → scalar fallback
//!
//! Build with `RUSTFLAGS="-C target-cpu=native"` (or `-C target-feature=+avx2`)
//! to get the AVX2 path. The flake's devShell sets `target-cpu=native`.
//!
//! Metadata pointer must be 32-byte aligned (the table allocator guarantees
//! this via `META_ALIGN = 32` on `Layout32`).

#[cfg(all(target_arch = "x86_64", not(miri)))]
use std::arch::x86_64::*;

use super::bitmask::BitMask32;

/// SIMD group operations for 32-slot designs.
///
/// `SLOT_MASK` is applied to all movemask results. For full 32-slot
/// designs (no embedded overflow byte), pass `0xFFFF_FFFF` and the
/// `& SLOT_MASK` compiles away.
pub struct Group32<const SLOT_MASK: u32>;

// ── x86_64 + AVX2 ─────────────────────────────────────────────────────────

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(miri)))]
impl<const SLOT_MASK: u32> Group32<SLOT_MASK> {
    /// SAFETY: `ptr` must be 32-byte aligned.
    #[inline(always)]
    pub unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask32 {
        unsafe {
            let data = _mm256_load_si256(ptr as *const __m256i);
            let needle = _mm256_set1_epi8(value as i8);
            let cmp = _mm256_cmpeq_epi8(data, needle);
            BitMask32(_mm256_movemask_epi8(cmp) as u32 & SLOT_MASK)
        }
    }

    #[inline(always)]
    pub unsafe fn match_empty(ptr: *const u8) -> BitMask32 {
        unsafe {
            let data = _mm256_load_si256(ptr as *const __m256i);
            let zero = _mm256_setzero_si256();
            let cmp = _mm256_cmpeq_epi8(data, zero);
            BitMask32(_mm256_movemask_epi8(cmp) as u32 & SLOT_MASK)
        }
    }

    #[inline(always)]
    pub unsafe fn match_non_empty(ptr: *const u8) -> BitMask32 {
        unsafe {
            let data = _mm256_load_si256(ptr as *const __m256i);
            let zero = _mm256_setzero_si256();
            let cmp = _mm256_cmpeq_epi8(data, zero);
            let mask = _mm256_movemask_epi8(cmp) as u32;
            BitMask32((!mask) & SLOT_MASK)
        }
    }

    #[inline(always)]
    pub unsafe fn match_byte_and_empty(ptr: *const u8, value: u8) -> (BitMask32, BitMask32) {
        unsafe {
            let data = _mm256_load_si256(ptr as *const __m256i);
            let needle = _mm256_set1_epi8(value as i8);
            let zero = _mm256_setzero_si256();
            let match_cmp = _mm256_cmpeq_epi8(data, needle);
            let empty_cmp = _mm256_cmpeq_epi8(data, zero);
            (
                BitMask32(_mm256_movemask_epi8(match_cmp) as u32 & SLOT_MASK),
                BitMask32(_mm256_movemask_epi8(empty_cmp) as u32 & SLOT_MASK),
            )
        }
    }

    #[inline(always)]
    pub unsafe fn prefetch_read(ptr: *const u8) {
        unsafe { _mm_prefetch(ptr as *const i8, _MM_HINT_T0); }
    }

    #[inline(always)]
    pub unsafe fn get_meta(ptr: *const u8, idx: usize) -> u8 {
        unsafe { *ptr.add(idx) }
    }

    #[inline(always)]
    pub unsafe fn set_meta(ptr: *mut u8, idx: usize, value: u8) {
        unsafe { *ptr.add(idx) = value; }
    }
}

// ── x86_64 SSE2 fallback (no AVX2) ────────────────────────────────────────

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2"), not(miri)))]
impl<const SLOT_MASK: u32> Group32<SLOT_MASK> {
    #[inline(always)]
    pub unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask32 {
        unsafe {
            let lo = _mm_load_si128(ptr as *const __m128i);
            let hi = _mm_load_si128(ptr.add(16) as *const __m128i);
            let needle = _mm_set1_epi8(value as i8);
            let lo_cmp = _mm_cmpeq_epi8(lo, needle);
            let hi_cmp = _mm_cmpeq_epi8(hi, needle);
            let lo_m = _mm_movemask_epi8(lo_cmp) as u32 & 0xFFFF;
            let hi_m = _mm_movemask_epi8(hi_cmp) as u32 & 0xFFFF;
            BitMask32(((hi_m << 16) | lo_m) & SLOT_MASK)
        }
    }

    #[inline(always)]
    pub unsafe fn match_empty(ptr: *const u8) -> BitMask32 {
        unsafe { Self::match_byte(ptr, 0) }
    }

    #[inline(always)]
    pub unsafe fn match_non_empty(ptr: *const u8) -> BitMask32 {
        unsafe {
            let lo = _mm_load_si128(ptr as *const __m128i);
            let hi = _mm_load_si128(ptr.add(16) as *const __m128i);
            let zero = _mm_setzero_si128();
            let lo_cmp = _mm_cmpeq_epi8(lo, zero);
            let hi_cmp = _mm_cmpeq_epi8(hi, zero);
            let lo_m = _mm_movemask_epi8(lo_cmp) as u32 & 0xFFFF;
            let hi_m = _mm_movemask_epi8(hi_cmp) as u32 & 0xFFFF;
            BitMask32((!((hi_m << 16) | lo_m)) & SLOT_MASK)
        }
    }

    #[inline(always)]
    pub unsafe fn match_byte_and_empty(ptr: *const u8, value: u8) -> (BitMask32, BitMask32) {
        unsafe { (Self::match_byte(ptr, value), Self::match_empty(ptr)) }
    }

    #[inline(always)]
    pub unsafe fn prefetch_read(ptr: *const u8) {
        unsafe { _mm_prefetch(ptr as *const i8, _MM_HINT_T0); }
    }

    #[inline(always)]
    pub unsafe fn get_meta(ptr: *const u8, idx: usize) -> u8 {
        unsafe { *ptr.add(idx) }
    }

    #[inline(always)]
    pub unsafe fn set_meta(ptr: *mut u8, idx: usize, value: u8) {
        unsafe { *ptr.add(idx) = value; }
    }
}

// ── Scalar fallback (non-x86_64 or Miri) ──────────────────────────────────

#[cfg(any(not(target_arch = "x86_64"), miri))]
impl<const SLOT_MASK: u32> Group32<SLOT_MASK> {
    #[inline(always)]
    pub unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask32 {
        let mut mask = 0u32;
        for i in 0..32 {
            if unsafe { *ptr.add(i) } == value {
                mask |= 1 << i;
            }
        }
        BitMask32(mask & SLOT_MASK)
    }

    #[inline(always)]
    pub unsafe fn match_empty(ptr: *const u8) -> BitMask32 {
        unsafe { Self::match_byte(ptr, 0) }
    }

    #[inline(always)]
    pub unsafe fn match_non_empty(ptr: *const u8) -> BitMask32 {
        let mut mask = 0u32;
        for i in 0..32 {
            if unsafe { *ptr.add(i) } != 0 {
                mask |= 1 << i;
            }
        }
        BitMask32(mask & SLOT_MASK)
    }

    #[inline(always)]
    pub unsafe fn match_byte_and_empty(ptr: *const u8, value: u8) -> (BitMask32, BitMask32) {
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
        unsafe { *ptr.add(idx) = value; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C, align(32))]
    struct Aligned32([u8; 32]);

    fn make() -> Aligned32 { Aligned32([0; 32]) }

    #[test]
    fn empty_group_all_match() {
        let buf = make();
        unsafe {
            let m = Group32::<0xFFFF_FFFF>::match_empty(buf.0.as_ptr());
            assert_eq!(m.0, 0xFFFF_FFFF);
            let n = Group32::<0xFFFF_FFFF>::match_non_empty(buf.0.as_ptr());
            assert_eq!(n.0, 0);
        }
    }

    #[test]
    fn match_byte_finds_hits() {
        let mut buf = make();
        buf.0[3] = 42;
        buf.0[19] = 42;
        buf.0[31] = 42;
        unsafe {
            let m = Group32::<0xFFFF_FFFF>::match_byte(buf.0.as_ptr(), 42);
            let hits: Vec<usize> = m.collect();
            assert_eq!(hits, vec![3, 19, 31]);
        }
    }

    #[test]
    fn match_non_empty_skips_zero() {
        let mut buf = make();
        buf.0[5] = 1;
        buf.0[20] = 99;
        unsafe {
            let m = Group32::<0xFFFF_FFFF>::match_non_empty(buf.0.as_ptr());
            let hits: Vec<usize> = m.collect();
            assert_eq!(hits, vec![5, 20]);
        }
    }

    #[test]
    fn match_byte_and_empty_split() {
        let mut buf = make();
        buf.0[0] = 7;
        buf.0[5] = 7;
        buf.0[10] = 99;
        unsafe {
            let (matches, empties) =
                Group32::<0xFFFF_FFFF>::match_byte_and_empty(buf.0.as_ptr(), 7);
            assert_eq!(matches.collect::<Vec<_>>(), vec![0, 5]);
            // Empty slots: everything except 0, 5, 10
            let empty_set: Vec<usize> = empties.collect();
            assert!(!empty_set.contains(&0));
            assert!(!empty_set.contains(&5));
            assert!(!empty_set.contains(&10));
            assert_eq!(empty_set.len(), 29);
        }
    }
}
