//! 64-slot group operations — one full cache line of metadata per group.
//!
//! Same encoding as IPO (0x00=EMPTY, 0x01=TOMBSTONE, 0x02-0xFF=hash) but
//! with 64 slots per group instead of 16. Uses runtime CPU feature detection
//! to select the best SIMD implementation:
//!
//! - **AVX-512BW**: 1 load (512-bit), mask-compare → 3 ops for match+empty
//! - **AVX2**: 2 loads (256-bit), compare+movemask → 6 ops for match+empty
//! - **SSE2**: 4 loads (128-bit), compare+movemask → 14 ops for match+empty

#[cfg(all(target_arch = "x86_64", not(miri)))]
use std::arch::x86_64::*;

pub use crate::raw::bitmask::BitMask64;

/// Number of element slots per group (one cache line of metadata).
pub const GROUP_SIZE: usize = 64;

/// Total metadata bytes per group.
pub const META_GROUP_BYTES: usize = 64;

/// Metadata byte: slot is empty.
pub const EMPTY: u8 = 0x00;

/// Metadata byte: slot was occupied but has been deleted.
pub const TOMBSTONE: u8 = 0x01;

// ── x86_64 implementation with runtime dispatch ─────────────────────────────

#[cfg(all(target_arch = "x86_64", not(miri)))]
pub struct Group;

#[cfg(all(target_arch = "x86_64", not(miri)))]
impl Group {
    // ── Public dispatch functions ────────────────────────────────────────
    //
    // Each public method dispatches to the best available SIMD tier.
    // For hot loops, callers should use the target_feature-annotated
    // methods directly (avx512/avx2/sse2) to avoid per-call dispatch.

    #[inline(always)]
    pub unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask64 {
        // Hot-path callers (find_by_hash etc.) should use direct avx512/avx2 methods.
        // This dispatch version is for non-hot-path code (iteration, clear, etc.)
        if is_x86_feature_detected!("avx512bw") {
            unsafe { Self::match_byte_avx512(ptr, value) }
        } else if is_x86_feature_detected!("avx2") {
            unsafe { Self::match_byte_avx2(ptr, value) }
        } else {
            unsafe { Self::match_byte_sse2(ptr, value) }
        }
    }

    #[inline(always)]
    pub unsafe fn match_empty(ptr: *const u8) -> BitMask64 {
        if is_x86_feature_detected!("avx512bw") {
            unsafe { Self::match_empty_avx512(ptr) }
        } else if is_x86_feature_detected!("avx2") {
            unsafe { Self::match_empty_avx2(ptr) }
        } else {
            unsafe { Self::match_empty_sse2(ptr) }
        }
    }

    #[inline(always)]
    pub unsafe fn match_byte_and_empty(ptr: *const u8, value: u8) -> (BitMask64, BitMask64) {
        if is_x86_feature_detected!("avx512bw") {
            unsafe { Self::match_byte_and_empty_avx512(ptr, value) }
        } else if is_x86_feature_detected!("avx2") {
            unsafe { Self::match_byte_and_empty_avx2(ptr, value) }
        } else {
            unsafe { Self::match_byte_and_empty_sse2(ptr, value) }
        }
    }

    #[inline(always)]
    pub unsafe fn match_empty_or_tombstone(ptr: *const u8) -> BitMask64 {
        if is_x86_feature_detected!("avx512bw") {
            unsafe { Self::match_empty_or_tombstone_avx512(ptr) }
        } else if is_x86_feature_detected!("avx2") {
            unsafe { Self::match_empty_or_tombstone_avx2(ptr) }
        } else {
            unsafe { Self::match_empty_or_tombstone_sse2(ptr) }
        }
    }

    #[inline(always)]
    pub unsafe fn match_occupied(ptr: *const u8) -> BitMask64 {
        if is_x86_feature_detected!("avx512bw") {
            unsafe { Self::match_occupied_avx512(ptr) }
        } else if is_x86_feature_detected!("avx2") {
            unsafe { Self::match_occupied_avx2(ptr) }
        } else {
            unsafe { Self::match_occupied_sse2(ptr) }
        }
    }

    // ── AVX-512BW: 1 load for 64 bytes ─────────────────────────────────

    #[target_feature(enable = "avx512bw")]
    pub(crate) unsafe fn match_byte_avx512(ptr: *const u8, value: u8) -> BitMask64 {
        unsafe {
            let data = _mm512_load_si512(ptr as *const __m512i);
            let needle = _mm512_set1_epi8(value as i8);
            BitMask64(_mm512_cmpeq_epi8_mask(data, needle))
        }
    }

    #[target_feature(enable = "avx512bw")]
    pub(crate) unsafe fn match_empty_avx512(ptr: *const u8) -> BitMask64 {
        unsafe {
            let data = _mm512_load_si512(ptr as *const __m512i);
            let zero = _mm512_setzero_si512();
            BitMask64(_mm512_cmpeq_epi8_mask(data, zero))
        }
    }

    #[target_feature(enable = "avx512bw")]
    pub(crate) unsafe fn match_byte_and_empty_avx512(
        ptr: *const u8,
        value: u8,
    ) -> (BitMask64, BitMask64) {
        unsafe {
            let data = _mm512_load_si512(ptr as *const __m512i);
            let needle = _mm512_set1_epi8(value as i8);
            let zero = _mm512_setzero_si512();
            (
                BitMask64(_mm512_cmpeq_epi8_mask(data, needle)),
                BitMask64(_mm512_cmpeq_epi8_mask(data, zero)),
            )
        }
    }

    #[target_feature(enable = "avx512bw")]
    pub(crate) unsafe fn match_empty_or_tombstone_avx512(ptr: *const u8) -> BitMask64 {
        unsafe {
            let data = _mm512_load_si512(ptr as *const __m512i);
            let zero = _mm512_setzero_si512();
            let one = _mm512_set1_epi8(1);
            let empty = _mm512_cmpeq_epi8_mask(data, zero);
            let tomb = _mm512_cmpeq_epi8_mask(data, one);
            BitMask64(empty | tomb)
        }
    }

    #[target_feature(enable = "avx512bw")]
    pub(crate) unsafe fn match_occupied_avx512(ptr: *const u8) -> BitMask64 {
        unsafe {
            let data = _mm512_load_si512(ptr as *const __m512i);
            let zero = _mm512_setzero_si512();
            let one = _mm512_set1_epi8(1);
            let empty = _mm512_cmpeq_epi8_mask(data, zero);
            let tomb = _mm512_cmpeq_epi8_mask(data, one);
            BitMask64(!(empty | tomb))
        }
    }

    // ── AVX2: 2 loads for 64 bytes ──────────────────────────────────────

    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn match_byte_avx2(ptr: *const u8, value: u8) -> BitMask64 {
        unsafe {
            let needle = _mm256_set1_epi8(value as i8);
            let d0 = _mm256_load_si256(ptr as *const __m256i);
            let d1 = _mm256_load_si256(ptr.add(32) as *const __m256i);
            let m0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(d0, needle)) as u32 as u64;
            let m1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(d1, needle)) as u32 as u64;
            BitMask64(m0 | (m1 << 32))
        }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn match_empty_avx2(ptr: *const u8) -> BitMask64 {
        unsafe {
            let zero = _mm256_setzero_si256();
            let d0 = _mm256_load_si256(ptr as *const __m256i);
            let d1 = _mm256_load_si256(ptr.add(32) as *const __m256i);
            let m0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(d0, zero)) as u32 as u64;
            let m1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(d1, zero)) as u32 as u64;
            BitMask64(m0 | (m1 << 32))
        }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn match_byte_and_empty_avx2(
        ptr: *const u8,
        value: u8,
    ) -> (BitMask64, BitMask64) {
        unsafe {
            let needle = _mm256_set1_epi8(value as i8);
            let zero = _mm256_setzero_si256();
            let d0 = _mm256_load_si256(ptr as *const __m256i);
            let d1 = _mm256_load_si256(ptr.add(32) as *const __m256i);
            let mm0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(d0, needle)) as u32 as u64;
            let mm1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(d1, needle)) as u32 as u64;
            let em0 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(d0, zero)) as u32 as u64;
            let em1 = _mm256_movemask_epi8(_mm256_cmpeq_epi8(d1, zero)) as u32 as u64;
            (BitMask64(mm0 | (mm1 << 32)), BitMask64(em0 | (em1 << 32)))
        }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn match_empty_or_tombstone_avx2(ptr: *const u8) -> BitMask64 {
        unsafe {
            let zero = _mm256_setzero_si256();
            let one = _mm256_set1_epi8(1);
            let d0 = _mm256_load_si256(ptr as *const __m256i);
            let d1 = _mm256_load_si256(ptr.add(32) as *const __m256i);
            let c = |d: __m256i| -> u64 {
                let e = _mm256_cmpeq_epi8(d, zero);
                let t = _mm256_cmpeq_epi8(d, one);
                _mm256_movemask_epi8(_mm256_or_si256(e, t)) as u32 as u64
            };
            BitMask64(c(d0) | (c(d1) << 32))
        }
    }

    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn match_occupied_avx2(ptr: *const u8) -> BitMask64 {
        unsafe {
            let zero = _mm256_setzero_si256();
            let one = _mm256_set1_epi8(1);
            let d0 = _mm256_load_si256(ptr as *const __m256i);
            let d1 = _mm256_load_si256(ptr.add(32) as *const __m256i);
            let c = |d: __m256i| -> u64 {
                let e = _mm256_cmpeq_epi8(d, zero);
                let t = _mm256_cmpeq_epi8(d, one);
                (!_mm256_movemask_epi8(_mm256_or_si256(e, t)) as u32) as u64
            };
            BitMask64(c(d0) | (c(d1) << 32))
        }
    }

    // ── SSE2: 4 loads for 64 bytes (fallback) ───────────────────────────

    unsafe fn match_byte_sse2(ptr: *const u8, value: u8) -> BitMask64 {
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

    unsafe fn match_empty_sse2(ptr: *const u8) -> BitMask64 {
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

    unsafe fn match_byte_and_empty_sse2(ptr: *const u8, value: u8) -> (BitMask64, BitMask64) {
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

    unsafe fn match_empty_or_tombstone_sse2(ptr: *const u8) -> BitMask64 {
        unsafe {
            let zero = _mm_setzero_si128();
            let one = _mm_set1_epi8(1);
            let mut mask = 0u64;
            for i in 0..4u64 {
                let d = _mm_load_si128(ptr.add((i as usize) * 16) as *const __m128i);
                let e = _mm_cmpeq_epi8(d, zero);
                let t = _mm_cmpeq_epi8(d, one);
                mask |= (_mm_movemask_epi8(_mm_or_si128(e, t)) as u64) << (i * 16);
            }
            BitMask64(mask)
        }
    }

    unsafe fn match_occupied_sse2(ptr: *const u8) -> BitMask64 {
        unsafe {
            let zero = _mm_setzero_si128();
            let one = _mm_set1_epi8(1);
            let mut mask = 0u64;
            for i in 0..4u64 {
                let d = _mm_load_si128(ptr.add((i as usize) * 16) as *const __m128i);
                let e = _mm_cmpeq_epi8(d, zero);
                let t = _mm_cmpeq_epi8(d, one);
                let dead = _mm_or_si128(e, t);
                mask |= ((!_mm_movemask_epi8(dead) as u16) as u64) << (i * 16);
            }
            BitMask64(mask)
        }
    }

    // ── Non-SIMD utilities ──────────────────────────────────────────────

    #[inline(always)]
    pub unsafe fn prefetch_read(ptr: *const u8) {
        unsafe {
            _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
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

// ── Fallback implementation ─────────────────────────────────────────────────

#[cfg(any(not(target_arch = "x86_64"), miri))]
pub struct Group;

#[cfg(any(not(target_arch = "x86_64"), miri))]
impl Group {
    #[inline(always)]
    pub unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask64 {
        let mut mask = 0u64;
        for i in 0..GROUP_SIZE {
            if unsafe { *ptr.add(i) } == value {
                mask |= 1 << i;
            }
        }
        BitMask64(mask)
    }

    #[inline(always)]
    pub unsafe fn match_empty(ptr: *const u8) -> BitMask64 {
        let mut mask = 0u64;
        for i in 0..GROUP_SIZE {
            if unsafe { *ptr.add(i) } == EMPTY {
                mask |= 1 << i;
            }
        }
        BitMask64(mask)
    }

    #[inline(always)]
    pub unsafe fn match_empty_or_tombstone(ptr: *const u8) -> BitMask64 {
        let mut mask = 0u64;
        for i in 0..GROUP_SIZE {
            let b = unsafe { *ptr.add(i) };
            if b <= TOMBSTONE {
                mask |= 1 << i;
            }
        }
        BitMask64(mask)
    }

    #[inline(always)]
    pub unsafe fn match_occupied(ptr: *const u8) -> BitMask64 {
        let mut mask = 0u64;
        for i in 0..GROUP_SIZE {
            if unsafe { *ptr.add(i) } >= 2 {
                mask |= 1 << i;
            }
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
        unsafe {
            *ptr.add(idx) = value;
        }
    }
}
