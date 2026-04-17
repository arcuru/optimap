#[cfg(all(target_arch = "x86_64", not(miri)))]
use std::arch::x86_64::*;

use super::bitmask::BitMask;

/// Number of element slots per group.
pub const GROUP_SIZE: usize = 15;

/// Total metadata bytes per group (15 hash bytes + 1 overflow byte).
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

/// A Group operates directly on a pointer to 16 bytes of metadata in-place.
/// No copying — all operations work on the metadata array directly via SSE2.
///
/// Layout in memory: `[hi0, hi1, ..., hi14, overflow]`
///
/// All metadata pointers MUST be 16-byte aligned (enforced by allocator).
#[cfg(all(target_arch = "x86_64", not(miri)))]
pub struct Group;

#[cfg(all(target_arch = "x86_64", not(miri)))]
impl Group {
    /// Return a bitmask of slots matching `value` using SSE2.
    /// Only the lower 15 bits are meaningful.
    /// SAFETY: `ptr` must be 16-byte aligned.
    #[inline(always)]
    pub unsafe fn match_byte(ptr: *const u8, value: u8) -> BitMask {
        unsafe {
            let data = _mm_load_si128(ptr as *const __m128i);
            let needle = _mm_set1_epi8(value as i8);
            let cmp = _mm_cmpeq_epi8(data, needle);
            let mask = _mm_movemask_epi8(cmp) as u16;
            BitMask(mask & 0x7FFF)
        }
    }

    /// Return a bitmask of empty slots.
    /// SAFETY: `ptr` must be 16-byte aligned.
    #[inline(always)]
    pub unsafe fn match_empty(ptr: *const u8) -> BitMask {
        unsafe {
            let data = _mm_load_si128(ptr as *const __m128i);
            let zero = _mm_setzero_si128();
            let cmp = _mm_cmpeq_epi8(data, zero);
            let mask = _mm_movemask_epi8(cmp) as u16;
            BitMask(mask & 0x7FFF)
        }
    }

    /// Return a bitmask of non-empty slots (occupied).
    /// Used for fast iteration — skip groups where this is zero.
    /// SAFETY: `ptr` must be 16-byte aligned.
    #[inline(always)]
    pub unsafe fn match_non_empty(ptr: *const u8) -> BitMask {
        unsafe {
            let data = _mm_load_si128(ptr as *const __m128i);
            let zero = _mm_setzero_si128();
            let cmp = _mm_cmpeq_epi8(data, zero);
            // cmp has 0xFF for empty, 0x00 for non-empty
            // movemask + invert gives us non-empty mask
            let mask = _mm_movemask_epi8(cmp) as u16;
            BitMask((!mask) & 0x7FFF)
        }
    }

    /// Return both match and empty bitmasks with a single SIMD load.
    /// SAFETY: `ptr` must be 16-byte aligned.
    #[inline(always)]
    pub unsafe fn match_byte_and_empty(ptr: *const u8, value: u8) -> (BitMask, BitMask) {
        unsafe {
            let data = _mm_load_si128(ptr as *const __m128i);
            let needle = _mm_set1_epi8(value as i8);
            let zero = _mm_setzero_si128();
            let match_cmp = _mm_cmpeq_epi8(data, needle);
            let empty_cmp = _mm_cmpeq_epi8(data, zero);
            let match_mask = _mm_movemask_epi8(match_cmp) as u16;
            let empty_mask = _mm_movemask_epi8(empty_cmp) as u16;
            (BitMask(match_mask & 0x7FFF), BitMask(empty_mask & 0x7FFF))
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
    #[inline(always)]
    pub unsafe fn has_overflow_bit(ptr: *const u8, bit: u8) -> bool {
        unsafe { (*ptr.add(GROUP_SIZE) & bit) != 0 }
    }

    /// Set a bit in the overflow byte.
    #[inline(always)]
    pub unsafe fn set_overflow_bit(ptr: *mut u8, bit: u8) {
        unsafe {
            *ptr.add(GROUP_SIZE) |= bit;
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

/// Fallback for non-x86_64 platforms.
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
    pub unsafe fn prefetch_read(_ptr: *const u8) {
        // No-op on non-x86_64
    }

    #[inline(always)]
    pub unsafe fn has_overflow_bit(ptr: *const u8, bit: u8) -> bool {
        unsafe { (*ptr.add(GROUP_SIZE) & bit) != 0 }
    }

    #[inline(always)]
    pub unsafe fn set_overflow_bit(ptr: *mut u8, bit: u8) {
        unsafe {
            *ptr.add(GROUP_SIZE) |= bit;
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

/// Initialize a group's metadata to all-empty (16 zero bytes).
#[inline(always)]
pub unsafe fn init_empty(ptr: *mut u8) {
    unsafe {
        std::ptr::write_bytes(ptr, 0, META_GROUP_BYTES);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C, align(16))]
    struct AlignedGroup([u8; META_GROUP_BYTES]);

    impl std::ops::Deref for AlignedGroup {
        type Target = [u8; META_GROUP_BYTES];
        fn deref(&self) -> &Self::Target { &self.0 }
    }
    impl std::ops::DerefMut for AlignedGroup {
        fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
    }
    impl AlignedGroup {
        fn as_ptr(&self) -> *const u8 { self.0.as_ptr() }
        fn as_mut_ptr(&mut self) -> *mut u8 { self.0.as_mut_ptr() }
    }

    fn make_aligned() -> AlignedGroup {
        AlignedGroup([0u8; META_GROUP_BYTES])
    }

    #[test]
    fn empty_group() {
        let buf = make_aligned();
        let ptr = buf.as_ptr();
        unsafe {
            let empty_mask = Group::match_empty(ptr);
            assert_eq!(empty_mask.0, 0x7FFF);
            assert!(!Group::has_overflow_bit(ptr, 0x01));
        }
    }

    #[test]
    fn match_byte_single() {
        let mut buf = make_aligned();
        buf[3] = 42;
        buf[7] = 42;
        buf[11] = 99;
        let ptr = buf.as_ptr();
        unsafe {
            let m = Group::match_byte(ptr, 42);
            let hits: Vec<usize> = m.collect();
            assert_eq!(hits, vec![3, 7]);
        }
    }

    #[test]
    fn overflow_bits() {
        let mut buf = make_aligned();
        let ptr = buf.as_mut_ptr();
        unsafe {
            assert!(!Group::has_overflow_bit(ptr, 0x04));
            Group::set_overflow_bit(ptr, 0x04);
            assert!(Group::has_overflow_bit(ptr, 0x04));
            assert!(!Group::has_overflow_bit(ptr, 0x02));
        }
    }

    #[test]
    fn reduced_hash_values() {
        assert_eq!(reduced_hash(0x00), 8); // 0 maps to 8 (preserves %8)
        assert_eq!(reduced_hash(0x01), 1); // 1 is now valid
        assert_eq!(reduced_hash(0x02), 2);
        assert_eq!(reduced_hash(0xFF), 255);

        for h in 0u64..=255 {
            let r = reduced_hash(h);
            assert!(r >= 1, "reduced_hash({h}) = {r}, must be >= 1");
            assert_eq!(r % 8, (h as u8) % 8);
        }
    }

    #[test]
    fn non_empty_mask() {
        let mut buf = make_aligned();
        buf[2] = 42;
        buf[5] = 99;
        buf[10] = 200;
        unsafe {
            let m = Group::match_non_empty(buf.as_ptr());
            let hits: Vec<usize> = m.collect();
            assert_eq!(hits, vec![2, 5, 10]);
        }
    }
}
