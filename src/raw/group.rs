use std::simd::{Simd, cmp::SimdPartialEq};

use super::bitmask::BitMask;

/// Number of element slots per group.
pub const GROUP_SIZE: usize = 15;

/// Total metadata bytes per group (15 hash bytes + 1 overflow byte).
pub const META_GROUP_BYTES: usize = 16;

/// Metadata byte: slot is empty.
pub const EMPTY: u8 = 0x00;

/// Metadata byte: sentinel (iteration terminator, placed after last group).
pub const SENTINEL: u8 = 0x01;

/// Compute the reduced hash value from the low byte of a hash.
/// Maps to range [2, 255] while preserving `result % 8 == h % 8`.
#[inline]
pub fn reduced_hash(h: u64) -> u8 {
    let low = (h & 0xFF) as u8;
    // We need: result >= 2 and result % 8 == low % 8
    // Simply: if low < 2, add 8 (not 2!) to preserve mod-8 alignment.
    // Actually, let's think more carefully:
    // low can be 0 or 1, both of which are reserved.
    // 0 % 8 = 0, so we want the smallest value >= 2 with v % 8 == 0, that's 8.
    // 1 % 8 = 1, so we want the smallest value >= 2 with v % 8 == 1, that's 9.
    // For low >= 2, low itself works.
    if low < 2 { low.wrapping_add(8) } else { low }
}

/// Overflow bit index for a given hash value.
#[inline]
pub fn overflow_bit(h: u64) -> u8 {
    1u8 << (h & 7)
}

/// A Group is a 16-byte metadata word: 15 slot bytes + 1 overflow byte.
///
/// Layout in memory: `[hi0, hi1, ..., hi14, overflow]`
///
/// We use std::simd portable SIMD for all comparisons.
#[derive(Clone, Copy)]
#[repr(align(16))]
pub struct Group {
    pub bytes: [u8; META_GROUP_BYTES],
}

impl Group {
    /// Create an empty group (all slots empty, overflow = 0).
    #[inline]
    pub fn empty() -> Self {
        Group {
            bytes: [EMPTY; META_GROUP_BYTES],
        }
    }

    /// Create a sentinel group (used as terminator for iteration).
    #[inline]
    pub fn sentinel() -> Self {
        let mut bytes = [EMPTY; META_GROUP_BYTES];
        bytes[0] = SENTINEL;
        Group { bytes }
    }

    /// Load a Group from a raw pointer. The pointer must be valid for 16 bytes.
    #[inline]
    pub unsafe fn load(ptr: *const u8) -> Self {
        let mut bytes = [0u8; META_GROUP_BYTES];
        unsafe { std::ptr::copy_nonoverlapping(ptr, bytes.as_mut_ptr(), META_GROUP_BYTES) };
        Group { bytes }
    }

    /// Store this group to a raw pointer.
    #[inline]
    pub unsafe fn store(&self, ptr: *mut u8) {
        unsafe { std::ptr::copy_nonoverlapping(self.bytes.as_ptr(), ptr, META_GROUP_BYTES) };
    }

    /// Return a bitmask of slots matching `value` (SIMD comparison).
    /// Only the lower 15 bits of the result are meaningful.
    #[inline]
    pub fn match_byte(&self, value: u8) -> BitMask {
        let data: Simd<u8, 16> = Simd::from_array(self.bytes);
        let needle: Simd<u8, 16> = Simd::splat(value);
        let cmp = data.simd_eq(needle);
        let mask = cmp.to_bitmask() as u16;
        // Mask off the overflow byte (bit 15)
        BitMask(mask & 0x7FFF)
    }

    /// Return a bitmask of empty slots (SIMD comparison against EMPTY).
    #[inline]
    pub fn match_empty(&self) -> BitMask {
        self.match_byte(EMPTY)
    }

    /// Returns true if all 15 slots are occupied (non-empty).
    #[inline]
    pub fn is_full(&self) -> bool {
        !self.match_empty().any_set()
    }

    /// Get the overflow byte.
    #[inline]
    pub fn overflow(&self) -> u8 {
        self.bytes[GROUP_SIZE]
    }

    /// Set a bit in the overflow byte.
    #[inline]
    pub fn set_overflow_bit(&mut self, bit: u8) {
        self.bytes[GROUP_SIZE] |= bit;
    }

    /// Check if a specific overflow bit is set.
    #[inline]
    pub fn has_overflow_bit(&self, bit: u8) -> bool {
        (self.bytes[GROUP_SIZE] & bit) != 0
    }

    /// Get the metadata byte for slot `idx`.
    #[inline]
    pub fn get(&self, idx: usize) -> u8 {
        debug_assert!(idx < GROUP_SIZE);
        self.bytes[idx]
    }

    /// Set the metadata byte for slot `idx`.
    #[inline]
    pub fn set(&mut self, idx: usize, value: u8) {
        debug_assert!(idx < GROUP_SIZE);
        self.bytes[idx] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_group() {
        let g = Group::empty();
        assert!(!g.is_full());
        let empty_mask = g.match_empty();
        assert_eq!(empty_mask.0, 0x7FFF); // all 15 slots empty
        assert_eq!(g.overflow(), 0);
    }

    #[test]
    fn sentinel_group() {
        let g = Group::sentinel();
        assert_eq!(g.get(0), SENTINEL);
        // SENTINEL (1) is not EMPTY (0), so slot 0 is not in match_empty
        let empty_mask = g.match_empty();
        assert_eq!(empty_mask.0, 0x7FFF & !1); // all except slot 0
    }

    #[test]
    fn match_byte_single() {
        let mut g = Group::empty();
        g.set(3, 42);
        g.set(7, 42);
        g.set(11, 99);
        let m = g.match_byte(42);
        let hits: Vec<usize> = m.collect();
        assert_eq!(hits, vec![3, 7]);
    }

    #[test]
    fn full_group() {
        let mut g = Group::empty();
        for i in 0..GROUP_SIZE {
            g.set(i, (i as u8) + 2); // all non-empty
        }
        assert!(g.is_full());
    }

    #[test]
    fn overflow_bits() {
        let mut g = Group::empty();
        assert!(!g.has_overflow_bit(0x01));
        g.set_overflow_bit(0x04); // bit 2
        assert!(g.has_overflow_bit(0x04));
        assert!(!g.has_overflow_bit(0x02));
        assert_eq!(g.overflow(), 0x04);
    }

    #[test]
    fn reduced_hash_values() {
        // 0 -> 8 (preserves mod 8 = 0)
        assert_eq!(reduced_hash(0x00), 8);
        assert_eq!(reduced_hash(0x00) % 8, 0);

        // 1 -> 9 (preserves mod 8 = 1)
        assert_eq!(reduced_hash(0x01), 9);
        assert_eq!(reduced_hash(0x01) % 8, 1);

        // 2 -> 2 (already valid)
        assert_eq!(reduced_hash(0x02), 2);

        // 255 -> 255
        assert_eq!(reduced_hash(0xFF), 255);

        // Check mod-8 preservation for a range
        for h in 0u64..=255 {
            let r = reduced_hash(h);
            assert!(r >= 2, "reduced_hash({h}) = {r} < 2");
            assert_eq!(r % 8, (h as u8) % 8, "mod-8 mismatch for h={h}");
        }
    }
}
