/// A bitmask over group slots, produced by SIMD comparison operations.
///
/// Each set bit corresponds to a matching slot within the group.
/// Only the lower 15 bits are meaningful (GROUP_SIZE = 15).
#[derive(Clone, Copy, Debug)]
pub struct BitMask(pub u16);

impl BitMask {
    /// Returns true if no bits are set.
    #[inline]
    pub fn any_set(self) -> bool {
        self.0 != 0
    }

    /// Returns the index of the lowest set bit, or None.
    #[inline]
    pub fn lowest_set_bit(self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            Some(self.0.trailing_zeros() as usize)
        }
    }
}

impl Iterator for BitMask {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.0 == 0 {
            None
        } else {
            let idx = self.0.trailing_zeros() as usize;
            self.0 &= self.0 - 1; // clear lowest set bit
            Some(idx)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bitmask() {
        let bm = BitMask(0);
        assert!(!bm.any_set());
        assert_eq!(bm.lowest_set_bit(), None);
    }

    #[test]
    fn single_bit() {
        let bm = BitMask(0b0000_0000_0000_0100);
        assert!(bm.any_set());
        assert_eq!(bm.lowest_set_bit(), Some(2));
    }

    #[test]
    fn iteration() {
        let bm = BitMask(0b0000_0100_0010_0001);
        let bits: Vec<usize> = bm.collect();
        assert_eq!(bits, vec![0, 5, 10]);
    }
}
