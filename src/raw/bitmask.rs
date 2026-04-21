/// Trait unifying the 16/32/64-slot bitmask types.
///
/// All three widths share the same iteration pattern (Brian Kernighan
/// `mask &= mask - 1`) and interrogation API — only the backing integer
/// width differs. This trait lets `GroupOps::Mask` carry the right width
/// without infecting the whole table code with generics over `u16`/`u32`/`u64`.
pub trait BitMaskOps: Copy + Iterator<Item = usize> {
    fn any_set(self) -> bool;
    fn lowest_set_bit(self) -> Option<usize>;
}

/// 16-bit bitmask: 16-slot (SSE2) groups.
///
/// Set bits correspond to matching slots within the group.
#[derive(Clone, Copy, Debug)]
pub struct BitMask(pub u16);

impl BitMask {
    #[inline]
    pub fn any_set(self) -> bool { self.0 != 0 }

    #[inline]
    pub fn lowest_set_bit(self) -> Option<usize> {
        if self.0 == 0 { None } else { Some(self.0.trailing_zeros() as usize) }
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
            self.0 &= self.0 - 1;
            Some(idx)
        }
    }
}

impl BitMaskOps for BitMask {
    #[inline] fn any_set(self) -> bool { self.any_set() }
    #[inline] fn lowest_set_bit(self) -> Option<usize> { self.lowest_set_bit() }
}

/// 32-bit bitmask: 32-slot (AVX2) groups.
#[derive(Clone, Copy, Debug)]
pub struct BitMask32(pub u32);

impl BitMask32 {
    #[inline]
    pub fn any_set(self) -> bool { self.0 != 0 }

    #[inline]
    pub fn lowest_set_bit(self) -> Option<usize> {
        if self.0 == 0 { None } else { Some(self.0.trailing_zeros() as usize) }
    }
}

impl Iterator for BitMask32 {
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

impl BitMaskOps for BitMask32 {
    #[inline] fn any_set(self) -> bool { self.any_set() }
    #[inline] fn lowest_set_bit(self) -> Option<usize> { self.lowest_set_bit() }
}

/// 64-bit bitmask: 64-slot (AVX-512 / tiered fallback) groups.
#[derive(Clone, Copy, Debug)]
pub struct BitMask64(pub u64);

impl BitMask64 {
    #[inline]
    pub fn any_set(self) -> bool { self.0 != 0 }

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

impl BitMaskOps for BitMask64 {
    #[inline] fn any_set(self) -> bool { self.any_set() }
    #[inline] fn lowest_set_bit(self) -> Option<usize> { self.lowest_set_bit() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bitmask() {
        assert!(!BitMask(0).any_set());
        assert_eq!(BitMask(0).lowest_set_bit(), None);
        assert!(!BitMask32(0).any_set());
        assert!(!BitMask64(0).any_set());
    }

    #[test]
    fn single_bit() {
        let bm = BitMask(0b0000_0000_0000_0100);
        assert!(bm.any_set());
        assert_eq!(bm.lowest_set_bit(), Some(2));
    }

    #[test]
    fn iteration() {
        let bits: Vec<usize> = BitMask(0b0000_0100_0010_0001).collect();
        assert_eq!(bits, vec![0, 5, 10]);

        let bits: Vec<usize> = BitMask32(0x8000_0001).collect();
        assert_eq!(bits, vec![0, 31]);

        let bits: Vec<usize> = BitMask64(0x8000_0000_0000_0001).collect();
        assert_eq!(bits, vec![0, 63]);
    }
}
