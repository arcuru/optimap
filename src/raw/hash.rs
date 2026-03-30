/// Post-mixing stage for hash values to compensate for poor hash functions.
///
/// Uses the xmx mixer (multiply-xor-multiply) from Jon Maiga's bit mixer
/// construction for 64-bit hashes.

/// 64-bit xmx post-mixer.
/// Ensures good bit avalanching even from mediocre hash functions.
#[inline]
pub fn mix_hash(mut h: u64) -> u64 {
    // xmx mixer constants from Jon Maiga
    // https://jonkagstrom.com/bit-mixer-construction/
    const C1: u64 = 0xbf58476d1ce4e5b9;
    const C2: u64 = 0x94d049bb133111eb;
    h ^= h >> 30;
    h = h.wrapping_mul(C1);
    h ^= h >> 27;
    h = h.wrapping_mul(C2);
    h ^= h >> 31;
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix_basic() {
        // Mixing should produce different outputs for adjacent inputs
        let a = mix_hash(0);
        let b = mix_hash(1);
        let c = mix_hash(2);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn mix_deterministic() {
        assert_eq!(mix_hash(42), mix_hash(42));
        assert_eq!(mix_hash(0xDEADBEEF), mix_hash(0xDEADBEEF));
    }

    #[test]
    fn mix_avalanche() {
        // Adjacent inputs should differ in many bits
        let a = mix_hash(0);
        let b = mix_hash(1);
        let diff = (a ^ b).count_ones();
        // Good mixer should flip roughly half the bits
        assert!(diff > 16, "only {diff} bits differ, expected good avalanche");
    }
}
