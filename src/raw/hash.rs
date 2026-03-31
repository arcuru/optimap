/// Post-mixing stage for hash values.
///
/// Uses a fast single-round multiply-xor-shift mixer (Fibonacci hashing).
/// This is much cheaper than the full xmx (3 multiplies) while still
/// providing adequate bit mixing for the reduced hash and group index.
/// The strong mixing is delegated to the underlying hash function (SipHash,
/// ahash, etc.) — our mixer just needs to spread bits for group indexing.

/// Fast 64-bit hash post-mixer.
/// A single multiply + xor-shift provides enough mixing to spread
/// hash bits for group indexing and reduced hash extraction.
#[inline(always)]
pub fn mix_hash(h: u64) -> u64 {
    // Fibonacci hashing constant (golden ratio * 2^64)
    // This single multiply spreads all input bits across the output.
    // The high bits (used for group index) get contributions from all input bits.
    let mixed = h.wrapping_mul(0x9E3779B97F4A7C15);
    // XOR-fold to improve the low bits (used for reduced hash)
    mixed ^ (mixed >> 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix_basic() {
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
    fn mix_distinct_low_bits() {
        // Sequential inputs should produce distinct low bytes
        // (important for reduced_hash)
        let mut seen = std::collections::HashSet::new();
        for i in 0u64..256 {
            let low = mix_hash(i) as u8;
            seen.insert(low);
        }
        // Should get good spread — at least 150 distinct low bytes from 256 inputs
        assert!(seen.len() > 150, "only {} distinct low bytes", seen.len());
    }
}
