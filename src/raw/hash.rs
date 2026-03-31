use std::hash::{BuildHasher, Hash, Hasher};

/// Marker trait for hash builders that already produce well-avalanched output.
///
/// When a `BuildHasher` implements this trait, the post-mixer is skipped,
/// avoiding redundant work. Hash functions like `ahash::RandomState`
/// produce hashes with good bit distribution and do not benefit from
/// additional mixing.
pub trait IsAvalanching: BuildHasher {}

// foldhash is an avalanching hasher (used as our default)
impl IsAvalanching for foldhash::fast::RandomState {}
impl IsAvalanching for foldhash::fast::FixedState {}
impl IsAvalanching for foldhash::fast::SeedableRandomState {}
impl IsAvalanching for foldhash::quality::RandomState {}
impl IsAvalanching for foldhash::quality::FixedState {}
impl IsAvalanching for foldhash::quality::SeedableRandomState {}

#[cfg(feature = "ahash")]
impl IsAvalanching for ahash::RandomState {}

/// Compute hash for a key, applying the post-mixer.
#[inline(always)]
pub fn hash_with_mix<K: Hash + ?Sized, S: BuildHasher>(key: &K, hash_builder: &S) -> u64 {
    let mut hasher = hash_builder.build_hasher();
    key.hash(&mut hasher);
    mix_hash(hasher.finish())
}

/// Compute hash for a key without post-mixing (for avalanching hashers).
#[inline(always)]
pub fn hash_no_mix<K: Hash + ?Sized, S: BuildHasher>(key: &K, hash_builder: &S) -> u64 {
    let mut hasher = hash_builder.build_hasher();
    key.hash(&mut hasher);
    hasher.finish()
}

/// Fast 64-bit hash post-mixer (Fibonacci hashing).
#[inline(always)]
pub fn mix_hash(h: u64) -> u64 {
    let mixed = h.wrapping_mul(0x9E3779B97F4A7C15);
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
        let mut seen = std::collections::HashSet::new();
        for i in 0u64..256 {
            let low = mix_hash(i) as u8;
            seen.insert(low);
        }
        assert!(seen.len() > 150, "only {} distinct low bytes", seen.len());
    }

    #[test]
    fn non_avalanching_applies_mixer() {
        let hasher = std::hash::RandomState::new();
        let h1 = hash_with_mix(&42u64, &hasher);
        let mut raw_hasher = hasher.build_hasher();
        42u64.hash(&mut raw_hasher);
        let h2 = raw_hasher.finish();
        assert_ne!(h1, h2);
        assert_eq!(h1, mix_hash(h2));
    }
}
