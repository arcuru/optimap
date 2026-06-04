//! Shared utilities for perfect-hash construction.
//!
//! The CHD and bucketed families both need (a) a splitmix64-style mixer
//! to derive an independent hash family from `(input_hash, seed)`,
//! (b) a way to project a mixed hash onto a bucket index, and (c) a
//! fail-fast duplicate-hash check before construction. Hoisted here so
//! both modules share one definition.

/// 64-bit mixer based on splitmix64. Combines `h` (the upstream hash)
/// with `seed` (an algorithm-specific seed) to produce an effectively
/// independent hash family — varying `seed` re-permutes outputs across
/// the full u64 range rather than shifting them.
#[inline(always)]
pub(super) fn mix(h: u64, seed: u64) -> u64 {
    let mut x = h ^ seed;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 30;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

/// Project a mixed hash onto a bucket index in `[0, r)`. Uses the top
/// 32 bits so the low byte stays independent of the bucket projection
/// — important when callers re-use the low byte as a slot tag.
#[inline(always)]
pub(super) fn bucket_of(h: u64, seed: u64, r: u64) -> usize {
    let mixed = mix(h, seed);
    ((mixed >> 32) % r) as usize
}

/// Sort-based duplicate check. Used as a fail-fast gate before
/// construction — two equal u64 hashes can never be perfect-hashed by
/// any algorithm. Sort avoids the allocation-heavy HashSet on the hot
/// construction path.
pub(super) fn has_duplicate(hashes: &[u64]) -> bool {
    let mut sorted: Vec<u64> = hashes.to_vec();
    sorted.sort_unstable();
    sorted.windows(2).any(|w| w[0] == w[1])
}
