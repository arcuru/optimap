//! Bucketed perfect-hash construction — single-constraint seed-retry.
//!
//! Algorithmic family separate from CHD-MPH. Each key is assigned to a
//! *bucket* of `K = SLOTS_PER_BUCKET = 16` slots (chosen to match the SSE2
//! group width) via `bucket(h, seed) mod r`. Buckets are sized for an
//! average load of `λ < K`; construction succeeds the first time every
//! bucket holds at most `K` keys. On overflow, the seed is incremented to
//! produce a fresh hash family and the assignment is retried.
//!
//! Lookup uses one bucket index, one SIMD tag scan, and a key compare on
//! hits. Tag collisions within a bucket are resolved by the key compare —
//! they don't break the build, only burn a key compare.
//!
//! Construction is O(n) per seed attempt with no per-bucket displacement
//! search (CHD's main cost). At λ = 4, K = 16, the per-attempt failure
//! probability is ~1e-6 so the first seed almost always succeeds; raising
//! λ pushes that probability up sharply (see [`DEFAULT_LAMBDA`]).
//!
//! Per-key storage in the PHF itself: 64 bits seed + 32 bits bucket count
//! ≈ ~0.0001 bits/key in the limit — the cost is paid in the *slot array*
//! (`K/λ ≈ 1.33×n` slots for entries) and the tag table (`8 · K/λ ≈ 10.7`
//! bits per key).

use crate::perfect::phf::BuildError;
use crate::perfect::util::{bucket_of, has_duplicate};

/// Slots per bucket — matches the SSE2 SIMD group width. Hard-coded; the
/// lookup path uses 128-bit intrinsics directly.
pub const SLOTS_PER_BUCKET: usize = 16;

/// Default average bucket load. λ ≪ `SLOTS_PER_BUCKET` because this is a
/// single-shot construction with no per-key overflow handling — a single
/// bucket exceeding K fails the entire seed. The Poisson tail dominates
/// the seed-success probability:
///
///   P(Poisson(λ) > K) ≈ 1e-6  at λ=4, K=16
///   P(Poisson(λ) > K) ≈ 2e-5  at λ=5, K=16
///   P(Poisson(λ) > K) ≈ 0.13  at λ=12, K=16  ← reliably fails at scale
///
/// At λ=4, expected overflowing buckets at n=1M is ~0.22 — first seed
/// almost always succeeds. Multi-level schemes (BBHash, PTHash) get
/// closer to λ ≈ K by overflowing into a secondary structure; that's a
/// distinct algorithmic family.
pub const DEFAULT_LAMBDA: f64 = 4.0;

/// How many independent seed hash families to try before giving up. Build
/// failure within this budget is extremely rare at λ = 4 — increase if
/// you tune λ closer to `SLOTS_PER_BUCKET`.
pub const MAX_SEED_RETRIES: u32 = 64;

/// Splitmix64-style starting seed. Kept distinct from the upstream
/// hasher's identity so the first attempt is a genuinely independent
/// hash family.
const SEED_BASE: u64 = 0xD6E8FEB86659FD93;

/// Bucketed perfect hash function. Compact runtime form: one seed + one
/// bucket count. No per-bucket displacement table — bucket-level
/// perfection is resolved at lookup time by SIMD tag scan + key compare.
#[derive(Debug, Clone)]
pub struct BucketedPhf {
    seed: u64,
    num_buckets: u32,
}

/// Per-key placement returned by [`BucketedPhf::build`]. Length `n`,
/// indexed by input hash position. `bucket * SLOTS_PER_BUCKET + slot` is
/// the linear entry-array index for that key.
#[derive(Debug)]
pub struct Placements {
    /// Bucket index for each input key.
    pub bucket: Box<[u32]>,
    /// Slot within the bucket for each input key (in `[0, SLOTS_PER_BUCKET)`).
    pub slot: Box<[u8]>,
}

impl BucketedPhf {
    /// Reconstruct a [`BucketedPhf`] from its compact runtime form. Used by
    /// the multi-level builder, which runs its own level-0 placement loop
    /// and then stamps the resulting `(seed, num_buckets)` into a
    /// [`BucketedPhf`] for shared lookup machinery.
    #[inline]
    pub(super) fn from_parts(seed: u64, num_buckets: u32) -> Self {
        Self { seed, num_buckets }
    }

    /// Build a bucketed PHF over `hashes`, with average bucket load
    /// `lambda`. `lambda` must satisfy `0 < lambda < SLOTS_PER_BUCKET as f64`.
    pub fn build(hashes: &[u64], lambda: f64) -> Result<(Self, Placements), BuildError> {
        assert!(
            lambda > 0.0 && lambda < SLOTS_PER_BUCKET as f64,
            "lambda must be in (0, {}), got {lambda}",
            SLOTS_PER_BUCKET
        );

        let n = hashes.len();
        if n == 0 {
            return Ok((
                Self {
                    seed: SEED_BASE,
                    num_buckets: 0,
                },
                Placements {
                    bucket: Box::new([]),
                    slot: Box::new([]),
                },
            ));
        }

        if has_duplicate(hashes) {
            return Err(BuildError::DuplicateHash);
        }

        // r = ceil(n / λ), at least 1. Use f64 ceiling to handle
        // non-integer λ exactly.
        let r = (((n as f64) / lambda).ceil() as usize).max(1);
        assert!(
            r <= u32::MAX as usize,
            "bucketed PHF does not support more than u32::MAX buckets"
        );

        let mut bucket = vec![0u32; n];
        let mut slot = vec![0u8; n];
        let mut bucket_size = vec![0u8; r];
        let r_u64 = r as u64;

        'seeds: for seed_try in 0..MAX_SEED_RETRIES {
            let seed = SEED_BASE.wrapping_add(seed_try as u64);
            // Reset bucket fill counters before each attempt.
            for s in bucket_size.iter_mut() {
                *s = 0;
            }

            for (i, &h) in hashes.iter().enumerate() {
                let b = bucket_of(h, seed, r_u64);
                // SAFETY: bucket_of returns a value in [0, r).
                let bs = unsafe { bucket_size.get_unchecked_mut(b) };
                if *bs >= SLOTS_PER_BUCKET as u8 {
                    // This seed overflowed a bucket. Try the next one.
                    continue 'seeds;
                }
                bucket[i] = b as u32;
                slot[i] = *bs;
                *bs += 1;
            }

            return Ok((
                Self {
                    seed,
                    num_buckets: r as u32,
                },
                Placements {
                    bucket: bucket.into_boxed_slice(),
                    slot: slot.into_boxed_slice(),
                },
            ));
        }
        Err(BuildError::Exhausted)
    }

    /// Map a hash to its bucket index.
    #[inline]
    pub fn bucket(&self, hash: u64) -> usize {
        // Empty PHF: callers must check first; return a sentinel.
        if self.num_buckets == 0 {
            return 0;
        }
        bucket_of(hash, self.seed, self.num_buckets as u64)
    }

    /// Number of buckets `r`. Total slot count is `r · SLOTS_PER_BUCKET`.
    #[inline]
    pub fn num_buckets(&self) -> usize {
        self.num_buckets as usize
    }

    /// Approximate space overhead in bits per built key for the PHF
    /// structure itself (not counting tags or entries). Diagnostic only.
    pub fn bits_per_key(&self, n: usize) -> f64 {
        if n == 0 {
            return 0.0;
        }
        // 64 bits seed + 32 bits bucket count, amortized.
        96.0 / (n as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perfect::util::mix;

    fn make_hashes(n: usize) -> Vec<u64> {
        // Well-mixed deterministic input. The PHF only cares that values
        // are unique.
        (0..n as u64)
            .map(|i| mix(i.wrapping_mul(0x9E3779B97F4A7C15), 0x6A09E667F3BCC908))
            .collect()
    }

    #[test]
    fn build_empty() {
        let (phf, placements) = BucketedPhf::build(&[], DEFAULT_LAMBDA).unwrap();
        assert_eq!(phf.num_buckets(), 0);
        assert!(placements.bucket.is_empty());
    }

    #[test]
    fn build_single() {
        let hashes = make_hashes(1);
        let (phf, placements) = BucketedPhf::build(&hashes, DEFAULT_LAMBDA).unwrap();
        assert_eq!(phf.num_buckets(), 1);
        assert_eq!(placements.bucket[0], 0);
        assert_eq!(placements.slot[0], 0);
    }

    #[test]
    fn build_small_round_trip() {
        for &n in &[2usize, 16, 64, 256, 1024] {
            let hashes = make_hashes(n);
            let (phf, placements) = BucketedPhf::build(&hashes, DEFAULT_LAMBDA)
                .expect("build at λ=4 should succeed for well-mixed input");

            // Every key has a valid (bucket, slot) pair within bounds and
            // distinct across input.
            let mut seen = std::collections::HashSet::new();
            for i in 0..n {
                let b = placements.bucket[i] as usize;
                let s = placements.slot[i] as usize;
                assert!(b < phf.num_buckets());
                assert!(s < SLOTS_PER_BUCKET);
                assert!(seen.insert((b, s)), "duplicate placement at key {i}");
            }
        }
    }

    #[test]
    fn build_medium_round_trip() {
        let n = 50_000;
        let hashes = make_hashes(n);
        let (phf, placements) = BucketedPhf::build(&hashes, DEFAULT_LAMBDA).unwrap();
        let mut seen = std::collections::HashSet::new();
        for (i, &h) in hashes.iter().enumerate().take(n) {
            let b = placements.bucket[i] as usize;
            let s = placements.slot[i] as usize;
            assert!(seen.insert((b, s)));
            // Lookup-side bucket should match build-time bucket.
            assert_eq!(phf.bucket(h), b);
        }
    }

    #[test]
    fn duplicate_hash_is_reported() {
        let hashes = vec![1, 2, 3, 4, 1];
        let err = BucketedPhf::build(&hashes, DEFAULT_LAMBDA).unwrap_err();
        assert_eq!(err, BuildError::DuplicateHash);
    }

    #[test]
    fn lambda_too_high_panics() {
        let hashes = make_hashes(8);
        let result = std::panic::catch_unwind(|| {
            let _ = BucketedPhf::build(&hashes, SLOTS_PER_BUCKET as f64);
        });
        assert!(result.is_err());
    }

}
