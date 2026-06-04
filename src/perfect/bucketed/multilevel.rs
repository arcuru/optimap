//! Multi-level bucketed perfect-hash construction.
//!
//! Single-level [`BucketedPhf`](super::BucketedPhf) is bound by a tight
//! Poisson-tail constraint: at λ = 4, K = 16 the per-attempt failure rate
//! is ~1e-6, but pushing λ higher to save memory makes single-shot
//! construction unreliable (λ = 12 fails ~13% of buckets per seed). Multi-
//! level lifts that ceiling by *accepting* level-0 overflow as a routine
//! outcome: oversized buckets are split off into a small level-1 PHF
//! instead of aborting the whole build.
//!
//! ## Algorithm
//!
//! 1. Level 0 hashes every key to a bucket via seed₀ with average load λ₀
//!    (default [`DEFAULT_LAMBDA_0`] = 8 — high enough to halve the slot-
//!    array size vs single-level λ = 4).
//! 2. Buckets whose size exceeds [`SLOTS_PER_BUCKET`] are *overflowed*:
//!    their keys are collected aside and the bucket is marked in a bitset.
//! 3. The overflow keys are fed into a regular single-level
//!    [`BucketedPhf`](super::BucketedPhf) at λ₁ ([`DEFAULT_LAMBDA_1`] = 4).
//!    Its independent seed family means level-1 placement is unaffected by
//!    level-0 collisions.
//!
//! ## Lookup
//!
//! One bucket projection at level 0 + one bit test in the overflow bitset.
//! In the common case (not overflowed), the bit test is the only added
//! cost vs single-level lookup.
//!
//! ## Space
//!
//! At λ₀ = 8, K = 16: `P(Poisson(8) > 16) ≈ 0.003` ⇒ ~0.6% of keys land
//! in level 1. The level-0 tag table (paid in the map wrapper, not this
//! struct) is `8·K/λ₀ = 16` bits/key plus a `1/λ₀ = 0.125` bits/key
//! overflow bitset. Level 1 contributes ~0.2 bits/key tag overhead —
//! negligible.

use super::algorithm::{BucketedPhf, Placements, SLOTS_PER_BUCKET};
use crate::perfect::phf::BuildError;
use crate::perfect::util::{bucket_of, has_duplicate};

/// Default average bucket load at level 0. Tuned so that the expected
/// fraction of keys reaching level 1 is small (~0.6%) while the level-0
/// slot array is ~`K/λ₀ = 2×` rather than the single-level ~4×.
pub const DEFAULT_LAMBDA_0: f64 = 8.0;

/// Default average bucket load at level 1. Matches the single-level
/// default — at the low expected overflow count, λ = 4 keeps level-1
/// construction reliable on the first seed.
pub const DEFAULT_LAMBDA_1: f64 = 4.0;

/// Independent seed base for level 0 of the multi-level family. Distinct
/// from the single-level [`super::BucketedPhf`] seed base so a future
/// composition (single-level over a derived key set, e.g.) does not share
/// a hash family with the multi-level layer above it.
const MULTILEVEL_SEED_BASE: u64 = 0xC4CEB9FE1A85EC53;

/// Two-level bucketed perfect hash function. Level 0 holds most of the
/// keys at high λ; the small overflow tail is handled by an independent
/// single-level PHF.
#[derive(Debug, Clone)]
pub struct MultilevelBucketedPhf {
    level_0: BucketedPhf,
    /// Bitset over level-0 buckets. A set bit means "this bucket
    /// overflowed at build time — redirect to level 1." Length is
    /// `ceil(r_0 / 64)` u64 words.
    overflow: Box<[u64]>,
    /// Present iff at least one bucket overflowed.
    level_1: Option<BucketedPhf>,
}

/// Per-key placement returned by [`MultilevelBucketedPhf::build`]. Length
/// `n`, indexed by input hash position.
///
/// For a key at index `i`:
/// * `level[i] == 0` → key lives in level 0 at `(bucket[i], slot[i])`
/// * `level[i] == 1` → key lives in level 1 at `(bucket[i], slot[i])`
#[derive(Debug)]
pub struct MultilevelPlacements {
    /// 0 → level 0, 1 → level 1.
    pub level: Box<[u8]>,
    /// Bucket index within the level recorded in `level[i]`.
    pub bucket: Box<[u32]>,
    /// Slot within the bucket (in `[0, SLOTS_PER_BUCKET)`).
    pub slot: Box<[u8]>,
}

impl MultilevelBucketedPhf {
    /// Build a multi-level bucketed PHF over `hashes`.
    ///
    /// `lambda_0` and `lambda_1` must each be in `(0, SLOTS_PER_BUCKET)`.
    /// Level-0 construction is single-seed (overflow is expected, not
    /// retried); level-1 construction is a regular single-level
    /// [`BucketedPhf::build`] with its own seed-retry budget.
    pub fn build(
        hashes: &[u64],
        lambda_0: f64,
        lambda_1: f64,
    ) -> Result<(Self, MultilevelPlacements), BuildError> {
        assert!(
            lambda_0 > 0.0 && lambda_0 < SLOTS_PER_BUCKET as f64,
            "lambda_0 must be in (0, {}), got {lambda_0}",
            SLOTS_PER_BUCKET
        );
        assert!(
            lambda_1 > 0.0 && lambda_1 < SLOTS_PER_BUCKET as f64,
            "lambda_1 must be in (0, {}), got {lambda_1}",
            SLOTS_PER_BUCKET
        );

        let n = hashes.len();
        if n == 0 {
            return Ok((
                Self {
                    level_0: BucketedPhf::from_parts(MULTILEVEL_SEED_BASE, 0),
                    overflow: Box::new([]),
                    level_1: None,
                },
                MultilevelPlacements {
                    level: Box::new([]),
                    bucket: Box::new([]),
                    slot: Box::new([]),
                },
            ));
        }

        if has_duplicate(hashes) {
            return Err(BuildError::DuplicateHash);
        }

        let r_0 = (((n as f64) / lambda_0).ceil() as usize).max(1);
        assert!(
            r_0 <= u32::MAX as usize,
            "multilevel PHF does not support more than u32::MAX level-0 buckets"
        );
        let seed_0 = MULTILEVEL_SEED_BASE;
        let r_0_u64 = r_0 as u64;

        // Pass 1: count bucket sizes. u32 fits since n ≤ u32::MAX (the
        // map wrapper enforces this; here we just trust the per-bucket
        // count cannot exceed n).
        let mut bucket_size = vec![0u32; r_0];
        for &h in hashes {
            let b = bucket_of(h, seed_0, r_0_u64);
            bucket_size[b] += 1;
        }

        // Pass 2: build the overflow bitset and count overflow keys.
        let words = r_0.div_ceil(64);
        let mut overflow = vec![0u64; words];
        let mut overflow_keys: u32 = 0;
        for (b, &sz) in bucket_size.iter().enumerate() {
            if sz > SLOTS_PER_BUCKET as u32 {
                overflow[b / 64] |= 1u64 << (b % 64);
                overflow_keys += sz;
            }
        }

        // Pass 3: place every key. Non-overflow keys get level-0 slots
        // assigned via a per-bucket cursor; overflow keys go aside for
        // level-1 construction.
        let mut level = vec![0u8; n];
        let mut bucket = vec![0u32; n];
        let mut slot = vec![0u8; n];
        let mut bucket_cursor = vec![0u8; r_0];
        let mut overflow_hashes: Vec<u64> = Vec::with_capacity(overflow_keys as usize);
        let mut overflow_orig: Vec<u32> = Vec::with_capacity(overflow_keys as usize);

        for (i, &h) in hashes.iter().enumerate() {
            let b = bucket_of(h, seed_0, r_0_u64);
            let word = b / 64;
            let bit = 1u64 << (b % 64);
            if overflow[word] & bit != 0 {
                level[i] = 1;
                overflow_orig.push(i as u32);
                overflow_hashes.push(h);
            } else {
                // level[i] already 0
                bucket[i] = b as u32;
                slot[i] = bucket_cursor[b];
                bucket_cursor[b] += 1;
            }
        }

        let level_0 = BucketedPhf::from_parts(seed_0, r_0 as u32);

        let level_1 = if overflow_hashes.is_empty() {
            None
        } else {
            let (l1_phf, l1_placements) = BucketedPhf::build(&overflow_hashes, lambda_1)?;
            // Patch level-1 placements back into the main per-key arrays
            // so callers see one unified placement view.
            patch_level_1(&overflow_orig, &l1_placements, &mut bucket, &mut slot);
            Some(l1_phf)
        };

        Ok((
            Self {
                level_0,
                overflow: overflow.into_boxed_slice(),
                level_1,
            },
            MultilevelPlacements {
                level: level.into_boxed_slice(),
                bucket: bucket.into_boxed_slice(),
                slot: slot.into_boxed_slice(),
            },
        ))
    }

    /// Map a hash to `(level, bucket_within_level)`. Callers use this to
    /// dispatch into the matching level's tag table + entry array.
    ///
    /// On an empty PHF, returns `(0, 0)` as a benign sentinel; callers
    /// must check emptiness before using the result.
    #[inline]
    pub fn classify(&self, hash: u64) -> (u8, usize) {
        if self.level_0.num_buckets() == 0 {
            return (0, 0);
        }
        let b0 = self.level_0.bucket(hash);
        let word = b0 / 64;
        let bit = 1u64 << (b0 % 64);
        // SAFETY: `b0 < num_buckets ≤ overflow.len() * 64`.
        let overflowed = unsafe { *self.overflow.get_unchecked(word) } & bit != 0;
        if overflowed {
            // SAFETY: an overflow bit is set only if at least one bucket
            // overflowed, in which case `level_1` is `Some`.
            let l1 = unsafe { self.level_1.as_ref().unwrap_unchecked() };
            (1, l1.bucket(hash))
        } else {
            (0, b0)
        }
    }

    /// Number of level-0 buckets.
    #[inline]
    pub fn num_buckets_level_0(&self) -> usize {
        self.level_0.num_buckets()
    }

    /// Number of level-1 buckets (zero if no overflow).
    #[inline]
    pub fn num_buckets_level_1(&self) -> usize {
        self.level_1.as_ref().map(|p| p.num_buckets()).unwrap_or(0)
    }

    /// Count of level-0 buckets that overflowed at build time.
    pub fn overflow_buckets(&self) -> usize {
        self.overflow.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// True iff a level-1 PHF was needed (at least one bucket overflowed).
    #[inline]
    pub fn has_level_1(&self) -> bool {
        self.level_1.is_some()
    }

    /// Approximate space overhead in bits per key for the PHF structure
    /// itself (seed + bucket counts + overflow bitset; level-1 seed if
    /// present). Excludes tag tables and entry arrays — those live in the
    /// map wrapper. Diagnostic only.
    pub fn bits_per_key(&self, n: usize) -> f64 {
        if n == 0 {
            return 0.0;
        }
        let level_0_bits = 96.0; // seed + num_buckets
        let overflow_bits = (self.overflow.len() * 64) as f64;
        let level_1_bits = if self.level_1.is_some() { 96.0 } else { 0.0 };
        (level_0_bits + overflow_bits + level_1_bits) / n as f64
    }
}

#[inline]
fn patch_level_1(
    overflow_orig: &[u32],
    l1_placements: &Placements,
    bucket: &mut [u32],
    slot: &mut [u8],
) {
    for (idx_in_overflow, &orig_idx) in overflow_orig.iter().enumerate() {
        bucket[orig_idx as usize] = l1_placements.bucket[idx_in_overflow];
        slot[orig_idx as usize] = l1_placements.slot[idx_in_overflow];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perfect::util::mix;

    fn make_hashes(n: usize) -> Vec<u64> {
        (0..n as u64)
            .map(|i| mix(i.wrapping_mul(0x9E3779B97F4A7C15), 0x6A09E667F3BCC908))
            .collect()
    }

    fn assert_placements_unique(placements: &MultilevelPlacements, n: usize) {
        let mut seen_l0 = std::collections::HashSet::new();
        let mut seen_l1 = std::collections::HashSet::new();
        for i in 0..n {
            let b = placements.bucket[i] as usize;
            let s = placements.slot[i] as usize;
            assert!(s < SLOTS_PER_BUCKET, "slot {s} out of range at key {i}");
            match placements.level[i] {
                0 => assert!(seen_l0.insert((b, s)), "duplicate level-0 placement at key {i}"),
                1 => assert!(seen_l1.insert((b, s)), "duplicate level-1 placement at key {i}"),
                other => panic!("invalid level marker {other} at key {i}"),
            }
        }
    }

    #[test]
    fn build_empty() {
        let (phf, placements) =
            MultilevelBucketedPhf::build(&[], DEFAULT_LAMBDA_0, DEFAULT_LAMBDA_1).unwrap();
        assert_eq!(phf.num_buckets_level_0(), 0);
        assert_eq!(phf.num_buckets_level_1(), 0);
        assert!(!phf.has_level_1());
        assert_eq!(phf.overflow_buckets(), 0);
        assert!(placements.level.is_empty());
    }

    #[test]
    fn build_single_no_overflow() {
        let hashes = make_hashes(1);
        let (phf, placements) =
            MultilevelBucketedPhf::build(&hashes, DEFAULT_LAMBDA_0, DEFAULT_LAMBDA_1).unwrap();
        assert_eq!(phf.num_buckets_level_0(), 1);
        assert!(!phf.has_level_1());
        assert_eq!(placements.level[0], 0);
        assert_eq!(placements.bucket[0], 0);
        assert_eq!(placements.slot[0], 0);
    }

    #[test]
    fn build_small_round_trip() {
        for &n in &[2usize, 16, 64, 256, 1024] {
            let hashes = make_hashes(n);
            let (phf, placements) =
                MultilevelBucketedPhf::build(&hashes, DEFAULT_LAMBDA_0, DEFAULT_LAMBDA_1).unwrap();
            assert_placements_unique(&placements, n);
            for (i, &h) in hashes.iter().enumerate() {
                let (level, bucket) = phf.classify(h);
                assert_eq!(level, placements.level[i], "level mismatch at key {i} (n={n})");
                assert_eq!(
                    bucket, placements.bucket[i] as usize,
                    "bucket mismatch at key {i} (n={n})"
                );
            }
        }
    }

    #[test]
    fn build_medium_round_trip() {
        let n = 50_000;
        let hashes = make_hashes(n);
        let (phf, placements) =
            MultilevelBucketedPhf::build(&hashes, DEFAULT_LAMBDA_0, DEFAULT_LAMBDA_1).unwrap();
        assert_placements_unique(&placements, n);
        for (i, &h) in hashes.iter().enumerate() {
            let (level, bucket) = phf.classify(h);
            assert_eq!(level, placements.level[i]);
            assert_eq!(bucket, placements.bucket[i] as usize);
        }
    }

    #[test]
    fn build_high_lambda_forces_overflow() {
        // λ₀ = 14 sits well above the K = 16 cap, so P(Poisson(14) > 16)
        // ≈ 0.27 — at n = 5000 we should see a substantial fraction of
        // buckets overflow and a populated level 1.
        let n = 5_000;
        let hashes = make_hashes(n);
        let (phf, placements) = MultilevelBucketedPhf::build(&hashes, 14.0, 4.0).unwrap();
        assert!(
            phf.has_level_1(),
            "λ₀=14 at n={n} should produce overflow"
        );
        assert!(phf.overflow_buckets() > 0);

        let level_1_count = placements.level.iter().filter(|&&l| l == 1).count();
        assert!(
            level_1_count > 0,
            "expected at least one level-1 key, got {level_1_count}"
        );
        assert_placements_unique(&placements, n);

        // Classify should agree with build-time placement for every key.
        for (i, &h) in hashes.iter().enumerate() {
            let (level, bucket) = phf.classify(h);
            assert_eq!(level, placements.level[i], "level mismatch at key {i}");
            assert_eq!(bucket, placements.bucket[i] as usize, "bucket mismatch at key {i}");
        }
    }

    #[test]
    fn duplicate_hash_is_reported() {
        let hashes = vec![1, 2, 3, 4, 1];
        let err =
            MultilevelBucketedPhf::build(&hashes, DEFAULT_LAMBDA_0, DEFAULT_LAMBDA_1).unwrap_err();
        assert_eq!(err, BuildError::DuplicateHash);
    }

    #[test]
    fn lambda_too_high_panics() {
        let hashes = make_hashes(8);
        let result = std::panic::catch_unwind(|| {
            let _ = MultilevelBucketedPhf::build(&hashes, SLOTS_PER_BUCKET as f64, 4.0);
        });
        assert!(result.is_err());
        let result = std::panic::catch_unwind(|| {
            let _ = MultilevelBucketedPhf::build(&hashes, 8.0, SLOTS_PER_BUCKET as f64);
        });
        assert!(result.is_err());
    }

    #[test]
    fn bits_per_key_sane_at_default_lambda() {
        // At λ₀=8 with n=10000, r_0 ≈ 1250 buckets, overflow bitset
        // ≈ 20 u64 words = 1280 bits. Plus ~96 bits each level. Total
        // ≈ 1472 bits over 10000 keys ≈ 0.15 bits/key.
        let n = 10_000;
        let hashes = make_hashes(n);
        let (phf, _) =
            MultilevelBucketedPhf::build(&hashes, DEFAULT_LAMBDA_0, DEFAULT_LAMBDA_1).unwrap();
        let bpk = phf.bits_per_key(n);
        assert!(bpk > 0.0 && bpk < 1.0, "bits_per_key {bpk} outside (0, 1)");
    }
}
