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

use super::BucketTags;
use super::algorithm::{BucketedPhf, Placements, SLOTS_PER_BUCKET};
use crate::map::DefaultHashBuilder;
use crate::perfect::phf::BuildError;
use crate::perfect::util::{bucket_of, has_duplicate};
use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::mem::MaybeUninit;

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

// ── MultilevelBucketedConfig ──────────────────────────────────────────────

/// Build configuration for the multi-level bucketed family. Exposes the
/// two average-bucket-load knobs; both default to the values tuned in
/// the algorithm module ([`DEFAULT_LAMBDA_0`], [`DEFAULT_LAMBDA_1`]).
#[derive(Debug, Clone)]
pub struct MultilevelBucketedConfig {
    /// Average level-0 bucket load. Must be in `(0, SLOTS_PER_BUCKET)`.
    /// Higher → fewer slots, more overflow into level 1.
    pub lambda_0: f64,
    /// Average level-1 bucket load. Must be in `(0, SLOTS_PER_BUCKET)`.
    /// Level 1 sees only the overflow tail — small at the default λ₀ = 8
    /// (~0.6 % of keys), so this knob has little effect there. As λ₀
    /// climbs toward `SLOTS_PER_BUCKET`, the overflow fraction grows
    /// rapidly and λ₁ starts to shape level-1 memory + build reliability
    /// directly.
    pub lambda_1: f64,
}

impl Default for MultilevelBucketedConfig {
    fn default() -> Self {
        Self {
            lambda_0: DEFAULT_LAMBDA_0,
            lambda_1: DEFAULT_LAMBDA_1,
        }
    }
}

impl MultilevelBucketedConfig {
    /// Convenience builder: set the level-0 average bucket load.
    pub fn with_lambda_0(mut self, lambda_0: f64) -> Self {
        self.lambda_0 = lambda_0;
        self
    }

    /// Convenience builder: set the level-1 average bucket load.
    pub fn with_lambda_1(mut self, lambda_1: f64) -> Self {
        self.lambda_1 = lambda_1;
        self
    }
}

// ── PerfectMapMultilevelBucketed ──────────────────────────────────────────

/// Read-only map backed by a multi-level bucketed perfect hash with SIMD
/// tag scan at each level. Sibling to [`PerfectMapBucketed`](super::PerfectMapBucketed);
/// pushes the level-0 average load to ~`K/2` to halve the slot-array
/// memory cost, falling back to an independent level-1 structure for the
/// small overflow tail.
///
/// # Example
///
/// ```
/// use optimap::PerfectMapMultilevelBucketed;
///
/// let m: PerfectMapMultilevelBucketed<u64, &'static str> =
///     PerfectMapMultilevelBucketed::from_iter_perfect(
///         [(1u64, "one"), (2, "two"), (3, "three")],
///     )
///     .unwrap();
/// assert_eq!(m.get(&2), Some(&"two"));
/// assert_eq!(m.get(&999), None);
/// ```
pub struct PerfectMapMultilevelBucketed<K, V, S = DefaultHashBuilder> {
    tags_l0: Box<[BucketTags]>,
    entries_l0: Box<[MaybeUninit<(K, V)>]>,
    tags_l1: Box<[BucketTags]>,
    entries_l1: Box<[MaybeUninit<(K, V)>]>,
    phf: MultilevelBucketedPhf,
    hash_builder: S,
    len: usize,
}

impl<K, V> PerfectMapMultilevelBucketed<K, V, DefaultHashBuilder>
where
    K: Hash + Eq,
{
    /// Build with the default hash builder and default `(λ₀, λ₁)`.
    pub fn from_iter_perfect<I>(entries: I) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        Self::from_entries(
            entries,
            DefaultHashBuilder::default(),
            &MultilevelBucketedConfig::default(),
        )
    }
}

impl<K, V, S> PerfectMapMultilevelBucketed<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Build with a custom hash builder and configuration.
    pub fn from_entries<I>(
        entries: I,
        hash_builder: S,
        config: &MultilevelBucketedConfig,
    ) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let entries: Vec<(K, V)> = entries.into_iter().collect();
        let n = entries.len();

        if n == 0 {
            return Ok(Self {
                tags_l0: Box::new([]),
                entries_l0: Box::new([]),
                tags_l1: Box::new([]),
                entries_l1: Box::new([]),
                phf: MultilevelBucketedPhf::build(&[], config.lambda_0, config.lambda_1)?.0,
                hash_builder,
                len: 0,
            });
        }

        let hashes: Vec<u64> = entries
            .iter()
            .map(|(k, _)| hash_builder.hash_one(k))
            .collect();
        let (phf, placements) =
            MultilevelBucketedPhf::build(&hashes, config.lambda_0, config.lambda_1)?;

        let r0 = phf.num_buckets_level_0();
        let r1 = phf.num_buckets_level_1();
        let slots_l0 = r0 * SLOTS_PER_BUCKET;
        let slots_l1 = r1 * SLOTS_PER_BUCKET;

        let mut tags_l0 = vec![BucketTags::EMPTY; r0];
        let mut entries_l0: Vec<MaybeUninit<(K, V)>> =
            (0..slots_l0).map(|_| MaybeUninit::uninit()).collect();
        let mut tags_l1 = vec![BucketTags::EMPTY; r1];
        let mut entries_l1: Vec<MaybeUninit<(K, V)>> =
            (0..slots_l1).map(|_| MaybeUninit::uninit()).collect();

        for ((k, v), (i, h)) in entries.into_iter().zip(hashes.iter().enumerate()) {
            let bucket = placements.bucket[i] as usize;
            let slot = placements.slot[i] as usize;
            let tag = crate::hash_tag(*h);
            match placements.level[i] {
                0 => {
                    debug_assert_eq!(tags_l0[bucket].0[slot], 0);
                    tags_l0[bucket].0[slot] = tag;
                    entries_l0[bucket * SLOTS_PER_BUCKET + slot].write((k, v));
                }
                1 => {
                    debug_assert_eq!(tags_l1[bucket].0[slot], 0);
                    tags_l1[bucket].0[slot] = tag;
                    entries_l1[bucket * SLOTS_PER_BUCKET + slot].write((k, v));
                }
                other => unreachable!("invalid level marker {other} from PHF placements"),
            }
        }

        Ok(Self {
            tags_l0: tags_l0.into_boxed_slice(),
            entries_l0: entries_l0.into_boxed_slice(),
            tags_l1: tags_l1.into_boxed_slice(),
            entries_l1: entries_l1.into_boxed_slice(),
            phf,
            hash_builder,
            len: n,
        })
    }

    /// Look up `key`. Returns `Some(&V)` if `key` was in the construction
    /// key set, `None` otherwise.
    #[inline]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if self.len == 0 {
            return None;
        }
        let hash = self.hash_builder.hash_one(key);
        let (level, bucket) = self.phf.classify(hash);
        let tag = crate::hash_tag(hash);

        let (tags, entries) = if level == 0 {
            (&self.tags_l0, &self.entries_l0)
        } else {
            (&self.tags_l1, &self.entries_l1)
        };

        // SAFETY: classify guarantees bucket < tags.len() in each level
        // (length matches num_buckets_level_{0,1}).
        let bucket_tags = unsafe { tags.get_unchecked(bucket) };
        let mask = unsafe { crate::raw::group::match_byte_full_16(bucket_tags.0.as_ptr(), tag) };

        let base = bucket * SLOTS_PER_BUCKET;
        for slot in mask {
            // SAFETY: tag != 0 means this slot was initialized at build time.
            let entry = unsafe { entries.get_unchecked(base + slot).assume_init_ref() };
            if entry.0.borrow() == key {
                return Some(&entry.1);
            }
        }
        None
    }

    /// True iff `key` was in the construction key set.
    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True iff `len == 0`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Level-0 bucket count.
    #[inline]
    pub fn num_buckets_level_0(&self) -> usize {
        self.phf.num_buckets_level_0()
    }

    /// Level-1 bucket count (0 if no overflow).
    #[inline]
    pub fn num_buckets_level_1(&self) -> usize {
        self.phf.num_buckets_level_1()
    }

    /// True iff at least one bucket overflowed and a level-1 substructure
    /// was built.
    #[inline]
    pub fn has_level_1(&self) -> bool {
        self.phf.has_level_1()
    }

    /// Iterate over `(&K, &V)` in (level, slot) order — an implementation-
    /// defined permutation of insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> + '_ {
        let l0 = iter_level(&self.tags_l0, &self.entries_l0);
        let l1 = iter_level(&self.tags_l1, &self.entries_l1);
        l0.chain(l1)
    }

    /// Iterate over `&K`.
    pub fn keys(&self) -> impl Iterator<Item = &K> + '_ {
        self.iter().map(|(k, _)| k)
    }

    /// Iterate over `&V`.
    pub fn values(&self) -> impl Iterator<Item = &V> + '_ {
        self.iter().map(|(_, v)| v)
    }

    /// Approximate space overhead in bits per key for the PHF data
    /// structure alone (seed + bucket counts + overflow bitset). Excludes
    /// tag tables and entries. Diagnostic only.
    pub fn phf_bits_per_key(&self) -> f64 {
        self.phf.bits_per_key(self.len)
    }

    /// Total tag-table bits per key across both levels, including
    /// empty-slot bytes. At λ₀ = 8, K = 16 this is ~16 bits/key from
    /// level 0 plus a small (~0.1 bits/key) contribution from level 1.
    pub fn tag_bits_per_key(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        let bits = (self.tags_l0.len() + self.tags_l1.len()) * SLOTS_PER_BUCKET * 8;
        bits as f64 / self.len as f64
    }

    /// Access the hash builder used at construction.
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }
}

fn iter_level<'a, K, V>(
    tags: &'a [BucketTags],
    entries: &'a [MaybeUninit<(K, V)>],
) -> impl Iterator<Item = (&'a K, &'a V)> + 'a {
    tags.iter().enumerate().flat_map(move |(b, bucket_tags)| {
        let base = b * SLOTS_PER_BUCKET;
        bucket_tags
            .0
            .iter()
            .enumerate()
            .filter_map(move |(s, &tag)| {
                if tag != 0 {
                    let entry = unsafe { entries.get_unchecked(base + s).assume_init_ref() };
                    Some((&entry.0, &entry.1))
                } else {
                    None
                }
            })
    })
}

impl<K, V, S> Drop for PerfectMapMultilevelBucketed<K, V, S> {
    fn drop(&mut self) {
        if std::mem::needs_drop::<(K, V)>() {
            drop_level(&self.tags_l0, &mut self.entries_l0);
            drop_level(&self.tags_l1, &mut self.entries_l1);
        }
    }
}

fn drop_level<T>(tags: &[BucketTags], entries: &mut [MaybeUninit<T>]) {
    for (b, bucket_tags) in tags.iter().enumerate() {
        let base = b * SLOTS_PER_BUCKET;
        for (s, &tag) in bucket_tags.0.iter().enumerate() {
            if tag != 0 {
                // SAFETY: tag != 0 ⇔ this slot was initialized at build
                // time and has not been moved out of.
                unsafe {
                    entries[base + s].assume_init_drop();
                }
            }
        }
    }
}

impl<K, V, S> fmt::Debug for PerfectMapMultilevelBucketed<K, V, S>
where
    K: Hash + Eq + fmt::Debug,
    V: fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

// ── PerfectSetMultilevelBucketed ──────────────────────────────────────────

/// Read-only set backed by a multi-level bucketed perfect hash. Sibling
/// to [`PerfectSetBucketed`](super::PerfectSetBucketed) with the same
/// memory/speed trade-off as [`PerfectMapMultilevelBucketed`].
///
/// # Example
///
/// ```
/// use optimap::PerfectSetMultilevelBucketed;
///
/// let s: PerfectSetMultilevelBucketed<&'static str> =
///     PerfectSetMultilevelBucketed::from_iter_perfect(["alpha", "beta", "gamma"]).unwrap();
/// assert!(s.contains("beta"));
/// assert!(!s.contains("delta"));
/// ```
pub struct PerfectSetMultilevelBucketed<K, S = DefaultHashBuilder> {
    tags_l0: Box<[BucketTags]>,
    keys_l0: Box<[MaybeUninit<K>]>,
    tags_l1: Box<[BucketTags]>,
    keys_l1: Box<[MaybeUninit<K>]>,
    phf: MultilevelBucketedPhf,
    hash_builder: S,
    len: usize,
}

impl<K> PerfectSetMultilevelBucketed<K, DefaultHashBuilder>
where
    K: Hash + Eq,
{
    /// Build with the default hash builder and default `(λ₀, λ₁)`.
    pub fn from_iter_perfect<I>(keys: I) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = K>,
    {
        Self::from_keys(
            keys,
            DefaultHashBuilder::default(),
            &MultilevelBucketedConfig::default(),
        )
    }
}

impl<K, S> PerfectSetMultilevelBucketed<K, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Build with a custom hash builder and configuration.
    pub fn from_keys<I>(
        keys: I,
        hash_builder: S,
        config: &MultilevelBucketedConfig,
    ) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = K>,
    {
        let keys: Vec<K> = keys.into_iter().collect();
        let n = keys.len();

        if n == 0 {
            return Ok(Self {
                tags_l0: Box::new([]),
                keys_l0: Box::new([]),
                tags_l1: Box::new([]),
                keys_l1: Box::new([]),
                phf: MultilevelBucketedPhf::build(&[], config.lambda_0, config.lambda_1)?.0,
                hash_builder,
                len: 0,
            });
        }

        let hashes: Vec<u64> = keys.iter().map(|k| hash_builder.hash_one(k)).collect();
        let (phf, placements) =
            MultilevelBucketedPhf::build(&hashes, config.lambda_0, config.lambda_1)?;

        let r0 = phf.num_buckets_level_0();
        let r1 = phf.num_buckets_level_1();
        let slots_l0 = r0 * SLOTS_PER_BUCKET;
        let slots_l1 = r1 * SLOTS_PER_BUCKET;

        let mut tags_l0 = vec![BucketTags::EMPTY; r0];
        let mut keys_l0: Vec<MaybeUninit<K>> =
            (0..slots_l0).map(|_| MaybeUninit::uninit()).collect();
        let mut tags_l1 = vec![BucketTags::EMPTY; r1];
        let mut keys_l1: Vec<MaybeUninit<K>> =
            (0..slots_l1).map(|_| MaybeUninit::uninit()).collect();

        for (k, (i, h)) in keys.into_iter().zip(hashes.iter().enumerate()) {
            let bucket = placements.bucket[i] as usize;
            let slot = placements.slot[i] as usize;
            let tag = crate::hash_tag(*h);
            match placements.level[i] {
                0 => {
                    debug_assert_eq!(tags_l0[bucket].0[slot], 0);
                    tags_l0[bucket].0[slot] = tag;
                    keys_l0[bucket * SLOTS_PER_BUCKET + slot].write(k);
                }
                1 => {
                    debug_assert_eq!(tags_l1[bucket].0[slot], 0);
                    tags_l1[bucket].0[slot] = tag;
                    keys_l1[bucket * SLOTS_PER_BUCKET + slot].write(k);
                }
                other => unreachable!("invalid level marker {other} from PHF placements"),
            }
        }

        Ok(Self {
            tags_l0: tags_l0.into_boxed_slice(),
            keys_l0: keys_l0.into_boxed_slice(),
            tags_l1: tags_l1.into_boxed_slice(),
            keys_l1: keys_l1.into_boxed_slice(),
            phf,
            hash_builder,
            len: n,
        })
    }

    /// True iff `key` was in the construction key set.
    #[inline]
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Borrow the stored key equal to `key`, or `None`. Useful for the
    /// intern pattern.
    #[inline]
    pub fn get<Q>(&self, key: &Q) -> Option<&K>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if self.len == 0 {
            return None;
        }
        let hash = self.hash_builder.hash_one(key);
        let (level, bucket) = self.phf.classify(hash);
        let tag = crate::hash_tag(hash);

        let (tags, stored) = if level == 0 {
            (&self.tags_l0, &self.keys_l0)
        } else {
            (&self.tags_l1, &self.keys_l1)
        };

        let bucket_tags = unsafe { tags.get_unchecked(bucket) };
        let mask = unsafe { crate::raw::group::match_byte_full_16(bucket_tags.0.as_ptr(), tag) };

        let base = bucket * SLOTS_PER_BUCKET;
        for slot in mask {
            let entry = unsafe { stored.get_unchecked(base + slot).assume_init_ref() };
            if entry.borrow() == key {
                return Some(entry);
            }
        }
        None
    }

    /// Number of keys.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True iff empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Iterate over `&K` in (level, slot) order.
    pub fn iter(&self) -> impl Iterator<Item = &K> + '_ {
        let l0 = iter_level_set(&self.tags_l0, &self.keys_l0);
        let l1 = iter_level_set(&self.tags_l1, &self.keys_l1);
        l0.chain(l1)
    }
}

fn iter_level_set<'a, K>(
    tags: &'a [BucketTags],
    stored: &'a [MaybeUninit<K>],
) -> impl Iterator<Item = &'a K> + 'a {
    tags.iter().enumerate().flat_map(move |(b, bucket_tags)| {
        let base = b * SLOTS_PER_BUCKET;
        bucket_tags
            .0
            .iter()
            .enumerate()
            .filter_map(move |(s, &tag)| {
                if tag != 0 {
                    Some(unsafe { stored.get_unchecked(base + s).assume_init_ref() })
                } else {
                    None
                }
            })
    })
}

impl<K, S> Drop for PerfectSetMultilevelBucketed<K, S> {
    fn drop(&mut self) {
        if std::mem::needs_drop::<K>() {
            drop_level(&self.tags_l0, &mut self.keys_l0);
            drop_level(&self.tags_l1, &mut self.keys_l1);
        }
    }
}

impl<K, S> fmt::Debug for PerfectSetMultilevelBucketed<K, S>
where
    K: Hash + Eq + fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
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
                0 => assert!(
                    seen_l0.insert((b, s)),
                    "duplicate level-0 placement at key {i}"
                ),
                1 => assert!(
                    seen_l1.insert((b, s)),
                    "duplicate level-1 placement at key {i}"
                ),
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
                assert_eq!(
                    level, placements.level[i],
                    "level mismatch at key {i} (n={n})"
                );
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
        assert!(phf.has_level_1(), "λ₀=14 at n={n} should produce overflow");
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
            assert_eq!(
                bucket, placements.bucket[i] as usize,
                "bucket mismatch at key {i}"
            );
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

    // ── PerfectMapMultilevelBucketed ──────────────────────────────────────

    fn entries_u64(n: u64) -> Vec<(u64, u64)> {
        (0..n).map(|i| (i, i.wrapping_mul(31))).collect()
    }

    #[test]
    fn map_empty_round_trip() {
        let m = PerfectMapMultilevelBucketed::<u64, u64>::from_iter_perfect(std::iter::empty())
            .unwrap();
        assert_eq!(m.len(), 0);
        assert!(m.is_empty());
        assert!(!m.has_level_1());
        assert_eq!(m.get(&0), None);
    }

    #[test]
    fn map_single_round_trip() {
        let m = PerfectMapMultilevelBucketed::<u64, &str>::from_iter_perfect([(42u64, "answer")])
            .unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m.get(&42), Some(&"answer"));
        assert_eq!(m.get(&0), None);
    }

    #[test]
    fn map_small_round_trip() {
        let n = 256u64;
        let entries = entries_u64(n);
        let m =
            PerfectMapMultilevelBucketed::<u64, u64>::from_iter_perfect(entries.clone()).unwrap();
        for (k, v) in &entries {
            assert_eq!(m.get(k), Some(v));
        }
        for k in n..n + 50 {
            assert_eq!(m.get(&k), None);
        }
    }

    #[test]
    fn map_medium_round_trip() {
        let n = 50_000u64;
        let entries = entries_u64(n);
        let m =
            PerfectMapMultilevelBucketed::<u64, u64>::from_iter_perfect(entries.clone()).unwrap();
        assert_eq!(m.len(), n as usize);
        for (k, v) in &entries {
            assert_eq!(m.get(k), Some(v));
        }
    }

    #[test]
    fn map_high_lambda_uses_level_1() {
        // λ₀=14 forces substantial overflow; verify that lookup goes
        // through level 1 correctly for the keys placed there.
        let n = 5_000u64;
        let entries = entries_u64(n);
        let config = MultilevelBucketedConfig::default().with_lambda_0(14.0);
        let m = PerfectMapMultilevelBucketed::<u64, u64>::from_entries(
            entries.clone(),
            DefaultHashBuilder::default(),
            &config,
        )
        .unwrap();
        assert!(m.has_level_1(), "λ₀=14 should produce overflow at n={n}");
        for (k, v) in &entries {
            assert_eq!(m.get(k), Some(v), "key={k}");
        }
    }

    #[test]
    fn map_string_keys() {
        let words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"];
        let entries: Vec<(String, usize)> = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.to_string(), i))
            .collect();
        let m = PerfectMapMultilevelBucketed::<String, usize>::from_iter_perfect(entries.clone())
            .unwrap();
        for (k, v) in &entries {
            assert_eq!(m.get(k.as_str()), Some(v));
        }
        assert_eq!(m.get("missing"), None);
    }

    #[test]
    fn map_drops_values_on_drop() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        struct Dropper(u32);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }
        DROPS.store(0, Ordering::Relaxed);
        {
            // Use λ₀=14 so the drop path exercises BOTH levels.
            let entries: Vec<(u64, Dropper)> =
                (0..500u64).map(|i| (i, Dropper(i as u32))).collect();
            let config = MultilevelBucketedConfig::default().with_lambda_0(14.0);
            let _m = PerfectMapMultilevelBucketed::<u64, Dropper>::from_entries(
                entries,
                DefaultHashBuilder::default(),
                &config,
            )
            .unwrap();
        }
        assert_eq!(DROPS.load(Ordering::Relaxed), 500);
    }

    #[test]
    fn map_iter_returns_all_entries() {
        let entries = entries_u64(200);
        // High λ₀ to exercise both levels in the iterator.
        let config = MultilevelBucketedConfig::default().with_lambda_0(14.0);
        let m = PerfectMapMultilevelBucketed::<u64, u64>::from_entries(
            entries.clone(),
            DefaultHashBuilder::default(),
            &config,
        )
        .unwrap();
        let mut got: Vec<(u64, u64)> = m.iter().map(|(k, v)| (*k, *v)).collect();
        got.sort();
        let mut want = entries.clone();
        want.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn map_duplicate_keys_rejected() {
        let entries = vec![(1u64, 1u64), (2, 2), (1, 3)];
        let err = PerfectMapMultilevelBucketed::<u64, u64>::from_iter_perfect(entries).unwrap_err();
        assert_eq!(err, BuildError::DuplicateHash);
    }

    // ── PerfectSetMultilevelBucketed ──────────────────────────────────────

    #[test]
    fn set_empty_round_trip() {
        let s = PerfectSetMultilevelBucketed::<u64>::from_iter_perfect(std::iter::empty()).unwrap();
        assert_eq!(s.len(), 0);
        assert!(!s.contains(&0));
    }

    #[test]
    fn set_small_round_trip() {
        let keys: Vec<u64> = (0..256).collect();
        let s = PerfectSetMultilevelBucketed::<u64>::from_iter_perfect(keys.clone()).unwrap();
        for k in &keys {
            assert!(s.contains(k));
        }
        for k in 256..300 {
            assert!(!s.contains(&k));
        }
    }

    #[test]
    fn set_high_lambda_uses_level_1() {
        let n = 5_000u64;
        let keys: Vec<u64> = (0..n).collect();
        let config = MultilevelBucketedConfig::default().with_lambda_0(14.0);
        let s = PerfectSetMultilevelBucketed::<u64>::from_keys(
            keys.clone(),
            DefaultHashBuilder::default(),
            &config,
        )
        .unwrap();
        for k in &keys {
            assert!(s.contains(k));
        }
        assert!(!s.contains(&(n + 1)));
    }

    #[test]
    fn set_intern_pattern() {
        let words = ["alpha", "beta", "gamma"];
        let entries: Vec<String> = words.iter().map(|w| w.to_string()).collect();
        let s = PerfectSetMultilevelBucketed::<String>::from_iter_perfect(entries).unwrap();
        let interned = s.get("beta").expect("beta is in the set");
        assert_eq!(interned, "beta");
        assert!(!s.contains("missing"));
    }

    #[test]
    fn set_drops_keys_on_drop() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        #[derive(Hash, PartialEq, Eq)]
        struct K(u32);
        impl Drop for K {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::Relaxed);
            }
        }
        DROPS.store(0, Ordering::Relaxed);
        {
            let keys: Vec<K> = (0..200u32).map(K).collect();
            let config = MultilevelBucketedConfig::default().with_lambda_0(14.0);
            let _s = PerfectSetMultilevelBucketed::<K>::from_keys(
                keys,
                DefaultHashBuilder::default(),
                &config,
            )
            .unwrap();
        }
        assert_eq!(DROPS.load(Ordering::Relaxed), 200);
    }
}
