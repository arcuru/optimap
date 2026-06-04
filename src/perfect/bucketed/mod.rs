//! Bucketed perfect-hash family — Swiss-table layout with build-time
//! bucket-fill guarantee.
//!
//! Sibling family to the CHD-MPH-based [`PerfectMap`](crate::PerfectMap).
//! Where CHD-MPH targets minimal (m = n) table size at the cost of a
//! per-bucket displacement search, this family targets fast construction
//! and SIMD-friendly miss-path lookup at the cost of `~K/λ ≈ 1.33×` slot
//! overhead.
//!
//! See [`docs/src/roadmap.md`](../../docs/src/roadmap.md) for the design
//! discussion and predicted comparison vs CHD-MPH.

mod algorithm;

pub use algorithm::{BucketedPhf, DEFAULT_LAMBDA, MAX_SEED_RETRIES, Placements, SLOTS_PER_BUCKET};

use crate::map::DefaultHashBuilder;
use crate::perfect::phf::BuildError;
use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::mem::MaybeUninit;

/// Build configuration for the bucketed family. Currently exposes only the
/// average bucket load λ — the SIMD group width K is fixed at 16.
#[derive(Debug, Clone)]
pub struct BucketedConfig {
    /// Average target keys per bucket. Must be in `(0, SLOTS_PER_BUCKET)`.
    /// Lower λ → larger slot array, easier construction. Higher λ → denser,
    /// more seed-retry pressure. Default [`DEFAULT_LAMBDA`] (12).
    pub lambda: f64,
}

impl Default for BucketedConfig {
    fn default() -> Self {
        Self {
            lambda: DEFAULT_LAMBDA,
        }
    }
}

impl BucketedConfig {
    /// Convenience builder: set the average bucket load.
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }
}

/// 16-byte aligned tag slab — one bucket's worth. `repr(align(16))` lets
/// the SIMD load path use the aligned `_mm_load_si128` intrinsic.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct BucketTags([u8; SLOTS_PER_BUCKET]);

impl BucketTags {
    const EMPTY: Self = Self([0u8; SLOTS_PER_BUCKET]);
}

// ── PerfectMapBucketed ────────────────────────────────────────────────────

/// Read-only map backed by a bucketed perfect hash with SIMD tag scan.
///
/// Construction targets fast build (no per-bucket displacement search)
/// and miss-path SIMD rejection (tag scan terminates without touching the
/// value array). The trade-off vs [`PerfectMap`](crate::PerfectMap) is
/// `~K/λ ≈ 1.33×` slot overhead and `8 bits/key` of tag table.
///
/// # Example
///
/// ```
/// use optimap::PerfectMapBucketed;
///
/// let m: PerfectMapBucketed<u64, &'static str> =
///     PerfectMapBucketed::from_iter_perfect([(1u64, "one"), (2, "two"), (3, "three")])
///         .unwrap();
/// assert_eq!(m.get(&2), Some(&"two"));
/// assert_eq!(m.get(&999), None);
/// ```
pub struct PerfectMapBucketed<K, V, S = DefaultHashBuilder> {
    tags: Box<[BucketTags]>,
    entries: Box<[MaybeUninit<(K, V)>]>,
    phf: BucketedPhf,
    hash_builder: S,
    len: usize,
}

impl<K, V> PerfectMapBucketed<K, V, DefaultHashBuilder>
where
    K: Hash + Eq,
{
    /// Build with the default hash builder and default λ.
    pub fn from_iter_perfect<I>(entries: I) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        Self::from_entries(entries, DefaultHashBuilder::default(), &BucketedConfig::default())
    }
}

impl<K, V, S> PerfectMapBucketed<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Build with a custom hash builder and configuration.
    pub fn from_entries<I>(
        entries: I,
        hash_builder: S,
        config: &BucketedConfig,
    ) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let entries: Vec<(K, V)> = entries.into_iter().collect();
        let n = entries.len();

        if n == 0 {
            return Ok(Self {
                tags: Box::new([]),
                entries: Box::new([]),
                phf: BucketedPhf::build(&[], config.lambda)?.0,
                hash_builder,
                len: 0,
            });
        }

        let hashes: Vec<u64> = entries
            .iter()
            .map(|(k, _)| hash_builder.hash_one(k))
            .collect();

        let (phf, placements) = BucketedPhf::build(&hashes, config.lambda)?;
        let r = phf.num_buckets();
        let total_slots = r * SLOTS_PER_BUCKET;

        let mut tags: Vec<BucketTags> = vec![BucketTags::EMPTY; r];
        let mut slots: Vec<MaybeUninit<(K, V)>> =
            (0..total_slots).map(|_| MaybeUninit::uninit()).collect();

        for ((k, v), (i, h)) in entries.into_iter().zip(hashes.iter().enumerate()) {
            let bucket = placements.bucket[i] as usize;
            let slot = placements.slot[i] as usize;
            let tag = crate::hash_tag(*h);

            debug_assert_eq!(tags[bucket].0[slot], 0, "build placed two keys in one slot");
            tags[bucket].0[slot] = tag;
            slots[bucket * SLOTS_PER_BUCKET + slot].write((k, v));
        }

        Ok(Self {
            tags: tags.into_boxed_slice(),
            entries: slots.into_boxed_slice(),
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
        if self.tags.is_empty() {
            return None;
        }
        let hash = self.hash_builder.hash_one(key);
        let bucket = self.phf.bucket(hash);
        let tag = crate::hash_tag(hash);

        // SAFETY: `bucket < num_buckets` and `tags.len() == num_buckets`.
        let bucket_tags = unsafe { self.tags.get_unchecked(bucket) };
        let mask = unsafe { algorithm::match_tag_16(bucket_tags.0.as_ptr(), tag) };

        let base = bucket * SLOTS_PER_BUCKET;
        for slot in mask {
            // SAFETY: tag != 0 means this slot was initialized at build time.
            let entry = unsafe { self.entries.get_unchecked(base + slot).assume_init_ref() };
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

    /// PHF bucket count.
    #[inline]
    pub fn num_buckets(&self) -> usize {
        self.phf.num_buckets()
    }

    /// Iterate over `(&K, &V)` in slot order (an implementation-defined
    /// permutation of insertion order).
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> + '_ {
        self.tags
            .iter()
            .enumerate()
            .flat_map(move |(b, bucket_tags)| {
                let base = b * SLOTS_PER_BUCKET;
                bucket_tags.0.iter().enumerate().filter_map(move |(s, &tag)| {
                    if tag != 0 {
                        let entry = unsafe { self.entries.get_unchecked(base + s).assume_init_ref() };
                        Some((&entry.0, &entry.1))
                    } else {
                        None
                    }
                })
            })
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
    /// structure alone (not counting tags or entries). Diagnostic only.
    pub fn phf_bits_per_key(&self) -> f64 {
        self.phf.bits_per_key(self.len)
    }

    /// Total tag-table bits per key, including empty-slot bytes. At
    /// λ = 12, K = 16 this is `8 · 16/12 ≈ 10.7` bits/key.
    pub fn tag_bits_per_key(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        (self.tags.len() * SLOTS_PER_BUCKET * 8) as f64 / self.len as f64
    }

    /// Access the hash builder used at construction.
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }
}

impl<K, V, S> Drop for PerfectMapBucketed<K, V, S> {
    fn drop(&mut self) {
        if std::mem::needs_drop::<(K, V)>() {
            for (b, bucket_tags) in self.tags.iter().enumerate() {
                let base = b * SLOTS_PER_BUCKET;
                for (s, &tag) in bucket_tags.0.iter().enumerate() {
                    if tag != 0 {
                        // SAFETY: tag != 0 ⇔ this slot was initialized
                        // at build time and has not been moved out of.
                        unsafe {
                            self.entries[base + s].assume_init_drop();
                        }
                    }
                }
            }
        }
    }
}

impl<K, V, S> fmt::Debug for PerfectMapBucketed<K, V, S>
where
    K: Hash + Eq + fmt::Debug,
    V: fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

// ── PerfectSetBucketed ────────────────────────────────────────────────────

/// Read-only set with bucketed perfect hash + SIMD tag scan. Sibling to
/// [`PerfectSet`](crate::PerfectSet); same construction trade-off as
/// [`PerfectMapBucketed`].
pub struct PerfectSetBucketed<K, S = DefaultHashBuilder> {
    tags: Box<[BucketTags]>,
    keys: Box<[MaybeUninit<K>]>,
    phf: BucketedPhf,
    hash_builder: S,
    len: usize,
}

impl<K> PerfectSetBucketed<K, DefaultHashBuilder>
where
    K: Hash + Eq,
{
    /// Build with the default hash builder and default λ.
    pub fn from_iter_perfect<I>(keys: I) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = K>,
    {
        Self::from_keys(keys, DefaultHashBuilder::default(), &BucketedConfig::default())
    }
}

impl<K, S> PerfectSetBucketed<K, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Build with a custom hash builder and configuration.
    pub fn from_keys<I>(keys: I, hash_builder: S, config: &BucketedConfig) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = K>,
    {
        let keys: Vec<K> = keys.into_iter().collect();
        let n = keys.len();

        if n == 0 {
            return Ok(Self {
                tags: Box::new([]),
                keys: Box::new([]),
                phf: BucketedPhf::build(&[], config.lambda)?.0,
                hash_builder,
                len: 0,
            });
        }

        let hashes: Vec<u64> = keys.iter().map(|k| hash_builder.hash_one(k)).collect();
        let (phf, placements) = BucketedPhf::build(&hashes, config.lambda)?;
        let r = phf.num_buckets();
        let total_slots = r * SLOTS_PER_BUCKET;

        let mut tags: Vec<BucketTags> = vec![BucketTags::EMPTY; r];
        let mut slots: Vec<MaybeUninit<K>> =
            (0..total_slots).map(|_| MaybeUninit::uninit()).collect();

        for (k, (i, h)) in keys.into_iter().zip(hashes.iter().enumerate()) {
            let bucket = placements.bucket[i] as usize;
            let slot = placements.slot[i] as usize;
            let tag = crate::hash_tag(*h);

            debug_assert_eq!(tags[bucket].0[slot], 0);
            tags[bucket].0[slot] = tag;
            slots[bucket * SLOTS_PER_BUCKET + slot].write(k);
        }

        Ok(Self {
            tags: tags.into_boxed_slice(),
            keys: slots.into_boxed_slice(),
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

    /// Borrow the stored key equal to `key`, or `None`. Useful for
    /// interning — look up by `&Q`, get back a `&K` that lives as long as
    /// the set.
    #[inline]
    pub fn get<Q>(&self, key: &Q) -> Option<&K>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if self.tags.is_empty() {
            return None;
        }
        let hash = self.hash_builder.hash_one(key);
        let bucket = self.phf.bucket(hash);
        let tag = crate::hash_tag(hash);

        let bucket_tags = unsafe { self.tags.get_unchecked(bucket) };
        let mask = unsafe { algorithm::match_tag_16(bucket_tags.0.as_ptr(), tag) };

        let base = bucket * SLOTS_PER_BUCKET;
        for slot in mask {
            let stored = unsafe { self.keys.get_unchecked(base + slot).assume_init_ref() };
            if stored.borrow() == key {
                return Some(stored);
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

    /// Iterate over `&K` in slot order.
    pub fn iter(&self) -> impl Iterator<Item = &K> + '_ {
        self.tags
            .iter()
            .enumerate()
            .flat_map(move |(b, bucket_tags)| {
                let base = b * SLOTS_PER_BUCKET;
                bucket_tags.0.iter().enumerate().filter_map(move |(s, &tag)| {
                    if tag != 0 {
                        Some(unsafe { self.keys.get_unchecked(base + s).assume_init_ref() })
                    } else {
                        None
                    }
                })
            })
    }
}

impl<K, S> Drop for PerfectSetBucketed<K, S> {
    fn drop(&mut self) {
        if std::mem::needs_drop::<K>() {
            for (b, bucket_tags) in self.tags.iter().enumerate() {
                let base = b * SLOTS_PER_BUCKET;
                for (s, &tag) in bucket_tags.0.iter().enumerate() {
                    if tag != 0 {
                        unsafe {
                            self.keys[base + s].assume_init_drop();
                        }
                    }
                }
            }
        }
    }
}

impl<K, S> fmt::Debug for PerfectSetBucketed<K, S>
where
    K: Hash + Eq + fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn entries_u64(n: u64) -> Vec<(u64, u64)> {
        (0..n).map(|i| (i, i.wrapping_mul(31))).collect()
    }

    // ── PerfectMapBucketed ────────────────────────────────────────────────

    #[test]
    fn map_empty_round_trip() {
        let m = PerfectMapBucketed::<u64, u64>::from_iter_perfect(std::iter::empty()).unwrap();
        assert_eq!(m.len(), 0);
        assert!(m.is_empty());
        assert_eq!(m.get(&0), None);
    }

    #[test]
    fn map_single_entry_round_trip() {
        let m =
            PerfectMapBucketed::<u64, &str>::from_iter_perfect([(42u64, "answer")]).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m.get(&42), Some(&"answer"));
        assert_eq!(m.get(&0), None);
    }

    #[test]
    fn map_small_round_trip() {
        let n = 256u64;
        let entries = entries_u64(n);
        let m = PerfectMapBucketed::<u64, u64>::from_iter_perfect(entries.clone()).unwrap();
        assert_eq!(m.len(), n as usize);
        for (k, v) in &entries {
            assert_eq!(m.get(k), Some(v));
        }
        for k in n..n + 50 {
            assert_eq!(m.get(&k), None);
        }
    }

    #[test]
    fn map_medium_round_trip() {
        let n = 10_000u64;
        let entries = entries_u64(n);
        let m = PerfectMapBucketed::<u64, u64>::from_iter_perfect(entries.clone()).unwrap();
        assert_eq!(m.len(), n as usize);
        for (k, v) in &entries {
            assert_eq!(m.get(k), Some(v));
        }
    }

    #[test]
    fn map_large_round_trip() {
        let n = 100_000u64;
        let entries = entries_u64(n);
        let m = PerfectMapBucketed::<u64, u64>::from_iter_perfect(entries.clone()).unwrap();
        assert_eq!(m.len(), n as usize);
        for (k, v) in &entries {
            assert_eq!(m.get(k), Some(v));
        }
    }

    #[test]
    fn map_string_keys_round_trip() {
        let words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"];
        let entries: Vec<(String, usize)> =
            words.iter().enumerate().map(|(i, w)| (w.to_string(), i)).collect();
        let m = PerfectMapBucketed::<String, usize>::from_iter_perfect(entries.clone()).unwrap();
        for (k, v) in &entries {
            assert_eq!(m.get(k.as_str()), Some(v), "key={k}");
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
            let entries: Vec<(u64, Dropper)> = (0..100u64).map(|i| (i, Dropper(i as u32))).collect();
            let _m = PerfectMapBucketed::<u64, Dropper>::from_iter_perfect(entries).unwrap();
        }
        assert_eq!(DROPS.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn map_iter_returns_all_entries() {
        let entries = entries_u64(100);
        let m = PerfectMapBucketed::<u64, u64>::from_iter_perfect(entries.clone()).unwrap();
        let mut got: Vec<(u64, u64)> = m.iter().map(|(k, v)| (*k, *v)).collect();
        got.sort();
        let mut want = entries.clone();
        want.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn map_duplicate_keys_rejected() {
        let entries = vec![(1u64, 1u64), (2, 2), (1, 3)];
        let err = PerfectMapBucketed::<u64, u64>::from_iter_perfect(entries).unwrap_err();
        assert_eq!(err, BuildError::DuplicateHash);
    }

    #[test]
    fn map_custom_lambda() {
        let n = 1000u64;
        let entries = entries_u64(n);
        let config = BucketedConfig::default().with_lambda(8.0);
        let m = PerfectMapBucketed::<u64, u64>::from_entries(
            entries.clone(),
            DefaultHashBuilder::default(),
            &config,
        )
        .unwrap();
        for (k, v) in &entries {
            assert_eq!(m.get(k), Some(v));
        }
        // λ=8 → r ≈ n/8 buckets; total slots ~2n
        assert!(m.num_buckets() >= (n as usize).div_ceil(8));
    }

    // ── PerfectSetBucketed ────────────────────────────────────────────────

    #[test]
    fn set_empty_round_trip() {
        let s = PerfectSetBucketed::<u64>::from_iter_perfect(std::iter::empty()).unwrap();
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        assert!(!s.contains(&0));
    }

    #[test]
    fn set_small_round_trip() {
        let keys: Vec<u64> = (0..256).collect();
        let s = PerfectSetBucketed::<u64>::from_iter_perfect(keys.clone()).unwrap();
        for k in &keys {
            assert!(s.contains(k));
        }
        for k in 256..300 {
            assert!(!s.contains(&k));
        }
    }

    #[test]
    fn set_medium_round_trip() {
        let keys: Vec<u64> = (0..10_000).collect();
        let s = PerfectSetBucketed::<u64>::from_iter_perfect(keys.clone()).unwrap();
        for k in &keys {
            assert!(s.contains(k));
        }
        assert_eq!(s.len(), 10_000);
    }

    #[test]
    fn set_string_intern_pattern() {
        let words = ["alpha", "beta", "gamma", "delta", "epsilon"];
        let entries: Vec<String> = words.iter().map(|w| w.to_string()).collect();
        let s = PerfectSetBucketed::<String>::from_iter_perfect(entries).unwrap();
        let interned = s.get("beta").expect("beta is in the set");
        assert_eq!(interned, "beta");
        assert!(s.contains("epsilon"));
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
            let keys: Vec<K> = (0..50u32).map(K).collect();
            let _s = PerfectSetBucketed::<K>::from_iter_perfect(keys).unwrap();
        }
        assert_eq!(DROPS.load(Ordering::Relaxed), 50);
    }

    // ── Tag layout sanity ─────────────────────────────────────────────────

    #[test]
    fn tag_slab_alignment_is_16() {
        // SIMD path uses _mm_load_si128, which requires 16-byte alignment.
        // repr(align(16)) on BucketTags pins this; verify here so a
        // change to the struct can't silently break the lookup.
        use std::mem::align_of;
        assert_eq!(align_of::<BucketTags>(), 16);
    }
}
