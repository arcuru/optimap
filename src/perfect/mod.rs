//! Read-only perfect-hash maps.
//!
//! A [`PerfectMap`] is built from a fixed key set (typically an existing
//! [`OptiMap`](crate::OptiMap)) and answers lookups in one indirection plus
//! one stored-key compare — no probing, no collisions for in-set keys.
//! [`PerfectMapUnchecked`] drops the stored key, trading miss-safety for
//! lookups that touch only the value array.
//!
//! Both flavors are parameterized over the PHF algorithm (default
//! [`ChdPhf`]) and the hash builder (default `foldhash`), matching
//! [`OptiMap`](crate::OptiMap)'s style — new algorithms slot in by
//! implementing [`PerfectHashFunction`].
//!
//! # Example
//!
//! ```
//! use optimap::OptiMap;
//!
//! let mut m: OptiMap<u64, &'static str> = OptiMap::new();
//! for (k, v) in [(1, "one"), (2, "two"), (3, "three")] {
//!     m.insert(k, v);
//! }
//! let perfect = m.make_perfect();
//! assert_eq!(perfect.get(&2), Some(&"two"));
//! assert_eq!(perfect.get(&999), None);
//! ```

mod chd;
mod phf;

pub use chd::ChdPhf;
pub use phf::{BuildError, PerfectHashFunction};

use crate::map::DefaultHashBuilder;
use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

// ── Configuration ──────────────────────────────────────────────────────────

/// Build options for [`PerfectMap`] / [`PerfectMapUnchecked`].
#[derive(Debug, Clone)]
pub struct PerfectMapConfig {
    /// Ratio of PHF table size to key count. `1.0` requests a minimal
    /// perfect hash (densest packing, hardest construction); values
    /// greater than `1.0` give the construction more slack and finish
    /// faster, at the cost of `(load_factor - 1) · n` extra slots in
    /// the slot array.
    ///
    /// Default: `1.0`.
    pub load_factor: f64,
}

impl Default for PerfectMapConfig {
    fn default() -> Self {
        Self { load_factor: 1.0 }
    }
}

impl PerfectMapConfig {
    /// Convenience builder: set the load factor.
    pub fn with_load_factor(mut self, load_factor: f64) -> Self {
        self.load_factor = load_factor;
        self
    }

    fn table_size(&self, n: usize) -> usize {
        assert!(
            self.load_factor >= 1.0 && self.load_factor.is_finite(),
            "PerfectMapConfig::load_factor must be a finite value >= 1.0 (got {})",
            self.load_factor
        );
        if n == 0 {
            return 0;
        }
        ((n as f64) * self.load_factor).ceil() as usize
    }
}

// ── PerfectMap (with stored keys) ─────────────────────────────────────────

/// Read-only map with guaranteed one-probe lookup over its construction key set.
///
/// Built via [`OptiMap::make_perfect`](crate::OptiMap::make_perfect),
/// [`OptiMap::make_perfect_with`](crate::OptiMap::make_perfect_with), or
/// directly through [`PerfectMap::from_entries`].
///
/// Each slot stores `Some((K, V))` for an in-set key, or `None` for the
/// `(load_factor - 1) · n` slack slots when `load_factor > 1.0`. Lookup
/// hashes the query key once, computes its PHF slot, and matches the
/// stored key — returning `Some(&V)` only on a true match. Unknown keys
/// either hit a `None` slot or fail the key compare; either way `get`
/// returns `None`.
pub struct PerfectMap<K, V, P = ChdPhf, S = DefaultHashBuilder> {
    slots: Box<[Option<(K, V)>]>,
    phf: P,
    hash_builder: S,
    len: usize,
}

impl<K, V> PerfectMap<K, V, ChdPhf, DefaultHashBuilder>
where
    K: Hash + Eq,
{
    /// Build a [`PerfectMap`] from key-value pairs using the default
    /// algorithm (CHD), default hash builder, and default config (minimal
    /// perfect hash, `load_factor = 1.0`).
    ///
    /// Fails with [`BuildError::DuplicateHash`] if two keys produce the same
    /// `u64` hash, and with [`BuildError::Exhausted`] if the construction
    /// can't fit the key set into a minimal table (rare; increase the load
    /// factor via [`PerfectMap::from_entries_with`] to give construction
    /// more slack).
    pub fn from_iter_perfect<I>(entries: I) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        Self::from_entries(entries, DefaultHashBuilder::default())
    }
}

impl<K, V, S> PerfectMap<K, V, ChdPhf, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Build a [`PerfectMap`] under the default config with a custom hash
    /// builder.
    pub fn from_entries<I>(entries: I, hash_builder: S) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        Self::from_entries_with(entries, hash_builder, &PerfectMapConfig::default())
    }
}

impl<K, V, P, S> PerfectMap<K, V, P, S>
where
    K: Hash + Eq,
    P: PerfectHashFunction,
    S: BuildHasher,
{
    /// Build a [`PerfectMap`] with a custom hash builder and explicit
    /// [`PerfectMapConfig`].
    pub fn from_entries_with<I>(
        entries: I,
        hash_builder: S,
        config: &PerfectMapConfig,
    ) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let entries: Vec<(K, V)> = entries.into_iter().collect();
        let n = entries.len();
        let m = config.table_size(n);

        let hashes: Vec<u64> = entries
            .iter()
            .map(|(k, _)| hash_builder.hash_one(k))
            .collect();
        let phf = P::build(&hashes, m)?;

        let mut slots: Vec<Option<(K, V)>> = (0..m).map(|_| None).collect();
        for ((k, v), h) in entries.into_iter().zip(hashes) {
            let slot = phf.index(h);
            debug_assert!(slot < m, "PHF produced out-of-range slot");
            debug_assert!(
                slots[slot].is_none(),
                "PHF placed two keys in the same slot — algorithm violated its contract"
            );
            slots[slot] = Some((k, v));
        }

        Ok(Self {
            slots: slots.into_boxed_slice(),
            phf,
            hash_builder,
            len: n,
        })
    }
}

impl<K, V, P, S> PerfectMap<K, V, P, S>
where
    K: Hash + Eq,
    P: PerfectHashFunction,
    S: BuildHasher,
{
    /// Look up `key`. Returns `Some(&V)` if `key` was in the construction
    /// key set, `None` otherwise.
    #[inline]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if self.slots.is_empty() {
            return None;
        }
        let hash = self.hash_builder.hash_one(key);
        let slot = self.phf.index(hash);
        let entry = self.slots.get(slot)?;
        match entry {
            Some((k, v)) if k.borrow() == key => Some(v),
            _ => None,
        }
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

    /// Number of entries (the size of the original key set).
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True iff `len == 0`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// PHF table size — equals `len` when built minimal, larger otherwise.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Iterate over `(&K, &V)` pairs in slot order (an implementation-
    /// defined permutation of insertion order).
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> + '_ {
        self.slots
            .iter()
            .filter_map(|e| e.as_ref().map(|(k, v)| (k, v)))
    }

    /// Iterate over `&K`.
    pub fn keys(&self) -> impl Iterator<Item = &K> + '_ {
        self.iter().map(|(k, _)| k)
    }

    /// Iterate over `&V`.
    pub fn values(&self) -> impl Iterator<Item = &V> + '_ {
        self.iter().map(|(_, v)| v)
    }

    /// Approximate space overhead in bits per key, per the underlying PHF
    /// algorithm. Diagnostic only.
    pub fn phf_bits_per_key(&self) -> f64 {
        self.phf.bits_per_key()
    }

    /// Access the hash builder used at construction.
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }
}

impl<K, V, P, S> fmt::Debug for PerfectMap<K, V, P, S>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(
                self.slots
                    .iter()
                    .filter_map(|e| e.as_ref().map(|(k, v)| (k, v))),
            )
            .finish()
    }
}

// ── PerfectMapUnchecked (no stored keys) ──────────────────────────────────

/// Read-only map that skips the stored-key check on lookup — the smallest
/// possible per-slot footprint, at the cost of returning garbage for keys
/// outside the original set.
///
/// Built via [`OptiMap::make_perfect_unchecked`](crate::OptiMap::make_perfect_unchecked)
/// or [`PerfectMapUnchecked::from_entries`]. Always minimal (m = n) — there
/// are no empty slots to handle without stored keys to distinguish them.
///
/// `get_unchecked(&Q) -> &V` returns the value at the PHF slot
/// unconditionally. Use only where the caller's invariants guarantee the
/// query key was in the construction set. For miss-safety, build a
/// [`PerfectMap`] instead.
pub struct PerfectMapUnchecked<K, V, P = ChdPhf, S = DefaultHashBuilder> {
    values: Box<[V]>,
    phf: P,
    hash_builder: S,
    _marker: PhantomData<K>,
}

impl<K, V> PerfectMapUnchecked<K, V, ChdPhf, DefaultHashBuilder>
where
    K: Hash + Eq,
{
    /// Build a [`PerfectMapUnchecked`] from key-value pairs using the
    /// default algorithm (CHD) and default hash builder. Always minimal.
    pub fn from_iter_perfect<I>(entries: I) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        Self::from_entries(entries, DefaultHashBuilder::default())
    }
}

impl<K, V, P, S> PerfectMapUnchecked<K, V, P, S>
where
    K: Hash + Eq,
    P: PerfectHashFunction,
    S: BuildHasher,
{
    /// Build a [`PerfectMapUnchecked`] with a custom hash builder.
    pub fn from_entries<I>(entries: I, hash_builder: S) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        let entries: Vec<(K, V)> = entries.into_iter().collect();
        let n = entries.len();

        let hashes: Vec<u64> = entries
            .iter()
            .map(|(k, _)| hash_builder.hash_one(k))
            .collect();
        let phf = P::build(&hashes, n)?;

        // Place values into the slot array. Every slot must be assigned
        // exactly once because m = n and PHF is perfect.
        let mut placed: Vec<Option<V>> = (0..n).map(|_| None).collect();
        for ((_, v), h) in entries.into_iter().zip(hashes) {
            let slot = phf.index(h);
            debug_assert!(slot < n);
            debug_assert!(placed[slot].is_none());
            placed[slot] = Some(v);
        }
        let values: Vec<V> = placed
            .into_iter()
            .map(|o| o.expect("minimal PHF must fill every slot"))
            .collect();

        Ok(Self {
            values: values.into_boxed_slice(),
            phf,
            hash_builder,
            _marker: PhantomData,
        })
    }

    /// Look up `key`. Returns the value at the PHF slot. **If `key` was not
    /// in the construction set, the returned reference is to an arbitrary
    /// value from the set.** Use [`PerfectMap`] if miss-safety matters.
    #[inline]
    pub fn get_unchecked<Q>(&self, key: &Q) -> &V
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hash_builder.hash_one(key);
        let slot = self.phf.index(hash);
        // SAFETY: PHF maps any hash into [0, n). `values.len() == n`.
        unsafe { self.values.get_unchecked(slot) }
    }

    /// Number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True iff `len == 0`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Iterate over `&V` in slot order.
    pub fn values(&self) -> impl Iterator<Item = &V> + '_ {
        self.values.iter()
    }

    /// Approximate space overhead in bits per key, per the underlying PHF
    /// algorithm. Diagnostic only.
    pub fn phf_bits_per_key(&self) -> f64 {
        self.phf.bits_per_key()
    }

    /// Access the hash builder used at construction.
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }
}

impl<K, V, P, S> fmt::Debug for PerfectMapUnchecked<K, V, P, S>
where
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PerfectMapUnchecked")
            .field("len", &self.values.len())
            .field("values", &&*self.values)
            .finish()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn entries_u64(n: u64) -> Vec<(u64, u64)> {
        (0..n).map(|i| (i, i.wrapping_mul(31))).collect()
    }

    #[test]
    fn empty_map_round_trip() {
        let m = PerfectMap::<u64, u64>::from_iter_perfect(std::iter::empty()).unwrap();
        assert_eq!(m.len(), 0);
        assert!(m.is_empty());
        assert_eq!(m.get(&0), None);
    }

    #[test]
    fn single_entry_round_trip() {
        let m = PerfectMap::<u64, &str>::from_iter_perfect([(42u64, "answer")]).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m.get(&42), Some(&"answer"));
        assert_eq!(m.get(&0), None);
    }

    #[test]
    fn small_round_trip_stored() {
        let n = 256u64;
        let entries = entries_u64(n);
        let m = PerfectMap::<u64, u64>::from_iter_perfect(entries.clone()).unwrap();
        for (k, v) in &entries {
            assert_eq!(m.get(k), Some(v));
        }
        assert_eq!(m.len(), n as usize);
        // Misses
        for k in n..n + 50 {
            assert_eq!(m.get(&k), None);
        }
    }

    #[test]
    fn medium_round_trip_stored() {
        let n = 10_000u64;
        let entries = entries_u64(n);
        let m = PerfectMap::<u64, u64>::from_iter_perfect(entries.clone()).unwrap();
        assert_eq!(m.len(), n as usize);
        for (k, v) in &entries {
            assert_eq!(m.get(k), Some(v));
        }
    }

    #[test]
    fn large_round_trip_stored() {
        let n = 100_000u64;
        let entries = entries_u64(n);
        let m = PerfectMap::<u64, u64>::from_iter_perfect(entries.clone()).unwrap();
        assert_eq!(m.len(), n as usize);
        for (k, v) in &entries {
            assert_eq!(m.get(k), Some(v));
        }
    }

    #[test]
    fn non_minimal_load_factor() {
        let n = 1000u64;
        let entries = entries_u64(n);
        let config = PerfectMapConfig::default().with_load_factor(1.5);
        let m = PerfectMap::<u64, u64>::from_entries_with(
            entries.clone(),
            DefaultHashBuilder::default(),
            &config,
        )
        .unwrap();
        assert!(m.capacity() >= (n as f64 * 1.5) as usize);
        assert_eq!(m.len(), n as usize);
        for (k, v) in &entries {
            assert_eq!(m.get(k), Some(v));
        }
    }

    #[test]
    fn unchecked_small_round_trip() {
        let n = 256u64;
        let entries = entries_u64(n);
        let m = PerfectMapUnchecked::<u64, u64>::from_iter_perfect(entries.clone()).unwrap();
        assert_eq!(m.len(), n as usize);
        for (k, v) in &entries {
            assert_eq!(m.get_unchecked(k), v);
        }
    }

    #[test]
    fn unchecked_medium_round_trip() {
        let n = 10_000u64;
        let entries = entries_u64(n);
        let m = PerfectMapUnchecked::<u64, u64>::from_iter_perfect(entries.clone()).unwrap();
        assert_eq!(m.len(), n as usize);
        for (k, v) in &entries {
            assert_eq!(m.get_unchecked(k), v);
        }
    }

    #[test]
    fn string_keys_round_trip() {
        let words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"];
        let entries: Vec<(String, usize)> =
            words.iter().enumerate().map(|(i, w)| (w.to_string(), i)).collect();
        let m = PerfectMap::<String, usize>::from_iter_perfect(entries.clone()).unwrap();
        for (k, v) in &entries {
            assert_eq!(m.get(k.as_str()), Some(v), "key={k}");
        }
        assert_eq!(m.get("missing"), None);
    }

    #[test]
    fn iter_returns_all_entries() {
        let entries = entries_u64(100);
        let m = PerfectMap::<u64, u64>::from_iter_perfect(entries.clone()).unwrap();
        let mut got: Vec<(u64, u64)> = m.iter().map(|(k, v)| (*k, *v)).collect();
        got.sort();
        let mut want = entries.clone();
        want.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn config_load_factor_below_one_panics() {
        let config = PerfectMapConfig::default().with_load_factor(0.5);
        let entries = entries_u64(10);
        let result = std::panic::catch_unwind(|| {
            let _ = PerfectMap::<u64, u64>::from_entries_with(
                entries,
                DefaultHashBuilder::default(),
                &config,
            );
        });
        assert!(result.is_err(), "load_factor < 1.0 should panic");
    }
}
