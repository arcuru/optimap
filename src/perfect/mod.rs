//! Read-only perfect-hash maps.
//!
//! Three flavors, all built once from a fixed key set and never mutated:
//!
//! | Type                    | Layout                          | Storage cost                    | Lookup contract              |
//! | ----------------------- | ------------------------------- | ------------------------------- | ---------------------------- |
//! | [`PerfectMap`]          | dense, `Box<[(K, V)]>`, m = n   | `n · (sizeof(K) + sizeof(V))`   | `get` returns `Option<&V>`   |
//! | [`PerfectMapSparse`]    | sparse, `Box<[Option<(K,V)>]>`  | `m · sizeof(Option<(K,V)>)`     | `get` returns `Option<&V>`   |
//! | [`PerfectMapUnchecked`] | dense, `Box<[V]>`, m = n        | `n · sizeof(V)`                 | `get_unchecked` returns `&V` |
//!
//! [`PerfectMap`] is the default — minimal perfect hash, dense storage,
//! key compare for miss safety. Use it unless you specifically need
//! something else.
//!
//! [`PerfectMapSparse`] exists for callers who want to trade space for
//! build time via a load factor > 1.0 — the construction has more slack
//! and finishes faster, at the cost of `(load_factor − 1) · n` empty slots.
//!
//! [`PerfectMapUnchecked`] drops the stored key entirely. Lookup returns a
//! `&V` directly with no membership check; out-of-set queries return an
//! arbitrary in-set value. Use only when membership is invariant.
//!
//! All three are parameterized over the PHF algorithm (default [`ChdPhf`])
//! and the hash builder (default `foldhash`). New algorithms slot in by
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

pub mod bucketed;
mod chd;
mod phf;
mod set;

pub use bucketed::{
    BucketedConfig, BucketedPhf, PerfectMapBucketed, PerfectSetBucketed,
};
pub use chd::ChdPhf;
pub use phf::{BuildError, PerfectHashFunction};
pub use set::PerfectSet;

use crate::map::DefaultHashBuilder;
use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

// ── PerfectMap (dense minimal, stored keys) ───────────────────────────────

/// Read-only map with guaranteed one-probe lookup. Dense storage — every
/// slot holds exactly one `(K, V)` pair, no `Option` wrapping. Minimal
/// perfect hash (table size equals key count).
///
/// The default flavor. Built via
/// [`OptiMap::make_perfect`](crate::OptiMap::make_perfect) or
/// [`PerfectMap::from_entries`]. For non-minimal load factors, see
/// [`PerfectMapSparse`]. For zero key storage, see [`PerfectMapUnchecked`].
pub struct PerfectMap<K, V, P = ChdPhf, S = DefaultHashBuilder> {
    slots: Box<[(K, V)]>,
    phf: P,
    hash_builder: S,
}

impl<K, V> PerfectMap<K, V, ChdPhf, DefaultHashBuilder>
where
    K: Hash + Eq,
{
    /// Build with the default algorithm (CHD) and default hash builder.
    ///
    /// Fails with [`BuildError::DuplicateHash`] on a u64 collision between
    /// any two keys, or [`BuildError::Exhausted`] if CHD can't fit the key
    /// set into a minimal table (rare; if it happens, use
    /// [`PerfectMapSparse`] with a load factor > 1.0).
    pub fn from_iter_perfect<I>(entries: I) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = (K, V)>,
    {
        Self::from_entries(entries, DefaultHashBuilder::default())
    }
}

impl<K, V, P, S> PerfectMap<K, V, P, S>
where
    K: Hash + Eq,
    P: PerfectHashFunction,
    S: BuildHasher,
{
    /// Build with a custom hash builder. Algorithm is `P` (default `ChdPhf`
    /// at the OptiMap entry point; pin a concrete `P` via the type
    /// parameter when constructing directly).
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

        // Place each entry at its PHF slot. Every slot ends up exactly once
        // because m = n and the PHF is perfect over the input hashes —
        // verified by the post-loop count check.
        let mut placed: Vec<Option<(K, V)>> = (0..n).map(|_| None).collect();
        for ((k, v), h) in entries.into_iter().zip(hashes) {
            let slot = phf.index(h);
            debug_assert!(slot < n, "PHF produced out-of-range slot");
            debug_assert!(
                placed[slot].is_none(),
                "PHF placed two keys in the same slot — algorithm violated its contract"
            );
            placed[slot] = Some((k, v));
        }
        let slots: Vec<(K, V)> = placed
            .into_iter()
            .map(|o| o.expect("minimal PHF must fill every slot"))
            .collect();

        Ok(Self {
            slots: slots.into_boxed_slice(),
            phf,
            hash_builder,
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
        if self.slots.is_empty() {
            return None;
        }
        let hash = self.hash_builder.hash_one(key);
        let slot = self.phf.index(hash);
        // SAFETY: `phf.index` returns a value in `[0, m)` and `slots.len() == m`.
        let (k, v) = unsafe { self.slots.get_unchecked(slot) };
        if k.borrow() == key { Some(v) } else { None }
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
        self.slots.len()
    }

    /// True iff `len == 0`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Iterate over `(&K, &V)` pairs in slot order (an implementation-
    /// defined permutation of insertion order).
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> + '_ {
        self.slots.iter().map(|(k, v)| (k, v))
    }

    /// Iterate over `&K`.
    pub fn keys(&self) -> impl Iterator<Item = &K> + '_ {
        self.slots.iter().map(|(k, _)| k)
    }

    /// Iterate over `&V`.
    pub fn values(&self) -> impl Iterator<Item = &V> + '_ {
        self.slots.iter().map(|(_, v)| v)
    }

    /// Approximate space overhead in bits per key for the PHF data
    /// structure (not counting the slot array). Diagnostic only.
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
            .entries(self.slots.iter().map(|(k, v)| (k, v)))
            .finish()
    }
}

// ── PerfectMapSparse (configurable load factor, stored keys) ──────────────

/// Build configuration for [`PerfectMapSparse`].
#[derive(Debug, Clone)]
pub struct PerfectMapConfig {
    /// Ratio of PHF table size to key count. `1.0` requests a minimal
    /// perfect hash — but if you want minimal you should use [`PerfectMap`]
    /// instead; this type only makes sense with values strictly greater
    /// than `1.0`. Values larger than `1.0` give the construction more
    /// slack (faster build) at the cost of `(load_factor − 1) · n` empty
    /// slots in the slot array.
    ///
    /// Default: `1.0` (degenerate — prefer [`PerfectMap`] at this point).
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

/// Read-only map with configurable PHF load factor. Sparse storage —
/// `m = ceil(n · load_factor)` slots, wrapped in `Option` so the `m − n`
/// slack slots can be empty.
///
/// Trades space for build time: high load factors let the PHF construction
/// finish in fewer displacement attempts. For the common case of minimal
/// (load factor = 1.0) dense storage, prefer [`PerfectMap`].
pub struct PerfectMapSparse<K, V, P = ChdPhf, S = DefaultHashBuilder> {
    slots: Box<[Option<(K, V)>]>,
    phf: P,
    hash_builder: S,
    len: usize,
}

impl<K, V, P, S> PerfectMapSparse<K, V, P, S>
where
    K: Hash + Eq,
    P: PerfectHashFunction,
    S: BuildHasher,
{
    /// Build with a custom hash builder and explicit [`PerfectMapConfig`].
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
                "PHF placed two keys in the same slot"
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

    /// Look up `key`. Returns `Some(&V)` if `key` was in the construction
    /// key set, `None` otherwise (either the slot is empty slack or the
    /// stored key doesn't match).
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

    /// PHF table size including slack slots.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Iterate over `(&K, &V)` pairs in slot order, skipping slack slots.
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

    /// Approximate space overhead in bits per key for the PHF data
    /// structure (not counting the slot array). Diagnostic only.
    pub fn phf_bits_per_key(&self) -> f64 {
        self.phf.bits_per_key()
    }

    /// Access the hash builder used at construction.
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }
}

impl<K, V, P, S> fmt::Debug for PerfectMapSparse<K, V, P, S>
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

// ── PerfectMapUnchecked (dense minimal, no stored keys) ───────────────────

/// Read-only map with no stored keys. Lookup returns `&V` directly without
/// a key compare — the cheapest possible per-slot footprint at the cost of
/// miss safety. Always minimal (m = n).
///
/// `get_unchecked(&Q) -> &V` returns the value at the PHF slot
/// unconditionally. If `Q` was not in the construction set the returned
/// reference is to an arbitrary in-set value — no UB (the slot array is
/// fully initialized), just garbage. Use only where membership is an
/// invariant. For miss-safety, use [`PerfectMap`].
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
    /// Build with the default algorithm (CHD) and default hash builder.
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
    /// Build with a custom hash builder.
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
    /// in-set value.** Use [`PerfectMap`] if miss-safety matters.
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

    // ── PerfectMap (dense minimal) ────────────────────────────────────────

    #[test]
    fn dense_empty_round_trip() {
        let m = PerfectMap::<u64, u64>::from_iter_perfect(std::iter::empty()).unwrap();
        assert_eq!(m.len(), 0);
        assert!(m.is_empty());
        assert_eq!(m.get(&0), None);
    }

    #[test]
    fn dense_single_entry_round_trip() {
        let m = PerfectMap::<u64, &str>::from_iter_perfect([(42u64, "answer")]).unwrap();
        assert_eq!(m.len(), 1);
        assert_eq!(m.get(&42), Some(&"answer"));
        assert_eq!(m.get(&0), None);
    }

    #[test]
    fn dense_small_round_trip() {
        let n = 256u64;
        let entries = entries_u64(n);
        let m = PerfectMap::<u64, u64>::from_iter_perfect(entries.clone()).unwrap();
        for (k, v) in &entries {
            assert_eq!(m.get(k), Some(v));
        }
        assert_eq!(m.len(), n as usize);
        for k in n..n + 50 {
            assert_eq!(m.get(&k), None);
        }
    }

    #[test]
    fn dense_medium_round_trip() {
        let n = 10_000u64;
        let entries = entries_u64(n);
        let m = PerfectMap::<u64, u64>::from_iter_perfect(entries.clone()).unwrap();
        assert_eq!(m.len(), n as usize);
        for (k, v) in &entries {
            assert_eq!(m.get(k), Some(v));
        }
    }

    #[test]
    fn dense_large_round_trip() {
        let n = 100_000u64;
        let entries = entries_u64(n);
        let m = PerfectMap::<u64, u64>::from_iter_perfect(entries.clone()).unwrap();
        assert_eq!(m.len(), n as usize);
        for (k, v) in &entries {
            assert_eq!(m.get(k), Some(v));
        }
    }

    #[test]
    fn dense_string_keys_round_trip() {
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
    fn dense_iter_returns_all_entries() {
        let entries = entries_u64(100);
        let m = PerfectMap::<u64, u64>::from_iter_perfect(entries.clone()).unwrap();
        let mut got: Vec<(u64, u64)> = m.iter().map(|(k, v)| (*k, *v)).collect();
        got.sort();
        let mut want = entries.clone();
        want.sort();
        assert_eq!(got, want);
    }

    // ── PerfectMapSparse (configurable load factor) ───────────────────────

    #[test]
    fn sparse_non_minimal_round_trip() {
        let n = 1000u64;
        let entries = entries_u64(n);
        let config = PerfectMapConfig::default().with_load_factor(1.5);
        let m = PerfectMapSparse::<u64, u64>::from_entries_with(
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
        for k in n..n + 50 {
            assert_eq!(m.get(&k), None);
        }
    }

    #[test]
    fn sparse_minimal_still_works() {
        let n = 1000u64;
        let entries = entries_u64(n);
        let config = PerfectMapConfig::default();
        let m = PerfectMapSparse::<u64, u64>::from_entries_with(
            entries.clone(),
            DefaultHashBuilder::default(),
            &config,
        )
        .unwrap();
        assert_eq!(m.capacity(), n as usize);
        assert_eq!(m.len(), n as usize);
        for (k, v) in &entries {
            assert_eq!(m.get(k), Some(v));
        }
    }

    #[test]
    fn sparse_load_factor_below_one_panics() {
        let config = PerfectMapConfig::default().with_load_factor(0.5);
        let entries = entries_u64(10);
        let result = std::panic::catch_unwind(|| {
            let _ = PerfectMapSparse::<u64, u64>::from_entries_with(
                entries,
                DefaultHashBuilder::default(),
                &config,
            );
        });
        assert!(result.is_err(), "load_factor < 1.0 should panic");
    }

    // ── PerfectMapUnchecked ───────────────────────────────────────────────

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

    // ── Storage-size sanity ───────────────────────────────────────────────

    #[test]
    fn dense_layout_skips_option_niche() {
        // (u64, u64) is 16 bytes — Option<(u64, u64)> is 24 bytes because
        // there's no niche to encode the discriminant. Dense layout uses
        // the former; Sparse uses the latter. This pins the contract.
        use std::mem::size_of;
        assert_eq!(size_of::<(u64, u64)>(), 16);
        assert_eq!(size_of::<Option<(u64, u64)>>(), 24);
    }
}
