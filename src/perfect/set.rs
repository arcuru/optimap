//! Read-only perfect-hash set — exact membership without false positives.
//!
//! Where `PerfectMap<K, V>` answers "what value does this key map to?",
//! `PerfectSet<K>` answers "is this key in the set?" with zero false
//! positives. Probabilistic alternatives (Bloom, Xor, Binary Fuse) trade
//! exact-ness for smaller space; use those when an occasional false
//! positive is acceptable and key storage cost is prohibitive.
//!
//! There is no `PerfectSetUnchecked` — a set without stored keys can't
//! distinguish in-set from out-of-set queries (it would have to always
//! answer "true"), which makes it indistinguishable from a no-op.

use super::chd::ChdPhf;
use super::phf::{BuildError, PerfectHashFunction};
use crate::map::DefaultHashBuilder;
use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash};

/// Read-only set with one-probe membership lookup. Dense storage —
/// `Box<[K]>` of size n, one key compare per `contains`.
pub struct PerfectSet<K, P = ChdPhf, S = DefaultHashBuilder> {
    slots: Box<[K]>,
    phf: P,
    hash_builder: S,
}

impl<K> PerfectSet<K, ChdPhf, DefaultHashBuilder>
where
    K: Hash + Eq,
{
    /// Build with the default algorithm (CHD) and default hash builder.
    pub fn from_iter_perfect<I>(keys: I) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = K>,
    {
        Self::from_keys(keys, DefaultHashBuilder::default())
    }
}

impl<K, P, S> PerfectSet<K, P, S>
where
    K: Hash + Eq,
    P: PerfectHashFunction,
    S: BuildHasher,
{
    /// Build with a custom hash builder.
    pub fn from_keys<I>(keys: I, hash_builder: S) -> Result<Self, BuildError>
    where
        I: IntoIterator<Item = K>,
    {
        let keys: Vec<K> = keys.into_iter().collect();
        let n = keys.len();

        let hashes: Vec<u64> = keys.iter().map(|k| hash_builder.hash_one(k)).collect();
        let phf = P::build(&hashes, n)?;

        // Place each key at its PHF slot.
        let mut placed: Vec<Option<K>> = (0..n).map(|_| None).collect();
        for (k, h) in keys.into_iter().zip(hashes) {
            let slot = phf.index(h);
            debug_assert!(slot < n);
            debug_assert!(placed[slot].is_none());
            placed[slot] = Some(k);
        }
        let slots: Vec<K> = placed
            .into_iter()
            .map(|o| o.expect("minimal PHF must fill every slot"))
            .collect();

        Ok(Self {
            slots: slots.into_boxed_slice(),
            phf,
            hash_builder,
        })
    }

    /// True iff `key` was in the construction key set. Exact — no false
    /// positives, no false negatives.
    #[inline]
    pub fn contains<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if self.slots.is_empty() {
            return false;
        }
        let hash = self.hash_builder.hash_one(key);
        let slot = self.phf.index(hash);
        // SAFETY: `phf.index` returns a value in `[0, m)` and `slots.len() == m`.
        let stored = unsafe { self.slots.get_unchecked(slot) };
        stored.borrow() == key
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
        if self.slots.is_empty() {
            return None;
        }
        let hash = self.hash_builder.hash_one(key);
        let slot = self.phf.index(hash);
        // SAFETY: same as contains.
        let stored = unsafe { self.slots.get_unchecked(slot) };
        if stored.borrow() == key { Some(stored) } else { None }
    }

    /// Number of keys.
    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// True iff empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Iterate over `&K` in slot order (an implementation-defined
    /// permutation of insertion order).
    pub fn iter(&self) -> impl Iterator<Item = &K> + '_ {
        self.slots.iter()
    }

    /// Approximate space overhead in bits per key for the PHF data
    /// structure (not counting the key array). Diagnostic only.
    pub fn phf_bits_per_key(&self) -> f64 {
        self.phf.bits_per_key()
    }

    /// Access the hash builder used at construction.
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }
}

impl<K, P, S> fmt::Debug for PerfectSet<K, P, S>
where
    K: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.slots.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_set() {
        let s = PerfectSet::<u64>::from_iter_perfect(std::iter::empty()).unwrap();
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        assert!(!s.contains(&0));
    }

    #[test]
    fn single_key_round_trip() {
        let s = PerfectSet::<u64>::from_iter_perfect([42u64]).unwrap();
        assert_eq!(s.len(), 1);
        assert!(s.contains(&42));
        assert!(!s.contains(&0));
        assert_eq!(s.get(&42), Some(&42));
        assert_eq!(s.get(&0), None);
    }

    #[test]
    fn small_round_trip() {
        let keys: Vec<u64> = (0..256).collect();
        let s = PerfectSet::<u64>::from_iter_perfect(keys.clone()).unwrap();
        for k in &keys {
            assert!(s.contains(k));
        }
        for k in 256..300 {
            assert!(!s.contains(&k));
        }
    }

    #[test]
    fn medium_round_trip() {
        let keys: Vec<u64> = (0..10_000).collect();
        let s = PerfectSet::<u64>::from_iter_perfect(keys.clone()).unwrap();
        for k in &keys {
            assert!(s.contains(k));
        }
        assert_eq!(s.len(), 10_000);
    }

    #[test]
    fn large_round_trip() {
        let keys: Vec<u64> = (0..100_000).collect();
        let s = PerfectSet::<u64>::from_iter_perfect(keys.clone()).unwrap();
        for k in &keys {
            assert!(s.contains(k));
        }
    }

    #[test]
    fn string_keys_intern_pattern() {
        let words = ["alpha", "beta", "gamma", "delta", "epsilon"];
        let entries: Vec<String> = words.iter().map(|w| w.to_string()).collect();
        let s = PerfectSet::<String>::from_iter_perfect(entries).unwrap();

        // The intern pattern: query with a borrowed &str, get back the owned String.
        let interned = s.get("beta").expect("beta is in the set");
        assert_eq!(interned, "beta");
        assert!(s.contains("epsilon"));
        assert!(!s.contains("missing"));
    }

    #[test]
    fn iter_returns_all_keys() {
        let keys: Vec<u64> = (0..100).collect();
        let s = PerfectSet::<u64>::from_iter_perfect(keys.clone()).unwrap();
        let mut got: Vec<u64> = s.iter().copied().collect();
        got.sort();
        assert_eq!(got, keys);
    }

    #[test]
    fn duplicate_keys_rejected() {
        // Two equal keys → two equal hashes → PHF rejects with DuplicateHash.
        let result = PerfectSet::<u64>::from_iter_perfect([1u64, 2, 3, 1]);
        assert_eq!(result.err(), Some(BuildError::DuplicateHash));
    }
}
