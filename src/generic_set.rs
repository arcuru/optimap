//! Generic set wrapper over any `Map<T, ()>` implementation.
//!
//! Provides `Set<T, M>` where `M` is any OptiMap map type (or hashbrown).
//! The set operations delegate to the underlying map with `()` values.

use std::borrow::Borrow;
use std::fmt;
use std::hash::Hash;

use crate::Map;

/// A hash set backed by any [`Map`] implementation.
///
/// `Set<T, M>` wraps `M` where `M: Map<T, ()>`. The map's key is the
/// set element; the value is zero-sized `()`.
///
/// # Type aliases
///
/// Each OptiMap design has a convenience alias:
/// - `UfmSet<T>` — backed by `UnorderedFlatMap`
/// - `SplitsiesSet<T>` — backed by `Splitsies`
/// - `IpoSet<T>` — backed by `InPlaceOverflow`
/// - `GapsSet<T>` — backed by `Gaps`
/// - `Ipo64Set<T>` — backed by `IPO64`
pub struct Set<T: Hash + Eq, M: Map<T, ()> = crate::UnorderedFlatMap<T, ()>> {
    map: M,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Hash + Eq, M: Map<T, ()>> Set<T, M> {
    /// Create an empty set.
    pub fn new() -> Self {
        Set { map: M::new(), _marker: std::marker::PhantomData }
    }

    /// Create a set with at least the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Set { map: M::with_capacity(capacity), _marker: std::marker::PhantomData }
    }

    /// Adds a value to the set. Returns `true` if newly inserted,
    /// `false` if already present.
    pub fn insert(&mut self, value: T) -> bool {
        self.map.insert(value, ()).is_none()
    }

    /// Returns `true` if the set contains the given value.
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.get(value).is_some()
    }

    /// Removes a value from the set. Returns `true` if it was present.
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.remove(value).is_some()
    }

    /// Number of elements in the set.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Number of elements the set can hold without rehashing.
    pub fn capacity(&self) -> usize {
        self.map.capacity()
    }

    /// Remove all elements, keeping allocated memory.
    pub fn clear(&mut self) {
        self.map.clear();
    }
}

// ── Set algebra operations ──────────────────────────────────────────────────

impl<T: Hash + Eq, M: Map<T, ()>> Set<T, M> {
    /// Returns `true` if `self` has no elements in common with `other`.
    pub fn is_disjoint<M2: Map<T, ()>>(&self, other: &Set<T, M2>) -> bool
    where
        T: Borrow<T>,
    {
        if self.len() <= other.len() {
            // Can't iterate generically without iterator support on Map trait.
            // For now, this requires the concrete type's iter().
            // TODO: Add iteration to the Map trait or use a separate approach.
            false // placeholder
        } else {
            false // placeholder
        }
    }
}

// ── Trait implementations ───────────────────────────────────────────────────

impl<T: Hash + Eq, M: Map<T, ()>> Default for Set<T, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Hash + Eq, M: Map<T, ()>> FromIterator<T> for Set<T, M> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut set = Self::with_capacity(lower);
        for item in iter {
            set.insert(item);
        }
        set
    }
}

impl<T: Hash + Eq, M: Map<T, ()>> Extend<T> for Set<T, M> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.insert(item);
        }
    }
}

impl<T: Hash + Eq + fmt::Debug, M: Map<T, ()>> fmt::Debug for Set<T, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Can't iterate generically — print len
        write!(f, "Set(len={})", self.len())
    }
}

// ── Type aliases for each design ────────────────────────────────────────────

/// Set backed by `UnorderedFlatMap` (15-slot Boost-style groups).
pub type UfmSet<T> = Set<T, crate::UnorderedFlatMap<T, ()>>;

/// Set backed by `Splitsies` (16-slot groups with separate overflow).
pub type SplitsiesSet<T> = Set<T, crate::Splitsies<T, ()>>;

/// Set backed by `InPlaceOverflow` (tombstone-based, 254 hash values).
pub type IpoSet<T> = Set<T, crate::InPlaceOverflow<T, ()>>;

/// Set backed by `Gaps` (15-slot groups with power-of-2 bucket stride).
pub type GapsSet<T> = Set<T, crate::Gaps<T, ()>>;

/// Set backed by `IPO64` (64-slot cache-line groups with AVX-512).
pub type Ipo64Set<T> = Set<T, crate::IPO64<T, ()>>;

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_ufm_set() {
        let mut set = UfmSet::new();
        assert!(set.is_empty());
        assert!(set.insert(1));
        assert!(set.insert(2));
        assert!(!set.insert(1)); // duplicate
        assert_eq!(set.len(), 2);
        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(!set.contains(&3));
    }

    #[test]
    fn basic_splitsies_set() {
        let mut set = SplitsiesSet::new();
        assert!(set.insert("hello".to_string()));
        assert!(set.insert("world".to_string()));
        assert!(!set.insert("hello".to_string()));
        assert_eq!(set.len(), 2);
        assert!(set.contains("hello"));
        assert!(set.contains("world"));
    }

    #[test]
    fn basic_ipo_set() {
        let mut set = IpoSet::new();
        for i in 0..1000 {
            set.insert(i);
        }
        assert_eq!(set.len(), 1000);
        for i in 0..1000 {
            assert!(set.contains(&i));
        }
    }

    #[test]
    fn remove() {
        let mut set = UfmSet::new();
        set.insert(1);
        set.insert(2);
        assert!(set.remove(&1));
        assert!(!set.remove(&1));
        assert_eq!(set.len(), 1);
        assert!(!set.contains(&1));
        assert!(set.contains(&2));
    }

    #[test]
    fn from_iter() {
        let set: SplitsiesSet<i32> = vec![1, 2, 3, 2, 1].into_iter().collect();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn clear() {
        let mut set = IpoSet::new();
        for i in 0..100 { set.insert(i); }
        set.clear();
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn with_capacity() {
        let set = UfmSet::<i32>::with_capacity(100);
        assert!(set.capacity() >= 100);
        assert!(set.is_empty());
    }

    #[test]
    fn gaps_set() {
        let mut set = GapsSet::new();
        for i in 0..500 { set.insert(i); }
        assert_eq!(set.len(), 500);
        for i in 0..250 { set.remove(&i); }
        assert_eq!(set.len(), 250);
    }

    #[test]
    fn ipo64_set() {
        let mut set = Ipo64Set::new();
        for i in 0..500 { set.insert(i); }
        assert_eq!(set.len(), 500);
        assert!(set.contains(&499));
        assert!(!set.contains(&500));
    }
}
