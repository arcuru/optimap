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

    /// Iterate over elements in arbitrary order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.map.iter().map(|(k, _)| k)
    }
}

// ── Set algebra operations ──────────────────────────────────────────────────

impl<T: Hash + Eq + Clone, M: Map<T, ()>> Set<T, M> {
    /// Returns `true` if `self` has no elements in common with `other`.
    pub fn is_disjoint<M2: Map<T, ()>>(&self, other: &Set<T, M2>) -> bool {
        if self.len() <= other.len() {
            self.iter().all(|v| !other.contains(v))
        } else {
            other.iter().all(|v| !self.contains(v))
        }
    }

    /// Returns `true` if every element in `self` is also in `other`.
    pub fn is_subset<M2: Map<T, ()>>(&self, other: &Set<T, M2>) -> bool {
        if self.len() > other.len() { return false; }
        self.iter().all(|v| other.contains(v))
    }

    /// Returns `true` if every element in `other` is also in `self`.
    pub fn is_superset<M2: Map<T, ()>>(&self, other: &Set<T, M2>) -> bool {
        other.is_subset(self)
    }

    /// Returns the union of `self` and `other` as a new set.
    pub fn union(&self, other: &Self) -> Self {
        let mut result = Self::with_capacity(self.len() + other.len());
        for item in self.iter() { result.insert(item.clone()); }
        for item in other.iter() { result.insert(item.clone()); }
        result
    }

    /// Returns the intersection of `self` and `other` as a new set.
    pub fn intersection<M2: Map<T, ()>>(&self, other: &Set<T, M2>) -> Self {
        let mut result = Self::new();
        let (smaller, check) = if self.len() <= other.len() {
            (self as &Self, other as &Set<T, M2>)
        } else {
            // Can't easily swap types, iterate self and check other
            return {
                let mut r = Self::new();
                for item in self.iter() {
                    if other.contains(item) { r.insert(item.clone()); }
                }
                r
            };
        };
        for item in smaller.iter() {
            if check.contains(item) { result.insert(item.clone()); }
        }
        result
    }

    /// Returns elements in `self` but not in `other`.
    pub fn difference<M2: Map<T, ()>>(&self, other: &Set<T, M2>) -> Self {
        let mut result = Self::new();
        for item in self.iter() {
            if !other.contains(item) { result.insert(item.clone()); }
        }
        result
    }

    /// Returns elements in either set but not both.
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        let mut result = Self::new();
        for item in self.iter() {
            if !other.contains(item) { result.insert(item.clone()); }
        }
        for item in other.iter() {
            if !self.contains(item) { result.insert(item.clone()); }
        }
        result
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
        f.debug_set().entries(self.iter()).finish()
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

    // ── Iterator tests ──────────────────────────────────────────────────

    #[test]
    fn iter_ufm() {
        let set: UfmSet<i32> = vec![1, 2, 3].into_iter().collect();
        let mut items: Vec<i32> = set.iter().copied().collect();
        items.sort();
        assert_eq!(items, vec![1, 2, 3]);
    }

    #[test]
    fn iter_splitsies() {
        let set: SplitsiesSet<i32> = (0..100).collect();
        assert_eq!(set.iter().count(), 100);
        let mut items: Vec<i32> = set.iter().copied().collect();
        items.sort();
        assert_eq!(items, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn iter_ipo() {
        let set: IpoSet<i32> = (0..50).collect();
        let sum: i32 = set.iter().sum();
        assert_eq!(sum, (0..50).sum());
    }

    // ── Set algebra tests ───────────────────────────────────────────────

    #[test]
    fn union() {
        let a: SplitsiesSet<i32> = vec![1, 2, 3].into_iter().collect();
        let b: SplitsiesSet<i32> = vec![3, 4, 5].into_iter().collect();
        let u = a.union(&b);
        assert_eq!(u.len(), 5);
        for i in 1..=5 { assert!(u.contains(&i)); }
    }

    #[test]
    fn intersection() {
        let a: IpoSet<i32> = vec![1, 2, 3, 4].into_iter().collect();
        let b: IpoSet<i32> = vec![3, 4, 5, 6].into_iter().collect();
        let inter = a.intersection(&b);
        assert_eq!(inter.len(), 2);
        assert!(inter.contains(&3));
        assert!(inter.contains(&4));
    }

    #[test]
    fn difference() {
        let a: UfmSet<i32> = vec![1, 2, 3, 4].into_iter().collect();
        let b: UfmSet<i32> = vec![3, 4, 5, 6].into_iter().collect();
        let diff = a.difference(&b);
        assert_eq!(diff.len(), 2);
        assert!(diff.contains(&1));
        assert!(diff.contains(&2));
    }

    #[test]
    fn symmetric_difference() {
        let a: SplitsiesSet<i32> = vec![1, 2, 3].into_iter().collect();
        let b: SplitsiesSet<i32> = vec![2, 3, 4].into_iter().collect();
        let sd = a.symmetric_difference(&b);
        assert_eq!(sd.len(), 2);
        assert!(sd.contains(&1));
        assert!(sd.contains(&4));
    }

    #[test]
    fn subset_superset() {
        let a: GapsSet<i32> = vec![1, 2].into_iter().collect();
        let b: GapsSet<i32> = vec![1, 2, 3].into_iter().collect();
        assert!(a.is_subset(&b));
        assert!(!b.is_subset(&a));
        assert!(b.is_superset(&a));
    }

    #[test]
    fn disjoint() {
        let a: IpoSet<i32> = vec![1, 2].into_iter().collect();
        let b: IpoSet<i32> = vec![3, 4].into_iter().collect();
        let c: IpoSet<i32> = vec![2, 3].into_iter().collect();
        assert!(a.is_disjoint(&b));
        assert!(!a.is_disjoint(&c));
    }

    // ── Map trait iter test ─────────────────────────────────────────────

    #[test]
    fn map_trait_iter_generic() {
        use crate::Map;
        fn sum_values<M: Map<i32, i32>>(m: &M) -> i32 {
            m.iter().map(|(_, v)| v).sum()
        }

        let mut m = crate::Splitsies::new();
        m.insert(1, 10);
        m.insert(2, 20);
        m.insert(3, 30);
        assert_eq!(sum_values(&m), 60);

        let mut m2 = crate::InPlaceOverflow::new();
        m2.insert(1, 100);
        m2.insert(2, 200);
        assert_eq!(sum_values(&m2), 300);
    }
}
