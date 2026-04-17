//! `OptiSet` — a smart wrapper that dynamically selects a hash set backend.
//!
//! This is the set counterpart to [`OptiMap`]. It wraps `OptiMap<T, ()>`
//! and provides the standard set interface, including workload hints and
//! backend pinning.

use std::borrow::Borrow;
use std::fmt;
use std::hash::Hash;

use crate::map::DefaultHashBuilder;
use crate::optimap::{Hint, MapType, OptiMap};

/// A smart hash set that dynamically selects its backend.
///
/// `OptiSet` is the set counterpart to [`OptiMap`]. Under the hood it
/// wraps an `OptiMap<T, ()>` and exposes the standard set API.
/// Backend selection, hints, and pinning all work identically.
///
/// # Examples
///
/// ```
/// use optimap::OptiSet;
///
/// // Let the policy choose:
/// let mut set = OptiSet::new();
/// set.insert("hello");
/// set.insert("world");
/// assert!(set.contains("hello"));
///
/// // Pin a specific backend:
/// let mut set = OptiSet::<u64>::ipo();
/// set.insert(42);
///
/// // Hint at workload:
/// use optimap::Hint;
/// let mut set = OptiSet::<u64>::with_hint(Hint::Churn);
/// ```
pub struct OptiSet<T, S = DefaultHashBuilder> {
    inner: OptiMap<T, (), S>,
}

// ── Constructors ───────────────────────────────────────────────────────────

impl<T: Hash + Eq> OptiSet<T> {
    /// Create an empty set, letting the policy engine choose the backend.
    pub fn new() -> Self {
        OptiSet { inner: OptiMap::new() }
    }

    /// Create a set with at least the given capacity, backend chosen by policy.
    pub fn with_capacity(capacity: usize) -> Self {
        OptiSet { inner: OptiMap::with_capacity(capacity) }
    }

    /// Create a set with the given workload hint.
    pub fn with_hint(hint: Hint) -> Self {
        OptiSet { inner: OptiMap::with_hint(hint) }
    }

    /// Create a set with both a capacity and a workload hint.
    pub fn with_capacity_and_hint(capacity: usize, hint: Hint) -> Self {
        OptiSet { inner: OptiMap::with_capacity_and_hint(capacity, hint) }
    }

    /// Create a set pinned to the `UnorderedFlatMap` backend.
    pub fn ufm() -> Self {
        OptiSet { inner: OptiMap::ufm() }
    }

    /// Create a set pinned to the `Splitsies` backend.
    pub fn splitsies() -> Self {
        OptiSet { inner: OptiMap::splitsies() }
    }

    /// Create a set pinned to the `InPlaceOverflow` backend.
    pub fn ipo() -> Self {
        OptiSet { inner: OptiMap::ipo() }
    }

    /// Create a set pinned to the `Gaps` backend.
    pub fn gaps() -> Self {
        OptiSet { inner: OptiMap::gaps() }
    }

    /// Create a set pinned to the `IPO64` backend.
    pub fn ipo64() -> Self {
        OptiSet { inner: OptiMap::ipo64() }
    }

    /// Create a set pinned to a specific backend type.
    pub fn with_type(map_type: MapType) -> Self {
        OptiSet { inner: OptiMap::with_type(map_type) }
    }

    /// Create a set pinned to a specific backend with the given capacity.
    pub fn with_type_and_capacity(map_type: MapType, capacity: usize) -> Self {
        OptiSet { inner: OptiMap::with_type_and_capacity(map_type, capacity) }
    }
}

// ── Core set operations ────────────────────────────────────────────────────

impl<T: Hash + Eq> OptiSet<T> {
    /// Adds a value to the set. Returns `true` if newly inserted.
    pub fn insert(&mut self, value: T) -> bool {
        self.inner.insert(value, ()).is_none()
    }

    /// Returns `true` if the set contains the given value.
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.contains_key(value)
    }

    /// Returns a reference to the value in the set matching the given value.
    pub fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.get_key_value(value).map(|(k, _)| k)
    }

    /// Removes a value from the set. Returns `true` if it was present.
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.remove(value).is_some()
    }

    /// Removes and returns the value in the set matching the given value.
    pub fn take<Q>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.inner.remove_entry(value).map(|(k, _)| k)
    }

    /// Number of elements in the set.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Number of elements the set can hold without rehashing.
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Remove all elements, keeping allocated memory.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Which backend is currently active.
    pub fn map_type(&self) -> MapType {
        self.inner.map_type()
    }

    /// Reserves capacity for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    /// Shrinks the capacity as much as possible.
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit();
    }

    /// Iterate over elements in arbitrary order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.inner.iter().map(|(k, _)| k)
    }

    /// Retains only the elements specified by the predicate.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.inner.retain(|k, _| f(k));
    }

    /// Clears the set, returning all elements as an iterator.
    pub fn drain(&mut self) -> impl Iterator<Item = T> {
        self.inner.drain().map(|(k, _)| k)
    }
}

// ── Set algebra operations ─────────────────────────────────────────────────

impl<T: Hash + Eq + Clone> OptiSet<T> {
    /// Returns `true` if `self` has no elements in common with `other`.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        if self.len() <= other.len() {
            self.iter().all(|v| !other.contains(v))
        } else {
            other.iter().all(|v| !self.contains(v))
        }
    }

    /// Returns `true` if every element in `self` is also in `other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        if self.len() > other.len() {
            return false;
        }
        self.iter().all(|v| other.contains(v))
    }

    /// Returns `true` if every element in `other` is also in `self`.
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    /// Returns the union of `self` and `other` as a new set.
    pub fn union(&self, other: &Self) -> Self {
        let mut result = Self::with_capacity(self.len() + other.len());
        for item in self.iter() {
            result.insert(item.clone());
        }
        for item in other.iter() {
            result.insert(item.clone());
        }
        result
    }

    /// Returns the intersection of `self` and `other` as a new set.
    pub fn intersection(&self, other: &Self) -> Self {
        let mut result = Self::new();
        let (smaller, larger) = if self.len() <= other.len() {
            (self, other)
        } else {
            (other, self)
        };
        for item in smaller.iter() {
            if larger.contains(item) {
                result.insert(item.clone());
            }
        }
        result
    }

    /// Returns elements in `self` but not in `other`.
    pub fn difference(&self, other: &Self) -> Self {
        let mut result = Self::new();
        for item in self.iter() {
            if !other.contains(item) {
                result.insert(item.clone());
            }
        }
        result
    }

    /// Returns elements in either set but not both.
    pub fn symmetric_difference(&self, other: &Self) -> Self {
        let mut result = Self::new();
        for item in self.iter() {
            if !other.contains(item) {
                result.insert(item.clone());
            }
        }
        for item in other.iter() {
            if !self.contains(item) {
                result.insert(item.clone());
            }
        }
        result
    }
}

// ── Trait implementations ──────────────────────────────────────────────────

impl<T: Hash + Eq> Default for OptiSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Hash + Eq + fmt::Debug> fmt::Debug for OptiSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T: Hash + Eq + Clone> Clone for OptiSet<T> {
    fn clone(&self) -> Self {
        OptiSet { inner: self.inner.clone() }
    }
}

impl<T: Hash + Eq> PartialEq for OptiSet<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().all(|v| other.contains(v))
    }
}

impl<T: Hash + Eq> Eq for OptiSet<T> {}

impl<T: Hash + Eq> FromIterator<T> for OptiSet<T> {
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

impl<T: Hash + Eq> Extend<T> for OptiSet<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.insert(item);
        }
    }
}

impl<T: Hash + Eq> IntoIterator for OptiSet<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(mut self) -> Self::IntoIter {
        self.drain().collect::<Vec<_>>().into_iter()
    }
}

// ── Set trait impl ─────────────────────────────────────────────────────────

impl<T: Hash + Eq> crate::Set<T> for OptiSet<T> {
    fn new() -> Self { OptiSet::new() }
    fn with_capacity(capacity: usize) -> Self { OptiSet::with_capacity(capacity) }
    fn insert(&mut self, value: T) -> bool { OptiSet::insert(self, value) }

    fn contains<Q>(&self, value: &Q) -> bool
    where T: Borrow<Q>, Q: Hash + Eq + ?Sized,
    { OptiSet::contains(self, value) }

    fn get<Q>(&self, value: &Q) -> Option<&T>
    where T: Borrow<Q>, Q: Hash + Eq + ?Sized,
    { OptiSet::get(self, value) }

    fn remove<Q>(&mut self, value: &Q) -> bool
    where T: Borrow<Q>, Q: Hash + Eq + ?Sized,
    { OptiSet::remove(self, value) }

    fn take<Q>(&mut self, value: &Q) -> Option<T>
    where T: Borrow<Q>, Q: Hash + Eq + ?Sized,
    { OptiSet::take(self, value) }

    fn len(&self) -> usize { OptiSet::len(self) }
    fn capacity(&self) -> usize { OptiSet::capacity(self) }
    fn clear(&mut self) { OptiSet::clear(self) }
    fn reserve(&mut self, additional: usize) { OptiSet::reserve(self, additional) }
    fn shrink_to_fit(&mut self) { OptiSet::shrink_to_fit(self) }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T> where T: 'a {
        OptiSet::iter(self)
    }

    fn retain<F>(&mut self, f: F) where F: FnMut(&T) -> bool {
        OptiSet::retain(self, f)
    }

    fn drain(&mut self) -> impl Iterator<Item = T> {
        OptiSet::drain(self)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_auto() {
        let mut set = OptiSet::new();
        set.insert("hello");
        set.insert("world");
        assert!(set.contains("hello"));
        assert!(set.contains("world"));
        assert!(!set.contains("foo"));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn pinned_backends() {
        for mt in [MapType::Ufm, MapType::Splitsies, MapType::Ipo, MapType::Gaps, MapType::Ipo64] {
            let mut set = OptiSet::<u64>::with_type(mt);
            for i in 0..100 {
                set.insert(i);
            }
            assert_eq!(set.len(), 100);
            assert_eq!(set.map_type(), mt);
            assert!(set.contains(&50));
        }
    }

    #[test]
    fn named_constructors() {
        assert_eq!(OptiSet::<u64>::ufm().map_type(), MapType::Ufm);
        assert_eq!(OptiSet::<u64>::splitsies().map_type(), MapType::Splitsies);
        assert_eq!(OptiSet::<u64>::ipo().map_type(), MapType::Ipo);
        assert_eq!(OptiSet::<u64>::gaps().map_type(), MapType::Gaps);
        assert_eq!(OptiSet::<u64>::ipo64().map_type(), MapType::Ipo64);
    }

    #[test]
    fn hint_constructors() {
        let s = OptiSet::<u64>::with_hint(Hint::ReadHeavy);
        assert_eq!(s.map_type(), MapType::Ipo);

        let s = OptiSet::<u64>::with_hint(Hint::Churn);
        assert_eq!(s.map_type(), MapType::Splitsies);

        let s = OptiSet::<u64>::with_hint(Hint::Iteration);
        assert_eq!(s.map_type(), MapType::Gaps);
    }

    #[test]
    fn remove_and_take() {
        let mut set = OptiSet::new();
        set.insert(1u64);
        set.insert(2);
        assert!(set.remove(&1));
        assert!(!set.remove(&1));
        assert_eq!(set.take(&2), Some(2));
        assert!(set.is_empty());
    }

    #[test]
    fn clear_and_capacity() {
        let mut set = OptiSet::<u64>::with_capacity(100);
        assert!(set.capacity() >= 100);
        for i in 0..50 {
            set.insert(i);
        }
        set.clear();
        assert!(set.is_empty());
    }

    #[test]
    fn iter_and_retain() {
        let mut set: OptiSet<u64> = (0..20).collect();
        set.retain(|&x| x % 2 == 0);
        assert_eq!(set.len(), 10);
        assert!(set.contains(&0));
        assert!(!set.contains(&1));
    }

    #[test]
    fn drain() {
        let mut set: OptiSet<u64> = (0..50).collect();
        let mut drained: Vec<u64> = set.drain().collect();
        drained.sort();
        assert_eq!(drained.len(), 50);
        assert!(set.is_empty());
    }

    #[test]
    fn from_iter_and_extend() {
        let mut set: OptiSet<u64> = vec![1, 2, 3].into_iter().collect();
        assert_eq!(set.len(), 3);
        set.extend(vec![3, 4, 5]);
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn clone_and_eq() {
        let set: OptiSet<u64> = (0..100).collect();
        let set2 = set.clone();
        assert_eq!(set, set2);
    }

    #[test]
    fn into_iterator() {
        let set: OptiSet<u64> = (0..50).collect();
        let mut items: Vec<u64> = set.into_iter().collect();
        items.sort();
        assert_eq!(items.len(), 50);
        assert_eq!(items[0], 0);
        assert_eq!(items[49], 49);
    }

    #[test]
    fn set_algebra() {
        let a: OptiSet<i32> = vec![1, 2, 3].into_iter().collect();
        let b: OptiSet<i32> = vec![2, 3, 4].into_iter().collect();

        let u = a.union(&b);
        assert_eq!(u.len(), 4);

        let i = a.intersection(&b);
        assert_eq!(i.len(), 2);
        assert!(i.contains(&2) && i.contains(&3));

        let d = a.difference(&b);
        assert_eq!(d.len(), 1);
        assert!(d.contains(&1));

        let sd = a.symmetric_difference(&b);
        assert_eq!(sd.len(), 2);
        assert!(sd.contains(&1) && sd.contains(&4));

        assert!(!a.is_disjoint(&b));
        let c: OptiSet<i32> = vec![10, 11].into_iter().collect();
        assert!(a.is_disjoint(&c));

        let sub: OptiSet<i32> = vec![1, 2].into_iter().collect();
        assert!(sub.is_subset(&a));
        assert!(a.is_superset(&sub));
    }

    #[test]
    fn set_trait_usage() {
        use crate::Set;

        fn fill<S: Set<u64>>(s: &mut S, n: u64) {
            for i in 0..n {
                s.insert(i);
            }
        }

        let mut set = OptiSet::new();
        fill(&mut set, 100);
        assert_eq!(set.len(), 100);
    }

    #[test]
    fn debug_display() {
        let mut set = OptiSet::new();
        set.insert(1u64);
        let s = format!("{:?}", set);
        assert!(s.contains("1"));
    }

    #[test]
    fn for_loop() {
        let set: OptiSet<u64> = vec![1, 2, 3].into_iter().collect();
        let mut sum = 0u64;
        for v in set {
            sum += v;
        }
        assert_eq!(sum, 6);
    }
}
