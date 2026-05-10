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

impl<T: Hash + Eq + Ord + Clone> OptiSet<T> {
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

    /// Create a set pinned to the `FlatBTree` backend (sorted iteration,
    /// range queries). Requires `T: Ord + Clone`.
    pub fn flat_btree() -> Self {
        OptiSet { inner: OptiMap::flat_btree() }
    }

    /// Create a set pinned to `FlatBTree` with the given capacity.
    /// Equivalent to `with_capacity_and_hint(capacity, Hint::Sorted)` but
    /// pinned (no auto-transition on resize).
    pub fn sorted_with_capacity(capacity: usize) -> Self {
        OptiSet { inner: OptiMap::sorted_with_capacity(capacity) }
    }

    /// Build a sorted set from input already sorted with no duplicates.
    /// Pins the backend to `FlatBTree`. See
    /// [`FlatBTree::from_sorted_iter`] for details.
    pub fn from_sorted_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        OptiSet {
            inner: OptiMap::from_sorted_iter(iter.into_iter().map(|t| (t, ()))),
        }
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

impl<T: Hash + Eq + Ord + Clone> OptiSet<T> {
    /// Adds a value to the set. Returns `true` if newly inserted.
    #[inline(always)]
    pub fn insert(&mut self, value: T) -> bool {
        self.inner.insert(value, ()).is_none()
    }

    /// Returns `true` if the set contains the given value.
    #[inline(always)]
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.inner.contains_key(value)
    }

    /// Returns a reference to the value in the set matching the given value.
    #[inline(always)]
    pub fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.inner.get_key_value(value).map(|(k, _)| k)
    }

    /// Removes a value from the set. Returns `true` if it was present.
    #[inline(always)]
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.inner.remove(value).is_some()
    }

    /// Removes and returns the value in the set matching the given value.
    #[inline(always)]
    pub fn take<Q>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.inner.remove_entry(value).map(|(k, _)| k)
    }

    /// Number of elements in the set.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the set is empty.
    #[inline(always)]
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

    // ── Sorted operations (require FlatBTree backend) ──────────────────────

    /// Returns a reference to the first (minimum) element.
    ///
    /// # Panics
    ///
    /// Panics if the current backend is not [`FlatBTree`]. Use
    /// [`OptiSet::flat_btree`] or [`Hint::Sorted`] to guarantee a sorted backend.
    pub fn first(&self) -> Option<&T> {
        self.inner.first_key_value().map(|(k, _)| k)
    }

    /// Returns a reference to the last (maximum) element.
    ///
    /// # Panics
    ///
    /// Panics if the current backend is not [`FlatBTree`].
    pub fn last(&self) -> Option<&T> {
        self.inner.last_key_value().map(|(k, _)| k)
    }

    /// Removes and returns the first (minimum) element.
    ///
    /// # Panics
    ///
    /// Panics if the current backend is not [`FlatBTree`].
    pub fn pop_first(&mut self) -> Option<T> {
        self.inner.pop_first().map(|(k, _)| k)
    }

    /// Removes and returns the last (maximum) element.
    ///
    /// # Panics
    ///
    /// Panics if the current backend is not [`FlatBTree`].
    pub fn pop_last(&mut self) -> Option<T> {
        self.inner.pop_last().map(|(k, _)| k)
    }

    /// Iterate over all elements in sorted order.
    ///
    /// # Panics
    ///
    /// Panics if the current backend is not [`FlatBTree`].
    pub fn iter_sorted(&self) -> impl Iterator<Item = &T> {
        self.inner.iter_sorted().map(|(k, _)| k)
    }

    /// Iterate over elements within the given range, in sorted order.
    ///
    /// # Panics
    ///
    /// Panics if the current backend is not [`FlatBTree`].
    pub fn range<'a, Q, R>(&'a self, range: R) -> impl Iterator<Item = &'a T>
    where
        T: Borrow<Q> + 'a,
        Q: Ord + ?Sized,
        R: std::ops::RangeBounds<Q> + 'a,
    {
        self.inner.range(range).map(|(k, _)| k)
    }

    /// Splits the set at `at`, keeping elements `< at` in `self` and returning
    /// a new set with elements `>= at`.
    ///
    /// # Panics
    ///
    /// Panics if the current backend is not [`FlatBTree`].
    pub fn split_off<Q>(&mut self, at: &Q) -> Self
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        OptiSet {
            inner: self.inner.split_off(at),
        }
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// # Panics
    ///
    /// Panics if either `self` or `other` has a non-[`FlatBTree`] backend.
    pub fn append(&mut self, other: &mut Self) {
        self.inner.append(&mut other.inner);
    }
}

// ── Set algebra operations ─────────────────────────────────────────────────

impl<T: Hash + Eq + Ord + Clone> OptiSet<T> {
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

impl<T: Hash + Eq + Ord + Clone> Default for OptiSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Hash + Eq + Ord + Clone + fmt::Debug> fmt::Debug for OptiSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T: Hash + Eq + Ord + Clone> Clone for OptiSet<T> {
    fn clone(&self) -> Self {
        OptiSet { inner: self.inner.clone() }
    }
}

impl<T: Hash + Eq + Ord + Clone> PartialEq for OptiSet<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().all(|v| other.contains(v))
    }
}

impl<T: Hash + Eq + Ord + Clone> Eq for OptiSet<T> {}

impl<T: Hash + Eq + Ord + Clone> FromIterator<T> for OptiSet<T> {
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

impl<T: Hash + Eq + Ord + Clone> Extend<T> for OptiSet<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.insert(item);
        }
    }
}

impl<T: Hash + Eq + Ord + Clone> IntoIterator for OptiSet<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(mut self) -> Self::IntoIter {
        self.drain().collect::<Vec<_>>().into_iter()
    }
}

// ── Bitwise operators (std::HashSet parity) ─────────────────────────────────

impl<T: Hash + Eq + Ord + Clone> std::ops::BitOr<&OptiSet<T>> for &OptiSet<T> {
    type Output = OptiSet<T>;
    /// Union: `a | b`.
    fn bitor(self, rhs: &OptiSet<T>) -> OptiSet<T> {
        self.union(rhs)
    }
}

impl<T: Hash + Eq + Ord + Clone> std::ops::BitAnd<&OptiSet<T>> for &OptiSet<T> {
    type Output = OptiSet<T>;
    /// Intersection: `a & b`.
    fn bitand(self, rhs: &OptiSet<T>) -> OptiSet<T> {
        self.intersection(rhs)
    }
}

impl<T: Hash + Eq + Ord + Clone> std::ops::BitXor<&OptiSet<T>> for &OptiSet<T> {
    type Output = OptiSet<T>;
    /// Symmetric difference: `a ^ b`.
    fn bitxor(self, rhs: &OptiSet<T>) -> OptiSet<T> {
        self.symmetric_difference(rhs)
    }
}

impl<T: Hash + Eq + Ord + Clone> std::ops::Sub<&OptiSet<T>> for &OptiSet<T> {
    type Output = OptiSet<T>;
    /// Difference: `a - b`.
    fn sub(self, rhs: &OptiSet<T>) -> OptiSet<T> {
        self.difference(rhs)
    }
}

// `Set` is intentionally NOT implemented for `OptiSet` — the FlatBTree
// variant brings `Q: Ord` into the dispatch path, which the `Set` trait's
// `Q: Hash + Eq` bound can't accommodate. Use inherent methods, or wrap
// a specific backend type if a `Set`-bound generic is needed.

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

    // The `set_trait_usage` test was removed when OptiSet stopped implementing
    // the `Set` trait (FlatBTree backend brings `Q: Ord`, incompatible with
    // `Set`'s `Q: Hash + Eq`). The inherent API still covers the same ground.

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

    #[test]
    fn set_operators() {
        let a: OptiSet<i32> = vec![1, 2, 3].into_iter().collect();
        let b: OptiSet<i32> = vec![2, 3, 4].into_iter().collect();
        assert_eq!((&a | &b).len(), 4);
        let inter = &a & &b;
        assert!(inter.contains(&2) && inter.contains(&3) && inter.len() == 2);
        let xor = &a ^ &b;
        assert!(xor.contains(&1) && xor.contains(&4) && xor.len() == 2);
        let diff = &a - &b;
        assert!(diff.contains(&1) && diff.len() == 1);
    }

    mod sorted_ops {
        use super::*;

        fn sorted_set() -> OptiSet<i32> {
            OptiSet::flat_btree()
        }

        #[test]
        fn first_last() {
            let mut set = sorted_set();
            for v in [5, 1, 3] {
                set.insert(v);
            }
            assert_eq!(set.first(), Some(&1));
            assert_eq!(set.last(), Some(&5));
        }

        #[test]
        fn first_last_empty() {
            let set: OptiSet<i32> = OptiSet::flat_btree();
            assert_eq!(set.first(), None);
            assert_eq!(set.last(), None);
        }

        #[test]
        fn pop_first_last() {
            let mut set = sorted_set();
            for v in 1..=5 {
                set.insert(v);
            }
            assert_eq!(set.pop_first(), Some(1));
            assert_eq!(set.pop_last(), Some(5));
            assert_eq!(set.len(), 3);
            assert_eq!(set.first(), Some(&2));
            assert_eq!(set.last(), Some(&4));
        }

        #[test]
        fn iter_sorted() {
            let mut set = sorted_set();
            for v in [5, 3, 1, 4, 2] {
                set.insert(v);
            }
            let items: Vec<_> = set.iter_sorted().copied().collect();
            assert_eq!(items, vec![1, 2, 3, 4, 5]);
        }

        #[test]
        fn range_query() {
            let mut set = sorted_set();
            for v in 0..10 {
                set.insert(v);
            }
            let range: Vec<_> = set.range(3..7).copied().collect();
            assert_eq!(range, vec![3, 4, 5, 6]);
        }

        #[test]
        #[should_panic(expected = "first_key_value requires a FlatBTree backend")]
        fn first_panics_on_hash_backend() {
            let set: OptiSet<u64> = OptiSet::splitsies();
            let _ = set.first();
        }

        #[test]
        #[should_panic(expected = "pop_first requires a FlatBTree backend")]
        fn pop_first_panics_on_hash_backend() {
            let mut set: OptiSet<u64> = OptiSet::ipo();
            let _ = set.pop_first();
        }

        #[test]
        #[should_panic(expected = "iter_sorted requires a FlatBTree backend")]
        fn iter_sorted_panics_on_hash_backend() {
            let set: OptiSet<u64> = OptiSet::gaps();
            let _ = set.iter_sorted();
        }

        #[test]
        #[should_panic(expected = "range requires a FlatBTree backend")]
        fn range_panics_on_hash_backend() {
            let set: OptiSet<u64> = OptiSet::ufm();
            let _ = set.range(..);
        }

        #[test]
        fn split_off() {
            let mut set = sorted_set();
            for v in 0..100 {
                set.insert(v);
            }
            let upper = set.split_off(&50);
            assert_eq!(set.len(), 50);
            assert_eq!(upper.len(), 50);
            assert_eq!(set.last(), Some(&49));
            assert_eq!(upper.first(), Some(&50));
            for v in 0..50 {
                assert!(set.contains(&v));
            }
            for v in 50..100 {
                assert!(!set.contains(&v));
                assert!(upper.contains(&v));
            }
        }

        #[test]
        fn split_off_empty_sides() {
            let mut set = sorted_set();
            for v in 0..10 {
                set.insert(v);
            }
            let upper = set.split_off(&0);
            assert!(set.is_empty());
            assert_eq!(upper.len(), 10);

            let mut set2 = sorted_set();
            set2.insert(5);
            let upper2 = set2.split_off(&100);
            assert!(upper2.is_empty());
            assert_eq!(set2.len(), 1);
        }

        #[test]
        fn append_disjoint() {
            let mut a = sorted_set();
            let mut b = sorted_set();
            for v in 0..50 {
                a.insert(v);
            }
            for v in 50..100 {
                b.insert(v);
            }
            a.append(&mut b);
            assert_eq!(a.len(), 100);
            assert!(b.is_empty());
            for v in 0..100 {
                assert!(a.contains(&v));
            }
        }

        #[test]
        fn append_empty() {
            let mut a: OptiSet<i32> = OptiSet::flat_btree();
            let mut b = sorted_set();
            b.insert(1);
            b.insert(2);
            a.append(&mut b);
            assert_eq!(a.len(), 2);
            assert!(b.is_empty());

            let mut c = sorted_set();
            let mut d: OptiSet<i32> = OptiSet::flat_btree();
            c.insert(1);
            let len_before = c.len();
            c.append(&mut d);
            assert_eq!(c.len(), len_before);
        }

        #[test]
        #[should_panic(expected = "split_off requires a FlatBTree backend")]
        fn split_off_panics_on_hash_backend() {
            let mut set: OptiSet<u64> = OptiSet::splitsies();
            let _ = set.split_off(&42);
        }

        #[test]
        #[should_panic(expected = "append requires a FlatBTree backend")]
        fn append_panics_on_hash_backend() {
            let mut a: OptiSet<u64> = OptiSet::splitsies();
            let mut b: OptiSet<u64> = OptiSet::flat_btree();
            a.append(&mut b);
        }

        #[test]
        fn split_off_then_append_roundtrip() {
            let mut set = sorted_set();
            for v in 0..200 {
                set.insert(v);
            }
            let mut upper = set.split_off(&100);
            assert_eq!(set.len(), 100);
            assert_eq!(upper.len(), 100);
            set.append(&mut upper);
            assert_eq!(set.len(), 200);
            assert!(upper.is_empty());
            for v in 0..200 {
                assert!(set.contains(&v));
            }
        }
    }
}
