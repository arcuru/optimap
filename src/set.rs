use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::iter::FusedIterator;

use crate::raw::RawTable;
use crate::raw::hash::mix_hash;

/// A hash set using open addressing with SIMD-accelerated group probing,
/// inspired by `boost::unordered_flat_set`.
///
/// Backed by the same engine as [`UnorderedFlatMap`], but stores only keys
/// (values are zero-sized `()`).
pub struct UnorderedFlatSet<T, S = RandomState> {
    table: RawTable<T, ()>,
    hash_builder: S,
}

// ── Constructors ────────────────────────────────────────────────────────────

impl<T> UnorderedFlatSet<T, RandomState> {
    /// Creates an empty set.
    pub fn new() -> Self {
        Self::with_hasher(RandomState::new())
    }

    /// Creates an empty set with at least the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, RandomState::new())
    }
}

impl<T, S> UnorderedFlatSet<T, S> {
    /// Creates an empty set with the given hasher.
    pub fn with_hasher(hash_builder: S) -> Self {
        UnorderedFlatSet {
            table: RawTable::new(),
            hash_builder,
        }
    }

    /// Creates an empty set with the given capacity and hasher.
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        UnorderedFlatSet {
            table: RawTable::with_capacity(capacity),
            hash_builder,
        }
    }

    /// Returns the number of elements in the set.
    #[inline]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Returns true if the set contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Returns the number of elements the set can hold without rehashing.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.table.capacity()
    }

    /// Returns a reference to the set's hasher.
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }

    /// Clears the set.
    pub fn clear(&mut self) {
        self.table.clear();
    }
}

// ── Core operations ─────────────────────────────────────────────────────────

impl<T, S> UnorderedFlatSet<T, S>
where
    T: Hash + Eq,
    S: BuildHasher,
{
    #[inline]
    fn hash_key<Q: Hash + ?Sized>(&self, key: &Q) -> u64 {
        use std::hash::Hasher;
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        mix_hash(hasher.finish())
    }

    /// Returns true if the set contains the given value.
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.table
            .get(unsafe { &*(value as *const Q as *const T) }, &self.hash_builder)
            .is_some()
    }

    /// Adds a value to the set. Returns true if newly inserted, false if already present.
    pub fn insert(&mut self, value: T) -> bool {
        if self.table.num_groups == 0 {
            self.table.allocate(1);
        }

        let h = self.hash_key(&value);

        if self.table.find_with_hash(&value, h).is_some() {
            return false;
        }

        if self.table.len >= self.table.max_load {
            let new_groups = if self.table.num_groups == 0 {
                1
            } else {
                self.table.num_groups * 2
            };
            self.table.rehash_with(new_groups, &self.hash_builder);
        }

        self.table.insert_no_check(h, value, ());
        true
    }

    /// Removes a value from the set. Returns true if it was present.
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.table
            .remove(
                unsafe { &*(value as *const Q as *const T) },
                &self.hash_builder,
            )
            .is_some()
    }

    /// Iterate over values.
    pub fn iter(&self) -> SetIter<'_, T> {
        SetIter {
            inner: self.table.iter_slots(),
        }
    }

    // ── Set operations ──────────────────────────────────────────────────

    /// Returns true if `self` has no elements in common with `other`.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        if self.len() <= other.len() {
            self.iter().all(|v| !other.contains(v))
        } else {
            other.iter().all(|v| !self.contains(v))
        }
    }

    /// Returns true if every element in `self` is also in `other`.
    pub fn is_subset(&self, other: &Self) -> bool {
        if self.len() > other.len() {
            return false;
        }
        self.iter().all(|v| other.contains(v))
    }

    /// Returns true if every element in `other` is also in `self`.
    pub fn is_superset(&self, other: &Self) -> bool {
        other.is_subset(self)
    }

    /// Returns the union of `self` and `other` as a new set.
    pub fn union(&self, other: &Self) -> Self
    where
        T: Clone,
        S: Clone + Default,
    {
        let mut result = self.clone();
        for item in other.iter() {
            result.insert(item.clone());
        }
        result
    }

    /// Returns the intersection of `self` and `other` as a new set.
    pub fn intersection(&self, other: &Self) -> Self
    where
        T: Clone,
        S: Default,
    {
        let (smaller, larger) = if self.len() <= other.len() {
            (self, other)
        } else {
            (other, self)
        };
        let mut result = Self::with_hasher(S::default());
        for item in smaller.iter() {
            if larger.contains(item) {
                result.insert(item.clone());
            }
        }
        result
    }

    /// Returns elements in `self` but not in `other` as a new set.
    pub fn difference(&self, other: &Self) -> Self
    where
        T: Clone,
        S: Default,
    {
        let mut result = Self::with_hasher(S::default());
        for item in self.iter() {
            if !other.contains(item) {
                result.insert(item.clone());
            }
        }
        result
    }

    /// Returns elements in either set but not both as a new set.
    pub fn symmetric_difference(&self, other: &Self) -> Self
    where
        T: Clone,
        S: Clone + Default,
    {
        let mut result = Self::with_hasher(S::default());
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

// ── Iterators ───────────────────────────────────────────────────────────────

pub struct SetIter<'a, T> {
    inner: crate::raw::SlotIter<'a, T, ()>,
}

impl<'a, T> Iterator for SetIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let (gi, si) = self.inner.next()?;
        let bucket = unsafe { &*self.inner.table.bucket_ptr(gi, si) };
        Some(&bucket.0)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T> FusedIterator for SetIter<'_, T> {}

pub struct SetIntoIter<T> {
    table: RawTable<T, ()>,
    group: usize,
    slot: usize,
}

impl<T> Iterator for SetIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            while self.group < self.table.num_groups {
                while self.slot < crate::raw::group::GROUP_SIZE {
                    let meta = *self
                        .table
                        .metadata
                        .add(self.group * crate::raw::group::META_GROUP_BYTES + self.slot);
                    let gi = self.group;
                    let si = self.slot;
                    self.slot += 1;
                    if meta >= 2 {
                        let ptr = self.table.bucket_ptr(gi, si);
                        let (key, ()) = ptr.read();
                        *self.table.metadata.add(
                            gi * crate::raw::group::META_GROUP_BYTES + si,
                        ) = crate::raw::group::EMPTY;
                        self.table.len -= 1;
                        return Some(key);
                    }
                }
                self.group += 1;
                self.slot = 0;
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.table.len, Some(self.table.len))
    }
}

impl<T> ExactSizeIterator for SetIntoIter<T> {}
impl<T> FusedIterator for SetIntoIter<T> {}

// ── Trait implementations ───────────────────────────────────────────────────

impl<T> Default for UnorderedFlatSet<T, RandomState> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, S> IntoIterator for UnorderedFlatSet<T, S> {
    type Item = T;
    type IntoIter = SetIntoIter<T>;

    fn into_iter(self) -> SetIntoIter<T> {
        let table = unsafe { std::ptr::read(&self.table) };
        std::mem::forget(self);
        SetIntoIter {
            table,
            group: 0,
            slot: 0,
        }
    }
}

impl<'a, T, S> IntoIterator for &'a UnorderedFlatSet<T, S>
where
    T: Hash + Eq,
    S: BuildHasher,
{
    type Item = &'a T;
    type IntoIter = SetIter<'a, T>;

    fn into_iter(self) -> SetIter<'a, T> {
        self.iter()
    }
}

impl<T, S> FromIterator<T> for UnorderedFlatSet<T, S>
where
    T: Hash + Eq,
    S: BuildHasher + Default,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut set = Self::with_capacity_and_hasher(lower, S::default());
        for item in iter {
            set.insert(item);
        }
        set
    }
}

impl<T, S> Extend<T> for UnorderedFlatSet<T, S>
where
    T: Hash + Eq,
    S: BuildHasher,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.insert(item);
        }
    }
}

impl<T, S> fmt::Debug for UnorderedFlatSet<T, S>
where
    T: Hash + Eq + fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl<T, S> Clone for UnorderedFlatSet<T, S>
where
    T: Clone,
    S: Clone,
{
    fn clone(&self) -> Self {
        UnorderedFlatSet {
            table: self.table.clone(),
            hash_builder: self.hash_builder.clone(),
        }
    }
}

impl<T, S> PartialEq for UnorderedFlatSet<T, S>
where
    T: Hash + Eq,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().all(|v| other.contains(v))
    }
}

impl<T, S> Eq for UnorderedFlatSet<T, S>
where
    T: Hash + Eq,
    S: BuildHasher,
{
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_set_operations() {
        let mut set = UnorderedFlatSet::new();
        assert!(set.is_empty());

        assert!(set.insert(1));
        assert!(set.insert(2));
        assert!(set.insert(3));
        assert!(!set.insert(2)); // duplicate

        assert_eq!(set.len(), 3);
        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(set.contains(&3));
        assert!(!set.contains(&4));
    }

    #[test]
    fn remove() {
        let mut set = UnorderedFlatSet::new();
        set.insert(1);
        set.insert(2);

        assert!(set.remove(&1));
        assert!(!set.remove(&1)); // already removed
        assert_eq!(set.len(), 1);
        assert!(!set.contains(&1));
        assert!(set.contains(&2));
    }

    #[test]
    fn from_iterator() {
        let set: UnorderedFlatSet<i32> = vec![1, 2, 3, 2, 1].into_iter().collect();
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn into_iter() {
        let mut set = UnorderedFlatSet::new();
        for i in 0..10 {
            set.insert(i);
        }

        let mut items: Vec<i32> = set.into_iter().collect();
        items.sort();
        assert_eq!(items, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn set_union() {
        let a: UnorderedFlatSet<i32> = vec![1, 2, 3].into_iter().collect();
        let b: UnorderedFlatSet<i32> = vec![3, 4, 5].into_iter().collect();

        let u = a.union(&b);
        assert_eq!(u.len(), 5);
        for i in 1..=5 {
            assert!(u.contains(&i));
        }
    }

    #[test]
    fn set_intersection() {
        let a: UnorderedFlatSet<i32> = vec![1, 2, 3, 4].into_iter().collect();
        let b: UnorderedFlatSet<i32> = vec![3, 4, 5, 6].into_iter().collect();

        let inter = a.intersection(&b);
        assert_eq!(inter.len(), 2);
        assert!(inter.contains(&3));
        assert!(inter.contains(&4));
    }

    #[test]
    fn set_difference() {
        let a: UnorderedFlatSet<i32> = vec![1, 2, 3, 4].into_iter().collect();
        let b: UnorderedFlatSet<i32> = vec![3, 4, 5, 6].into_iter().collect();

        let diff = a.difference(&b);
        assert_eq!(diff.len(), 2);
        assert!(diff.contains(&1));
        assert!(diff.contains(&2));
    }

    #[test]
    fn set_symmetric_difference() {
        let a: UnorderedFlatSet<i32> = vec![1, 2, 3].into_iter().collect();
        let b: UnorderedFlatSet<i32> = vec![2, 3, 4].into_iter().collect();

        let sd = a.symmetric_difference(&b);
        assert_eq!(sd.len(), 2);
        assert!(sd.contains(&1));
        assert!(sd.contains(&4));
    }

    #[test]
    fn subset_superset() {
        let a: UnorderedFlatSet<i32> = vec![1, 2].into_iter().collect();
        let b: UnorderedFlatSet<i32> = vec![1, 2, 3].into_iter().collect();

        assert!(a.is_subset(&b));
        assert!(!b.is_subset(&a));
        assert!(b.is_superset(&a));
    }

    #[test]
    fn disjoint() {
        let a: UnorderedFlatSet<i32> = vec![1, 2].into_iter().collect();
        let b: UnorderedFlatSet<i32> = vec![3, 4].into_iter().collect();
        let c: UnorderedFlatSet<i32> = vec![2, 3].into_iter().collect();

        assert!(a.is_disjoint(&b));
        assert!(!a.is_disjoint(&c));
    }

    #[test]
    fn large_set() {
        let mut set = UnorderedFlatSet::new();
        for i in 0..10_000 {
            set.insert(i);
        }
        assert_eq!(set.len(), 10_000);
        for i in 0..10_000 {
            assert!(set.contains(&i));
        }
    }

    #[test]
    fn clone_and_eq() {
        let a: UnorderedFlatSet<i32> = vec![1, 2, 3].into_iter().collect();
        let b = a.clone();
        assert_eq!(a, b);

        let c: UnorderedFlatSet<i32> = vec![1, 2].into_iter().collect();
        assert_ne!(a, c);
    }
}
