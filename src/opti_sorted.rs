//! `OptiSortedMap` and `OptiSortedSet` — smart wrappers for sorted containers.
//!
//! Currently backed by [`FlatBTree`], the only sorted backend. These wrappers
//! provide a consistent API surface alongside [`OptiMap`] and [`OptiSet`],
//! and are the natural extension point if additional sorted backends are added.

use std::borrow::Borrow;
use std::fmt;
use std::hash::Hash;
use std::hash::BuildHasher;

use crate::flat_btree::FlatBTree;
use crate::map::DefaultHashBuilder;
use crate::traits::SortedMap;

// ═══════════════════════════════════════════════════════════════════════════
// OptiSortedMap
// ═══════════════════════════════════════════════════════════════════════════

/// A smart sorted map backed by [`FlatBTree`].
///
/// `OptiSortedMap` provides sorted iteration, range queries, and
/// first/last access in addition to the standard map operations.
/// Currently delegates to `FlatBTree`; the wrapper exists for API
/// consistency with [`OptiMap`] and as an extension point for future
/// sorted backends.
///
/// # Examples
///
/// ```
/// use optimap::OptiSortedMap;
///
/// let mut map = OptiSortedMap::new();
/// map.insert(3, "three");
/// map.insert(1, "one");
/// map.insert(2, "two");
///
/// // Sorted iteration:
/// let keys: Vec<_> = map.iter_sorted().map(|(k, _)| *k).collect();
/// assert_eq!(keys, vec![1, 2, 3]);
///
/// // Range queries:
/// let range: Vec<_> = map.range(1..3).map(|(k, _)| *k).collect();
/// assert_eq!(range, vec![1, 2]);
///
/// // First/last:
/// assert_eq!(map.first_key_value(), Some((&1, &"one")));
/// assert_eq!(map.last_key_value(), Some((&3, &"three")));
/// ```
pub struct OptiSortedMap<K, V, S = DefaultHashBuilder> {
    inner: FlatBTree<K, V, S>,
}

// ── Constructors ───────────────────────────────────────────────────────────

impl<K: Ord + Clone, V> OptiSortedMap<K, V> {
    /// Create an empty sorted map.
    pub fn new() -> Self {
        OptiSortedMap { inner: FlatBTree::new() }
    }

    /// Create a sorted map with at least the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        OptiSortedMap { inner: FlatBTree::with_capacity(capacity) }
    }
}

// ── Core map operations ────────────────────────────────────────────────────

impl<K: Ord + Clone, V, S: BuildHasher + Default> OptiSortedMap<K, V, S> {
    /// Insert a key-value pair. Returns the previous value if the key existed.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.inner.insert(key, value)
    }

    /// Look up a value by key.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.get(key)
    }

    /// Returns the key-value pair corresponding to the key.
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.get_key_value(key)
    }

    /// Look up a value by key, returning a mutable reference.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.get_mut(key)
    }

    /// Remove a key, returning its value if present.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.remove(key)
    }

    /// Removes a key, returning the key and value if present.
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.remove_entry(key)
    }

    /// Whether the map contains the given key.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.contains_key(key)
    }

    /// Number of elements in the map.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Number of elements the map can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Remove all elements, keeping allocated memory.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Reserves capacity for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        self.inner.reserve(additional);
    }

    /// Shrinks the capacity as much as possible.
    pub fn shrink_to_fit(&mut self) {
        self.inner.shrink_to_fit();
    }

    /// Iterate over key-value pairs in insertion order (unordered).
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.inner.iter()
    }

    /// Iterate over key-value pairs with mutable values.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.inner.iter_mut()
    }

    /// Iterate over keys.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.inner.keys()
    }

    /// Iterate over values.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.inner.values()
    }

    /// Iterate over mutable values.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.inner.values_mut()
    }

    /// Retains only the elements specified by the predicate.
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        self.inner.retain(f);
    }

    /// Clears the map, returning all key-value pairs as an iterator.
    pub fn drain(&mut self) -> impl Iterator<Item = (K, V)> {
        self.inner.drain()
    }

    /// Tries to insert a key-value pair, failing if the key already exists.
    pub fn try_insert(&mut self, key: K, value: V) -> Result<(), crate::traits::OccupiedError<K, V>> {
        self.inner.try_insert(key, value)
    }

    /// Creates a consuming iterator over the keys.
    pub fn into_keys(self) -> impl Iterator<Item = K> {
        self.inner.into_keys()
    }

    /// Creates a consuming iterator over the values.
    pub fn into_values(self) -> impl Iterator<Item = V> {
        self.inner.into_values()
    }

    // ── Sorted operations ──────────────────────────────────────────────────

    /// Returns a reference to the first (minimum) key-value pair.
    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        self.inner.first_key_value()
    }

    /// Returns a reference to the last (maximum) key-value pair.
    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        self.inner.last_key_value()
    }

    /// Removes and returns the first (minimum) key-value pair.
    pub fn pop_first(&mut self) -> Option<(K, V)> {
        self.inner.pop_first()
    }

    /// Removes and returns the last (maximum) key-value pair.
    pub fn pop_last(&mut self) -> Option<(K, V)> {
        self.inner.pop_last()
    }

    /// Iterate over all key-value pairs in sorted order.
    pub fn iter_sorted(&self) -> impl Iterator<Item = (&K, &V)> {
        self.inner.iter_sorted()
    }

    /// Iterate over key-value pairs within the given range, in sorted order.
    pub fn range<'a, Q, R>(&'a self, range: R) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: Borrow<Q> + 'a,
        V: 'a,
        Q: Ord + ?Sized,
        R: std::ops::RangeBounds<Q> + 'a,
    {
        self.inner.range(range)
    }
}

// ── Trait implementations ──────────────────────────────────────────────────

impl<K: Ord + Clone, V> Default for OptiSortedMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord + Clone + fmt::Debug, V: fmt::Debug, S: BuildHasher + Default> fmt::Debug
    for OptiSortedMap<K, V, S>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.inner.iter_sorted()).finish()
    }
}

impl<K: Ord + Clone, V: Clone, S: BuildHasher + Default + Clone> Clone
    for OptiSortedMap<K, V, S>
{
    fn clone(&self) -> Self {
        OptiSortedMap { inner: self.inner.clone() }
    }
}

impl<K: Ord + Clone + Hash + Eq, V: PartialEq, S: BuildHasher + Default> PartialEq
    for OptiSortedMap<K, V, S>
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<K: Ord + Clone + Hash + Eq, V: Eq, S: BuildHasher + Default> Eq
    for OptiSortedMap<K, V, S> {}

impl<K: Hash + Eq + Ord + Clone, V> FromIterator<(K, V)> for OptiSortedMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut map = Self::with_capacity(lower);
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

impl<K: Hash + Eq + Ord + Clone, V> Extend<(K, V)> for OptiSortedMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K: Ord + Clone, V, S: BuildHasher + Default> IntoIterator for OptiSortedMap<K, V, S> {
    type Item = (K, V);
    type IntoIter = <FlatBTree<K, V, S> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<K, Q, V, S> std::ops::Index<&Q> for OptiSortedMap<K, V, S>
where
    K: Ord + Clone + Borrow<Q>,
    Q: Ord + ?Sized,
    S: BuildHasher + Default,
{
    type Output = V;

    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

// ── Map trait impl ─────────────────────────────────────────────────────────

impl<K: Hash + Eq + Ord + Clone, V, S: BuildHasher + Default> crate::Map<K, V>
    for OptiSortedMap<K, V, S>
{
    fn new() -> Self {
        OptiSortedMap { inner: FlatBTree::with_hasher(S::default()) }
    }
    fn with_capacity(capacity: usize) -> Self {
        OptiSortedMap { inner: FlatBTree::with_capacity_and_hasher(capacity, S::default()) }
    }
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.inner.insert(key, value)
    }
    fn get<Q>(&self, key: &Q) -> Option<&V>
    where K: Borrow<Q>, Q: Hash + Eq + ?Sized,
    {
        // FlatBTree's Map impl handles Hash+Eq lookup
        crate::Map::get(&self.inner, key)
    }
    fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where K: Borrow<Q>, Q: Hash + Eq + ?Sized,
    {
        crate::Map::get_key_value(&self.inner, key)
    }
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where K: Borrow<Q>, Q: Hash + Eq + ?Sized,
    {
        crate::Map::get_mut(&mut self.inner, key)
    }
    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where K: Borrow<Q>, Q: Hash + Eq + ?Sized,
    {
        crate::Map::remove(&mut self.inner, key)
    }
    fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where K: Borrow<Q>, Q: Hash + Eq + ?Sized,
    {
        crate::Map::remove_entry(&mut self.inner, key)
    }
    fn contains_key<Q>(&self, key: &Q) -> bool
    where K: Borrow<Q>, Q: Hash + Eq + ?Sized,
    {
        crate::Map::contains_key(&self.inner, key)
    }
    fn len(&self) -> usize { self.inner.len() }
    fn capacity(&self) -> usize { self.inner.capacity() }
    fn clear(&mut self) { self.inner.clear() }
    fn reserve(&mut self, additional: usize) { self.inner.reserve(additional) }
    fn shrink_to_fit(&mut self) { self.inner.shrink_to_fit() }

    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where K: 'a, V: 'a,
    { self.inner.iter() }

    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (&'a K, &'a mut V)>
    where K: 'a, V: 'a,
    { self.inner.iter_mut() }

    fn retain<F>(&mut self, f: F)
    where F: FnMut(&K, &mut V) -> bool,
    { self.inner.retain(f) }

    fn drain(&mut self) -> impl Iterator<Item = (K, V)> {
        self.inner.drain()
    }

    fn try_insert(
        &mut self,
        key: K,
        value: V,
    ) -> Result<(), crate::traits::OccupiedError<K, V>> {
        OptiSortedMap::try_insert(self, key, value)
    }

    fn into_keys(self) -> impl Iterator<Item = K> {
        OptiSortedMap::into_keys(self)
    }

    fn into_values(self) -> impl Iterator<Item = V> {
        OptiSortedMap::into_values(self)
    }
}

// ── SortedMap trait impl ───────────────────────────────────────────────────

impl<K: Ord + Clone, V, S: BuildHasher + Default> crate::SortedMap<K, V>
    for OptiSortedMap<K, V, S>
{
    fn first_key_value(&self) -> Option<(&K, &V)> {
        self.inner.first_key_value()
    }
    fn last_key_value(&self) -> Option<(&K, &V)> {
        self.inner.last_key_value()
    }
    fn pop_first(&mut self) -> Option<(K, V)> {
        self.inner.pop_first()
    }
    fn pop_last(&mut self) -> Option<(K, V)> {
        self.inner.pop_last()
    }
    fn iter_sorted<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where K: 'a, V: 'a,
    {
        self.inner.iter_sorted()
    }
    fn range<'a, Q, R>(&'a self, range: R) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: Borrow<Q> + 'a,
        V: 'a,
        Q: Ord + ?Sized,
        R: std::ops::RangeBounds<Q> + 'a,
    {
        self.inner.range(range)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// OptiSortedSet
// ═══════════════════════════════════════════════════════════════════════════

/// A smart sorted set backed by [`FlatBTree`].
///
/// `OptiSortedSet` provides sorted iteration, range queries, and
/// first/last access in addition to the standard set operations.
/// Under the hood it wraps an `OptiSortedMap<T, ()>`.
///
/// # Examples
///
/// ```
/// use optimap::OptiSortedSet;
///
/// let mut set = OptiSortedSet::new();
/// set.insert(3);
/// set.insert(1);
/// set.insert(2);
///
/// // Sorted iteration:
/// let items: Vec<_> = set.iter_sorted().copied().collect();
/// assert_eq!(items, vec![1, 2, 3]);
///
/// // Range queries:
/// let range: Vec<_> = set.range(1..3).copied().collect();
/// assert_eq!(range, vec![1, 2]);
///
/// // First/last:
/// assert_eq!(set.first(), Some(&1));
/// assert_eq!(set.last(), Some(&3));
/// ```
pub struct OptiSortedSet<T, S = DefaultHashBuilder> {
    inner: OptiSortedMap<T, (), S>,
}

// ── Constructors ───────────────────────────────────────────────────────────

impl<T: Ord + Clone> OptiSortedSet<T> {
    /// Create an empty sorted set.
    pub fn new() -> Self {
        OptiSortedSet { inner: OptiSortedMap::new() }
    }

    /// Create a sorted set with at least the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        OptiSortedSet { inner: OptiSortedMap::with_capacity(capacity) }
    }
}

// ── Core set operations ────────────────────────────────────────────────────

impl<T: Ord + Clone, S: BuildHasher + Default> OptiSortedSet<T, S> {
    /// Adds a value to the set. Returns `true` if newly inserted.
    pub fn insert(&mut self, value: T) -> bool {
        self.inner.insert(value, ()).is_none()
    }

    /// Returns `true` if the set contains the given value.
    pub fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.contains_key(value)
    }

    /// Returns a reference to the value in the set matching the given value.
    pub fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.get_key_value(value).map(|(k, _)| k)
    }

    /// Removes a value from the set. Returns `true` if it was present.
    pub fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.inner.remove(value).is_some()
    }

    /// Removes and returns the value in the set matching the given value.
    pub fn take<Q>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
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

    /// Number of elements the set can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Remove all elements, keeping allocated memory.
    pub fn clear(&mut self) {
        self.inner.clear();
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

    // ── Sorted operations ──────────────────────────────────────────────────

    /// Returns a reference to the first (minimum) element.
    pub fn first(&self) -> Option<&T> {
        self.inner.first_key_value().map(|(k, _)| k)
    }

    /// Returns a reference to the last (maximum) element.
    pub fn last(&self) -> Option<&T> {
        self.inner.last_key_value().map(|(k, _)| k)
    }

    /// Removes and returns the first (minimum) element.
    pub fn pop_first(&mut self) -> Option<T> {
        self.inner.pop_first().map(|(k, _)| k)
    }

    /// Removes and returns the last (maximum) element.
    pub fn pop_last(&mut self) -> Option<T> {
        self.inner.pop_last().map(|(k, _)| k)
    }

    /// Iterate over all elements in sorted order.
    pub fn iter_sorted(&self) -> impl Iterator<Item = &T> {
        self.inner.iter_sorted().map(|(k, _)| k)
    }

    /// Iterate over elements within the given range, in sorted order.
    pub fn range<'a, Q, R>(&'a self, range: R) -> impl Iterator<Item = &'a T>
    where
        T: Borrow<Q> + 'a,
        Q: Ord + ?Sized,
        R: std::ops::RangeBounds<Q> + 'a,
    {
        self.inner.range(range).map(|(k, _)| k)
    }
}

// ── Set algebra operations ─────────────────────────────────────────────────

impl<T: Hash + Eq + Ord + Clone> OptiSortedSet<T> {
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

impl<T: Ord + Clone> Default for OptiSortedSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord + Clone + fmt::Debug, S: BuildHasher + Default> fmt::Debug
    for OptiSortedSet<T, S>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter_sorted()).finish()
    }
}

impl<T: Ord + Clone, S: BuildHasher + Default + Clone> Clone for OptiSortedSet<T, S> {
    fn clone(&self) -> Self {
        OptiSortedSet { inner: self.inner.clone() }
    }
}

impl<T: Hash + Eq + Ord + Clone> PartialEq for OptiSortedSet<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().all(|v| other.contains(v))
    }
}

impl<T: Hash + Eq + Ord + Clone> Eq for OptiSortedSet<T> {}

impl<T: Hash + Eq + Ord + Clone> FromIterator<T> for OptiSortedSet<T> {
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

impl<T: Hash + Eq + Ord + Clone> Extend<T> for OptiSortedSet<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.insert(item);
        }
    }
}

impl<T: Ord + Clone, S: BuildHasher + Default> IntoIterator for OptiSortedSet<T, S> {
    type Item = T;
    type IntoIter = std::iter::Map<
        <FlatBTree<T, (), S> as IntoIterator>::IntoIter,
        fn((T, ())) -> T,
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter().map(|(k, _)| k)
    }
}

// ── Set trait impl ─────────────────────────────────────────────────────────

impl<T: Hash + Eq + Ord + Clone> crate::Set<T> for OptiSortedSet<T> {
    fn new() -> Self { OptiSortedSet::new() }
    fn with_capacity(capacity: usize) -> Self { OptiSortedSet::with_capacity(capacity) }
    fn insert(&mut self, value: T) -> bool { OptiSortedSet::insert(self, value) }

    fn contains<Q>(&self, value: &Q) -> bool
    where T: Borrow<Q>, Q: Hash + Eq + ?Sized,
    {
        // Set trait requires Hash+Eq, delegate through Map trait on inner
        crate::Map::contains_key(&self.inner, value)
    }

    fn get<Q>(&self, value: &Q) -> Option<&T>
    where T: Borrow<Q>, Q: Hash + Eq + ?Sized,
    {
        crate::Map::get_key_value(&self.inner, value).map(|(k, _)| k)
    }

    fn remove<Q>(&mut self, value: &Q) -> bool
    where T: Borrow<Q>, Q: Hash + Eq + ?Sized,
    {
        crate::Map::remove(&mut self.inner, value).is_some()
    }

    fn take<Q>(&mut self, value: &Q) -> Option<T>
    where T: Borrow<Q>, Q: Hash + Eq + ?Sized,
    {
        crate::Map::remove_entry(&mut self.inner, value).map(|(k, _)| k)
    }

    fn len(&self) -> usize { self.inner.len() }
    fn capacity(&self) -> usize { self.inner.capacity() }
    fn clear(&mut self) { self.inner.clear() }
    fn reserve(&mut self, additional: usize) { self.inner.reserve(additional) }
    fn shrink_to_fit(&mut self) { self.inner.shrink_to_fit() }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T> where T: 'a {
        self.inner.iter().map(|(k, _)| k)
    }

    fn retain<F>(&mut self, mut f: F) where F: FnMut(&T) -> bool {
        self.inner.retain(|k, _| f(k));
    }

    fn drain(&mut self) -> impl Iterator<Item = T> {
        self.inner.drain().map(|(k, _)| k)
    }
}

// ── SortedSet trait impl ───────────────────────────────────────────────────

impl<T: Hash + Eq + Ord + Clone> crate::SortedSet<T> for OptiSortedSet<T> {
    fn first(&self) -> Option<&T> {
        OptiSortedSet::first(self)
    }
    fn last(&self) -> Option<&T> {
        OptiSortedSet::last(self)
    }
    fn pop_first(&mut self) -> Option<T> {
        OptiSortedSet::pop_first(self)
    }
    fn pop_last(&mut self) -> Option<T> {
        OptiSortedSet::pop_last(self)
    }
    fn iter_sorted<'a>(&'a self) -> impl Iterator<Item = &'a T> where T: 'a {
        OptiSortedSet::iter_sorted(self)
    }
    fn range<'a, Q, R>(&'a self, range: R) -> impl Iterator<Item = &'a T>
    where
        T: Borrow<Q> + 'a,
        Q: Ord + ?Sized,
        R: std::ops::RangeBounds<Q> + 'a,
    {
        OptiSortedSet::range(self, range)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── OptiSortedMap tests ────────────────────────────────────────────────

    mod sorted_map {
        use super::*;

        #[test]
        fn basic() {
            let mut map = OptiSortedMap::new();
            map.insert(3, "three");
            map.insert(1, "one");
            map.insert(2, "two");
            assert_eq!(map.len(), 3);
            assert_eq!(map.get(&2), Some(&"two"));
        }

        #[test]
        fn sorted_iteration() {
            let mut map = OptiSortedMap::new();
            for i in [5, 3, 1, 4, 2] {
                map.insert(i, i * 10);
            }
            let keys: Vec<_> = map.iter_sorted().map(|(k, _)| *k).collect();
            assert_eq!(keys, vec![1, 2, 3, 4, 5]);
        }

        #[test]
        fn range_query() {
            let map: OptiSortedMap<i32, i32> =
                (0..10).map(|i| (i, i * 10)).collect();
            let range: Vec<_> = map.range(3..7).map(|(k, _)| *k).collect();
            assert_eq!(range, vec![3, 4, 5, 6]);
        }

        #[test]
        fn first_last() {
            let map: OptiSortedMap<i32, &str> =
                vec![(1, "a"), (5, "e"), (3, "c")].into_iter().collect();
            assert_eq!(map.first_key_value(), Some((&1, &"a")));
            assert_eq!(map.last_key_value(), Some((&5, &"e")));
        }

        #[test]
        fn pop_first_last() {
            let mut map: OptiSortedMap<i32, i32> =
                (1..=5).map(|i| (i, i)).collect();
            assert_eq!(map.pop_first(), Some((1, 1)));
            assert_eq!(map.pop_last(), Some((5, 5)));
            assert_eq!(map.len(), 3);
        }

        #[test]
        fn remove_and_contains() {
            let mut map = OptiSortedMap::new();
            map.insert(1, 10);
            map.insert(2, 20);
            assert!(map.contains_key(&1));
            assert_eq!(map.remove(&1), Some(10));
            assert!(!map.contains_key(&1));
        }

        #[test]
        fn clear_and_capacity() {
            let mut map = OptiSortedMap::<i32, i32>::with_capacity(100);
            for i in 0..50 {
                map.insert(i, i);
            }
            map.clear();
            assert!(map.is_empty());
        }

        #[test]
        fn retain() {
            let mut map: OptiSortedMap<i32, i32> =
                (0..20).map(|i| (i, i)).collect();
            map.retain(|&k, _| k % 2 == 0);
            assert_eq!(map.len(), 10);
            assert!(map.contains_key(&0));
            assert!(!map.contains_key(&1));
        }

        #[test]
        fn drain() {
            let mut map: OptiSortedMap<i32, i32> =
                (0..10).map(|i| (i, i)).collect();
            let mut drained: Vec<_> = map.drain().collect();
            drained.sort();
            assert_eq!(drained.len(), 10);
            assert!(map.is_empty());
        }

        #[test]
        fn clone_and_eq() {
            let map: OptiSortedMap<i32, i32> =
                (0..50).map(|i| (i, i)).collect();
            let map2 = map.clone();
            assert_eq!(map, map2);
        }

        #[test]
        fn into_iterator() {
            let map: OptiSortedMap<i32, i32> =
                vec![(3, 30), (1, 10), (2, 20)].into_iter().collect();
            let mut pairs: Vec<_> = map.into_iter().collect();
            pairs.sort();
            assert_eq!(pairs, vec![(1, 10), (2, 20), (3, 30)]);
        }

        #[test]
        fn index() {
            let mut map = OptiSortedMap::new();
            map.insert("a", 1);
            map.insert("b", 2);
            assert_eq!(map[&"a"], 1);
        }

        #[test]
        fn map_trait_usage() {
            use crate::Map;

            fn fill<M: Map<i32, i32>>(m: &mut M, n: i32) {
                for i in 0..n {
                    m.insert(i, i);
                }
            }

            let mut map = OptiSortedMap::new();
            fill(&mut map, 100);
            assert_eq!(map.len(), 100);
        }

        #[test]
        fn sorted_map_trait_usage() {
            use crate::SortedMap;

            fn first_key<M: SortedMap<i32, i32>>(m: &M) -> Option<i32> {
                m.first_key_value().map(|(k, _)| *k)
            }

            let map: OptiSortedMap<i32, i32> =
                vec![(5, 50), (1, 10), (3, 30)].into_iter().collect();
            assert_eq!(first_key(&map), Some(1));
        }
    }

    // ── OptiSortedSet tests ────────────────────────────────────────────────

    mod sorted_set {
        use super::*;

        #[test]
        fn basic() {
            let mut set = OptiSortedSet::new();
            assert!(set.insert(3));
            assert!(set.insert(1));
            assert!(set.insert(2));
            assert!(!set.insert(2));
            assert_eq!(set.len(), 3);
            assert!(set.contains(&2));
        }

        #[test]
        fn sorted_iteration() {
            let set: OptiSortedSet<i32> = vec![5, 3, 1, 4, 2].into_iter().collect();
            let items: Vec<_> = set.iter_sorted().copied().collect();
            assert_eq!(items, vec![1, 2, 3, 4, 5]);
        }

        #[test]
        fn range_query() {
            let set: OptiSortedSet<i32> = (0..10).collect();
            let range: Vec<_> = set.range(3..7).copied().collect();
            assert_eq!(range, vec![3, 4, 5, 6]);
        }

        #[test]
        fn first_last() {
            let set: OptiSortedSet<i32> = vec![5, 1, 3].into_iter().collect();
            assert_eq!(set.first(), Some(&1));
            assert_eq!(set.last(), Some(&5));
        }

        #[test]
        fn pop_first_last() {
            let mut set: OptiSortedSet<i32> = (1..=5).collect();
            assert_eq!(set.pop_first(), Some(1));
            assert_eq!(set.pop_last(), Some(5));
            assert_eq!(set.len(), 3);
        }

        #[test]
        fn remove_and_take() {
            let mut set = OptiSortedSet::new();
            set.insert(1);
            set.insert(2);
            assert!(set.remove(&1));
            assert!(!set.remove(&1));
            assert_eq!(set.take(&2), Some(2));
            assert!(set.is_empty());
        }

        #[test]
        fn retain() {
            let mut set: OptiSortedSet<i32> = (0..20).collect();
            set.retain(|&x| x % 2 == 0);
            assert_eq!(set.len(), 10);
        }

        #[test]
        fn drain() {
            let mut set: OptiSortedSet<i32> = (0..10).collect();
            let mut drained: Vec<_> = set.drain().collect();
            drained.sort();
            assert_eq!(drained.len(), 10);
            assert!(set.is_empty());
        }

        #[test]
        fn clone_and_eq() {
            let set: OptiSortedSet<i32> = (0..50).collect();
            let set2 = set.clone();
            assert_eq!(set, set2);
        }

        #[test]
        fn into_iterator() {
            let set: OptiSortedSet<i32> = vec![3, 1, 2].into_iter().collect();
            let mut items: Vec<_> = set.into_iter().collect();
            items.sort();
            assert_eq!(items, vec![1, 2, 3]);
        }

        #[test]
        fn set_algebra() {
            let a: OptiSortedSet<i32> = vec![1, 2, 3].into_iter().collect();
            let b: OptiSortedSet<i32> = vec![2, 3, 4].into_iter().collect();

            assert_eq!(a.union(&b).len(), 4);
            assert_eq!(a.intersection(&b).len(), 2);
            assert_eq!(a.difference(&b).len(), 1);
            assert_eq!(a.symmetric_difference(&b).len(), 2);
            assert!(!a.is_disjoint(&b));
            let c: OptiSortedSet<i32> = vec![10, 11].into_iter().collect();
            assert!(a.is_disjoint(&c));
        }

        #[test]
        fn set_trait_usage() {
            use crate::Set;

            fn fill<S: Set<i32>>(s: &mut S, n: i32) {
                for i in 0..n {
                    s.insert(i);
                }
            }

            let mut set = OptiSortedSet::new();
            fill(&mut set, 100);
            assert_eq!(set.len(), 100);
        }

        #[test]
        fn sorted_set_trait_usage() {
            use crate::SortedSet;

            fn first_elem<S: SortedSet<i32>>(s: &S) -> Option<i32> {
                s.first().copied()
            }

            let set: OptiSortedSet<i32> = vec![5, 1, 3].into_iter().collect();
            assert_eq!(first_elem(&set), Some(1));
        }

        #[test]
        fn for_loop() {
            let set: OptiSortedSet<i32> = vec![1, 2, 3].into_iter().collect();
            let mut sum = 0;
            for v in set {
                sum += v;
            }
            assert_eq!(sum, 6);
        }
    }
}
