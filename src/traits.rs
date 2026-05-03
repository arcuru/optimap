//! Common traits for all OptiMap map implementations.
//!
//! Three trait flavors:
//!
//! - [`HashedMap`] — hash-dispatched (`K: Hash + Eq`, `Q: Hash + Eq`). The
//!   natural fit for `UnorderedFlatMap`, `Splitsies`, `InPlaceOverflow`,
//!   `IPO64`, `Gaps`, hashbrown, `std::HashMap`.
//! - [`SortedMap`] — ord-dispatched (`K: Ord`, `Q: Ord`). Adds sorted-only
//!   ops: `first/last_key_value`, `pop_first/last`, `range`, `split_off`,
//!   `append`. Implemented by `FlatBTree`, `std::BTreeMap`, `OptiSortedMap`.
//! - [`Map`] — universal facade (`K: Hash + Eq + Ord`). Use this in generic
//!   code that wants to accept both flavors and doesn't care about ordering.
//!
//! Most concrete types implement only one of `HashedMap` / `SortedMap`. The
//! exceptions are `FlatBTree` and `OptiSortedMap`, which carry a `HashedMap`
//! impl (O(n) leaf scan) for backward compatibility with `GenericSet`'s
//! `HashedMap` bound — the [`Map`] facade for these types delegates to their
//! cheap `SortedMap` path. The hash function is an implementation detail of
//! each concrete type, not part of any trait.
//!
//! Users calling methods on concrete types (e.g. `Splitsies::insert`)
//! do NOT need to import these traits — inherent methods work automatically.
//! The traits are only needed for generic code over multiple implementations.

use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash};

/// Error returned by [`HashedMap::try_insert`] when the key already exists.
///
/// Contains the key and value that were not inserted.
#[derive(Debug, PartialEq, Eq)]
pub struct OccupiedError<K, V> {
    /// The key that was not inserted.
    pub key: K,
    /// The value that was not inserted.
    pub value: V,
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Display for OccupiedError<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "failed to insert {:?}, key {:?} already exists",
            self.value, self.key
        )
    }
}

impl<K: fmt::Debug, V: fmt::Debug> std::error::Error for OccupiedError<K, V> {}

/// Core hash-dispatched map interface. Maps keys to values via `Hash + Eq`.
///
/// The hash function is an implementation detail — each concrete type
/// carries its own hasher internally. Generic code uses `HashedMap<K, V>`
/// without knowing or caring about the hasher.
///
/// Sorted-dispatch maps (e.g. `std::BTreeMap`) implement [`SortedMap`]
/// instead. `FlatBTree` and `OptiSortedMap` carry a `HashedMap` impl too
/// (O(n) leaf scan, kept for `GenericSet` compatibility) — generic code
/// that doesn't care about ordering should prefer the [`Map`] facade,
/// which routes those types through `SortedMap`.
///
/// # Usage
///
/// For concrete types, use inherent methods directly (no import needed):
/// ```
/// let mut map = optimap::Splitsies::new();
/// map.insert("hello", 42);
/// ```
///
/// For generic code, import the trait:
/// ```
/// use optimap::HashedMap;
/// fn count<M: HashedMap<String, usize>>(m: &mut M, key: String) {
///     let val = m.get(&key).copied().unwrap_or(0);
///     m.insert(key, val + 1);
/// }
/// ```
pub trait HashedMap<K: Hash + Eq, V> {
    /// Create an empty map with the default hasher.
    fn new() -> Self;

    /// Create a map with at least the specified capacity.
    fn with_capacity(capacity: usize) -> Self;

    /// Insert a key-value pair. Returns the previous value if the key existed.
    fn insert(&mut self, key: K, value: V) -> Option<V>;

    /// Look up a value by key.
    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized;

    /// Returns the key-value pair corresponding to the key.
    fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized;

    /// Look up a value by key, returning a mutable reference.
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized;

    /// Remove a key, returning its value if present.
    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized;

    /// Removes a key from the map, returning the key and value if it was present.
    fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized;

    /// Whether the map contains the given key.
    fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Number of elements in the map.
    fn len(&self) -> usize;

    /// Whether the map is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of elements the map can hold without rehashing.
    fn capacity(&self) -> usize;

    /// Remove all elements, keeping allocated memory.
    fn clear(&mut self);

    /// Reserves capacity for at least `additional` more elements.
    fn reserve(&mut self, additional: usize);

    /// Shrinks the capacity as much as possible.
    fn shrink_to_fit(&mut self);

    /// Iterate over key-value pairs in arbitrary order.
    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a;

    /// Iterate over key-value pairs with mutable values.
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (&'a K, &'a mut V)>
    where
        K: 'a,
        V: 'a;

    /// Iterate over keys.
    fn keys<'a>(&'a self) -> impl Iterator<Item = &'a K>
    where
        K: 'a,
        V: 'a,
    {
        self.iter().map(|(k, _)| k)
    }

    /// Iterate over values.
    fn values<'a>(&'a self) -> impl Iterator<Item = &'a V>
    where
        K: 'a,
        V: 'a,
    {
        self.iter().map(|(_, v)| v)
    }

    /// Iterate over mutable values.
    fn values_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut V>
    where
        K: 'a,
        V: 'a,
    {
        self.iter_mut().map(|(_, v)| v)
    }

    /// Retains only the elements specified by the predicate.
    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&K, &mut V) -> bool;

    /// Clears the map, returning all key-value pairs as an iterator.
    fn drain(&mut self) -> impl Iterator<Item = (K, V)>;

    /// Tries to insert a key-value pair into the map, failing if the key
    /// already exists.
    ///
    /// Returns `Ok(())` if the pair was inserted, or `Err(OccupiedError)`
    /// containing the key and value that were not inserted.
    fn try_insert(&mut self, key: K, value: V) -> Result<(), OccupiedError<K, V>> {
        if self.contains_key(&key) {
            Err(OccupiedError { key, value })
        } else {
            self.insert(key, value);
            Ok(())
        }
    }

    /// Creates a consuming iterator over the keys of the map.
    fn into_keys(self) -> impl Iterator<Item = K>
    where
        Self: Sized;

    /// Creates a consuming iterator over the values of the map.
    fn into_values(self) -> impl Iterator<Item = V>
    where
        Self: Sized;
}

/// Core ord-dispatched map interface. Maps keys to values via `Ord`.
///
/// Unlike [`HashedMap`], this does not require `Hash` — it works with any
/// key type that supports ordering. Implementors store keys in sorted order
/// and use comparison to navigate (e.g. `FlatBTree`, `std::BTreeMap`).
///
/// Most types implement only one of `HashedMap` / `SortedMap`. `FlatBTree`
/// and `OptiSortedMap` carry a `HashedMap` impl (O(n) leaf scan) for
/// `GenericSet` compatibility — the [`Map`] facade routes them through
/// `SortedMap` instead.
pub trait SortedMap<K: Ord, V> {
    // ── Construction ────────────────────────────────────────────────────

    /// Create an empty map.
    fn new() -> Self;

    /// Create a map with at least the specified capacity.
    fn with_capacity(capacity: usize) -> Self;

    // ── Core CRUD ───────────────────────────────────────────────────────

    /// Insert a key-value pair. Returns the previous value if the key existed.
    fn insert(&mut self, key: K, value: V) -> Option<V>;

    /// Look up a value by key.
    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized;

    /// Returns the key-value pair corresponding to the key.
    fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized;

    /// Look up a value by key, returning a mutable reference.
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized;

    /// Remove a key, returning its value if present.
    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized;

    /// Removes a key from the map, returning the key and value if it was present.
    fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized;

    /// Whether the map contains the given key.
    fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Tries to insert a key-value pair into the map, failing if the key
    /// already exists.
    fn try_insert(&mut self, key: K, value: V) -> Result<(), OccupiedError<K, V>> {
        if self.contains_key(&key) {
            Err(OccupiedError { key, value })
        } else {
            self.insert(key, value);
            Ok(())
        }
    }

    // ── Size / capacity ─────────────────────────────────────────────────

    /// Number of elements in the map.
    fn len(&self) -> usize;

    /// Whether the map is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of elements the map can hold without reallocation.
    fn capacity(&self) -> usize;

    /// Remove all elements, keeping allocated memory.
    fn clear(&mut self);

    /// Reserves capacity for at least `additional` more elements.
    fn reserve(&mut self, additional: usize);

    /// Shrinks the capacity as much as possible.
    fn shrink_to_fit(&mut self);

    // ── Iteration ───────────────────────────────────────────────────────

    /// Iterate over key-value pairs. For sorted maps this yields pairs in
    /// ascending key order — same as [`Self::iter_sorted`].
    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a;

    /// Iterate over key-value pairs with mutable values, in ascending key order.
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (&'a K, &'a mut V)>
    where
        K: 'a,
        V: 'a;

    /// Iterate over keys in ascending order.
    fn keys<'a>(&'a self) -> impl Iterator<Item = &'a K>
    where
        K: 'a,
        V: 'a,
    {
        self.iter().map(|(k, _)| k)
    }

    /// Iterate over values in ascending key order.
    fn values<'a>(&'a self) -> impl Iterator<Item = &'a V>
    where
        K: 'a,
        V: 'a,
    {
        self.iter().map(|(_, v)| v)
    }

    /// Iterate over mutable values in ascending key order.
    fn values_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut V>
    where
        K: 'a,
        V: 'a,
    {
        self.iter_mut().map(|(_, v)| v)
    }

    /// Retains only the elements specified by the predicate.
    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&K, &mut V) -> bool;

    /// Clears the map, returning all key-value pairs as an iterator.
    fn drain(&mut self) -> impl Iterator<Item = (K, V)>;

    /// Creates a consuming iterator over the keys of the map.
    fn into_keys(self) -> impl Iterator<Item = K>
    where
        Self: Sized;

    /// Creates a consuming iterator over the values of the map.
    fn into_values(self) -> impl Iterator<Item = V>
    where
        Self: Sized;

    // ── Sorted-only operations ──────────────────────────────────────────

    /// Returns the first (minimum) key-value pair.
    fn first_key_value(&self) -> Option<(&K, &V)>;

    /// Returns the last (maximum) key-value pair.
    fn last_key_value(&self) -> Option<(&K, &V)>;

    /// Removes and returns the first (minimum) key-value pair.
    fn pop_first(&mut self) -> Option<(K, V)>;

    /// Removes and returns the last (maximum) key-value pair.
    fn pop_last(&mut self) -> Option<(K, V)>;

    /// Iterate over all key-value pairs in sorted order. Same as [`Self::iter`]
    /// for `SortedMap` impls — provided as an explicit name for callers that
    /// want to make the ordering guarantee visible at the call site.
    fn iter_sorted<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a,
    {
        self.iter()
    }

    /// Iterate over key-value pairs within the given range, in sorted order.
    fn range<'a, Q, R>(&'a self, range: R) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: Borrow<Q> + 'a,
        V: 'a,
        Q: Ord + ?Sized,
        R: std::ops::RangeBounds<Q> + 'a;

    /// Splits the map at `at`, keeping keys `< at` in self and returning a
    /// new map containing keys `>= at`.
    fn split_off<Q>(&mut self, at: &Q) -> Self
    where
        Self: Sized,
        K: Borrow<Q>,
        Q: Ord + ?Sized;

    /// Moves all entries from `other` into `self`, leaving `other` empty.
    /// On key collision, `other`'s value wins (matches `std::BTreeMap::append`).
    fn append(&mut self, other: &mut Self)
    where
        Self: Sized;
}

// ── SortedMap impl for std::BTreeMap ────────────────────────────────────────

impl<K: Ord, V> SortedMap<K, V> for std::collections::BTreeMap<K, V> {
    fn new() -> Self {
        std::collections::BTreeMap::new()
    }

    fn with_capacity(_capacity: usize) -> Self {
        // BTreeMap doesn't support capacity hints — accepts and ignores.
        std::collections::BTreeMap::new()
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        std::collections::BTreeMap::insert(self, key, value)
    }

    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        std::collections::BTreeMap::get(self, key)
    }

    fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        std::collections::BTreeMap::get_key_value(self, key)
    }

    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        std::collections::BTreeMap::get_mut(self, key)
    }

    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        std::collections::BTreeMap::remove(self, key)
    }

    fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        std::collections::BTreeMap::remove_entry(self, key)
    }

    fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        std::collections::BTreeMap::contains_key(self, key)
    }

    fn len(&self) -> usize {
        std::collections::BTreeMap::len(self)
    }

    fn is_empty(&self) -> bool {
        std::collections::BTreeMap::is_empty(self)
    }

    fn capacity(&self) -> usize {
        // BTreeMap has no notion of capacity beyond `len`.
        self.len()
    }

    fn clear(&mut self) {
        std::collections::BTreeMap::clear(self)
    }

    fn reserve(&mut self, _additional: usize) {
        // No-op: BTreeMap doesn't support pre-reservation.
    }

    fn shrink_to_fit(&mut self) {
        // No-op: BTreeMap nodes are individually heap-allocated; nothing to shrink.
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a,
    {
        std::collections::BTreeMap::iter(self)
    }

    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (&'a K, &'a mut V)>
    where
        K: 'a,
        V: 'a,
    {
        std::collections::BTreeMap::iter_mut(self)
    }

    fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        std::collections::BTreeMap::retain(self, |k, v| f(k, v))
    }

    fn drain(&mut self) -> impl Iterator<Item = (K, V)> {
        // BTreeMap has no native `drain`; emulate via take + into_iter.
        std::mem::take(self).into_iter()
    }

    fn into_keys(self) -> impl Iterator<Item = K> {
        std::collections::BTreeMap::into_keys(self)
    }

    fn into_values(self) -> impl Iterator<Item = V> {
        std::collections::BTreeMap::into_values(self)
    }

    fn first_key_value(&self) -> Option<(&K, &V)> {
        std::collections::BTreeMap::first_key_value(self)
    }

    fn last_key_value(&self) -> Option<(&K, &V)> {
        std::collections::BTreeMap::last_key_value(self)
    }

    fn pop_first(&mut self) -> Option<(K, V)> {
        std::collections::BTreeMap::pop_first(self)
    }

    fn pop_last(&mut self) -> Option<(K, V)> {
        std::collections::BTreeMap::pop_last(self)
    }

    fn range<'a, Q, R>(&'a self, range: R) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: Borrow<Q> + 'a,
        V: 'a,
        Q: Ord + ?Sized,
        R: std::ops::RangeBounds<Q> + 'a,
    {
        std::collections::BTreeMap::range(self, range)
    }

    fn split_off<Q>(&mut self, at: &Q) -> Self
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        std::collections::BTreeMap::split_off(self, at)
    }

    fn append(&mut self, other: &mut Self) {
        std::collections::BTreeMap::append(self, other)
    }
}

// ── Map facade trait ────────────────────────────────────────────────────────

/// Universal map facade. Delegates to whichever underlying trait
/// ([`HashedMap`] or [`SortedMap`]) the implementor exposes most cheaply.
///
/// Use `Map<K, V>` when generic code wants to accept both hash-dispatched
/// and ord-dispatched maps and doesn't care which storage strategy is used
/// (e.g. a function that only needs CRUD + iteration). The trait's `K`
/// requires both `Hash + Eq` and `Ord` so the same `Q`-borrow signatures
/// satisfy both flavors.
///
/// The trait is intentionally **not dyn-safe** — the generic `<Q>` methods
/// and RPITIT (`impl Iterator<...>`) returns block object safety. For
/// runtime-polymorphic dispatch over heterogeneous backends, use [`OptiMap`].
///
/// Sorted-only operations (`first/last_key_value`, `pop_first/last`,
/// `range`, `split_off`, `append`) stay on [`SortedMap`].
///
/// # Example
///
/// ```
/// use optimap::Map;
/// fn count<M: Map<String, usize>>(m: &mut M, key: String) {
///     let val = Map::get(m, &key).copied().unwrap_or(0);
///     m.insert(key, val + 1);
/// }
/// ```
///
/// [`OptiMap`]: crate::OptiMap
pub trait Map<K, V>
where
    K: Hash + Eq + Ord,
{
    // ── Construction ────────────────────────────────────────────────────

    /// Create an empty map.
    fn new() -> Self;

    /// Create a map with at least the specified capacity.
    fn with_capacity(capacity: usize) -> Self;

    // ── Core CRUD ───────────────────────────────────────────────────────

    /// Insert a key-value pair. Returns the previous value if the key existed.
    fn insert(&mut self, key: K, value: V) -> Option<V>;

    /// Look up a value by key.
    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized;

    /// Returns the key-value pair corresponding to the key.
    fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized;

    /// Look up a value by key, returning a mutable reference.
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized;

    /// Remove a key, returning its value if present.
    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized;

    /// Removes a key from the map, returning the key and value if it was present.
    fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized;

    /// Whether the map contains the given key.
    fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Tries to insert a key-value pair, failing if the key already exists.
    fn try_insert(&mut self, key: K, value: V) -> Result<(), OccupiedError<K, V>> {
        if self.contains_key(&key) {
            Err(OccupiedError { key, value })
        } else {
            self.insert(key, value);
            Ok(())
        }
    }

    // ── Size / capacity ─────────────────────────────────────────────────

    /// Number of elements in the map.
    fn len(&self) -> usize;

    /// Whether the map is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of elements the map can hold without reallocation.
    fn capacity(&self) -> usize;

    /// Remove all elements, keeping allocated memory.
    fn clear(&mut self);

    /// Reserves capacity for at least `additional` more elements.
    fn reserve(&mut self, additional: usize);

    /// Shrinks the capacity as much as possible.
    fn shrink_to_fit(&mut self);

    // ── Iteration ───────────────────────────────────────────────────────

    /// Iterate over key-value pairs. Order depends on the implementor —
    /// hash-dispatched maps yield arbitrary order, sorted maps yield sorted.
    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a;

    /// Iterate over key-value pairs with mutable values.
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (&'a K, &'a mut V)>
    where
        K: 'a,
        V: 'a;

    /// Iterate over keys.
    fn keys<'a>(&'a self) -> impl Iterator<Item = &'a K>
    where
        K: 'a,
        V: 'a,
    {
        self.iter().map(|(k, _)| k)
    }

    /// Iterate over values.
    fn values<'a>(&'a self) -> impl Iterator<Item = &'a V>
    where
        K: 'a,
        V: 'a,
    {
        self.iter().map(|(_, v)| v)
    }

    /// Iterate over mutable values.
    fn values_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut V>
    where
        K: 'a,
        V: 'a,
    {
        self.iter_mut().map(|(_, v)| v)
    }

    /// Retain only the elements specified by the predicate.
    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&K, &mut V) -> bool;

    /// Clear the map, returning all key-value pairs as an iterator.
    fn drain(&mut self) -> impl Iterator<Item = (K, V)>;

    /// Consuming iterator over keys.
    fn into_keys(self) -> impl Iterator<Item = K>
    where
        Self: Sized;

    /// Consuming iterator over values.
    fn into_values(self) -> impl Iterator<Item = V>
    where
        Self: Sized;
}

// ── Macro to generate trait impl that delegates to inherent methods ──────────

macro_rules! impl_map_trait {
    ($type:ident) => {
        impl<K, V, S> $crate::traits::HashedMap<K, V> for $type<K, V, S>
        where
            K: ::std::hash::Hash + Eq,
            S: ::std::hash::BuildHasher + Default,
        {
            fn new() -> Self {
                Self::with_hasher(S::default())
            }
            fn with_capacity(capacity: usize) -> Self {
                Self::with_capacity_and_hasher(capacity, S::default())
            }
            fn insert(&mut self, key: K, value: V) -> Option<V> {
                $type::insert(self, key, value)
            }
            fn get<Q>(&self, key: &Q) -> Option<&V>
            where
                K: ::std::borrow::Borrow<Q>,
                Q: ::std::hash::Hash + Eq + ?Sized,
            {
                $type::get(self, key)
            }
            fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
            where
                K: ::std::borrow::Borrow<Q>,
                Q: ::std::hash::Hash + Eq + ?Sized,
            {
                $type::get_key_value(self, key)
            }
            fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
            where
                K: ::std::borrow::Borrow<Q>,
                Q: ::std::hash::Hash + Eq + ?Sized,
            {
                $type::get_mut(self, key)
            }
            fn remove<Q>(&mut self, key: &Q) -> Option<V>
            where
                K: ::std::borrow::Borrow<Q>,
                Q: ::std::hash::Hash + Eq + ?Sized,
            {
                $type::remove(self, key)
            }
            fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
            where
                K: ::std::borrow::Borrow<Q>,
                Q: ::std::hash::Hash + Eq + ?Sized,
            {
                $type::remove_entry(self, key)
            }
            fn contains_key<Q>(&self, key: &Q) -> bool
            where
                K: ::std::borrow::Borrow<Q>,
                Q: ::std::hash::Hash + Eq + ?Sized,
            {
                $type::contains_key(self, key)
            }
            fn len(&self) -> usize {
                $type::len(self)
            }
            fn capacity(&self) -> usize {
                $type::capacity(self)
            }
            fn clear(&mut self) {
                $type::clear(self)
            }
            fn reserve(&mut self, additional: usize) {
                $type::reserve(self, additional)
            }
            fn shrink_to_fit(&mut self) {
                $type::shrink_to_fit(self)
            }
            fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
            where
                K: 'a,
                V: 'a,
            {
                $type::iter(self)
            }
            fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (&'a K, &'a mut V)>
            where
                K: 'a,
                V: 'a,
            {
                $type::iter_mut(self)
            }
            fn retain<F>(&mut self, f: F)
            where
                F: FnMut(&K, &mut V) -> bool,
            {
                $type::retain(self, f)
            }
            fn drain(&mut self) -> impl Iterator<Item = (K, V)> {
                $type::drain(self)
            }
            fn try_insert(
                &mut self,
                key: K,
                value: V,
            ) -> Result<(), $crate::traits::OccupiedError<K, V>> {
                $type::try_insert(self, key, value)
            }
            fn into_keys(self) -> impl Iterator<Item = K> {
                $type::into_keys(self)
            }
            fn into_values(self) -> impl Iterator<Item = V> {
                $type::into_values(self)
            }
        }

        // Map facade — delegates to the same inherent methods. Bound on
        // K is tightened to Hash + Eq + Ord (vs HashedMap's Hash + Eq).
        impl<K, V, S> $crate::traits::Map<K, V> for $type<K, V, S>
        where
            K: ::std::hash::Hash + Eq + ::std::cmp::Ord,
            S: ::std::hash::BuildHasher + Default,
        {
            fn new() -> Self {
                Self::with_hasher(S::default())
            }
            fn with_capacity(capacity: usize) -> Self {
                Self::with_capacity_and_hasher(capacity, S::default())
            }
            fn insert(&mut self, key: K, value: V) -> Option<V> {
                $type::insert(self, key, value)
            }
            fn get<Q>(&self, key: &Q) -> Option<&V>
            where
                K: ::std::borrow::Borrow<Q>,
                Q: ::std::hash::Hash + Eq + ::std::cmp::Ord + ?Sized,
            {
                $type::get(self, key)
            }
            fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
            where
                K: ::std::borrow::Borrow<Q>,
                Q: ::std::hash::Hash + Eq + ::std::cmp::Ord + ?Sized,
            {
                $type::get_key_value(self, key)
            }
            fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
            where
                K: ::std::borrow::Borrow<Q>,
                Q: ::std::hash::Hash + Eq + ::std::cmp::Ord + ?Sized,
            {
                $type::get_mut(self, key)
            }
            fn remove<Q>(&mut self, key: &Q) -> Option<V>
            where
                K: ::std::borrow::Borrow<Q>,
                Q: ::std::hash::Hash + Eq + ::std::cmp::Ord + ?Sized,
            {
                $type::remove(self, key)
            }
            fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
            where
                K: ::std::borrow::Borrow<Q>,
                Q: ::std::hash::Hash + Eq + ::std::cmp::Ord + ?Sized,
            {
                $type::remove_entry(self, key)
            }
            fn contains_key<Q>(&self, key: &Q) -> bool
            where
                K: ::std::borrow::Borrow<Q>,
                Q: ::std::hash::Hash + Eq + ::std::cmp::Ord + ?Sized,
            {
                $type::contains_key(self, key)
            }
            fn try_insert(
                &mut self,
                key: K,
                value: V,
            ) -> Result<(), $crate::traits::OccupiedError<K, V>> {
                $type::try_insert(self, key, value)
            }
            fn len(&self) -> usize {
                $type::len(self)
            }
            fn capacity(&self) -> usize {
                $type::capacity(self)
            }
            fn clear(&mut self) {
                $type::clear(self)
            }
            fn reserve(&mut self, additional: usize) {
                $type::reserve(self, additional)
            }
            fn shrink_to_fit(&mut self) {
                $type::shrink_to_fit(self)
            }
            fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
            where
                K: 'a,
                V: 'a,
            {
                $type::iter(self)
            }
            fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (&'a K, &'a mut V)>
            where
                K: 'a,
                V: 'a,
            {
                $type::iter_mut(self)
            }
            fn retain<F>(&mut self, f: F)
            where
                F: FnMut(&K, &mut V) -> bool,
            {
                $type::retain(self, f)
            }
            fn drain(&mut self) -> impl Iterator<Item = (K, V)> {
                $type::drain(self)
            }
            fn into_keys(self) -> impl Iterator<Item = K> {
                $type::into_keys(self)
            }
            fn into_values(self) -> impl Iterator<Item = V> {
                $type::into_values(self)
            }
        }
    };
}

pub(crate) use impl_map_trait;

// ── hashbrown implementation ─────────────────────────────────────────────────

impl<K, V, S> HashedMap<K, V> for hashbrown::HashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher + Default,
{
    fn new() -> Self {
        Self::with_hasher(S::default())
    }
    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, S::default())
    }
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        hashbrown::HashMap::insert(self, key, value)
    }
    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        hashbrown::HashMap::get(self, key)
    }
    fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        hashbrown::HashMap::get_key_value(self, key)
    }
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        hashbrown::HashMap::get_mut(self, key)
    }
    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        hashbrown::HashMap::remove(self, key)
    }
    fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        hashbrown::HashMap::remove_entry(self, key)
    }
    fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        hashbrown::HashMap::contains_key(self, key)
    }
    fn len(&self) -> usize {
        hashbrown::HashMap::len(self)
    }
    fn capacity(&self) -> usize {
        hashbrown::HashMap::capacity(self)
    }
    fn clear(&mut self) {
        hashbrown::HashMap::clear(self)
    }
    fn reserve(&mut self, additional: usize) {
        hashbrown::HashMap::reserve(self, additional)
    }
    fn shrink_to_fit(&mut self) {
        hashbrown::HashMap::shrink_to_fit(self)
    }
    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a,
    {
        hashbrown::HashMap::iter(self)
    }
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (&'a K, &'a mut V)>
    where
        K: 'a,
        V: 'a,
    {
        hashbrown::HashMap::iter_mut(self)
    }
    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        hashbrown::HashMap::retain(self, f)
    }
    fn drain(&mut self) -> impl Iterator<Item = (K, V)> {
        hashbrown::HashMap::drain(self)
    }
    fn into_keys(self) -> impl Iterator<Item = K> {
        hashbrown::HashMap::into_keys(self)
    }
    fn into_values(self) -> impl Iterator<Item = V> {
        hashbrown::HashMap::into_values(self)
    }
}

impl<K, V, S> Map<K, V> for hashbrown::HashMap<K, V, S>
where
    K: Hash + Eq + Ord,
    S: BuildHasher + Default,
{
    fn new() -> Self {
        Self::with_hasher(S::default())
    }
    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, S::default())
    }
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        hashbrown::HashMap::insert(self, key, value)
    }
    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        hashbrown::HashMap::get(self, key)
    }
    fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        hashbrown::HashMap::get_key_value(self, key)
    }
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        hashbrown::HashMap::get_mut(self, key)
    }
    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        hashbrown::HashMap::remove(self, key)
    }
    fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        hashbrown::HashMap::remove_entry(self, key)
    }
    fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        hashbrown::HashMap::contains_key(self, key)
    }
    fn len(&self) -> usize {
        hashbrown::HashMap::len(self)
    }
    fn capacity(&self) -> usize {
        hashbrown::HashMap::capacity(self)
    }
    fn clear(&mut self) {
        hashbrown::HashMap::clear(self)
    }
    fn reserve(&mut self, additional: usize) {
        hashbrown::HashMap::reserve(self, additional)
    }
    fn shrink_to_fit(&mut self) {
        hashbrown::HashMap::shrink_to_fit(self)
    }
    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a,
    {
        hashbrown::HashMap::iter(self)
    }
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (&'a K, &'a mut V)>
    where
        K: 'a,
        V: 'a,
    {
        hashbrown::HashMap::iter_mut(self)
    }
    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        hashbrown::HashMap::retain(self, f)
    }
    fn drain(&mut self) -> impl Iterator<Item = (K, V)> {
        hashbrown::HashMap::drain(self)
    }
    fn into_keys(self) -> impl Iterator<Item = K> {
        hashbrown::HashMap::into_keys(self)
    }
    fn into_values(self) -> impl Iterator<Item = V> {
        hashbrown::HashMap::into_values(self)
    }
}

// ── std::HashMap implementation ─────────────────────────────────────────────

impl<K, V, S> HashedMap<K, V> for std::collections::HashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher + Default,
{
    fn new() -> Self {
        Self::with_hasher(S::default())
    }
    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, S::default())
    }
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        std::collections::HashMap::insert(self, key, value)
    }
    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        std::collections::HashMap::get(self, key)
    }
    fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        std::collections::HashMap::get_key_value(self, key)
    }
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        std::collections::HashMap::get_mut(self, key)
    }
    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        std::collections::HashMap::remove(self, key)
    }
    fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        std::collections::HashMap::remove_entry(self, key)
    }
    fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        std::collections::HashMap::contains_key(self, key)
    }
    fn len(&self) -> usize {
        std::collections::HashMap::len(self)
    }
    fn capacity(&self) -> usize {
        std::collections::HashMap::capacity(self)
    }
    fn clear(&mut self) {
        std::collections::HashMap::clear(self)
    }
    fn reserve(&mut self, additional: usize) {
        std::collections::HashMap::reserve(self, additional)
    }
    fn shrink_to_fit(&mut self) {
        std::collections::HashMap::shrink_to_fit(self)
    }
    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a,
    {
        std::collections::HashMap::iter(self)
    }
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (&'a K, &'a mut V)>
    where
        K: 'a,
        V: 'a,
    {
        std::collections::HashMap::iter_mut(self)
    }
    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        std::collections::HashMap::retain(self, f)
    }
    fn drain(&mut self) -> impl Iterator<Item = (K, V)> {
        std::collections::HashMap::drain(self)
    }
    fn into_keys(self) -> impl Iterator<Item = K> {
        std::collections::HashMap::into_keys(self)
    }
    fn into_values(self) -> impl Iterator<Item = V> {
        std::collections::HashMap::into_values(self)
    }
}

impl<K, V, S> Map<K, V> for std::collections::HashMap<K, V, S>
where
    K: Hash + Eq + Ord,
    S: BuildHasher + Default,
{
    fn new() -> Self {
        Self::with_hasher(S::default())
    }
    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, S::default())
    }
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        std::collections::HashMap::insert(self, key, value)
    }
    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        std::collections::HashMap::get(self, key)
    }
    fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        std::collections::HashMap::get_key_value(self, key)
    }
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        std::collections::HashMap::get_mut(self, key)
    }
    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        std::collections::HashMap::remove(self, key)
    }
    fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        std::collections::HashMap::remove_entry(self, key)
    }
    fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        std::collections::HashMap::contains_key(self, key)
    }
    fn len(&self) -> usize {
        std::collections::HashMap::len(self)
    }
    fn capacity(&self) -> usize {
        std::collections::HashMap::capacity(self)
    }
    fn clear(&mut self) {
        std::collections::HashMap::clear(self)
    }
    fn reserve(&mut self, additional: usize) {
        std::collections::HashMap::reserve(self, additional)
    }
    fn shrink_to_fit(&mut self) {
        std::collections::HashMap::shrink_to_fit(self)
    }
    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a,
    {
        std::collections::HashMap::iter(self)
    }
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (&'a K, &'a mut V)>
    where
        K: 'a,
        V: 'a,
    {
        std::collections::HashMap::iter_mut(self)
    }
    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        std::collections::HashMap::retain(self, f)
    }
    fn drain(&mut self) -> impl Iterator<Item = (K, V)> {
        std::collections::HashMap::drain(self)
    }
    fn into_keys(self) -> impl Iterator<Item = K> {
        std::collections::HashMap::into_keys(self)
    }
    fn into_values(self) -> impl Iterator<Item = V> {
        std::collections::HashMap::into_values(self)
    }
}

// ── Map facade impl for std::BTreeMap ───────────────────────────────────────

impl<K: Hash + Eq + Ord, V> Map<K, V> for std::collections::BTreeMap<K, V> {
    fn new() -> Self {
        std::collections::BTreeMap::new()
    }
    fn with_capacity(_capacity: usize) -> Self {
        std::collections::BTreeMap::new()
    }
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        std::collections::BTreeMap::insert(self, key, value)
    }
    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        std::collections::BTreeMap::get(self, key)
    }
    fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        std::collections::BTreeMap::get_key_value(self, key)
    }
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        std::collections::BTreeMap::get_mut(self, key)
    }
    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        std::collections::BTreeMap::remove(self, key)
    }
    fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        std::collections::BTreeMap::remove_entry(self, key)
    }
    fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        std::collections::BTreeMap::contains_key(self, key)
    }
    fn len(&self) -> usize {
        std::collections::BTreeMap::len(self)
    }
    fn capacity(&self) -> usize {
        self.len()
    }
    fn clear(&mut self) {
        std::collections::BTreeMap::clear(self)
    }
    fn reserve(&mut self, _additional: usize) {}
    fn shrink_to_fit(&mut self) {}
    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a,
    {
        std::collections::BTreeMap::iter(self)
    }
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (&'a K, &'a mut V)>
    where
        K: 'a,
        V: 'a,
    {
        std::collections::BTreeMap::iter_mut(self)
    }
    fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        std::collections::BTreeMap::retain(self, |k, v| f(k, v))
    }
    fn drain(&mut self) -> impl Iterator<Item = (K, V)> {
        std::mem::take(self).into_iter()
    }
    fn into_keys(self) -> impl Iterator<Item = K> {
        std::collections::BTreeMap::into_keys(self)
    }
    fn into_values(self) -> impl Iterator<Item = V> {
        std::collections::BTreeMap::into_values(self)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Set traits
// ════════════════════════════════════════════════════════════════════════════

/// Core hash set interface.
///
/// The hash function is an implementation detail — each concrete type
/// carries its own hasher internally. Generic code uses `Set<T>`
/// without knowing or caring about the hasher.
///
/// # Usage
///
/// For concrete types, use inherent methods directly (no import needed):
/// ```
/// let mut set = optimap::UnorderedFlatSet::new();
/// set.insert("hello");
/// assert!(set.contains("hello"));
/// ```
///
/// For generic code, import the trait:
/// ```
/// use optimap::Set;
/// fn has_duplicates<S: Set<i32>>(items: &[i32]) -> bool {
///     let mut seen = S::new();
///     items.iter().any(|&x| !seen.insert(x))
/// }
/// ```
pub trait Set<T: Hash + Eq> {
    /// Create an empty set with the default hasher.
    fn new() -> Self;

    /// Create a set with at least the specified capacity.
    fn with_capacity(capacity: usize) -> Self;

    /// Adds a value to the set. Returns `true` if newly inserted,
    /// `false` if already present.
    fn insert(&mut self, value: T) -> bool;

    /// Returns `true` if the set contains the given value.
    fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized;

    /// Returns a reference to the value in the set matching the given value.
    fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized;

    /// Removes a value from the set. Returns `true` if it was present.
    fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized;

    /// Removes and returns the value in the set matching the given value.
    fn take<Q>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized;

    /// Number of elements in the set.
    fn len(&self) -> usize;

    /// Whether the set is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of elements the set can hold without rehashing.
    fn capacity(&self) -> usize;

    /// Remove all elements, keeping allocated memory.
    fn clear(&mut self);

    /// Reserves capacity for at least `additional` more elements.
    fn reserve(&mut self, additional: usize);

    /// Shrinks the capacity as much as possible.
    fn shrink_to_fit(&mut self);

    /// Iterate over elements in arbitrary order.
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a;

    /// Retains only the elements specified by the predicate.
    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool;

    /// Clears the set, returning all elements as an iterator.
    fn drain(&mut self) -> impl Iterator<Item = T>;
}

/// Trait for sorted set implementations that support ordered operations.
pub trait SortedSet<T> {
    /// Returns a reference to the first (minimum) element.
    fn first(&self) -> Option<&T>;

    /// Returns a reference to the last (maximum) element.
    fn last(&self) -> Option<&T>;

    /// Removes and returns the first (minimum) element.
    fn pop_first(&mut self) -> Option<T>;

    /// Removes and returns the last (maximum) element.
    fn pop_last(&mut self) -> Option<T>;

    /// Iterate over all elements in sorted order.
    fn iter_sorted<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a;

    /// Iterate over elements within the given range, in sorted order.
    fn range<'a, Q, R>(&'a self, range: R) -> impl Iterator<Item = &'a T>
    where
        T: Borrow<Q> + 'a,
        Q: Ord + ?Sized,
        R: std::ops::RangeBounds<Q> + 'a;

    /// Splits the set at `at`, keeping elements `< at` in self and returning
    /// a new set containing elements `>= at`.
    fn split_off<Q>(&mut self, at: &Q) -> Self
    where
        Self: Sized,
        T: Borrow<Q>,
        Q: Ord + ?Sized;

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    fn append(&mut self, other: &mut Self)
    where
        Self: Sized;
}

// ── Macro to generate Set trait impl ────────────────────────────────────────

macro_rules! impl_set_trait {
    ($type:ident) => {
        impl<T, S> $crate::traits::Set<T> for $type<T, S>
        where
            T: ::std::hash::Hash + Eq,
            S: ::std::hash::BuildHasher + Default,
        {
            fn new() -> Self {
                Self::with_hasher(S::default())
            }
            fn with_capacity(capacity: usize) -> Self {
                Self::with_capacity_and_hasher(capacity, S::default())
            }
            fn insert(&mut self, value: T) -> bool {
                $type::insert(self, value)
            }
            fn contains<Q>(&self, value: &Q) -> bool
            where
                T: ::std::borrow::Borrow<Q>,
                Q: ::std::hash::Hash + Eq + ?Sized,
            {
                $type::contains(self, value)
            }
            fn get<Q>(&self, value: &Q) -> Option<&T>
            where
                T: ::std::borrow::Borrow<Q>,
                Q: ::std::hash::Hash + Eq + ?Sized,
            {
                $type::get(self, value)
            }
            fn remove<Q>(&mut self, value: &Q) -> bool
            where
                T: ::std::borrow::Borrow<Q>,
                Q: ::std::hash::Hash + Eq + ?Sized,
            {
                $type::remove(self, value)
            }
            fn take<Q>(&mut self, value: &Q) -> Option<T>
            where
                T: ::std::borrow::Borrow<Q>,
                Q: ::std::hash::Hash + Eq + ?Sized,
            {
                $type::take(self, value)
            }
            fn len(&self) -> usize {
                $type::len(self)
            }
            fn capacity(&self) -> usize {
                $type::capacity(self)
            }
            fn clear(&mut self) {
                $type::clear(self)
            }
            fn reserve(&mut self, additional: usize) {
                $type::reserve(self, additional)
            }
            fn shrink_to_fit(&mut self) {
                $type::shrink_to_fit(self)
            }
            fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
            where
                T: 'a,
            {
                $type::iter(self)
            }
            fn retain<F>(&mut self, f: F)
            where
                F: FnMut(&T) -> bool,
            {
                $type::retain(self, f)
            }
            fn drain(&mut self) -> impl Iterator<Item = T> {
                $type::drain(self)
            }
        }
    };
}

pub(crate) use impl_set_trait;

// ── Set impl for GenericSet<T, M> ───────────────────────────────────────────

impl<T, M> Set<T> for crate::generic_set::GenericSet<T, M>
where
    T: Hash + Eq,
    M: HashedMap<T, ()>,
{
    fn new() -> Self {
        crate::generic_set::GenericSet::new()
    }
    fn with_capacity(capacity: usize) -> Self {
        crate::generic_set::GenericSet::with_capacity(capacity)
    }
    fn insert(&mut self, value: T) -> bool {
        crate::generic_set::GenericSet::insert(self, value)
    }
    fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        crate::generic_set::GenericSet::contains(self, value)
    }
    fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        crate::generic_set::GenericSet::get(self, value)
    }
    fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        crate::generic_set::GenericSet::remove(self, value)
    }
    fn take<Q>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        crate::generic_set::GenericSet::take(self, value)
    }
    fn len(&self) -> usize {
        crate::generic_set::GenericSet::len(self)
    }
    fn capacity(&self) -> usize {
        crate::generic_set::GenericSet::capacity(self)
    }
    fn clear(&mut self) {
        crate::generic_set::GenericSet::clear(self)
    }
    fn reserve(&mut self, additional: usize) {
        crate::generic_set::GenericSet::reserve(self, additional)
    }
    fn shrink_to_fit(&mut self) {
        crate::generic_set::GenericSet::shrink_to_fit(self)
    }
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        crate::generic_set::GenericSet::iter(self)
    }
    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        crate::generic_set::GenericSet::retain(self, f)
    }
    fn drain(&mut self) -> impl Iterator<Item = T> {
        crate::generic_set::GenericSet::drain(self)
    }
}

// ── Set impl for hashbrown::HashSet ─────────────────────────────────────────

impl<T, S> Set<T> for hashbrown::HashSet<T, S>
where
    T: Hash + Eq,
    S: BuildHasher + Default,
{
    fn new() -> Self {
        Self::with_hasher(S::default())
    }
    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, S::default())
    }
    fn insert(&mut self, value: T) -> bool {
        hashbrown::HashSet::insert(self, value)
    }
    fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        hashbrown::HashSet::contains(self, value)
    }
    fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        hashbrown::HashSet::get(self, value)
    }
    fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        hashbrown::HashSet::remove(self, value)
    }
    fn take<Q>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        hashbrown::HashSet::take(self, value)
    }
    fn len(&self) -> usize {
        hashbrown::HashSet::len(self)
    }
    fn capacity(&self) -> usize {
        hashbrown::HashSet::capacity(self)
    }
    fn clear(&mut self) {
        hashbrown::HashSet::clear(self)
    }
    fn reserve(&mut self, additional: usize) {
        hashbrown::HashSet::reserve(self, additional)
    }
    fn shrink_to_fit(&mut self) {
        hashbrown::HashSet::shrink_to_fit(self)
    }
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        hashbrown::HashSet::iter(self)
    }
    fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        hashbrown::HashSet::retain(self, |v| f(v))
    }
    fn drain(&mut self) -> impl Iterator<Item = T> {
        hashbrown::HashSet::drain(self)
    }
}

// ── Set impl for std::HashSet ───────────────────────────────────────────────

impl<T, S> Set<T> for std::collections::HashSet<T, S>
where
    T: Hash + Eq,
    S: BuildHasher + Default,
{
    fn new() -> Self {
        Self::with_hasher(S::default())
    }
    fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, S::default())
    }
    fn insert(&mut self, value: T) -> bool {
        std::collections::HashSet::insert(self, value)
    }
    fn contains<Q>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        std::collections::HashSet::contains(self, value)
    }
    fn get<Q>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        std::collections::HashSet::get(self, value)
    }
    fn remove<Q>(&mut self, value: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        std::collections::HashSet::remove(self, value)
    }
    fn take<Q>(&mut self, value: &Q) -> Option<T>
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        std::collections::HashSet::take(self, value)
    }
    fn len(&self) -> usize {
        std::collections::HashSet::len(self)
    }
    fn capacity(&self) -> usize {
        std::collections::HashSet::capacity(self)
    }
    fn clear(&mut self) {
        std::collections::HashSet::clear(self)
    }
    fn reserve(&mut self, additional: usize) {
        std::collections::HashSet::reserve(self, additional)
    }
    fn shrink_to_fit(&mut self) {
        std::collections::HashSet::shrink_to_fit(self)
    }
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        std::collections::HashSet::iter(self)
    }
    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        std::collections::HashSet::retain(self, f)
    }
    fn drain(&mut self) -> impl Iterator<Item = T> {
        std::collections::HashSet::drain(self)
    }
}

// ── SortedSet impl for GenericSet (when backing map is SortedMap) ───────────

impl<T, M> SortedSet<T> for crate::generic_set::GenericSet<T, M>
where
    T: Hash + Eq + Ord,
    M: HashedMap<T, ()> + crate::SortedMap<T, ()>,
{
    fn first(&self) -> Option<&T> {
        crate::generic_set::GenericSet::first(self)
    }

    fn last(&self) -> Option<&T> {
        crate::generic_set::GenericSet::last(self)
    }

    fn pop_first(&mut self) -> Option<T> {
        crate::generic_set::GenericSet::pop_first(self)
    }

    fn pop_last(&mut self) -> Option<T> {
        crate::generic_set::GenericSet::pop_last(self)
    }

    fn iter_sorted<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        crate::generic_set::GenericSet::iter_sorted(self)
    }

    fn range<'a, Q, R>(&'a self, range: R) -> impl Iterator<Item = &'a T>
    where
        T: Borrow<Q> + 'a,
        Q: Ord + ?Sized,
        R: std::ops::RangeBounds<Q> + 'a,
    {
        crate::generic_set::GenericSet::range(self, range)
    }

    fn split_off<Q>(&mut self, at: &Q) -> Self
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        crate::generic_set::GenericSet::split_off(self, at)
    }

    fn append(&mut self, other: &mut Self) {
        crate::generic_set::GenericSet::append(self, other)
    }
}

// ── SortedSet impl for std::BTreeSet ────────────────────────────────────────

impl<T: Ord> SortedSet<T> for std::collections::BTreeSet<T> {
    fn first(&self) -> Option<&T> {
        self.iter().next()
    }

    fn last(&self) -> Option<&T> {
        self.iter().next_back()
    }

    fn pop_first(&mut self) -> Option<T> {
        std::collections::BTreeSet::pop_first(self)
    }

    fn pop_last(&mut self) -> Option<T> {
        std::collections::BTreeSet::pop_last(self)
    }

    fn iter_sorted<'a>(&'a self) -> impl Iterator<Item = &'a T>
    where
        T: 'a,
    {
        self.iter()
    }

    fn range<'a, Q, R>(&'a self, range: R) -> impl Iterator<Item = &'a T>
    where
        T: Borrow<Q> + 'a,
        Q: Ord + ?Sized,
        R: std::ops::RangeBounds<Q> + 'a,
    {
        std::collections::BTreeSet::range(self, range)
    }

    fn split_off<Q>(&mut self, at: &Q) -> Self
    where
        T: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        std::collections::BTreeSet::split_off(self, at)
    }

    fn append(&mut self, other: &mut Self) {
        std::collections::BTreeSet::append(self, other)
    }
}
