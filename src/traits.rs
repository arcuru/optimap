//! Common trait for all OptiMap hash map implementations.
//!
//! The `Map` trait defines the key-to-value mapping interface.
//! The hash function is an implementation detail of each concrete type,
//! not part of the trait.
//!
//! Users calling methods on concrete types (e.g. `Splitsies::insert`)
//! do NOT need to import this trait — inherent methods work automatically.
//! The trait is only needed for generic code over multiple implementations.

use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash};

/// Core hash map interface. Maps keys to values.
///
/// The hash function is an implementation detail — each concrete type
/// carries its own hasher internally. Generic code uses `Map<K, V>`
/// without knowing or caring about the hasher.
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
/// use optimap::Map;
/// fn count<M: Map<String, usize>>(m: &mut M, key: String) {
///     let val = m.get(&key).copied().unwrap_or(0);
///     m.insert(key, val + 1);
/// }
/// ```
pub trait Map<K: Hash + Eq, V> {
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
}

/// Trait for sorted map implementations that support ordered operations.
///
/// Unlike [`Map`], this does not require `Hash` — it works with any
/// key type that supports ordering.
pub trait SortedMap<K, V> {
    /// Returns the first (minimum) key-value pair.
    fn first_key_value(&self) -> Option<(&K, &V)>;

    /// Returns the last (maximum) key-value pair.
    fn last_key_value(&self) -> Option<(&K, &V)>;

    /// Removes and returns the first (minimum) key-value pair.
    fn pop_first(&mut self) -> Option<(K, V)>;

    /// Removes and returns the last (maximum) key-value pair.
    fn pop_last(&mut self) -> Option<(K, V)>;

    /// Iterate over all key-value pairs in sorted order.
    fn iter_sorted<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a;

    /// Iterate over key-value pairs within the given range, in sorted order.
    fn range<'a, Q, R>(&'a self, range: R) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: Borrow<Q> + 'a,
        V: 'a,
        Q: Ord + ?Sized,
        R: std::ops::RangeBounds<Q> + 'a;
}

// ── SortedMap impl for std::BTreeMap ────────────────────────────────────────

impl<K: Ord, V> SortedMap<K, V> for std::collections::BTreeMap<K, V> {
    fn first_key_value(&self) -> Option<(&K, &V)> {
        self.iter().next()
    }

    fn last_key_value(&self) -> Option<(&K, &V)> {
        self.iter().next_back()
    }

    fn pop_first(&mut self) -> Option<(K, V)> {
        std::collections::BTreeMap::pop_first(self)
    }

    fn pop_last(&mut self) -> Option<(K, V)> {
        std::collections::BTreeMap::pop_last(self)
    }

    fn iter_sorted<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a,
    {
        self.iter()
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
}

// ── Macro to generate trait impl that delegates to inherent methods ──────────

macro_rules! impl_map_trait {
    ($type:ident) => {
        impl<K, V, S> $crate::traits::Map<K, V> for $type<K, V, S>
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
        }
    };
}

pub(crate) use impl_map_trait;

// ── hashbrown implementation ─────────────────────────────────────────────────

impl<K, V, S> Map<K, V> for hashbrown::HashMap<K, V, S>
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
}

// ── std::HashMap implementation ─────────────────────────────────────────────

impl<K, V, S> Map<K, V> for std::collections::HashMap<K, V, S>
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
}
