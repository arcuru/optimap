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

    /// Iterate over key-value pairs in arbitrary order.
    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a;
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
            fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
            where
                K: 'a,
                V: 'a,
            {
                $type::iter(self)
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
    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a,
    {
        hashbrown::HashMap::iter(self)
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
    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a,
    {
        std::collections::HashMap::iter(self)
    }
}
