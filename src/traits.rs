//! Common trait for all OptiMap hash map implementations.
//!
//! The `Map` trait defines the core operations shared across all designs.
//! It uses `map_` prefixed method names to avoid conflicts with inherent
//! methods (which have richer signatures with `Borrow<Q>` bounds).
//!
//! Use this trait for:
//! - Writing generic code that works with any OptiMap implementation
//! - Benchmarking multiple designs with a single generic function
//! - Swapping implementations without changing calling code

use std::hash::Hash;

/// Core hash map operations shared by all OptiMap implementations.
///
/// Method names use the `map_` prefix to coexist with inherent methods
/// (which support `Borrow<Q>` for flexible key lookups). When calling
/// methods on a concrete type, prefer the inherent methods. Use this
/// trait when you need generic code over multiple map implementations.
pub trait Map<K: Hash + Eq, V> {
    /// Create an empty map.
    fn map_new() -> Self;

    /// Create a map with at least the specified capacity.
    fn map_with_capacity(capacity: usize) -> Self;

    /// Insert a key-value pair. Returns the previous value if the key existed.
    fn map_insert(&mut self, key: K, value: V) -> Option<V>;

    /// Look up a value by key.
    fn map_get(&self, key: &K) -> Option<&V>;

    /// Remove a key, returning its value if present.
    fn map_remove(&mut self, key: &K) -> Option<V>;

    /// Number of elements in the map.
    fn map_len(&self) -> usize;

    /// Whether the map is empty.
    fn map_is_empty(&self) -> bool {
        self.map_len() == 0
    }

    /// Number of elements the map can hold without rehashing.
    fn map_capacity(&self) -> usize;

    /// Remove all elements, keeping allocated memory.
    fn map_clear(&mut self);

    /// Whether the map contains the given key.
    fn map_contains_key(&self, key: &K) -> bool {
        self.map_get(key).is_some()
    }
}
