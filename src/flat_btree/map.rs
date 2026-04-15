//! Public FlatBTree API, trait implementations, and iterators.

use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

use super::node::*;
use super::raw::RawBTree;

pub type DefaultHashBuilder = foldhash::fast::RandomState;

/// A cache-line-optimized B+ tree map.
///
/// Keys are stored in sorted order. Iteration yields elements in ascending
/// key order. Lookup, insert, and remove are O(log n).
///
/// The hasher `S` is carried for [`Map`](crate::Map) trait compatibility
/// but is never used — all operations use `K: Ord`.
///
/// ```
/// use optimap::FlatBTree;
///
/// let mut map = FlatBTree::new();
/// map.insert(3, "three");
/// map.insert(1, "one");
/// map.insert(2, "two");
///
/// // Iteration is sorted
/// let keys: Vec<_> = map.iter().map(|(k, _)| *k).collect();
/// assert_eq!(keys, vec![1, 2, 3]);
/// ```
pub struct FlatBTree<K, V, S = DefaultHashBuilder> {
    tree: RawBTree<K, V>,
    _hasher: PhantomData<S>,
}

// ── Constructors ────────────────────────────────────────────────────────

impl<K: Ord, V> FlatBTree<K, V> {
    /// Create an empty FlatBTree.
    pub fn new() -> Self {
        FlatBTree {
            tree: RawBTree::new(),
            _hasher: PhantomData,
        }
    }

    /// Create a FlatBTree with pre-allocated capacity for at least `capacity` elements.
    pub fn with_capacity(capacity: usize) -> Self {
        FlatBTree {
            tree: RawBTree::with_capacity(capacity),
            _hasher: PhantomData,
        }
    }
}

impl<K: Ord, V, S> FlatBTree<K, V, S> {
    /// Create an empty FlatBTree with a specific hasher (for Map trait compatibility).
    pub fn with_hasher(_hash_builder: S) -> Self {
        FlatBTree {
            tree: RawBTree::new(),
            _hasher: PhantomData,
        }
    }

    /// Create a FlatBTree with capacity and a specific hasher.
    pub fn with_capacity_and_hasher(capacity: usize, _hash_builder: S) -> Self {
        FlatBTree {
            tree: RawBTree::with_capacity(capacity),
            _hasher: PhantomData,
        }
    }
}

// ── Core operations (K: Ord, O(log n)) ──────────────────────────────────

impl<K: Ord + Clone, V, S> FlatBTree<K, V, S> {
    /// Insert a key-value pair. Returns the previous value if the key existed.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.tree.insert(key, value)
    }
}

impl<K: Ord, V, S> FlatBTree<K, V, S> {
    /// Look up a value by key.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.tree.get(key)
    }

    /// Look up a value by key, returning a mutable reference.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.tree.get_mut(key)
    }

    /// Remove a key, returning its value if present.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.tree.remove(key)
    }

    /// Whether the map contains the given key.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.tree.search(key).is_some()
    }

    /// Number of elements in the map.
    #[inline]
    pub fn len(&self) -> usize {
        self.tree.len()
    }

    /// Whether the map is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    /// Approximate capacity.
    pub fn capacity(&self) -> usize {
        self.tree.capacity()
    }

    /// Remove all elements.
    pub fn clear(&mut self) {
        self.tree.clear();
    }

    /// Returns the first (minimum) key-value pair.
    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        if self.tree.first_leaf == NO_NODE {
            return None;
        }
        let node = self.tree.arena.node_ptr(self.tree.first_leaf);
        let header = unsafe { NodeLayout::<K, V>::header(node) };
        if header.len == 0 {
            return None;
        }
        Some(unsafe {
            (
                &*NodeLayout::<K, V>::leaf_key_ptr(node, 0),
                &*NodeLayout::<K, V>::leaf_val_ptr(node, 0),
            )
        })
    }

    /// Returns the last (maximum) key-value pair.
    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        if self.tree.last_leaf == NO_NODE {
            return None;
        }
        let node = self.tree.arena.node_ptr(self.tree.last_leaf);
        let header = unsafe { NodeLayout::<K, V>::header(node) };
        if header.len == 0 {
            return None;
        }
        let last_idx = header.len as usize - 1;
        Some(unsafe {
            (
                &*NodeLayout::<K, V>::leaf_key_ptr(node, last_idx),
                &*NodeLayout::<K, V>::leaf_val_ptr(node, last_idx),
            )
        })
    }

    /// Iterate over key-value pairs in sorted order.
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            tree: &self.tree,
            current_leaf: self.tree.first_leaf,
            current_idx: 0,
            remaining: self.tree.len(),
        }
    }

    /// Iterate over keys in sorted order.
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { inner: self.iter() }
    }

    /// Iterate over values in key order.
    pub fn values(&self) -> Values<'_, K, V> {
        Values { inner: self.iter() }
    }
}

// ── Map trait impl (K: Hash + Eq + Ord) ─────────────────────────────────

impl<K, V, S> crate::Map<K, V> for FlatBTree<K, V, S>
where
    K: Hash + Eq + Ord + Clone,
    S: BuildHasher + Default,
{
    fn new() -> Self {
        FlatBTree::with_hasher(S::default())
    }

    fn with_capacity(capacity: usize) -> Self {
        FlatBTree::with_capacity_and_hasher(capacity, S::default())
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        FlatBTree::insert(self, key, value)
    }

    // O(n) fallback: the Map trait's Q bound is Hash + Eq, not Ord.
    // We can't do a tree search without Ord, so we scan the leaf chain.
    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.tree.get_by_eq(key)
    }

    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.tree.get_mut_by_eq(key)
    }

    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.tree.remove_by_eq(key)
    }

    fn len(&self) -> usize {
        self.tree.len()
    }

    fn capacity(&self) -> usize {
        self.tree.capacity()
    }

    fn clear(&mut self) {
        self.tree.clear();
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a,
    {
        FlatBTree::iter(self)
    }
}

// ── Iterators ───────────────────────────────────────────────────────────

/// Iterator over `(&K, &V)` pairs in sorted order.
pub struct Iter<'a, K, V> {
    tree: &'a RawBTree<K, V>,
    current_leaf: NodeIdx,
    current_idx: usize,
    remaining: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_leaf == NO_NODE {
                return None;
            }

            let node = self.tree.arena.node_ptr(self.current_leaf);
            let header = unsafe { NodeLayout::<K, V>::header(node) };
            let len = header.len as usize;

            if self.current_idx < len {
                let k = unsafe { &*NodeLayout::<K, V>::leaf_key_ptr(node, self.current_idx) };
                let v = unsafe { &*NodeLayout::<K, V>::leaf_val_ptr(node, self.current_idx) };
                self.current_idx += 1;
                self.remaining -= 1;
                return Some((k, v));
            }

            // Move to next leaf
            self.current_leaf = unsafe { NodeLayout::<K, V>::leaf_next_ptr(node).read() };
            self.current_idx = 0;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {}
impl<K, V> std::iter::FusedIterator for Iter<'_, K, V> {}

/// Iterator over keys in sorted order.
pub struct Keys<'a, K, V> {
    inner: Iter<'a, K, V>,
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

/// Iterator over values in key order.
pub struct Values<'a, K, V> {
    inner: Iter<'a, K, V>,
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

// ── Standard traits ─────────────────────────────────────────────────────

impl<K: Ord, V> Default for FlatBTree<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord + Clone, V: Clone, S: Default> Clone for FlatBTree<K, V, S> {
    fn clone(&self) -> Self {
        let mut new = FlatBTree::with_capacity_and_hasher(self.len(), S::default());
        for (k, v) in self.iter() {
            new.insert(k.clone(), v.clone());
        }
        new
    }
}

impl<K: Ord + fmt::Debug, V: fmt::Debug, S> fmt::Debug for FlatBTree<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K: Ord + PartialEq, V: PartialEq, S> PartialEq for FlatBTree<K, V, S> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<K: Ord + Eq, V: Eq, S> Eq for FlatBTree<K, V, S> {}

impl<K: Ord + Clone, V, S: Default> FromIterator<(K, V)> for FlatBTree<K, V, S> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut map = FlatBTree::with_capacity_and_hasher(lower, S::default());
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

impl<K: Ord + Clone, V, S> Extend<(K, V)> for FlatBTree<K, V, S> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<'a, K, V, S> IntoIterator for &'a FlatBTree<K, V, S>
where
    K: Ord,
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_map() {
        let map: FlatBTree<i32, i32> = FlatBTree::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert_eq!(map.get(&1), None);
        assert_eq!(map.first_key_value(), None);
        assert_eq!(map.last_key_value(), None);
    }

    #[test]
    fn insert_and_get() {
        let mut map = FlatBTree::new();
        assert_eq!(map.insert(1, "one"), None);
        assert_eq!(map.insert(2, "two"), None);
        assert_eq!(map.insert(3, "three"), None);

        assert_eq!(map.get(&1), Some(&"one"));
        assert_eq!(map.get(&2), Some(&"two"));
        assert_eq!(map.get(&3), Some(&"three"));
        assert_eq!(map.get(&4), None);
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn insert_replace() {
        let mut map = FlatBTree::new();
        assert_eq!(map.insert(1, "one"), None);
        assert_eq!(map.insert(1, "ONE"), Some("one"));
        assert_eq!(map.get(&1), Some(&"ONE"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn sorted_iteration() {
        let mut map = FlatBTree::new();
        for i in (0..100).rev() {
            map.insert(i, i * 10);
        }

        let keys: Vec<_> = map.keys().copied().collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
        assert_eq!(keys.len(), 100);
    }

    #[test]
    fn first_and_last() {
        let mut map = FlatBTree::new();
        map.insert(5, "five");
        map.insert(1, "one");
        map.insert(9, "nine");

        assert_eq!(map.first_key_value(), Some((&1, &"one")));
        assert_eq!(map.last_key_value(), Some((&9, &"nine")));
    }

    #[test]
    fn remove_basic() {
        let mut map = FlatBTree::new();
        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");

        assert_eq!(map.remove(&2), Some("two"));
        assert_eq!(map.get(&2), None);
        assert_eq!(map.len(), 2);
        assert_eq!(map.remove(&2), None);
    }

    #[test]
    fn clear_and_reuse() {
        let mut map = FlatBTree::new();
        for i in 0..50 {
            map.insert(i, i);
        }
        assert_eq!(map.len(), 50);
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.first_key_value(), None);

        // Reuse after clear
        map.insert(42, 42);
        assert_eq!(map.get(&42), Some(&42));
    }

    #[test]
    fn many_inserts_with_splits() {
        let mut map = FlatBTree::new();
        let n = 1000;

        // Insert in reverse order to force many splits
        for i in (0..n).rev() {
            map.insert(i, i * 10);
        }

        assert_eq!(map.len(), n);

        // Verify all elements present
        for i in 0..n {
            assert_eq!(map.get(&i), Some(&(i * 10)), "missing key {i}");
        }

        // Verify sorted iteration
        let keys: Vec<_> = map.keys().copied().collect();
        assert_eq!(keys.len(), n);
        for i in 0..n {
            assert_eq!(keys[i], i, "iteration order wrong at {i}");
        }
    }

    #[test]
    fn string_keys() {
        let mut map = FlatBTree::new();
        map.insert("banana".to_string(), 1);
        map.insert("apple".to_string(), 2);
        map.insert("cherry".to_string(), 3);

        assert_eq!(map.get("apple"), Some(&2));
        assert_eq!(map.get("banana"), Some(&1));
        assert_eq!(map.get("cherry"), Some(&3));

        let keys: Vec<_> = map.keys().collect();
        assert_eq!(keys, vec!["apple", "banana", "cherry"]);
    }

    #[test]
    fn map_trait_get() {
        use crate::Map;

        fn check<M: Map<i32, &'static str>>(m: &M) {
            assert_eq!(m.get(&1), Some(&"one"));
            assert_eq!(m.get(&3), None);
        }

        let mut map = FlatBTree::new();
        map.insert(1, "one");
        map.insert(2, "two");

        // Map trait get uses O(n) eq scan
        check(&map);
    }

    #[test]
    fn from_iterator() {
        let map: FlatBTree<i32, i32> = (0..100).map(|i| (i, i * 2)).collect();
        assert_eq!(map.len(), 100);
        assert_eq!(map.get(&50), Some(&100));
    }

    #[test]
    fn clone_map() {
        let mut map = FlatBTree::new();
        for i in 0..50 {
            map.insert(i, i);
        }
        let clone = map.clone();
        assert_eq!(map.len(), clone.len());
        for (a, b) in map.iter().zip(clone.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn stress_insert_remove() {
        use std::collections::BTreeMap;

        let mut ours = FlatBTree::new();
        let mut std_map = BTreeMap::new();

        // Insert 500 elements
        for i in (0..500).rev() {
            let key = (i * 7) % 300; // some duplicates
            ours.insert(key, key * 10);
            std_map.insert(key, key * 10);
        }

        assert_eq!(ours.len(), std_map.len());

        // Verify all elements match
        for (k, v) in std_map.iter() {
            assert_eq!(ours.get(k), Some(v), "mismatch for key {k}");
        }

        // Remove half
        for i in 0..150 {
            let key = (i * 7) % 300;
            let ours_val = ours.remove(&key);
            let std_val = std_map.remove(&key);
            assert_eq!(ours_val, std_val, "remove mismatch for key {key}");
        }

        assert_eq!(ours.len(), std_map.len());

        // Verify remaining
        for (k, v) in std_map.iter() {
            assert_eq!(ours.get(k), Some(v), "post-remove mismatch for key {k}");
        }
    }
}
