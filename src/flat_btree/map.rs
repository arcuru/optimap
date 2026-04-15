//! Public FlatBTree API, trait implementations, and iterators.

use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::ops::RangeBounds;

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

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        use super::raw::EntrySearch;
        match self.tree.entry_search(&key) {
            EntrySearch::Occupied(leaf_idx, slot_idx) => {
                let node = self.tree.arena.node_ptr(leaf_idx);
                let value = unsafe { &mut *NodeLayout::<K, V>::leaf_val_ptr(node, slot_idx) };
                Entry::Occupied(OccupiedEntry { key, value })
            }
            EntrySearch::Vacant(leaf_idx, pos, path) => Entry::Vacant(VacantEntry {
                key,
                leaf_idx,
                pos,
                path,
                tree: &mut self.tree,
            }),
            EntrySearch::EmptyTree => Entry::Vacant(VacantEntry {
                key,
                leaf_idx: NO_NODE,
                pos: 0,
                path: Vec::new(),
                tree: &mut self.tree,
            }),
        }
    }
}

/// A view into a single entry in a FlatBTree, which may be vacant or occupied.
pub enum Entry<'a, K, V> {
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, K, V>),
    /// A vacant entry.
    Vacant(VacantEntry<'a, K, V>),
}

/// A view into an occupied entry in a FlatBTree.
pub struct OccupiedEntry<'a, K, V> {
    key: K,
    value: &'a mut V,
}

/// A view into a vacant entry in a FlatBTree.
pub struct VacantEntry<'a, K, V> {
    key: K,
    leaf_idx: NodeIdx,
    pos: usize,
    path: Vec<(NodeIdx, usize)>,
    tree: &'a mut RawBTree<K, V>,
}

impl<'a, K: Ord + Clone, V> Entry<'a, K, V> {
    /// Ensures a value is in the entry by inserting the default if empty,
    /// and returns a mutable reference to the value.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(default),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the function
    /// if empty, and returns a mutable reference to the value.
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(default()),
        }
    }

    /// Returns a reference to this entry's key.
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(e) => &e.key,
            Entry::Vacant(e) => &e.key,
        }
    }

    /// Provides in-place mutable access to an occupied entry.
    pub fn and_modify<F: FnOnce(&mut V)>(self, f: F) -> Self {
        match self {
            Entry::Occupied(mut e) => {
                f(e.get_mut());
                Entry::Occupied(e)
            }
            Entry::Vacant(e) => Entry::Vacant(e),
        }
    }
}

impl<'a, K: Ord + Clone, V: Default> Entry<'a, K, V> {
    /// Ensures a value is in the entry by inserting the default value if empty.
    pub fn or_default(self) -> &'a mut V {
        self.or_insert(V::default())
    }
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    /// Gets a reference to the value in the entry.
    pub fn get(&self) -> &V {
        self.value
    }

    /// Gets a mutable reference to the value in the entry.
    pub fn get_mut(&mut self) -> &mut V {
        self.value
    }

    /// Converts the entry into a mutable reference to the value,
    /// with a lifetime bound to the map.
    pub fn into_mut(self) -> &'a mut V {
        self.value
    }

    /// Sets the value of the entry, returning the old value.
    pub fn insert(&mut self, value: V) -> V {
        std::mem::replace(self.value, value)
    }

    /// Returns a reference to the entry's key.
    pub fn key(&self) -> &K {
        &self.key
    }
}

impl<'a, K: Ord + Clone, V> VacantEntry<'a, K, V> {
    /// Sets the value of the entry and returns a mutable reference to it.
    pub fn insert(self, value: V) -> &'a mut V {
        if self.leaf_idx == NO_NODE {
            // Empty tree
            self.tree.insert_first(self.key.clone(), value);
            let node = self.tree.arena.node_ptr(self.tree.first_leaf);
            unsafe { &mut *NodeLayout::<K, V>::leaf_val_ptr(node, 0) }
        } else {
            self.tree
                .insert_at_vacant(self.leaf_idx, self.pos, self.path, self.key.clone(), value);
            // Find the value we just inserted. After insert (possibly with split),
            // we need to search for it since the leaf may have split.
            // The key was just inserted, so search will find it.
            let (leaf_idx, slot_idx) = self.tree.search(&self.key).expect("just inserted");
            let node = self.tree.arena.node_ptr(leaf_idx);
            unsafe { &mut *NodeLayout::<K, V>::leaf_val_ptr(node, slot_idx) }
        }
    }

    /// Returns a reference to the entry's key.
    pub fn key(&self) -> &K {
        &self.key
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

    /// Pre-allocate arena space for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        let leaf_cap = NodeLayout::<K, V>::LEAF_CAP.max(1);
        let needed_leaves = additional.div_ceil(leaf_cap);
        let current = self.tree.arena.allocated_nodes();
        // Leaves + ~25% overhead for internal nodes
        let target = current + needed_leaves as u32 + needed_leaves as u32 / 4;
        self.tree.arena.ensure_capacity(target);
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
        let back_leaf = self.tree.last_leaf;
        let back_idx = if back_leaf != NO_NODE {
            let node = self.tree.arena.node_ptr(back_leaf);
            unsafe { NodeLayout::<K, V>::header(node).len as usize }
        } else {
            0
        };
        Iter {
            tree: &self.tree,
            front_leaf: self.tree.first_leaf,
            front_idx: 0,
            back_leaf,
            back_idx,
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

    /// Iterate over mutable values in key order.
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        let first = self.tree.first_leaf;
        let len = self.tree.len();
        ValuesMut {
            inner: IterMut {
                tree: &mut self.tree,
                current_leaf: first,
                current_idx: 0,
                remaining: len,
            },
        }
    }

    /// Iterate over mutable key-value pairs in sorted order.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        let first = self.tree.first_leaf;
        let len = self.tree.len();
        IterMut {
            tree: &mut self.tree,
            current_leaf: first,
            current_idx: 0,
            remaining: len,
        }
    }

    /// Iterate over key-value pairs within the given range, in sorted order.
    pub fn range<Q, R>(&self, range: R) -> RangeIter<'_, K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
        R: RangeBounds<Q>,
    {
        use std::ops::Bound;

        // Find start position
        let (start_leaf, start_idx) = match range.start_bound() {
            Bound::Included(key) => self.tree.lower_bound(key).unwrap_or((NO_NODE, 0)),
            Bound::Excluded(key) => {
                if let Some((leaf, idx)) = self.tree.lower_bound(key) {
                    let node = self.tree.arena.node_ptr(leaf);
                    let k = unsafe { &*NodeLayout::<K, V>::leaf_key_ptr(node, idx) };
                    if k.borrow() == key {
                        let header = unsafe { NodeLayout::<K, V>::header(node) };
                        if idx + 1 < header.len as usize {
                            (leaf, idx + 1)
                        } else {
                            let next = unsafe { NodeLayout::<K, V>::leaf_next_ptr(node).read() };
                            (next, 0)
                        }
                    } else {
                        (leaf, idx)
                    }
                } else {
                    (NO_NODE, 0)
                }
            }
            Bound::Unbounded => (self.tree.first_leaf, 0),
        };

        // Find end position: (leaf, idx) of the last element in range, exclusive
        let (end_leaf, end_idx) = match range.end_bound() {
            Bound::Included(key) => {
                // Find the position AFTER the last included key
                if let Some((leaf, idx)) = self.tree.lower_bound(key) {
                    let node = self.tree.arena.node_ptr(leaf);
                    let k = unsafe { &*NodeLayout::<K, V>::leaf_key_ptr(node, idx) };
                    if k.borrow() == key {
                        // Include this element: end is one past
                        let header = unsafe { NodeLayout::<K, V>::header(node) };
                        if idx + 1 < header.len as usize {
                            (leaf, idx + 1)
                        } else {
                            let next = unsafe { NodeLayout::<K, V>::leaf_next_ptr(node).read() };
                            (next, 0)
                        }
                    } else {
                        // key not in tree; lower_bound points to first key > key
                        // so end is this position
                        (leaf, idx)
                    }
                } else {
                    // All keys <= target, so include everything
                    (NO_NODE, 0)
                }
            }
            Bound::Excluded(key) => {
                // End at first key >= target
                self.tree.lower_bound(key).unwrap_or((NO_NODE, 0))
            }
            Bound::Unbounded => (NO_NODE, 0),
        };

        RangeIter {
            tree: &self.tree,
            current_leaf: start_leaf,
            current_idx: start_idx,
            end_leaf,
            end_idx,
        }
    }
}

impl<K, V, S> FlatBTree<K, V, S> {
    /// Retains only the elements specified by the predicate.
    /// Elements are visited in sorted key order.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        let mut leaf_idx = self.tree.first_leaf;
        while leaf_idx != NO_NODE {
            let node = self.tree.arena.node_ptr(leaf_idx);
            let next = unsafe { NodeLayout::<K, V>::leaf_next_ptr(node).read() };
            let header = unsafe { NodeLayout::<K, V>::header_mut(node) };
            let mut len = header.len as usize;
            let mut i = 0;

            while i < len {
                let k = unsafe { &*NodeLayout::<K, V>::leaf_key_ptr(node, i) };
                let v = unsafe { &mut *NodeLayout::<K, V>::leaf_val_ptr(node, i) };

                if f(k, v) {
                    i += 1;
                } else {
                    // Remove element at i: drop it and shift remaining left
                    unsafe {
                        std::ptr::drop_in_place(NodeLayout::<K, V>::leaf_key_ptr(node, i));
                        std::ptr::drop_in_place(NodeLayout::<K, V>::leaf_val_ptr(node, i));

                        for j in i..len - 1 {
                            let src_k = NodeLayout::<K, V>::leaf_key_ptr(node, j + 1);
                            let dst_k = NodeLayout::<K, V>::leaf_key_ptr(node, j);
                            std::ptr::copy_nonoverlapping(src_k, dst_k, 1);

                            let src_v = NodeLayout::<K, V>::leaf_val_ptr(node, j + 1);
                            let dst_v = NodeLayout::<K, V>::leaf_val_ptr(node, j);
                            std::ptr::copy_nonoverlapping(src_v, dst_v, 1);
                        }
                    }
                    len -= 1;
                    header.len = len as u16;
                    self.tree.len -= 1;
                    // Don't increment i — the next element shifted into position i
                }
            }

            leaf_idx = next;
        }
    }

    /// Creates a draining iterator that removes all elements from the map
    /// and yields them in sorted key order. The map is empty after this call.
    pub fn drain(&mut self) -> Drain<'_, K, V> {
        let first = self.tree.first_leaf;
        let len = self.tree.len();
        Drain {
            tree: &mut self.tree,
            current_leaf: first,
            current_idx: 0,
            remaining: len,
        }
    }
}

/// A draining iterator over `(K, V)` pairs in sorted order.
pub struct Drain<'a, K, V> {
    tree: &'a mut RawBTree<K, V>,
    current_leaf: NodeIdx,
    current_idx: usize,
    remaining: usize,
}

impl<K, V> Iterator for Drain<'_, K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_leaf == NO_NODE {
                return None;
            }

            let node = self.tree.arena.node_ptr(self.current_leaf);
            let header = unsafe { NodeLayout::<K, V>::header(node) };
            let len = header.len as usize;

            if self.current_idx < len {
                let k = unsafe { NodeLayout::<K, V>::leaf_key_ptr(node, self.current_idx).read() };
                let v = unsafe { NodeLayout::<K, V>::leaf_val_ptr(node, self.current_idx).read() };
                self.current_idx += 1;
                self.remaining -= 1;
                self.tree.len -= 1;
                return Some((k, v));
            }

            // Mark leaf as consumed
            unsafe { NodeLayout::<K, V>::header_mut(node).len = 0 };
            self.current_leaf = unsafe { NodeLayout::<K, V>::leaf_next_ptr(node).read() };
            self.current_idx = 0;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> Drop for Drain<'_, K, V> {
    fn drop(&mut self) {
        // Consume remaining elements
        while self.next().is_some() {}
    }
}

impl<K, V> ExactSizeIterator for Drain<'_, K, V> {}
impl<K, V> std::iter::FusedIterator for Drain<'_, K, V> {}

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

/// Iterator over `(&K, &V)` pairs in sorted order. Supports double-ended iteration.
pub struct Iter<'a, K, V> {
    tree: &'a RawBTree<K, V>,
    front_leaf: NodeIdx,
    front_idx: usize,
    back_leaf: NodeIdx,
    /// One past the last valid index in back_leaf (exclusive).
    back_idx: usize,
    remaining: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.remaining == 0 {
                return None;
            }

            let node = self.tree.arena.node_ptr(self.front_leaf);

            // Determine the effective end for the current front leaf
            let end = if self.front_leaf == self.back_leaf {
                self.back_idx
            } else {
                unsafe { NodeLayout::<K, V>::header(node).len as usize }
            };

            if self.front_idx < end {
                let k = unsafe { &*NodeLayout::<K, V>::leaf_key_ptr(node, self.front_idx) };
                let v = unsafe { &*NodeLayout::<K, V>::leaf_val_ptr(node, self.front_idx) };
                self.front_idx += 1;
                self.remaining -= 1;
                return Some((k, v));
            }

            // Move to next leaf
            self.front_leaf = unsafe { NodeLayout::<K, V>::leaf_next_ptr(node).read() };
            self.front_idx = 0;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> DoubleEndedIterator for Iter<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if self.remaining == 0 {
                return None;
            }

            // Determine effective start for back leaf
            let start = if self.back_leaf == self.front_leaf {
                self.front_idx
            } else {
                0
            };

            if self.back_idx > start {
                self.back_idx -= 1;
                let node = self.tree.arena.node_ptr(self.back_leaf);
                let k = unsafe { &*NodeLayout::<K, V>::leaf_key_ptr(node, self.back_idx) };
                let v = unsafe { &*NodeLayout::<K, V>::leaf_val_ptr(node, self.back_idx) };
                self.remaining -= 1;
                return Some((k, v));
            }

            // Move to previous leaf
            let node = self.tree.arena.node_ptr(self.back_leaf);
            self.back_leaf = unsafe { NodeLayout::<K, V>::leaf_prev_ptr(node).read() };
            if self.back_leaf != NO_NODE {
                let prev_node = self.tree.arena.node_ptr(self.back_leaf);
                self.back_idx = unsafe { NodeLayout::<K, V>::header(prev_node).len as usize };
            } else {
                // Exhausted
                self.remaining = 0;
                return None;
            }
        }
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

/// Mutable iterator over `(&K, &mut V)` pairs in sorted order.
pub struct IterMut<'a, K, V> {
    tree: &'a mut RawBTree<K, V>,
    current_leaf: NodeIdx,
    current_idx: usize,
    remaining: usize,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

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
                let v = unsafe { &mut *NodeLayout::<K, V>::leaf_val_ptr(node, self.current_idx) };
                self.current_idx += 1;
                self.remaining -= 1;
                return Some((k, v));
            }

            self.current_leaf = unsafe { NodeLayout::<K, V>::leaf_next_ptr(node).read() };
            self.current_idx = 0;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> ExactSizeIterator for IterMut<'_, K, V> {}
impl<K, V> std::iter::FusedIterator for IterMut<'_, K, V> {}

/// Iterator over mutable values in key order.
pub struct ValuesMut<'a, K, V> {
    inner: IterMut<'a, K, V>,
}

impl<'a, K, V> Iterator for ValuesMut<'a, K, V> {
    type Item = &'a mut V;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

/// Owning iterator over `(K, V)` pairs.
pub struct IntoIter<K, V> {
    tree: RawBTree<K, V>,
    current_leaf: NodeIdx,
    current_idx: usize,
    /// Index of first unconsumed element in current leaf.
    /// Elements [0..consumed_start) have been read out and must NOT be dropped again.
    consumed_start: usize,
    remaining: usize,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_leaf == NO_NODE {
                return None;
            }

            let node = self.tree.arena.node_ptr(self.current_leaf);
            let header = unsafe { NodeLayout::<K, V>::header(node) };
            let len = header.len as usize;

            if self.current_idx < len {
                let k = unsafe { NodeLayout::<K, V>::leaf_key_ptr(node, self.current_idx).read() };
                let v = unsafe { NodeLayout::<K, V>::leaf_val_ptr(node, self.current_idx).read() };
                self.current_idx += 1;
                self.consumed_start = self.current_idx;
                self.remaining -= 1;
                return Some((k, v));
            }

            // Move to next leaf
            let next = unsafe { NodeLayout::<K, V>::leaf_next_ptr(node).read() };
            // Mark current leaf as fully consumed (set len = 0 so Drop skips it)
            unsafe { NodeLayout::<K, V>::header_mut(node).len = 0 };
            self.current_leaf = next;
            self.current_idx = 0;
            self.consumed_start = 0;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<K, V> ExactSizeIterator for IntoIter<K, V> {}

impl<K, V> Drop for IntoIter<K, V> {
    fn drop(&mut self) {
        // Drop remaining unconsumed elements in the current leaf
        if self.current_leaf != NO_NODE {
            let node = self.tree.arena.node_ptr(self.current_leaf);
            let header = unsafe { NodeLayout::<K, V>::header(node) };
            let len = header.len as usize;

            // Drop elements [consumed_start..len) that we haven't read
            if std::mem::needs_drop::<K>() || std::mem::needs_drop::<V>() {
                for i in self.consumed_start..len {
                    unsafe {
                        std::ptr::drop_in_place(NodeLayout::<K, V>::leaf_key_ptr(node, i));
                        std::ptr::drop_in_place(NodeLayout::<K, V>::leaf_val_ptr(node, i));
                    }
                }
            }
            // Mark as consumed
            unsafe { NodeLayout::<K, V>::header_mut(node).len = 0 };

            // Drop all remaining leaves
            let mut leaf_idx = unsafe { NodeLayout::<K, V>::leaf_next_ptr(node).read() };
            while leaf_idx != NO_NODE {
                let node = self.tree.arena.node_ptr(leaf_idx);
                let header = unsafe { NodeLayout::<K, V>::header(node) };
                let nlen = header.len as usize;
                let next = unsafe { NodeLayout::<K, V>::leaf_next_ptr(node).read() };

                if std::mem::needs_drop::<K>() || std::mem::needs_drop::<V>() {
                    for i in 0..nlen {
                        unsafe {
                            std::ptr::drop_in_place(NodeLayout::<K, V>::leaf_key_ptr(node, i));
                            std::ptr::drop_in_place(NodeLayout::<K, V>::leaf_val_ptr(node, i));
                        }
                    }
                }
                unsafe { NodeLayout::<K, V>::header_mut(node).len = 0 };
                leaf_idx = next;
            }
        }

        // All leaf contents are now consumed/dropped. Clear tree state
        // so RawBTree::drop doesn't try to drop them again.
        self.tree.first_leaf = NO_NODE;
        self.tree.root = NO_NODE;
        // RawBTree::drop will still drop internal node keys and free the arena.
    }
}

/// Iterator over `(&K, &V)` pairs within a key range.
/// End is tracked as a (leaf, idx) position. NO_NODE means unbounded.
pub struct RangeIter<'a, K, V> {
    tree: &'a RawBTree<K, V>,
    current_leaf: NodeIdx,
    current_idx: usize,
    end_leaf: NodeIdx,
    end_idx: usize,
}

impl<'a, K, V> Iterator for RangeIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_leaf == NO_NODE {
                return None;
            }

            // Check if we've reached the end position
            if self.current_leaf == self.end_leaf && self.current_idx >= self.end_idx {
                self.current_leaf = NO_NODE;
                return None;
            }

            let node = self.tree.arena.node_ptr(self.current_leaf);
            let header = unsafe { NodeLayout::<K, V>::header(node) };
            let len = header.len as usize;

            if self.current_idx < len {
                let k = unsafe { &*NodeLayout::<K, V>::leaf_key_ptr(node, self.current_idx) };
                let v = unsafe { &*NodeLayout::<K, V>::leaf_val_ptr(node, self.current_idx) };
                self.current_idx += 1;
                return Some((k, v));
            }

            self.current_leaf = unsafe { NodeLayout::<K, V>::leaf_next_ptr(node).read() };
            self.current_idx = 0;
        }
    }
}

impl<K, V> std::iter::FusedIterator for RangeIter<'_, K, V> {}

// ── SortedMap trait impl ────────────────────────────────────────────────

impl<K: Ord, V, S> crate::SortedMap<K, V> for FlatBTree<K, V, S> {
    fn first_key_value(&self) -> Option<(&K, &V)> {
        FlatBTree::first_key_value(self)
    }

    fn last_key_value(&self) -> Option<(&K, &V)> {
        FlatBTree::last_key_value(self)
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
        R: RangeBounds<Q> + 'a,
    {
        FlatBTree::range(self, range)
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
        FlatBTree {
            tree: self.tree.clone_tree(),
            _hasher: PhantomData,
        }
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

impl<K: Ord, V, S, Q> std::ops::Index<&Q> for FlatBTree<K, V, S>
where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
{
    type Output = V;
    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

impl<K: Ord + Clone, V, S: Default> FromIterator<(K, V)> for FlatBTree<K, V, S> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut pairs: Vec<(K, V)> = iter.into_iter().collect();
        if pairs.is_empty() {
            return FlatBTree::with_hasher(S::default());
        }

        // Sort and deduplicate (keep last value for duplicate keys)
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        pairs.dedup_by(|b, a| {
            if a.0 == b.0 {
                // Keep the later value (b), move it to a's slot
                std::mem::swap(&mut a.1, &mut b.1);
                true
            } else {
                false
            }
        });

        FlatBTree {
            tree: RawBTree::bulk_load(pairs),
            _hasher: PhantomData,
        }
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

impl<K, V, S> IntoIterator for FlatBTree<K, V, S>
where
    K: Ord,
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;
    fn into_iter(self) -> Self::IntoIter {
        let first = self.tree.first_leaf;
        let len = self.tree.len();
        // We need to move the tree out without running FlatBTree's drop
        let tree = unsafe { std::ptr::read(&self.tree) };
        std::mem::forget(self);
        IntoIter {
            tree,
            current_leaf: first,
            current_idx: 0,
            consumed_start: 0,
            remaining: len,
        }
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
        for (i, &k) in keys.iter().enumerate() {
            assert_eq!(k, i, "iteration order wrong at {i}");
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

    #[test]
    fn range_query() {
        let mut map = FlatBTree::new();
        for i in 0..100 {
            map.insert(i, i * 10);
        }

        // Inclusive range
        let r: Vec<_> = map.range(10..=20).map(|(k, _)| *k).collect();
        assert_eq!(r, (10..=20).collect::<Vec<_>>());

        // Exclusive end
        let r: Vec<_> = map.range(90..95).map(|(k, _)| *k).collect();
        assert_eq!(r, vec![90, 91, 92, 93, 94]);

        // From start
        let r: Vec<_> = map.range(..3).map(|(k, _)| *k).collect();
        assert_eq!(r, vec![0, 1, 2]);

        // To end
        let r: Vec<_> = map.range(97..).map(|(k, _)| *k).collect();
        assert_eq!(r, vec![97, 98, 99]);

        // Full range
        let r: Vec<_> = map.range(..).map(|(k, _)| *k).collect();
        assert_eq!(r.len(), 100);

        // Empty range
        let r: Vec<_> = map.range(200..300).collect();
        assert!(r.is_empty());
    }

    #[test]
    fn range_with_string_keys() {
        let mut map = FlatBTree::new();
        map.insert("apple".to_string(), 1);
        map.insert("banana".to_string(), 2);
        map.insert("cherry".to_string(), 3);
        map.insert("date".to_string(), 4);
        map.insert("elderberry".to_string(), 5);

        let r: Vec<_> = map
            .range("banana".to_string()..="date".to_string())
            .map(|(k, _)| k.as_str())
            .collect();
        assert_eq!(r, vec!["banana", "cherry", "date"]);
    }

    #[test]
    fn iter_mut() {
        let mut map = FlatBTree::new();
        for i in 0..10 {
            map.insert(i, i);
        }

        for (_, v) in map.iter_mut() {
            *v *= 2;
        }

        for i in 0..10 {
            assert_eq!(map.get(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn into_iter() {
        let mut map = FlatBTree::new();
        for i in 0..50 {
            map.insert(i, i * 10);
        }

        let pairs: Vec<(i32, i32)> = map.into_iter().collect();
        assert_eq!(pairs.len(), 50);
        // Should be sorted
        for (i, (k, v)) in pairs.iter().enumerate() {
            assert_eq!(*k, i as i32);
            assert_eq!(*v, (i * 10) as i32);
        }
    }

    #[test]
    fn sorted_map_trait() {
        use crate::SortedMap;

        fn check<M: SortedMap<i32, i32>>(m: &M) {
            assert_eq!(m.first_key_value(), Some((&0, &0)));
            assert_eq!(m.last_key_value(), Some((&9, &90)));
            let r: Vec<_> = m.range(3..7).map(|(k, _)| *k).collect();
            assert_eq!(r, vec![3, 4, 5, 6]);
        }

        let mut map = FlatBTree::new();
        for i in 0..10 {
            map.insert(i, i * 10);
        }
        check(&map);
    }

    #[test]
    fn flat_btree_set() {
        let mut set = crate::FlatBTreeSet::new();
        set.insert(3);
        set.insert(1);
        set.insert(2);
        assert!(set.contains(&1));
        assert!(!set.contains(&4));
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn values_mut() {
        let mut map = FlatBTree::new();
        map.insert(1, "hello".to_string());
        map.insert(2, "world".to_string());

        for v in map.values_mut() {
            v.push('!');
        }

        assert_eq!(map.get(&1), Some(&"hello!".to_string()));
        assert_eq!(map.get(&2), Some(&"world!".to_string()));
    }

    #[test]
    fn into_iter_drops_correctly() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        struct Counted(i32);
        impl Drop for Counted {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);
        let mut map = FlatBTree::new();
        for i in 0..10 {
            map.insert(Counted(i), Counted(i));
        }

        // Partial iteration then drop
        let mut iter = map.into_iter();
        let _ = iter.next(); // consume 1
        drop(iter); // should drop remaining 9 + the 1 we consumed

        // 10 keys + 10 values = 20 Counted objects total
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 20);
    }

    #[test]
    fn entry_or_insert() {
        let mut map = FlatBTree::new();
        map.entry(1).or_insert("one");
        map.entry(1).or_insert("ONE"); // should not replace
        assert_eq!(map.get(&1), Some(&"one"));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn entry_or_default() {
        let mut map: FlatBTree<i32, i32> = FlatBTree::new();
        *map.entry(1).or_default() += 10;
        *map.entry(1).or_default() += 20;
        assert_eq!(map.get(&1), Some(&30));
    }

    #[test]
    fn entry_or_insert_with() {
        let mut map = FlatBTree::new();
        let val = map.entry(42).or_insert_with(|| "computed".to_string());
        assert_eq!(val, "computed");
        // Should not recompute
        let val = map
            .entry(42)
            .or_insert_with(|| panic!("should not be called"));
        assert_eq!(val, "computed");
    }

    #[test]
    fn entry_and_modify() {
        let mut map = FlatBTree::new();
        map.insert(1, 10);

        map.entry(1).and_modify(|v| *v += 5).or_insert(0);
        assert_eq!(map.get(&1), Some(&15));

        map.entry(2).and_modify(|v| *v += 5).or_insert(0);
        assert_eq!(map.get(&2), Some(&0));
    }

    #[test]
    fn entry_occupied_methods() {
        let mut map = FlatBTree::new();
        map.insert(1, "hello");

        match map.entry(1) {
            Entry::Occupied(mut e) => {
                assert_eq!(e.get(), &"hello");
                assert_eq!(e.key(), &1);
                let old = e.insert("world");
                assert_eq!(old, "hello");
                assert_eq!(e.get(), &"world");
            }
            Entry::Vacant(_) => panic!("expected occupied"),
        }

        assert_eq!(map.get(&1), Some(&"world"));
    }

    #[test]
    fn entry_vacant_key() {
        let mut map: FlatBTree<i32, i32> = FlatBTree::new();
        match map.entry(42) {
            Entry::Vacant(e) => {
                assert_eq!(e.key(), &42);
                e.insert(100);
            }
            Entry::Occupied(_) => panic!("expected vacant"),
        }
        assert_eq!(map.get(&42), Some(&100));
    }

    #[test]
    fn entry_counting_pattern() {
        let mut map = FlatBTree::new();
        let words = ["the", "cat", "sat", "on", "the", "mat", "the"];

        for word in words {
            *map.entry(word).or_insert(0) += 1;
        }

        assert_eq!(map.get("the"), Some(&3));
        assert_eq!(map.get("cat"), Some(&1));
        assert_eq!(map.get("on"), Some(&1));
        assert_eq!(map.len(), 5);
    }

    #[test]
    fn entry_with_splits() {
        // Force many splits via entry API
        let mut map = FlatBTree::new();
        for i in (0..500).rev() {
            *map.entry(i).or_insert(0) += 1;
        }
        // Insert same keys again
        for i in 0..500 {
            *map.entry(i).or_insert(0) += 1;
        }

        assert_eq!(map.len(), 500);
        for i in 0..500 {
            assert_eq!(map.get(&i), Some(&2), "wrong count for {i}");
        }
    }

    #[test]
    fn entry_empty_tree() {
        let mut map: FlatBTree<String, Vec<i32>> = FlatBTree::new();
        map.entry("hello".to_string()).or_default().push(1);
        map.entry("hello".to_string()).or_default().push(2);

        assert_eq!(map.get("hello"), Some(&vec![1, 2]));
    }

    #[test]
    fn double_ended_iter() {
        let mut map = FlatBTree::new();
        for i in 0..100 {
            map.insert(i, i);
        }

        // Forward
        let fwd: Vec<i32> = map.iter().map(|(&k, _)| k).collect();
        assert_eq!(fwd, (0..100).collect::<Vec<_>>());

        // Backward
        let bwd: Vec<i32> = map.iter().rev().map(|(&k, _)| k).collect();
        assert_eq!(bwd, (0..100).rev().collect::<Vec<_>>());

        // Mixed: front and back meeting in the middle
        let mut iter = map.iter();
        assert_eq!(iter.next().map(|(&k, _)| k), Some(0));
        assert_eq!(iter.next_back().map(|(&k, _)| k), Some(99));
        assert_eq!(iter.next().map(|(&k, _)| k), Some(1));
        assert_eq!(iter.next_back().map(|(&k, _)| k), Some(98));
        assert_eq!(iter.len(), 96);
    }

    #[test]
    fn double_ended_empty() {
        let map: FlatBTree<i32, i32> = FlatBTree::new();
        let mut iter = map.iter();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn double_ended_single() {
        let mut map = FlatBTree::new();
        map.insert(1, 10);
        let mut iter = map.iter();
        assert_eq!(iter.next_back(), Some((&1, &10)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn retain_basic() {
        let mut map = FlatBTree::new();
        for i in 0..100 {
            map.insert(i, i);
        }

        map.retain(|&k, _| k % 2 == 0);
        assert_eq!(map.len(), 50);
        for i in 0..100 {
            if i % 2 == 0 {
                assert_eq!(map.get(&i), Some(&i));
            } else {
                assert_eq!(map.get(&i), None);
            }
        }
    }

    #[test]
    fn retain_modify_values() {
        let mut map = FlatBTree::new();
        for i in 0..10 {
            map.insert(i, i);
        }

        map.retain(|_, v| {
            *v *= 2;
            true
        });

        for i in 0..10 {
            assert_eq!(map.get(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn drain_basic() {
        let mut map = FlatBTree::new();
        for i in 0..100 {
            map.insert(i, i * 10);
        }

        let drained: Vec<(i32, i32)> = map.drain().collect();
        assert_eq!(drained.len(), 100);
        assert!(map.is_empty());

        // Should be in sorted order
        for (i, (k, v)) in drained.iter().enumerate() {
            assert_eq!(*k, i as i32);
            assert_eq!(*v, (i * 10) as i32);
        }
    }

    #[test]
    fn drain_partial_drop() {
        let mut map = FlatBTree::new();
        for i in 0..50 {
            map.insert(i, i);
        }

        {
            let mut drain = map.drain();
            let _ = drain.next(); // consume 1
            // drop drain — should consume remaining
        }

        assert!(map.is_empty());
    }
}
