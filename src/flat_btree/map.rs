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

impl<K: Ord + Clone, V> FlatBTree<K, V> {
    /// Build a FlatBTree from input that is already sorted by key, with no
    /// duplicate keys.
    ///
    /// Faster than `from_iter` for already-sorted input because it skips
    /// the sort+dedup pass. Leaves are filled to `LEAF_CAP`, internal levels
    /// built bottom-up — the resulting tree is denser (fewer half-full
    /// leaves) and shallower than one built by repeated insertion.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if the input is not strictly ascending. In
    /// release builds, the tree may be invalid if the precondition is
    /// violated.
    pub fn from_sorted_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let pairs: Vec<(K, V)> = iter.into_iter().collect();
        debug_assert!(
            pairs.windows(2).all(|w| w[0].0 < w[1].0),
            "from_sorted_iter requires strictly ascending keys"
        );
        FlatBTree {
            tree: super::raw::RawBTree::bulk_load(pairs),
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

    /// Remove a key, returning its value if present.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.tree.remove(key)
    }

    /// Removes a key from the map, returning the key and value if it was present.
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.tree.remove_entry(key)
    }

    /// Removes and returns the first (minimum) key-value pair.
    pub fn pop_first(&mut self) -> Option<(K, V)> {
        if self.tree.first_leaf == NO_NODE {
            return None;
        }
        let node = self.tree.arena.node_ptr(self.tree.first_leaf);
        let header = unsafe { NodeLayout::<K, V>::header(node) };
        if header.len == 0 {
            return None;
        }
        let leaf_idx = self.tree.first_leaf;
        Some(self.tree.leaf_remove_at(leaf_idx, 0))
    }

    /// Removes and returns the last (maximum) key-value pair.
    pub fn pop_last(&mut self) -> Option<(K, V)> {
        if self.tree.last_leaf == NO_NODE {
            return None;
        }
        let node = self.tree.arena.node_ptr(self.tree.last_leaf);
        let header = unsafe { NodeLayout::<K, V>::header(node) };
        if header.len == 0 {
            return None;
        }
        let last_idx = header.len as usize - 1;
        let leaf_idx = self.tree.last_leaf;
        Some(self.tree.leaf_remove_at(leaf_idx, last_idx))
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        use super::raw::{EntrySearch, PathBuf};
        let mut path = PathBuf::new();
        match self.tree.entry_search(&key, &mut path) {
            EntrySearch::Occupied(leaf_idx, slot_idx) => {
                let node = self.tree.arena.node_ptr(leaf_idx);
                let value = unsafe { &mut *NodeLayout::<K, V>::leaf_val_ptr(node, slot_idx) };
                Entry::Occupied(OccupiedEntry { key, value })
            }
            EntrySearch::Vacant(leaf_idx, pos) => Entry::Vacant(VacantEntry {
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
                path,
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
    /// Stack-allocated descent path captured by `entry_search`. Consumed by
    /// `insert_at_vacant` if a split cascade fires; otherwise dropped cheaply.
    path: super::raw::PathBuf,
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

    /// Ensures a value is in the entry by inserting the result of the
    /// function (which receives the key) if empty.
    pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let value = default(&e.key);
                e.insert(value)
            }
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
    pub fn insert(mut self, value: V) -> &'a mut V {
        if self.leaf_idx == NO_NODE {
            // Empty tree
            self.tree.insert_first(self.key, value);
            let node = self.tree.arena.node_ptr(self.tree.first_leaf);
            unsafe { &mut *NodeLayout::<K, V>::leaf_val_ptr(node, 0) }
        } else {
            let (leaf_idx, slot_idx) = self.tree.insert_at_vacant(
                self.leaf_idx,
                self.pos,
                &mut self.path,
                self.key,
                value,
            );
            let node = self.tree.arena.node_ptr(leaf_idx);
            unsafe { &mut *NodeLayout::<K, V>::leaf_val_ptr(node, slot_idx) }
        }
    }

    /// Returns a reference to the entry's key.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Takes ownership of the key.
    pub fn into_key(self) -> K {
        self.key
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

    /// Returns the key-value pair corresponding to the key.
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (leaf_idx, slot_idx) = self.tree.search(key)?;
        let node = self.tree.arena.node_ptr(leaf_idx);
        Some(unsafe {
            (
                &*NodeLayout::<K, V>::leaf_key_ptr(node, slot_idx),
                &*NodeLayout::<K, V>::leaf_val_ptr(node, slot_idx),
            )
        })
    }

    /// Look up a value by key, returning a mutable reference.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.tree.get_mut(key)
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
        let (start_leaf, start_idx, end_leaf, end_idx) = self.tree.resolve_range_bounds(range);

        RangeIter {
            tree: &self.tree,
            current_leaf: start_leaf,
            current_idx: start_idx,
            end_leaf,
            end_idx,
        }
    }

    /// Iterate over key-value pairs within the given range, yielding mutable
    /// values, in sorted order.
    pub fn range_mut<Q, R>(&mut self, range: R) -> RangeIterMut<'_, K, V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
        R: RangeBounds<Q>,
    {
        let (start_leaf, start_idx, end_leaf, end_idx) =
            self.tree.resolve_range_bounds(range);
        RangeIterMut {
            tree: &mut self.tree as *mut RawBTree<K, V>,
            current_leaf: start_leaf,
            current_idx: start_idx,
            end_leaf,
            end_idx,
            _marker: PhantomData,
        }
    }
}

impl<K: Ord + Clone, V, S: Default> FlatBTree<K, V, S> {
    /// Shrinks the capacity as much as possible.
    ///
    /// Rebuilds the tree by draining all entries (in sorted order) into a
    /// fresh arena via `bulk_load`. This both releases unused arena nodes
    /// from the free list *and* compacts leaves to `LEAF_CAP` (vs the
    /// ~50% leaf utilization left behind by repeated split-on-insert).
    pub fn shrink_to_fit(&mut self) {
        if self.tree.is_empty() {
            // Empty: drop the arena entirely.
            self.tree = RawBTree::new();
            return;
        }
        // Move the old tree out (replacing with an empty one), drain it
        // in sorted order, bulk-load a fresh tree from the pairs.
        let old_tree = std::mem::replace(&mut self.tree, RawBTree::new());
        let first = old_tree.first_leaf;
        let len = old_tree.len();
        let into_iter = IntoIter {
            tree: old_tree,
            current_leaf: first,
            current_idx: 0,
            consumed_start: 0,
            remaining: len,
        };
        let pairs: Vec<(K, V)> = into_iter.collect();
        self.tree = RawBTree::bulk_load(pairs);
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

impl<K: Ord + Clone, V, S> FlatBTree<K, V, S> {
    /// Tries to insert a key-value pair, failing if the key already exists.
    pub fn try_insert(&mut self, key: K, value: V) -> Result<(), crate::traits::OccupiedError<K, V>> {
        match self.entry(key) {
            Entry::Occupied(e) => Err(crate::traits::OccupiedError { key: e.key, value }),
            Entry::Vacant(e) => { e.insert(value); Ok(()) }
        }
    }

    /// Moves all elements from `other` into `self`, leaving `other` empty.
    ///
    /// On key collision, the value from `other` wins (matches `std::BTreeMap::append`).
    ///
    /// Dispatches between three strategies based on a cheap O(1) disjointness
    /// check on `self.last_key` vs `other.first_key`:
    ///
    /// - **Disjoint adjacent** (`self.last < other.first`): tree-surgery graft
    ///   via [`append_graft`]. O(other_arena_nodes + log self.height) — does
    ///   NOT touch self's existing leaves at all. Falls back internally to
    ///   [`append_drain`] when `self.height < other.height` (rare).
    /// - **Reverse adjacent** (`other.last < self.first`): swap and graft
    ///   in the opposite direction.
    /// - **Overlapping**: drain both into sorted vectors, merge, bulk-load via
    ///   [`append_drain`]. O(n + m).
    ///
    /// [`append_graft`]: Self::append_graft
    /// [`append_drain`]: Self::append_drain
    pub fn append(&mut self, other: &mut Self) {
        if other.tree.is_empty() {
            return;
        }
        if self.tree.is_empty() {
            std::mem::swap(&mut self.tree, &mut other.tree);
            return;
        }

        // O(1) disjointness check via min/max keys.
        // SAFETY: both trees non-empty per the early-return above.
        let self_max = self.last_key_value().map(|(k, _)| k).unwrap();
        let other_min = other.first_key_value().map(|(k, _)| k).unwrap();
        if self_max < other_min {
            // Common case: adjacent disjoint, self's range first.
            self.append_graft(other);
            return;
        }
        let self_min = self.first_key_value().map(|(k, _)| k).unwrap();
        let other_max = other.last_key_value().map(|(k, _)| k).unwrap();
        if other_max < self_min {
            // Reverse adjacent: swap so self holds the smaller-key tree, then graft.
            std::mem::swap(&mut self.tree, &mut other.tree);
            self.append_graft(other);
            return;
        }

        // Overlapping ranges: fall back to drain+merge+bulk_load.
        self.append_drain(other);
    }

    /// Drain-and-rebuild append: drains both trees in sorted order, merges
    /// them with key-collision resolution (other wins on equal keys), and
    /// bulk-loads the result. O(n + m). Always correct regardless of whether
    /// the trees' key ranges overlap.
    #[doc(hidden)]
    pub fn append_drain(&mut self, other: &mut Self) {
        if other.tree.is_empty() {
            return;
        }
        if self.tree.is_empty() {
            std::mem::swap(&mut self.tree, &mut other.tree);
            return;
        }

        let a: Vec<(K, V)> = self.drain().collect();
        let b: Vec<(K, V)> = other.drain().collect();
        let mut a_iter = a.into_iter();
        let mut b_iter = b.into_iter();
        let mut out: Vec<(K, V)> = Vec::with_capacity(a_iter.len() + b_iter.len());
        let mut a_head = a_iter.next();
        let mut b_head = b_iter.next();
        loop {
            match (&a_head, &b_head) {
                (None, None) => break,
                (Some(_), None) => {
                    out.push(a_head.take().unwrap());
                    a_head = a_iter.next();
                }
                (None, Some(_)) => {
                    out.push(b_head.take().unwrap());
                    b_head = b_iter.next();
                }
                (Some(av), Some(bv)) => match av.0.cmp(&bv.0) {
                    std::cmp::Ordering::Less => {
                        out.push(a_head.take().unwrap());
                        a_head = a_iter.next();
                    }
                    std::cmp::Ordering::Greater => {
                        out.push(b_head.take().unwrap());
                        b_head = b_iter.next();
                    }
                    std::cmp::Ordering::Equal => {
                        // other wins on collision
                        a_head = a_iter.next();
                        out.push(b_head.take().unwrap());
                        b_head = b_iter.next();
                    }
                },
            }
        }
        self.tree = RawBTree::bulk_load(out);
    }

    /// Concat append: when `self.last_key < other.first_key`, the merge step
    /// in [`append_drain`] is unnecessary — drained sequences are already
    /// disjoint and ordered. This variant chains them and bulk-loads. O(n + m)
    /// but with lower constant factor than `append_drain`.
    ///
    /// **Precondition**: `self.last_key < other.first_key` (or one is empty).
    /// If violated, the result tree may have unsorted keys.
    ///
    /// [`append_drain`]: Self::append_drain
    #[doc(hidden)]
    pub fn append_concat(&mut self, other: &mut Self) {
        if other.tree.is_empty() {
            return;
        }
        if self.tree.is_empty() {
            std::mem::swap(&mut self.tree, &mut other.tree);
            return;
        }
        debug_assert!(
            self.last_key_value().map(|(k, _)| k) < other.first_key_value().map(|(k, _)| k),
            "append_concat requires self.last_key < other.first_key"
        );

        let mut a: Vec<(K, V)> = self.drain().collect();
        let b: Vec<(K, V)> = other.drain().collect();
        a.reserve(b.len());
        a.extend(b);
        self.tree = RawBTree::bulk_load(a);
    }

    /// Tail-extend append: drains `other` into a sorted vector and inserts each
    /// pair into `self`. When `self.last_key < other.first_key`, every insert
    /// hits the tail fast path → O(m). Compared to `append_drain` this avoids
    /// rebuilding `self` from scratch — leaves up to `self.last_leaf` keep
    /// their physical layout.
    ///
    /// Falls back to per-key insert (still correct) when ranges overlap, but
    /// is slower than `append_drain` in that case because it loses the
    /// bulk-load constant factor.
    #[doc(hidden)]
    pub fn append_extend(&mut self, other: &mut Self) {
        if other.tree.is_empty() {
            return;
        }
        if self.tree.is_empty() {
            std::mem::swap(&mut self.tree, &mut other.tree);
            return;
        }
        let pairs: Vec<(K, V)> = other.drain().collect();
        for (k, v) in pairs {
            self.tree.insert(k, v);
        }
    }

    /// Tree-surgery graft: for the disjoint adjacent case
    /// (`self.last_key < other.first_key`), avoid touching self's existing
    /// leaves entirely. Other's nodes are byte-copied into self's arena and
    /// the leaf chain is spliced; the spines are bridged with at most a
    /// single new internal node (equal heights) or by appending other's root
    /// as a new last child somewhere on self's right spine (self taller).
    ///
    /// Falls back to [`append_drain`] when `self.height < other.height` —
    /// supporting that direction is symmetric work but rare in practice
    /// (other being taller means it has more entries than self, which is
    /// unusual for an "append-into-self" call).
    ///
    /// **Precondition**: `self.last_key < other.first_key`.
    ///
    /// [`append_drain`]: Self::append_drain
    #[doc(hidden)]
    pub fn append_graft(&mut self, other: &mut Self) {
        if other.tree.is_empty() {
            return;
        }
        if self.tree.is_empty() {
            std::mem::swap(&mut self.tree, &mut other.tree);
            return;
        }
        debug_assert!(
            self.last_key_value().map(|(k, _)| k) < other.first_key_value().map(|(k, _)| k),
            "append_graft requires self.last_key < other.first_key"
        );
        if !self.tree.append_graft_disjoint(&mut other.tree) {
            // Height delta unsupported (self.height < other.height): fallback.
            self.append_drain(other);
        }
    }

    /// Splits the map into two at the given key. Returns everything with
    /// keys >= `at` as a new map; `self` keeps everything with keys < `at`.
    ///
    /// Picks the surgical direction that deep-copies the SMALLER side using
    /// a cheap O(log n) right-fraction estimate of the spine. Drain is no
    /// longer used: with both directions available, the side-of-min surgical
    /// is always at least as fast as drain (and 2-7× faster at asymmetric
    /// pivots on large trees).
    ///
    /// - `right_frac` ≤ ~0.5 → `surgical_right` (deep-copy the right side)
    /// - `right_frac` >  ~0.5 → `surgical_left`  (deep-copy the left side)
    ///
    /// Each surgical impl is O(log n + side_size).
    pub fn split_off<Q>(&mut self, at: &Q) -> Self
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        self.split_off_with_cutoff(at, Self::DEFAULT_RIGHT_FRAC_LEFT_CUTOFF)
    }

    /// Default dispatcher cutoff for `split_off`. See [`split_off_with_cutoff`]
    /// for the semantics and tuning rationale.
    ///
    /// [`split_off_with_cutoff`]: Self::split_off_with_cutoff
    pub const DEFAULT_RIGHT_FRAC_LEFT_CUTOFF: f64 = 0.70;

    /// Variant of [`split_off`] with a tunable estimator cutoff. Exposed
    /// `#[doc(hidden)]` for benchmarking the routing policy. The default
    /// cutoff used by [`split_off`] is [`DEFAULT_RIGHT_FRAC_LEFT_CUTOFF`].
    ///
    /// Two compounding biases shape the right cutoff:
    ///
    /// 1. **Estimator bias**: `estimate_right_fraction` weights every
    ///    subtree uniformly at `LEAF_CAP × (IC+1)^k`, but the rightmost
    ///    top-level subtree is usually partial. This adds ~+0.10 to the
    ///    estimated `r` at the symmetric pivot.
    ///
    /// 2. **Cost asymmetry between surgical_left and surgical_right**:
    ///    for the same amount of copied data, `surgical_left` is ~2–3×
    ///    slower than `surgical_right` on random-build 1M trees. Causes:
    ///    `surgical_left` does a per-spine-level array compaction shift
    ///    that `surgical_right` skips, an extra `mem::swap` of the two
    ///    RawBTrees at the end, and a "temp chain prepend" pass to keep
    ///    leaf chain order correct. The empirical break-even sits at
    ///    `actual r ≈ 0.65` (left side ≤ ~35%): below that, copy-right
    ///    wins despite copying more entries; above, copy-left wins.
    ///
    /// Combining: the biased-r boundary corresponding to actual r=0.65
    /// is ~0.75. Cutoff 0.70 leaves some margin for noise — at p030
    /// (actual r=0.70, biased ~0.80) we still pick LEFT correctly, and
    /// at p045 (actual r=0.55, biased ~0.65) we pick RIGHT correctly.
    /// At cutoff 0.60 (the prior default), p045 mis-routes to LEFT and
    /// runs 2.2× slower (12.3ms → 5.7ms at 1M).
    ///
    /// - `right_frac` > cutoff → `surgical_left`  (deep-copy the left side)
    /// - `right_frac` ≤ cutoff → `surgical_right` (deep-copy the right side)
    ///
    /// [`split_off`]: Self::split_off
    /// [`DEFAULT_RIGHT_FRAC_LEFT_CUTOFF`]: Self::DEFAULT_RIGHT_FRAC_LEFT_CUTOFF
    #[doc(hidden)]
    pub fn split_off_with_cutoff<Q>(&mut self, at: &Q, cutoff: f64) -> Self
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        match self.tree.estimate_right_fraction(at) {
            Some(rf) if rf > cutoff => self.split_off_surgical_left(at),
            _ => self.split_off_surgical_right(at),
        }
    }

    /// Drain-and-rebuild split: O(n) regardless of split position. Both halves
    /// rebuilt via `bulk_load`. Compact arenas, simple invariants.
    #[doc(hidden)]
    pub fn split_off_drain<Q>(&mut self, at: &Q) -> Self
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if self.tree.is_empty() {
            return FlatBTree {
                tree: RawBTree::new(),
                _hasher: PhantomData,
            };
        }

        // Trivial case 1: every key is < at → right is empty, self unchanged.
        match self.tree.lower_bound(at) {
            None => {
                return FlatBTree {
                    tree: RawBTree::new(),
                    _hasher: PhantomData,
                };
            }
            // Trivial case 2: at falls before any element → entire tree moves right.
            Some((leaf_idx, 0)) if leaf_idx == self.tree.first_leaf => {
                let stolen = std::mem::replace(&mut self.tree, RawBTree::new());
                return FlatBTree {
                    tree: stolen,
                    _hasher: PhantomData,
                };
            }
            _ => {}
        }

        let pairs: Vec<(K, V)> = self.drain().collect();
        // drain() yields in sorted order, so partition_point finds the boundary.
        let split = pairs.partition_point(|(k, _)| k.borrow() < at);
        let mut left = pairs;
        let right = left.split_off(split);

        if !left.is_empty() {
            self.tree = RawBTree::bulk_load(left);
        }

        let right_tree = if right.is_empty() {
            RawBTree::new()
        } else {
            RawBTree::bulk_load(right)
        };

        FlatBTree {
            tree: right_tree,
            _hasher: PhantomData,
        }
    }

    /// Tree-surgery split, deep-copying the RIGHT side. O(log n + right_nodes).
    /// Mutates `self` in place to retain `< at`; the deep-copied right subtree
    /// is returned. Wins when the right side is the smaller of the two.
    #[doc(hidden)]
    pub fn split_off_surgical_right<Q>(&mut self, at: &Q) -> Self
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let right_tree = self.tree.split_off_surgical_right(at);
        FlatBTree {
            tree: right_tree,
            _hasher: PhantomData,
        }
    }

    /// Tree-surgery split, deep-copying the LEFT side. O(log n + left_nodes).
    /// The kept side stays in self.arena (mutated in place), the LEFT side is
    /// deep-copied to a fresh arena, then a `mem::swap` flips them so `self`
    /// holds `< at` and the returned tree holds `>= at`. Wins when the left
    /// side is the smaller of the two.
    #[doc(hidden)]
    pub fn split_off_surgical_left<Q>(&mut self, at: &Q) -> Self
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let right_tree = self.tree.split_off_surgical_left(at);
        FlatBTree {
            tree: right_tree,
            _hasher: PhantomData,
        }
    }
}

impl<K: Ord, V, S> FlatBTree<K, V, S> {
    /// Creates a consuming iterator over the keys.
    pub fn into_keys(self) -> impl Iterator<Item = K> {
        self.into_iter().map(|(k, _)| k)
    }

    /// Creates a consuming iterator over the values.
    pub fn into_values(self) -> impl Iterator<Item = V> {
        self.into_iter().map(|(_, v)| v)
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

    fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.tree.get_key_value_by_eq(key)
    }

    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.tree.remove_by_eq(key)
    }

    fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.tree.remove_entry_by_eq(key)
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

    fn reserve(&mut self, additional: usize) {
        FlatBTree::reserve(self, additional)
    }

    fn shrink_to_fit(&mut self) {
        FlatBTree::shrink_to_fit(self)
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a,
    {
        FlatBTree::iter(self)
    }

    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (&'a K, &'a mut V)>
    where
        K: 'a,
        V: 'a,
    {
        FlatBTree::iter_mut(self)
    }

    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        FlatBTree::retain(self, f)
    }

    fn drain(&mut self) -> impl Iterator<Item = (K, V)> {
        FlatBTree::drain(self)
    }

    fn try_insert(
        &mut self,
        key: K,
        value: V,
    ) -> Result<(), crate::traits::OccupiedError<K, V>> {
        FlatBTree::try_insert(self, key, value)
    }

    fn into_keys(self) -> impl Iterator<Item = K> {
        FlatBTree::into_keys(self)
    }

    fn into_values(self) -> impl Iterator<Item = V> {
        FlatBTree::into_values(self)
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

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let node = self.tree.arena.node_ptr(self.front_leaf);
        let len = unsafe { NodeLayout::<K, V>::header(node).len as usize };

        if self.front_idx < len {
            let k = unsafe { &*NodeLayout::<K, V>::leaf_key_ptr(node, self.front_idx) };
            let v = unsafe { &*NodeLayout::<K, V>::leaf_val_ptr(node, self.front_idx) };
            self.front_idx += 1;
            self.remaining -= 1;
            return Some((k, v));
        }

        // Move to next leaf (cold path)
        self.advance_front()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, K, V> Iter<'a, K, V> {
    #[cold]
    #[inline(never)]
    fn advance_front(&mut self) -> Option<(&'a K, &'a V)> {
        let node = self.tree.arena.node_ptr(self.front_leaf);
        self.front_leaf = unsafe { NodeLayout::<K, V>::leaf_next_ptr(node).read() };
        self.front_idx = 0;
        // Recurse into next() — the new leaf has elements (remaining > 0 guarantees this)
        self.next()
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

/// Iterator over `(&K, &mut V)` pairs within a key range.
///
/// `tree` is held as a raw pointer so we can hand out independent `&mut V`
/// borrows for each yielded item without aliasing across yields. End is
/// tracked the same way as [`RangeIter`].
pub struct RangeIterMut<'a, K, V> {
    tree: *mut RawBTree<K, V>,
    current_leaf: NodeIdx,
    current_idx: usize,
    end_leaf: NodeIdx,
    end_idx: usize,
    _marker: PhantomData<&'a mut RawBTree<K, V>>,
}

// SAFETY: behaves like &mut RawBTree wrt thread safety.
unsafe impl<K: Send, V: Send> Send for RangeIterMut<'_, K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for RangeIterMut<'_, K, V> {}

impl<'a, K, V> Iterator for RangeIterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_leaf == NO_NODE {
                return None;
            }

            if self.current_leaf == self.end_leaf && self.current_idx >= self.end_idx {
                self.current_leaf = NO_NODE;
                return None;
            }

            // SAFETY: tree pointer is valid for 'a; each yield borrows a distinct slot.
            let arena_ptr = unsafe { (*self.tree).arena.node_ptr(self.current_leaf) };
            let header = unsafe { NodeLayout::<K, V>::header(arena_ptr) };
            let len = header.len as usize;

            if self.current_idx < len {
                let k = unsafe {
                    &*NodeLayout::<K, V>::leaf_key_ptr(arena_ptr, self.current_idx)
                };
                let v = unsafe {
                    &mut *NodeLayout::<K, V>::leaf_val_ptr(arena_ptr, self.current_idx)
                };
                self.current_idx += 1;
                return Some((k, v));
            }

            self.current_leaf = unsafe {
                NodeLayout::<K, V>::leaf_next_ptr(arena_ptr).read()
            };
            self.current_idx = 0;
        }
    }
}

impl<K, V> std::iter::FusedIterator for RangeIterMut<'_, K, V> {}

// ── SortedMap trait impl ────────────────────────────────────────────────

impl<K: Ord + Clone, V, S> crate::SortedMap<K, V> for FlatBTree<K, V, S> {
    fn first_key_value(&self) -> Option<(&K, &V)> {
        FlatBTree::first_key_value(self)
    }

    fn last_key_value(&self) -> Option<(&K, &V)> {
        FlatBTree::last_key_value(self)
    }

    fn pop_first(&mut self) -> Option<(K, V)> {
        FlatBTree::pop_first(self)
    }

    fn pop_last(&mut self) -> Option<(K, V)> {
        FlatBTree::pop_last(self)
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

    fn split_off<Q>(&mut self, at: &Q) -> Self
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        FlatBTree::split_off(self, at)
    }

    fn append(&mut self, other: &mut Self) {
        FlatBTree::append(self, other);
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

impl<K: Ord + Hash, V: Hash, S> Hash for FlatBTree<K, V, S> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Match std::BTreeMap: hash length followed by each (k, v) in sorted order.
        state.write_usize(self.len());
        for kv in self.iter() {
            kv.hash(state);
        }
    }
}

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

impl<'a, K, V, S> IntoIterator for &'a mut FlatBTree<K, V, S>
where
    K: Ord,
{
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
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
    fn shrink_to_fit_rebuilds() {
        // Insert lots, remove half, shrink. After shrink: capacity should
        // shrink, and lookups should still work.
        let mut map: FlatBTree<i32, i32> =
            (0i32..5_000).map(|i| (i, i * 3)).collect();
        for i in 0i32..2_500 {
            assert_eq!(map.remove(&i), Some(i * 3));
        }
        let cap_before = map.capacity();
        map.shrink_to_fit();
        let cap_after = map.capacity();
        assert!(
            cap_after <= cap_before,
            "shrink should not grow ({cap_after} > {cap_before})"
        );
        assert_eq!(map.len(), 2_500);
        for i in 2_500i32..5_000 {
            assert_eq!(map.get(&i), Some(&(i * 3)), "missing key {i} after shrink");
        }
        for i in 0i32..2_500 {
            assert_eq!(map.get(&i), None, "stale key {i} after shrink");
        }
    }

    #[test]
    fn shrink_to_fit_empty() {
        let mut map: FlatBTree<i32, i32> = (0..100).map(|i| (i, i)).collect();
        for i in 0..100 {
            map.remove(&i);
        }
        map.shrink_to_fit();
        assert_eq!(map.len(), 0);
        assert_eq!(map.capacity(), 0);
    }

    #[test]
    fn from_iterator_multi_level() {
        // 100K entries forces a 3-level tree, exercising bulk_load's
        // recursive separator-key promotion (which previously read the
        // wrong key for height ≥ 2 internal nodes).
        let map: FlatBTree<i32, i32> =
            (0i32..100_000).map(|i| (i, i.wrapping_mul(13))).collect();
        assert_eq!(map.len(), 100_000);
        for i in 0i32..100_000 {
            assert_eq!(map.get(&i), Some(&i.wrapping_mul(13)), "missing key {i}");
        }
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
    fn range_mut_basic() {
        let mut map = FlatBTree::new();
        for i in 0..200 {
            map.insert(i, i);
        }
        for (_, v) in map.range_mut(50..150) {
            *v += 1000;
        }
        for i in 0..200 {
            let expected = if (50..150).contains(&i) { i + 1000 } else { i };
            assert_eq!(map.get(&i), Some(&expected), "key {i}");
        }
    }

    #[test]
    fn range_mut_inclusive_excluded() {
        let mut map: FlatBTree<i32, i32> = (0..20).map(|i| (i, i)).collect();
        // (5..=10) → 5, 6, 7, 8, 9, 10
        let keys: Vec<i32> = map.range_mut(5..=10).map(|(&k, _)| k).collect();
        assert_eq!(keys, vec![5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn from_sorted_iter_basic() {
        let map = FlatBTree::from_sorted_iter((0..1000).map(|i| (i, i * 10)));
        assert_eq!(map.len(), 1000);
        for i in 0..1000 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
        }
        // Sorted iteration should match original order
        let pairs: Vec<_> = map.iter().map(|(&k, _)| k).collect();
        assert_eq!(pairs, (0..1000).collect::<Vec<_>>());
    }

    #[test]
    fn from_sorted_iter_multi_level() {
        // 100K elements: with LEAF_CAP=15 (u64/u64) we get ~6700 leaves,
        // INTERNAL_CAP=20 gives ~320 level-1 internals, ~16 level-2, 1 root.
        // Forces a 3-level tree which exercises the recursive separator fix.
        let n = 100_000u64;
        let map = FlatBTree::from_sorted_iter((0..n).map(|i| (i, i.wrapping_mul(7))));
        assert_eq!(map.len() as u64, n);
        for i in 0..n {
            assert_eq!(map.get(&i), Some(&i.wrapping_mul(7)), "missing key {i}");
        }
        for i in n..(n + 100) {
            assert_eq!(map.get(&i), None, "spurious key {i}");
        }
    }

    #[test]
    fn from_sorted_iter_empty() {
        let map: FlatBTree<i32, i32> = FlatBTree::from_sorted_iter(std::iter::empty());
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
    }

    #[test]
    fn from_sorted_iter_single_leaf() {
        // Fewer than LEAF_CAP elements (15 for u64/u64) → single root leaf.
        let map = FlatBTree::from_sorted_iter((0..5).map(|i| (i, i)));
        assert_eq!(map.len(), 5);
        for i in 0..5 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }

    #[test]
    #[should_panic(expected = "strictly ascending")]
    fn from_sorted_iter_unsorted_panics_in_debug() {
        // Only debug builds panic. The test runs under cfg(test) which is debug.
        let _: FlatBTree<i32, i32> =
            FlatBTree::from_sorted_iter([(2, 0), (1, 0), (3, 0)]);
    }

    #[test]
    fn range_mut_unbounded() {
        let mut map: FlatBTree<i32, i32> = (0..30).map(|i| (i, i)).collect();
        let mut count = 0;
        for (_, v) in map.range_mut(..) {
            *v += 100;
            count += 1;
        }
        assert_eq!(count, 30);
        assert_eq!(map.get(&0), Some(&100));
        assert_eq!(map.get(&29), Some(&129));
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
    fn into_iter_mut_yields_mutable_refs() {
        let mut map: FlatBTree<i32, i32> = (0..30).map(|i| (i, i)).collect();
        for (_k, v) in &mut map {
            *v *= 10;
        }
        for i in 0..30 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
        }
    }

    #[test]
    fn hash_matches_equal_maps() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let a: FlatBTree<i32, i32> = (0..50).map(|i| (i, i * 2)).collect();
        let b: FlatBTree<i32, i32> = (0..50).rev().map(|i| (i, i * 2)).collect();
        let c: FlatBTree<i32, i32> = (0..49).map(|i| (i, i * 2)).collect();

        let h = |m: &FlatBTree<i32, i32>| {
            let mut s = DefaultHasher::new();
            m.hash(&mut s);
            s.finish()
        };

        assert_eq!(a, b);
        assert_eq!(h(&a), h(&b));
        assert_ne!(h(&a), h(&c));
    }

    #[test]
    fn split_off_basic() {
        let mut map: FlatBTree<i32, i32> = (0..200).map(|i| (i, i * 10)).collect();
        let right = map.split_off(&100);

        assert_eq!(map.len(), 100);
        assert_eq!(right.len(), 100);

        for i in 0..100 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
            assert_eq!(right.get(&i), None);
        }
        for i in 100..200 {
            assert_eq!(map.get(&i), None);
            assert_eq!(right.get(&i), Some(&(i * 10)));
        }
    }

    #[test]
    fn split_off_at_missing_key() {
        // Splitting at a key not in the map: everything strictly less stays in self.
        let mut map: FlatBTree<i32, i32> = [10, 20, 30, 40].iter().map(|&i| (i, i)).collect();
        let right = map.split_off(&25);
        assert_eq!(map.iter().map(|(k, _)| *k).collect::<Vec<_>>(), vec![10, 20]);
        assert_eq!(right.iter().map(|(k, _)| *k).collect::<Vec<_>>(), vec![30, 40]);
    }

    #[test]
    fn split_off_edges() {
        // Split before everything → self empty, right gets all.
        let mut a: FlatBTree<i32, i32> = (1..=10).map(|i| (i, i)).collect();
        let r = a.split_off(&0);
        assert!(a.is_empty());
        assert_eq!(r.len(), 10);

        // Split after everything → self keeps all, right empty.
        let mut b: FlatBTree<i32, i32> = (1..=10).map(|i| (i, i)).collect();
        let r = b.split_off(&100);
        assert_eq!(b.len(), 10);
        assert!(r.is_empty());

        // Empty self → empty right.
        let mut c: FlatBTree<i32, i32> = FlatBTree::new();
        let r = c.split_off(&5);
        assert!(c.is_empty());
        assert!(r.is_empty());
    }

    #[test]
    fn split_off_large_then_mutate() {
        // After split, both halves must support insert/remove correctly.
        let mut map: FlatBTree<u64, u64> = (0..5_000).map(|i| (i, i)).collect();
        let mut right = map.split_off(&2_500);
        assert_eq!(map.len(), 2500);
        assert_eq!(right.len(), 2500);

        // Mutate both halves and re-check.
        for i in 0..2500 {
            map.insert(i, i + 1);
            right.insert(i + 2500, i + 2501);
        }
        for i in 0..2500 {
            assert_eq!(map.get(&i), Some(&(i + 1)));
            assert_eq!(right.get(&(i + 2500)), Some(&(i + 2501)));
        }

        // Removes still work after split.
        for i in (0..2500).step_by(7) {
            assert_eq!(map.remove(&i), Some(i + 1));
        }
        assert_eq!(map.len(), 2500 - (2500_usize.div_ceil(7)));
    }

    // ── Surgical split_off tests ───────────────────────────────────────────
    //
    // The surgical variant has independent invariants from drain+bulk_load,
    // so it gets its own coverage. Tests run via `split_off_surgical_right()` directly.

    fn assert_sorted_and_len<K: Ord + std::fmt::Debug, V>(tree: &FlatBTree<K, V>) {
        let keys: Vec<&K> = tree.iter().map(|(k, _)| k).collect();
        for w in keys.windows(2) {
            assert!(w[0] < w[1], "iter not sorted: {:?} >= {:?}", w[0], w[1]);
        }
        assert_eq!(keys.len(), tree.len(), "iter count != len()");
    }

    #[test]
    fn surgical_split_off_basic() {
        let mut map: FlatBTree<i32, i32> = (0..200).map(|i| (i, i * 10)).collect();
        let right = map.split_off_surgical_right(&100);

        assert_eq!(map.len(), 100);
        assert_eq!(right.len(), 100);
        assert_sorted_and_len(&map);
        assert_sorted_and_len(&right);
        for i in 0..100 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
            assert_eq!(right.get(&i), None);
        }
        for i in 100..200 {
            assert_eq!(map.get(&i), None);
            assert_eq!(right.get(&i), Some(&(i * 10)));
        }
    }

    #[test]
    fn surgical_split_off_at_missing_key() {
        let mut map: FlatBTree<i32, i32> = [10, 20, 30, 40].iter().map(|&i| (i, i)).collect();
        let right = map.split_off_surgical_right(&25);
        assert_eq!(map.iter().map(|(k, _)| *k).collect::<Vec<_>>(), vec![10, 20]);
        assert_eq!(right.iter().map(|(k, _)| *k).collect::<Vec<_>>(), vec![30, 40]);
    }

    #[test]
    fn surgical_split_off_edges() {
        // Split before everything → self empty, right gets all.
        let mut a: FlatBTree<i32, i32> = (1..=10).map(|i| (i, i)).collect();
        let r = a.split_off_surgical_right(&0);
        assert!(a.is_empty());
        assert_eq!(r.len(), 10);
        assert_sorted_and_len(&r);

        // Split at the smallest key still moves everything to right (lower_bound = first leaf, slot 0).
        let mut a2: FlatBTree<i32, i32> = (1..=10).map(|i| (i, i)).collect();
        let r2 = a2.split_off_surgical_right(&1);
        assert!(a2.is_empty());
        assert_eq!(r2.len(), 10);

        // Split after everything → self keeps all, right empty.
        let mut b: FlatBTree<i32, i32> = (1..=10).map(|i| (i, i)).collect();
        let r = b.split_off_surgical_right(&100);
        assert_eq!(b.len(), 10);
        assert!(r.is_empty());
        assert_sorted_and_len(&b);

        // Empty self → empty right.
        let mut c: FlatBTree<i32, i32> = FlatBTree::new();
        let r = c.split_off_surgical_right(&5);
        assert!(c.is_empty());
        assert!(r.is_empty());

        // Single-element tree, split below key.
        let mut d: FlatBTree<i32, i32> = [(5, 50)].into_iter().collect();
        let r = d.split_off_surgical_right(&5);
        assert!(d.is_empty());
        assert_eq!(r.len(), 1);

        // Single-element tree, split above key.
        let mut e: FlatBTree<i32, i32> = [(5, 50)].into_iter().collect();
        let r = e.split_off_surgical_right(&6);
        assert_eq!(e.len(), 1);
        assert!(r.is_empty());
    }

    #[test]
    fn surgical_split_off_single_leaf_root() {
        // Tree fits in one leaf (height 0). Surgical must handle the empty-spine case.
        let mut map: FlatBTree<i32, i32> = (0..10).map(|i| (i, i)).collect();
        let right = map.split_off_surgical_right(&5);
        assert_eq!(map.iter().map(|(k, _)| *k).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4]);
        assert_eq!(right.iter().map(|(k, _)| *k).collect::<Vec<_>>(), vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn surgical_split_off_at_leaf_boundary() {
        // Pivot matches the first key of a non-first leaf → slot_idx == 0
        // for a non-first leaf; tests the "whole leaf moves right" code path.
        let mut map: FlatBTree<u64, u64> = (0..1_000).map(|i| (i, i)).collect();
        // LEAF_CAP for u64/u64 is 15. So leaf 0 holds keys 0..15, leaf 1 holds 15..30, etc.
        // Pivot = 15 lands at slot 0 of leaf 1.
        let right = map.split_off_surgical_right(&15);
        assert_eq!(map.len(), 15);
        assert_eq!(right.len(), 985);
        assert_sorted_and_len(&map);
        assert_sorted_and_len(&right);
        for i in 0..15 {
            assert_eq!(map.get(&i), Some(&i));
        }
        for i in 15..1000 {
            assert_eq!(right.get(&i), Some(&i));
        }
    }

    #[test]
    fn surgical_split_off_mid_leaf() {
        // Pivot lands inside a leaf, exercising the physical leaf split path.
        let mut map: FlatBTree<u64, u64> = (0..1_000).map(|i| (i, i)).collect();
        // 22 lands somewhere inside leaf 1 (keys 15..30 with LEAF_CAP=15).
        let right = map.split_off_surgical_right(&22);
        assert_eq!(map.len(), 22);
        assert_eq!(right.len(), 978);
        assert_sorted_and_len(&map);
        assert_sorted_and_len(&right);
    }

    #[test]
    fn surgical_split_off_left_collapses() {
        // Pivot near the start: most of the tree moves right, the spine on
        // the left has child_pos == 0 at multiple levels and produces a chain
        // of degenerate internals that Phase 3 must collapse.
        let mut map: FlatBTree<u64, u64> = (0..10_000).map(|i| (i, i)).collect();
        let right = map.split_off_surgical_right(&50);
        assert_eq!(map.len(), 50);
        assert_eq!(right.len(), 9950);
        assert_sorted_and_len(&map);
        assert_sorted_and_len(&right);
        for i in 0..50 {
            assert_eq!(map.get(&i), Some(&i));
        }
        for i in 50..10_000 {
            assert_eq!(right.get(&i), Some(&i));
        }
    }

    #[test]
    fn surgical_split_off_right_collapses() {
        // Pivot near the end: spine descends rightmost children, copied subtrees
        // are empty, right side accumulates degenerate internals along its
        // leftmost edge that Phase 3 must collapse.
        let mut map: FlatBTree<u64, u64> = (0..10_000).map(|i| (i, i)).collect();
        let right = map.split_off_surgical_right(&9_950);
        assert_eq!(map.len(), 9950);
        assert_eq!(right.len(), 50);
        assert_sorted_and_len(&map);
        assert_sorted_and_len(&right);
        for i in 0..9950 {
            assert_eq!(map.get(&i), Some(&i));
        }
        for i in 9950..10_000 {
            assert_eq!(right.get(&i), Some(&i));
        }
    }

    #[test]
    fn surgical_split_off_post_split_mutate() {
        let mut map: FlatBTree<u64, u64> = (0..5_000).map(|i| (i, i)).collect();
        let mut right = map.split_off_surgical_right(&2_500);
        assert_eq!(map.len(), 2500);
        assert_eq!(right.len(), 2500);

        // Insert + replace into both halves — exercises split paths in both arenas.
        for i in 0..2500 {
            map.insert(i, i + 1);
            right.insert(i + 2500, i + 2501);
        }
        assert_eq!(map.len(), 2500);
        assert_eq!(right.len(), 2500);
        for i in 0..2500 {
            assert_eq!(map.get(&i), Some(&(i + 1)));
            assert_eq!(right.get(&(i + 2500)), Some(&(i + 2501)));
        }

        // Removes — exercises rebalance on the surgically-built spines.
        for i in (0..2500).step_by(7) {
            assert_eq!(map.remove(&i), Some(i + 1));
        }
        assert_eq!(map.len(), 2500 - (2500_usize.div_ceil(7)));
        assert_sorted_and_len(&map);
        assert_sorted_and_len(&right);
    }

    #[test]
    fn surgical_matches_drain_across_pivots() {
        // Run both implementations side by side on the same input + pivot,
        // and assert the resulting (left, right) trees have identical contents.
        for &n in &[1usize, 7, 15, 16, 30, 100, 500, 5_000, 50_000] {
            // Pick a representative spread of pivots.
            let pivots: Vec<i64> = vec![
                -1,                    // before everything
                0,                     // first key
                (n / 1000).max(1) as i64,    // very-near start
                (n / 10) as i64,             // 10%
                (n / 2) as i64,              // mid
                (n - n / 10) as i64,         // 90%
                (n - 1).max(0) as i64,       // last key
                n as i64,                    // beyond everything
            ];
            for &pivot in &pivots {
                let mut a: FlatBTree<i64, i64> = (0..n as i64).map(|i| (i, i * 3)).collect();
                let mut b: FlatBTree<i64, i64> = (0..n as i64).map(|i| (i, i * 3)).collect();

                let a_right = a.split_off_drain(&pivot);
                let b_right = b.split_off_surgical_right(&pivot);

                let a_left_keys: Vec<i64> = a.iter().map(|(k, _)| *k).collect();
                let b_left_keys: Vec<i64> = b.iter().map(|(k, _)| *k).collect();
                let a_right_keys: Vec<i64> = a_right.iter().map(|(k, _)| *k).collect();
                let b_right_keys: Vec<i64> = b_right.iter().map(|(k, _)| *k).collect();

                assert_eq!(
                    a_left_keys, b_left_keys,
                    "left mismatch (n={n}, pivot={pivot})"
                );
                assert_eq!(
                    a_right_keys, b_right_keys,
                    "right mismatch (n={n}, pivot={pivot})"
                );
                assert_eq!(a.len(), b.len());
                assert_eq!(a_right.len(), b_right.len());
            }
        }
    }

    #[test]
    fn surgical_split_off_drop_types() {
        // String keys and values exercise the Drop logic for moved-out K/V.
        // No leaks, no double-drops: rely on miri or the test runner's leak
        // detection plus the explicit content check.
        let mut map: FlatBTree<String, String> = (0..200)
            .map(|i| (format!("key_{i:03}"), format!("val_{i}")))
            .collect();
        let pivot = "key_100".to_string();
        let right = map.split_off_surgical_right(&pivot);
        assert_eq!(map.len(), 100);
        assert_eq!(right.len(), 100);
        for i in 0..100 {
            assert_eq!(map.get(&format!("key_{i:03}")), Some(&format!("val_{i}")));
        }
        for i in 100..200 {
            assert_eq!(right.get(&format!("key_{i:03}")), Some(&format!("val_{i}")));
        }
        assert_sorted_and_len(&map);
        assert_sorted_and_len(&right);
    }

    #[test]
    fn surgical_split_off_iterate_full_chain() {
        // Walking the leaf chain via iter() in both halves verifies prev/next
        // pointers are consistent on both sides after the surgery.
        let mut map: FlatBTree<u64, u64> = (0..50_000).map(|i| (i, i)).collect();
        let pivot = 25_137_u64; // arbitrary mid-leaf pivot
        let right = map.split_off_surgical_right(&pivot);

        let left_keys: Vec<u64> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(left_keys, (0..pivot).collect::<Vec<_>>());
        let right_keys: Vec<u64> = right.iter().map(|(k, _)| *k).collect();
        assert_eq!(right_keys, (pivot..50_000).collect::<Vec<_>>());
    }

    #[test]
    fn estimate_right_fraction_matches_dispatch_intent() {
        // Sanity check that the dispatcher's estimator routes the bench-tested
        // pivot positions to the correct surgical direction. The default
        // cutoff is 0.70 — high enough to (1) absorb the ~+0.10 systematic
        // estimator bias from uniform-subtree-weighting, AND (2) avoid
        // routing to surgical_left when surgical_right is faster despite
        // copying more (the variants have a ~2-3× constant-factor asymmetry).
        let cutoff = FlatBTree::<u64, u64>::DEFAULT_RIGHT_FRAC_LEFT_CUTOFF;
        let pairs: Vec<(u64, u64)> = (0..1000_u64).map(|i| (i, i)).collect();
        let map: FlatBTree<u64, u64> = pairs.iter().copied().collect();
        // p001 / p010: estimator says ≈ 0.99 / 0.92 (actual 0.99 / 0.90)
        //              → above 0.70 → surgical_left (deep-copy small left)
        assert!(map.tree.estimate_right_fraction(&10).unwrap() > cutoff);
        assert!(map.tree.estimate_right_fraction(&100).unwrap() > cutoff);
        // p050: estimator says ≈ 0.60 (actual 0.50). Below cutoff →
        // surgical_right (small constant-factor advantage at symmetric pivot).
        let r050 = map.tree.estimate_right_fraction(&500).unwrap();
        assert!(r050 > 0.55 && r050 <= 0.65);
        assert!(r050 <= cutoff, "p050 estimator must route to surgical_right");
        // p090 / p099: estimator says ≈ 0.29 / 0.01 (actual 0.10 / 0.01)
        //              → below 0.70 → surgical_right (deep-copy small right)
        assert!(map.tree.estimate_right_fraction(&900).unwrap() <= cutoff);
        assert!(map.tree.estimate_right_fraction(&990).unwrap() <= cutoff);
    }

    #[test]
    fn surgical_split_off_repeated() {
        // Repeatedly split off the high half. Each call exercises a smaller
        // tree, ensuring the post-split structure remains valid for the next
        // operation.
        let mut map: FlatBTree<u64, u64> = (0..10_000).map(|i| (i, i)).collect();
        let mut total = 10_000_u64;
        for cut in &[5_000_u64, 2_500, 1_250, 625, 312, 156, 78, 39, 19, 9, 4, 1] {
            let right = map.split_off_surgical_right(cut);
            assert_eq!(map.len() + right.len(), total as usize);
            assert_sorted_and_len(&map);
            assert_sorted_and_len(&right);
            total = *cut;
        }
        assert_eq!(map.len() as u64, total);
    }

    // ── Surgical copy-LEFT split_off tests ────────────────────────────────
    //
    // Mirror coverage for `split_off_surgical_left`. Each test asserts the
    // same invariants as its copy-right counterpart, verifying both (left,
    // right) contents and structural soundness via assert_sorted_and_len.

    #[test]
    fn surgical_left_split_off_basic() {
        let mut map: FlatBTree<i32, i32> = (0..200).map(|i| (i, i * 10)).collect();
        let right = map.split_off_surgical_left(&100);
        assert_eq!(map.len(), 100);
        assert_eq!(right.len(), 100);
        assert_sorted_and_len(&map);
        assert_sorted_and_len(&right);
        for i in 0..100 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
            assert_eq!(right.get(&i), None);
        }
        for i in 100..200 {
            assert_eq!(map.get(&i), None);
            assert_eq!(right.get(&i), Some(&(i * 10)));
        }
    }

    #[test]
    fn surgical_left_split_off_at_missing_key() {
        let mut map: FlatBTree<i32, i32> =
            (0..50).map(|i| (i * 2, i)).collect(); // even keys only
        let right = map.split_off_surgical_left(&25); // 25 is not present
        assert_eq!(map.len(), 13);
        assert_eq!(right.len(), 37);
        assert_sorted_and_len(&map);
        assert_sorted_and_len(&right);
        assert_eq!(map.get(&24), Some(&12));
        assert_eq!(right.get(&26), Some(&13));
    }

    #[test]
    fn surgical_left_split_off_edges() {
        // pivot before first key → entire tree to right.
        let mut a: FlatBTree<i32, i32> = (10..50).map(|i| (i, i)).collect();
        let r = a.split_off_surgical_left(&0);
        assert!(a.is_empty());
        assert_eq!(r.len(), 40);

        let mut a2: FlatBTree<i32, i32> = (1..5).map(|i| (i, i)).collect();
        let r2 = a2.split_off_surgical_left(&1);
        assert!(a2.is_empty());
        assert_eq!(r2.len(), 4);

        // pivot after last key → right empty.
        let mut b: FlatBTree<i32, i32> = (0..50).map(|i| (i, i)).collect();
        let r = b.split_off_surgical_left(&100);
        assert!(r.is_empty());
        assert_eq!(b.len(), 50);

        // pivot equals interior key.
        let mut c: FlatBTree<i32, i32> = (0..10).map(|i| (i, i * 3)).collect();
        let r = c.split_off_surgical_left(&5);
        assert_eq!(c.len(), 5);
        assert_eq!(r.len(), 5);

        // pivot at a missing key just below an existing one.
        let mut d: FlatBTree<i32, i32> = (0..10).map(|i| (i * 2, i)).collect();
        let r = d.split_off_surgical_left(&5); // first key >= 5 is 6
        assert_eq!(d.len(), 3); // 0,2,4
        assert_eq!(r.len(), 7); // 6,8,...,18

        // missing pivot just above, picks last key precisely.
        let mut e: FlatBTree<i32, i32> = (0..10).map(|i| (i * 2, i)).collect();
        let r = e.split_off_surgical_left(&6);
        assert_eq!(e.len(), 3);
        assert_eq!(r.len(), 7);
    }

    #[test]
    fn surgical_left_split_off_single_leaf_root() {
        // Tree fits in one leaf (height 0). Spine walk runs zero iterations.
        let mut map: FlatBTree<u64, u64> = (0..10).map(|i| (i, i)).collect();
        let right = map.split_off_surgical_left(&5);
        assert_eq!(map.len(), 5);
        assert_eq!(right.len(), 5);
        assert_sorted_and_len(&map);
        assert_sorted_and_len(&right);
        for i in 0..5 {
            assert_eq!(map.get(&i), Some(&i));
        }
        for i in 5..10 {
            assert_eq!(right.get(&i), Some(&i));
        }
    }

    #[test]
    fn surgical_left_split_off_at_leaf_boundary() {
        // Pivot = first key of a non-first leaf → slot_idx == 0, entire boundary
        // leaf stays in self.arena (no in-place shift). Exercises the
        // slot_idx == 0 branch.
        let mut map: FlatBTree<u64, u64> = (0..200).map(|i| (i, i)).collect();
        // LEAF_CAP for u64/u64 is 15, so leaves boundaries are at multiples of 15.
        // Pivot at exactly key 15 should align with a leaf boundary.
        let right = map.split_off_surgical_left(&15);
        assert_eq!(map.len(), 15);
        assert_eq!(right.len(), 185);
        assert_sorted_and_len(&map);
        assert_sorted_and_len(&right);
    }

    #[test]
    fn surgical_left_split_off_mid_leaf() {
        // Pivot mid-leaf (slot_idx > 0) — boundary leaf physically split,
        // keys[slot_idx..] shifted in place to slots [0..].
        let mut map: FlatBTree<u64, u64> = (0..200).map(|i| (i, i)).collect();
        let right = map.split_off_surgical_left(&22); // 22 % 15 = 7, so mid-leaf
        assert_eq!(map.len(), 22);
        assert_eq!(right.len(), 178);
        assert_sorted_and_len(&map);
        assert_sorted_and_len(&right);
    }

    #[test]
    fn surgical_left_split_off_left_collapses() {
        // Pivot near the END forces left-side spine to be a chain of degenerates
        // along the rightmost edge. Exercises Phase 4 collapse_along_edge(Rightmost).
        let mut map: FlatBTree<u64, u64> = (0..100).map(|i| (i, i)).collect();
        let right = map.split_off_surgical_left(&99);
        assert_eq!(map.len(), 99);
        assert_eq!(right.len(), 1);
        assert_sorted_and_len(&map);
        assert_sorted_and_len(&right);
    }

    #[test]
    fn surgical_left_split_off_right_collapses() {
        // Pivot near the START forces kept (right) spine to be degenerates
        // along the leftmost edge. Exercises Phase 3 collapse_along_edge(Leftmost).
        let mut map: FlatBTree<u64, u64> = (0..100).map(|i| (i, i)).collect();
        let right = map.split_off_surgical_left(&1);
        assert_eq!(map.len(), 1);
        assert_eq!(right.len(), 99);
        assert_sorted_and_len(&map);
        assert_sorted_and_len(&right);
    }

    #[test]
    fn surgical_left_split_off_post_split_mutate() {
        let mut map: FlatBTree<u64, u64> = (0..5_000).map(|i| (i, i)).collect();
        let mut right = map.split_off_surgical_left(&2_500);
        assert_eq!(map.len(), 2500);
        assert_eq!(right.len(), 2500);
        for i in 0..2500 {
            map.insert(i, i + 1);
            right.insert(i + 2500, i + 2501);
        }
        assert_eq!(map.len(), 2500);
        assert_eq!(right.len(), 2500);
        for i in 0..2500 {
            assert_eq!(map.get(&i), Some(&(i + 1)));
            assert_eq!(right.get(&(i + 2500)), Some(&(i + 2501)));
        }
        for i in (0..2500).step_by(7) {
            assert_eq!(map.remove(&i), Some(i + 1));
        }
        assert_eq!(map.len(), 2500 - (2500_usize.div_ceil(7)));
        assert_sorted_and_len(&map);
        assert_sorted_and_len(&right);
    }

    #[test]
    fn surgical_left_split_off_drop_types() {
        // String K/V exercises Drop on the deep-copied LEFT side.
        let mut map: FlatBTree<String, String> = (0..200)
            .map(|i| (format!("key_{i:03}"), format!("val_{i}")))
            .collect();
        let pivot = "key_100".to_string();
        let right = map.split_off_surgical_left(&pivot);
        assert_eq!(map.len(), 100);
        assert_eq!(right.len(), 100);
        for i in 0..100 {
            assert_eq!(map.get(&format!("key_{i:03}")), Some(&format!("val_{i}")));
        }
        for i in 100..200 {
            assert_eq!(right.get(&format!("key_{i:03}")), Some(&format!("val_{i}")));
        }
        assert_sorted_and_len(&map);
        assert_sorted_and_len(&right);
    }

    #[test]
    fn surgical_left_split_off_iterate_full_chain() {
        let mut map: FlatBTree<u64, u64> = (0..50_000).map(|i| (i, i)).collect();
        let pivot = 25_137_u64;
        let right = map.split_off_surgical_left(&pivot);

        let left_keys: Vec<u64> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(left_keys, (0..pivot).collect::<Vec<_>>());
        let right_keys: Vec<u64> = right.iter().map(|(k, _)| *k).collect();
        assert_eq!(right_keys, (pivot..50_000).collect::<Vec<_>>());
    }

    #[test]
    fn surgical_left_split_off_repeated() {
        // Repeatedly split off the LOW half. Each call exercises a smaller
        // tree on the kept (right) side, while the deep-copy chain accumulates
        // the discarded prefixes — must remain sorted-iterable each round.
        let mut map: FlatBTree<u64, u64> = (0..10_000).map(|i| (i, i)).collect();
        let cuts = [
            500_u64, 1_000, 1_500, 2_500, 3_500, 5_000, 6_500, 8_000, 9_000, 9_500, 9_900, 9_999,
        ];
        for cut in &cuts {
            let len_before = map.len();
            let left = map.split_off_surgical_left(cut);
            assert!(left.len() <= len_before);
            assert_eq!(left.len() + map.len(), len_before);
            assert_sorted_and_len(&map);
            assert_sorted_and_len(&left);
        }
    }

    #[test]
    fn surgical_left_matches_drain_and_right() {
        // Three-way cross-validation: drain, surgical_right, surgical_left
        // must all produce identical (left_contents, right_contents) trees
        // for the same input + pivot. Covers many sizes including:
        // single-leaf (height 0), height 1 (15..225), height 2 (225..3375),
        // height 3 (>3375), and irregular fills near boundaries.
        for &n in &[0usize, 1, 7, 15, 16, 30, 100, 224, 225, 226, 500, 5_000, 50_000] {
            // Pick a representative spread of pivots, including out-of-range.
            let pivots: Vec<i64> = if n == 0 {
                vec![0, 5]
            } else {
                vec![
                    -1,                          // before everything
                    0,                           // first key
                    (n / 1000).max(1) as i64,    // very-near start
                    (n / 100).max(1) as i64,     // p001
                    (n / 10) as i64,             // p010
                    (n / 2) as i64,              // p050
                    (n - n / 10) as i64,         // p090
                    (n - n / 100).max(0) as i64, // p099
                    (n - 1).max(0) as i64,       // last key
                    n as i64,                    // beyond everything
                ]
            };
            for &pivot in &pivots {
                let mut a: FlatBTree<i64, i64> = (0..n as i64).map(|i| (i, i * 3)).collect();
                let mut b: FlatBTree<i64, i64> = (0..n as i64).map(|i| (i, i * 3)).collect();
                let mut c: FlatBTree<i64, i64> = (0..n as i64).map(|i| (i, i * 3)).collect();

                let a_right = a.split_off_drain(&pivot);
                let b_right = b.split_off_surgical_right(&pivot);
                let c_right = c.split_off_surgical_left(&pivot);

                let a_left_keys: Vec<i64> = a.iter().map(|(k, _)| *k).collect();
                let b_left_keys: Vec<i64> = b.iter().map(|(k, _)| *k).collect();
                let c_left_keys: Vec<i64> = c.iter().map(|(k, _)| *k).collect();
                let a_right_keys: Vec<i64> = a_right.iter().map(|(k, _)| *k).collect();
                let b_right_keys: Vec<i64> = b_right.iter().map(|(k, _)| *k).collect();
                let c_right_keys: Vec<i64> = c_right.iter().map(|(k, _)| *k).collect();

                assert_eq!(
                    a_left_keys, b_left_keys,
                    "left mismatch drain vs surgical_right (n={n}, pivot={pivot})"
                );
                assert_eq!(
                    a_left_keys, c_left_keys,
                    "left mismatch drain vs surgical_left (n={n}, pivot={pivot})"
                );
                assert_eq!(
                    a_right_keys, b_right_keys,
                    "right mismatch drain vs surgical_right (n={n}, pivot={pivot})"
                );
                assert_eq!(
                    a_right_keys, c_right_keys,
                    "right mismatch drain vs surgical_left (n={n}, pivot={pivot})"
                );

                // Cross-check values too, not just keys.
                let a_left_vals: Vec<i64> = a.iter().map(|(_, v)| *v).collect();
                let c_left_vals: Vec<i64> = c.iter().map(|(_, v)| *v).collect();
                assert_eq!(
                    a_left_vals, c_left_vals,
                    "left value mismatch drain vs surgical_left (n={n}, pivot={pivot})"
                );
                let a_right_vals: Vec<i64> = a_right.iter().map(|(_, v)| *v).collect();
                let c_right_vals: Vec<i64> = c_right.iter().map(|(_, v)| *v).collect();
                assert_eq!(
                    a_right_vals, c_right_vals,
                    "right value mismatch drain vs surgical_left (n={n}, pivot={pivot})"
                );
            }
        }
    }

    #[test]
    fn append_disjoint() {
        let mut a: FlatBTree<i32, i32> = (0..50).map(|i| (i, i)).collect();
        let mut b: FlatBTree<i32, i32> = (50..100).map(|i| (i, i + 1000)).collect();
        a.append(&mut b);
        assert!(b.is_empty());
        assert_eq!(a.len(), 100);
        for i in 0..50 {
            assert_eq!(a.get(&i), Some(&i));
        }
        for i in 50..100 {
            assert_eq!(a.get(&i), Some(&(i + 1000)));
        }
    }

    #[test]
    fn append_overlapping_other_wins() {
        // Verify std::BTreeMap::append semantics: on key collision, other wins.
        let mut a: FlatBTree<i32, &str> = [(1, "a1"), (2, "a2"), (3, "a3")].into_iter().collect();
        let mut b: FlatBTree<i32, &str> = [(2, "b2"), (3, "b3"), (4, "b4")].into_iter().collect();
        a.append(&mut b);
        assert!(b.is_empty());
        assert_eq!(a.get(&1), Some(&"a1"));
        assert_eq!(a.get(&2), Some(&"b2"));
        assert_eq!(a.get(&3), Some(&"b3"));
        assert_eq!(a.get(&4), Some(&"b4"));
    }

    #[test]
    fn append_empty_cases() {
        // Append into empty: self gets all of other.
        let mut a: FlatBTree<i32, i32> = FlatBTree::new();
        let mut b: FlatBTree<i32, i32> = (0..10).map(|i| (i, i)).collect();
        a.append(&mut b);
        assert!(b.is_empty());
        assert_eq!(a.len(), 10);

        // Append empty: self unchanged.
        let mut a: FlatBTree<i32, i32> = (0..10).map(|i| (i, i)).collect();
        let mut b: FlatBTree<i32, i32> = FlatBTree::new();
        a.append(&mut b);
        assert!(b.is_empty());
        assert_eq!(a.len(), 10);
    }

    #[test]
    fn append_large_then_check_invariants() {
        let mut a: FlatBTree<u64, u64> = (0..3000).map(|i| (i * 2, i)).collect();
        let mut b: FlatBTree<u64, u64> = (0..3000).map(|i| (i * 2 + 1, i + 100_000)).collect();
        a.append(&mut b);
        assert_eq!(a.len(), 6000);
        // Sorted iteration check
        let keys: Vec<u64> = a.iter().map(|(k, _)| *k).collect();
        assert!(keys.windows(2).all(|w| w[0] < w[1]));
        // Spot-check
        assert_eq!(a.get(&100), Some(&50));
        assert_eq!(a.get(&101), Some(&(50 + 100_000)));
    }

    #[test]
    fn append_reverse_order_swap() {
        // self.first > other.last → dispatcher must swap then extend.
        let mut a: FlatBTree<u64, u64> = (100..150).map(|i| (i, i + 1000)).collect();
        let mut b: FlatBTree<u64, u64> = (0..50).map(|i| (i, i)).collect();
        a.append(&mut b);
        assert!(b.is_empty());
        assert_eq!(a.len(), 100);
        for i in 0..50 {
            assert_eq!(a.get(&i), Some(&i));
        }
        for i in 100..150 {
            assert_eq!(a.get(&i), Some(&(i + 1000)));
        }
        // sorted invariant
        let keys: Vec<u64> = a.iter().map(|(k, _)| *k).collect();
        assert!(keys.windows(2).all(|w| w[0] < w[1]));
    }

    // ── Per-variant correctness tests ─────────────────────────────────

    fn check_append_invariants(a: &FlatBTree<u64, u64>, expected: &[(u64, u64)]) {
        assert_eq!(a.len(), expected.len());
        let actual: Vec<(u64, u64)> = a.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(actual, expected);
    }

    fn build_disjoint_pair(
        n_a: usize,
        n_b: usize,
    ) -> (FlatBTree<u64, u64>, FlatBTree<u64, u64>, Vec<(u64, u64)>) {
        let a: FlatBTree<u64, u64> = (0..n_a as u64).map(|i| (i, i)).collect();
        let b: FlatBTree<u64, u64> = (n_a as u64..(n_a + n_b) as u64)
            .map(|i| (i, i * 2))
            .collect();
        let mut expected: Vec<(u64, u64)> = (0..n_a as u64).map(|i| (i, i)).collect();
        expected.extend((n_a as u64..(n_a + n_b) as u64).map(|i| (i, i * 2)));
        (a, b, expected)
    }

    #[test]
    fn append_drain_overlapping() {
        // append_drain handles overlap correctly (other wins).
        let mut a: FlatBTree<u64, u64> =
            [(1u64, 11u64), (2, 12), (3, 13)].into_iter().collect();
        let mut b: FlatBTree<u64, u64> = [(2u64, 22u64), (3, 23), (4, 24)].into_iter().collect();
        a.append_drain(&mut b);
        assert!(b.is_empty());
        check_append_invariants(&a, &[(1, 11), (2, 22), (3, 23), (4, 24)]);
    }

    #[test]
    fn append_concat_disjoint_small() {
        let (mut a, mut b, expected) = build_disjoint_pair(50, 50);
        a.append_concat(&mut b);
        assert!(b.is_empty());
        check_append_invariants(&a, &expected);
    }

    #[test]
    fn append_concat_disjoint_multilevel() {
        // 3000 + 3000 → multi-level B+ tree.
        let (mut a, mut b, expected) = build_disjoint_pair(3000, 3000);
        a.append_concat(&mut b);
        assert!(b.is_empty());
        check_append_invariants(&a, &expected);
    }

    #[test]
    fn append_extend_disjoint_small() {
        let (mut a, mut b, expected) = build_disjoint_pair(50, 50);
        a.append_extend(&mut b);
        assert!(b.is_empty());
        check_append_invariants(&a, &expected);
    }

    #[test]
    fn append_extend_disjoint_multilevel() {
        let (mut a, mut b, expected) = build_disjoint_pair(3000, 3000);
        a.append_extend(&mut b);
        assert!(b.is_empty());
        check_append_invariants(&a, &expected);
    }

    #[test]
    fn append_graft_disjoint_small() {
        // Both small → both single-leaf trees, equal heights.
        let (mut a, mut b, expected) = build_disjoint_pair(5, 5);
        a.append_graft(&mut b);
        assert!(b.is_empty());
        check_append_invariants(&a, &expected);
    }

    #[test]
    fn append_graft_equal_heights_multilevel() {
        // Two 3000-entry trees, both 3-level — exercises the full graft path
        // including the new-root-above-both case.
        let (mut a, mut b, expected) = build_disjoint_pair(3000, 3000);
        a.append_graft(&mut b);
        assert!(b.is_empty());
        check_append_invariants(&a, &expected);
    }

    #[test]
    fn append_graft_self_taller() {
        // self has 100K entries (height 3 for u64), other has 30 entries
        // (height 1 — single internal node above 2 leaves). Exercises the
        // descent-spine codepath.
        let (mut a, mut b, expected) = build_disjoint_pair(100_000, 30);
        a.append_graft(&mut b);
        assert!(b.is_empty());
        check_append_invariants(&a, &expected);
    }

    #[test]
    fn append_graft_self_taller_full_seam() {
        // Provoke the cascade-split path: self has height 3 with the
        // rightmost spine fully packed, then graft other (small, single
        // leaf). The descended cur should be full → split + propagate.
        let (mut a, mut b, expected) = build_disjoint_pair(50_000, 5);
        a.append_graft(&mut b);
        assert!(b.is_empty());
        check_append_invariants(&a, &expected);
    }

    #[test]
    fn append_graft_other_taller_falls_back() {
        // self.height < other.height → graft falls back to drain. Result
        // must still be correct.
        let (mut a, mut b, expected) = build_disjoint_pair(30, 100_000);
        a.append_graft(&mut b);
        assert!(b.is_empty());
        check_append_invariants(&a, &expected);
    }

    #[test]
    fn append_graft_huge_disjoint() {
        // 1M-entry stress to flush out arena-growth corner cases during graft.
        let (mut a, mut b, expected) = build_disjoint_pair(500_000, 500_000);
        a.append_graft(&mut b);
        assert!(b.is_empty());
        check_append_invariants(&a, &expected);
    }

    #[test]
    fn append_graft_then_mutate() {
        // After graft, follow-up insert/remove operations must work normally.
        let (mut a, mut b, _) = build_disjoint_pair(3000, 3000);
        a.append_graft(&mut b);

        // Insert into the seam region.
        a.insert(2999_500, 999_500);
        assert_eq!(a.get(&2999_500), Some(&999_500));

        // Remove around the seam.
        for i in 2_990..3_010u64 {
            assert!(a.remove(&i).is_some());
        }
        assert_eq!(a.len(), 6000 + 1 - 20);

        // Sorted iteration still holds.
        let keys: Vec<u64> = a.iter().map(|(k, _)| *k).collect();
        assert!(keys.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn append_dispatcher_disjoint_uses_extend() {
        // Public append must produce identical results to all variants.
        let (mut a, mut b, expected) = build_disjoint_pair(3000, 3000);
        a.append(&mut b);
        assert!(b.is_empty());
        check_append_invariants(&a, &expected);
    }

    #[test]
    fn append_cross_validation() {
        // All four variants must produce identical results on disjoint input.
        let (a_drain_lhs, a_drain_rhs, _) = build_disjoint_pair(2000, 2000);
        let (a_concat_lhs, a_concat_rhs, _) = build_disjoint_pair(2000, 2000);
        let (a_extend_lhs, a_extend_rhs, _) = build_disjoint_pair(2000, 2000);
        let (a_graft_lhs, a_graft_rhs, _) = build_disjoint_pair(2000, 2000);

        let mut a = a_drain_lhs;
        let mut b = a_drain_rhs;
        a.append_drain(&mut b);
        let drain_out: Vec<(u64, u64)> = a.iter().map(|(k, v)| (*k, *v)).collect();

        let mut a = a_concat_lhs;
        let mut b = a_concat_rhs;
        a.append_concat(&mut b);
        let concat_out: Vec<(u64, u64)> = a.iter().map(|(k, v)| (*k, *v)).collect();

        let mut a = a_extend_lhs;
        let mut b = a_extend_rhs;
        a.append_extend(&mut b);
        let extend_out: Vec<(u64, u64)> = a.iter().map(|(k, v)| (*k, *v)).collect();

        let mut a = a_graft_lhs;
        let mut b = a_graft_rhs;
        a.append_graft(&mut b);
        let graft_out: Vec<(u64, u64)> = a.iter().map(|(k, v)| (*k, *v)).collect();

        assert_eq!(drain_out, concat_out);
        assert_eq!(drain_out, extend_out);
        assert_eq!(drain_out, graft_out);
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
