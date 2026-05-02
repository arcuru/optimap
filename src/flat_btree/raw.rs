//! Core B+ tree engine: arena allocator, search, insert/split.

use std::alloc::{self, Layout};
use std::borrow::Borrow;
use std::marker::PhantomData;

use super::node::*;

// ── Arena ───────────────────────────────────────────────────────────────

/// Slab allocator for 256-byte-aligned node blocks.
pub(crate) struct Arena {
    /// Base pointer to the allocation (null when empty).
    ptr: *mut u8,
    /// Number of node slots allocated.
    cap: u32,
    /// High-water mark: next fresh slot index.
    len: u32,
    /// Head of free list (freed nodes linked via first 4 bytes).
    free_head: NodeIdx,
}

impl Arena {
    const INITIAL_CAP: u32 = 4;

    pub fn new() -> Self {
        Arena {
            ptr: std::ptr::null_mut(),
            cap: 0,
            len: 0,
            free_head: NO_NODE,
        }
    }

    pub fn with_capacity(node_count: u32) -> Self {
        if node_count == 0 {
            return Self::new();
        }
        let cap = node_count.next_power_of_two().max(Self::INITIAL_CAP);
        let ptr = unsafe { alloc::alloc_zeroed(Self::layout(cap)) };
        if ptr.is_null() {
            alloc::handle_alloc_error(Self::layout(cap));
        }
        Arena {
            ptr,
            cap,
            len: 0,
            free_head: NO_NODE,
        }
    }

    fn layout(cap: u32) -> Layout {
        Layout::from_size_align(cap as usize * NODE_SIZE, NODE_SIZE).unwrap()
    }

    /// Get a raw pointer to the node at the given index.
    #[inline(always)]
    pub fn node_ptr(&self, idx: NodeIdx) -> *mut u8 {
        debug_assert!(idx != NO_NODE);
        debug_assert!(idx < self.len);
        unsafe { self.ptr.add(idx as usize * NODE_SIZE) }
    }

    /// Allocate a new node, returning its index. The node is zeroed.
    pub fn alloc_node(&mut self) -> NodeIdx {
        // Try free list first
        if self.free_head != NO_NODE {
            let idx = self.free_head;
            let node = self.node_ptr(idx);
            // Read next free pointer from the freed node's first 4 bytes
            self.free_head = unsafe { node.cast::<NodeIdx>().read() };
            // Zero the node
            unsafe { std::ptr::write_bytes(node, 0, NODE_SIZE) };
            return idx;
        }

        // Grow if needed
        if self.len >= self.cap {
            self.grow();
        }

        let idx = self.len;
        self.len += 1;
        // Node is already zeroed from alloc_zeroed or grow
        idx
    }

    /// Return a node to the free list.
    pub fn free_node(&mut self, idx: NodeIdx) {
        let node = self.node_ptr(idx);
        // Write current free_head into the node's first 4 bytes
        unsafe { node.cast::<NodeIdx>().write(self.free_head) };
        self.free_head = idx;
    }

    fn grow(&mut self) {
        let new_cap = if self.cap == 0 {
            Self::INITIAL_CAP
        } else {
            self.cap * 2
        };
        let new_layout = Self::layout(new_cap);

        let new_ptr = if self.ptr.is_null() {
            unsafe { alloc::alloc_zeroed(new_layout) }
        } else {
            let old_layout = Self::layout(self.cap);
            let new_ptr = unsafe { alloc::realloc(self.ptr, old_layout, new_layout.size()) };
            if !new_ptr.is_null() {
                // Zero the new portion
                let old_size = self.cap as usize * NODE_SIZE;
                unsafe {
                    std::ptr::write_bytes(new_ptr.add(old_size), 0, new_layout.size() - old_size);
                }
            }
            new_ptr
        };

        if new_ptr.is_null() {
            alloc::handle_alloc_error(new_layout);
        }

        self.ptr = new_ptr;
        self.cap = new_cap;
    }

    /// Create a byte-for-byte copy of the arena. All node indices remain valid.
    pub fn clone_arena(&self) -> Arena {
        if self.ptr.is_null() {
            return Arena::new();
        }
        let layout = Self::layout(self.cap);
        let new_ptr = unsafe { alloc::alloc(layout) };
        if new_ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        // Copy the used portion
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr, new_ptr, self.len as usize * NODE_SIZE);
        }
        Arena {
            ptr: new_ptr,
            cap: self.cap,
            len: self.len,
            free_head: self.free_head,
        }
    }

    /// Number of allocated node slots (high-water, not accounting for free list).
    pub fn allocated_nodes(&self) -> u32 {
        self.len
    }

    /// Ensure capacity for at least `min_cap` nodes.
    pub fn ensure_capacity(&mut self, min_cap: u32) {
        while self.cap < min_cap {
            self.grow();
        }
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { alloc::dealloc(self.ptr, Self::layout(self.cap)) };
        }
    }
}

// ── PathBuf ─────────────────────────────────────────────────────────────

/// Maximum tree height supported by [`PathBuf`].
///
/// 16 covers any practical (K, V) pair where K + V ≤ 64 bytes up to
/// `u32::MAX` entries (the arena limit). For example: `(String, String)`
/// has LEAF_CAP=5, INTERNAL_CAP=8 → max height ≈ 11 at 4 billion entries.
/// Trees with extreme key sizes that exceed this will hit the
/// `debug_assert!` in `push`.
pub(crate) const MAX_PATH_HEIGHT: usize = 16;

/// Stack-allocated path buffer used during B-tree descent. Records
/// `(parent_node_idx, child_slot_in_parent)` from root → leaf so that a
/// subsequent `propagate_split` can walk back up. Replaces a per-insert
/// `Vec` heap allocation in the hot insert path.
pub(crate) struct PathBuf {
    items: [(NodeIdx, usize); MAX_PATH_HEIGHT],
    len: u32,
}

impl PathBuf {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            items: [(NO_NODE, 0); MAX_PATH_HEIGHT],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn push(&mut self, item: (NodeIdx, usize)) {
        debug_assert!(
            (self.len as usize) < MAX_PATH_HEIGHT,
            "PathBuf overflow: tree height exceeded MAX_PATH_HEIGHT={MAX_PATH_HEIGHT}"
        );
        // SAFETY: debug_assert above guards in debug; release relies on
        // height ≤ MAX_PATH_HEIGHT (checked at insert sites by tree shape).
        unsafe { *self.items.get_unchecked_mut(self.len as usize) = item };
        self.len += 1;
    }

    #[inline(always)]
    pub fn pop(&mut self) -> Option<(NodeIdx, usize)> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        // SAFETY: 0 <= self.len < MAX_PATH_HEIGHT after the decrement.
        Some(unsafe { *self.items.get_unchecked(self.len as usize) })
    }

    /// Reverse the path in place. Used by `path_to_node` which records
    /// leaf→root and needs root→leaf order.
    #[inline]
    pub fn reverse(&mut self) {
        let len = self.len as usize;
        if len > 1 {
            self.items[..len].reverse();
        }
    }

    /// Live slice of recorded path entries in their current (insertion / reverse) order.
    #[inline]
    pub fn as_slice(&self) -> &[(NodeIdx, usize)] {
        &self.items[..self.len as usize]
    }
}

// ── RawBTree ────────────────────────────────────────────────────────────

/// Core B+ tree structure, parameterized by K and V.
pub(crate) struct RawBTree<K, V> {
    pub(crate) arena: Arena,
    pub(crate) root: NodeIdx,
    pub(crate) first_leaf: NodeIdx,
    pub(crate) last_leaf: NodeIdx,
    pub(crate) len: usize,
    pub(crate) height: u32,
    _marker: PhantomData<(K, V)>,
}

impl<K, V> RawBTree<K, V> {
    pub fn new() -> Self {
        NodeLayout::<K, V>::assert_capacities();
        RawBTree {
            arena: Arena::new(),
            root: NO_NODE,
            first_leaf: NO_NODE,
            last_leaf: NO_NODE,
            len: 0,
            height: 0,
            _marker: PhantomData,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        NodeLayout::<K, V>::assert_capacities();
        // Estimate nodes needed: capacity / leaf_cap + some internal nodes
        let leaf_cap = NodeLayout::<K, V>::LEAF_CAP;
        let leaves = capacity.div_ceil(leaf_cap.max(1));
        // Internal nodes are roughly leaves / internal_cap per level; overshoot a bit
        let estimated = (leaves as u32).saturating_add(leaves as u32 / 4).max(4);
        RawBTree {
            arena: Arena::with_capacity(estimated),
            root: NO_NODE,
            first_leaf: NO_NODE,
            last_leaf: NO_NODE,
            len: 0,
            height: 0,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// First non-empty leaf, walking forward from `first_leaf` past any
    /// empty leaves (which can be left behind by an in-place op like a
    /// partial `retain`). Returns None when no non-empty leaf exists.
    pub(crate) fn first_nonempty_leaf(&self) -> Option<NodeIdx> {
        let mut idx = self.first_leaf;
        while idx != NO_NODE {
            let n = self.arena.node_ptr(idx);
            let h = unsafe { NodeLayout::<K, V>::header(n) };
            if h.len > 0 {
                return Some(idx);
            }
            idx = unsafe { NodeLayout::<K, V>::leaf_next_ptr(n).read() };
        }
        None
    }

    /// Last non-empty leaf, walking backward from `last_leaf` past any
    /// empty leaves.
    pub(crate) fn last_nonempty_leaf(&self) -> Option<NodeIdx> {
        let mut idx = self.last_leaf;
        while idx != NO_NODE {
            let n = self.arena.node_ptr(idx);
            let h = unsafe { NodeLayout::<K, V>::header(n) };
            if h.len > 0 {
                return Some(idx);
            }
            idx = unsafe { NodeLayout::<K, V>::leaf_prev_ptr(n).read() };
        }
        None
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Approximate capacity (leaf slots allocated so far).
    pub fn capacity(&self) -> usize {
        self.arena.allocated_nodes() as usize * NodeLayout::<K, V>::LEAF_CAP
    }
}

impl<K: Ord, V> RawBTree<K, V> {
    /// Resolve a range to `(start_leaf, start_idx, end_leaf, end_idx)` where
    /// `end_*` is the exclusive sentinel (NO_NODE = unbounded).
    pub fn resolve_range_bounds<Q, R>(&self, range: R) -> (NodeIdx, usize, NodeIdx, usize)
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
        R: std::ops::RangeBounds<Q>,
    {
        use std::ops::Bound;

        let (start_leaf, start_idx) = match range.start_bound() {
            Bound::Included(key) => self.lower_bound(key).unwrap_or((NO_NODE, 0)),
            Bound::Excluded(key) => {
                if let Some((leaf, idx)) = self.lower_bound(key) {
                    let node = self.arena.node_ptr(leaf);
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
            Bound::Unbounded => (self.first_leaf, 0),
        };

        let (end_leaf, end_idx) = match range.end_bound() {
            Bound::Included(key) => {
                if let Some((leaf, idx)) = self.lower_bound(key) {
                    let node = self.arena.node_ptr(leaf);
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
            Bound::Excluded(key) => self.lower_bound(key).unwrap_or((NO_NODE, 0)),
            Bound::Unbounded => (NO_NODE, 0),
        };

        (start_leaf, start_idx, end_leaf, end_idx)
    }

    /// Search for a key, returning the leaf node index and slot index if found.
    ///
    /// Uses linear scan for shallow trees (better branch prediction in the
    /// hot loop) and binary search for taller trees (the saved comparisons
    /// pay off when more nodes are visited and the working set spills L1).
    /// Crossover height calibrated on `lookup_hit/lookup_miss` benches:
    /// linear wins ~75% at height ≤ 2 (≤ ~1K entries for u64/u64), binary
    /// wins ~20% at height ≥ 3 (≥ ~10K entries).
    #[inline]
    pub fn search<Q>(&self, key: &Q) -> Option<(NodeIdx, usize)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if self.root == NO_NODE {
            return None;
        }
        if self.height < 3 {
            self.search_linear(key)
        } else {
            self.search_binary(key)
        }
    }

    #[inline(always)]
    fn search_linear<Q>(&self, key: &Q) -> Option<(NodeIdx, usize)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut node_idx = self.root;

        for _ in 0..self.height {
            let node = self.arena.node_ptr(node_idx);
            let header = unsafe { NodeLayout::<K, V>::header(node) };
            let len = header.len as usize;

            let mut child_idx = len;
            for i in 0..len {
                let k = unsafe { &*NodeLayout::<K, V>::internal_key_ptr(node, i) };
                if key.cmp(k.borrow()) == std::cmp::Ordering::Less {
                    child_idx = i;
                    break;
                }
            }

            node_idx = unsafe { NodeLayout::<K, V>::internal_child_ptr(node, child_idx).read() };
        }

        let node = self.arena.node_ptr(node_idx);
        let header = unsafe { NodeLayout::<K, V>::header(node) };
        let len = header.len as usize;

        for i in 0..len {
            let k = unsafe { &*NodeLayout::<K, V>::leaf_key_ptr(node, i) };
            match key.cmp(k.borrow()) {
                std::cmp::Ordering::Equal => return Some((node_idx, i)),
                std::cmp::Ordering::Less => return None,
                std::cmp::Ordering::Greater => {}
            }
        }
        None
    }

    #[inline(always)]
    fn search_binary<Q>(&self, key: &Q) -> Option<(NodeIdx, usize)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let mut node_idx = self.root;

        for _ in 0..self.height {
            let node = self.arena.node_ptr(node_idx);
            let header = unsafe { NodeLayout::<K, V>::header(node) };
            let len = header.len as usize;

            let keys = unsafe {
                std::slice::from_raw_parts(NodeLayout::<K, V>::internal_key_ptr(node, 0), len)
            };
            let child_idx =
                keys.partition_point(|k| k.borrow().cmp(key) != std::cmp::Ordering::Greater);

            node_idx = unsafe { NodeLayout::<K, V>::internal_child_ptr(node, child_idx).read() };
        }

        let node = self.arena.node_ptr(node_idx);
        let header = unsafe { NodeLayout::<K, V>::header(node) };
        let len = header.len as usize;
        let keys = unsafe {
            std::slice::from_raw_parts(NodeLayout::<K, V>::leaf_key_ptr(node, 0), len)
        };
        match keys.binary_search_by(|k| k.borrow().cmp(key)) {
            Ok(i) => Some((node_idx, i)),
            Err(_) => None,
        }
    }

    /// O(log n) estimate of the right side's fraction for `split_off(at)`.
    /// Walks the spine top-down; at each level, accumulates contributions
    /// from "pure" left and right children (sized by approximate balanced
    /// subtree weights) and recurses into the boundary subtree. The boundary
    /// itself becomes exact at the leaf level. Used by the `split_off`
    /// dispatcher to pick between drain (large right) and surgical (small right).
    ///
    /// Returns None if the tree is empty.
    pub(crate) fn estimate_right_fraction<Q>(&self, at: &Q) -> Option<f64>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if self.root == NO_NODE {
            return None;
        }

        let leaf_cap = NodeLayout::<K, V>::LEAF_CAP as u64;
        let n_children_full = (NodeLayout::<K, V>::INTERNAL_CAP + 1) as u64;

        // child_size at the current level = approximate entries per child of the
        // current internal node. Starts at leaf_cap × n_children^(height-1) and
        // divides by n_children each descent. Assumes balanced fill; off by a
        // constant factor when nodes are partially full, but the right_fraction
        // ratio largely cancels the error out.
        let h = self.height;
        let mut child_size: u64 = leaf_cap;
        for _ in 0..h.saturating_sub(1) {
            child_size = child_size.saturating_mul(n_children_full);
        }

        let mut left: u64 = 0;
        let mut right: u64 = 0;
        let mut node = self.root;

        loop {
            let header = unsafe { NodeLayout::<K, V>::header(self.arena.node_ptr(node)) };
            let len = header.len as usize;
            if header.is_leaf() {
                if len == 0 {
                    break;
                }
                let keys = unsafe {
                    std::slice::from_raw_parts(
                        NodeLayout::<K, V>::leaf_key_ptr(self.arena.node_ptr(node), 0),
                        len,
                    )
                };
                let pos = keys.partition_point(|k| k.borrow().cmp(at) == std::cmp::Ordering::Less);
                left += pos as u64;
                right += (len - pos) as u64;
                break;
            }
            let keys = unsafe {
                std::slice::from_raw_parts(
                    NodeLayout::<K, V>::internal_key_ptr(self.arena.node_ptr(node), 0),
                    len,
                )
            };
            let child_idx =
                keys.partition_point(|k| k.borrow().cmp(at) == std::cmp::Ordering::Less);
            // In B+ tree, keys[i] = first leaf-key of child[i+1]. If the pivot
            // exactly equals keys[child_idx], the boundary is at the leftmost
            // leaf of child[child_idx+1] — that whole child is entirely right.
            let on_boundary_key = child_idx < len
                && unsafe {
                    keys.get_unchecked(child_idx).borrow().cmp(at) == std::cmp::Ordering::Equal
                };
            if on_boundary_key {
                left += ((child_idx + 1) as u64).saturating_mul(child_size);
                right += ((len - child_idx) as u64).saturating_mul(child_size);
                break;
            }
            // Standard: children[0..child_idx] left, children[child_idx+1..=len] right,
            // children[child_idx] is the boundary subtree — descend into it.
            left += (child_idx as u64).saturating_mul(child_size);
            right += ((len - child_idx) as u64).saturating_mul(child_size);
            node = unsafe {
                NodeLayout::<K, V>::internal_child_ptr(self.arena.node_ptr(node), child_idx).read()
            };
            child_size = (child_size / n_children_full).max(1);
        }

        let total = left + right;
        if total == 0 {
            None
        } else {
            Some((right as f64) / (total as f64))
        }
    }

    /// Find the first (leaf, position) where key >= target.
    /// Returns (leaf_idx, slot_idx) or None if all keys are less than target.
    #[inline]
    pub fn lower_bound<Q>(&self, key: &Q) -> Option<(NodeIdx, usize)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if self.root == NO_NODE {
            return None;
        }

        let mut node_idx = self.root;

        if self.height < 3 {
            // Linear navigation in shallow trees.
            for _ in 0..self.height {
                let node = self.arena.node_ptr(node_idx);
                let header = unsafe { NodeLayout::<K, V>::header(node) };
                let len = header.len as usize;

                let mut child_idx = len;
                for i in 0..len {
                    let k = unsafe { &*NodeLayout::<K, V>::internal_key_ptr(node, i) };
                    if key.cmp(k.borrow()) != std::cmp::Ordering::Greater {
                        child_idx = i;
                        break;
                    }
                }

                node_idx =
                    unsafe { NodeLayout::<K, V>::internal_child_ptr(node, child_idx).read() };
            }
        } else {
            // Binary navigation in taller trees.
            for _ in 0..self.height {
                let node = self.arena.node_ptr(node_idx);
                let header = unsafe { NodeLayout::<K, V>::header(node) };
                let len = header.len as usize;

                let keys = unsafe {
                    std::slice::from_raw_parts(NodeLayout::<K, V>::internal_key_ptr(node, 0), len)
                };
                // child_idx = first i where keys[i] >= target.
                let child_idx =
                    keys.partition_point(|k| k.borrow().cmp(key) == std::cmp::Ordering::Less);

                node_idx =
                    unsafe { NodeLayout::<K, V>::internal_child_ptr(node, child_idx).read() };
            }
        }

        // At leaf: find first key >= target
        let node = self.arena.node_ptr(node_idx);
        let header = unsafe { NodeLayout::<K, V>::header(node) };
        let len = header.len as usize;

        let leaf_keys = unsafe {
            std::slice::from_raw_parts(NodeLayout::<K, V>::leaf_key_ptr(node, 0), len)
        };
        let pos = if self.height < 3 {
            let mut p = len;
            for (i, k) in leaf_keys.iter().enumerate() {
                if key.cmp(k.borrow()) != std::cmp::Ordering::Greater {
                    p = i;
                    break;
                }
            }
            p
        } else {
            leaf_keys.partition_point(|k| k.borrow().cmp(key) == std::cmp::Ordering::Less)
        };
        if pos < len {
            return Some((node_idx, pos));
        }

        // All keys in this leaf are less — check next leaf
        let next = unsafe { NodeLayout::<K, V>::leaf_next_ptr(node).read() };
        if next != NO_NODE {
            let next_node = self.arena.node_ptr(next);
            let next_header = unsafe { NodeLayout::<K, V>::header(next_node) };
            if next_header.len > 0 {
                return Some((next, 0));
            }
        }

        None
    }

    /// Search for the leaf where a key should be inserted.
    /// Returns (leaf_idx, insert_position) where insert_position is the
    /// index at which the key should go to maintain sorted order.
    /// Records the descent path of `(node_idx, child_index)` pairs into
    /// `path` for split propagation.
    fn search_for_insert(&self, key: &K, path: &mut PathBuf) -> (NodeIdx, usize) {
        debug_assert!(self.root != NO_NODE);
        debug_assert!(path.is_empty());

        let mut node_idx = self.root;

        // Navigate internal nodes
        for _ in 0..self.height {
            let node = self.arena.node_ptr(node_idx);
            let header = unsafe { NodeLayout::<K, V>::header(node) };
            let len = header.len as usize;

            let mut child_idx = len;
            for i in 0..len {
                let k = unsafe { &*NodeLayout::<K, V>::internal_key_ptr(node, i) };
                if key < k {
                    child_idx = i;
                    break;
                }
            }

            path.push((node_idx, child_idx));
            node_idx = unsafe { NodeLayout::<K, V>::internal_child_ptr(node, child_idx).read() };
        }

        // At leaf: find insertion position
        let node = self.arena.node_ptr(node_idx);
        let header = unsafe { NodeLayout::<K, V>::header(node) };
        let len = header.len as usize;

        let mut pos = len;
        for i in 0..len {
            let k = unsafe { &*NodeLayout::<K, V>::leaf_key_ptr(node, i) };
            if key <= k {
                pos = i;
                break;
            }
        }

        (node_idx, pos)
    }

    /// Search for a key, returning either the existing location or the
    /// insertion point. The descent path is written into `path` for use
    /// by `insert_at_vacant` if a split cascade is needed.
    pub(crate) fn entry_search(&self, key: &K, path: &mut PathBuf) -> EntrySearch {
        if self.root == NO_NODE {
            return EntrySearch::EmptyTree;
        }

        let (leaf_idx, pos) = self.search_for_insert(key, path);

        // Check if key already exists at this position
        let node = self.arena.node_ptr(leaf_idx);
        let header = unsafe { NodeLayout::<K, V>::header(node) };
        let len = header.len as usize;

        if pos < len {
            let existing_key = unsafe { &*NodeLayout::<K, V>::leaf_key_ptr(node, pos) };
            if *existing_key == *key {
                return EntrySearch::Occupied(leaf_idx, pos);
            }
        }

        EntrySearch::Vacant(leaf_idx, pos)
    }

    /// Insert a value at a pre-located vacant position.
    /// Returns (leaf_idx, slot_idx) of the inserted element.
    /// Used by VacantEntry::insert.
    pub(crate) fn insert_at_vacant(
        &mut self,
        leaf_idx: NodeIdx,
        pos: usize,
        path: &mut PathBuf,
        key: K,
        value: V,
    ) -> (NodeIdx, usize)
    where
        K: Clone,
    {
        let node = self.arena.node_ptr(leaf_idx);
        let len = unsafe { NodeLayout::<K, V>::header(node).len } as usize;

        if len < NodeLayout::<K, V>::LEAF_CAP {
            self.leaf_insert_at(leaf_idx, pos, key, value);
            self.len += 1;
            (leaf_idx, pos)
        } else {
            let mid = (NodeLayout::<K, V>::LEAF_CAP + 1) / 2;
            let (promoted_key, new_leaf_idx) =
                self.leaf_split_and_insert(leaf_idx, pos, key, value);
            self.len += 1;
            self.propagate_split(path, promoted_key, new_leaf_idx);
            // Determine which leaf the element ended up in
            if pos < mid {
                (leaf_idx, pos)
            } else {
                (new_leaf_idx, pos - mid)
            }
        }
    }

    /// Create the first leaf for an empty tree and insert into it.
    /// Used by VacantEntry::insert when tree is empty.
    pub(crate) fn insert_first(&mut self, key: K, value: V) {
        let leaf_idx = self.arena.alloc_node();
        let node = self.arena.node_ptr(leaf_idx);
        unsafe {
            let header = NodeLayout::<K, V>::header_mut(node);
            header.len = 1;
            header.flags = NodeHeader::IS_LEAF;
            header.parent = NO_NODE;
            NodeLayout::<K, V>::leaf_key_ptr(node, 0).write(key);
            NodeLayout::<K, V>::leaf_val_ptr(node, 0).write(value);
            NodeLayout::<K, V>::leaf_prev_ptr(node).write(NO_NODE);
            NodeLayout::<K, V>::leaf_next_ptr(node).write(NO_NODE);
        }
        self.root = leaf_idx;
        self.first_leaf = leaf_idx;
        self.last_leaf = leaf_idx;
        self.len = 1;
    }

    /// Build the path from root to a given node by following parent pointers.
    /// Writes into `path` (which must be empty on entry).
    fn path_to_node(&self, target: NodeIdx, path: &mut PathBuf) {
        debug_assert!(path.is_empty());
        let mut node_idx = target;

        loop {
            let node = self.arena.node_ptr(node_idx);
            let parent = unsafe { NodeLayout::<K, V>::header(node).parent };
            if parent == NO_NODE {
                break;
            }

            let parent_node = self.arena.node_ptr(parent);
            let parent_len = unsafe { NodeLayout::<K, V>::header(parent_node).len } as usize;
            let mut child_slot = parent_len;
            for i in 0..=parent_len {
                let child =
                    unsafe { NodeLayout::<K, V>::internal_child_ptr(parent_node, i).read() };
                if child == node_idx {
                    child_slot = i;
                    break;
                }
            }

            path.push((parent, child_slot));
            node_idx = parent;
        }

        path.reverse();
    }
}

/// Result of an entry search on the B-tree.
///
/// The descent path is *not* embedded in the variants — callers pass a
/// `&mut PathBuf` into `entry_search` and own the path buffer themselves.
/// Keeping this enum small (~16 bytes vs ~280 bytes if PathBuf were
/// embedded) avoids large stack copies on every entry() call.
pub(crate) enum EntrySearch {
    /// Tree is empty.
    EmptyTree,
    /// Key found at (leaf_idx, slot_idx).
    Occupied(NodeIdx, usize),
    /// Key not found. Insert at (leaf_idx, pos). The descent path is in the
    /// caller-provided `PathBuf` passed to `entry_search`.
    Vacant(NodeIdx, usize),
}

impl<K: Ord + Clone, V> RawBTree<K, V> {
    /// Insert a key-value pair. Returns the previous value if the key existed.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Empty tree: create root leaf
        if self.root == NO_NODE {
            let leaf_idx = self.arena.alloc_node();
            let node = self.arena.node_ptr(leaf_idx);
            unsafe {
                let header = NodeLayout::<K, V>::header_mut(node);
                header.len = 1;
                header.flags = NodeHeader::IS_LEAF;
                header.parent = NO_NODE;
                NodeLayout::<K, V>::leaf_key_ptr(node, 0).write(key);
                NodeLayout::<K, V>::leaf_val_ptr(node, 0).write(value);
                NodeLayout::<K, V>::leaf_prev_ptr(node).write(NO_NODE);
                NodeLayout::<K, V>::leaf_next_ptr(node).write(NO_NODE);
            }
            self.root = leaf_idx;
            self.first_leaf = leaf_idx;
            self.last_leaf = leaf_idx;
            self.len = 1;
            return None;
        }

        // Fast path: append to the end (key > all existing keys).
        // This is common for sorted/sequential inserts.
        {
            let last_node = self.arena.node_ptr(self.last_leaf);
            let last_header = unsafe { NodeLayout::<K, V>::header(last_node) };
            let last_len = last_header.len as usize;
            if last_len > 0 {
                let last_key =
                    unsafe { &*NodeLayout::<K, V>::leaf_key_ptr(last_node, last_len - 1) };
                if key > *last_key {
                    // Key goes at the end of the last leaf
                    if last_len < NodeLayout::<K, V>::LEAF_CAP {
                        unsafe {
                            NodeLayout::<K, V>::leaf_key_ptr(last_node, last_len).write(key);
                            NodeLayout::<K, V>::leaf_val_ptr(last_node, last_len).write(value);
                            NodeLayout::<K, V>::header_mut(last_node).len = (last_len + 1) as u16;
                        }
                        self.len += 1;
                        return None;
                    }
                    // Last leaf is full: need split. Build path to last leaf.
                    let mut path = PathBuf::new();
                    self.path_to_node(self.last_leaf, &mut path);
                    let (promoted_key, new_leaf_idx) =
                        self.leaf_split_and_insert(self.last_leaf, last_len, key, value);
                    self.len += 1;
                    self.propagate_split(&mut path, promoted_key, new_leaf_idx);
                    return None;
                }
            }
        }

        // Fast path: prepend to the front (key < all existing keys).
        // This is common for reverse-sorted inserts.
        {
            let first_node = self.arena.node_ptr(self.first_leaf);
            let first_header = unsafe { NodeLayout::<K, V>::header(first_node) };
            let first_len = first_header.len as usize;
            if first_len > 0 {
                let first_key = unsafe { &*NodeLayout::<K, V>::leaf_key_ptr(first_node, 0) };
                if key < *first_key {
                    if first_len < NodeLayout::<K, V>::LEAF_CAP {
                        self.leaf_insert_at(self.first_leaf, 0, key, value);
                        self.len += 1;
                        return None;
                    }
                    let mut path = PathBuf::new();
                    self.path_to_node(self.first_leaf, &mut path);
                    let (promoted_key, new_leaf_idx) =
                        self.leaf_split_and_insert(self.first_leaf, 0, key, value);
                    self.len += 1;
                    self.propagate_split(&mut path, promoted_key, new_leaf_idx);
                    return None;
                }
            }
        }

        let mut path = PathBuf::new();
        let (leaf_idx, pos) = self.search_for_insert(&key, &mut path);

        // Check if key already exists at this position
        let node = self.arena.node_ptr(leaf_idx);
        let header = unsafe { NodeLayout::<K, V>::header(node) };
        let len = header.len as usize;

        if pos < len {
            let existing_key = unsafe { &*NodeLayout::<K, V>::leaf_key_ptr(node, pos) };
            if *existing_key == key {
                // Replace value
                let val_ptr = unsafe { NodeLayout::<K, V>::leaf_val_ptr(node, pos) };
                let old = unsafe { val_ptr.read() };
                unsafe { val_ptr.write(value) };
                return Some(old);
            }
        }

        // Insert into leaf
        if len < NodeLayout::<K, V>::LEAF_CAP {
            self.leaf_insert_at(leaf_idx, pos, key, value);
            self.len += 1;
            None
        } else {
            let (promoted_key, new_leaf_idx) =
                self.leaf_split_and_insert(leaf_idx, pos, key, value);
            self.len += 1;
            self.propagate_split(&mut path, promoted_key, new_leaf_idx);
            None
        }
    }

    /// Insert key+value at position `pos` in a leaf that has room.
    fn leaf_insert_at(&mut self, leaf_idx: NodeIdx, pos: usize, key: K, value: V) {
        let node = self.arena.node_ptr(leaf_idx);
        let header = unsafe { NodeLayout::<K, V>::header_mut(node) };
        let len = header.len as usize;
        debug_assert!(len < NodeLayout::<K, V>::LEAF_CAP);

        // Shift keys and values right
        unsafe {
            for i in (pos..len).rev() {
                let src_k = NodeLayout::<K, V>::leaf_key_ptr(node, i);
                let dst_k = NodeLayout::<K, V>::leaf_key_ptr(node, i + 1);
                std::ptr::copy_nonoverlapping(src_k, dst_k, 1);

                let src_v = NodeLayout::<K, V>::leaf_val_ptr(node, i);
                let dst_v = NodeLayout::<K, V>::leaf_val_ptr(node, i + 1);
                std::ptr::copy_nonoverlapping(src_v, dst_v, 1);
            }

            NodeLayout::<K, V>::leaf_key_ptr(node, pos).write(key);
            NodeLayout::<K, V>::leaf_val_ptr(node, pos).write(value);
        }

        header.len = (len + 1) as u16;
    }

    /// Split a full leaf and insert the new key+value.
    /// Returns (promoted_key, new_right_leaf_idx).
    fn leaf_split_and_insert(
        &mut self,
        left_idx: NodeIdx,
        pos: usize,
        key: K,
        value: V,
    ) -> (K, NodeIdx)
    where
        K: Clone,
    {
        let leaf_cap = NodeLayout::<K, V>::LEAF_CAP;
        let mid = (leaf_cap + 1) / 2;

        // Allocate new right leaf
        let right_idx = self.arena.alloc_node();

        // Read current left leaf state
        let left_node = self.arena.node_ptr(left_idx);
        let old_next = unsafe { NodeLayout::<K, V>::leaf_next_ptr(left_node).read() };
        let left_parent = unsafe { NodeLayout::<K, V>::header(left_node).parent };

        // Collect all keys+values from left leaf, plus the new one at `pos`
        // We work with the indices to know what goes left vs right
        // Left keeps [0..mid), right gets [mid..leaf_cap+1)

        // Initialize right leaf header
        let right_node = self.arena.node_ptr(right_idx);
        unsafe {
            let right_header = NodeLayout::<K, V>::header_mut(right_node);
            right_header.flags = NodeHeader::IS_LEAF;
            right_header.parent = left_parent;
        }

        // Determine how many elements go to each side after insert
        // Total after insert = leaf_cap + 1
        // Left keeps `mid`, right gets `leaf_cap + 1 - mid`
        let right_count = leaf_cap + 1 - mid;

        if pos < mid {
            // New element goes to the left half
            // Move keys[mid-1..leaf_cap) to right[0..right_count)
            // (we lose one from left because the insert will add one)
            let move_start = mid - 1;
            let move_count = leaf_cap - move_start;
            let left_node = self.arena.node_ptr(left_idx);
            let right_node = self.arena.node_ptr(right_idx);
            unsafe {
                for i in 0..move_count {
                    let src_k = NodeLayout::<K, V>::leaf_key_ptr(left_node, move_start + i);
                    let dst_k = NodeLayout::<K, V>::leaf_key_ptr(right_node, i);
                    std::ptr::copy_nonoverlapping(src_k, dst_k, 1);

                    let src_v = NodeLayout::<K, V>::leaf_val_ptr(left_node, move_start + i);
                    let dst_v = NodeLayout::<K, V>::leaf_val_ptr(right_node, i);
                    std::ptr::copy_nonoverlapping(src_v, dst_v, 1);
                }
            }

            // Update lengths
            let left_node = self.arena.node_ptr(left_idx);
            unsafe {
                NodeLayout::<K, V>::header_mut(left_node).len = (mid - 1) as u16;
            }
            let right_node = self.arena.node_ptr(right_idx);
            unsafe {
                NodeLayout::<K, V>::header_mut(right_node).len = move_count as u16;
            }

            // Now insert into left leaf (which has mid-1 elements, room for one more)
            self.leaf_insert_at(left_idx, pos, key, value);
        } else {
            // New element goes to the right half
            // Move keys[mid..leaf_cap) to right, inserting the new element at the right position
            let right_pos = pos - mid;
            let left_node = self.arena.node_ptr(left_idx);
            let right_node = self.arena.node_ptr(right_idx);

            unsafe {
                // Copy elements before the insertion point
                for i in 0..right_pos {
                    let src_k = NodeLayout::<K, V>::leaf_key_ptr(left_node, mid + i);
                    let dst_k = NodeLayout::<K, V>::leaf_key_ptr(right_node, i);
                    std::ptr::copy_nonoverlapping(src_k, dst_k, 1);

                    let src_v = NodeLayout::<K, V>::leaf_val_ptr(left_node, mid + i);
                    let dst_v = NodeLayout::<K, V>::leaf_val_ptr(right_node, i);
                    std::ptr::copy_nonoverlapping(src_v, dst_v, 1);
                }

                // Write the new element
                NodeLayout::<K, V>::leaf_key_ptr(right_node, right_pos).write(key);
                NodeLayout::<K, V>::leaf_val_ptr(right_node, right_pos).write(value);

                // Copy elements after the insertion point
                for i in right_pos..(leaf_cap - mid) {
                    let src_k = NodeLayout::<K, V>::leaf_key_ptr(left_node, mid + i);
                    let dst_k = NodeLayout::<K, V>::leaf_key_ptr(right_node, i + 1);
                    std::ptr::copy_nonoverlapping(src_k, dst_k, 1);

                    let src_v = NodeLayout::<K, V>::leaf_val_ptr(left_node, mid + i);
                    let dst_v = NodeLayout::<K, V>::leaf_val_ptr(right_node, i + 1);
                    std::ptr::copy_nonoverlapping(src_v, dst_v, 1);
                }
            }

            // Update lengths
            let left_node = self.arena.node_ptr(left_idx);
            unsafe {
                NodeLayout::<K, V>::header_mut(left_node).len = mid as u16;
            }
            let right_node = self.arena.node_ptr(right_idx);
            unsafe {
                NodeLayout::<K, V>::header_mut(right_node).len = right_count as u16;
            }
        }

        // Wire leaf chain: left <-> right <-> old_next
        let left_node = self.arena.node_ptr(left_idx);
        let right_node = self.arena.node_ptr(right_idx);
        unsafe {
            NodeLayout::<K, V>::leaf_next_ptr(left_node).write(right_idx);
            NodeLayout::<K, V>::leaf_prev_ptr(right_node).write(left_idx);
            NodeLayout::<K, V>::leaf_next_ptr(right_node).write(old_next);
        }
        if old_next != NO_NODE {
            let old_next_node = self.arena.node_ptr(old_next);
            unsafe {
                NodeLayout::<K, V>::leaf_prev_ptr(old_next_node).write(right_idx);
            }
        } else {
            self.last_leaf = right_idx;
        }

        // Promoted key = first key of right leaf (clone, since B+ tree keeps it in the leaf)
        let right_node = self.arena.node_ptr(right_idx);
        let promoted = unsafe { (*NodeLayout::<K, V>::leaf_key_ptr(right_node, 0)).clone() };

        (promoted, right_idx)
    }

    /// Propagate a split upward from child to parent(s).
    fn propagate_split(
        &mut self,
        path: &mut PathBuf,
        mut key: K,
        mut new_child: NodeIdx,
    ) where
        K: Clone,
    {
        while let Some((parent_idx, child_pos)) = path.pop() {
            let parent_node = self.arena.node_ptr(parent_idx);
            let parent_header = unsafe { NodeLayout::<K, V>::header(parent_node) };
            let parent_len = parent_header.len as usize;

            if parent_len < NodeLayout::<K, V>::INTERNAL_CAP {
                // Room in parent: insert key and child pointer
                self.internal_insert_at(parent_idx, child_pos, key, new_child);
                // Update the new child's parent pointer
                let child_node = self.arena.node_ptr(new_child);
                unsafe {
                    NodeLayout::<K, V>::header_mut(child_node).parent = parent_idx;
                }
                return;
            }

            // Parent is full: split it
            let (promoted, new_internal) =
                self.internal_split_and_insert(parent_idx, child_pos, key, new_child);
            key = promoted;
            new_child = new_internal;
        }

        // We've split all the way to the root: create a new root
        let new_root = self.arena.alloc_node();
        let new_root_node = self.arena.node_ptr(new_root);
        unsafe {
            let header = NodeLayout::<K, V>::header_mut(new_root_node);
            header.len = 1;
            header.flags = 0; // internal
            header.parent = NO_NODE;

            NodeLayout::<K, V>::internal_key_ptr(new_root_node, 0).write(key);
            NodeLayout::<K, V>::internal_child_ptr(new_root_node, 0).write(self.root);
            NodeLayout::<K, V>::internal_child_ptr(new_root_node, 1).write(new_child);
        }

        // Update old root's and new child's parent
        let old_root_node = self.arena.node_ptr(self.root);
        unsafe {
            NodeLayout::<K, V>::header_mut(old_root_node).parent = new_root;
        }
        let new_child_node = self.arena.node_ptr(new_child);
        unsafe {
            NodeLayout::<K, V>::header_mut(new_child_node).parent = new_root;
        }

        self.root = new_root;
        self.height += 1;
    }

    /// Insert a key and right-child at position `pos` in an internal node that has room.
    fn internal_insert_at(&mut self, node_idx: NodeIdx, pos: usize, key: K, right_child: NodeIdx) {
        let node = self.arena.node_ptr(node_idx);
        let header = unsafe { NodeLayout::<K, V>::header_mut(node) };
        let len = header.len as usize;
        debug_assert!(len < NodeLayout::<K, V>::INTERNAL_CAP);

        unsafe {
            // Shift keys right
            for i in (pos..len).rev() {
                let src = NodeLayout::<K, V>::internal_key_ptr(node, i);
                let dst = NodeLayout::<K, V>::internal_key_ptr(node, i + 1);
                std::ptr::copy_nonoverlapping(src, dst, 1);
            }
            NodeLayout::<K, V>::internal_key_ptr(node, pos).write(key);

            // Shift children right (children are at positions pos+1..=len, shift to pos+2..=len+1)
            for i in (pos + 1..=len).rev() {
                let src = NodeLayout::<K, V>::internal_child_ptr(node, i);
                let dst = NodeLayout::<K, V>::internal_child_ptr(node, i + 1);
                std::ptr::copy_nonoverlapping(src, dst, 1);
            }
            NodeLayout::<K, V>::internal_child_ptr(node, pos + 1).write(right_child);
        }

        header.len = (len + 1) as u16;
    }

    /// Split a full internal node and insert a key + right_child.
    /// Returns (promoted_key, new_right_internal_idx).
    #[allow(clippy::needless_range_loop)]
    fn internal_split_and_insert(
        &mut self,
        left_idx: NodeIdx,
        pos: usize,
        key: K,
        right_child: NodeIdx,
    ) -> (K, NodeIdx) {
        let cap = NodeLayout::<K, V>::INTERNAL_CAP;
        let mid = cap / 2;

        // Allocate right internal node
        let right_idx = self.arena.alloc_node();

        let left_node = self.arena.node_ptr(left_idx);
        let left_parent = unsafe { NodeLayout::<K, V>::header(left_node).parent };

        let right_node = self.arena.node_ptr(right_idx);
        unsafe {
            let header = NodeLayout::<K, V>::header_mut(right_node);
            header.flags = 0; // internal
            header.parent = left_parent;
        }

        // We have `cap` keys + 1 new key to distribute:
        // Left gets keys[0..mid), promoted = keys[mid], right gets keys[mid+1..cap] + new key at pos
        // This is complex, so we use a temporary buffer approach for correctness.

        // Collect all cap+1 keys and cap+2 children into temp arrays
        // (We use Vec here for simplicity; this is a cold path)
        let mut all_keys: Vec<K> = Vec::with_capacity(cap + 1);
        let mut all_children: Vec<NodeIdx> = Vec::with_capacity(cap + 2);

        let left_node = self.arena.node_ptr(left_idx);
        unsafe {
            // Collect keys, inserting new key at `pos`
            for i in 0..pos {
                all_keys.push(NodeLayout::<K, V>::internal_key_ptr(left_node, i).read());
            }
            all_keys.push(key);
            for i in pos..cap {
                all_keys.push(NodeLayout::<K, V>::internal_key_ptr(left_node, i).read());
            }

            // Collect children, inserting new child at `pos + 1`
            for i in 0..=pos {
                all_children.push(NodeLayout::<K, V>::internal_child_ptr(left_node, i).read());
            }
            all_children.push(right_child);
            for i in (pos + 1)..=cap {
                all_children.push(NodeLayout::<K, V>::internal_child_ptr(left_node, i).read());
            }
        }

        debug_assert_eq!(all_keys.len(), cap + 1);
        debug_assert_eq!(all_children.len(), cap + 2);

        // Distribute: left[0..mid], promoted = all_keys[mid], right[mid+1..]
        let promoted = unsafe { std::ptr::read(&all_keys[mid]) };
        let right_key_count = cap - mid; // cap+1 total - mid left - 1 promoted

        // Write left side
        let left_node = self.arena.node_ptr(left_idx);
        unsafe {
            for i in 0..mid {
                NodeLayout::<K, V>::internal_key_ptr(left_node, i)
                    .write(std::ptr::read(&all_keys[i]));
            }
            for i in 0..=mid {
                NodeLayout::<K, V>::internal_child_ptr(left_node, i)
                    .write(std::ptr::read(&all_children[i]));
            }
            NodeLayout::<K, V>::header_mut(left_node).len = mid as u16;
        }

        // Write right side
        let right_node = self.arena.node_ptr(right_idx);
        unsafe {
            for i in 0..right_key_count {
                NodeLayout::<K, V>::internal_key_ptr(right_node, i)
                    .write(std::ptr::read(&all_keys[mid + 1 + i]));
            }
            for i in 0..=right_key_count {
                let child = std::ptr::read(&all_children[mid + 1 + i]);
                NodeLayout::<K, V>::internal_child_ptr(right_node, i).write(child);
                // Update child's parent pointer
                let child_node = self.arena.node_ptr(child);
                NodeLayout::<K, V>::header_mut(child_node).parent = right_idx;
            }
            NodeLayout::<K, V>::header_mut(right_node).len = right_key_count as u16;
        }

        // Prevent Vec from dropping the moved-out elements
        unsafe {
            all_keys.set_len(0);
            all_children.set_len(0);
        }

        (promoted, right_idx)
    }
}

impl<K: Ord, V> RawBTree<K, V> {
    /// Get a reference to the value for a key (O(log n), requires Q: Ord).
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (leaf_idx, slot_idx) = self.search(key)?;
        let node = self.arena.node_ptr(leaf_idx);
        Some(unsafe { &*NodeLayout::<K, V>::leaf_val_ptr(node, slot_idx) })
    }

    /// Get a mutable reference to the value for a key (O(log n), requires Q: Ord).
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (leaf_idx, slot_idx) = self.search(key)?;
        let node = self.arena.node_ptr(leaf_idx);
        Some(unsafe { &mut *NodeLayout::<K, V>::leaf_val_ptr(node, slot_idx) })
    }

    /// Get by equality only (O(n) leaf scan). Used by Map trait impl.
    pub fn get_by_eq<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let mut leaf_idx = self.first_leaf;
        while leaf_idx != NO_NODE {
            let node = self.arena.node_ptr(leaf_idx);
            let header = unsafe { NodeLayout::<K, V>::header(node) };
            let len = header.len as usize;

            for i in 0..len {
                let k = unsafe { &*NodeLayout::<K, V>::leaf_key_ptr(node, i) };
                if k.borrow() == key {
                    return Some(unsafe { &*NodeLayout::<K, V>::leaf_val_ptr(node, i) });
                }
            }

            leaf_idx = unsafe { NodeLayout::<K, V>::leaf_next_ptr(node).read() };
        }
        None
    }

    /// Get key-value pair by equality only (O(n) leaf scan). Used by Map trait impl.
    pub fn get_key_value_by_eq<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let mut leaf_idx = self.first_leaf;
        while leaf_idx != NO_NODE {
            let node = self.arena.node_ptr(leaf_idx);
            let header = unsafe { NodeLayout::<K, V>::header(node) };
            let len = header.len as usize;

            for i in 0..len {
                let k = unsafe { &*NodeLayout::<K, V>::leaf_key_ptr(node, i) };
                if k.borrow() == key {
                    let v = unsafe { &*NodeLayout::<K, V>::leaf_val_ptr(node, i) };
                    return Some((k, v));
                }
            }

            leaf_idx = unsafe { NodeLayout::<K, V>::leaf_next_ptr(node).read() };
        }
        None
    }

    /// Get mutable by equality only (O(n) leaf scan). Used by Map trait impl.
    pub fn get_mut_by_eq<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let mut leaf_idx = self.first_leaf;
        while leaf_idx != NO_NODE {
            let node = self.arena.node_ptr(leaf_idx);
            let header = unsafe { NodeLayout::<K, V>::header(node) };
            let len = header.len as usize;

            for i in 0..len {
                let k = unsafe { &*NodeLayout::<K, V>::leaf_key_ptr(node, i) };
                if k.borrow() == key {
                    return Some(unsafe { &mut *NodeLayout::<K, V>::leaf_val_ptr(node, i) });
                }
            }

            leaf_idx = unsafe { NodeLayout::<K, V>::leaf_next_ptr(node).read() };
        }
        None
    }
}

impl<K: Ord + Clone, V> RawBTree<K, V> {
    /// Remove by equality (O(n) leaf scan). Used by Map trait impl.
    pub fn remove_by_eq<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.remove_entry_by_eq(key).map(|(_, v)| v)
    }

    /// Remove entry by equality (O(n) leaf scan). Used by Map trait impl.
    pub fn remove_entry_by_eq<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        let mut leaf_idx = self.first_leaf;
        while leaf_idx != NO_NODE {
            let node = self.arena.node_ptr(leaf_idx);
            let header = unsafe { NodeLayout::<K, V>::header(node) };
            let len = header.len as usize;

            for i in 0..len {
                let k = unsafe { &*NodeLayout::<K, V>::leaf_key_ptr(node, i) };
                if k.borrow() == key {
                    return Some(self.leaf_remove_at(leaf_idx, i));
                }
            }

            leaf_idx = unsafe { NodeLayout::<K, V>::leaf_next_ptr(node).read() };
        }
        None
    }

    /// Remove a key by Ord search (O(log n)).
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (leaf_idx, slot_idx) = self.search(key)?;
        let (_, v) = self.leaf_remove_at(leaf_idx, slot_idx);
        Some(v)
    }

    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        let (leaf_idx, slot_idx) = self.search(key)?;
        Some(self.leaf_remove_at(leaf_idx, slot_idx))
    }

    /// Remove the element at position `idx` in a leaf, then rebalance if needed.
    pub(crate) fn leaf_remove_at(&mut self, leaf_idx: NodeIdx, idx: usize) -> (K, V) {
        let node = self.arena.node_ptr(leaf_idx);
        let header = unsafe { NodeLayout::<K, V>::header_mut(node) };
        let len = header.len as usize;
        debug_assert!(idx < len);

        let kv = unsafe {
            let key = NodeLayout::<K, V>::leaf_key_ptr(node, idx).read();
            let value = NodeLayout::<K, V>::leaf_val_ptr(node, idx).read();

            for i in idx..len - 1 {
                let src_k = NodeLayout::<K, V>::leaf_key_ptr(node, i + 1);
                let dst_k = NodeLayout::<K, V>::leaf_key_ptr(node, i);
                std::ptr::copy_nonoverlapping(src_k, dst_k, 1);

                let src_v = NodeLayout::<K, V>::leaf_val_ptr(node, i + 1);
                let dst_v = NodeLayout::<K, V>::leaf_val_ptr(node, i);
                std::ptr::copy_nonoverlapping(src_v, dst_v, 1);
            }

            header.len = (len - 1) as u16;
            self.len -= 1;
            (key, value)
        };

        // Rebalance if underflow (skip for root leaf or single-node tree)
        let new_len = len - 1;
        let min_keys = NodeLayout::<K, V>::LEAF_CAP / 2;
        if new_len < min_keys && self.height > 0 {
            self.rebalance_leaf(leaf_idx);
        }

        kv
    }

    /// Rebalance a leaf that has fewer than LEAF_CAP/2 elements.
    fn rebalance_leaf(&mut self, leaf_idx: NodeIdx)
    where
        K: Clone,
    {
        let node = self.arena.node_ptr(leaf_idx);
        let parent_idx = unsafe { NodeLayout::<K, V>::header(node).parent };
        if parent_idx == NO_NODE {
            return; // root leaf, nothing to rebalance
        }

        let leaf_len = unsafe { NodeLayout::<K, V>::header(node).len } as usize;
        let min_keys = NodeLayout::<K, V>::LEAF_CAP / 2;

        // Find our position in parent
        let parent_node = self.arena.node_ptr(parent_idx);
        let parent_len = unsafe { NodeLayout::<K, V>::header(parent_node).len } as usize;
        let mut child_pos = 0;
        for i in 0..=parent_len {
            if unsafe { NodeLayout::<K, V>::internal_child_ptr(parent_node, i).read() } == leaf_idx
            {
                child_pos = i;
                break;
            }
        }

        // Try steal from right sibling
        if child_pos < parent_len {
            let right_idx = unsafe {
                NodeLayout::<K, V>::internal_child_ptr(parent_node, child_pos + 1).read()
            };
            let right_node = self.arena.node_ptr(right_idx);
            let right_len = unsafe { NodeLayout::<K, V>::header(right_node).len } as usize;

            if right_len > min_keys {
                // Steal first element from right sibling
                unsafe {
                    let node = self.arena.node_ptr(leaf_idx);
                    // Append right's first key+value to our end
                    let right_node = self.arena.node_ptr(right_idx);
                    let stolen_k = NodeLayout::<K, V>::leaf_key_ptr(right_node, 0).read();
                    let stolen_v = NodeLayout::<K, V>::leaf_val_ptr(right_node, 0).read();
                    NodeLayout::<K, V>::leaf_key_ptr(node, leaf_len).write(stolen_k);
                    NodeLayout::<K, V>::leaf_val_ptr(node, leaf_len).write(stolen_v);
                    NodeLayout::<K, V>::header_mut(node).len = (leaf_len + 1) as u16;

                    // Shift right sibling left
                    let right_node = self.arena.node_ptr(right_idx);
                    for i in 0..right_len - 1 {
                        let src_k = NodeLayout::<K, V>::leaf_key_ptr(right_node, i + 1);
                        let dst_k = NodeLayout::<K, V>::leaf_key_ptr(right_node, i);
                        std::ptr::copy_nonoverlapping(src_k, dst_k, 1);
                        let src_v = NodeLayout::<K, V>::leaf_val_ptr(right_node, i + 1);
                        let dst_v = NodeLayout::<K, V>::leaf_val_ptr(right_node, i);
                        std::ptr::copy_nonoverlapping(src_v, dst_v, 1);
                    }
                    NodeLayout::<K, V>::header_mut(right_node).len = (right_len - 1) as u16;

                    // Update separator key in parent (= new first key of right)
                    let right_node = self.arena.node_ptr(right_idx);
                    let new_sep = (*NodeLayout::<K, V>::leaf_key_ptr(right_node, 0)).clone();
                    let parent_node = self.arena.node_ptr(parent_idx);
                    let old_sep =
                        NodeLayout::<K, V>::internal_key_ptr(parent_node, child_pos).read();
                    drop(old_sep);
                    NodeLayout::<K, V>::internal_key_ptr(parent_node, child_pos).write(new_sep);
                }
                return;
            }
        }

        // Try steal from left sibling
        if child_pos > 0 {
            let left_idx = unsafe {
                NodeLayout::<K, V>::internal_child_ptr(parent_node, child_pos - 1).read()
            };
            let left_node = self.arena.node_ptr(left_idx);
            let left_len = unsafe { NodeLayout::<K, V>::header(left_node).len } as usize;

            if left_len > min_keys {
                unsafe {
                    let node = self.arena.node_ptr(leaf_idx);
                    // Shift our elements right to make room at position 0
                    for i in (0..leaf_len).rev() {
                        let src_k = NodeLayout::<K, V>::leaf_key_ptr(node, i);
                        let dst_k = NodeLayout::<K, V>::leaf_key_ptr(node, i + 1);
                        std::ptr::copy_nonoverlapping(src_k, dst_k, 1);
                        let src_v = NodeLayout::<K, V>::leaf_val_ptr(node, i);
                        let dst_v = NodeLayout::<K, V>::leaf_val_ptr(node, i + 1);
                        std::ptr::copy_nonoverlapping(src_v, dst_v, 1);
                    }

                    // Move last element from left sibling to our position 0
                    let left_node = self.arena.node_ptr(left_idx);
                    let stolen_k = NodeLayout::<K, V>::leaf_key_ptr(left_node, left_len - 1).read();
                    let stolen_v = NodeLayout::<K, V>::leaf_val_ptr(left_node, left_len - 1).read();
                    let node = self.arena.node_ptr(leaf_idx);
                    NodeLayout::<K, V>::leaf_key_ptr(node, 0).write(stolen_k);
                    NodeLayout::<K, V>::leaf_val_ptr(node, 0).write(stolen_v);
                    NodeLayout::<K, V>::header_mut(node).len = (leaf_len + 1) as u16;

                    let left_node = self.arena.node_ptr(left_idx);
                    NodeLayout::<K, V>::header_mut(left_node).len = (left_len - 1) as u16;

                    // Update separator key in parent (= new first key of us)
                    let node = self.arena.node_ptr(leaf_idx);
                    let new_sep = (*NodeLayout::<K, V>::leaf_key_ptr(node, 0)).clone();
                    let parent_node = self.arena.node_ptr(parent_idx);
                    let old_sep =
                        NodeLayout::<K, V>::internal_key_ptr(parent_node, child_pos - 1).read();
                    drop(old_sep);
                    NodeLayout::<K, V>::internal_key_ptr(parent_node, child_pos - 1).write(new_sep);
                }
                return;
            }
        }

        // Neither sibling can donate: merge with a sibling
        if child_pos < parent_len {
            // Merge with right sibling (move right's elements into us)
            self.merge_leaves(leaf_idx, child_pos, parent_idx);
        } else {
            // Merge with left sibling (move our elements into left)
            let left_idx = unsafe {
                NodeLayout::<K, V>::internal_child_ptr(
                    self.arena.node_ptr(parent_idx),
                    child_pos - 1,
                )
                .read()
            };
            self.merge_leaves(left_idx, child_pos - 1, parent_idx);
        }
    }

    /// Merge leaf at `left_child_pos` with the leaf at `left_child_pos + 1`.
    /// Removes the separator key from parent.
    fn merge_leaves(&mut self, left_idx: NodeIdx, left_child_pos: usize, parent_idx: NodeIdx)
    where
        K: Clone,
    {
        let parent_node = self.arena.node_ptr(parent_idx);
        let right_idx = unsafe {
            NodeLayout::<K, V>::internal_child_ptr(parent_node, left_child_pos + 1).read()
        };

        let left_node = self.arena.node_ptr(left_idx);
        let left_len = unsafe { NodeLayout::<K, V>::header(left_node).len } as usize;
        let right_node = self.arena.node_ptr(right_idx);
        let right_len = unsafe { NodeLayout::<K, V>::header(right_node).len } as usize;

        // Move all elements from right into left
        unsafe {
            let left_node = self.arena.node_ptr(left_idx);
            let right_node = self.arena.node_ptr(right_idx);
            for i in 0..right_len {
                let k = NodeLayout::<K, V>::leaf_key_ptr(right_node, i).read();
                let v = NodeLayout::<K, V>::leaf_val_ptr(right_node, i).read();
                NodeLayout::<K, V>::leaf_key_ptr(left_node, left_len + i).write(k);
                NodeLayout::<K, V>::leaf_val_ptr(left_node, left_len + i).write(v);
            }
            NodeLayout::<K, V>::header_mut(left_node).len = (left_len + right_len) as u16;
        }

        // Update leaf chain: left.next = right.next
        let right_next =
            unsafe { NodeLayout::<K, V>::leaf_next_ptr(self.arena.node_ptr(right_idx)).read() };
        unsafe {
            NodeLayout::<K, V>::leaf_next_ptr(self.arena.node_ptr(left_idx)).write(right_next);
        }
        if right_next != NO_NODE {
            unsafe {
                NodeLayout::<K, V>::leaf_prev_ptr(self.arena.node_ptr(right_next)).write(left_idx);
            }
        } else {
            self.last_leaf = left_idx;
        }

        // Free the right leaf
        self.arena.free_node(right_idx);

        // Remove separator key + right child pointer from parent
        self.internal_remove_at(parent_idx, left_child_pos);
    }

    /// Remove key at `idx` and child at `idx + 1` from an internal node.
    /// Then rebalance the internal node if needed.
    fn internal_remove_at(&mut self, node_idx: NodeIdx, idx: usize)
    where
        K: Clone,
    {
        let node = self.arena.node_ptr(node_idx);
        let header = unsafe { NodeLayout::<K, V>::header_mut(node) };
        let len = header.len as usize;

        unsafe {
            // Drop the separator key
            std::ptr::drop_in_place(NodeLayout::<K, V>::internal_key_ptr(node, idx));

            // Shift keys left
            for i in idx..len - 1 {
                let src = NodeLayout::<K, V>::internal_key_ptr(node, i + 1);
                let dst = NodeLayout::<K, V>::internal_key_ptr(node, i);
                std::ptr::copy_nonoverlapping(src, dst, 1);
            }

            // Shift children left (remove child at idx + 1)
            for i in (idx + 1)..len {
                let src = NodeLayout::<K, V>::internal_child_ptr(node, i + 1);
                let dst = NodeLayout::<K, V>::internal_child_ptr(node, i);
                std::ptr::copy_nonoverlapping(src, dst, 1);
            }

            header.len = (len - 1) as u16;
        }

        // If root becomes empty (0 keys, 1 child), collapse
        let new_len = len - 1;
        if node_idx == self.root && new_len == 0 {
            let only_child = unsafe { NodeLayout::<K, V>::internal_child_ptr(node, 0).read() };
            let child_node = self.arena.node_ptr(only_child);
            unsafe {
                NodeLayout::<K, V>::header_mut(child_node).parent = NO_NODE;
            }
            self.arena.free_node(self.root);
            self.root = only_child;
            self.height -= 1;
            return;
        }

        // Rebalance internal node if underflow
        let min_keys = NodeLayout::<K, V>::INTERNAL_CAP / 2;
        if new_len < min_keys && node_idx != self.root {
            self.rebalance_internal(node_idx);
        }
    }

    /// Rebalance an internal node that has fewer than INTERNAL_CAP/2 keys.
    fn rebalance_internal(&mut self, node_idx: NodeIdx)
    where
        K: Clone,
    {
        let node = self.arena.node_ptr(node_idx);
        let parent_idx = unsafe { NodeLayout::<K, V>::header(node).parent };
        if parent_idx == NO_NODE {
            return;
        }

        let node_len = unsafe { NodeLayout::<K, V>::header(node).len } as usize;
        let min_keys = NodeLayout::<K, V>::INTERNAL_CAP / 2;

        // Find our position in parent
        let parent_node = self.arena.node_ptr(parent_idx);
        let parent_len = unsafe { NodeLayout::<K, V>::header(parent_node).len } as usize;
        let mut child_pos = 0;
        for i in 0..=parent_len {
            if unsafe { NodeLayout::<K, V>::internal_child_ptr(parent_node, i).read() } == node_idx
            {
                child_pos = i;
                break;
            }
        }

        // Try steal from right sibling
        if child_pos < parent_len {
            let right_idx = unsafe {
                NodeLayout::<K, V>::internal_child_ptr(parent_node, child_pos + 1).read()
            };
            let right_len =
                unsafe { NodeLayout::<K, V>::header(self.arena.node_ptr(right_idx)).len } as usize;

            if right_len > min_keys {
                unsafe {
                    // Rotate: parent separator → end of us, right's first key → parent separator
                    let parent_node = self.arena.node_ptr(parent_idx);
                    let sep_key =
                        NodeLayout::<K, V>::internal_key_ptr(parent_node, child_pos).read();

                    let node = self.arena.node_ptr(node_idx);
                    NodeLayout::<K, V>::internal_key_ptr(node, node_len).write(sep_key);

                    // Move right's first child to our end
                    let right_node = self.arena.node_ptr(right_idx);
                    let moved_child = NodeLayout::<K, V>::internal_child_ptr(right_node, 0).read();
                    let node = self.arena.node_ptr(node_idx);
                    NodeLayout::<K, V>::internal_child_ptr(node, node_len + 1).write(moved_child);
                    NodeLayout::<K, V>::header_mut(node).len = (node_len + 1) as u16;

                    // Update moved child's parent
                    NodeLayout::<K, V>::header_mut(self.arena.node_ptr(moved_child)).parent =
                        node_idx;

                    // Right's first key becomes new parent separator
                    let right_node = self.arena.node_ptr(right_idx);
                    let new_sep = NodeLayout::<K, V>::internal_key_ptr(right_node, 0).read();
                    let parent_node = self.arena.node_ptr(parent_idx);
                    NodeLayout::<K, V>::internal_key_ptr(parent_node, child_pos).write(new_sep);

                    // Shift right's keys and children left
                    let right_node = self.arena.node_ptr(right_idx);
                    for i in 0..right_len - 1 {
                        let src = NodeLayout::<K, V>::internal_key_ptr(right_node, i + 1);
                        let dst = NodeLayout::<K, V>::internal_key_ptr(right_node, i);
                        std::ptr::copy_nonoverlapping(src, dst, 1);
                    }
                    for i in 0..right_len {
                        let src = NodeLayout::<K, V>::internal_child_ptr(right_node, i + 1);
                        let dst = NodeLayout::<K, V>::internal_child_ptr(right_node, i);
                        std::ptr::copy_nonoverlapping(src, dst, 1);
                    }
                    NodeLayout::<K, V>::header_mut(right_node).len = (right_len - 1) as u16;
                }
                return;
            }
        }

        // Try steal from left sibling
        if child_pos > 0 {
            let left_idx = unsafe {
                NodeLayout::<K, V>::internal_child_ptr(parent_node, child_pos - 1).read()
            };
            let left_len =
                unsafe { NodeLayout::<K, V>::header(self.arena.node_ptr(left_idx)).len } as usize;

            if left_len > min_keys {
                unsafe {
                    // Rotate: parent separator → start of us, left's last key → parent separator
                    let parent_node = self.arena.node_ptr(parent_idx);
                    let sep_key =
                        NodeLayout::<K, V>::internal_key_ptr(parent_node, child_pos - 1).read();

                    // Shift our keys and children right
                    let node = self.arena.node_ptr(node_idx);
                    for i in (0..node_len).rev() {
                        let src = NodeLayout::<K, V>::internal_key_ptr(node, i);
                        let dst = NodeLayout::<K, V>::internal_key_ptr(node, i + 1);
                        std::ptr::copy_nonoverlapping(src, dst, 1);
                    }
                    for i in (0..=node_len).rev() {
                        let src = NodeLayout::<K, V>::internal_child_ptr(node, i);
                        let dst = NodeLayout::<K, V>::internal_child_ptr(node, i + 1);
                        std::ptr::copy_nonoverlapping(src, dst, 1);
                    }
                    NodeLayout::<K, V>::internal_key_ptr(node, 0).write(sep_key);

                    // Move left's last child to our position 0
                    let left_node = self.arena.node_ptr(left_idx);
                    let moved_child =
                        NodeLayout::<K, V>::internal_child_ptr(left_node, left_len).read();
                    let node = self.arena.node_ptr(node_idx);
                    NodeLayout::<K, V>::internal_child_ptr(node, 0).write(moved_child);
                    NodeLayout::<K, V>::header_mut(node).len = (node_len + 1) as u16;

                    // Update moved child's parent
                    NodeLayout::<K, V>::header_mut(self.arena.node_ptr(moved_child)).parent =
                        node_idx;

                    // Left's last key becomes new parent separator
                    let left_node = self.arena.node_ptr(left_idx);
                    let new_sep =
                        NodeLayout::<K, V>::internal_key_ptr(left_node, left_len - 1).read();
                    NodeLayout::<K, V>::header_mut(left_node).len = (left_len - 1) as u16;
                    let parent_node = self.arena.node_ptr(parent_idx);
                    NodeLayout::<K, V>::internal_key_ptr(parent_node, child_pos - 1).write(new_sep);
                }
                return;
            }
        }

        // Neither sibling can donate: merge
        if child_pos < parent_len {
            self.merge_internals(node_idx, child_pos, parent_idx);
        } else {
            let left_idx = unsafe {
                NodeLayout::<K, V>::internal_child_ptr(
                    self.arena.node_ptr(parent_idx),
                    child_pos - 1,
                )
                .read()
            };
            self.merge_internals(left_idx, child_pos - 1, parent_idx);
        }
    }

    /// Merge internal node at `left_child_pos` with node at `left_child_pos + 1`.
    fn merge_internals(&mut self, left_idx: NodeIdx, left_child_pos: usize, parent_idx: NodeIdx)
    where
        K: Clone,
    {
        let parent_node = self.arena.node_ptr(parent_idx);
        let right_idx = unsafe {
            NodeLayout::<K, V>::internal_child_ptr(parent_node, left_child_pos + 1).read()
        };

        let left_len =
            unsafe { NodeLayout::<K, V>::header(self.arena.node_ptr(left_idx)).len } as usize;
        let right_len =
            unsafe { NodeLayout::<K, V>::header(self.arena.node_ptr(right_idx)).len } as usize;

        unsafe {
            // Pull separator key from parent into left
            let parent_node = self.arena.node_ptr(parent_idx);
            let sep_key = NodeLayout::<K, V>::internal_key_ptr(parent_node, left_child_pos).read();
            let left_node = self.arena.node_ptr(left_idx);
            NodeLayout::<K, V>::internal_key_ptr(left_node, left_len).write(sep_key);

            // Move all keys and children from right into left
            let right_node = self.arena.node_ptr(right_idx);
            let left_node = self.arena.node_ptr(left_idx);
            for i in 0..right_len {
                let k = NodeLayout::<K, V>::internal_key_ptr(right_node, i).read();
                NodeLayout::<K, V>::internal_key_ptr(left_node, left_len + 1 + i).write(k);
            }
            for i in 0..=right_len {
                let child = NodeLayout::<K, V>::internal_child_ptr(right_node, i).read();
                NodeLayout::<K, V>::internal_child_ptr(left_node, left_len + 1 + i).write(child);
                // Update child's parent
                NodeLayout::<K, V>::header_mut(self.arena.node_ptr(child)).parent = left_idx;
            }
            NodeLayout::<K, V>::header_mut(left_node).len = (left_len + 1 + right_len) as u16;
        }

        // Free the right internal node
        self.arena.free_node(right_idx);

        // Remove separator + right child from parent
        self.internal_remove_at(parent_idx, left_child_pos);
    }
}

impl<K, V> RawBTree<K, V> {
    /// Clear all elements, dropping keys and values.
    pub fn clear(&mut self) {
        self.drop_all_contents();

        // Reset state (don't free arena memory — keep it for reuse)
        self.root = NO_NODE;
        self.first_leaf = NO_NODE;
        self.last_leaf = NO_NODE;
        self.len = 0;
        self.height = 0;
        // Reset arena high-water mark and free list
        self.arena.len = 0;
        self.arena.free_head = NO_NODE;
    }

    /// Recursively drop keys in internal nodes. No trait bounds required.
    fn drop_internal_keys(&self, node_idx: NodeIdx) {
        if node_idx == NO_NODE {
            return;
        }
        let node = self.arena.node_ptr(node_idx);
        let header = unsafe { NodeLayout::<K, V>::header(node) };
        if header.is_leaf() {
            return;
        }
        let len = header.len as usize;
        for i in 0..len {
            unsafe {
                std::ptr::drop_in_place(NodeLayout::<K, V>::internal_key_ptr(node, i));
            }
        }
        // Recurse into children
        for i in 0..=len {
            let child = unsafe { NodeLayout::<K, V>::internal_child_ptr(node, i).read() };
            self.drop_internal_keys(child);
        }
    }

    /// Drop all leaf contents and internal keys. Used by Drop and clear.
    fn drop_all_contents(&mut self) {
        if self.root == NO_NODE {
            return;
        }

        // Walk all leaves and drop their contents
        let mut leaf_idx = self.first_leaf;
        while leaf_idx != NO_NODE {
            let node = self.arena.node_ptr(leaf_idx);
            let header = unsafe { NodeLayout::<K, V>::header(node) };
            let len = header.len as usize;
            let next = unsafe { NodeLayout::<K, V>::leaf_next_ptr(node).read() };

            if std::mem::needs_drop::<K>() || std::mem::needs_drop::<V>() {
                for i in 0..len {
                    unsafe {
                        std::ptr::drop_in_place(NodeLayout::<K, V>::leaf_key_ptr(node, i));
                        std::ptr::drop_in_place(NodeLayout::<K, V>::leaf_val_ptr(node, i));
                    }
                }
            }

            leaf_idx = next;
        }

        // Drop keys in internal nodes
        if std::mem::needs_drop::<K>() {
            self.drop_internal_keys(self.root);
        }
    }
}

impl<K: Clone, V: Clone> RawBTree<K, V> {
    /// Clone the tree by bulk-copying the arena, then cloning K/V values in-place.
    /// Much faster than re-inserting every element through the tree.
    pub fn clone_tree(&self) -> Self {
        if self.root == NO_NODE {
            return RawBTree {
                arena: Arena::new(),
                root: NO_NODE,
                first_leaf: NO_NODE,
                last_leaf: NO_NODE,
                len: 0,
                height: 0,
                _marker: PhantomData,
            };
        }

        // Bulk-copy the entire arena (all node indices stay valid)
        let new_arena = self.arena.clone_arena();

        // Now clone the K/V values in leaf nodes in-place.
        // The arena copy has bitwise copies of K and V, which is only valid
        // for Copy types. For non-Copy types, we need to clone each value
        // and write it over the bitwise copy (which we must NOT drop).
        let mut leaf_idx = self.first_leaf;
        while leaf_idx != NO_NODE {
            let src_node = self.arena.node_ptr(leaf_idx);
            let dst_node = new_arena.node_ptr(leaf_idx);
            let header = unsafe { NodeLayout::<K, V>::header(src_node) };
            let len = header.len as usize;
            let next = unsafe { NodeLayout::<K, V>::leaf_next_ptr(src_node).read() };

            for i in 0..len {
                unsafe {
                    // Read from source, clone, write to dest (overwriting bitwise copy)
                    let src_k = &*NodeLayout::<K, V>::leaf_key_ptr(src_node, i);
                    let src_v = &*NodeLayout::<K, V>::leaf_val_ptr(src_node, i);
                    let dst_k = NodeLayout::<K, V>::leaf_key_ptr(dst_node, i);
                    let dst_v = NodeLayout::<K, V>::leaf_val_ptr(dst_node, i);
                    // Write cloned values over the bitwise copy without dropping it
                    std::ptr::write(dst_k, src_k.clone());
                    std::ptr::write(dst_v, src_v.clone());
                }
            }

            leaf_idx = next;
        }

        // Clone keys in internal nodes
        self.clone_internal_keys(&new_arena, self.root);

        RawBTree {
            arena: new_arena,
            root: self.root,
            first_leaf: self.first_leaf,
            last_leaf: self.last_leaf,
            len: self.len,
            height: self.height,
            _marker: PhantomData,
        }
    }

    fn clone_internal_keys(&self, new_arena: &Arena, node_idx: NodeIdx) {
        if node_idx == NO_NODE {
            return;
        }
        let src_node = self.arena.node_ptr(node_idx);
        let header = unsafe { NodeLayout::<K, V>::header(src_node) };
        if header.is_leaf() {
            return;
        }
        let len = header.len as usize;
        let dst_node = new_arena.node_ptr(node_idx);

        for i in 0..len {
            unsafe {
                let src_k = &*NodeLayout::<K, V>::internal_key_ptr(src_node, i);
                let dst_k = NodeLayout::<K, V>::internal_key_ptr(dst_node, i);
                std::ptr::write(dst_k, src_k.clone());
            }
        }

        // Recurse into children
        for i in 0..=len {
            let child = unsafe { NodeLayout::<K, V>::internal_child_ptr(src_node, i).read() };
            self.clone_internal_keys(new_arena, child);
        }
    }
}

impl<K: Ord + Clone, V> RawBTree<K, V> {
    /// Build a tree from a pre-sorted, deduplicated vector of key-value pairs.
    /// Much faster than N individual inserts — O(n) vs O(n log n).
    pub fn bulk_load(mut pairs: Vec<(K, V)>) -> Self {
        NodeLayout::<K, V>::assert_capacities();

        if pairs.is_empty() {
            return Self::new();
        }

        let n = pairs.len();
        let leaf_cap = NodeLayout::<K, V>::LEAF_CAP;
        let num_leaves = n.div_ceil(leaf_cap);

        // Pre-allocate arena
        // Rough estimate: leaves + internal nodes (leaves / internal_cap per level)
        let estimated_nodes = num_leaves + num_leaves / 2 + 4;
        let mut arena = Arena::with_capacity(estimated_nodes as u32);

        // Phase 1: Build leaf nodes from the sorted pairs
        let mut leaf_indices: Vec<NodeIdx> = Vec::with_capacity(num_leaves);
        let mut pair_iter = pairs.drain(..);

        for _ in 0..num_leaves {
            let leaf_idx = arena.alloc_node();
            let node = arena.node_ptr(leaf_idx);

            let mut count = 0;
            for _ in 0..leaf_cap {
                if let Some((k, v)) = pair_iter.next() {
                    unsafe {
                        NodeLayout::<K, V>::leaf_key_ptr(node, count).write(k);
                        NodeLayout::<K, V>::leaf_val_ptr(node, count).write(v);
                    }
                    count += 1;
                } else {
                    break;
                }
            }

            unsafe {
                let header = NodeLayout::<K, V>::header_mut(node);
                header.len = count as u16;
                header.flags = NodeHeader::IS_LEAF;
                header.parent = NO_NODE;
            }

            leaf_indices.push(leaf_idx);
        }

        // Phase 2: Wire leaf chain
        let first_leaf = leaf_indices[0];
        let last_leaf = *leaf_indices.last().unwrap();

        for i in 0..leaf_indices.len() {
            let node = arena.node_ptr(leaf_indices[i]);
            unsafe {
                NodeLayout::<K, V>::leaf_prev_ptr(node).write(if i > 0 {
                    leaf_indices[i - 1]
                } else {
                    NO_NODE
                });
                NodeLayout::<K, V>::leaf_next_ptr(node).write(if i + 1 < leaf_indices.len() {
                    leaf_indices[i + 1]
                } else {
                    NO_NODE
                });
            }
        }

        // Phase 3: Build internal nodes bottom-up.
        //
        // Each level keeps a parallel `min_keys` vector: the leftmost leaf-key
        // in every subtree at the current level. The separator promoted into
        // a parent at position j is `min_keys[i + j + 1]` — the leftmost leaf
        // key of the subtree to the right of that separator. Pulling the key
        // from the child's first internal-key is wrong at level ≥ 2 because
        // that key splits the child's *own* subtrees, not the boundary
        // between siblings.
        let mut children = leaf_indices;
        // For leaves: leftmost key = first key of the leaf.
        let mut min_keys: Vec<K> = children.iter().map(|&leaf_idx| {
            let node = arena.node_ptr(leaf_idx);
            unsafe { (*NodeLayout::<K, V>::leaf_key_ptr(node, 0)).clone() }
        }).collect();
        let mut height = 0u32;

        while children.len() > 1 {
            let internal_cap = NodeLayout::<K, V>::INTERNAL_CAP;
            let mut parents: Vec<NodeIdx> = Vec::new();
            let mut parent_min_keys: Vec<K> = Vec::new();
            let mut i = 0;

            while i < children.len() {
                let parent_idx = arena.alloc_node();
                let parent_node = arena.node_ptr(parent_idx);

                // How many children fit in this internal node?
                let remaining = children.len() - i;
                let n_children = remaining.min(internal_cap + 1);
                let n_keys = n_children - 1;

                unsafe {
                    let header = NodeLayout::<K, V>::header_mut(parent_node);
                    header.len = n_keys as u16;
                    header.flags = 0; // internal
                    header.parent = NO_NODE;

                    // Write first child
                    NodeLayout::<K, V>::internal_child_ptr(parent_node, 0).write(children[i]);

                    // Write keys and remaining children
                    for j in 0..n_keys {
                        let child_pos = i + j + 1;
                        // Separator = leftmost leaf-key in the subtree right of this
                        // separator (i.e. the subtree rooted at children[child_pos]).
                        let sep_key = min_keys[child_pos].clone();

                        NodeLayout::<K, V>::internal_key_ptr(parent_node, j).write(sep_key);
                        NodeLayout::<K, V>::internal_child_ptr(parent_node, j + 1)
                            .write(children[child_pos]);
                    }

                    // Set parent pointers on children
                    for j in 0..n_children {
                        let child_node = arena.node_ptr(children[i + j]);
                        NodeLayout::<K, V>::header_mut(child_node).parent = parent_idx;
                    }
                }

                // The new parent's leftmost key = leftmost key of its first child.
                parent_min_keys.push(min_keys[i].clone());
                parents.push(parent_idx);
                i += n_children;
            }

            children = parents;
            min_keys = parent_min_keys;
            height += 1;
        }

        let root = children[0];

        RawBTree {
            arena,
            root,
            first_leaf,
            last_leaf,
            len: n,
            height,
            _marker: PhantomData,
        }
    }
}

// ── Tree-surgery append (disjoint adjacent graft) ───────────────────────

impl<K: Ord + Clone, V> RawBTree<K, V> {
    /// Recursively copy a subtree from `src_arena` (rooted at `src`) into
    /// `dst_arena`, returning the new root index in `dst_arena`. Wires copied
    /// leaves into the running chain `(dst_first_leaf, dst_chain_tail)` in
    /// sorted order — caller pre-seeds `dst_chain_tail` with the splice point
    /// in dst (e.g. `self.last_leaf` for an "append at the tail" graft).
    /// `src_arena` is NOT modified — values are byte-copied; the caller is
    /// responsible for preventing double-drop by deallocating src's arena
    /// without traversing nodes (e.g. `*src = Self::new()`).
    fn clone_subtree_into(
        src_arena: &Arena,
        src: NodeIdx,
        dst_arena: &mut Arena,
        dst_first_leaf: &mut NodeIdx,
        dst_chain_tail: &mut NodeIdx,
    ) -> NodeIdx {
        let (is_leaf, len) = unsafe {
            let header = NodeLayout::<K, V>::header(src_arena.node_ptr(src));
            (header.is_leaf(), header.len as usize)
        };

        let dst = dst_arena.alloc_node();

        if is_leaf {
            // Bitwise-copy header+keys+values; do NOT copy prev/next links.
            let copy_size = NODE_SIZE - 2 * std::mem::size_of::<NodeIdx>();
            unsafe {
                let src_ptr = src_arena.node_ptr(src);
                let dst_ptr = dst_arena.node_ptr(dst);
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, copy_size);
                let dst_header = NodeLayout::<K, V>::header_mut(dst_ptr);
                dst_header.parent = NO_NODE;

                NodeLayout::<K, V>::leaf_prev_ptr(dst_ptr).write(*dst_chain_tail);
                NodeLayout::<K, V>::leaf_next_ptr(dst_ptr).write(NO_NODE);
                if *dst_chain_tail != NO_NODE {
                    let prev_ptr = dst_arena.node_ptr(*dst_chain_tail);
                    NodeLayout::<K, V>::leaf_next_ptr(prev_ptr).write(dst);
                }
                // First *new* leaf in this clone op — distinct from the splice
                // anchor `dst_chain_tail` may have been pre-seeded with.
                if *dst_first_leaf == NO_NODE {
                    *dst_first_leaf = dst;
                }
                *dst_chain_tail = dst;
            }
        } else {
            // Internal: copy bytes (with stale child indices) then re-write
            // child indices after recursive copies.
            unsafe {
                let src_ptr = src_arena.node_ptr(src);
                let dst_ptr = dst_arena.node_ptr(dst);
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, NODE_SIZE);
                NodeLayout::<K, V>::header_mut(dst_ptr).parent = NO_NODE;
            }

            let mut children: Vec<NodeIdx> = Vec::with_capacity(len + 1);
            for i in 0..=len {
                children.push(unsafe {
                    NodeLayout::<K, V>::internal_child_ptr(src_arena.node_ptr(src), i).read()
                });
            }

            for (i, c) in children.iter().enumerate() {
                let new_c = Self::clone_subtree_into(
                    src_arena,
                    *c,
                    dst_arena,
                    dst_first_leaf,
                    dst_chain_tail,
                );
                unsafe {
                    let dst_ptr = dst_arena.node_ptr(dst);
                    NodeLayout::<K, V>::internal_child_ptr(dst_ptr, i).write(new_c);
                    NodeLayout::<K, V>::header_mut(dst_arena.node_ptr(new_c)).parent = dst;
                }
            }
        }
        dst
    }

    /// Tree-surgery graft: precondition `self.last_key < other.first_key`,
    /// and both are non-empty. Copies `other`'s tree into `self.arena`,
    /// splices the leaf chain at `self.last_leaf`, and bridges the spines.
    ///
    /// Returns `true` on success, `false` if the height delta isn't currently
    /// supported (caller should fall back to drain). For now we support
    /// `self.height >= other.height`. The asymmetric case (other taller than
    /// self) is rare in practice and falls through to drain.
    ///
    /// O(other_arena_nodes + log self.height). When other is significantly
    /// smaller than self, this avoids touching most of self's existing nodes.
    pub(crate) fn append_graft_disjoint(&mut self, other: &mut Self) -> bool {
        debug_assert!(!self.is_empty() && !other.is_empty());
        if self.height < other.height {
            return false;
        }

        // Step 1: copy other's tree into self.arena; splice the chain at self.last_leaf.
        let mut dst_first_leaf: NodeIdx = NO_NODE;
        let mut dst_chain_tail: NodeIdx = self.last_leaf;
        let other_root_in_self = Self::clone_subtree_into(
            &other.arena,
            other.root,
            &mut self.arena,
            &mut dst_first_leaf,
            &mut dst_chain_tail,
        );

        // Step 2: link self.last_leaf.next -> dst_first_leaf
        // (clone_subtree_into already wired the *prev* of dst_first_leaf to dst_chain_tail's
        //  initial value, which was self.last_leaf — so the leaf chain is now bidirectional).
        unsafe {
            let prev_ptr = self.arena.node_ptr(self.last_leaf);
            NodeLayout::<K, V>::leaf_next_ptr(prev_ptr).write(dst_first_leaf);
        }
        let new_last_leaf = dst_chain_tail;

        // Step 3: bridge spines. Separator = first key of dst_first_leaf
        //         (= other's smallest key).
        let separator: K = unsafe {
            let n = self.arena.node_ptr(dst_first_leaf);
            (*NodeLayout::<K, V>::leaf_key_ptr(n, 0)).clone()
        };
        let other_height = other.height;
        self.graft_combine_roots(other_root_in_self, separator, other_height);

        // Step 4: finalize state.
        self.last_leaf = new_last_leaf;
        self.len += other.len;
        // Reset other to empty. Its arena's bytes are dropped by Arena::drop
        // (no per-node walk), so the K/V values that were byte-copied into
        // self.arena are NOT double-dropped.
        other.arena = Arena::new();
        other.root = NO_NODE;
        other.first_leaf = NO_NODE;
        other.last_leaf = NO_NODE;
        other.len = 0;
        other.height = 0;

        true
    }

    /// Bridge two roots after grafting `other_root` (now in self.arena) onto
    /// the right side of `self.root`. Handles the cap-cascade if internal
    /// nodes overflow during the splice.
    /// Precondition: `self.height >= other_height`.
    fn graft_combine_roots(&mut self, other_root: NodeIdx, separator: K, other_height: u32) {
        debug_assert!(self.height >= other_height);

        if self.height == other_height {
            // New root above both children.
            let new_root = self.arena.alloc_node();
            unsafe {
                let new_root_node = self.arena.node_ptr(new_root);
                let header = NodeLayout::<K, V>::header_mut(new_root_node);
                header.len = 1;
                header.flags = 0; // internal
                header.parent = NO_NODE;
                NodeLayout::<K, V>::internal_key_ptr(new_root_node, 0).write(separator);
                NodeLayout::<K, V>::internal_child_ptr(new_root_node, 0).write(self.root);
                NodeLayout::<K, V>::internal_child_ptr(new_root_node, 1).write(other_root);
                NodeLayout::<K, V>::header_mut(self.arena.node_ptr(self.root)).parent = new_root;
                NodeLayout::<K, V>::header_mut(self.arena.node_ptr(other_root)).parent = new_root;
            }
            self.root = new_root;
            self.height += 1;
            return;
        }

        // self.height > other_height: descend self's right spine to depth
        // (self.height - other_height - 1). The node we land on (`cur`) is at
        // level (other_height + 1) — exactly one level above other_root.
        // We then add `(separator, other_root)` as cur's new last child;
        // if cur is full, cascade splits up via propagate_split.
        let descend = self.height - other_height - 1;
        let mut cur = self.root;
        let mut path = PathBuf::new();
        for _ in 0..descend {
            let cur_len = unsafe {
                NodeLayout::<K, V>::header(self.arena.node_ptr(cur)).len as usize
            };
            // cur's slot in its parent IS the next-pos+1 (rightmost child).
            // For propagate_split semantics, child_pos == cur's slot in parent
            // (= len position, where the new right-sibling's separator key
            //  will go).
            path.push((cur, cur_len));
            // Descend into rightmost child.
            cur = unsafe {
                NodeLayout::<K, V>::internal_child_ptr(self.arena.node_ptr(cur), cur_len).read()
            };
        }
        let cur_len = unsafe {
            NodeLayout::<K, V>::header(self.arena.node_ptr(cur)).len as usize
        };

        if cur_len < NodeLayout::<K, V>::INTERNAL_CAP {
            // Room: append separator + other_root at cur's tail.
            self.internal_insert_at(cur, cur_len, separator, other_root);
            unsafe {
                NodeLayout::<K, V>::header_mut(self.arena.node_ptr(other_root)).parent = cur;
            }
            return;
        }

        // cur is full: split, then cascade up.
        let (promoted, new_internal) =
            self.internal_split_and_insert(cur, cur_len, separator, other_root);
        self.propagate_split(&mut path, promoted, new_internal);
    }
}

// ── Tree-surgery split_off ──────────────────────────────────────────────

#[derive(Clone, Copy)]
pub(crate) enum CollapseEdge {
    Leftmost,
    Rightmost,
}

impl<K, V> RawBTree<K, V> {
    /// Walk down a tree along the leftmost or rightmost edge, collapsing
    /// degenerate internal nodes (1 child, 0 keys) along the way. After a
    /// surgical split, the spine of the resulting tree may have such
    /// degenerates; this normalizes the structure and returns the new root.
    pub(crate) fn collapse_along_edge(
        arena: &mut Arena,
        root: NodeIdx,
        edge: CollapseEdge,
    ) -> NodeIdx {
        if root == NO_NODE {
            return NO_NODE;
        }
        let mut new_root = root;
        // Track the parent that holds the link to `current` and the slot index there.
        // None at the very top (current == new_root).
        let mut parent_node: Option<NodeIdx> = None;
        let mut parent_slot: usize = 0;
        let mut current = root;

        loop {
            let header = unsafe { NodeLayout::<K, V>::header(arena.node_ptr(current)) };
            if header.is_leaf() {
                return new_root;
            }
            let len = header.len as usize;
            if len == 0 {
                // Degenerate. Replace `current` with its sole child.
                let only =
                    unsafe { NodeLayout::<K, V>::internal_child_ptr(arena.node_ptr(current), 0).read() };
                unsafe {
                    NodeLayout::<K, V>::header_mut(arena.node_ptr(only)).parent =
                        parent_node.unwrap_or(NO_NODE);
                }
                if let Some(p) = parent_node {
                    unsafe {
                        NodeLayout::<K, V>::internal_child_ptr(arena.node_ptr(p), parent_slot)
                            .write(only);
                    }
                } else {
                    new_root = only;
                }
                arena.free_node(current);
                current = only;
                // parent_node / parent_slot unchanged: we replaced one child with another at the same slot.
            } else {
                // Non-degenerate. Descend chosen edge.
                let slot = match edge {
                    CollapseEdge::Leftmost => 0,
                    CollapseEdge::Rightmost => len, // children[len] is rightmost
                };
                parent_node = Some(current);
                parent_slot = slot;
                current = unsafe {
                    NodeLayout::<K, V>::internal_child_ptr(arena.node_ptr(current), slot).read()
                };
            }
        }
    }

    /// Compute the height of the tree by walking root → leaf via the leftmost child.
    pub(crate) fn compute_height(arena: &Arena, root: NodeIdx) -> u32 {
        if root == NO_NODE {
            return 0;
        }
        let mut h: u32 = 0;
        let mut node = root;
        loop {
            let header = unsafe { NodeLayout::<K, V>::header(arena.node_ptr(node)) };
            if header.is_leaf() {
                return h;
            }
            h += 1;
            node = unsafe { NodeLayout::<K, V>::internal_child_ptr(arena.node_ptr(node), 0).read() };
        }
    }

    /// Deep-copy a subtree rooted at `src` (in `self.arena`) into `dst_arena`,
    /// freeing source nodes as it goes. Each copied leaf is appended to the
    /// running leaf chain in `dst_arena` (tracked via `dst_first_leaf` / `dst_chain_tail`).
    /// Returns the new root index in `dst_arena`.
    fn deep_copy_subtree(
        &mut self,
        src: NodeIdx,
        dst_arena: &mut Arena,
        dst_first_leaf: &mut NodeIdx,
        dst_chain_tail: &mut NodeIdx,
        dst_count: &mut usize,
    ) -> NodeIdx {
        // Snapshot shape info before any mutation.
        let (is_leaf, len) = {
            let header = unsafe { NodeLayout::<K, V>::header(self.arena.node_ptr(src)) };
            (header.is_leaf(), header.len as usize)
        };

        let dst = dst_arena.alloc_node(); // may grow dst_arena

        if is_leaf {
            // Bitwise-copy header+keys+values; do NOT copy prev/next links.
            let copy_size = NODE_SIZE - 2 * std::mem::size_of::<NodeIdx>();
            unsafe {
                let src_ptr = self.arena.node_ptr(src);
                let dst_ptr = dst_arena.node_ptr(dst);
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, copy_size);
                let dst_header = NodeLayout::<K, V>::header_mut(dst_ptr);
                dst_header.parent = NO_NODE;

                // Wire dst into the leaf chain.
                NodeLayout::<K, V>::leaf_prev_ptr(dst_ptr).write(*dst_chain_tail);
                NodeLayout::<K, V>::leaf_next_ptr(dst_ptr).write(NO_NODE);
                if *dst_chain_tail != NO_NODE {
                    let prev_ptr = dst_arena.node_ptr(*dst_chain_tail);
                    NodeLayout::<K, V>::leaf_next_ptr(prev_ptr).write(dst);
                } else {
                    *dst_first_leaf = dst;
                }
                *dst_chain_tail = dst;

                // Mark src as moved-out so it won't be walked / dropped.
                let src_header = NodeLayout::<K, V>::header_mut(self.arena.node_ptr(src));
                src_header.len = 0;
            }
            *dst_count += len;
            self.arena.free_node(src);
        } else {
            // Internal: copy bytes (with stale child indices) then re-write children
            // after recursive copies.
            unsafe {
                let src_ptr = self.arena.node_ptr(src);
                let dst_ptr = dst_arena.node_ptr(dst);
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, NODE_SIZE);
                NodeLayout::<K, V>::header_mut(dst_ptr).parent = NO_NODE;
            }

            // Snapshot children indices.
            let mut children: Vec<NodeIdx> = Vec::with_capacity(len + 1);
            for i in 0..=len {
                children.push(unsafe {
                    NodeLayout::<K, V>::internal_child_ptr(self.arena.node_ptr(src), i).read()
                });
            }

            // Recurse leftmost-first so the leaf chain extends in sorted order.
            for (i, c) in children.iter().enumerate() {
                let new_c =
                    self.deep_copy_subtree(*c, dst_arena, dst_first_leaf, dst_chain_tail, dst_count);
                unsafe {
                    let dst_ptr = dst_arena.node_ptr(dst);
                    NodeLayout::<K, V>::internal_child_ptr(dst_ptr, i).write(new_c);
                    NodeLayout::<K, V>::header_mut(dst_arena.node_ptr(new_c)).parent = dst;
                }
            }

            // Mark src as moved-out (its keys were bitwise-copied into dst).
            unsafe {
                NodeLayout::<K, V>::header_mut(self.arena.node_ptr(src)).len = 0;
            }
            self.arena.free_node(src);
        }

        dst
    }
}

impl<K: Ord, V> RawBTree<K, V> {
    /// Tree-surgery split_off. Mutates self in place to retain keys `< at`,
    /// deep-copies the right subtree into a fresh arena. O(log n + right_nodes)
    /// rather than O(n) for drain+bulk_load.
    ///
    /// Currently always copies the right side. For asymmetric splits where the
    /// left side is smaller, the larger right side dominates cost; see roadmap
    /// for "adaptive split direction" follow-up.
    pub(crate) fn split_off_surgical_right<Q>(&mut self, at: &Q) -> Self
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        // Empty tree → empty result.
        if self.root == NO_NODE {
            return Self::new();
        }

        // Find the boundary leaf+slot.
        let (leaf_idx, slot_idx) = match self.lower_bound(at) {
            Some(p) => p,
            None => return Self::new(), // all keys < at
        };

        // Trivial: at lands before all keys → entire tree moves right.
        if leaf_idx == self.first_leaf && slot_idx == 0 {
            let mut moved = Self::new();
            std::mem::swap(self, &mut moved);
            return moved;
        }

        // Spine from root down to the boundary leaf (one entry per internal level).
        let mut spine = PathBuf::new();
        self.path_to_node(leaf_idx, &mut spine);
        debug_assert_eq!(spine.len() as u32, self.height);

        // Capture the boundary leaf's original prev (for later if the whole leaf moves right).
        let (leaf_len, original_leaf_prev) = unsafe {
            let n = self.arena.node_ptr(leaf_idx);
            (
                NodeLayout::<K, V>::header(n).len as usize,
                NodeLayout::<K, V>::leaf_prev_ptr(n).read(),
            )
        };

        // ── Phase 1: split or move the boundary leaf ───────────────────────
        let mut right_arena = Arena::new();
        let mut right_first_leaf: NodeIdx = NO_NODE;
        let mut right_chain_tail: NodeIdx = NO_NODE;
        let mut right_count: usize = 0;

        let (mut left_root_below, mut right_root_below): (Option<NodeIdx>, Option<NodeIdx>) =
            if slot_idx == 0 {
                // Whole boundary leaf goes right.
                let dst = self.deep_copy_subtree(
                    leaf_idx,
                    &mut right_arena,
                    &mut right_first_leaf,
                    &mut right_chain_tail,
                    &mut right_count,
                );
                (None, Some(dst))
            } else {
                // Physical split: keys[0..slot_idx] stay, keys[slot_idx..len] go right.
                let right_dst = right_arena.alloc_node();
                let n_to_right = leaf_len - slot_idx;
                unsafe {
                    let dst_ptr = right_arena.node_ptr(right_dst);
                    let dst_header = NodeLayout::<K, V>::header_mut(dst_ptr);
                    dst_header.len = n_to_right as u16;
                    dst_header.flags = NodeHeader::IS_LEAF;
                    dst_header.parent = NO_NODE;
                    NodeLayout::<K, V>::leaf_prev_ptr(dst_ptr).write(NO_NODE);
                    NodeLayout::<K, V>::leaf_next_ptr(dst_ptr).write(NO_NODE);
                }
                for i in 0..n_to_right {
                    unsafe {
                        let src_ptr = self.arena.node_ptr(leaf_idx);
                        let k = NodeLayout::<K, V>::leaf_key_ptr(src_ptr, slot_idx + i).read();
                        let v = NodeLayout::<K, V>::leaf_val_ptr(src_ptr, slot_idx + i).read();
                        let dst_ptr = right_arena.node_ptr(right_dst);
                        NodeLayout::<K, V>::leaf_key_ptr(dst_ptr, i).write(k);
                        NodeLayout::<K, V>::leaf_val_ptr(dst_ptr, i).write(v);
                    }
                }
                unsafe {
                    let src_ptr = self.arena.node_ptr(leaf_idx);
                    NodeLayout::<K, V>::header_mut(src_ptr).len = slot_idx as u16;
                    // Cut the next-leaf link; subsequent leaves are gone in self.
                    NodeLayout::<K, V>::leaf_next_ptr(src_ptr).write(NO_NODE);
                }
                right_first_leaf = right_dst;
                right_chain_tail = right_dst;
                right_count += n_to_right;
                (Some(leaf_idx), Some(right_dst))
            };

        // ── Phase 2: spine walk bottom-up ──────────────────────────────────
        // No collapsing during the walk — we always build internal nodes (even
        // degenerate 1-child/0-keys ones) so that subtree heights stay aligned
        // when we combine `*_root_below` with deep-copied siblings. Degenerates
        // are removed in Phase 3.
        for &(parent_idx, child_pos) in spine.as_slice().iter().rev() {
            let n = unsafe {
                NodeLayout::<K, V>::header(self.arena.node_ptr(parent_idx)).len as usize
            };

            // Snapshot children to move (children[child_pos+1..=n]) and the
            // surrounding keys, BEFORE deep_copy_subtree mutates self.arena.
            let mut moved_children: Vec<NodeIdx> = Vec::with_capacity(n.saturating_sub(child_pos));
            for i in (child_pos + 1)..=n {
                moved_children.push(unsafe {
                    NodeLayout::<K, V>::internal_child_ptr(self.arena.node_ptr(parent_idx), i).read()
                });
            }

            // separator_at_split = keys[child_pos] (B+ tree: this equals the
            // first leaf-key in original child[child_pos+1]'s subtree, which is
            // also the first leaf-key of the right side at this level).
            let separator_at_split: Option<K> = if child_pos < n {
                Some(unsafe {
                    NodeLayout::<K, V>::internal_key_ptr(self.arena.node_ptr(parent_idx), child_pos)
                        .read()
                })
            } else {
                None
            };
            let mut keys_after_split: Vec<K> =
                Vec::with_capacity(n.saturating_sub(child_pos + 1));
            for i in (child_pos + 1)..n {
                keys_after_split.push(unsafe {
                    NodeLayout::<K, V>::internal_key_ptr(self.arena.node_ptr(parent_idx), i).read()
                });
            }

            // Deep-copy each moved child's subtree to right_arena.
            let mut copied_children: Vec<NodeIdx> = Vec::with_capacity(moved_children.len());
            for c in moved_children {
                let new_c = self.deep_copy_subtree(
                    c,
                    &mut right_arena,
                    &mut right_first_leaf,
                    &mut right_chain_tail,
                    &mut right_count,
                );
                copied_children.push(new_c);
            }

            // Build new right at this level.
            let mut new_right_keys: Vec<K> = Vec::new();
            let mut new_right_children: Vec<NodeIdx> = Vec::new();

            if let Some(rrb) = right_root_below {
                new_right_children.push(rrb);
                if !copied_children.is_empty() {
                    // separator_at_split is Some iff copied_children non-empty
                    // (both are equivalent to child_pos < n).
                    new_right_keys.push(
                        separator_at_split.expect("separator_at_split must be Some when copied non-empty"),
                    );
                } else {
                    debug_assert!(separator_at_split.is_none());
                }
            } else {
                // Drop the separator: its left side (the dropped boundary subtree) is gone.
                drop(separator_at_split);
            }
            new_right_children.extend(copied_children);
            new_right_keys.extend(keys_after_split);

            right_root_below = if new_right_children.is_empty() {
                None
            } else {
                let new_right = right_arena.alloc_node();
                let n_keys = new_right_keys.len();
                let n_children = new_right_children.len();
                unsafe {
                    let dst_ptr = right_arena.node_ptr(new_right);
                    let header = NodeLayout::<K, V>::header_mut(dst_ptr);
                    header.len = n_keys as u16;
                    header.flags = 0; // internal
                    header.parent = NO_NODE;
                }
                for (i, k) in new_right_keys.into_iter().enumerate() {
                    unsafe {
                        NodeLayout::<K, V>::internal_key_ptr(right_arena.node_ptr(new_right), i)
                            .write(k);
                    }
                }
                for i in 0..n_children {
                    let c = new_right_children[i];
                    unsafe {
                        NodeLayout::<K, V>::internal_child_ptr(
                            right_arena.node_ptr(new_right),
                            i,
                        )
                        .write(c);
                        NodeLayout::<K, V>::header_mut(right_arena.node_ptr(c)).parent = new_right;
                    }
                }
                Some(new_right)
            };

            // Mutate `parent_idx` in self.arena to be the new left at this level.
            left_root_below = if let Some(lrb) = left_root_below {
                unsafe {
                    let parent_node = self.arena.node_ptr(parent_idx);
                    NodeLayout::<K, V>::internal_child_ptr(parent_node, child_pos).write(lrb);
                    NodeLayout::<K, V>::header_mut(self.arena.node_ptr(lrb)).parent = parent_idx;
                    NodeLayout::<K, V>::header_mut(parent_node).len = child_pos as u16;
                }
                Some(parent_idx)
            } else if child_pos == 0 {
                // No keys, no children retained: free parent.
                self.arena.free_node(parent_idx);
                None
            } else {
                // Drop separator before the now-removed child[child_pos].
                unsafe {
                    let parent_node = self.arena.node_ptr(parent_idx);
                    std::ptr::drop_in_place(
                        NodeLayout::<K, V>::internal_key_ptr(parent_node, child_pos - 1),
                    );
                    NodeLayout::<K, V>::header_mut(parent_node).len = (child_pos - 1) as u16;
                }
                Some(parent_idx)
            };
        }

        // ── Phase 3: collapse degenerates + finalize self ──────────────────
        let mut self_root = left_root_below.unwrap_or(NO_NODE);
        if self_root != NO_NODE {
            unsafe {
                NodeLayout::<K, V>::header_mut(self.arena.node_ptr(self_root)).parent = NO_NODE;
            }
            self_root = Self::collapse_along_edge(
                &mut self.arena,
                self_root,
                CollapseEdge::Rightmost,
            );
        }

        let new_self_last_leaf = if slot_idx > 0 {
            leaf_idx
        } else {
            // Boundary leaf moved entirely; new last leaf = original predecessor.
            original_leaf_prev
        };
        if new_self_last_leaf != NO_NODE {
            unsafe {
                NodeLayout::<K, V>::leaf_next_ptr(self.arena.node_ptr(new_self_last_leaf))
                    .write(NO_NODE);
            }
        }
        let new_self_first_leaf = if new_self_last_leaf == NO_NODE {
            NO_NODE
        } else {
            self.first_leaf
        };

        let new_self_height = Self::compute_height(&self.arena, self_root);
        let new_self_len = self.len - right_count;

        self.root = self_root;
        self.first_leaf = new_self_first_leaf;
        self.last_leaf = new_self_last_leaf;
        self.height = new_self_height;
        self.len = new_self_len;

        // ── Phase 4: build the right tree ──────────────────────────────────
        let mut right_root = right_root_below.unwrap_or(NO_NODE);
        if right_root != NO_NODE {
            unsafe {
                NodeLayout::<K, V>::header_mut(right_arena.node_ptr(right_root)).parent = NO_NODE;
            }
            right_root = Self::collapse_along_edge(
                &mut right_arena,
                right_root,
                CollapseEdge::Leftmost,
            );
        }
        if right_first_leaf != NO_NODE {
            unsafe {
                NodeLayout::<K, V>::leaf_prev_ptr(right_arena.node_ptr(right_first_leaf))
                    .write(NO_NODE);
            }
        }
        if right_chain_tail != NO_NODE {
            unsafe {
                NodeLayout::<K, V>::leaf_next_ptr(right_arena.node_ptr(right_chain_tail))
                    .write(NO_NODE);
            }
        }

        let right_height = Self::compute_height(&right_arena, right_root);

        RawBTree {
            arena: right_arena,
            root: right_root,
            first_leaf: right_first_leaf,
            last_leaf: right_chain_tail,
            len: right_count,
            height: right_height,
            _marker: PhantomData,
        }
    }

    /// Mirror of [`split_off_surgical_right`]: the kept side stays in
    /// `self.arena` (mutated in place), and the LEFT side (keys < at) is
    /// deep-copied into a fresh arena. A final `mem::swap` flips the result
    /// so `self` ends up with `< at` and the returned tree holds `>= at`,
    /// matching the public `split_off` contract.
    ///
    /// O(log n + left_nodes) — preferable to copy-right when the left side
    /// is the smaller of the two.
    pub(crate) fn split_off_surgical_left<Q>(&mut self, at: &Q) -> Self
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if self.root == NO_NODE {
            return Self::new();
        }

        let (leaf_idx, slot_idx) = match self.lower_bound(at) {
            Some(p) => p,
            None => return Self::new(), // all keys < at
        };

        // Trivial: at lands before all keys → entire tree moves to returned.
        if leaf_idx == self.first_leaf && slot_idx == 0 {
            let mut moved = Self::new();
            std::mem::swap(self, &mut moved);
            return moved;
        }

        let mut spine = PathBuf::new();
        self.path_to_node(leaf_idx, &mut spine);
        debug_assert_eq!(spine.len() as u32, self.height);

        let leaf_len = unsafe {
            NodeLayout::<K, V>::header(self.arena.node_ptr(leaf_idx)).len as usize
        };

        // ── Phase 1: split or move boundary leaf ───────────────────────────
        // The kept side ALWAYS retains the boundary leaf (since keys[slot_idx..]
        // are all >= at and stay), so right_root_below is Some throughout.
        let mut left_arena = Arena::new();
        let mut left_first_leaf: NodeIdx = NO_NODE;
        let mut left_chain_tail: NodeIdx = NO_NODE;
        let mut left_count: usize = 0;

        let mut left_root_below: Option<NodeIdx> = if slot_idx > 0 {
            // Physical split: keys[0..slot_idx] go LEFT, keys[slot_idx..] stay.
            let left_dst = left_arena.alloc_node();
            let n_to_left = slot_idx;
            unsafe {
                let dst_ptr = left_arena.node_ptr(left_dst);
                let dst_header = NodeLayout::<K, V>::header_mut(dst_ptr);
                dst_header.len = n_to_left as u16;
                dst_header.flags = NodeHeader::IS_LEAF;
                dst_header.parent = NO_NODE;
                NodeLayout::<K, V>::leaf_prev_ptr(dst_ptr).write(NO_NODE);
                NodeLayout::<K, V>::leaf_next_ptr(dst_ptr).write(NO_NODE);
            }
            for i in 0..n_to_left {
                unsafe {
                    let src_ptr = self.arena.node_ptr(leaf_idx);
                    let k = NodeLayout::<K, V>::leaf_key_ptr(src_ptr, i).read();
                    let v = NodeLayout::<K, V>::leaf_val_ptr(src_ptr, i).read();
                    let dst_ptr = left_arena.node_ptr(left_dst);
                    NodeLayout::<K, V>::leaf_key_ptr(dst_ptr, i).write(k);
                    NodeLayout::<K, V>::leaf_val_ptr(dst_ptr, i).write(v);
                }
            }
            // Shift keys[slot_idx..leaf_len] → keys[0..leaf_len-slot_idx] and
            // values likewise. Source/dest overlap; use memmove (ptr::copy).
            let n_kept = leaf_len - slot_idx;
            unsafe {
                let src_ptr = self.arena.node_ptr(leaf_idx);
                std::ptr::copy(
                    NodeLayout::<K, V>::leaf_key_ptr(src_ptr, slot_idx),
                    NodeLayout::<K, V>::leaf_key_ptr(src_ptr, 0),
                    n_kept,
                );
                std::ptr::copy(
                    NodeLayout::<K, V>::leaf_val_ptr(src_ptr, slot_idx),
                    NodeLayout::<K, V>::leaf_val_ptr(src_ptr, 0),
                    n_kept,
                );
                NodeLayout::<K, V>::header_mut(src_ptr).len = n_kept as u16;
                NodeLayout::<K, V>::leaf_prev_ptr(src_ptr).write(NO_NODE);
            }
            left_first_leaf = left_dst;
            left_chain_tail = left_dst;
            left_count += n_to_left;
            Some(left_dst)
        } else {
            // slot_idx == 0: boundary leaf stays unchanged (all keys >= at).
            // Predecessors will be deep-copied; we set boundary.prev = NO_NODE
            // in Phase 3 along the kept-side finalize.
            None
        };
        let mut right_root_below: Option<NodeIdx> = Some(leaf_idx);

        // ── Phase 2: spine walk bottom-up ──────────────────────────────────
        for &(parent_idx, child_pos) in spine.as_slice().iter().rev() {
            let n = unsafe {
                NodeLayout::<K, V>::header(self.arena.node_ptr(parent_idx)).len as usize
            };

            // Snapshot moved_children = c[0..child_pos] (NodeIdx is Copy).
            let mut moved_children: Vec<NodeIdx> = Vec::with_capacity(child_pos);
            for i in 0..child_pos {
                moved_children.push(unsafe {
                    NodeLayout::<K, V>::internal_child_ptr(self.arena.node_ptr(parent_idx), i).read()
                });
            }

            // separator_at_split = keys[child_pos - 1]: separator between
            // c[child_pos-1] (last moved) and c[child_pos] (boundary subtree).
            // In B+ tree, this equals the first leaf-key of c[child_pos], which
            // is also the first leaf-key of lrb (left portion of boundary subtree).
            let separator_at_split: Option<K> = if child_pos > 0 {
                Some(unsafe {
                    NodeLayout::<K, V>::internal_key_ptr(
                        self.arena.node_ptr(parent_idx),
                        child_pos - 1,
                    )
                    .read()
                })
            } else {
                None
            };
            let mut keys_before_split: Vec<K> =
                Vec::with_capacity(child_pos.saturating_sub(1));
            for i in 0..child_pos.saturating_sub(1) {
                keys_before_split.push(unsafe {
                    NodeLayout::<K, V>::internal_key_ptr(self.arena.node_ptr(parent_idx), i).read()
                });
            }

            // Deep-copy moved_children into left_arena via a TEMP chain.
            // Bottom-up walk visits deepest siblings first (closer to boundary
            // = larger keys), so each batch must PREPEND to the main chain to
            // keep sorted order.
            let mut temp_first_leaf: NodeIdx = NO_NODE;
            let mut temp_chain_tail: NodeIdx = NO_NODE;
            let mut temp_count: usize = 0;
            let mut copied_children: Vec<NodeIdx> = Vec::with_capacity(moved_children.len());
            for c in moved_children {
                let new_c = self.deep_copy_subtree(
                    c,
                    &mut left_arena,
                    &mut temp_first_leaf,
                    &mut temp_chain_tail,
                    &mut temp_count,
                );
                copied_children.push(new_c);
            }

            // Build new_left at this level: copied_children + (lrb if Some).
            let mut new_left_keys: Vec<K> = Vec::new();
            let mut new_left_children: Vec<NodeIdx> = Vec::new();
            new_left_children.extend(copied_children);
            new_left_keys.extend(keys_before_split);
            if let Some(lrb) = left_root_below {
                if !new_left_children.is_empty() {
                    new_left_keys.push(
                        separator_at_split
                            .expect("separator must be Some when copied_children non-empty"),
                    );
                } else {
                    debug_assert!(separator_at_split.is_none());
                }
                new_left_children.push(lrb);
            } else {
                drop(separator_at_split);
            }

            left_root_below = if new_left_children.is_empty() {
                None
            } else {
                let new_left = left_arena.alloc_node();
                let n_keys = new_left_keys.len();
                let n_children = new_left_children.len();
                unsafe {
                    let dst_ptr = left_arena.node_ptr(new_left);
                    let header = NodeLayout::<K, V>::header_mut(dst_ptr);
                    header.len = n_keys as u16;
                    header.flags = 0; // internal
                    header.parent = NO_NODE;
                }
                for (i, k) in new_left_keys.into_iter().enumerate() {
                    unsafe {
                        NodeLayout::<K, V>::internal_key_ptr(left_arena.node_ptr(new_left), i)
                            .write(k);
                    }
                }
                for i in 0..n_children {
                    let c = new_left_children[i];
                    unsafe {
                        NodeLayout::<K, V>::internal_child_ptr(
                            left_arena.node_ptr(new_left),
                            i,
                        )
                        .write(c);
                        NodeLayout::<K, V>::header_mut(left_arena.node_ptr(c)).parent = new_left;
                    }
                }
                Some(new_left)
            };

            // Prepend temp chain to main chain (sorted order: temp before main).
            if temp_first_leaf != NO_NODE {
                if left_first_leaf != NO_NODE {
                    unsafe {
                        NodeLayout::<K, V>::leaf_next_ptr(
                            left_arena.node_ptr(temp_chain_tail),
                        )
                        .write(left_first_leaf);
                        NodeLayout::<K, V>::leaf_prev_ptr(
                            left_arena.node_ptr(left_first_leaf),
                        )
                        .write(temp_chain_tail);
                    }
                    left_first_leaf = temp_first_leaf;
                } else {
                    left_first_leaf = temp_first_leaf;
                    left_chain_tail = temp_chain_tail;
                }
                left_count += temp_count;
            }

            // Mutate parent_idx in self.arena: keep right suffix.
            // rrb is always Some in copy-LEFT (boundary leaf is always retained).
            let rrb = right_root_below
                .expect("right_root_below must be Some throughout copy-LEFT spine walk");
            let count = n - child_pos;
            unsafe {
                let parent_node = self.arena.node_ptr(parent_idx);
                if child_pos > 0 && count > 0 {
                    // Shift kept range to start of arrays.
                    // Keys: [child_pos..n] (count elems) → [0..count]
                    // Children: [child_pos+1..=n] (count elems) → [1..=count]
                    std::ptr::copy(
                        NodeLayout::<K, V>::internal_key_ptr(parent_node, child_pos),
                        NodeLayout::<K, V>::internal_key_ptr(parent_node, 0),
                        count,
                    );
                    std::ptr::copy(
                        NodeLayout::<K, V>::internal_child_ptr(parent_node, child_pos + 1),
                        NodeLayout::<K, V>::internal_child_ptr(parent_node, 1),
                        count,
                    );
                }
                NodeLayout::<K, V>::internal_child_ptr(parent_node, 0).write(rrb);
                NodeLayout::<K, V>::header_mut(self.arena.node_ptr(rrb)).parent = parent_idx;
                NodeLayout::<K, V>::header_mut(parent_node).len = count as u16;
            }
            right_root_below = Some(parent_idx);
        }

        // ── Phase 3: collapse degenerates + finalize self (kept = right) ───
        let mut self_root = right_root_below.unwrap_or(NO_NODE);
        if self_root != NO_NODE {
            unsafe {
                NodeLayout::<K, V>::header_mut(self.arena.node_ptr(self_root)).parent = NO_NODE;
            }
            self_root = Self::collapse_along_edge(
                &mut self.arena,
                self_root,
                CollapseEdge::Leftmost,
            );
        }

        // boundary leaf is always the new first leaf of the kept side.
        if self_root != NO_NODE {
            unsafe {
                NodeLayout::<K, V>::leaf_prev_ptr(self.arena.node_ptr(leaf_idx))
                    .write(NO_NODE);
            }
        }
        let new_self_first_leaf = if self_root == NO_NODE { NO_NODE } else { leaf_idx };
        let new_self_last_leaf = if self_root == NO_NODE { NO_NODE } else { self.last_leaf };
        let new_self_height = Self::compute_height(&self.arena, self_root);
        let new_self_len = self.len - left_count;

        self.root = self_root;
        self.first_leaf = new_self_first_leaf;
        self.last_leaf = new_self_last_leaf;
        self.height = new_self_height;
        self.len = new_self_len;

        // ── Phase 4: build the left tree, swap arenas, return right side ───
        let mut left_root = left_root_below.unwrap_or(NO_NODE);
        if left_root != NO_NODE {
            unsafe {
                NodeLayout::<K, V>::header_mut(left_arena.node_ptr(left_root)).parent = NO_NODE;
            }
            left_root = Self::collapse_along_edge(
                &mut left_arena,
                left_root,
                CollapseEdge::Rightmost,
            );
        }
        if left_first_leaf != NO_NODE {
            unsafe {
                NodeLayout::<K, V>::leaf_prev_ptr(left_arena.node_ptr(left_first_leaf))
                    .write(NO_NODE);
            }
        }
        if left_chain_tail != NO_NODE {
            unsafe {
                NodeLayout::<K, V>::leaf_next_ptr(left_arena.node_ptr(left_chain_tail))
                    .write(NO_NODE);
            }
        }

        let left_height = Self::compute_height(&left_arena, left_root);

        // After construction:
        //   self    holds RIGHT side (kept, mutated original arena, >= at)
        //   returned holds LEFT side (deep-copied, < at)
        // Swap so self ends up with LEFT and returned with RIGHT, matching the
        // split_off contract (self keeps < at, return holds >= at).
        let mut returned = RawBTree {
            arena: left_arena,
            root: left_root,
            first_leaf: left_first_leaf,
            last_leaf: left_chain_tail,
            len: left_count,
            height: left_height,
            _marker: PhantomData,
        };
        std::mem::swap(self, &mut returned);
        returned
    }
}

impl<K, V> Drop for RawBTree<K, V> {
    fn drop(&mut self) {
        self.drop_all_contents();
        // Arena's Drop handles deallocation
    }
}
