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
        debug_assert!((idx as u32) < self.len);
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

    /// Number of allocated node slots (high-water, not accounting for free list).
    pub fn allocated_nodes(&self) -> u32 {
        self.len
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { alloc::dealloc(self.ptr, Self::layout(self.cap)) };
        }
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
        let leaves = if leaf_cap > 0 {
            (capacity + leaf_cap - 1) / leaf_cap
        } else {
            capacity
        };
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
    /// Search for a key, returning the leaf node index and slot index if found.
    pub fn search<Q>(&self, key: &Q) -> Option<(NodeIdx, usize)>
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        if self.root == NO_NODE {
            return None;
        }

        let mut node_idx = self.root;

        // Navigate internal nodes
        for _ in 0..self.height {
            let node = self.arena.node_ptr(node_idx);
            let header = unsafe { NodeLayout::<K, V>::header(node) };
            let len = header.len as usize;

            // Linear scan to find child index
            let mut child_idx = len; // default: rightmost child
            for i in 0..len {
                let k = unsafe { &*NodeLayout::<K, V>::internal_key_ptr(node, i) };
                if key.cmp(k.borrow()) == std::cmp::Ordering::Less {
                    child_idx = i;
                    break;
                }
            }

            node_idx = unsafe { NodeLayout::<K, V>::internal_child_ptr(node, child_idx).read() };
        }

        // At leaf: linear scan for exact match
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

    /// Search for the leaf where a key should be inserted.
    /// Returns (leaf_idx, insert_position) where insert_position is the
    /// index at which the key should go to maintain sorted order.
    /// Also returns the path of (node_idx, child_index) pairs for split propagation.
    fn search_for_insert(&self, key: &K) -> (NodeIdx, usize, Vec<(NodeIdx, usize)>) {
        debug_assert!(self.root != NO_NODE);

        let mut node_idx = self.root;
        let mut path = Vec::new();

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

        (node_idx, pos, path)
    }
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

        let (leaf_idx, pos, path) = self.search_for_insert(&key);

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
            // Room in leaf: shift right and insert
            self.leaf_insert_at(leaf_idx, pos, key, value);
            self.len += 1;
            None
        } else {
            // Leaf is full: split
            let (promoted_key, new_leaf_idx) =
                self.leaf_split_and_insert(leaf_idx, pos, key, value);
            self.len += 1;

            // Propagate split upward
            self.propagate_split(path, promoted_key, new_leaf_idx);
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
        mut path: Vec<(NodeIdx, usize)>,
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

    /// Remove by equality (O(n) leaf scan). Used by Map trait impl.
    pub fn remove_by_eq<Q>(&mut self, key: &Q) -> Option<V>
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
        Some(self.leaf_remove_at(leaf_idx, slot_idx))
    }

    /// Remove the element at position `idx` in a leaf. No rebalancing (lazy).
    fn leaf_remove_at(&mut self, leaf_idx: NodeIdx, idx: usize) -> V {
        let node = self.arena.node_ptr(leaf_idx);
        let header = unsafe { NodeLayout::<K, V>::header_mut(node) };
        let len = header.len as usize;
        debug_assert!(idx < len);

        unsafe {
            // Read the key and value to return
            let _key = NodeLayout::<K, V>::leaf_key_ptr(node, idx).read();
            let value = NodeLayout::<K, V>::leaf_val_ptr(node, idx).read();

            // Shift remaining elements left
            for i in idx..len - 1 {
                let src_k = NodeLayout::<K, V>::leaf_key_ptr(node, i + 1);
                let dst_k = NodeLayout::<K, V>::leaf_key_ptr(node, i);
                std::ptr::copy_nonoverlapping(src_k, dst_k, 1);

                let src_v = NodeLayout::<K, V>::leaf_val_ptr(node, i + 1);
                let dst_v = NodeLayout::<K, V>::leaf_val_ptr(node, i);
                std::ptr::copy_nonoverlapping(src_v, dst_v, 1);
            }

            header.len = (len - 1) as u16;
            // Drop the key
            drop(_key);
            self.len -= 1;
            value
        }
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

impl<K, V> Drop for RawBTree<K, V> {
    fn drop(&mut self) {
        self.drop_all_contents();
        // Arena's Drop handles deallocation
    }
}
