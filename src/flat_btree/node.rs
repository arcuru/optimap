//! Node memory layout and pointer arithmetic for 256-byte B+ tree nodes.
//!
//! Two node types share the same 256-byte allocation:
//! - **Leaf**: [Header 8B][keys: K × LEAF_CAP][values: V × LEAF_CAP][prev: u32][next: u32]
//! - **Internal**: [Header 8B][keys: K × INTERNAL_CAP][children: u32 × (INTERNAL_CAP + 1)]

use std::marker::PhantomData;

/// Size of each node in bytes (4 cache lines on x86-64).
pub const NODE_SIZE: usize = 256;

/// Index into the arena. u32 supports ~4 billion nodes.
pub type NodeIdx = u32;

/// Sentinel value for "no node".
pub const NO_NODE: NodeIdx = u32::MAX;

/// Node header, shared by leaf and internal nodes.
/// Stored at the start of every 256-byte node block.
#[repr(C)]
pub struct NodeHeader {
    /// Number of keys currently in this node.
    pub len: u16,
    /// Bit 0: is_leaf. Remaining bits reserved.
    pub flags: u16,
    /// Parent node index (NO_NODE for root).
    pub parent: NodeIdx,
}

const HEADER_SIZE: usize = std::mem::size_of::<NodeHeader>();
const _: () = assert!(HEADER_SIZE == 8, "NodeHeader must be exactly 8 bytes");

/// Leaf chain pointers stored at the end of a leaf node.
const LEAF_LINK_SIZE: usize = 2 * std::mem::size_of::<NodeIdx>(); // prev + next = 8 bytes

impl NodeHeader {
    pub const IS_LEAF: u16 = 1;

    #[inline(always)]
    pub fn is_leaf(&self) -> bool {
        self.flags & Self::IS_LEAF != 0
    }
}

/// Compile-time layout information for nodes parameterized by K and V.
pub struct NodeLayout<K, V> {
    _marker: PhantomData<(K, V)>,
}

impl<K, V> NodeLayout<K, V> {
    // Available space after header
    const PAYLOAD: usize = NODE_SIZE - HEADER_SIZE;

    /// Maximum keys in a leaf node.
    /// Layout: [keys: K × N][values: V × N][prev: u32][next: u32]
    pub const LEAF_CAP: usize = {
        let kv_size = std::mem::size_of::<K>() + std::mem::size_of::<V>();
        if kv_size == 0 {
            // ZST keys+values: arbitrary cap, bounded by u16::MAX
            128
        } else {
            (Self::PAYLOAD - LEAF_LINK_SIZE) / kv_size
        }
    };

    /// Maximum keys in an internal node.
    /// Layout: [keys: K × N][children: u32 × (N+1)]
    /// Constraint: N * size_of::<K>() + (N+1) * 4 <= PAYLOAD
    pub const INTERNAL_CAP: usize = {
        let k_plus_child = std::mem::size_of::<K>() + std::mem::size_of::<NodeIdx>();
        if k_plus_child == 0 {
            128
        } else {
            // N * (K + 4) + 4 <= PAYLOAD  →  N <= (PAYLOAD - 4) / (K + 4)
            (Self::PAYLOAD - std::mem::size_of::<NodeIdx>()) / k_plus_child
        }
    };

    // Compile-time assertions
    const _ASSERT_LEAF: () = assert!(
        Self::LEAF_CAP >= 1,
        "K + V too large for 256-byte leaf node. Consider using Box<V>."
    );
    const _ASSERT_INTERNAL: () = assert!(
        Self::INTERNAL_CAP >= 2,
        "K too large for 256-byte internal node. Consider using Box<K>."
    );

    /// Force compile-time assertion evaluation.
    #[inline(always)]
    pub fn assert_capacities() {
        let _ = Self::_ASSERT_LEAF;
        let _ = Self::_ASSERT_INTERNAL;
    }

    // ── Leaf pointer arithmetic ─────────────────────────────────────

    /// Pointer to the i-th key in a leaf node.
    #[inline(always)]
    pub unsafe fn leaf_key_ptr(node: *mut u8, idx: usize) -> *mut K {
        debug_assert!(idx < Self::LEAF_CAP);
        node.add(HEADER_SIZE + idx * std::mem::size_of::<K>())
            .cast::<K>()
    }

    /// Pointer to the i-th value in a leaf node.
    #[inline(always)]
    pub unsafe fn leaf_val_ptr(node: *mut u8, idx: usize) -> *mut V {
        debug_assert!(idx < Self::LEAF_CAP);
        let vals_offset = HEADER_SIZE + Self::LEAF_CAP * std::mem::size_of::<K>();
        node.add(vals_offset + idx * std::mem::size_of::<V>())
            .cast::<V>()
    }

    /// Pointer to the `prev` leaf link.
    #[inline(always)]
    pub unsafe fn leaf_prev_ptr(node: *mut u8) -> *mut NodeIdx {
        node.add(NODE_SIZE - LEAF_LINK_SIZE).cast::<NodeIdx>()
    }

    /// Pointer to the `next` leaf link.
    #[inline(always)]
    pub unsafe fn leaf_next_ptr(node: *mut u8) -> *mut NodeIdx {
        node.add(NODE_SIZE - std::mem::size_of::<NodeIdx>())
            .cast::<NodeIdx>()
    }

    // ── Internal pointer arithmetic ─────────────────────────────────

    /// Pointer to the i-th key in an internal node.
    #[inline(always)]
    pub unsafe fn internal_key_ptr(node: *mut u8, idx: usize) -> *mut K {
        debug_assert!(idx < Self::INTERNAL_CAP);
        node.add(HEADER_SIZE + idx * std::mem::size_of::<K>())
            .cast::<K>()
    }

    /// Pointer to the i-th child in an internal node.
    /// Internal nodes have INTERNAL_CAP + 1 children (one more than keys).
    #[inline(always)]
    pub unsafe fn internal_child_ptr(node: *mut u8, idx: usize) -> *mut NodeIdx {
        debug_assert!(idx <= Self::INTERNAL_CAP);
        let children_offset = HEADER_SIZE + Self::INTERNAL_CAP * std::mem::size_of::<K>();
        node.add(children_offset + idx * std::mem::size_of::<NodeIdx>())
            .cast::<NodeIdx>()
    }

    // ── Header access ───────────────────────────────────────────────

    /// Read the header from a node.
    #[inline(always)]
    pub unsafe fn header(node: *const u8) -> &'static NodeHeader {
        &*node.cast::<NodeHeader>()
    }

    /// Mutable header access.
    #[inline(always)]
    pub unsafe fn header_mut(node: *mut u8) -> &'static mut NodeHeader {
        &mut *node.cast::<NodeHeader>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_u64_u64() {
        // K=8, V=8: (248 - 8) / 16 = 15 leaf, (248 - 4) / 12 = 20 internal
        assert_eq!(NodeLayout::<u64, u64>::LEAF_CAP, 15);
        assert_eq!(NodeLayout::<u64, u64>::INTERNAL_CAP, 20);
    }

    #[test]
    fn capacity_u32_u32() {
        // K=4, V=4: (248 - 8) / 8 = 30 leaf, (248 - 4) / 8 = 30 internal
        assert_eq!(NodeLayout::<u32, u32>::LEAF_CAP, 30);
        assert_eq!(NodeLayout::<u32, u32>::INTERNAL_CAP, 30);
    }

    #[test]
    fn capacity_string_string() {
        // K=24, V=24: (248 - 8) / 48 = 5 leaf, (248 - 4) / 28 = 8 internal
        assert_eq!(NodeLayout::<String, String>::LEAF_CAP, 5);
        assert_eq!(NodeLayout::<String, String>::INTERNAL_CAP, 8);
    }

    #[test]
    fn capacity_u64_u128() {
        // K=8, V=16: (248 - 8) / 24 = 10 leaf, (248 - 4) / 12 = 20 internal
        assert_eq!(NodeLayout::<u64, u128>::LEAF_CAP, 10);
        assert_eq!(NodeLayout::<u64, u128>::INTERNAL_CAP, 20);
    }

    #[test]
    fn capacity_u8_u8() {
        // K=1, V=1: (248 - 8) / 2 = 120 leaf, (248 - 4) / 5 = 48 internal
        assert_eq!(NodeLayout::<u8, u8>::LEAF_CAP, 120);
        assert_eq!(NodeLayout::<u8, u8>::INTERNAL_CAP, 48);
    }

    #[test]
    fn header_size() {
        assert_eq!(std::mem::size_of::<NodeHeader>(), 8);
    }

    #[test]
    fn leaf_layout_fits_in_node() {
        // Verify leaf layout doesn't exceed NODE_SIZE for u64/u64
        type L = NodeLayout<u64, u64>;
        let keys_end = HEADER_SIZE + L::LEAF_CAP * 8;
        let vals_end = keys_end + L::LEAF_CAP * 8;
        let total = vals_end + LEAF_LINK_SIZE;
        assert!(
            total <= NODE_SIZE,
            "leaf layout overflows: {total} > {NODE_SIZE}"
        );
    }

    #[test]
    fn internal_layout_fits_in_node() {
        // Verify internal layout doesn't exceed NODE_SIZE for u64/u64
        type L = NodeLayout<u64, u64>;
        let keys_end = HEADER_SIZE + L::INTERNAL_CAP * 8;
        let children_end = keys_end + (L::INTERNAL_CAP + 1) * 4;
        assert!(
            children_end <= NODE_SIZE,
            "internal layout overflows: {children_end} > {NODE_SIZE}"
        );
    }
}
