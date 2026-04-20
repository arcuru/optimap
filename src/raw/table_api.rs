//! Shared trait for all raw hash table engines.
//!
//! `RawTableApi` defines the internal contract that `GenericMap` needs from any
//! raw table backend. Each design (overflow-bit, tombstone, etc.) implements this
//! trait with its own probe strategy, SIMD operations, and growth policy.

use std::hash::{BuildHasher, Hash};

/// Result of a fused find-or-entry probe.
pub enum EntryProbe {
    /// Key was found at (group_index, slot_index).
    Found(usize, usize),
    /// Key was not found. Optional pre-located slot: (gi, si, full_mask).
    /// If None, the table needs to grow before inserting.
    Vacant(Option<(usize, usize, u8)>),
}

/// Internal contract for all raw hash table engines.
///
/// Every method here is `#[inline(always)]`-friendly and fully monomorphized
/// at each call site. GenericMap<K,V,S,R> calls these methods; the concrete
/// R type determines the probe strategy and layout.
pub trait RawTableApi<K, V>: Sized {
    /// Iterator over occupied (group_index, slot_index) pairs.
    type SlotIter<'a>: Iterator<Item = (usize, usize)>
    where
        Self: 'a,
        K: 'a,
        V: 'a;

    /// Owning iterator that consumes the table.
    type IntoIter: Iterator<Item = (K, V)> + ExactSizeIterator;

    // ── Construction ───────────────────────────────────────────────────────

    fn new() -> Self;
    fn with_capacity(cap: usize) -> Self;

    // ── Queries ────────────────────────────────────────────────────────────

    fn len(&self) -> usize;

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn capacity(&self) -> usize;
    fn is_allocated(&self) -> bool;
    fn num_groups(&self) -> usize;
    fn groups_for_capacity(cap: usize) -> usize;

    // ── State mutation ─────────────────────────────────────────────────────

    fn clear(&mut self);

    // ── Slot access ────────────────────────────────────────────────────────

    /// Pointer to the key at (group_index, slot_index).
    ///
    /// # Safety
    /// `gi` and `si` must be within bounds of an allocated table.
    unsafe fn key_ptr(&self, gi: usize, si: usize) -> *const K;

    /// Pointer to the value at (group_index, slot_index).
    ///
    /// # Safety
    /// `gi` and `si` must be within bounds of an allocated table.
    unsafe fn value_ptr(&self, gi: usize, si: usize) -> *mut V;

    // ── Lookups ────────────────────────────────────────────────────────────

    /// Find a key, returning its (group_index, slot_index).
    fn find_by_hash<F: Fn(&K) -> bool>(&self, h: u64, eq: F) -> Option<(usize, usize)>;

    // ── Insert (high-level, with fused fast path) ──────────────────────────

    /// Insert a key-value pair. Returns the old value if the key already existed.
    ///
    /// Each design implements its own fused home-group fast path here.
    fn insert_or_replace<S: BuildHasher>(
        &mut self,
        key: K,
        value: V,
        hb: &S,
    ) -> Option<V>
    where
        K: Hash + Eq;

    // ── Entry support ──────────────────────────────────────────────────────

    /// Fused entry probe: check for existing key with home-group fast path.
    ///
    /// Returns `Found(gi, si)` if the key exists, or `Vacant(slot)` if absent.
    /// When at capacity, returns `Vacant(None)` since pre-located slots would
    /// be invalidated by growth.
    fn find_for_entry(&self, h: u64, key: &K) -> EntryProbe
    where
        K: Eq;

    /// Write a key-value pair into a known-empty slot, setting overflow bits
    /// (or handling tombstone recycling) as appropriate.
    fn insert_at(&mut self, h: u64, gi: usize, si: usize, k: K, v: V, mask: u8);

    /// Insert without checking for duplicates or capacity.
    fn insert_no_check(&mut self, h: u64, k: K, v: V) -> (usize, usize);

    /// Ensure there is room for at least one more insert. Grows if needed.
    fn ensure_capacity<S: BuildHasher>(&mut self, hb: &S)
    where
        K: Hash;

    // ── Removal ────────────────────────────────────────────────────────────

    /// Remove a key, returning the key-value pair if found.
    fn remove_by_hash<F: Fn(&K) -> bool>(&mut self, h: u64, eq: F) -> Option<(K, V)>;

    /// Erase an occupied slot at (gi, si) given the key's hash.
    ///
    /// The bucket contents have already been dropped by the caller.
    /// Overflow-bit designs: set EMPTY + anti-drift.
    /// Tombstone designs: set TOMBSTONE.
    ///
    /// # Safety
    /// The slot at (gi, si) must have been occupied and its contents dropped.
    unsafe fn erase_slot(&mut self, h: u64, gi: usize, si: usize);

    // ── Capacity management ────────────────────────────────────────────────

    fn rehash_with<S: BuildHasher>(&mut self, new_num_groups: usize, hb: &S)
    where
        K: Hash;

    fn reserve<S: BuildHasher>(&mut self, additional: usize, hb: &S)
    where
        K: Hash;

    fn shrink_to_fit<S: BuildHasher>(&mut self, hb: &S)
    where
        K: Hash;

    // ── Iteration ──────────────────────────────────────────────────────────

    /// Iterate over occupied (group_index, slot_index) pairs.
    fn iter_slots(&self) -> Self::SlotIter<'_>;

    /// Consume the table into an owning iterator.
    fn into_iter_impl(self) -> Self::IntoIter;

    /// Drain all entries, leaving the table in a valid empty state.
    fn drain_impl(&mut self) -> Self::IntoIter;

    // ── Retain support ─────────────────────────────────────────────────────

    /// Iterate over all occupied slots for retain/filter operations.
    /// Returns (group_index, slot_index) pairs. Unlike iter_slots, this
    /// collects all positions upfront to allow mutation during iteration.
    fn occupied_positions(&self) -> Vec<(usize, usize)> {
        self.iter_slots().collect()
    }

    // ── Clone support ──────────────────────────────────────────────────────

    /// Clone the raw table. Only available when K and V are Clone.
    fn clone_table(&self) -> Self
    where
        K: Clone,
        V: Clone;
}
