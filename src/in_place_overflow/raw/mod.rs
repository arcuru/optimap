pub mod group;

use std::alloc::{self, Layout};
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::ptr;

use crate::raw::bitmask;
use crate::raw::hash;
use crate::raw::kv_storage::{AoS, KvStorage};
use crate::raw::tag_strategy::{Byte7_254, TombstoneTag};
use group::{EMPTY, GROUP_SIZE, Group, META_GROUP_BYTES, TOMBSTONE};

/// Static sentinel for empty tables: 16-byte-aligned zeros.
/// SIMD loads on this produce all-EMPTY matches, terminating probes immediately.
/// Avoids a branch on every lookup for unallocated tables.
#[repr(align(16))]
struct EmptySentinel([u8; 16]);
static EMPTY_SENTINEL: EmptySentinel = EmptySentinel([0; 16]);

/// Result of a fused find-or-locate probe.
pub(crate) enum ProbeResult {
    /// Key was found at (group_index, slot_index).
    Found(usize, usize),
    /// Key was not found; first available slot at (group_index, slot_index).
    /// The u8 field is unused in this design (always 0).
    InsertSlot(usize, usize, u8),
    /// Key was not found; no available slot was encountered during the probe.
    /// Caller must fall back to insert_no_check for a full probe.
    NotFound,
}

/// Maximum load factor (fixed at 7/8 = 0.875).
const MAX_LOAD_FACTOR_NUM: usize = 7;
const MAX_LOAD_FACTOR_DEN: usize = 8;

#[inline(always)]
fn max_load_for_capacity(capacity: usize) -> usize {
    capacity * MAX_LOAD_FACTOR_NUM / MAX_LOAD_FACTOR_DEN
}

/// The core hash table engine (16-slot groups, tombstone-based deletion).
///
/// # Memory layout: mid-pointer design (inspired by hashbrown)
///
/// A single allocation holds buckets and metadata, with `ctrl` pointing
/// to the boundary between them:
///
/// ```text
///   low addresses ──────────────────────────────────────► high addresses
///
///   ┌─────────────────────────────────────┬──────────────────────────────┐
///   │ Buckets (KV pairs)                  │ Metadata (control bytes)     │
///   │                                     │                              │
///   │ [slot N-1] ... [slot 1] [slot 0]    │ [ctrl 0..15] [ctrl 16..31]...│
///   │ ◄── addressed via negative offset   │ addressed via positive ──►   │
///   └─────────────────────────────────────┴──────────────────────────────┘
///   ↑ alloc_ptr (computed for dealloc)     ↑ ctrl (the only stored pointer)
/// ```
///
/// - **Metadata**: `ctrl[gi * 16 + si]` — forward from `ctrl`.
/// - **Buckets**: `ctrl.cast::<(K,V)>().sub(slot_index + 1)` — backward from `ctrl`.
///   Slot index is `gi * 16 + si`, so slot 0 is right before `ctrl`.
///
/// This eliminates a struct field and an address computation in the hot path:
/// both metadata and bucket access derive from the single `ctrl` pointer,
/// just in opposite directions. hashbrown uses the same trick.
///
/// ## Why this works for tombstone designs but not overflow-bit designs
///
/// Tombstone designs have exactly two memory regions (metadata + buckets),
/// so one mid-pointer serves both directions. Overflow-bit designs
/// (Splitsies, UFM, matrix variants) have a third region (overflow bytes)
/// between metadata and buckets — no single pointer can serve all three
/// without adding offset computations for the third region.
pub struct RawTable<K, V, T: TombstoneTag = Byte7_254, S: KvStorage<K, V> = AoS> {
    /// num_groups - 1. Used for probe wraparound: `gi & mask`.
    pub(crate) mask: usize,
    /// Points to the boundary between buckets (backward) and metadata (forward).
    /// For unallocated tables, points to EMPTY_SENTINEL.
    pub(crate) ctrl: *mut u8,
    /// Extra storage state. AoS: `()` (zero-size). SoA: `*mut u8` (values pointer).
    pub(crate) extra: S::Extra,
    pub(crate) len: usize,
    /// Number of EMPTY slots remaining before we must grow or rehash.
    pub(crate) growth_left: usize,
    /// Maximum number of entries before growth is required. Zero when unallocated.
    pub(crate) max_load: usize,
    _marker: PhantomData<(K, V, T, S)>,
}

impl<K, V, T: TombstoneTag, S: KvStorage<K, V>> RawTable<K, V, T, S> {
    pub fn new() -> Self {
        RawTable {
            mask: 0,
            ctrl: EMPTY_SENTINEL.0.as_ptr() as *mut u8,
            extra: S::extra_null(),
            len: 0,
            growth_left: 0,
            max_load: 0,
            _marker: PhantomData,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::new();
        }
        let mut table = Self::new();
        let num_groups = Self::groups_for_capacity(capacity);
        table.allocate(num_groups);
        table
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn capacity(&self) -> usize {
        if self.is_allocated() {
            self.num_groups() * GROUP_SIZE
        } else {
            0
        }
    }

    /// Number of groups. Derived from mask.
    #[inline(always)]
    pub(crate) fn num_groups(&self) -> usize {
        self.mask + 1
    }

    /// Whether the table has a real allocation (not the empty sentinel).
    #[inline(always)]
    pub(crate) fn is_allocated(&self) -> bool {
        self.max_load > 0
    }

    pub(crate) fn groups_for_capacity(capacity: usize) -> usize {
        let min_slots =
            (capacity * MAX_LOAD_FACTOR_DEN + MAX_LOAD_FACTOR_NUM - 1) / MAX_LOAD_FACTOR_NUM;
        let min_groups = (min_slots + GROUP_SIZE - 1) / GROUP_SIZE;
        min_groups.next_power_of_two()
    }

    #[inline(always)]
    fn backward_size(num_groups: usize) -> usize {
        S::backward_size(num_groups * GROUP_SIZE)
    }

    fn values_offset(num_groups: usize) -> usize {
        let meta_size = num_groups * META_GROUP_BYTES;
        let val_align = S::values_align();
        (meta_size + val_align - 1) & !(val_align - 1)
    }

    fn combined_layout(num_groups: usize) -> Layout {
        let backward = Self::backward_size(num_groups);
        let values_offset = Self::values_offset(num_groups);
        let values_size = S::values_region_size(num_groups * GROUP_SIZE);
        let total_size = backward + values_offset + values_size;
        let align = S::alloc_align();
        Layout::from_size_align(total_size.max(align), align).unwrap()
    }

    pub(crate) fn allocate(&mut self, num_groups: usize) {
        debug_assert!(num_groups.is_power_of_two());

        let layout = Self::combined_layout(num_groups);
        let backward = Self::backward_size(num_groups);
        let values_offset = Self::values_offset(num_groups);
        let meta_size = num_groups * META_GROUP_BYTES;
        let total_buckets = num_groups * GROUP_SIZE;

        unsafe {
            let alloc_ptr = alloc::alloc(layout);
            if alloc_ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }

            self.ctrl = alloc_ptr.add(backward);
            self.extra = S::init_extra(self.ctrl, values_offset);

            // Zero metadata only
            ptr::write_bytes(self.ctrl, 0, meta_size);
        }

        self.mask = num_groups - 1;
        self.max_load = max_load_for_capacity(total_buckets);
        self.growth_left = self.max_load;
    }

    #[inline(always)]
    unsafe fn alloc_ptr(&self) -> *mut u8 {
        unsafe { self.ctrl.sub(Self::backward_size(self.num_groups())) }
    }

    unsafe fn deallocate(&mut self) {
        if !self.is_allocated() {
            return;
        }
        let layout = Self::combined_layout(self.num_groups());
        unsafe {
            alloc::dealloc(self.alloc_ptr(), layout);
        }
        self.ctrl = EMPTY_SENTINEL.0.as_ptr() as *mut u8;
        self.extra = S::extra_null();
        self.max_load = 0;
    }

    /// Map hash to group index via AND (like hashbrown's h1).
    /// Single instruction vs shift-based indexing.
    /// Tag strategies must use bits outside the mask range to avoid correlation.
    #[inline(always)]
    pub(crate) fn group_index(&self, h: u64) -> usize {
        (h as usize) & self.mask
    }

    /// Pointer to group metadata (16-byte aligned). Forward from ctrl.
    #[inline(always)]
    pub(crate) unsafe fn meta_ptr(&self, gi: usize) -> *mut u8 {
        unsafe { self.ctrl.add(gi * META_GROUP_BYTES) }
    }

    #[inline(always)]
    fn bucket_index(gi: usize, si: usize) -> usize {
        (gi << 4) | si
    }

    #[inline(always)]
    pub(crate) unsafe fn key_ptr_impl(&self, gi: usize, si: usize) -> *mut K {
        unsafe { S::key_ptr(self.ctrl, Self::bucket_index(gi, si)) }
    }

    #[inline(always)]
    pub(crate) unsafe fn value_ptr_impl(&self, gi: usize, si: usize) -> *mut V {
        unsafe { S::value_ptr(self.ctrl, self.extra, Self::bucket_index(gi, si)) }
    }

    #[inline(always)]
    fn hash_key<H: BuildHasher>(key: &K, hash_builder: &H) -> u64
    where
        K: Hash,
    {
        hash::hash_no_mix(key, hash_builder)
    }

    /// Find a key in the table.
    pub fn find<H: BuildHasher>(&self, key: &K, hash_builder: &H) -> Option<(usize, usize)>
    where
        K: Hash + Eq,
    {
        let h = Self::hash_key(key, hash_builder);
        self.find_by_hash(h, |k| k == key)
    }

    pub(crate) fn find_with_hash(&self, key: &K, h: u64) -> Option<(usize, usize)>
    where
        K: Eq,
    {
        self.find_by_hash(h, |k| k == key)
    }

    /// Core lookup: SIMD match + EMPTY-based probe termination + prefetch.
    #[inline(always)]
    pub(crate) fn find_by_hash<F>(&self, h: u64, eq: F) -> Option<(usize, usize)>
    where
        F: Fn(&K) -> bool,
    {
        let reduced = T::reduced_hash(h);
        let mut gi = self.group_index(h);
        let mut probe = 0usize;

        loop {
            let meta = unsafe { self.meta_ptr(gi) };

            let data = unsafe { Group::load(meta) };

            for si in unsafe { Group::loaded_match_byte(data, reduced) } {
                let key = unsafe { &*self.key_ptr_impl(gi, si) };
                if eq(key) {
                    return Some((gi, si));
                }
            }

            if unsafe { Group::loaded_match_empty(data).any_set() } {
                return None;
            }

            probe += 1;
            gi = (gi.wrapping_add(probe)) & self.mask;

            unsafe {
                Group::prefetch_read(self.meta_ptr(gi) as *const u8);
                Group::prefetch_read(self.key_ptr_impl(gi, 0) as *const u8);
            }
        }
    }

    pub(crate) fn remove_by_hash<F>(&mut self, h: u64, eq: F) -> Option<(K, V)>
    where
        F: Fn(&K) -> bool,
    {
        let (gi, si) = self.find_by_hash(h, eq)?;

        unsafe {
            let bucket = S::read(self.ctrl, self.extra, Self::bucket_index(gi, si));

            let meta = self.meta_ptr(gi);
            Group::set_meta(meta, si, TOMBSTONE);

            self.len -= 1;

            Some(bucket)
        }
    }

    /// Insert without checking for duplicates or capacity.
    /// Probes until an EMPTY slot is found, tracking the first tombstone seen.
    #[inline(always)]
    pub(crate) fn insert_no_check(&mut self, h: u64, key: K, value: V) -> (usize, usize) {
        let reduced = T::reduced_hash(h);
        let mut gi = self.group_index(h);
        let mut probe = 0usize;
        let mut first_tombstone: Option<(usize, usize)> = None;

        loop {
            let meta = unsafe { self.meta_ptr(gi) };

            // Track first tombstone slot
            if first_tombstone.is_none()
                && let Some(si) = unsafe { Group::match_byte(meta, TOMBSTONE) }.lowest_set_bit()
            {
                first_tombstone = Some((gi, si));
            }

            // Check for EMPTY slot — this is our termination condition
            if let Some(si) = unsafe { Group::match_empty(meta) }.lowest_set_bit() {
                let (ins_gi, ins_si, decrement) = if let Some((tgi, tsi)) = first_tombstone {
                    (tgi, tsi, false)
                } else {
                    (gi, si, true)
                };

                unsafe {
                    let ins_meta = self.meta_ptr(ins_gi);
                    Group::set_meta(ins_meta, ins_si, reduced);
                    S::write(self.ctrl, self.extra, Self::bucket_index(ins_gi, ins_si), key, value);
                }
                self.len += 1;
                if decrement {
                    self.growth_left -= 1;
                }
                return (ins_gi, ins_si);
            }

            probe += 1;
            gi = (gi.wrapping_add(probe)) & self.mask;
        }
    }

    /// Fused find-or-locate: probes for the key and tracks the first available slot.
    #[inline(always)]
    pub(crate) fn find_or_locate<F>(&self, h: u64, eq: F) -> ProbeResult
    where
        F: Fn(&K) -> bool,
    {
        let reduced = T::reduced_hash(h);
        let gi = self.group_index(h);

        // Home group fast path
        let meta = unsafe { self.meta_ptr(gi) };
        let (matches, empties) = unsafe { Group::match_byte_and_empty(meta, reduced) };

        for si in matches {
            let key = unsafe { &*self.key_ptr_impl(gi, si) };
            if eq(key) {
                return ProbeResult::Found(gi, si);
            }
        }

        if let Some(si) = empties.lowest_set_bit() {
            if let Some(tsi) = unsafe { Group::match_byte(meta, TOMBSTONE) }.lowest_set_bit() {
                return ProbeResult::InsertSlot(gi, tsi, 0);
            }
            return ProbeResult::InsertSlot(gi, si, 0);
        }

        let first_tombstone = unsafe { Group::match_byte(meta, TOMBSTONE) }
            .lowest_set_bit()
            .map(|tsi| (gi, tsi));

        self.find_or_locate_overflow(h, eq, reduced, gi, first_tombstone)
    }

    /// Slow path for find_or_locate when home group has no EMPTY slots.
    #[inline(never)]
    fn find_or_locate_overflow<F>(
        &self,
        _h: u64,
        eq: F,
        reduced: u8,
        home_gi: usize,
        mut first_available: Option<(usize, usize)>,
    ) -> ProbeResult
    where
        F: Fn(&K) -> bool,
    {
        let mut probe = 1usize;
        let mut gi = (home_gi.wrapping_add(probe)) & self.mask;

        loop {
            let meta = unsafe { self.meta_ptr(gi) };
            let (matches, empties) = unsafe { Group::match_byte_and_empty(meta, reduced) };

            for si in matches {
                let key = unsafe { &*self.key_ptr_impl(gi, si) };
                if eq(key) {
                    return ProbeResult::Found(gi, si);
                }
            }

            if first_available.is_none() {
                if let Some(tsi) = unsafe { Group::match_byte(meta, TOMBSTONE) }.lowest_set_bit() {
                    first_available = Some((gi, tsi));
                } else if let Some(si) = empties.lowest_set_bit() {
                    first_available = Some((gi, si));
                }
            }

            if empties.any_set() {
                return match first_available {
                    Some((ins_gi, ins_si)) => ProbeResult::InsertSlot(ins_gi, ins_si, 0),
                    None => ProbeResult::NotFound,
                };
            }

            probe += 1;
            gi = (gi.wrapping_add(probe)) & self.mask;

            unsafe {
                Group::prefetch_read(self.meta_ptr(gi) as *const u8);
                Group::prefetch_read(self.key_ptr_impl(gi, 0) as *const u8);
            }
        }
    }

    /// Write a key-value pair into a known-available slot.
    #[inline(always)]
    pub(crate) fn insert_at(
        &mut self,
        h: u64,
        gi: usize,
        si: usize,
        key: K,
        value: V,
        _full_mask: u8,
    ) {
        let reduced = T::reduced_hash(h);

        unsafe {
            let meta = self.meta_ptr(gi);
            let old_meta = Group::get_meta(meta, si);
            Group::set_meta(meta, si, reduced);
            S::write(self.ctrl, self.extra, Self::bucket_index(gi, si), key, value);

            if old_meta == EMPTY {
                self.growth_left -= 1;
            }
        }
        self.len += 1;
    }

    pub fn rehash_with<H: BuildHasher>(&mut self, new_num_groups: usize, hash_builder: &H)
    where
        K: Hash,
    {
        let was_allocated = self.is_allocated();
        let old_num_groups = self.num_groups();
        let old_ctrl = self.ctrl;
        let old_layout = if was_allocated {
            Some(Self::combined_layout(old_num_groups))
        } else {
            None
        };
        let old_backward = if was_allocated {
            Self::backward_size(old_num_groups)
        } else {
            0
        };
        let old_extra = self.extra;

        self.ctrl = EMPTY_SENTINEL.0.as_ptr() as *mut u8;
        self.extra = S::extra_null();
        self.mask = 0;
        self.len = 0;
        self.max_load = 0;
        self.allocate(new_num_groups);

        if !was_allocated {
            return;
        }

        unsafe {
            for gi in 0..old_num_groups {
                let group_meta = old_ctrl.add(gi * META_GROUP_BYTES);
                for si in Group::match_occupied(group_meta) {
                    let idx = (gi << 4) | si;
                    let (key, value) = S::read(old_ctrl, old_extra, idx);
                    let h = Self::hash_key(&key, hash_builder);
                    self.insert_no_check(h, key, value);
                }
            }

            let old_alloc = old_ctrl.sub(old_backward);
            alloc::dealloc(old_alloc, old_layout.unwrap());
        }
    }

    pub fn insert_with_rehash<H: BuildHasher>(
        &mut self,
        key: K,
        value: V,
        hash_builder: &H,
    ) -> (&mut V, bool)
    where
        K: Hash + Eq,
    {
        if !self.is_allocated() {
            self.allocate(1);
        }

        let h = Self::hash_key(&key, hash_builder);

        if let Some((gi, si)) = self.find_with_hash(&key, h) {
            drop(key);
            drop(value);
            let v = unsafe { &mut *self.value_ptr_impl(gi, si) };
            return (v, false);
        }

        if self.growth_left == 0 {
            self.grow_or_rehash(hash_builder);
        }

        let (gi, si) = self.insert_no_check(h, key, value);
        let v = unsafe { &mut *self.value_ptr_impl(gi, si) };
        (v, true)
    }

    /// If len >= capacity * 7/8, grow (double). Else rehash in place (compact tombstones).
    fn grow_or_rehash<H: BuildHasher>(&mut self, hash_builder: &H)
    where
        K: Hash,
    {
        let new_groups = if !self.is_allocated() {
            1
        } else {
            let cap = self.num_groups() * GROUP_SIZE;
            if self.len >= max_load_for_capacity(cap) {
                self.num_groups() * 2
            } else {
                self.num_groups()
            }
        };
        self.rehash_with(new_groups, hash_builder);
    }

    pub fn remove<H: BuildHasher>(&mut self, key: &K, hash_builder: &H) -> Option<V>
    where
        K: Hash + Eq,
    {
        if !self.is_allocated() {
            return None;
        }
        let h = Self::hash_key(key, hash_builder);
        let (gi, si) = self.find_with_hash(key, h)?;

        unsafe {
            let bucket = S::read(self.ctrl, self.extra, Self::bucket_index(gi, si));
            let meta = self.meta_ptr(gi);
            Group::set_meta(meta, si, TOMBSTONE);
            self.len -= 1;

            let (_k, v) = bucket;
            Some(v)
        }
    }

    pub fn get<H: BuildHasher>(&self, key: &K, hash_builder: &H) -> Option<&V>
    where
        K: Hash + Eq,
    {
        let (gi, si) = self.find(key, hash_builder)?;
        Some(unsafe { &*self.value_ptr_impl(gi, si) })
    }

    pub fn get_mut<H: BuildHasher>(&mut self, key: &K, hash_builder: &H) -> Option<&mut V>
    where
        K: Hash + Eq,
    {
        let (gi, si) = self.find(key, hash_builder)?;
        Some(unsafe { &mut *self.value_ptr_impl(gi, si) })
    }

    pub fn clear(&mut self) {
        if !self.is_allocated() {
            return;
        }

        unsafe {
            if S::needs_drop() {
                for gi in 0..self.num_groups() {
                    let group_meta = self.ctrl.add(gi * META_GROUP_BYTES);
                    for si in Group::match_occupied(group_meta) {
                        S::drop_slot(self.ctrl, self.extra, Self::bucket_index(gi, si));
                    }
                }
            }

            let meta_size = self.num_groups() * META_GROUP_BYTES;
            ptr::write_bytes(self.ctrl, 0, meta_size);
        }

        self.len = 0;
        self.growth_left = max_load_for_capacity(self.num_groups() * GROUP_SIZE);
    }

    /// Iterate over all occupied slots using SIMD to skip empty/tombstone groups.
    pub fn iter_slots(&self) -> SlotIter<'_, K, V, T, S> {
        SlotIter {
            table: self,
            group: 0,
            current_mask: if !self.is_allocated() {
                bitmask::BitMask(0)
            } else {
                unsafe { Group::match_occupied(self.ctrl) }
            },
        }
    }
}

impl<K, V, T: TombstoneTag, S: KvStorage<K, V>> Drop for RawTable<K, V, T, S> {
    fn drop(&mut self) {
        if !self.is_allocated() {
            return;
        }
        if S::needs_drop() {
            unsafe {
                for gi in 0..self.num_groups() {
                    let group_meta = self.ctrl.add(gi * META_GROUP_BYTES);
                    for si in Group::match_occupied(group_meta) {
                        S::drop_slot(self.ctrl, self.extra, Self::bucket_index(gi, si));
                    }
                }
            }
        }
        unsafe {
            self.deallocate();
        }
    }
}

unsafe impl<K: Send, V: Send, T: TombstoneTag, S: KvStorage<K, V>> Send for RawTable<K, V, T, S> {}
unsafe impl<K: Sync, V: Sync, T: TombstoneTag, S: KvStorage<K, V>> Sync for RawTable<K, V, T, S> {}

impl<K: Clone, V: Clone, T: TombstoneTag, S: KvStorage<K, V>> Clone for RawTable<K, V, T, S> {
    fn clone(&self) -> Self {
        if !self.is_allocated() {
            return Self::new();
        }

        let mut new_table = Self::new();
        new_table.allocate(self.num_groups());

        unsafe {
            // Copy metadata
            let meta_size = self.num_groups() * META_GROUP_BYTES;
            ptr::copy_nonoverlapping(self.ctrl, new_table.ctrl, meta_size);

            for gi in 0..self.num_groups() {
                let group_meta = self.ctrl.add(gi * META_GROUP_BYTES);
                for si in Group::match_occupied(group_meta) {
                    let idx = Self::bucket_index(gi, si);
                    S::clone_slot(self.ctrl, self.extra, new_table.ctrl, new_table.extra, idx);
                }
            }
        }

        new_table.len = self.len;
        new_table.growth_left = self.growth_left;
        new_table
    }
}

/// SIMD-accelerated iterator over occupied slot positions.
pub struct SlotIter<'a, K, V, T: TombstoneTag = Byte7_254, S: KvStorage<K, V> = AoS> {
    pub(crate) table: &'a RawTable<K, V, T, S>,
    group: usize,
    current_mask: bitmask::BitMask,
}

impl<'a, K, V, T: TombstoneTag, S: KvStorage<K, V>> Iterator for SlotIter<'a, K, V, T, S> {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<(usize, usize)> {
        loop {
            if let Some(si) = self.current_mask.next() {
                return Some((self.group, si));
            }
            self.group += 1;
            if self.group > self.table.mask {
                return None;
            }
            self.current_mask = unsafe { Group::match_occupied(self.table.meta_ptr(self.group)) };
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.table.len))
    }
}

// ── IntoIter ───────────────────────────────────────────────────────────────

pub struct IntoIter<K, V, T: TombstoneTag = Byte7_254, S: KvStorage<K, V> = AoS> {
    table: RawTable<K, V, T, S>,
    group: usize,
    current_mask: bitmask::BitMask,
}

impl<K, V, T: TombstoneTag, S: KvStorage<K, V>> Iterator for IntoIter<K, V, T, S> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(si) = self.current_mask.next() {
                let gi = self.group;
                unsafe {
                    let idx = RawTable::<K, V, T, S>::bucket_index(gi, si);
                    let kv = S::read(self.table.ctrl, self.table.extra, idx);
                    let meta = self.table.ctrl.add(gi * META_GROUP_BYTES + si);
                    *meta = EMPTY;
                    self.table.len -= 1;
                    return Some(kv);
                }
            }
            self.group += 1;
            if self.group > self.table.mask {
                return None;
            }
            self.current_mask = unsafe {
                Group::match_occupied(self.table.ctrl.add(self.group * META_GROUP_BYTES))
            };
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.table.len, Some(self.table.len))
    }
}

impl<K, V, T: TombstoneTag, S: KvStorage<K, V>> ExactSizeIterator for IntoIter<K, V, T, S> {}
impl<K, V, T: TombstoneTag, S: KvStorage<K, V>> std::iter::FusedIterator for IntoIter<K, V, T, S> {}

// ── RawTableApi ────────────────────────────────────────────────────────────

use crate::raw::table_api::{EntryProbe, RawTableApi};

impl<K, V, T: TombstoneTag, S: KvStorage<K, V>> RawTableApi<K, V> for RawTable<K, V, T, S> {
    type SlotIter<'a> = SlotIter<'a, K, V, T, S> where K: 'a, V: 'a;
    type IntoIter = IntoIter<K, V, T, S>;

    fn new() -> Self { RawTable::new() }
    fn with_capacity(cap: usize) -> Self { RawTable::with_capacity(cap) }

    #[inline(always)]
    fn len(&self) -> usize { self.len }

    #[inline(always)]
    fn capacity(&self) -> usize {
        if self.is_allocated() { self.num_groups() * GROUP_SIZE } else { 0 }
    }

    #[inline(always)]
    fn is_allocated(&self) -> bool { self.max_load > 0 }

    #[inline(always)]
    fn num_groups(&self) -> usize { self.mask + 1 }

    fn groups_for_capacity(capacity: usize) -> usize {
        let min_slots = (capacity * MAX_LOAD_FACTOR_DEN + MAX_LOAD_FACTOR_NUM - 1) / MAX_LOAD_FACTOR_NUM;
        let min_groups = (min_slots + GROUP_SIZE - 1) / GROUP_SIZE;
        min_groups.next_power_of_two()
    }

    fn clear(&mut self) { self.clear(); }

    #[inline(always)]
    unsafe fn key_ptr(&self, gi: usize, si: usize) -> *const K {
        unsafe { self.key_ptr_impl(gi, si) }
    }

    #[inline(always)]
    unsafe fn value_ptr(&self, gi: usize, si: usize) -> *mut V {
        unsafe { self.value_ptr_impl(gi, si) }
    }

    #[inline(always)]
    fn find_by_hash<F: Fn(&K) -> bool>(&self, h: u64, eq: F) -> Option<(usize, usize)> {
        self.find_by_hash(h, eq)
    }

    fn insert_or_replace<H: BuildHasher>(&mut self, key: K, value: V, hb: &H) -> Option<V>
    where K: Hash + Eq,
    {
        if !self.is_allocated() {
            self.allocate(1);
        }
        let h = hash::hash_no_mix(&key, hb);

        if self.growth_left == 0 {
            if let Some((gi, si)) = self.find_by_hash(h, |k| k == &key) {
                let v = unsafe { &mut *self.value_ptr_impl(gi, si) };
                return Some(std::mem::replace(v, value));
            }
            self.grow_or_rehash(hb);
            self.insert_no_check(h, key, value);
            return None;
        }

        // Fused home-group fast path
        let reduced = T::reduced_hash(h);
        let gi = self.group_index(h);
        let meta = unsafe { self.meta_ptr(gi) };
        let (matches, empties) = unsafe { Group::match_byte_and_empty(meta, reduced) };

        for si in matches {
            let k = unsafe { &*self.key_ptr_impl(gi, si) };
            if *k == key {
                let v = unsafe { &mut *self.value_ptr_impl(gi, si) };
                return Some(std::mem::replace(v, value));
            }
        }

        if let Some(si) = empties.lowest_set_bit() {
            unsafe {
                Group::set_meta(meta, si, reduced);
                S::write(self.ctrl, self.extra, Self::bucket_index(gi, si), key, value);
            }
            self.len += 1;
            self.growth_left -= 1;
            return None;
        }

        // No EMPTY in home group — full probe
        if let Some((gi, si)) = self.find_by_hash(h, |k| k == &key) {
            let v = unsafe { &mut *self.value_ptr_impl(gi, si) };
            return Some(std::mem::replace(v, value));
        }
        if self.growth_left == 0 {
            self.grow_or_rehash(hb);
        }
        self.insert_no_check(h, key, value);
        None
    }

    fn find_for_entry(&self, h: u64, key: &K) -> EntryProbe
    where K: Eq,
    {
        if self.growth_left == 0 {
            if let Some((gi, si)) = self.find_by_hash(h, |k| k == key) {
                return EntryProbe::Found(gi, si);
            }
            return EntryProbe::Vacant(None);
        }

        let reduced = T::reduced_hash(h);
        let gi = self.group_index(h);
        let meta = unsafe { self.meta_ptr(gi) };
        let (matches, empties) = unsafe { Group::match_byte_and_empty(meta, reduced) };

        for si in matches {
            let k = unsafe { &*self.key_ptr_impl(gi, si) };
            if *k == *key {
                return EntryProbe::Found(gi, si);
            }
        }

        if let Some(si) = empties.lowest_set_bit() {
            return EntryProbe::Vacant(Some((gi, si, 0)));
        }

        match self.find_or_locate(h, |k| k == key) {
            ProbeResult::Found(gi, si) => EntryProbe::Found(gi, si),
            ProbeResult::InsertSlot(gi, si, mask) => EntryProbe::Vacant(Some((gi, si, mask))),
            ProbeResult::NotFound => EntryProbe::Vacant(None),
        }
    }

    #[inline(always)]
    fn insert_at(&mut self, h: u64, gi: usize, si: usize, k: K, v: V, _mask: u8) {
        self.insert_at(h, gi, si, k, v, _mask);
    }

    #[inline(always)]
    fn insert_no_check(&mut self, h: u64, k: K, v: V) -> (usize, usize) {
        self.insert_no_check(h, k, v)
    }

    fn ensure_capacity<H: BuildHasher>(&mut self, hb: &H) where K: Hash {
        if self.growth_left == 0 {
            self.grow_or_rehash(hb);
        }
    }

    fn remove_by_hash<F: Fn(&K) -> bool>(&mut self, h: u64, eq: F) -> Option<(K, V)> {
        self.remove_by_hash(h, eq)
    }

    unsafe fn erase_slot(&mut self, _h: u64, gi: usize, si: usize) {
        unsafe {
            let meta = self.meta_ptr(gi);
            Group::set_meta(meta, si, TOMBSTONE);
            self.len -= 1;
        }
    }

    fn reserve<H: BuildHasher>(&mut self, additional: usize, hb: &H) where K: Hash {
        let needed = self.len.checked_add(additional).expect("capacity overflow");
        if !self.is_allocated() {
            if additional > 0 {
                self.allocate(Self::groups_for_capacity(needed));
            }
            return;
        }
        if needed > max_load_for_capacity(self.num_groups() * GROUP_SIZE) {
            let new_groups = Self::groups_for_capacity(needed);
            if new_groups > self.num_groups() {
                self.rehash_with(new_groups, hb);
            }
        }
    }

    fn shrink_to_fit<H: BuildHasher>(&mut self, hb: &H) where K: Hash {
        if self.len == 0 {
            let mut empty = Self::new();
            std::mem::swap(self, &mut empty);
            return;
        }
        let min_groups = Self::groups_for_capacity(self.len);
        if min_groups < self.num_groups() {
            self.rehash_with(min_groups, hb);
        }
    }

    fn rehash_with<H: BuildHasher>(&mut self, new_num_groups: usize, hb: &H) where K: Hash {
        self.rehash_with(new_num_groups, hb);
    }

    fn iter_slots(&self) -> SlotIter<'_, K, V, T, S> { self.iter_slots() }

    fn into_iter_impl(self) -> IntoIter<K, V, T, S> {
        let mask = if !self.is_allocated() {
            bitmask::BitMask(0)
        } else {
            unsafe { Group::match_occupied(self.ctrl) }
        };
        let table = unsafe { ptr::read(&self) };
        std::mem::forget(self);
        IntoIter { table, group: 0, current_mask: mask }
    }

    fn drain_impl(&mut self) -> IntoIter<K, V, T, S> {
        let table = std::mem::replace(self, Self::new());
        table.into_iter_impl()
    }

    fn clone_table(&self) -> Self where K: Clone, V: Clone {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::tag_strategy::{Byte2_254, Byte7_128};
    use std::hash::RandomState;

    // Generic test helpers — parameterized by TombstoneTag
    fn test_basic<T: TombstoneTag>() {
        let hb = RandomState::new();
        let mut table: RawTable<u64, u64, T> = RawTable::with_capacity(16);

        let (_v, inserted) = table.insert_with_rehash(42, 100, &hb);
        assert!(inserted);
        assert_eq!(table.len(), 1);
        assert_eq!(table.get(&42, &hb), Some(&100));
        assert_eq!(table.get(&999, &hb), None);

        assert_eq!(table.remove(&42, &hb), Some(100));
        assert!(table.is_empty());
    }

    fn test_grow<T: TombstoneTag>() {
        let hb = RandomState::new();
        let mut table: RawTable<u64, u64, T> = RawTable::new();
        for i in 0..200 {
            table.insert_with_rehash(i, i * 10, &hb);
        }
        assert_eq!(table.len(), 200);
        for i in 0..200 {
            assert_eq!(table.get(&i, &hb), Some(&(i * 10)));
        }
    }

    fn test_clone<T: TombstoneTag>() {
        let hb = RandomState::new();
        let mut table: RawTable<u64, u64, T> = RawTable::new();
        for i in 0..50 {
            table.insert_with_rehash(i, i * 10, &hb);
        }
        let cloned = table.clone();
        assert_eq!(cloned.len(), 50);
        for i in 0..50 {
            assert_eq!(cloned.get(&i, &hb), Some(&(i * 10)));
        }
    }

    fn test_remove_cycle<T: TombstoneTag>() {
        let hb = RandomState::new();
        let mut table: RawTable<u64, u64, T> = RawTable::new();
        for cycle in 0..3 {
            for i in 0..100 {
                table.insert_with_rehash(i, i + cycle * 1000, &hb);
            }
            for i in 0..100 {
                table.remove(&i, &hb);
            }
            assert_eq!(table.len(), 0);
        }
    }

    fn test_iter<T: TombstoneTag>() {
        let hb = RandomState::new();
        let mut table: RawTable<u64, u64, T> = RawTable::new();
        for i in 0..50 {
            table.insert_with_rehash(i, i * 10, &hb);
        }
        assert_eq!(table.iter_slots().count(), 50);
    }

    // Byte7_254 (current IPO default — top-byte, decorrelated from AND group index)
    #[test] fn b7_254_basic() { test_basic::<Byte7_254>(); }
    #[test] fn b7_254_grow() { test_grow::<Byte7_254>(); }
    #[test] fn b7_254_clone() { test_clone::<Byte7_254>(); }
    #[test] fn b7_254_remove_cycle() { test_remove_cycle::<Byte7_254>(); }
    #[test] fn b7_254_iter() { test_iter::<Byte7_254>(); }

    // Byte2_254 (pre-fix default; correlated with AND mask above 2^16 groups)
    #[test] fn b2_254_basic() { test_basic::<Byte2_254>(); }
    #[test] fn b2_254_grow() { test_grow::<Byte2_254>(); }
    #[test] fn b2_254_clone() { test_clone::<Byte2_254>(); }
    #[test] fn b2_254_remove_cycle() { test_remove_cycle::<Byte2_254>(); }
    #[test] fn b2_254_iter() { test_iter::<Byte2_254>(); }

    // Byte7_128 (consolidated alternative — TopTag128 + HighByte128 + TopByte128)
    #[test] fn b7_128_basic() { test_basic::<Byte7_128>(); }
    #[test] fn b7_128_grow() { test_grow::<Byte7_128>(); }
    #[test] fn b7_128_clone() { test_clone::<Byte7_128>(); }
    #[test] fn b7_128_remove_cycle() { test_remove_cycle::<Byte7_128>(); }
    #[test] fn b7_128_iter() { test_iter::<Byte7_128>(); }

    // Extra: string keys (verifies Drop + non-Copy types)
    #[test]
    fn string_keys() {
        let hb = RandomState::new();
        let mut table: RawTable<String, i32> = RawTable::new();
        table.insert_with_rehash("hello".to_string(), 1, &hb);
        table.insert_with_rehash("world".to_string(), 2, &hb);
        assert_eq!(table.get(&"hello".to_string(), &hb), Some(&1));
        assert_eq!(table.get(&"missing".to_string(), &hb), None);
    }

    // Extra: clear + reuse
    #[test]
    fn clear_table() {
        let hb = RandomState::new();
        let mut table: RawTable<u64, u64> = RawTable::new();
        for i in 0..50 { table.insert_with_rehash(i, i, &hb); }
        table.clear();
        assert_eq!(table.len(), 0);
        assert!(table.capacity() > 0);
        table.insert_with_rehash(1, 1, &hb);
        assert_eq!(table.get(&1, &hb), Some(&1));
    }
}
