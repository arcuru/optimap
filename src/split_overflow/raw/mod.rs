pub mod group;

use std::alloc::{self, Layout};
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::ptr;

use group::{Group, GROUP_SIZE, META_GROUP_BYTES, EMPTY, reduced_hash, overflow_bit};
use crate::raw::bitmask;
use crate::raw::hash;

/// Result of a fused find-or-locate probe.
pub(crate) enum ProbeResult {
    /// Key was found at (group_index, slot_index).
    Found(usize, usize),
    /// Key was not found; first available empty slot at (group_index, slot_index).
    /// The u8 bitmask records which probe steps had full groups (bit i = step i was full).
    /// Used by insert_at to set overflow bits without re-walking the probe chain.
    InsertSlot(usize, usize, u8),
    /// Key was not found; no empty slot was encountered during the probe.
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

/// The core hash table engine (16-slot groups, separate overflow array).
///
/// Single combined allocation:
///   - metadata: `num_groups * 16` bytes, 16-byte aligned (all 16 are slot metadata)
///   - overflow: `num_groups` bytes (one overflow byte per group)
///   - (padding to bucket alignment)
///   - padding to bucket alignment
///   - buckets: `num_groups * 16 * sizeof((K,V))`
pub struct RawTable<K, V> {
    /// num_groups - 1. Used directly for probe wraparound and group_index masking.
    /// For empty tables (no allocation), mask = 0 and metadata is null.
    /// For 1 group, mask = 0 and metadata is non-null.
    pub(crate) mask: usize,
    pub(crate) metadata: *mut u8,
    overflow: *mut u8,
    buckets: *mut u8,
    pub(crate) len: usize,
    pub(crate) max_load: usize,
    /// group_index = hash >> shift. For num_groups=1, shift=64.
    pub(crate) shift: u32,
    _marker: PhantomData<(K, V)>,
}

impl<K, V> RawTable<K, V> {
    pub fn new() -> Self {
        RawTable {
            mask: 0,
            metadata: ptr::null_mut(),
            overflow: ptr::null_mut(),
            buckets: ptr::null_mut(),
            len: 0,
            max_load: 0,
            shift: 64,
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

    /// Whether the table has any allocation.
    #[inline(always)]
    pub(crate) fn is_allocated(&self) -> bool {
        !self.metadata.is_null()
    }

    fn groups_for_capacity(capacity: usize) -> usize {
        let min_slots = (capacity * MAX_LOAD_FACTOR_DEN + MAX_LOAD_FACTOR_NUM - 1)
            / MAX_LOAD_FACTOR_NUM;
        let min_groups = (min_slots + GROUP_SIZE - 1) / GROUP_SIZE;
        min_groups.next_power_of_two()
    }

    /// Compute the overflow offset within the combined allocation.
    /// Overflow array starts right after metadata.
    #[inline(always)]
    fn overflow_offset(num_groups: usize) -> usize {
        num_groups * META_GROUP_BYTES
    }

    /// Compute the bucket offset within the combined allocation.
    /// Layout: [metadata: num_groups*16] [overflow: num_groups bytes] [padding] [buckets]
    #[inline(always)]
    fn bucket_offset(num_groups: usize) -> usize {
        let meta_size = num_groups * META_GROUP_BYTES;
        let before_buckets = meta_size + num_groups;
        let bucket_align = std::mem::align_of::<(K, V)>().max(1);
        // Round up to bucket alignment
        (before_buckets + bucket_align - 1) & !(bucket_align - 1)
    }

    /// Compute the layout for the single combined allocation.
    fn combined_layout(num_groups: usize) -> Layout {
        let bucket_size = std::mem::size_of::<(K, V)>();
        let total_buckets = num_groups * GROUP_SIZE;
        let bucket_offset = Self::bucket_offset(num_groups);
        let total_size = bucket_offset + total_buckets * bucket_size;
        // Align to 16 for SIMD metadata loads
        Layout::from_size_align(total_size.max(16), 16).unwrap()
    }

    pub(crate) fn allocate(&mut self, num_groups: usize) {
        debug_assert!(num_groups.is_power_of_two());

        let layout = Self::combined_layout(num_groups);
        let bucket_offset = Self::bucket_offset(num_groups);
        let overflow_offset = Self::overflow_offset(num_groups);
        let meta_size = num_groups * META_GROUP_BYTES;
        let total_buckets = num_groups * GROUP_SIZE;

        unsafe {
            let ptr = alloc::alloc(layout);
            if ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }

            self.metadata = ptr;
            self.overflow = ptr.add(overflow_offset);
            self.buckets = ptr.add(bucket_offset);

            // Zero all metadata (empty groups) and overflow bytes
            ptr::write_bytes(self.metadata, 0, meta_size);
            ptr::write_bytes(self.overflow, 0, num_groups);
        }

        self.mask = num_groups - 1;
        self.max_load = max_load_for_capacity(total_buckets);
        self.shift = 64u32.wrapping_sub(num_groups.trailing_zeros());
    }

    unsafe fn deallocate(&mut self) {
        if self.metadata.is_null() {
            return;
        }
        let layout = Self::combined_layout(self.num_groups());
        unsafe { alloc::dealloc(self.metadata, layout); }
        self.metadata = ptr::null_mut();
        self.overflow = ptr::null_mut();
        self.buckets = ptr::null_mut();
    }

    /// Map hash to group index.
    #[inline(always)]
    pub(crate) fn group_index(&self, h: u64) -> usize {
        (h.wrapping_shr(self.shift) as usize) & self.mask
    }

    /// Pointer to group metadata (16-byte aligned).
    #[inline(always)]
    pub(crate) unsafe fn meta_ptr(&self, gi: usize) -> *mut u8 {
        unsafe { self.metadata.add(gi * META_GROUP_BYTES) }
    }

    /// Pointer to the overflow byte for group `gi`.
    #[inline(always)]
    pub(crate) unsafe fn overflow_ptr(&self, gi: usize) -> *mut u8 {
        unsafe { self.overflow.add(gi) }
    }

    /// Pointer to bucket slot. Uses shift+or since GROUP_SIZE=16.
    #[inline(always)]
    pub(crate) unsafe fn bucket_ptr(&self, gi: usize, si: usize) -> *mut (K, V) {
        let bucket_size = std::mem::size_of::<(K, V)>();
        let idx = (gi << 4) | si;
        unsafe { self.buckets.add(idx * bucket_size).cast::<(K, V)>() }
    }

    #[inline(always)]
    fn hash_key<S: BuildHasher>(key: &K, hash_builder: &S) -> u64
    where
        K: Hash,
    {
        hash::hash_no_mix(key, hash_builder)
    }

    /// Find a key in the table.
    pub fn find<S: BuildHasher>(&self, key: &K, hash_builder: &S) -> Option<(usize, usize)>
    where
        K: Hash + Eq,
    {
        if !self.is_allocated() {
            return None;
        }
        let h = Self::hash_key(key, hash_builder);
        self.find_by_hash(h, |k| k == key)
    }

    pub(crate) fn find_with_hash(&self, key: &K, h: u64) -> Option<(usize, usize)>
    where
        K: Eq,
    {
        self.find_by_hash(h, |k| k == key)
    }

    /// Core lookup: SIMD match (aligned) + overflow-bit probe termination + prefetch.
    #[inline(always)]
    pub(crate) fn find_by_hash<F>(&self, h: u64, eq: F) -> Option<(usize, usize)>
    where
        F: Fn(&K) -> bool,
    {
        if !self.is_allocated() {
            return None;
        }

        let reduced = reduced_hash(h);
        let mut gi = self.group_index(h);
        let ofw_bit = overflow_bit(h);
        let mut probe = 0usize;

        // Prefetch overflow byte for home group
        unsafe { Group::prefetch_read(self.overflow_ptr(gi) as *const u8); }

        loop {
            let meta = unsafe { self.meta_ptr(gi) };

            for si in unsafe { Group::match_byte(meta, reduced) } {
                let bucket = unsafe { &*self.bucket_ptr(gi, si) };
                if eq(&bucket.0) {
                    return Some((gi, si));
                }
            }

            if !unsafe { Group::has_overflow_bit(self.overflow_ptr(gi), ofw_bit) } {
                return None;
            }

            probe += 1;
            gi = (gi.wrapping_add(probe)) & self.mask;

            // Prefetch only on overflow -- doesn't fire on miss fast path
            unsafe {
                Group::prefetch_read(self.meta_ptr(gi) as *const u8);
                Group::prefetch_read(self.bucket_ptr(gi, 0) as *const u8);
                Group::prefetch_read(self.overflow_ptr(gi) as *const u8);
            }
        }
    }

    pub(crate) fn remove_by_hash<F>(&mut self, h: u64, eq: F) -> Option<V>
    where
        F: Fn(&K) -> bool,
    {
        let (gi, si) = self.find_by_hash(h, eq)?;

        unsafe {
            let bucket = self.bucket_ptr(gi, si).read();

            let meta = self.meta_ptr(gi);
            Group::set_meta(meta, si, EMPTY);

            self.len -= 1;

            // Anti-drift
            let initial_gi = self.group_index(h);
            let ofw_bit = overflow_bit(h);
            if Group::has_overflow_bit(self.overflow_ptr(initial_gi), ofw_bit) {
                self.max_load = self.max_load.saturating_sub(1);
            }

            let (_k, v) = bucket;
            Some(v)
        }
    }

    /// Insert without checking for duplicates or capacity.
    #[inline(always)]
    pub(crate) fn insert_no_check(&mut self, h: u64, key: K, value: V) -> (usize, usize) {
        let reduced = reduced_hash(h);
        let ofw_bit = overflow_bit(h);
        let mut gi = self.group_index(h);
        let mut probe = 0usize;

        loop {
            let meta = unsafe { self.meta_ptr(gi) };

            if let Some(si) = unsafe { Group::match_empty(meta) }.lowest_set_bit() {
                unsafe {
                    Group::set_meta(meta, si, reduced);
                    self.bucket_ptr(gi, si).write((key, value));
                }
                self.len += 1;
                return (gi, si);
            }

            // Group full -- set overflow bit in overflow array
            unsafe { Group::set_overflow_bit(self.overflow_ptr(gi), ofw_bit); }

            probe += 1;
            gi = (gi.wrapping_add(probe)) & self.mask;
        }
    }

    /// Fused find-or-locate: probes for the key and tracks the first empty slot.
    /// Returns Found(gi, si) if the key exists, or InsertSlot(gi, si) with the
    /// first available empty slot if the key is absent.
    /// Caller must ensure the table is non-empty and has capacity for an insert.
    #[inline(always)]
    pub(crate) fn find_or_locate<F>(&self, h: u64, eq: F) -> ProbeResult
    where
        F: Fn(&K) -> bool,
    {
        let reduced = reduced_hash(h);
        let ofw_bit = overflow_bit(h);
        let gi = self.group_index(h);

        // Home group fast path: covers the vast majority of operations
        // at load factor < 87.5%.
        let meta = unsafe { self.meta_ptr(gi) };
        let (matches, empties) = unsafe { Group::match_byte_and_empty(meta, reduced) };

        for si in matches {
            let bucket = unsafe { &*self.bucket_ptr(gi, si) };
            if eq(&bucket.0) {
                return ProbeResult::Found(gi, si);
            }
        }

        // If home group has empty slots and no overflow, key is absent
        if let Some(si) = empties.lowest_set_bit() {
            if !unsafe { Group::has_overflow_bit(self.overflow_ptr(gi), ofw_bit) } {
                // No overflow, empty slot in home group -- no overflow bits to set
                return ProbeResult::InsertSlot(gi, si, 0);
            }
            // Has overflow -- continue probing, remember this empty slot
            return self.find_or_locate_overflow(h, eq, reduced, ofw_bit, gi, Some((gi, si)), 0);
        }

        // Home group full -- continue probing
        if !unsafe { Group::has_overflow_bit(self.overflow_ptr(gi), ofw_bit) } {
            return ProbeResult::NotFound;
        }

        // Home group was full (bit 0 = step 0 was full)
        self.find_or_locate_overflow(h, eq, reduced, ofw_bit, gi, None, 1)
    }

    /// Slow path for find_or_locate when home group overflows.
    #[inline(never)]
    fn find_or_locate_overflow<F>(
        &self,
        _h: u64,
        eq: F,
        reduced: u8,
        ofw_bit: u8,
        home_gi: usize,
        mut first_empty: Option<(usize, usize)>,
        mut full_mask: u8,
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
                let bucket = unsafe { &*self.bucket_ptr(gi, si) };
                if eq(&bucket.0) {
                    return ProbeResult::Found(gi, si);
                }
            }

            if first_empty.is_none() {
                if let Some(si) = empties.lowest_set_bit() {
                    first_empty = Some((gi, si));
                } else {
                    // This group is full -- record in bitmask (probe steps 1..7)
                    if probe < 8 {
                        full_mask |= 1 << probe;
                    }
                }
            }

            if !unsafe { Group::has_overflow_bit(self.overflow_ptr(gi), ofw_bit) } {
                return match first_empty {
                    Some((ins_gi, ins_si)) => ProbeResult::InsertSlot(ins_gi, ins_si, full_mask),
                    None => ProbeResult::NotFound,
                };
            }

            probe += 1;
            gi = (gi.wrapping_add(probe)) & self.mask;

            unsafe {
                Group::prefetch_read(self.meta_ptr(gi) as *const u8);
                Group::prefetch_read(self.bucket_ptr(gi, 0) as *const u8);
                Group::prefetch_read(self.overflow_ptr(gi) as *const u8);
            }
        }
    }

    /// Write a key-value pair into a known-empty slot and set overflow bits.
    /// `full_mask` is a bitmask from find_or_locate: bit i means probe step i
    /// was a full group that needs its overflow bit set.
    #[inline(always)]
    pub(crate) fn insert_at(
        &mut self,
        h: u64,
        gi: usize,
        si: usize,
        key: K,
        value: V,
        full_mask: u8,
    ) {
        let reduced = reduced_hash(h);
        let ofw_bit = overflow_bit(h);

        // Set overflow bits on full groups recorded during probe.
        // Uses the bitmask from find_or_locate to avoid re-reading metadata.
        if full_mask != 0 {
            let home_gi = self.group_index(h);
            let mut set_probe = 0usize;
            let mut set_gi = home_gi;
            let mut mask = full_mask;
            while mask != 0 {
                if mask & 1 != 0 {
                    unsafe { Group::set_overflow_bit(self.overflow_ptr(set_gi), ofw_bit); }
                }
                mask >>= 1;
                set_probe += 1;
                set_gi = (set_gi.wrapping_add(set_probe)) & self.mask;
            }
        }

        unsafe {
            let meta = self.meta_ptr(gi);
            Group::set_meta(meta, si, reduced);
            self.bucket_ptr(gi, si).write((key, value));
        }
        self.len += 1;
    }

    pub fn rehash_with<S: BuildHasher>(&mut self, new_num_groups: usize, hash_builder: &S)
    where
        K: Hash,
    {
        let old_num_groups = self.num_groups();
        let old_metadata = self.metadata;
        let old_buckets = self.buckets;
        let old_layout = if old_metadata.is_null() {
            None
        } else {
            Some(Self::combined_layout(old_num_groups))
        };

        let bucket_size = std::mem::size_of::<(K, V)>();

        self.metadata = ptr::null_mut();
        self.overflow = ptr::null_mut();
        self.buckets = ptr::null_mut();
        self.mask = 0;
        self.len = 0;
        self.allocate(new_num_groups);

        if old_metadata.is_null() {
            return;
        }

        unsafe {
            for gi in 0..old_num_groups {
                let group_meta = old_metadata.add(gi * META_GROUP_BYTES);
                for si in Group::match_non_empty(group_meta) {
                    let old_bucket = old_buckets
                        .add(((gi << 4) | si) * bucket_size)
                        .cast::<(K, V)>();
                    let (key, value) = old_bucket.read();
                    let h = Self::hash_key(&key, hash_builder);
                    self.insert_no_check(h, key, value);
                }
            }

            alloc::dealloc(old_metadata, old_layout.unwrap());
        }
    }

    pub fn insert_with_rehash<S: BuildHasher>(
        &mut self,
        key: K,
        value: V,
        hash_builder: &S,
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
            let bucket = unsafe { &mut *self.bucket_ptr(gi, si) };
            return (&mut bucket.1, false);
        }

        if self.len >= self.max_load {
            let new_groups = if !self.is_allocated() { 1 } else { self.num_groups() * 2 };
            self.rehash_with(new_groups, hash_builder);
        }

        let (gi, si) = self.insert_no_check(h, key, value);
        let bucket = unsafe { &mut *self.bucket_ptr(gi, si) };
        (&mut bucket.1, true)
    }

    pub fn remove<S: BuildHasher>(&mut self, key: &K, hash_builder: &S) -> Option<V>
    where
        K: Hash + Eq,
    {
        if !self.is_allocated() {
            return None;
        }
        let h = Self::hash_key(key, hash_builder);
        let (gi, si) = self.find_with_hash(key, h)?;

        unsafe {
            let bucket = self.bucket_ptr(gi, si).read();
            let meta = self.meta_ptr(gi);
            Group::set_meta(meta, si, EMPTY);
            self.len -= 1;

            let initial_gi = self.group_index(h);
            let ofw_bit = overflow_bit(h);
            if Group::has_overflow_bit(self.overflow_ptr(initial_gi), ofw_bit) {
                self.max_load = self.max_load.saturating_sub(1);
            }

            let (_k, v) = bucket;
            Some(v)
        }
    }

    pub fn get<S: BuildHasher>(&self, key: &K, hash_builder: &S) -> Option<&V>
    where
        K: Hash + Eq,
    {
        let (gi, si) = self.find(key, hash_builder)?;
        let bucket = unsafe { &*self.bucket_ptr(gi, si) };
        Some(&bucket.1)
    }

    pub fn get_mut<S: BuildHasher>(&mut self, key: &K, hash_builder: &S) -> Option<&mut V>
    where
        K: Hash + Eq,
    {
        let (gi, si) = self.find(key, hash_builder)?;
        let bucket = unsafe { &mut *self.bucket_ptr(gi, si) };
        Some(&mut bucket.1)
    }

    pub fn clear(&mut self) {
        if self.metadata.is_null() {
            return;
        }

        unsafe {
            if std::mem::needs_drop::<(K, V)>() {
                for gi in 0..self.num_groups() {
                    let group_meta = self.metadata.add(gi * META_GROUP_BYTES);
                    for si in Group::match_non_empty(group_meta) {
                        ptr::drop_in_place(self.bucket_ptr(gi, si));
                    }
                }
            }

            let meta_size = self.num_groups() * META_GROUP_BYTES;
            ptr::write_bytes(self.metadata, 0, meta_size);
            ptr::write_bytes(self.overflow, 0, self.num_groups());
        }

        self.len = 0;
        self.max_load = max_load_for_capacity(self.num_groups() * GROUP_SIZE);
    }

    /// Iterate over all occupied slots using SIMD to skip empty groups.
    pub fn iter_slots(&self) -> SlotIter<'_, K, V> {
        SlotIter {
            table: self,
            group: 0,
            current_mask: if self.metadata.is_null() {
                bitmask::BitMask(0)
            } else {
                unsafe { Group::match_non_empty(self.metadata) }
            },
        }
    }
}

impl<K, V> Drop for RawTable<K, V> {
    fn drop(&mut self) {
        if self.metadata.is_null() {
            return;
        }
        if std::mem::needs_drop::<(K, V)>() {
            unsafe {
                for gi in 0..self.num_groups() {
                    let group_meta = self.metadata.add(gi * META_GROUP_BYTES);
                    for si in Group::match_non_empty(group_meta) {
                        ptr::drop_in_place(self.bucket_ptr(gi, si));
                    }
                }
            }
        }
        unsafe { self.deallocate(); }
    }
}

unsafe impl<K: Send, V: Send> Send for RawTable<K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for RawTable<K, V> {}

impl<K: Clone, V: Clone> Clone for RawTable<K, V> {
    fn clone(&self) -> Self {
        if self.metadata.is_null() {
            return Self::new();
        }

        let mut new_table = Self::new();
        new_table.allocate(self.num_groups());

        unsafe {
            // Copy metadata (num_groups * 16 bytes)
            let meta_size = self.num_groups() * META_GROUP_BYTES;
            ptr::copy_nonoverlapping(self.metadata, new_table.metadata, meta_size);

            // Copy overflow bytes
            ptr::copy_nonoverlapping(self.overflow, new_table.overflow, self.num_groups());

            let bucket_size = std::mem::size_of::<(K, V)>();
            if bucket_size > 0 {
                for gi in 0..self.num_groups() {
                    let group_meta = self.metadata.add(gi * META_GROUP_BYTES);
                    for si in Group::match_non_empty(group_meta) {
                        let src = &*self.bucket_ptr(gi, si);
                        new_table.bucket_ptr(gi, si).write(src.clone());
                    }
                }
            }
        }

        new_table.len = self.len;
        new_table.max_load = self.max_load;
        new_table
    }
}

/// SIMD-accelerated iterator over occupied slot positions.
pub struct SlotIter<'a, K, V> {
    pub(crate) table: &'a RawTable<K, V>,
    group: usize,
    current_mask: bitmask::BitMask,
}

impl<'a, K, V> Iterator for SlotIter<'a, K, V> {
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
            self.current_mask = unsafe {
                Group::match_non_empty(self.table.meta_ptr(self.group))
            };
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.table.len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::RandomState;

    #[test]
    fn new_table_is_empty() {
        let table: RawTable<u64, u64> = RawTable::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
        assert_eq!(table.capacity(), 0);
    }

    #[test]
    fn insert_and_find() {
        let hash_builder = RandomState::new();
        let mut table: RawTable<u64, u64> = RawTable::with_capacity(16);

        let (_v, inserted) = table.insert_with_rehash(42, 100, &hash_builder);
        assert!(inserted);
        assert_eq!(table.len(), 1);

        assert_eq!(table.get(&42, &hash_builder), Some(&100));
        assert_eq!(table.get(&999, &hash_builder), None);
    }

    #[test]
    fn insert_duplicate() {
        let hash_builder = RandomState::new();
        let mut table: RawTable<u64, u64> = RawTable::with_capacity(16);

        table.insert_with_rehash(1, 10, &hash_builder);
        let (_v, inserted) = table.insert_with_rehash(1, 20, &hash_builder);
        assert!(!inserted);
        assert_eq!(table.len(), 1);
        assert_eq!(table.get(&1, &hash_builder), Some(&10));
    }

    #[test]
    fn remove_existing() {
        let hash_builder = RandomState::new();
        let mut table: RawTable<u64, u64> = RawTable::with_capacity(16);

        table.insert_with_rehash(1, 10, &hash_builder);
        table.insert_with_rehash(2, 20, &hash_builder);

        assert_eq!(table.remove(&1, &hash_builder), Some(10));
        assert_eq!(table.len(), 1);
        assert_eq!(table.get(&1, &hash_builder), None);
        assert_eq!(table.get(&2, &hash_builder), Some(&20));
    }

    #[test]
    fn remove_nonexistent() {
        let hash_builder = RandomState::new();
        let mut table: RawTable<u64, u64> = RawTable::with_capacity(16);
        assert_eq!(table.remove(&42, &hash_builder), None);
    }

    #[test]
    fn grow_and_rehash() {
        let hash_builder = RandomState::new();
        let mut table: RawTable<u64, u64> = RawTable::new();

        for i in 0..200 {
            table.insert_with_rehash(i, i * 10, &hash_builder);
        }
        assert_eq!(table.len(), 200);
        for i in 0..200 {
            assert_eq!(table.get(&i, &hash_builder), Some(&(i * 10)));
        }
    }

    #[test]
    fn clear_table() {
        let hash_builder = RandomState::new();
        let mut table: RawTable<u64, u64> = RawTable::new();

        for i in 0..50 {
            table.insert_with_rehash(i, i, &hash_builder);
        }

        table.clear();
        assert_eq!(table.len(), 0);
        assert!(table.capacity() > 0);

        table.insert_with_rehash(1, 1, &hash_builder);
        assert_eq!(table.get(&1, &hash_builder), Some(&1));
    }

    #[test]
    fn string_keys() {
        let hash_builder = RandomState::new();
        let mut table: RawTable<String, i32> = RawTable::new();

        table.insert_with_rehash("hello".to_string(), 1, &hash_builder);
        table.insert_with_rehash("world".to_string(), 2, &hash_builder);

        assert_eq!(table.get(&"hello".to_string(), &hash_builder), Some(&1));
        assert_eq!(table.get(&"world".to_string(), &hash_builder), Some(&2));
        assert_eq!(table.get(&"missing".to_string(), &hash_builder), None);
    }

    #[test]
    fn clone_table() {
        let hash_builder = RandomState::new();
        let mut table: RawTable<u64, u64> = RawTable::new();

        for i in 0..50 {
            table.insert_with_rehash(i, i * 10, &hash_builder);
        }

        let cloned = table.clone();
        assert_eq!(cloned.len(), 50);
        for i in 0..50 {
            assert_eq!(cloned.get(&i, &hash_builder), Some(&(i * 10)));
        }
    }

    #[test]
    fn insert_remove_insert_cycle() {
        let hash_builder = RandomState::new();
        let mut table: RawTable<u64, u64> = RawTable::new();

        for cycle in 0..3 {
            for i in 0..100 {
                table.insert_with_rehash(i, i + cycle * 1000, &hash_builder);
            }
            for i in 0..100 {
                table.remove(&i, &hash_builder);
            }
            assert_eq!(table.len(), 0);
        }
    }

    #[test]
    fn iter_slots_works() {
        let hash_builder = RandomState::new();
        let mut table: RawTable<u64, u64> = RawTable::new();

        for i in 0..50 {
            table.insert_with_rehash(i, i * 10, &hash_builder);
        }

        let count = table.iter_slots().count();
        assert_eq!(count, 50);
    }
}
