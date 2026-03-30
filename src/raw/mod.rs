pub mod bitmask;
pub mod group;
pub mod hash;

use std::alloc::{self, Layout};
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::ptr;

use group::{Group, GROUP_SIZE, META_GROUP_BYTES, EMPTY, reduced_hash, overflow_bit};
use hash::mix_hash;

/// Maximum load factor (fixed at 7/8 = 0.875).
const MAX_LOAD_FACTOR_NUM: usize = 7;
const MAX_LOAD_FACTOR_DEN: usize = 8;

/// Compute max load for a given capacity.
#[inline]
fn max_load_for_capacity(capacity: usize) -> usize {
    capacity * MAX_LOAD_FACTOR_NUM / MAX_LOAD_FACTOR_DEN
}

/// The core hash table engine.
///
/// Stores key-value pairs in a flat bucket array with a companion metadata
/// array organized in groups of 15, using SIMD for fast matching.
pub struct RawTable<K, V> {
    /// Number of groups (always a power of 2, minimum 1 when allocated).
    pub(crate) num_groups: usize,

    /// Pointer to metadata array: `num_groups + 1` groups
    /// (extra sentinel group at the end for iteration).
    pub(crate) metadata: *mut u8,

    /// Pointer to bucket array: `num_groups * GROUP_SIZE` slots.
    /// Each slot is `MaybeUninit<(K, V)>`.
    pub(crate) buckets: *mut u8,

    /// Number of occupied slots.
    pub(crate) len: usize,

    /// Maximum number of elements before rehash.
    /// Starts at max_load_for_capacity(num_groups * GROUP_SIZE) and
    /// decreases on overflow-bit erasures (anti-drift).
    pub(crate) max_load: usize,

    /// Shift amount for mapping hash → group index.
    /// group_index = hash >> shift, where shift = 64 - log2(num_groups).
    pub(crate) shift: u32,

    _marker: PhantomData<(K, V)>,
}

impl<K, V> RawTable<K, V> {
    /// Create a new empty table with no allocation.
    pub fn new() -> Self {
        RawTable {
            num_groups: 0,
            metadata: ptr::null_mut(),
            buckets: ptr::null_mut(),
            len: 0,
            max_load: 0,
            shift: 64,
            _marker: PhantomData,
        }
    }

    /// Create a table pre-allocated for at least `capacity` elements.
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::new();
        }
        let mut table = Self::new();
        let num_groups = Self::groups_for_capacity(capacity);
        table.allocate(num_groups);
        table
    }

    /// Number of occupied elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Is the table empty?
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total number of bucket slots.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.num_groups * GROUP_SIZE
    }

    /// Compute the minimum number of groups needed for a given capacity.
    fn groups_for_capacity(capacity: usize) -> usize {
        // We need: num_groups * GROUP_SIZE * 7/8 >= capacity
        // So: num_groups >= capacity * 8 / (GROUP_SIZE * 7)
        let min_slots = (capacity * MAX_LOAD_FACTOR_DEN + MAX_LOAD_FACTOR_NUM - 1)
            / MAX_LOAD_FACTOR_NUM;
        let min_groups = (min_slots + GROUP_SIZE - 1) / GROUP_SIZE;
        // Round up to power of 2
        min_groups.next_power_of_two()
    }

    /// Allocate metadata and bucket arrays for `num_groups` groups.
    pub(crate) fn allocate(&mut self, num_groups: usize) {
        debug_assert!(num_groups.is_power_of_two());

        let bucket_size = std::mem::size_of::<(K, V)>();
        let bucket_align = std::mem::align_of::<(K, V)>();

        // Metadata: (num_groups + 1) * 16 bytes, 16-byte aligned
        let meta_size = (num_groups + 1) * META_GROUP_BYTES;
        let meta_layout = Layout::from_size_align(meta_size, 16).unwrap();

        // Buckets: num_groups * GROUP_SIZE * sizeof((K,V))
        let total_buckets = num_groups * GROUP_SIZE;
        let bucket_layout = if bucket_size > 0 {
            Layout::from_size_align(total_buckets * bucket_size, bucket_align).unwrap()
        } else {
            // ZST buckets
            Layout::from_size_align(0, 1).unwrap()
        };

        unsafe {
            self.metadata = alloc::alloc(meta_layout);
            if self.metadata.is_null() {
                alloc::handle_alloc_error(meta_layout);
            }

            if bucket_size > 0 {
                self.buckets = alloc::alloc(bucket_layout);
                if self.buckets.is_null() {
                    alloc::dealloc(self.metadata, meta_layout);
                    alloc::handle_alloc_error(bucket_layout);
                }
            }

            // Initialize all metadata groups to empty
            for i in 0..num_groups {
                let g = Group::empty();
                g.store(self.metadata.add(i * META_GROUP_BYTES));
            }
            // Sentinel group at the end
            let sentinel = Group::sentinel();
            sentinel.store(self.metadata.add(num_groups * META_GROUP_BYTES));
        }

        self.num_groups = num_groups;
        self.max_load = max_load_for_capacity(total_buckets);
        self.shift = if num_groups == 0 {
            64
        } else {
            64 - num_groups.trailing_zeros()
        };
    }

    /// Free the metadata and bucket arrays.
    unsafe fn deallocate(&mut self) {
        if self.metadata.is_null() {
            return;
        }

        let bucket_size = std::mem::size_of::<(K, V)>();
        let bucket_align = std::mem::align_of::<(K, V)>();

        let meta_size = (self.num_groups + 1) * META_GROUP_BYTES;
        let meta_layout = Layout::from_size_align(meta_size, 16).unwrap();
        unsafe {
            alloc::dealloc(self.metadata, meta_layout);
        }

        if bucket_size > 0 {
            let total_buckets = self.num_groups * GROUP_SIZE;
            let bucket_layout =
                Layout::from_size_align(total_buckets * bucket_size, bucket_align).unwrap();
            unsafe {
                alloc::dealloc(self.buckets, bucket_layout);
            }
        }

        self.metadata = ptr::null_mut();
        self.buckets = ptr::null_mut();
    }

    /// Map a hash value to a group index.
    #[inline]
    fn group_index(&self, h: u64) -> usize {
        // When num_groups == 1, shift == 64 and we want to return 0.
        // Avoid UB/panic by clamping shift to 63 and masking with num_groups - 1.
        if self.num_groups <= 1 {
            return 0;
        }
        (h >> self.shift) as usize
    }

    /// Get a pointer to the metadata for group `gi`.
    #[inline]
    unsafe fn meta_ptr(&self, gi: usize) -> *mut u8 {
        unsafe { self.metadata.add(gi * META_GROUP_BYTES) }
    }

    /// Get a pointer to bucket slot at (group_index, slot_index).
    #[inline]
    pub(crate) unsafe fn bucket_ptr(&self, gi: usize, si: usize) -> *mut (K, V) {
        let bucket_size = std::mem::size_of::<(K, V)>();
        let idx = gi * GROUP_SIZE + si;
        unsafe { self.buckets.add(idx * bucket_size).cast::<(K, V)>() }
    }

    /// Load the Group metadata for group `gi`.
    #[inline]
    unsafe fn load_group(&self, gi: usize) -> Group {
        unsafe { Group::load(self.meta_ptr(gi)) }
    }

    /// Compute the hash of a key.
    #[inline]
    fn hash_key<S: BuildHasher>(key: &K, hash_builder: &S) -> u64
    where
        K: Hash,
    {
        let mut hasher = hash_builder.build_hasher();
        key.hash(&mut hasher);
        mix_hash(hasher.finish())
    }

    /// Find a key in the table. Returns (group_index, slot_index) if found.
    pub fn find<S: BuildHasher>(&self, key: &K, hash_builder: &S) -> Option<(usize, usize)>
    where
        K: Hash + Eq,
    {
        if self.num_groups == 0 {
            return None;
        }

        let h = Self::hash_key(key, hash_builder);
        let reduced = reduced_hash(h);
        let ofw_bit = overflow_bit(h);
        let mut gi = self.group_index(h);
        let mut probe = 0usize;

        loop {
            let group = unsafe { self.load_group(gi) };

            // SIMD match: find slots with matching reduced hash
            for si in group.match_byte(reduced) {
                let bucket = unsafe { &*self.bucket_ptr(gi, si) };
                if bucket.0 == *key {
                    return Some((gi, si));
                }
            }

            // Termination: if group is not full, or overflow bit not set, key is absent
            if !group.has_overflow_bit(ofw_bit) {
                return None;
            }

            // Quadratic probing: next group
            probe += 1;
            gi = (gi + probe) & (self.num_groups - 1);
        }
    }

    /// Find a key using a pre-computed hash. Returns (group_index, slot_index) if found.
    pub(crate) fn find_with_hash(&self, key: &K, h: u64) -> Option<(usize, usize)>
    where
        K: Eq,
    {
        self.find_by_hash(h, |k| k == key)
    }

    /// Find an element using a pre-computed hash and a custom equality predicate.
    /// This supports Borrow-based lookups where Q != K.
    pub(crate) fn find_by_hash<F>(&self, h: u64, eq: F) -> Option<(usize, usize)>
    where
        F: Fn(&K) -> bool,
    {
        if self.num_groups == 0 {
            return None;
        }

        let reduced = reduced_hash(h);
        let ofw_bit = overflow_bit(h);
        let mut gi = self.group_index(h);
        let mut probe = 0usize;

        loop {
            let group = unsafe { self.load_group(gi) };

            for si in group.match_byte(reduced) {
                let bucket = unsafe { &*self.bucket_ptr(gi, si) };
                if eq(&bucket.0) {
                    return Some((gi, si));
                }
            }

            if !group.has_overflow_bit(ofw_bit) {
                return None;
            }

            probe += 1;
            gi = (gi + probe) & (self.num_groups - 1);
        }
    }

    /// Remove an element using a pre-computed hash and equality predicate.
    pub(crate) fn remove_by_hash<F>(&mut self, h: u64, eq: F) -> Option<V>
    where
        F: Fn(&K) -> bool,
    {
        let (gi, si) = self.find_by_hash(h, eq)?;

        unsafe {
            let bucket = self.bucket_ptr(gi, si).read();
            let mut group = self.load_group(gi);
            group.set(si, EMPTY);
            group.store(self.meta_ptr(gi));
            self.len -= 1;

            let initial_gi = self.group_index(h);
            let initial_group = self.load_group(initial_gi);
            let ofw_bit = overflow_bit(h);
            if initial_group.has_overflow_bit(ofw_bit) {
                self.max_load = self.max_load.saturating_sub(1);
            }

            let (_k, v) = bucket;
            Some(v)
        }
    }

    /// Insert without checking for duplicates or capacity.
    /// Used during rehash and after find confirms absence.
    pub(crate) fn insert_no_check(&mut self, h: u64, key: K, value: V) -> (usize, usize) {
        let reduced = reduced_hash(h);
        let ofw_bit = overflow_bit(h);
        let mut gi = self.group_index(h);
        let mut probe = 0usize;

        loop {
            let mut group = unsafe { self.load_group(gi) };

            if let Some(si) = group.match_empty().lowest_set_bit() {
                // Found an empty slot
                group.set(si, reduced);
                unsafe {
                    group.store(self.meta_ptr(gi));
                    self.bucket_ptr(gi, si).write((key, value));
                }
                self.len += 1;
                return (gi, si);
            }

            // Group is full — set overflow bit and probe to next
            group.set_overflow_bit(ofw_bit);
            unsafe {
                group.store(self.meta_ptr(gi));
            }

            probe += 1;
            gi = (gi + probe) & (self.num_groups - 1);
        }
    }

    /// Rehash the table using a hash builder to recompute hashes.
    pub fn rehash_with<S: BuildHasher>(&mut self, new_num_groups: usize, hash_builder: &S)
    where
        K: Hash,
    {
        let old_num_groups = self.num_groups;
        let old_metadata = self.metadata;
        let old_buckets = self.buckets;

        let bucket_size = std::mem::size_of::<(K, V)>();
        let bucket_align = std::mem::align_of::<(K, V)>();

        // Reset and allocate new
        self.metadata = ptr::null_mut();
        self.buckets = ptr::null_mut();
        self.num_groups = 0;
        self.len = 0;
        self.allocate(new_num_groups);

        if old_metadata.is_null() {
            return;
        }

        unsafe {
            for gi in 0..old_num_groups {
                let group_meta = old_metadata.add(gi * META_GROUP_BYTES);
                for si in 0..GROUP_SIZE {
                    let meta = *group_meta.add(si);
                    if meta >= 2 {
                        let old_bucket = old_buckets
                            .add((gi * GROUP_SIZE + si) * bucket_size)
                            .cast::<(K, V)>();
                        let (key, value) = old_bucket.read();
                        let h = Self::hash_key(&key, hash_builder);
                        self.insert_no_check(h, key, value);
                    }
                }
            }

            // Deallocate old arrays
            let old_meta_size = (old_num_groups + 1) * META_GROUP_BYTES;
            let old_meta_layout = Layout::from_size_align(old_meta_size, 16).unwrap();
            alloc::dealloc(old_metadata, old_meta_layout);

            if bucket_size > 0 {
                let old_total = old_num_groups * GROUP_SIZE;
                let old_bucket_layout =
                    Layout::from_size_align(old_total * bucket_size, bucket_align).unwrap();
                alloc::dealloc(old_buckets, old_bucket_layout);
            }
        }
    }

    /// Insert with automatic rehash support.
    pub fn insert_with_rehash<S: BuildHasher>(
        &mut self,
        key: K,
        value: V,
        hash_builder: &S,
    ) -> (&mut V, bool)
    where
        K: Hash + Eq,
    {
        if self.num_groups == 0 {
            self.allocate(1);
        }

        let h = Self::hash_key(&key, hash_builder);

        // Check if key already exists
        if let Some((gi, si)) = self.find_with_hash(&key, h) {
            drop(key);
            drop(value);
            let bucket = unsafe { &mut *self.bucket_ptr(gi, si) };
            return (&mut bucket.1, false);
        }

        // Check if rehash needed
        if self.len >= self.max_load {
            let new_groups = if self.num_groups == 0 { 1 } else { self.num_groups * 2 };
            self.rehash_with(new_groups, hash_builder);
        }

        let (gi, si) = self.insert_no_check(h, key, value);
        let bucket = unsafe { &mut *self.bucket_ptr(gi, si) };
        (&mut bucket.1, true)
    }

    /// Remove a key from the table. Returns the value if found.
    pub fn remove<S: BuildHasher>(&mut self, key: &K, hash_builder: &S) -> Option<V>
    where
        K: Hash + Eq,
    {
        if self.num_groups == 0 {
            return None;
        }

        let h = Self::hash_key(key, hash_builder);
        let (gi, si) = self.find_with_hash(key, h)?;

        unsafe {
            // Read out the bucket
            let bucket = self.bucket_ptr(gi, si).read();

            // Mark slot as empty in metadata
            let mut group = self.load_group(gi);
            group.set(si, EMPTY);
            group.store(self.meta_ptr(gi));

            self.len -= 1;

            // Anti-drift: if overflow bit is set for this hash's bit position
            // in the element's initial group, decrease max_load
            let initial_gi = self.group_index(h);
            let initial_group = self.load_group(initial_gi);
            let ofw_bit = overflow_bit(h);
            if initial_group.has_overflow_bit(ofw_bit) {
                self.max_load = self.max_load.saturating_sub(1);
            }

            // Return the value, drop the key
            let (_k, v) = bucket;
            Some(v)
        }
    }

    /// Get a reference to the value for a key.
    pub fn get<S: BuildHasher>(&self, key: &K, hash_builder: &S) -> Option<&V>
    where
        K: Hash + Eq,
    {
        let (gi, si) = self.find(key, hash_builder)?;
        let bucket = unsafe { &*self.bucket_ptr(gi, si) };
        Some(&bucket.1)
    }

    /// Get a mutable reference to the value for a key.
    pub fn get_mut<S: BuildHasher>(&mut self, key: &K, hash_builder: &S) -> Option<&mut V>
    where
        K: Hash + Eq,
    {
        let (gi, si) = self.find(key, hash_builder)?;
        let bucket = unsafe { &mut *self.bucket_ptr(gi, si) };
        Some(&mut bucket.1)
    }

    /// Clear all elements without deallocating.
    pub fn clear(&mut self) {
        if self.metadata.is_null() {
            return;
        }

        unsafe {
            // Drop all occupied elements
            for gi in 0..self.num_groups {
                let group_meta = self.metadata.add(gi * META_GROUP_BYTES);
                for si in 0..GROUP_SIZE {
                    let meta = *group_meta.add(si);
                    if meta >= 2 {
                        ptr::drop_in_place(self.bucket_ptr(gi, si));
                    }
                }
            }

            // Reset all metadata to empty
            for gi in 0..self.num_groups {
                let g = Group::empty();
                g.store(self.meta_ptr(gi));
            }
            // Re-write sentinel
            let sentinel = Group::sentinel();
            sentinel.store(self.meta_ptr(self.num_groups));
        }

        self.len = 0;
        self.max_load = max_load_for_capacity(self.num_groups * GROUP_SIZE);
    }

    /// Iterate over all occupied slots, returning (group_index, slot_index) pairs.
    pub fn iter_slots(&self) -> SlotIter<'_, K, V> {
        SlotIter {
            table: self,
            group: 0,
            slot: 0,
        }
    }
}

impl<K, V> Drop for RawTable<K, V> {
    fn drop(&mut self) {
        if self.metadata.is_null() {
            return;
        }

        // Drop all occupied elements
        unsafe {
            let bucket_size = std::mem::size_of::<(K, V)>();
            for gi in 0..self.num_groups {
                let group_meta = self.metadata.add(gi * META_GROUP_BYTES);
                for si in 0..GROUP_SIZE {
                    let meta = *group_meta.add(si);
                    if meta >= 2 {
                        if bucket_size > 0 {
                            ptr::drop_in_place(self.bucket_ptr(gi, si));
                        }
                    }
                }
            }
        }

        unsafe {
            self.deallocate();
        }
    }
}

// SAFETY: RawTable can be sent between threads if K and V can.
unsafe impl<K: Send, V: Send> Send for RawTable<K, V> {}
unsafe impl<K: Sync, V: Sync> Sync for RawTable<K, V> {}

impl<K: Clone, V: Clone> Clone for RawTable<K, V> {
    fn clone(&self) -> Self {
        if self.metadata.is_null() {
            return Self::new();
        }

        let mut new_table = Self::new();
        new_table.allocate(self.num_groups);

        unsafe {
            // Copy metadata verbatim (including overflow bits and sentinel)
            let meta_size = (self.num_groups + 1) * META_GROUP_BYTES;
            ptr::copy_nonoverlapping(self.metadata, new_table.metadata, meta_size);

            // Clone each occupied bucket
            let bucket_size = std::mem::size_of::<(K, V)>();
            for gi in 0..self.num_groups {
                let group_meta = self.metadata.add(gi * META_GROUP_BYTES);
                for si in 0..GROUP_SIZE {
                    let meta = *group_meta.add(si);
                    if meta >= 2 {
                        let src = &*self.bucket_ptr(gi, si);
                        let cloned = src.clone();
                        if bucket_size > 0 {
                            new_table.bucket_ptr(gi, si).write(cloned);
                        }
                    }
                }
            }
        }

        new_table.len = self.len;
        new_table.max_load = self.max_load;
        new_table
    }
}

/// Iterator over occupied slot positions.
pub struct SlotIter<'a, K, V> {
    pub(crate) table: &'a RawTable<K, V>,
    group: usize,
    slot: usize,
}

impl<'a, K, V> Iterator for SlotIter<'a, K, V> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.table.metadata.is_null() {
            return None;
        }
        unsafe {
            while self.group < self.table.num_groups {
                while self.slot < GROUP_SIZE {
                    let meta = *self
                        .table
                        .metadata
                        .add(self.group * META_GROUP_BYTES + self.slot);
                    let gi = self.group;
                    let si = self.slot;
                    self.slot += 1;
                    if meta >= 2 {
                        return Some((gi, si));
                    }
                }
                self.group += 1;
                self.slot = 0;
            }
        }
        None
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

        let val = table.get(&42, &hash_builder);
        assert_eq!(val, Some(&100));

        let val = table.get(&999, &hash_builder);
        assert_eq!(val, None);
    }

    #[test]
    fn insert_duplicate() {
        let hash_builder = RandomState::new();
        let mut table: RawTable<u64, u64> = RawTable::with_capacity(16);

        table.insert_with_rehash(1, 10, &hash_builder);
        let (_v, inserted) = table.insert_with_rehash(1, 20, &hash_builder);
        assert!(!inserted);
        assert_eq!(table.len(), 1);
        assert_eq!(table.get(&1, &hash_builder), Some(&10)); // original value
    }

    #[test]
    fn remove_existing() {
        let hash_builder = RandomState::new();
        let mut table: RawTable<u64, u64> = RawTable::with_capacity(16);

        table.insert_with_rehash(1, 10, &hash_builder);
        table.insert_with_rehash(2, 20, &hash_builder);

        let removed = table.remove(&1, &hash_builder);
        assert_eq!(removed, Some(10));
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
        assert!(table.is_empty());

        // Capacity should still be allocated
        assert!(table.capacity() > 0);

        // Should be able to insert again
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

        // Insert, remove, re-insert to test tombstone-free deletion
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
}
