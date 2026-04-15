use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::iter::FusedIterator;
use std::ops::Index;

use super::raw::{ProbeResult, RawTable};
use crate::raw::hash;

/// Default hasher for the map. Uses foldhash for speed
/// (same fast hasher used by hashbrown).
pub type DefaultHashBuilder = foldhash::fast::RandomState;

/// A hash map using open addressing with SIMD-accelerated group probing,
/// inspired by `boost::optimap`.
///
/// Elements are stored contiguously in a flat bucket array (no indirection).
/// A companion metadata array with 15-byte groups enables fast SIMD lookups.
///
/// The maximum load factor is fixed at 0.875 and cannot be changed.
pub struct Gaps<K, V, S = DefaultHashBuilder> {
    table: RawTable<K, V>,
    hash_builder: S,
}

// ── Constructors ────────────────────────────────────────────────────────────

impl<K, V> Gaps<K, V, DefaultHashBuilder> {
    /// Creates an empty map.
    pub fn new() -> Self {
        Self::with_hasher(DefaultHashBuilder::default())
    }

    /// Creates an empty map with at least the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, DefaultHashBuilder::default())
    }
}

impl<K, V, S> Gaps<K, V, S> {
    /// Creates an empty map with the given hasher.
    pub fn with_hasher(hash_builder: S) -> Self {
        Gaps {
            table: RawTable::new(),
            hash_builder,
        }
    }

    /// Creates an empty map with the given capacity and hasher.
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        Gaps {
            table: RawTable::with_capacity(capacity),
            hash_builder,
        }
    }

    /// Returns the number of elements in the map.
    #[inline]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Returns true if the map contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Returns the number of elements the map can hold without rehashing.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.table.capacity()
    }

    /// Returns a reference to the map's hasher.
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }

    /// Clears the map, removing all elements but keeping allocated memory.
    pub fn clear(&mut self) {
        self.table.clear();
    }
}

// ── Core operations ─────────────────────────────────────────────────────────

impl<K, V, S> Gaps<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    #[inline(always)]
    fn hash_key<Q: Hash + ?Sized>(&self, key: &Q) -> u64 {
        hash::hash_no_mix(key, &self.hash_builder)
    }

    /// Returns a reference to the value corresponding to the key.
    #[inline]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let h = self.hash_key(key);
        let (gi, si) = self.table.find_by_hash(h, |k| k.borrow() == key)?;
        let bucket = unsafe { &*self.table.bucket_ptr(gi, si) };
        Some(&bucket.1)
    }

    /// Returns a mutable reference to the value corresponding to the key.
    #[inline]
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let h = self.hash_key(key);
        let (gi, si) = self.table.find_by_hash(h, |k| k.borrow() == key)?;
        let bucket = unsafe { &mut *self.table.bucket_ptr(gi, si) };
        Some(&mut bucket.1)
    }

    /// Returns true if the map contains the given key.
    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the key already exists, the value is replaced and the old value is returned.
    ///
    /// Uses a fused home-group fast path: a single SIMD load produces both the
    /// key-match and empty-slot bitmasks, avoiding a second metadata load when
    /// the key is absent and the home group has space.
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        use super::raw::group::{Group, overflow_bit, reduced_hash};

        if !self.table.is_allocated() {
            self.table.allocate(1);
        }

        let h = self.hash_key(&key);

        // At capacity: use standard two-pass (growth would invalidate any slot info)
        if self.table.len >= self.table.max_load {
            return self.insert_at_capacity(h, key, value);
        }

        let reduced = reduced_hash(h);
        let gi = self.table.group_index(h);
        let meta = unsafe { self.table.meta_ptr(gi) };

        // Single SIMD load → both match and empty bitmasks
        let (matches, empties) = unsafe { Group::match_byte_and_empty(meta, reduced) };

        // Check home group for existing key
        for si in matches {
            let bucket = unsafe { &mut *self.table.bucket_ptr(gi, si) };
            if bucket.0 == key {
                return Some(std::mem::replace(&mut bucket.1, value));
            }
        }

        // No overflow from home group → key is absent, insert directly
        let ofw_bit = overflow_bit(h);
        if let Some(si) = empties.lowest_set_bit()
            && !unsafe { Group::has_overflow_bit(meta, ofw_bit) }
        {
            unsafe {
                Group::set_meta(meta, si, reduced);
                self.table.bucket_ptr(gi, si).write((key, value));
            }
            self.table.len += 1;
            return None;
        }

        // Cold: overflow or full home group — full probe needed
        self.insert_overflow(h, key, value)
    }

    /// Cold path: insert when the home group has overflow or is full.
    /// Does a full probe to check for the key, then inserts.
    #[cold]
    #[inline(never)]
    fn insert_overflow(&mut self, h: u64, key: K, value: V) -> Option<V> {
        // Full probe for the key (starting from home group)
        if let Some((gi, si)) = self.table.find_by_hash(h, |k| k == &key) {
            let bucket = unsafe { &mut *self.table.bucket_ptr(gi, si) };
            return Some(std::mem::replace(&mut bucket.1, value));
        }

        if self.table.len >= self.table.max_load {
            self.grow_and_rehash();
        }
        self.table.insert_no_check(h, key, value);
        None
    }

    /// Cold path: insert when already at capacity.
    #[cold]
    #[inline(never)]
    fn insert_at_capacity(&mut self, h: u64, key: K, value: V) -> Option<V> {
        if let Some((gi, si)) = self.table.find_by_hash(h, |k| k == &key) {
            let bucket = unsafe { &mut *self.table.bucket_ptr(gi, si) };
            return Some(std::mem::replace(&mut bucket.1, value));
        }
        self.grow_and_rehash();
        self.table.insert_no_check(h, key, value);
        None
    }

    #[cold]
    #[inline(never)]
    fn grow_and_rehash(&mut self) {
        let new_groups = if !self.table.is_allocated() {
            1
        } else {
            self.table.num_groups() * 2
        };
        self.table.rehash_with(new_groups, &self.hash_builder);
    }

    /// Removes a key from the map, returning the value if it was present.
    #[inline]
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let h = self.hash_key(key);
        self.table.remove_by_hash(h, |k| k.borrow() == key)
    }

    /// Gets the given key's entry in the map for in-place manipulation.
    ///
    /// Uses the same fused home-group pattern as insert(): a single SIMD load
    /// checks for the key and locates an empty slot simultaneously.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, S> {
        use super::raw::group::{Group, overflow_bit, reduced_hash};

        if !self.table.is_allocated() {
            self.table.allocate(1);
        }

        let h = self.hash_key(&key);

        // At capacity: can't pre-locate a slot (growth would invalidate it)
        if self.table.len >= self.table.max_load {
            return self.entry_at_capacity(h, key);
        }

        let reduced = reduced_hash(h);
        let gi = self.table.group_index(h);
        let meta = unsafe { self.table.meta_ptr(gi) };
        let (matches, empties) = unsafe { Group::match_byte_and_empty(meta, reduced) };

        // Check home group for existing key
        for si in matches {
            let bucket = unsafe { &mut *self.table.bucket_ptr(gi, si) };
            if bucket.0 == key {
                return Entry::Occupied(OccupiedEntry {
                    key,
                    value: &mut bucket.1,
                });
            }
        }

        // No overflow from home group → key absent, slot in home group
        let ofw_bit = overflow_bit(h);
        if let Some(si) = empties.lowest_set_bit()
            && !unsafe { Group::has_overflow_bit(meta, ofw_bit) }
        {
            return Entry::Vacant(VacantEntry {
                key,
                hash: h,
                slot: Some((gi, si, 0)),
                table: &mut self.table,
                hash_builder: &self.hash_builder,
            });
        }

        // Cold: overflow or full home group — use full find_or_locate
        self.entry_overflow(h, key)
    }

    /// Cold path: entry when already at capacity.
    #[cold]
    #[inline(never)]
    fn entry_at_capacity(&mut self, h: u64, key: K) -> Entry<'_, K, V, S> {
        if let Some((gi, si)) = self.table.find_by_hash(h, |k| k == &key) {
            let bucket = unsafe { &mut *self.table.bucket_ptr(gi, si) };
            return Entry::Occupied(OccupiedEntry {
                key,
                value: &mut bucket.1,
            });
        }
        Entry::Vacant(VacantEntry {
            key,
            hash: h,
            slot: None,
            table: &mut self.table,
            hash_builder: &self.hash_builder,
        })
    }

    /// Cold path: entry when home group has overflow or is full.
    #[cold]
    #[inline(never)]
    fn entry_overflow(&mut self, h: u64, key: K) -> Entry<'_, K, V, S> {
        match self.table.find_or_locate(h, |k| k == &key) {
            ProbeResult::Found(gi, si) => {
                let bucket = unsafe { &mut *self.table.bucket_ptr(gi, si) };
                Entry::Occupied(OccupiedEntry {
                    key,
                    value: &mut bucket.1,
                })
            }
            ProbeResult::InsertSlot(gi, si, full_mask) => Entry::Vacant(VacantEntry {
                key,
                hash: h,
                slot: Some((gi, si, full_mask)),
                table: &mut self.table,
                hash_builder: &self.hash_builder,
            }),
            ProbeResult::NotFound => Entry::Vacant(VacantEntry {
                key,
                hash: h,
                slot: None,
                table: &mut self.table,
                hash_builder: &self.hash_builder,
            }),
        }
    }

    /// Iterate over key-value pairs.
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            inner: self.table.iter_slots(),
        }
    }

    /// Iterate over key-value pairs with mutable values.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            inner: self.table.iter_slots(),
        }
    }

    /// Iterate over keys.
    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { inner: self.iter() }
    }

    /// Iterate over values.
    pub fn values(&self) -> Values<'_, K, V> {
        Values { inner: self.iter() }
    }

    /// Iterate over mutable values.
    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut {
            inner: self.iter_mut(),
        }
    }
}

// ── Entry API ───────────────────────────────────────────────────────────────

pub enum Entry<'a, K, V, S> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V, S>),
}

pub struct OccupiedEntry<'a, K, V> {
    key: K,
    value: &'a mut V,
}

pub struct VacantEntry<'a, K, V, S> {
    key: K,
    hash: u64,
    /// Pre-located insertion slot from fused find_or_locate, if available.
    /// Tuple: (group_index, slot_index, full_groups_bitmask).
    slot: Option<(usize, usize, u8)>,
    table: &'a mut RawTable<K, V>,
    hash_builder: &'a S,
}

impl<'a, K: Hash + Eq, V, S: BuildHasher> Entry<'a, K, V, S> {
    /// Ensures a value is in the entry by inserting the default if empty.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.value,
            Entry::Vacant(e) => e.insert(default),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the
    /// function if empty.
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.value,
            Entry::Vacant(e) => e.insert(default()),
        }
    }

    /// Ensures a value is in the entry by inserting the default value if empty.
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }

    /// Returns a reference to this entry's key.
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(e) => &e.key,
            Entry::Vacant(e) => &e.key,
        }
    }
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    /// Gets a reference to the value.
    pub fn get(&self) -> &V {
        self.value
    }

    /// Gets a mutable reference to the value.
    pub fn get_mut(&mut self) -> &mut V {
        self.value
    }

    /// Sets the value and returns the old value.
    pub fn insert(&mut self, value: V) -> V {
        std::mem::replace(self.value, value)
    }

    /// Converts to a mutable reference to the value.
    pub fn into_mut(self) -> &'a mut V {
        self.value
    }
}

impl<'a, K: Hash + Eq, V, S: BuildHasher> VacantEntry<'a, K, V, S> {
    /// Insert a value and return a mutable reference.
    pub fn insert(self, value: V) -> &'a mut V {
        if let Some((gi, si, full_mask)) = self.slot {
            // Fast path: use pre-located slot from fused probe
            self.table
                .insert_at(self.hash, gi, si, self.key, value, full_mask);
            let bucket = unsafe { &mut *self.table.bucket_ptr(gi, si) };
            &mut bucket.1
        } else {
            // Slow path: need to grow first, then insert
            if self.table.len >= self.table.max_load {
                let new_groups = if !self.table.is_allocated() {
                    1
                } else {
                    self.table.num_groups() * 2
                };
                self.table.rehash_with(new_groups, self.hash_builder);
            }
            let (gi, si) = self.table.insert_no_check(self.hash, self.key, value);
            let bucket = unsafe { &mut *self.table.bucket_ptr(gi, si) };
            &mut bucket.1
        }
    }

    /// Gets a reference to the key.
    pub fn key(&self) -> &K {
        &self.key
    }
}

// ── Iterators ───────────────────────────────────────────────────────────────

pub struct Iter<'a, K, V> {
    inner: super::raw::SlotIter<'a, K, V>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let (gi, si) = self.inner.next()?;
        let bucket = unsafe { &*self.inner.table.bucket_ptr(gi, si) };
        Some((&bucket.0, &bucket.1))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> FusedIterator for Iter<'_, K, V> {}

pub struct IterMut<'a, K, V> {
    inner: super::raw::SlotIter<'a, K, V>,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        let (gi, si) = self.inner.next()?;
        // SAFETY: We have &mut self, and each slot is visited only once.
        let bucket = unsafe { &mut *self.inner.table.bucket_ptr(gi, si) };
        Some((&bucket.0, &mut bucket.1))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<K, V> FusedIterator for IterMut<'_, K, V> {}

pub struct IntoIter<K, V> {
    table: RawTable<K, V>,
    group: usize,
    current_mask: crate::raw::bitmask::BitMask,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(si) = self.current_mask.next() {
                let gi = self.group;
                unsafe {
                    let ptr = self.table.bucket_ptr(gi, si);
                    let kv = ptr.read();
                    // Mark as empty so Drop doesn't double-free
                    let meta = self
                        .table
                        .metadata
                        .add(gi * super::raw::group::META_GROUP_BYTES + si);
                    *meta = super::raw::group::EMPTY;
                    self.table.len -= 1;
                    return Some(kv);
                }
            }
            self.group += 1;
            if self.group > self.table.mask {
                return None;
            }
            self.current_mask = unsafe {
                super::raw::group::Group::match_non_empty(
                    self.table
                        .metadata
                        .add(self.group * super::raw::group::META_GROUP_BYTES),
                )
            };
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.table.len, Some(self.table.len))
    }
}

impl<K, V> ExactSizeIterator for IntoIter<K, V> {}
impl<K, V> FusedIterator for IntoIter<K, V> {}

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

impl<K, V> FusedIterator for Keys<'_, K, V> {}

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

impl<K, V> FusedIterator for Values<'_, K, V> {}

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

impl<K, V> FusedIterator for ValuesMut<'_, K, V> {}

// ── Trait implementations ───────────────────────────────────────────────────

impl<K, V> Default for Gaps<K, V, DefaultHashBuilder> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, S> IntoIterator for Gaps<K, V, S> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> IntoIter<K, V> {
        let table = unsafe { std::ptr::read(&self.table) };
        std::mem::forget(self);
        let mask = if table.metadata.is_null() {
            crate::raw::bitmask::BitMask(0)
        } else {
            unsafe { super::raw::group::Group::match_non_empty(table.metadata) }
        };
        IntoIter {
            table,
            group: 0,
            current_mask: mask,
        }
    }
}

impl<'a, K, V, S> IntoIterator for &'a Gaps<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Iter<'a, K, V> {
        self.iter()
    }
}

impl<K, V, S> FromIterator<(K, V)> for Gaps<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher + Default,
{
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut map = Self::with_capacity_and_hasher(lower, S::default());
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

impl<K, V, S> Extend<(K, V)> for Gaps<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K, V, S, Q> Index<&Q> for Gaps<K, V, S>
where
    K: Hash + Eq + Borrow<Q>,
    Q: Hash + Eq + ?Sized,
    S: BuildHasher,
{
    type Output = V;

    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

impl<K, V, S> fmt::Debug for Gaps<K, V, S>
where
    K: Hash + Eq + fmt::Debug,
    V: fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V, S> Clone for Gaps<K, V, S>
where
    K: Clone,
    V: Clone,
    S: Clone,
{
    fn clone(&self) -> Self {
        Gaps {
            table: self.table.clone(),
            hash_builder: self.hash_builder.clone(),
        }
    }
}

impl<K, V, S> PartialEq for Gaps<K, V, S>
where
    K: Hash + Eq,
    V: PartialEq,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        self.iter().all(|(k, v)| other.get(k) == Some(v))
    }
}

impl<K, V, S> Eq for Gaps<K, V, S>
where
    K: Hash + Eq,
    V: Eq,
    S: BuildHasher,
{
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_operations() {
        let mut map = Gaps::new();
        assert!(map.is_empty());

        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");

        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&1), Some(&"one"));
        assert_eq!(map.get(&2), Some(&"two"));
        assert_eq!(map.get(&3), Some(&"three"));
        assert_eq!(map.get(&4), None);
    }

    #[test]
    fn insert_replace() {
        let mut map = Gaps::new();
        assert_eq!(map.insert(1, "a"), None);
        assert_eq!(map.insert(1, "b"), Some("a"));
        assert_eq!(map.get(&1), Some(&"b"));
    }

    #[test]
    fn remove() {
        let mut map = Gaps::new();
        map.insert(1, 10);
        map.insert(2, 20);

        assert_eq!(map.remove(&1), Some(10));
        assert_eq!(map.len(), 1);
        assert!(!map.contains_key(&1));
        assert!(map.contains_key(&2));

        assert_eq!(map.remove(&99), None);
    }

    #[test]
    fn entry_api() {
        let mut map = Gaps::new();

        map.entry(1).or_insert(10);
        assert_eq!(map.get(&1), Some(&10));

        map.entry(1).or_insert(20);
        assert_eq!(map.get(&1), Some(&10)); // not replaced

        *map.entry(2).or_insert(0) += 5;
        assert_eq!(map.get(&2), Some(&5));
    }

    #[test]
    fn from_iterator() {
        let map: Gaps<i32, &str> = vec![(1, "a"), (2, "b"), (3, "c")].into_iter().collect();
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&2), Some(&"b"));
    }

    #[test]
    fn into_iter() {
        let mut map = Gaps::new();
        for i in 0..10 {
            map.insert(i, i * 10);
        }

        let mut pairs: Vec<(i32, i32)> = map.into_iter().collect();
        pairs.sort();
        assert_eq!(pairs.len(), 10);
        for (i, pair) in pairs.iter().enumerate().take(10) {
            assert_eq!(*pair, (i as i32, (i * 10) as i32));
        }
    }

    #[test]
    fn keys_values() {
        let mut map = Gaps::new();
        map.insert(1, "a");
        map.insert(2, "b");

        let mut keys: Vec<_> = map.keys().copied().collect();
        keys.sort();
        assert_eq!(keys, vec![1, 2]);

        let mut values: Vec<_> = map.values().copied().collect();
        values.sort();
        assert_eq!(values, vec!["a", "b"]);
    }

    #[test]
    fn debug_display() {
        let mut map = Gaps::new();
        map.insert(1, "one");
        let debug = format!("{:?}", map);
        assert!(debug.contains("1"));
        assert!(debug.contains("one"));
    }

    #[test]
    fn clone_and_eq() {
        let mut map = Gaps::new();
        map.insert(1, 10);
        map.insert(2, 20);

        let cloned = map.clone();
        assert_eq!(map, cloned);

        let mut different = Gaps::new();
        different.insert(1, 10);
        assert_ne!(map, different);
    }

    #[test]
    fn index_operator() {
        let mut map = Gaps::new();
        map.insert("hello", 42);
        assert_eq!(map[&"hello"], 42);
    }

    #[test]
    #[should_panic(expected = "no entry found")]
    fn index_missing_key() {
        let map: Gaps<i32, i32> = Gaps::new();
        let _ = map[&1];
    }

    #[test]
    fn large_insert() {
        let mut map = Gaps::new();
        for i in 0..10_000 {
            map.insert(i, i * 2);
        }
        assert_eq!(map.len(), 10_000);
        for i in 0..10_000 {
            assert_eq!(map.get(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn extend() {
        let mut map = Gaps::new();
        map.insert(1, 10);
        map.extend(vec![(2, 20), (3, 30)]);
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn with_capacity() {
        let map: Gaps<i32, i32> = Gaps::with_capacity(100);
        assert!(map.capacity() >= 100);
        assert!(map.is_empty());
    }
}

crate::traits::impl_map_trait!(Gaps);
