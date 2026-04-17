//! Splitsies for the 16-slot split-overflow design.
//!
//! This is a copy of the main map.rs adapted to use the split_overflow
//! raw table. Key differences:
//! - Uses `super::raw::group` (16-slot) instead of `crate::raw::group` (15-slot)
//! - Fused insert/entry paths pass overflow_ptr to Group methods

use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::iter::FusedIterator;
use std::ops::Index;

use super::raw::{ProbeResult, RawTable};
use crate::raw::hash;

pub type DefaultHashBuilder = foldhash::fast::RandomState;

pub struct Splitsies<K, V, S = DefaultHashBuilder> {
    table: RawTable<K, V>,
    hash_builder: S,
}

// ── Constructors ────────────────────────────────────────────────────────────

impl<K, V> Splitsies<K, V, DefaultHashBuilder> {
    pub fn new() -> Self {
        Self::with_hasher(DefaultHashBuilder::default())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, DefaultHashBuilder::default())
    }
}

impl<K, V, S> Splitsies<K, V, S> {
    pub fn with_hasher(hash_builder: S) -> Self {
        Splitsies {
            table: RawTable::new(),
            hash_builder,
        }
    }

    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        Splitsies {
            table: RawTable::with_capacity(capacity),
            hash_builder,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.table.capacity()
    }

    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }

    pub fn clear(&mut self) {
        self.table.clear();
    }
}

// ── Core operations ─────────────────────────────────────────────────────────

impl<K, V, S> Splitsies<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    #[inline(always)]
    fn hash_key<Q: Hash + ?Sized>(&self, key: &Q) -> u64 {
        hash::hash_no_mix(key, &self.hash_builder)
    }

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

    /// Returns the key-value pair corresponding to the key.
    #[inline]
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let h = self.hash_key(key);
        let (gi, si) = self.table.find_by_hash(h, |k| k.borrow() == key)?;
        let bucket = unsafe { &*self.table.bucket_ptr(gi, si) };
        Some((&bucket.0, &bucket.1))
    }

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

    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get(key).is_some()
    }

    /// Fused home-group insert using the 16-slot split-overflow design.
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        use super::raw::group::{Group, overflow_bit, reduced_hash};

        if !self.table.is_allocated() {
            self.table.allocate(1);
        }

        let h = self.hash_key(&key);

        if self.table.len >= self.table.max_load {
            return self.insert_at_capacity(h, key, value);
        }

        let reduced = reduced_hash(h);
        let gi = self.table.group_index(h);

        // Prefetch overflow byte while SIMD match executes
        unsafe {
            Group::prefetch_read(self.table.overflow_ptr(gi) as *const u8);
        }

        let meta = unsafe { self.table.meta_ptr(gi) };
        let (matches, empties) = unsafe { Group::match_byte_and_empty(meta, reduced) };

        for si in matches {
            let bucket = unsafe { &mut *self.table.bucket_ptr(gi, si) };
            if bucket.0 == key {
                return Some(std::mem::replace(&mut bucket.1, value));
            }
        }

        // Overflow byte should be in cache by now
        let ofw_bit = overflow_bit(h);
        if let Some(si) = empties.lowest_set_bit() {
            let ofw_ptr = unsafe { self.table.overflow_ptr(gi) };
            if !unsafe { Group::has_overflow_bit(ofw_ptr, ofw_bit) } {
                unsafe {
                    Group::set_meta(meta, si, reduced);
                    self.table.bucket_ptr(gi, si).write((key, value));
                }
                self.table.len += 1;
                return None;
            }
        }

        self.insert_overflow(h, key, value)
    }

    #[cold]
    #[inline(never)]
    fn insert_overflow(&mut self, h: u64, key: K, value: V) -> Option<V> {
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

    #[inline]
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let h = self.hash_key(key);
        self.table
            .remove_by_hash(h, |k| k.borrow() == key)
            .map(|(_, v)| v)
    }

    /// Removes a key from the map, returning the key and value if it was present.
    #[inline]
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let h = self.hash_key(key);
        self.table.remove_by_hash(h, |k| k.borrow() == key)
    }

    /// Fused home-group entry using split-overflow.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, S> {
        use super::raw::group::{Group, overflow_bit, reduced_hash};

        if !self.table.is_allocated() {
            self.table.allocate(1);
        }

        let h = self.hash_key(&key);

        if self.table.len >= self.table.max_load {
            return self.entry_at_capacity(h, key);
        }

        let reduced = reduced_hash(h);
        let gi = self.table.group_index(h);

        // Prefetch overflow byte while SIMD match executes
        unsafe {
            Group::prefetch_read(self.table.overflow_ptr(gi) as *const u8);
        }

        let meta = unsafe { self.table.meta_ptr(gi) };
        let (matches, empties) = unsafe { Group::match_byte_and_empty(meta, reduced) };

        for si in matches {
            let bucket = unsafe { &mut *self.table.bucket_ptr(gi, si) };
            if bucket.0 == key {
                return Entry::Occupied(OccupiedEntry {
                    key,
                    value: &mut bucket.1,
                });
            }
        }

        let ofw_bit = overflow_bit(h);
        if let Some(si) = empties.lowest_set_bit() {
            let ofw_ptr = unsafe { self.table.overflow_ptr(gi) };
            if !unsafe { Group::has_overflow_bit(ofw_ptr, ofw_bit) } {
                return Entry::Vacant(VacantEntry {
                    key,
                    hash: h,
                    slot: Some((gi, si, 0)),
                    table: &mut self.table,
                    hash_builder: &self.hash_builder,
                });
            }
        }

        self.entry_overflow(h, key)
    }

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

    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            inner: self.table.iter_slots(),
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            inner: self.table.iter_slots(),
        }
    }

    pub fn keys(&self) -> Keys<'_, K, V> {
        Keys { inner: self.iter() }
    }

    pub fn values(&self) -> Values<'_, K, V> {
        Values { inner: self.iter() }
    }

    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V> {
        ValuesMut {
            inner: self.iter_mut(),
        }
    }

    /// Retains only the elements specified by the predicate.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        use super::raw::group::{Group, EMPTY, overflow_bit};

        if !self.table.is_allocated() {
            return;
        }

        for gi in 0..=self.table.mask {
            let meta = unsafe { self.table.meta_ptr(gi) };
            let occupied = unsafe { Group::match_non_empty(meta) };
            for si in occupied {
                let bucket = unsafe { &mut *self.table.bucket_ptr(gi, si) };
                if !f(&bucket.0, &mut bucket.1) {
                    let h = self.hash_key(&bucket.0);
                    unsafe {
                        std::ptr::drop_in_place(bucket);
                        Group::set_meta(meta, si, EMPTY);
                    }
                    self.table.len -= 1;

                    // Anti-drift (Splitsies uses overflow_ptr, not meta_ptr)
                    let initial_gi = self.table.group_index(h);
                    let ofw_bit = overflow_bit(h);
                    if unsafe { Group::has_overflow_bit(self.table.overflow_ptr(initial_gi), ofw_bit) } {
                        self.table.max_load = self.table.max_load.saturating_sub(1);
                    }
                }
            }
        }
    }

    /// Reserves capacity for at least `additional` more elements.
    pub fn reserve(&mut self, additional: usize) {
        use super::raw::RawTable;
        let needed = self.table.len.checked_add(additional).expect("capacity overflow");
        if !self.table.is_allocated() {
            if additional > 0 {
                self.table.allocate(RawTable::<K, V>::groups_for_capacity(needed));
            }
            return;
        }
        if needed > self.table.max_load {
            let new_groups = RawTable::<K, V>::groups_for_capacity(needed);
            if new_groups > self.table.num_groups() {
                self.table.rehash_with(new_groups, &self.hash_builder);
            }
        }
    }

    /// Shrinks the capacity as much as possible.
    pub fn shrink_to_fit(&mut self) {
        use super::raw::RawTable;
        if self.table.len == 0 {
            let mut empty = RawTable::new();
            std::mem::swap(&mut self.table, &mut empty);
            return;
        }
        let min_groups = RawTable::<K, V>::groups_for_capacity(self.table.len);
        if min_groups < self.table.num_groups() {
            self.table.rehash_with(min_groups, &self.hash_builder);
        }
    }

    /// Clears the map, returning all key-value pairs as an iterator.
    pub fn drain(&mut self) -> IntoIter<K, V> {
        use super::raw::group::Group;
        use super::raw::RawTable;
        let table = std::mem::replace(&mut self.table, RawTable::new());
        let mask = if table.metadata.is_null() {
            crate::raw::bitmask::BitMask(0)
        } else {
            unsafe { Group::match_non_empty(table.metadata) }
        };
        IntoIter {
            table,
            group: 0,
            current_mask: mask,
        }
    }

    /// Tries to insert a key-value pair, failing if the key already exists.
    pub fn try_insert(&mut self, key: K, value: V) -> Result<(), crate::traits::OccupiedError<K, V>> {
        match self.entry(key) {
            Entry::Occupied(e) => Err(crate::traits::OccupiedError { key: e.key, value }),
            Entry::Vacant(e) => { e.insert(value); Ok(()) }
        }
    }

    /// Creates a consuming iterator over the keys.
    pub fn into_keys(self) -> impl Iterator<Item = K> {
        self.into_iter().map(|(k, _)| k)
    }

    /// Creates a consuming iterator over the values.
    pub fn into_values(self) -> impl Iterator<Item = V> {
        self.into_iter().map(|(_, v)| v)
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
    slot: Option<(usize, usize, u8)>,
    table: &'a mut RawTable<K, V>,
    hash_builder: &'a S,
}

impl<'a, K: Hash + Eq, V, S: BuildHasher> Entry<'a, K, V, S> {
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.value,
            Entry::Vacant(e) => e.insert(default),
        }
    }

    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.value,
            Entry::Vacant(e) => e.insert(default()),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the
    /// function (which receives the key) if empty.
    pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.value,
            Entry::Vacant(e) => {
                let value = default(&e.key);
                e.insert(value)
            }
        }
    }

    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }

    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(e) => &e.key,
            Entry::Vacant(e) => &e.key,
        }
    }

    /// Provides in-place mutable access to an occupied entry before any
    /// potential inserts.
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

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    pub fn key(&self) -> &K {
        &self.key
    }
    pub fn get(&self) -> &V {
        self.value
    }
    pub fn get_mut(&mut self) -> &mut V {
        self.value
    }
    pub fn insert(&mut self, value: V) -> V {
        std::mem::replace(self.value, value)
    }
    pub fn into_mut(self) -> &'a mut V {
        self.value
    }
}

impl<'a, K: Hash + Eq, V, S: BuildHasher> VacantEntry<'a, K, V, S> {
    /// Takes ownership of the key.
    pub fn into_key(self) -> K {
        self.key
    }

    pub fn insert(self, value: V) -> &'a mut V {
        if let Some((gi, si, full_mask)) = self.slot {
            self.table
                .insert_at(self.hash, gi, si, self.key, value, full_mask);
            let bucket = unsafe { &mut *self.table.bucket_ptr(gi, si) };
            &mut bucket.1
        } else {
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
        use super::raw::group::{EMPTY, Group, META_GROUP_BYTES};
        loop {
            if let Some(si) = self.current_mask.next() {
                let gi = self.group;
                unsafe {
                    let ptr = self.table.bucket_ptr(gi, si);
                    let kv = ptr.read();
                    let meta = self.table.metadata.add(gi * META_GROUP_BYTES + si);
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
                Group::match_non_empty(self.table.metadata.add(self.group * META_GROUP_BYTES))
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

impl<K, V> Default for Splitsies<K, V, DefaultHashBuilder> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, S> IntoIterator for Splitsies<K, V, S> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> IntoIter<K, V> {
        use super::raw::group::Group;
        let table = unsafe { std::ptr::read(&self.table) };
        std::mem::forget(self);
        let mask = if table.metadata.is_null() {
            crate::raw::bitmask::BitMask(0)
        } else {
            unsafe { Group::match_non_empty(table.metadata) }
        };
        IntoIter {
            table,
            group: 0,
            current_mask: mask,
        }
    }
}

impl<'a, K, V, S> IntoIterator for &'a Splitsies<K, V, S>
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

impl<K, V, S> FromIterator<(K, V)> for Splitsies<K, V, S>
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

impl<K, V, S> Extend<(K, V)> for Splitsies<K, V, S>
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

impl<K, V, S, Q> Index<&Q> for Splitsies<K, V, S>
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

impl<K, V, S> fmt::Debug for Splitsies<K, V, S>
where
    K: Hash + Eq + fmt::Debug,
    V: fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V, S> Clone for Splitsies<K, V, S>
where
    K: Clone,
    V: Clone,
    S: Clone,
{
    fn clone(&self) -> Self {
        Splitsies {
            table: self.table.clone(),
            hash_builder: self.hash_builder.clone(),
        }
    }
}

impl<K, V, S> PartialEq for Splitsies<K, V, S>
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

impl<K, V, S> Eq for Splitsies<K, V, S>
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
        let mut map = Splitsies::new();
        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&1), Some(&"one"));
        assert_eq!(map.get(&4), None);
    }

    #[test]
    fn insert_replace() {
        let mut map = Splitsies::new();
        assert_eq!(map.insert(1, "a"), None);
        assert_eq!(map.insert(1, "b"), Some("a"));
        assert_eq!(map.get(&1), Some(&"b"));
    }

    #[test]
    fn remove() {
        let mut map = Splitsies::new();
        map.insert(1, 10);
        map.insert(2, 20);
        assert_eq!(map.remove(&1), Some(10));
        assert_eq!(map.len(), 1);
        assert!(!map.contains_key(&1));
    }

    #[test]
    fn entry_api() {
        let mut map = Splitsies::new();
        map.entry(1).or_insert(10);
        assert_eq!(map.get(&1), Some(&10));
        map.entry(1).or_insert(20);
        assert_eq!(map.get(&1), Some(&10));
        *map.entry(2).or_insert(0) += 5;
        assert_eq!(map.get(&2), Some(&5));
    }

    #[test]
    fn large_insert() {
        let mut map = Splitsies::new();
        for i in 0..10_000 {
            map.insert(i, i * 2);
        }
        assert_eq!(map.len(), 10_000);
        for i in 0..10_000 {
            assert_eq!(map.get(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn from_iterator() {
        let map: Splitsies<i32, &str> = vec![(1, "a"), (2, "b"), (3, "c")].into_iter().collect();
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&2), Some(&"b"));
    }

    #[test]
    fn into_iter() {
        let mut map = Splitsies::new();
        for i in 0..10 {
            map.insert(i, i * 10);
        }
        let mut pairs: Vec<(i32, i32)> = map.into_iter().collect();
        pairs.sort();
        assert_eq!(pairs.len(), 10);
    }

    #[test]
    fn clone_and_eq() {
        let mut map = Splitsies::new();
        map.insert(1, 10);
        map.insert(2, 20);
        let cloned = map.clone();
        assert_eq!(map, cloned);
    }

    #[test]
    fn with_capacity() {
        let map: Splitsies<i32, i32> = Splitsies::with_capacity(100);
        assert!(map.capacity() >= 100);
        assert!(map.is_empty());
    }

    #[test]
    fn insert_remove_cycle() {
        let mut map = Splitsies::new();
        for cycle in 0..3 {
            for i in 0..100 {
                map.insert(i, i + cycle * 1000);
            }
            for i in 0..100 {
                map.remove(&i);
            }
            assert_eq!(map.len(), 0);
        }
    }
}

crate::traits::impl_map_trait!(Splitsies);
