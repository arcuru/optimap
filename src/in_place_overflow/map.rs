//! InPlaceOverflow — tombstone-based Swiss-table design (no overflow bytes).

use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::iter::FusedIterator;
use std::ops::Index;

use super::raw::{ProbeResult, RawTable};
use crate::raw::hash;

pub type DefaultHashBuilder = foldhash::fast::RandomState;

pub struct InPlaceOverflow<K, V, S = DefaultHashBuilder> {
    table: RawTable<K, V>,
    hash_builder: S,
}

// ── Constructors ────────────────────────────────────────────────────────────

impl<K, V> InPlaceOverflow<K, V, DefaultHashBuilder> {
    pub fn new() -> Self {
        Self::with_hasher(DefaultHashBuilder::default())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, DefaultHashBuilder::default())
    }
}

impl<K, V, S> InPlaceOverflow<K, V, S> {
    pub fn with_hasher(hash_builder: S) -> Self {
        InPlaceOverflow {
            table: RawTable::new(),
            hash_builder,
        }
    }

    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        InPlaceOverflow {
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

impl<K, V, S> InPlaceOverflow<K, V, S>
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

    /// Fused home-group insert: EMPTY in home group proves absence.
    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        use super::raw::group::{Group, reduced_hash};

        if !self.table.is_allocated() {
            self.table.allocate(1);
        }

        let h = self.hash_key(&key);

        if self.table.growth_left == 0 {
            return self.insert_at_capacity(h, key, value);
        }

        let reduced = reduced_hash(h);
        let gi = self.table.group_index(h);

        let meta = unsafe { self.table.meta_ptr(gi) };
        let (matches, empties) = unsafe { Group::match_byte_and_empty(meta, reduced) };

        for si in matches {
            let bucket = unsafe { &mut *self.table.bucket_ptr(gi, si) };
            if bucket.0 == key {
                return Some(std::mem::replace(&mut bucket.1, value));
            }
        }

        // EMPTY in home group proves key is absent — insert directly.
        // Don't bother checking for tombstones in the home group; the EMPTY
        // slot is equally close to the home position. Tombstone preference
        // only matters in insert_no_check (cold path, longer probe chains).
        if let Some(si) = empties.lowest_set_bit() {
            unsafe {
                Group::set_meta(meta, si, reduced);
                self.table.bucket_ptr(gi, si).write((key, value));
            }
            self.table.len += 1;
            self.table.growth_left -= 1;
            return None;
        }

        // No EMPTY in home group — must probe further
        self.insert_overflow(h, key, value)
    }

    #[cold]
    #[inline(never)]
    fn insert_overflow(&mut self, h: u64, key: K, value: V) -> Option<V> {
        if let Some((gi, si)) = self.table.find_by_hash(h, |k| k == &key) {
            let bucket = unsafe { &mut *self.table.bucket_ptr(gi, si) };
            return Some(std::mem::replace(&mut bucket.1, value));
        }

        if self.table.growth_left == 0 {
            self.grow_or_rehash();
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
        self.grow_or_rehash();
        self.table.insert_no_check(h, key, value);
        None
    }

    #[cold]
    #[inline(never)]
    fn grow_or_rehash(&mut self) {
        let new_groups = if !self.table.is_allocated() {
            1
        } else {
            let cap = self.table.num_groups() * super::raw::group::GROUP_SIZE;
            if self.table.len >= cap * 7 / 8 {
                self.table.num_groups() * 2
            } else {
                // Tombstones are eating growth_left — rehash in place
                self.table.num_groups()
            }
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
        self.table.remove_by_hash(h, |k| k.borrow() == key)
    }

    /// Fused home-group entry: EMPTY in home group proves absence.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, S> {
        use super::raw::group::{Group, reduced_hash};

        if !self.table.is_allocated() {
            self.table.allocate(1);
        }

        let h = self.hash_key(&key);

        if self.table.growth_left == 0 {
            return self.entry_at_capacity(h, key);
        }

        let reduced = reduced_hash(h);
        let gi = self.table.group_index(h);

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

        // EMPTY in home group proves absence — use the empty slot directly
        if let Some(si) = empties.lowest_set_bit() {
            return Entry::Vacant(VacantEntry {
                key,
                hash: h,
                slot: Some((gi, si, 0)),
                table: &mut self.table,
                hash_builder: &self.hash_builder,
            });
        }

        // No EMPTY — must probe
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
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
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
    pub fn insert(self, value: V) -> &'a mut V {
        if let Some((gi, si, full_mask)) = self.slot {
            self.table
                .insert_at(self.hash, gi, si, self.key, value, full_mask);
            let bucket = unsafe { &mut *self.table.bucket_ptr(gi, si) };
            &mut bucket.1
        } else {
            if self.table.growth_left == 0 {
                let cap = self.table.num_groups() * super::raw::group::GROUP_SIZE;
                let new_groups = if !self.table.is_allocated() {
                    1
                } else if self.table.len >= cap * 7 / 8 {
                    self.table.num_groups() * 2
                } else {
                    self.table.num_groups()
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
                Group::match_occupied(self.table.metadata.add(self.group * META_GROUP_BYTES))
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

impl<K, V> Default for InPlaceOverflow<K, V, DefaultHashBuilder> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, S> IntoIterator for InPlaceOverflow<K, V, S> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> IntoIter<K, V> {
        use super::raw::group::Group;
        let table = unsafe { std::ptr::read(&self.table) };
        std::mem::forget(self);
        let mask = if table.metadata.is_null() {
            crate::raw::bitmask::BitMask(0)
        } else {
            unsafe { Group::match_occupied(table.metadata) }
        };
        IntoIter {
            table,
            group: 0,
            current_mask: mask,
        }
    }
}

impl<'a, K, V, S> IntoIterator for &'a InPlaceOverflow<K, V, S>
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

impl<K, V, S> FromIterator<(K, V)> for InPlaceOverflow<K, V, S>
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

impl<K, V, S> Extend<(K, V)> for InPlaceOverflow<K, V, S>
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

impl<K, V, S, Q> Index<&Q> for InPlaceOverflow<K, V, S>
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

impl<K, V, S> fmt::Debug for InPlaceOverflow<K, V, S>
where
    K: Hash + Eq + fmt::Debug,
    V: fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V, S> Clone for InPlaceOverflow<K, V, S>
where
    K: Clone,
    V: Clone,
    S: Clone,
{
    fn clone(&self) -> Self {
        InPlaceOverflow {
            table: self.table.clone(),
            hash_builder: self.hash_builder.clone(),
        }
    }
}

impl<K, V, S> PartialEq for InPlaceOverflow<K, V, S>
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

impl<K, V, S> Eq for InPlaceOverflow<K, V, S>
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
        let mut map = InPlaceOverflow::new();
        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&1), Some(&"one"));
        assert_eq!(map.get(&4), None);
    }

    #[test]
    fn insert_replace() {
        let mut map = InPlaceOverflow::new();
        assert_eq!(map.insert(1, "a"), None);
        assert_eq!(map.insert(1, "b"), Some("a"));
        assert_eq!(map.get(&1), Some(&"b"));
    }

    #[test]
    fn remove() {
        let mut map = InPlaceOverflow::new();
        map.insert(1, 10);
        map.insert(2, 20);
        assert_eq!(map.remove(&1), Some(10));
        assert_eq!(map.len(), 1);
        assert!(!map.contains_key(&1));
    }

    #[test]
    fn entry_api() {
        let mut map = InPlaceOverflow::new();
        map.entry(1).or_insert(10);
        assert_eq!(map.get(&1), Some(&10));
        map.entry(1).or_insert(20);
        assert_eq!(map.get(&1), Some(&10));
        *map.entry(2).or_insert(0) += 5;
        assert_eq!(map.get(&2), Some(&5));
    }

    #[test]
    fn large_insert() {
        let mut map = InPlaceOverflow::new();
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
        let map: InPlaceOverflow<i32, &str> =
            vec![(1, "a"), (2, "b"), (3, "c")].into_iter().collect();
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&2), Some(&"b"));
    }

    #[test]
    fn into_iter() {
        let mut map = InPlaceOverflow::new();
        for i in 0..10 {
            map.insert(i, i * 10);
        }
        let mut pairs: Vec<(i32, i32)> = map.into_iter().collect();
        pairs.sort();
        assert_eq!(pairs.len(), 10);
    }

    #[test]
    fn clone_and_eq() {
        let mut map = InPlaceOverflow::new();
        map.insert(1, 10);
        map.insert(2, 20);
        let cloned = map.clone();
        assert_eq!(map, cloned);
    }

    #[test]
    fn with_capacity() {
        let map: InPlaceOverflow<i32, i32> = InPlaceOverflow::with_capacity(100);
        assert!(map.capacity() >= 100);
        assert!(map.is_empty());
    }

    #[test]
    fn insert_remove_cycle() {
        let mut map = InPlaceOverflow::new();
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

    #[test]
    fn tombstone_reuse() {
        let mut map = InPlaceOverflow::new();

        // Insert and remove to create tombstones
        for i in 0..50 {
            map.insert(i, i);
        }
        for i in 0..50 {
            map.remove(&i);
        }
        assert_eq!(map.len(), 0);

        // Re-insert — should reuse tombstone slots
        for i in 0..50 {
            map.insert(i, i * 100);
        }
        assert_eq!(map.len(), 50);
        for i in 0..50 {
            assert_eq!(map.get(&i), Some(&(i * 100)));
        }
    }
}

crate::traits::impl_map_trait!(InPlaceOverflow);
