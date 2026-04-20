//! Generic hash map wrapper over any `RawTableApi` backend.
//!
//! `GenericMap<K, V, S, R>` provides the full map API (constructors, entry API,
//! iterators, trait impls) once. Concrete map types are type aliases:
//! - `UnorderedFlatMap<K, V, S>` = `GenericMap<K, V, S, overflow::RawTable<K, V, UfmLayout>>`
//! - `Splitsies<K, V, S>` = `GenericMap<K, V, S, overflow::RawTable<K, V, SplitsiesLayout>>`
//! - etc.

use std::borrow::Borrow;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::iter::FusedIterator;
use std::ops::Index;
use std::ptr;

use crate::raw::hash;
use crate::raw::table_api::{EntryProbe, RawTableApi};

/// Default hasher for all map types.
pub type DefaultHashBuilder = foldhash::fast::RandomState;

/// A hash map generic over its raw table backend.
///
/// `R: RawTableApi<K, V>` provides the probe strategy, SIMD operations,
/// and growth policy. This wrapper adds the public API, entry API,
/// iterators, and trait implementations.
pub struct GenericMap<K, V, S = DefaultHashBuilder, R: RawTableApi<K, V> = crate::raw::overflow_table::RawTable<K, V, crate::raw::group_layout::UfmLayout>> {
    pub(crate) table: R,
    pub(crate) hash_builder: S,
    _marker: std::marker::PhantomData<(K, V)>,
}

// ── Constructors ───────────────────────────────────────────────────────────

impl<K, V, R: RawTableApi<K, V>> GenericMap<K, V, DefaultHashBuilder, R> {
    pub fn new() -> Self {
        Self::with_hasher(DefaultHashBuilder::default())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, DefaultHashBuilder::default())
    }
}

impl<K, V, S, R: RawTableApi<K, V>> GenericMap<K, V, S, R> {
    pub fn with_hasher(hash_builder: S) -> Self {
        GenericMap {
            table: R::new(),
            hash_builder,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        GenericMap {
            table: R::with_capacity(capacity),
            hash_builder,
            _marker: std::marker::PhantomData,
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

// ── Core operations ────────────────────────────────────────────────────────

impl<K, V, S, R: RawTableApi<K, V>> GenericMap<K, V, S, R>
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
        Some(unsafe { &*self.table.value_ptr(gi, si) })
    }

    #[inline]
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let h = self.hash_key(key);
        let (gi, si) = self.table.find_by_hash(h, |k| k.borrow() == key)?;
        unsafe { Some((&*self.table.key_ptr(gi, si), &*self.table.value_ptr(gi, si))) }
    }

    #[inline]
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let h = self.hash_key(key);
        let (gi, si) = self.table.find_by_hash(h, |k| k.borrow() == key)?;
        Some(unsafe { &mut *self.table.value_ptr(gi, si) })
    }

    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get(key).is_some()
    }

    #[inline]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.table.insert_or_replace(key, value, &self.hash_builder)
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

    #[inline]
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let h = self.hash_key(key);
        self.table.remove_by_hash(h, |k| k.borrow() == key)
    }

    /// Gets the given key's entry in the map for in-place manipulation.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V, S, R> {
        if !self.table.is_allocated() {
            // Ensure table is allocated for entry probing
            self.table.reserve(1, &self.hash_builder);
        }

        let h = self.hash_key(&key);

        match self.table.find_for_entry(h, &key) {
            EntryProbe::Found(gi, si) => {
                let value = unsafe { &mut *self.table.value_ptr(gi, si) };
                Entry::Occupied(OccupiedEntry {
                    key,
                    value,
                })
            }
            EntryProbe::Vacant(slot) => Entry::Vacant(VacantEntry {
                key,
                hash: h,
                slot,
                table: &mut self.table,
                hash_builder: &self.hash_builder,
                _marker: std::marker::PhantomData,
            }),
        }
    }

    pub fn iter(&self) -> Iter<'_, K, V, R> {
        Iter {
            inner: self.table.iter_slots(),
            table: &self.table,
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, K, V, R> {
        IterMut {
            inner: self.table.iter_slots(),
            table: &self.table,
        }
    }

    pub fn keys(&self) -> Keys<'_, K, V, R> {
        Keys { inner: self.iter() }
    }

    pub fn values(&self) -> Values<'_, K, V, R> {
        Values { inner: self.iter() }
    }

    pub fn values_mut(&mut self) -> ValuesMut<'_, K, V, R> {
        ValuesMut {
            inner: self.iter_mut(),
        }
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        if !self.table.is_allocated() {
            return;
        }

        let positions = self.table.occupied_positions();
        for (gi, si) in positions {
            let key = unsafe { &*self.table.key_ptr(gi, si) };
            let value = unsafe { &mut *self.table.value_ptr(gi, si) };
            if !f(key, value) {
                let h = self.hash_key(key);
                unsafe {
                    ptr::drop_in_place(self.table.key_ptr(gi, si) as *mut K);
                    ptr::drop_in_place(self.table.value_ptr(gi, si));
                    self.table.erase_slot(h, gi, si);
                }
            }
        }
    }

    pub fn reserve(&mut self, additional: usize) {
        self.table.reserve(additional, &self.hash_builder);
    }

    pub fn shrink_to_fit(&mut self) {
        self.table.shrink_to_fit(&self.hash_builder);
    }

    pub fn drain(&mut self) -> R::IntoIter {
        self.table.drain_impl()
    }

    pub fn try_insert(
        &mut self,
        key: K,
        value: V,
    ) -> Result<(), crate::traits::OccupiedError<K, V>> {
        match self.entry(key) {
            Entry::Occupied(e) => Err(crate::traits::OccupiedError {
                key: e.key,
                value,
            }),
            Entry::Vacant(e) => {
                e.insert(value);
                Ok(())
            }
        }
    }

    pub fn into_keys(self) -> impl Iterator<Item = K> {
        self.into_iter().map(|(k, _)| k)
    }

    pub fn into_values(self) -> impl Iterator<Item = V> {
        self.into_iter().map(|(_, v)| v)
    }
}

// ── Entry API ──────────────────────────────────────────────────────────────

pub enum Entry<'a, K, V, S, R: RawTableApi<K, V>> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V, S, R>),
}

pub struct OccupiedEntry<'a, K, V: 'a> {
    pub(crate) key: K,
    pub(crate) value: &'a mut V,
}

pub struct VacantEntry<'a, K, V, S, R: RawTableApi<K, V>> {
    pub(crate) key: K,
    pub(crate) hash: u64,
    pub(crate) slot: Option<(usize, usize, u8)>,
    pub(crate) table: &'a mut R,
    pub(crate) hash_builder: &'a S,
    _marker: std::marker::PhantomData<V>,
}

impl<'a, K: Hash + Eq, V, S: BuildHasher, R: RawTableApi<K, V>> Entry<'a, K, V, S, R> {
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

    pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.value,
            Entry::Vacant(e) => {
                let value = default(&e.key);
                e.insert(value)
            }
        }
    }

    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(e) => &e.key,
            Entry::Vacant(e) => &e.key,
        }
    }

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

impl<'a, K: Hash + Eq, V, S: BuildHasher, R: RawTableApi<K, V>> VacantEntry<'a, K, V, S, R> {
    pub fn insert(self, value: V) -> &'a mut V {
        if let Some((gi, si, full_mask)) = self.slot {
            self.table
                .insert_at(self.hash, gi, si, self.key, value, full_mask);
            unsafe { &mut *self.table.value_ptr(gi, si) }
        } else {
            self.table.ensure_capacity(self.hash_builder);
            let (gi, si) = self.table.insert_no_check(self.hash, self.key, value);
            unsafe { &mut *self.table.value_ptr(gi, si) }
        }
    }

    pub fn key(&self) -> &K {
        &self.key
    }

    pub fn into_key(self) -> K {
        self.key
    }
}

// ── Iterators ──────────────────────────────────────────────────────────────

pub struct Iter<'a, K: 'a, V: 'a, R: RawTableApi<K, V> + 'a> {
    inner: R::SlotIter<'a>,
    table: &'a R,
}

impl<'a, K: 'a, V: 'a, R: RawTableApi<K, V> + 'a> Iterator for Iter<'a, K, V, R> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        let (gi, si) = self.inner.next()?;
        unsafe { Some((&*self.table.key_ptr(gi, si), &*self.table.value_ptr(gi, si))) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K: 'a, V: 'a, R: RawTableApi<K, V> + 'a> FusedIterator for Iter<'a, K, V, R> {}

pub struct IterMut<'a, K: 'a, V: 'a, R: RawTableApi<K, V> + 'a> {
    inner: R::SlotIter<'a>,
    table: &'a R,
}

impl<'a, K: 'a, V: 'a, R: RawTableApi<K, V> + 'a> Iterator for IterMut<'a, K, V, R> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        let (gi, si) = self.inner.next()?;
        unsafe { Some((&*self.table.key_ptr(gi, si), &mut *self.table.value_ptr(gi, si))) }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K: 'a, V: 'a, R: RawTableApi<K, V> + 'a> FusedIterator for IterMut<'a, K, V, R> {}

pub struct Keys<'a, K: 'a, V: 'a, R: RawTableApi<K, V> + 'a> {
    inner: Iter<'a, K, V, R>,
}

impl<'a, K: 'a, V: 'a, R: RawTableApi<K, V> + 'a> Iterator for Keys<'a, K, V, R> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(k, _)| k)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K: 'a, V: 'a, R: RawTableApi<K, V> + 'a> FusedIterator for Keys<'a, K, V, R> {}

pub struct Values<'a, K: 'a, V: 'a, R: RawTableApi<K, V> + 'a> {
    inner: Iter<'a, K, V, R>,
}

impl<'a, K: 'a, V: 'a, R: RawTableApi<K, V> + 'a> Iterator for Values<'a, K, V, R> {
    type Item = &'a V;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K: 'a, V: 'a, R: RawTableApi<K, V> + 'a> FusedIterator for Values<'a, K, V, R> {}

pub struct ValuesMut<'a, K: 'a, V: 'a, R: RawTableApi<K, V> + 'a> {
    inner: IterMut<'a, K, V, R>,
}

impl<'a, K: 'a, V: 'a, R: RawTableApi<K, V> + 'a> Iterator for ValuesMut<'a, K, V, R> {
    type Item = &'a mut V;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, K: 'a, V: 'a, R: RawTableApi<K, V> + 'a> FusedIterator for ValuesMut<'a, K, V, R> {}

// ── Trait implementations ──────────────────────────────────────────────────

impl<K, V, R: RawTableApi<K, V>> Default for GenericMap<K, V, DefaultHashBuilder, R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, S, R: RawTableApi<K, V>> IntoIterator for GenericMap<K, V, S, R> {
    type Item = (K, V);
    type IntoIter = R::IntoIter;

    fn into_iter(self) -> R::IntoIter {
        self.table.into_iter_impl()
    }
}

impl<'a, K, V, S, R: RawTableApi<K, V>> IntoIterator for &'a GenericMap<K, V, S, R>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V, R>;

    fn into_iter(self) -> Iter<'a, K, V, R> {
        self.iter()
    }
}

impl<K, V, S, R: RawTableApi<K, V>> FromIterator<(K, V)> for GenericMap<K, V, S, R>
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

impl<K, V, S, R: RawTableApi<K, V>> Extend<(K, V)> for GenericMap<K, V, S, R>
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

impl<K, V, S, R: RawTableApi<K, V>, Q> Index<&Q> for GenericMap<K, V, S, R>
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

impl<K, V, S, R: RawTableApi<K, V>> fmt::Debug for GenericMap<K, V, S, R>
where
    K: Hash + Eq + fmt::Debug,
    V: fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<K, V, S, R: RawTableApi<K, V>> Clone for GenericMap<K, V, S, R>
where
    K: Clone,
    V: Clone,
    S: Clone,
{
    fn clone(&self) -> Self {
        GenericMap {
            table: self.table.clone_table(),
            hash_builder: self.hash_builder.clone(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<K, V, S, R: RawTableApi<K, V>> PartialEq for GenericMap<K, V, S, R>
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

impl<K, V, S, R: RawTableApi<K, V>> Eq for GenericMap<K, V, S, R>
where
    K: Hash + Eq,
    V: Eq,
    S: BuildHasher,
{
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::group_layout::{GapsLayout, SplitsiesLayout, UfmLayout};
    use crate::raw::overflow_table::RawTable;

    type UfmMap<K, V> = GenericMap<K, V, DefaultHashBuilder, RawTable<K, V, UfmLayout>>;
    type SplitsiesMap<K, V> = GenericMap<K, V, DefaultHashBuilder, RawTable<K, V, SplitsiesLayout>>;
    type GapsMap<K, V> = GenericMap<K, V, DefaultHashBuilder, RawTable<K, V, GapsLayout>>;

    fn test_basic<R: RawTableApi<u64, &'static str>>()
    where
        GenericMap<u64, &'static str, DefaultHashBuilder, R>:,
    {
        let mut map = GenericMap::<u64, &str, DefaultHashBuilder, R>::new();
        assert!(map.is_empty());

        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");

        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&1), Some(&"one"));
        assert_eq!(map.get(&4), None);
    }

    fn test_entry<R: RawTableApi<u64, u64>>() {
        let mut map = GenericMap::<u64, u64, DefaultHashBuilder, R>::new();

        map.entry(1).or_insert(10);
        assert_eq!(map.get(&1), Some(&10));

        map.entry(1).or_insert(20);
        assert_eq!(map.get(&1), Some(&10));

        *map.entry(2).or_insert(0) += 5;
        assert_eq!(map.get(&2), Some(&5));
    }

    fn test_large<R: RawTableApi<u64, u64>>() {
        let mut map = GenericMap::<u64, u64, DefaultHashBuilder, R>::new();
        for i in 0..10_000 {
            map.insert(i, i * 2);
        }
        assert_eq!(map.len(), 10_000);
        for i in 0..10_000 {
            assert_eq!(map.get(&i), Some(&(i * 2)));
        }
    }

    fn test_clone_eq<R: RawTableApi<u64, u64>>() {
        let mut map = GenericMap::<u64, u64, DefaultHashBuilder, R>::new();
        map.insert(1, 10);
        map.insert(2, 20);

        let cloned = map.clone();
        assert_eq!(map, cloned);
    }

    fn test_into_iter<R: RawTableApi<i32, i32>>() {
        let mut map = GenericMap::<i32, i32, DefaultHashBuilder, R>::new();
        for i in 0..10 {
            map.insert(i, i * 10);
        }
        let mut pairs: Vec<_> = map.into_iter().collect();
        pairs.sort();
        assert_eq!(pairs.len(), 10);
        assert_eq!(pairs[0], (0, 0));
        assert_eq!(pairs[9], (9, 90));
    }

    fn test_retain<R: RawTableApi<u64, u64>>() {
        let mut map = GenericMap::<u64, u64, DefaultHashBuilder, R>::new();
        for i in 0..20 {
            map.insert(i, i);
        }
        map.retain(|_, v| *v % 2 == 0);
        assert_eq!(map.len(), 10);
        for i in 0..20 {
            if i % 2 == 0 {
                assert_eq!(map.get(&i), Some(&i));
            } else {
                assert_eq!(map.get(&i), None);
            }
        }
    }

    fn test_from_iter<R: RawTableApi<i32, &'static str>>()
    where
        GenericMap<i32, &'static str, DefaultHashBuilder, R>:
            FromIterator<(i32, &'static str)>,
    {
        let map: GenericMap<i32, &str, DefaultHashBuilder, R> =
            vec![(1, "a"), (2, "b"), (3, "c")].into_iter().collect();
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&2), Some(&"b"));
    }

    #[test] fn ufm_basic() { test_basic::<RawTable<u64, &str, UfmLayout>>(); }
    #[test] fn ufm_entry() { test_entry::<RawTable<u64, u64, UfmLayout>>(); }
    #[test] fn ufm_large() { test_large::<RawTable<u64, u64, UfmLayout>>(); }
    #[test] fn ufm_clone_eq() { test_clone_eq::<RawTable<u64, u64, UfmLayout>>(); }
    #[test] fn ufm_into_iter() { test_into_iter::<RawTable<i32, i32, UfmLayout>>(); }
    #[test] fn ufm_retain() { test_retain::<RawTable<u64, u64, UfmLayout>>(); }
    #[test] fn ufm_from_iter() { test_from_iter::<RawTable<i32, &str, UfmLayout>>(); }

    #[test] fn splitsies_basic() { test_basic::<RawTable<u64, &str, SplitsiesLayout>>(); }
    #[test] fn splitsies_entry() { test_entry::<RawTable<u64, u64, SplitsiesLayout>>(); }
    #[test] fn splitsies_large() { test_large::<RawTable<u64, u64, SplitsiesLayout>>(); }
    #[test] fn splitsies_clone_eq() { test_clone_eq::<RawTable<u64, u64, SplitsiesLayout>>(); }
    #[test] fn splitsies_into_iter() { test_into_iter::<RawTable<i32, i32, SplitsiesLayout>>(); }
    #[test] fn splitsies_retain() { test_retain::<RawTable<u64, u64, SplitsiesLayout>>(); }
    #[test] fn splitsies_from_iter() { test_from_iter::<RawTable<i32, &str, SplitsiesLayout>>(); }

    #[test] fn gaps_basic() { test_basic::<RawTable<u64, &str, GapsLayout>>(); }
    #[test] fn gaps_entry() { test_entry::<RawTable<u64, u64, GapsLayout>>(); }
    #[test] fn gaps_large() { test_large::<RawTable<u64, u64, GapsLayout>>(); }
    #[test] fn gaps_clone_eq() { test_clone_eq::<RawTable<u64, u64, GapsLayout>>(); }
    #[test] fn gaps_into_iter() { test_into_iter::<RawTable<i32, i32, GapsLayout>>(); }
    #[test] fn gaps_retain() { test_retain::<RawTable<u64, u64, GapsLayout>>(); }
    #[test] fn gaps_from_iter() { test_from_iter::<RawTable<i32, &str, GapsLayout>>(); }
}
