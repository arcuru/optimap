//! `OptiMap` — a smart wrapper that dynamically selects a hash map backend.
//!
//! Users can let the policy engine choose the best backend based on capacity
//! and key/value sizes, pin a specific backend, or provide workload hints.

use std::borrow::Borrow;
use std::fmt;
use std::hash::Hash;
use std::mem;

use crate::map::DefaultHashBuilder;
use crate::{Gaps, IPO64, InPlaceOverflow, Splitsies, UnorderedFlatMap};

// ── Public types ───────────────────────────────────────────────────────────

/// Which concrete hash map backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapType {
    Ufm,
    Splitsies,
    Ipo,
    Gaps,
    Ipo64,
}

/// Workload hint for backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Hint {
    /// General purpose — the policy picks based on size.
    #[default]
    Auto,
    /// Read-heavy: optimise for lookup hit.
    ReadHeavy,
    /// Write-heavy: optimise for insert throughput.
    WriteHeavy,
    /// High churn: frequent insert + delete of the same keys.
    Churn,
    /// Iteration-heavy: optimise for sequential scan.
    Iteration,
}

/// Backend selection strategy.
#[derive(Debug, Clone, Copy)]
enum Backend {
    /// Policy decides and may transition on resize.
    Auto(Hint),
    /// User chose explicitly — never transitions.
    Pinned,
}

/// A smart hash map that dynamically selects its backend.
///
/// `OptiMap` wraps the five core hash map designs behind an enum and
/// delegates every operation to the active backend. When constructed
/// with [`OptiMap::new`] or [`OptiMap::with_hint`], the backend is
/// chosen by a policy engine and may transition at resize boundaries.
/// When constructed with an explicit backend (e.g. [`OptiMap::splitsies`]),
/// the choice is pinned for the lifetime of the map.
///
/// # Examples
///
/// ```
/// use optimap::OptiMap;
///
/// // Let the policy choose:
/// let mut map = OptiMap::<String, i32>::new();
/// map.insert("hello".into(), 42);
///
/// // Pin a specific backend:
/// let mut map = OptiMap::<u64, u64>::ipo();
/// map.insert(1, 2);
///
/// // Hint at workload:
/// use optimap::optimap::Hint;
/// let mut map = OptiMap::<u64, u64>::with_hint(Hint::Churn);
/// ```
pub struct OptiMap<K, V, S = DefaultHashBuilder> {
    inner: Inner<K, V, S>,
    backend: Backend,
}

// ── Inner enum ─────────────────────────────────────────────────────────────

enum Inner<K, V, S = DefaultHashBuilder> {
    Ufm(UnorderedFlatMap<K, V, S>),
    Splitsies(Splitsies<K, V, S>),
    Ipo(InPlaceOverflow<K, V, S>),
    Gaps(Gaps<K, V, S>),
    Ipo64(IPO64<K, V, S>),
}

// ── Policy engine ──────────────────────────────────────────────────────────

/// Choose a backend for the given conditions.
fn select_backend<K, V>(hint: Hint, capacity: usize) -> MapType {
    let kv_size = mem::size_of::<K>() + mem::size_of::<V>();

    match hint {
        Hint::ReadHeavy => MapType::Ipo,
        Hint::WriteHeavy => MapType::Ipo,
        Hint::Churn => MapType::Splitsies,
        Hint::Iteration => MapType::Gaps,
        Hint::Auto => {
            if kv_size <= 16 && capacity >= 4096 {
                // Small KV at scale — IPO64's 64-slot groups shine
                MapType::Ipo64
            } else if capacity >= 1024 {
                // General large — IPO is the best all-rounder
                MapType::Ipo
            } else {
                // Small/medium — Splitsies: good balance, tombstone-free
                MapType::Splitsies
            }
        }
    }
}

// ── Dispatch macro ─────────────────────────────────────────────────────────

/// Dispatch a method call to whichever inner variant is active.
macro_rules! dispatch {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match &$self.inner {
            Inner::Ufm(m) => m.$method($($arg),*),
            Inner::Splitsies(m) => m.$method($($arg),*),
            Inner::Ipo(m) => m.$method($($arg),*),
            Inner::Gaps(m) => m.$method($($arg),*),
            Inner::Ipo64(m) => m.$method($($arg),*),
        }
    };
}

macro_rules! dispatch_mut {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match &mut $self.inner {
            Inner::Ufm(m) => m.$method($($arg),*),
            Inner::Splitsies(m) => m.$method($($arg),*),
            Inner::Ipo(m) => m.$method($($arg),*),
            Inner::Gaps(m) => m.$method($($arg),*),
            Inner::Ipo64(m) => m.$method($($arg),*),
        }
    };
}

// ── Constructors ───────────────────────────────────────────────────────────

impl<K: Hash + Eq, V> OptiMap<K, V> {
    /// Create an empty map, letting the policy engine choose the backend.
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Create a map with at least the given capacity, backend chosen by policy.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hint(capacity, Hint::Auto)
    }

    /// Create a map with the given workload hint.
    pub fn with_hint(hint: Hint) -> Self {
        Self::with_capacity_and_hint(0, hint)
    }

    /// Create a map with both a capacity and a workload hint.
    pub fn with_capacity_and_hint(capacity: usize, hint: Hint) -> Self {
        let map_type = select_backend::<K, V>(hint, capacity);
        let inner = build_inner(map_type, capacity);
        OptiMap {
            inner,
            backend: Backend::Auto(hint),
        }
    }

    /// Create a map pinned to the `UnorderedFlatMap` backend.
    pub fn ufm() -> Self {
        Self::pinned(MapType::Ufm, 0)
    }

    /// Create a map pinned to the `Splitsies` backend.
    pub fn splitsies() -> Self {
        Self::pinned(MapType::Splitsies, 0)
    }

    /// Create a map pinned to the `InPlaceOverflow` backend.
    pub fn ipo() -> Self {
        Self::pinned(MapType::Ipo, 0)
    }

    /// Create a map pinned to the `Gaps` backend.
    pub fn gaps() -> Self {
        Self::pinned(MapType::Gaps, 0)
    }

    /// Create a map pinned to the `IPO64` backend.
    pub fn ipo64() -> Self {
        Self::pinned(MapType::Ipo64, 0)
    }

    /// Create a map pinned to a specific backend type.
    pub fn with_type(map_type: MapType) -> Self {
        Self::pinned(map_type, 0)
    }

    /// Create a map pinned to a specific backend with the given capacity.
    pub fn with_type_and_capacity(map_type: MapType, capacity: usize) -> Self {
        Self::pinned(map_type, capacity)
    }

    fn pinned(map_type: MapType, capacity: usize) -> Self {
        OptiMap {
            inner: build_inner(map_type, capacity),
            backend: Backend::Pinned,
        }
    }
}

fn build_inner<K: Hash + Eq, V>(
    map_type: MapType,
    capacity: usize,
) -> Inner<K, V> {
    match map_type {
        MapType::Ufm => Inner::Ufm(UnorderedFlatMap::with_capacity(capacity)),
        MapType::Splitsies => Inner::Splitsies(Splitsies::with_capacity(capacity)),
        MapType::Ipo => Inner::Ipo(InPlaceOverflow::with_capacity(capacity)),
        MapType::Gaps => Inner::Gaps(Gaps::with_capacity(capacity)),
        MapType::Ipo64 => Inner::Ipo64(IPO64::with_capacity(capacity)),
    }
}

// ── Core map operations ────────────────────────────────────────────────────

impl<K: Hash + Eq, V> OptiMap<K, V> {
    /// Insert a key-value pair. Returns the previous value if the key existed.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        dispatch_mut!(self, insert, key, value)
    }

    /// Look up a value by key.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        dispatch!(self, get, key)
    }

    /// Returns the key-value pair corresponding to the key.
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        dispatch!(self, get_key_value, key)
    }

    /// Look up a value by key, returning a mutable reference.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        dispatch_mut!(self, get_mut, key)
    }

    /// Remove a key, returning its value if present.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        dispatch_mut!(self, remove, key)
    }

    /// Removes a key, returning the key and value if present.
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        dispatch_mut!(self, remove_entry, key)
    }

    /// Whether the map contains the given key.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        dispatch!(self, contains_key, key)
    }

    /// Number of elements in the map.
    pub fn len(&self) -> usize {
        dispatch!(self, len)
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of elements the map can hold without rehashing.
    pub fn capacity(&self) -> usize {
        dispatch!(self, capacity)
    }

    /// Remove all elements, keeping allocated memory.
    pub fn clear(&mut self) {
        dispatch_mut!(self, clear)
    }

    /// Which backend is currently active.
    pub fn map_type(&self) -> MapType {
        match &self.inner {
            Inner::Ufm(_) => MapType::Ufm,
            Inner::Splitsies(_) => MapType::Splitsies,
            Inner::Ipo(_) => MapType::Ipo,
            Inner::Gaps(_) => MapType::Gaps,
            Inner::Ipo64(_) => MapType::Ipo64,
        }
    }

    /// Reserves capacity for at least `additional` more elements.
    ///
    /// For `Auto` backends, this may transition to a different backend
    /// if the new capacity crosses a policy threshold.
    pub fn reserve(&mut self, additional: usize) {
        let new_cap = self.len() + additional;
        if let Backend::Auto(hint) = self.backend {
            let desired = select_backend::<K, V>(hint, new_cap);
            if desired != self.map_type() {
                self.transition_to(desired, new_cap);
                return;
            }
        }
        dispatch_mut!(self, reserve, additional)
    }

    /// Shrinks the capacity as much as possible.
    pub fn shrink_to_fit(&mut self) {
        dispatch_mut!(self, shrink_to_fit)
    }

    /// Iterate over key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        match &self.inner {
            Inner::Ufm(m) => Box::new(m.iter()) as Box<dyn Iterator<Item = _>>,
            Inner::Splitsies(m) => Box::new(m.iter()),
            Inner::Ipo(m) => Box::new(m.iter()),
            Inner::Gaps(m) => Box::new(m.iter()),
            Inner::Ipo64(m) => Box::new(m.iter()),
        }
    }

    /// Iterate over key-value pairs with mutable values.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        match &mut self.inner {
            Inner::Ufm(m) => Box::new(m.iter_mut()) as Box<dyn Iterator<Item = _>>,
            Inner::Splitsies(m) => Box::new(m.iter_mut()),
            Inner::Ipo(m) => Box::new(m.iter_mut()),
            Inner::Gaps(m) => Box::new(m.iter_mut()),
            Inner::Ipo64(m) => Box::new(m.iter_mut()),
        }
    }

    /// Iterate over keys.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|(k, _)| k)
    }

    /// Iterate over values.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }

    /// Iterate over mutable values.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.iter_mut().map(|(_, v)| v)
    }

    /// Retains only the elements specified by the predicate.
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        dispatch_mut!(self, retain, f)
    }

    /// Clears the map, returning all key-value pairs as an iterator.
    pub fn drain(&mut self) -> impl Iterator<Item = (K, V)> {
        let items: Vec<(K, V)> = match &mut self.inner {
            Inner::Ufm(m) => m.drain().collect(),
            Inner::Splitsies(m) => m.drain().collect(),
            Inner::Ipo(m) => m.drain().collect(),
            Inner::Gaps(m) => m.drain().collect(),
            Inner::Ipo64(m) => m.drain().collect(),
        };
        items.into_iter()
    }

    // ── Private helpers ────────────────────────────────────────────────────

    /// Drain all entries from the current backend and re-insert into a new one.
    fn transition_to(&mut self, map_type: MapType, capacity: usize) {
        let old = mem::replace(&mut self.inner, build_inner::<K, V>(map_type, capacity));
        let entries: Vec<(K, V)> = match old {
            Inner::Ufm(mut m) => m.drain().collect(),
            Inner::Splitsies(mut m) => m.drain().collect(),
            Inner::Ipo(mut m) => m.drain().collect(),
            Inner::Gaps(mut m) => m.drain().collect(),
            Inner::Ipo64(mut m) => m.drain().collect(),
        };
        for (k, v) in entries {
            dispatch_mut!(self, insert, k, v);
        }
    }
}

// ── Default, Debug, Clone ──────────────────────────────────────────────────

impl<K: Hash + Eq, V> Default for OptiMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + fmt::Debug, V: fmt::Debug> fmt::Debug for OptiMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();
        match &self.inner {
            Inner::Ufm(m) => { for (k, v) in m.iter() { map.entry(k, v); } }
            Inner::Splitsies(m) => { for (k, v) in m.iter() { map.entry(k, v); } }
            Inner::Ipo(m) => { for (k, v) in m.iter() { map.entry(k, v); } }
            Inner::Gaps(m) => { for (k, v) in m.iter() { map.entry(k, v); } }
            Inner::Ipo64(m) => { for (k, v) in m.iter() { map.entry(k, v); } }
        }
        map.finish()
    }
}

impl<K: Hash + Eq + Clone, V: Clone> Clone for OptiMap<K, V> {
    fn clone(&self) -> Self {
        OptiMap {
            inner: match &self.inner {
                Inner::Ufm(m) => Inner::Ufm(m.clone()),
                Inner::Splitsies(m) => Inner::Splitsies(m.clone()),
                Inner::Ipo(m) => Inner::Ipo(m.clone()),
                Inner::Gaps(m) => Inner::Gaps(m.clone()),
                Inner::Ipo64(m) => Inner::Ipo64(m.clone()),
            },
            backend: self.backend,
        }
    }
}

impl<K: Hash + Eq, V: PartialEq> PartialEq for OptiMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        // Compare by iterating self and looking up each key in other
        let mut eq = true;
        macro_rules! check_eq {
            ($m:expr) => {
                for (k, v) in $m.iter() {
                    match other.get(k) {
                        Some(v2) if v == v2 => {}
                        _ => { eq = false; break; }
                    }
                }
            };
        }
        match &self.inner {
            Inner::Ufm(m) => check_eq!(m),
            Inner::Splitsies(m) => check_eq!(m),
            Inner::Ipo(m) => check_eq!(m),
            Inner::Gaps(m) => check_eq!(m),
            Inner::Ipo64(m) => check_eq!(m),
        }
        eq
    }
}

impl<K: Hash + Eq, V: Eq> Eq for OptiMap<K, V> {}

impl<K: Hash + Eq, V> FromIterator<(K, V)> for OptiMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut map = Self::with_capacity(lower);
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

impl<K: Hash + Eq, V> Extend<(K, V)> for OptiMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

// ── Map trait impl ─────────────────────────────────────────────────────────

impl<K: Hash + Eq, V> crate::Map<K, V> for OptiMap<K, V> {
    fn new() -> Self {
        OptiMap::new()
    }

    fn with_capacity(capacity: usize) -> Self {
        OptiMap::with_capacity(capacity)
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        OptiMap::insert(self, key, value)
    }

    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        OptiMap::get(self, key)
    }

    fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        OptiMap::get_key_value(self, key)
    }

    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        OptiMap::get_mut(self, key)
    }

    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        OptiMap::remove(self, key)
    }

    fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        OptiMap::remove_entry(self, key)
    }

    fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        OptiMap::contains_key(self, key)
    }

    fn len(&self) -> usize {
        OptiMap::len(self)
    }

    fn capacity(&self) -> usize {
        OptiMap::capacity(self)
    }

    fn clear(&mut self) {
        OptiMap::clear(self)
    }

    fn reserve(&mut self, additional: usize) {
        OptiMap::reserve(self, additional)
    }

    fn shrink_to_fit(&mut self) {
        OptiMap::shrink_to_fit(self)
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: 'a,
        V: 'a,
    {
        OptiMap::iter(self)
    }

    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (&'a K, &'a mut V)>
    where
        K: 'a,
        V: 'a,
    {
        OptiMap::iter_mut(self)
    }

    fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        OptiMap::retain(self, f)
    }

    fn drain(&mut self) -> impl Iterator<Item = (K, V)> {
        OptiMap::drain(self)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_auto() {
        let mut map = OptiMap::new();
        map.insert("hello", 42);
        map.insert("world", 99);
        assert_eq!(map.get("hello"), Some(&42));
        assert_eq!(map.get("world"), Some(&99));
        assert_eq!(map.len(), 2);
        assert!(!map.is_empty());
    }

    #[test]
    fn pinned_backends() {
        for mt in [MapType::Ufm, MapType::Splitsies, MapType::Ipo, MapType::Gaps, MapType::Ipo64] {
            let mut map = OptiMap::<u64, u64>::with_type(mt);
            for i in 0..100 {
                map.insert(i, i * 2);
            }
            assert_eq!(map.len(), 100);
            assert_eq!(map.map_type(), mt);
            assert_eq!(map.get(&50), Some(&100));
        }
    }

    #[test]
    fn named_constructors() {
        assert_eq!(OptiMap::<u64, u64>::ufm().map_type(), MapType::Ufm);
        assert_eq!(OptiMap::<u64, u64>::splitsies().map_type(), MapType::Splitsies);
        assert_eq!(OptiMap::<u64, u64>::ipo().map_type(), MapType::Ipo);
        assert_eq!(OptiMap::<u64, u64>::gaps().map_type(), MapType::Gaps);
        assert_eq!(OptiMap::<u64, u64>::ipo64().map_type(), MapType::Ipo64);
    }

    #[test]
    fn hint_constructors() {
        let m = OptiMap::<u64, u64>::with_hint(Hint::ReadHeavy);
        assert_eq!(m.map_type(), MapType::Ipo);

        let m = OptiMap::<u64, u64>::with_hint(Hint::Churn);
        assert_eq!(m.map_type(), MapType::Splitsies);

        let m = OptiMap::<u64, u64>::with_hint(Hint::Iteration);
        assert_eq!(m.map_type(), MapType::Gaps);
    }

    #[test]
    fn remove_and_contains() {
        let mut map = OptiMap::new();
        map.insert(1u64, 10u64);
        map.insert(2, 20);
        assert!(map.contains_key(&1));
        assert_eq!(map.remove(&1), Some(10));
        assert!(!map.contains_key(&1));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn clear_and_capacity() {
        let mut map = OptiMap::<u64, u64>::with_capacity(100);
        assert!(map.capacity() >= 100);
        for i in 0..50 {
            map.insert(i, i);
        }
        map.clear();
        assert!(map.is_empty());
        assert!(map.capacity() >= 100);
    }

    #[test]
    fn iter() {
        let mut map = OptiMap::new();
        for i in 0u64..100 {
            map.insert(i, i * 3);
        }
        let mut pairs: Vec<(u64, u64)> = map.iter().map(|(&k, &v)| (k, v)).collect();
        pairs.sort();
        assert_eq!(pairs.len(), 100);
        assert_eq!(pairs[0], (0, 0));
        assert_eq!(pairs[99], (99, 297));
    }

    #[test]
    fn iter_mut() {
        let mut map = OptiMap::new();
        for i in 0u64..10 {
            map.insert(i, i);
        }
        for (_, v) in map.iter_mut() {
            *v *= 10;
        }
        assert_eq!(map.get(&5), Some(&50));
    }

    #[test]
    fn retain() {
        let mut map: OptiMap<u64, u64> = (0..20).map(|i| (i, i)).collect();
        map.retain(|&k, _| k % 2 == 0);
        assert_eq!(map.len(), 10);
        assert!(map.contains_key(&0));
        assert!(!map.contains_key(&1));
    }

    #[test]
    fn drain() {
        let mut map: OptiMap<u64, u64> = (0..50).map(|i| (i, i)).collect();
        let mut drained: Vec<(u64, u64)> = map.drain().collect();
        drained.sort();
        assert_eq!(drained.len(), 50);
        assert!(map.is_empty());
    }

    #[test]
    fn from_iter_and_extend() {
        let mut map: OptiMap<u64, u64> = vec![(1, 10), (2, 20)].into_iter().collect();
        assert_eq!(map.len(), 2);
        map.extend(vec![(3, 30), (4, 40)]);
        assert_eq!(map.len(), 4);
        assert_eq!(map.get(&3), Some(&30));
    }

    #[test]
    fn clone_and_eq() {
        let map: OptiMap<u64, u64> = (0..100).map(|i| (i, i)).collect();
        let map2 = map.clone();
        assert_eq!(map, map2);
    }

    #[test]
    fn auto_transition_on_reserve() {
        // Start small — policy picks Splitsies for small capacity
        let mut map = OptiMap::<u8, u8>::new();
        assert_eq!(map.map_type(), MapType::Splitsies);
        for i in 0..10u8 {
            map.insert(i, i);
        }
        // Reserve a lot — policy may switch to IPO64 (small KV, large capacity)
        map.reserve(10_000);
        // Verify data survived the transition
        for i in 0..10u8 {
            assert_eq!(map.get(&i), Some(&i), "lost key {i} after transition");
        }
    }

    #[test]
    fn pinned_no_transition() {
        let mut map = OptiMap::<u8, u8>::splitsies();
        for i in 0..10u8 {
            map.insert(i, i);
        }
        map.reserve(10_000);
        // Pinned — must stay Splitsies
        assert_eq!(map.map_type(), MapType::Splitsies);
        for i in 0..10u8 {
            assert_eq!(map.get(&i), Some(&i));
        }
    }

    #[test]
    fn string_keys() {
        let mut map = OptiMap::new();
        map.insert("hello".to_string(), 1);
        map.insert("world".to_string(), 2);
        assert_eq!(map.get("hello"), Some(&1));
        assert_eq!(map.get("world"), Some(&2));
        assert!(!map.contains_key("foo"));
    }

    #[test]
    fn large_scale() {
        let mut map = OptiMap::new();
        for i in 0u64..5000 {
            map.insert(i, i * 7);
        }
        assert_eq!(map.len(), 5000);
        for i in 0..5000u64 {
            assert_eq!(map.get(&i), Some(&(i * 7)));
        }
        for i in 0..2500u64 {
            assert!(map.remove(&i).is_some());
        }
        assert_eq!(map.len(), 2500);
    }

    #[test]
    fn map_trait_usage() {
        use crate::Map;

        fn fill<M: Map<u64, u64>>(m: &mut M, n: u64) {
            for i in 0..n {
                m.insert(i, i);
            }
        }

        let mut map = OptiMap::new();
        fill(&mut map, 100);
        assert_eq!(map.len(), 100);
    }

    #[test]
    fn debug_display() {
        let mut map = OptiMap::new();
        map.insert(1, 2);
        let s = format!("{:?}", map);
        assert!(s.contains("1"));
        assert!(s.contains("2"));
    }
}
