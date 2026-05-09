//! `OptiMap` — a smart wrapper that dynamically selects a hash map backend.
//!
//! Users can let the policy engine choose the best backend based on capacity
//! and key/value sizes, pin a specific backend, or provide workload hints.

use std::borrow::Borrow;
use std::fmt;
use std::hash::Hash;
use std::iter::FusedIterator;
use std::mem;

use crate::map::DefaultHashBuilder;
use crate::{FlatBTree, Gaps, IPO64, InPlaceOverflow, Splitsies, UnorderedFlatMap};

// ── Enum iterators (zero-cost dispatch, no Box<dyn>) ──────────────────────

/// Iterator over `(&K, &V)` pairs in an [`OptiMap`].
pub enum Iter<'a, K, V> {
    Ufm(crate::map::Iter<'a, K, V>),
    Splitsies(crate::split_overflow::map::Iter<'a, K, V>),
    Ipo(crate::in_place_overflow::map::Iter<'a, K, V>),
    Gaps(crate::gaps::map::Iter<'a, K, V>),
    Ipo64(crate::ipo64::map::Iter<'a, K, V>),
    FlatBTree(crate::flat_btree::map::Iter<'a, K, V>),
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Iter::Ufm(i) => i.next(),
            Iter::Splitsies(i) => i.next(),
            Iter::Ipo(i) => i.next(),
            Iter::Gaps(i) => i.next(),
            Iter::Ipo64(i) => i.next(),
            Iter::FlatBTree(i) => i.next(),
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Iter::Ufm(i) => i.size_hint(),
            Iter::Splitsies(i) => i.size_hint(),
            Iter::Ipo(i) => i.size_hint(),
            Iter::Gaps(i) => i.size_hint(),
            Iter::Ipo64(i) => i.size_hint(),
            Iter::FlatBTree(i) => i.size_hint(),
        }
    }
}
impl<K, V> FusedIterator for Iter<'_, K, V> {}

/// Mutable iterator over `(&K, &mut V)` pairs in an [`OptiMap`].
pub enum IterMut<'a, K, V> {
    Ufm(crate::map::IterMut<'a, K, V>),
    Splitsies(crate::split_overflow::map::IterMut<'a, K, V>),
    Ipo(crate::in_place_overflow::map::IterMut<'a, K, V>),
    Gaps(crate::gaps::map::IterMut<'a, K, V>),
    Ipo64(crate::ipo64::map::IterMut<'a, K, V>),
    FlatBTree(crate::flat_btree::map::IterMut<'a, K, V>),
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IterMut::Ufm(i) => i.next(),
            IterMut::Splitsies(i) => i.next(),
            IterMut::Ipo(i) => i.next(),
            IterMut::Gaps(i) => i.next(),
            IterMut::Ipo64(i) => i.next(),
            IterMut::FlatBTree(i) => i.next(),
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            IterMut::Ufm(i) => i.size_hint(),
            IterMut::Splitsies(i) => i.size_hint(),
            IterMut::Ipo(i) => i.size_hint(),
            IterMut::Gaps(i) => i.size_hint(),
            IterMut::Ipo64(i) => i.size_hint(),
            IterMut::FlatBTree(i) => i.size_hint(),
        }
    }
}
impl<K, V> FusedIterator for IterMut<'_, K, V> {}

/// Owning iterator over `(K, V)` pairs from an [`OptiMap`].
pub enum IntoIter<K, V> {
    Ufm(crate::map::IntoIter<K, V>),
    Splitsies(crate::split_overflow::map::IntoIter<K, V>),
    Ipo(crate::in_place_overflow::map::IntoIter<K, V>),
    Gaps(crate::gaps::map::IntoIter<K, V>),
    Ipo64(crate::ipo64::map::IntoIter<K, V>),
    FlatBTree(crate::flat_btree::map::IntoIter<K, V>),
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IntoIter::Ufm(i) => i.next(),
            IntoIter::Splitsies(i) => i.next(),
            IntoIter::Ipo(i) => i.next(),
            IntoIter::Gaps(i) => i.next(),
            IntoIter::Ipo64(i) => i.next(),
            IntoIter::FlatBTree(i) => i.next(),
        }
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            IntoIter::Ufm(i) => i.size_hint(),
            IntoIter::Splitsies(i) => i.size_hint(),
            IntoIter::Ipo(i) => i.size_hint(),
            IntoIter::Gaps(i) => i.size_hint(),
            IntoIter::Ipo64(i) => i.size_hint(),
            IntoIter::FlatBTree(i) => i.size_hint(),
        }
    }
}
impl<K, V> FusedIterator for IntoIter<K, V> {}

// ── Enum entry types (zero-cost dispatch, no Box<dyn>) ──────────────────────

type S = DefaultHashBuilder;

/// Dispatches a method on a 6-variant enum, returning the result directly.
macro_rules! entry_match {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            Self::Ufm(e) => e.$method($($arg),*),
            Self::Splitsies(e) => e.$method($($arg),*),
            Self::Ipo(e) => e.$method($($arg),*),
            Self::Gaps(e) => e.$method($($arg),*),
            Self::Ipo64(e) => e.$method($($arg),*),
            Self::FlatBTree(e) => e.$method($($arg),*),
        }
    };
}

/// A view into a single entry in an [`OptiMap`], which may either be vacant
/// or occupied.
pub enum Entry<'a, K, V> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

/// A view into an occupied entry in an [`OptiMap`].
pub enum OccupiedEntry<'a, K, V> {
    Ufm(crate::map::OccupiedEntry<'a, K, V>),
    Splitsies(crate::split_overflow::map::OccupiedEntry<'a, K, V>),
    Ipo(crate::in_place_overflow::map::OccupiedEntry<'a, K, V>),
    Gaps(crate::gaps::map::OccupiedEntry<'a, K, V>),
    Ipo64(crate::ipo64::map::OccupiedEntry<'a, K, V>),
    FlatBTree(crate::flat_btree::map::OccupiedEntry<'a, K, V>),
}

/// A view into a vacant entry in an [`OptiMap`].
pub enum VacantEntry<'a, K, V> {
    Ufm(crate::map::VacantEntry<'a, K, V, S>),
    Splitsies(crate::split_overflow::map::VacantEntry<'a, K, V, S>),
    Ipo(crate::in_place_overflow::map::VacantEntry<'a, K, V, S>),
    Gaps(crate::gaps::map::VacantEntry<'a, K, V, S>),
    Ipo64(crate::ipo64::map::VacantEntry<'a, K, V, S>),
    FlatBTree(crate::flat_btree::map::VacantEntry<'a, K, V>),
}

impl<'a, K: Hash + Eq + Ord + Clone, V> Entry<'a, K, V> {
    /// Ensures a value is in the entry by inserting the default if empty.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => e.insert(default),
        }
    }

    /// Ensures a value is in the entry by inserting the result of the
    /// function if empty.
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
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

    /// Ensures a value is in the entry by inserting the result of the
    /// function (which receives the key) if empty.
    pub fn or_insert_with_key<F: FnOnce(&K) -> V>(self, default: F) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let value = default(e.key());
                e.insert(value)
            }
        }
    }

    /// Returns a reference to this entry's key.
    pub fn key(&self) -> &K {
        match self {
            Entry::Occupied(e) => e.key(),
            Entry::Vacant(e) => e.key(),
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
    /// Gets a reference to the key.
    pub fn key(&self) -> &K {
        entry_match!(self, key)
    }

    /// Gets a reference to the value.
    pub fn get(&self) -> &V {
        entry_match!(self, get)
    }

    /// Gets a mutable reference to the value.
    pub fn get_mut(&mut self) -> &mut V {
        entry_match!(self, get_mut)
    }

    /// Sets the value and returns the old value.
    pub fn insert(&mut self, value: V) -> V {
        entry_match!(self, insert, value)
    }

    /// Converts to a mutable reference to the value.
    pub fn into_mut(self) -> &'a mut V {
        entry_match!(self, into_mut)
    }
}

impl<'a, K: Hash + Eq + Ord + Clone, V> VacantEntry<'a, K, V> {
    /// Insert a value and return a mutable reference.
    pub fn insert(self, value: V) -> &'a mut V {
        entry_match!(self, insert, value)
    }

    /// Gets a reference to the key.
    pub fn key(&self) -> &K {
        entry_match!(self, key)
    }

    /// Takes ownership of the key.
    pub fn into_key(self) -> K {
        entry_match!(self, into_key)
    }
}

// ── Public types ───────────────────────────────────────────────────────────

/// Which concrete map backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapType {
    Ufm,
    Splitsies,
    Ipo,
    Gaps,
    Ipo64,
    /// Sorted backend ([`FlatBTree`]). Selected by `Hint::Sorted` or pinned
    /// explicitly. Requires `K: Ord + Clone`.
    FlatBTree,
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
    /// Sorted iteration / range queries — picks [`FlatBTree`].
    /// Requires `K: Ord + Clone`.
    Sorted,
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
/// use optimap::Hint;
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
    FlatBTree(FlatBTree<K, V, S>),
}

// ── Policy engine ──────────────────────────────────────────────────────────

/// Capacity threshold above which tombstone designs (IPO, IPO64) start
/// suffering from the "tombstone-at-DRAM" cliff: lookup_miss probe chains
/// elongate as deletions accumulate and the table no longer fits in cache.
///
/// Sweep data (sweep-2026-04-20) shows IPO miss latency spiking from ~6 ns
/// at 100K entries to ~104 ns at 100M (vs ~5 ns for overflow-bit designs at
/// the same size). Above this threshold, default to a tombstone-free design.
const TOMBSTONE_DRAM_CLIFF: usize = 1_000_000;

/// Capacity threshold below which dispatch overhead and rehash sawtooth
/// dominate; pick the lowest-overhead design (Splitsies — tombstone-free,
/// no probe-length penalty).
const SMALL_CAPACITY: usize = 1024;

/// Choose a backend for the given conditions.
///
/// Selection rationale (from sweep-2026-04-20.csv at 100K-100M entries):
///
/// | Op          | Best at small/med  | Best at ≥1M                  |
/// |-------------|--------------------|------------------------------|
/// | lookup_hit  | IPO/Hi128_Tomb     | hashbrown / UFM (overflow)   |
/// | lookup_miss | Splitsies / SoaMap | Splitsies / Gaps             |
/// | insert      | hashbrown / IPO    | UFM / Lo8_1bit (overflow)    |
/// | remove      | Hi128_Tomb / IPO   | Hi128_Tomb (still wins)      |
///
/// Tombstone designs (IPO, IPO64) win remove + hit at small/medium but hit
/// a 5-13× cliff on lookup_miss at ≥1M entries due to tombstone accumulation.
/// Auto policy avoids that cliff by switching to Splitsies (tombstone-free,
/// balanced, no DRAM-scale regression).
fn select_backend<K, V>(hint: Hint, capacity: usize) -> MapType {
    let _ = mem::size_of::<K>() + mem::size_of::<V>();

    match hint {
        Hint::ReadHeavy => {
            // Hit-heavy: IPO wins up to ~1M, then UFM/overflow wins at DRAM scale.
            // Splitsies is the closest drop-in tombstone-free design.
            if capacity >= TOMBSTONE_DRAM_CLIFF {
                MapType::Splitsies
            } else {
                MapType::Ipo
            }
        }
        Hint::WriteHeavy => {
            // Inserts: IPO is competitive at cache-resident sizes; tombstone
            // accumulation eats it at large N. Splitsies stays flat.
            if capacity >= TOMBSTONE_DRAM_CLIFF {
                MapType::Splitsies
            } else {
                MapType::Ipo
            }
        }
        Hint::Churn => MapType::Splitsies, // tombstone-free, flat at high load
        Hint::Iteration => MapType::Gaps,
        Hint::Sorted => MapType::FlatBTree, // sorted iteration / range queries
        Hint::Auto => {
            if capacity >= TOMBSTONE_DRAM_CLIFF {
                // Avoid the tombstone-at-DRAM cliff (5-13× miss regression).
                MapType::Splitsies
            } else if capacity >= SMALL_CAPACITY {
                // Cache-resident: IPO wins hit + remove decisively.
                MapType::Ipo
            } else {
                // Small: rehash sawtooth dominates; pick the lowest-overhead
                // tombstone-free design.
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
            Inner::FlatBTree(m) => m.$method($($arg),*),
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
            Inner::FlatBTree(m) => m.$method($($arg),*),
        }
    };
}

// ── Constructors ───────────────────────────────────────────────────────────

impl<K: Hash + Eq + Ord + Clone, V> OptiMap<K, V> {
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

    /// Create a map pinned to the `FlatBTree` backend (sorted iteration,
    /// range queries). Requires `K: Ord + Clone`.
    pub fn flat_btree() -> Self {
        Self::pinned(MapType::FlatBTree, 0)
    }

    /// Create a map pinned to `FlatBTree` with the given capacity.
    /// Equivalent to `with_capacity_and_hint(capacity, Hint::Sorted)` but
    /// pinned (no auto-transition on resize).
    pub fn sorted_with_capacity(capacity: usize) -> Self {
        Self::pinned(MapType::FlatBTree, capacity)
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

fn build_inner<K: Hash + Eq + Ord + Clone, V>(map_type: MapType, capacity: usize) -> Inner<K, V> {
    match map_type {
        MapType::Ufm => Inner::Ufm(UnorderedFlatMap::with_capacity(capacity)),
        MapType::Splitsies => Inner::Splitsies(Splitsies::with_capacity(capacity)),
        MapType::Ipo => Inner::Ipo(InPlaceOverflow::with_capacity(capacity)),
        MapType::Gaps => Inner::Gaps(Gaps::with_capacity(capacity)),
        MapType::Ipo64 => Inner::Ipo64(IPO64::with_capacity(capacity)),
        MapType::FlatBTree => Inner::FlatBTree(FlatBTree::with_capacity(capacity)),
    }
}

// ── Core map operations ────────────────────────────────────────────────────

impl<K: Hash + Eq + Ord + Clone, V> OptiMap<K, V> {
    /// Insert a key-value pair. Returns the previous value if the key existed.
    #[inline(always)]
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        dispatch_mut!(self, insert, key, value)
    }

    /// Look up a value by key.
    #[inline(always)]
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        dispatch!(self, get, key)
    }

    /// Returns the key-value pair corresponding to the key.
    #[inline(always)]
    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        dispatch!(self, get_key_value, key)
    }

    /// Look up a value by key, returning a mutable reference.
    #[inline(always)]
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        dispatch_mut!(self, get_mut, key)
    }

    /// Remove a key, returning its value if present.
    #[inline(always)]
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        dispatch_mut!(self, remove, key)
    }

    /// Removes a key, returning the key and value if present.
    #[inline(always)]
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        dispatch_mut!(self, remove_entry, key)
    }

    /// Whether the map contains the given key.
    #[inline(always)]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        dispatch!(self, contains_key, key)
    }

    /// Number of elements in the map.
    #[inline]
    pub fn len(&self) -> usize {
        dispatch!(self, len)
    }

    /// Whether the map is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of elements the map can hold without rehashing.
    #[inline]
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
            Inner::FlatBTree(_) => MapType::FlatBTree,
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
    pub fn iter(&self) -> Iter<'_, K, V> {
        match &self.inner {
            Inner::Ufm(m) => Iter::Ufm(m.iter()),
            Inner::Splitsies(m) => Iter::Splitsies(m.iter()),
            Inner::Ipo(m) => Iter::Ipo(m.iter()),
            Inner::Gaps(m) => Iter::Gaps(m.iter()),
            Inner::Ipo64(m) => Iter::Ipo64(m.iter()),
            Inner::FlatBTree(m) => Iter::FlatBTree(m.iter()),
        }
    }

    /// Iterate over key-value pairs with mutable values.
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        match &mut self.inner {
            Inner::Ufm(m) => IterMut::Ufm(m.iter_mut()),
            Inner::Splitsies(m) => IterMut::Splitsies(m.iter_mut()),
            Inner::Ipo(m) => IterMut::Ipo(m.iter_mut()),
            Inner::Gaps(m) => IterMut::Gaps(m.iter_mut()),
            Inner::Ipo64(m) => IterMut::Ipo64(m.iter_mut()),
            Inner::FlatBTree(m) => IterMut::FlatBTree(m.iter_mut()),
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

    /// Tries to insert a key-value pair, failing if the key already exists.
    pub fn try_insert(
        &mut self,
        key: K,
        value: V,
    ) -> Result<(), crate::traits::OccupiedError<K, V>> {
        dispatch_mut!(self, try_insert, key, value)
    }

    /// Gets the given key's corresponding entry in the map for in-place
    /// manipulation.
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        match &mut self.inner {
            Inner::Ufm(m) => match m.entry(key) {
                crate::map::Entry::Occupied(e) => Entry::Occupied(OccupiedEntry::Ufm(e)),
                crate::map::Entry::Vacant(e) => Entry::Vacant(VacantEntry::Ufm(e)),
            },
            Inner::Splitsies(m) => match m.entry(key) {
                crate::split_overflow::map::Entry::Occupied(e) => {
                    Entry::Occupied(OccupiedEntry::Splitsies(e))
                }
                crate::split_overflow::map::Entry::Vacant(e) => {
                    Entry::Vacant(VacantEntry::Splitsies(e))
                }
            },
            Inner::Ipo(m) => match m.entry(key) {
                crate::in_place_overflow::map::Entry::Occupied(e) => {
                    Entry::Occupied(OccupiedEntry::Ipo(e))
                }
                crate::in_place_overflow::map::Entry::Vacant(e) => {
                    Entry::Vacant(VacantEntry::Ipo(e))
                }
            },
            Inner::Gaps(m) => match m.entry(key) {
                crate::gaps::map::Entry::Occupied(e) => Entry::Occupied(OccupiedEntry::Gaps(e)),
                crate::gaps::map::Entry::Vacant(e) => Entry::Vacant(VacantEntry::Gaps(e)),
            },
            Inner::Ipo64(m) => match m.entry(key) {
                crate::ipo64::map::Entry::Occupied(e) => Entry::Occupied(OccupiedEntry::Ipo64(e)),
                crate::ipo64::map::Entry::Vacant(e) => Entry::Vacant(VacantEntry::Ipo64(e)),
            },
            Inner::FlatBTree(m) => match m.entry(key) {
                crate::flat_btree::map::Entry::Occupied(e) => {
                    Entry::Occupied(OccupiedEntry::FlatBTree(e))
                }
                crate::flat_btree::map::Entry::Vacant(e) => {
                    Entry::Vacant(VacantEntry::FlatBTree(e))
                }
            },
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

    /// Clears the map, returning all key-value pairs as an iterator.
    pub fn drain(&mut self) -> impl Iterator<Item = (K, V)> {
        let items: Vec<(K, V)> = match &mut self.inner {
            Inner::Ufm(m) => m.drain().collect(),
            Inner::Splitsies(m) => m.drain().collect(),
            Inner::Ipo(m) => m.drain().collect(),
            Inner::Gaps(m) => m.drain().collect(),
            Inner::Ipo64(m) => m.drain().collect(),
            Inner::FlatBTree(m) => m.drain().collect(),
        };
        items.into_iter()
    }

    // ── Sorted operations (require FlatBTree backend) ──────────────────────

    /// Returns the first (minimum) key-value pair.
    ///
    /// # Panics
    ///
    /// Panics if the current backend is not [`FlatBTree`]. Use
    /// [`OptiMap::flat_btree`] or [`Hint::Sorted`] to guarantee a sorted backend.
    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        match &self.inner {
            Inner::FlatBTree(m) => m.first_key_value(),
            _ => not_flat_btree("first_key_value"),
        }
    }

    /// Returns the last (maximum) key-value pair.
    ///
    /// # Panics
    ///
    /// Panics if the current backend is not [`FlatBTree`].
    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        match &self.inner {
            Inner::FlatBTree(m) => m.last_key_value(),
            _ => not_flat_btree("last_key_value"),
        }
    }

    /// Removes and returns the first (minimum) key-value pair.
    ///
    /// # Panics
    ///
    /// Panics if the current backend is not [`FlatBTree`].
    pub fn pop_first(&mut self) -> Option<(K, V)> {
        match &mut self.inner {
            Inner::FlatBTree(m) => m.pop_first(),
            _ => not_flat_btree("pop_first"),
        }
    }

    /// Removes and returns the last (maximum) key-value pair.
    ///
    /// # Panics
    ///
    /// Panics if the current backend is not [`FlatBTree`].
    pub fn pop_last(&mut self) -> Option<(K, V)> {
        match &mut self.inner {
            Inner::FlatBTree(m) => m.pop_last(),
            _ => not_flat_btree("pop_last"),
        }
    }

    /// Iterate over all key-value pairs in sorted order.
    ///
    /// # Panics
    ///
    /// Panics if the current backend is not [`FlatBTree`].
    pub fn iter_sorted(&self) -> impl Iterator<Item = (&K, &V)> {
        match &self.inner {
            Inner::FlatBTree(m) => {
                let iter: crate::flat_btree::map::Iter<'_, K, V> = m.iter();
                iter
            }
            _ => not_flat_btree("iter_sorted"),
        }
    }

    /// Iterate over key-value pairs within the given range, in sorted order.
    ///
    /// # Panics
    ///
    /// Panics if the current backend is not [`FlatBTree`].
    pub fn range<'a, Q, R>(&'a self, range: R) -> impl Iterator<Item = (&'a K, &'a V)>
    where
        K: Borrow<Q> + 'a,
        Q: Ord + ?Sized,
        R: std::ops::RangeBounds<Q> + 'a,
    {
        match &self.inner {
            Inner::FlatBTree(m) => {
                let iter: crate::flat_btree::map::RangeIter<'_, K, V> = m.range(range);
                iter
            }
            _ => not_flat_btree("range"),
        }
    }

    /// Iterate over key-value pairs within the given range, yielding mutable
    /// values, in sorted order.
    ///
    /// # Panics
    ///
    /// Panics if the current backend is not [`FlatBTree`].
    pub fn range_mut<'a, Q, R>(&'a mut self, range: R) -> impl Iterator<Item = (&'a K, &'a mut V)>
    where
        K: Borrow<Q> + 'a,
        Q: Ord + ?Sized,
        R: std::ops::RangeBounds<Q> + 'a,
    {
        match &mut self.inner {
            Inner::FlatBTree(m) => {
                let iter: crate::flat_btree::map::RangeIterMut<'_, K, V> = m.range_mut(range);
                iter
            }
            _ => not_flat_btree("range_mut"),
        }
    }

    /// Splits the map at `at`, keeping keys `< at` in `self` and returning
    /// a new map with keys `>= at`.
    ///
    /// # Panics
    ///
    /// Panics if the current backend is not [`FlatBTree`].
    pub fn split_off<Q>(&mut self, at: &Q) -> Self
    where
        K: Borrow<Q>,
        Q: Ord + ?Sized,
    {
        match &mut self.inner {
            Inner::FlatBTree(m) => {
                let other = m.split_off(at);
                OptiMap {
                    inner: Inner::FlatBTree(other),
                    backend: Backend::Pinned,
                }
            }
            _ => not_flat_btree("split_off"),
        }
    }

    /// Moves all entries from `other` into `self`, leaving `other` empty.
    ///
    /// On key collision, `other`'s value wins (matching
    /// [`std::collections::BTreeMap::append`]).
    ///
    /// # Panics
    ///
    /// Panics if either `self` or `other` has a non-[`FlatBTree`] backend.
    pub fn append(&mut self, other: &mut Self) {
        match (&mut self.inner, &mut other.inner) {
            (Inner::FlatBTree(a), Inner::FlatBTree(b)) => a.append(b),
            _ => not_flat_btree("append"),
        }
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
            Inner::FlatBTree(mut m) => m.drain().collect(),
        };
        for (k, v) in entries {
            dispatch_mut!(self, insert, k, v);
        }
    }
}

// ── Default, Debug, Clone ──────────────────────────────────────────────────

impl<K: Hash + Eq + Ord + Clone, V> Default for OptiMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Ord + Clone + fmt::Debug, V: fmt::Debug> fmt::Debug for OptiMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();
        match &self.inner {
            Inner::Ufm(m) => {
                for (k, v) in m.iter() {
                    map.entry(k, v);
                }
            }
            Inner::Splitsies(m) => {
                for (k, v) in m.iter() {
                    map.entry(k, v);
                }
            }
            Inner::Ipo(m) => {
                for (k, v) in m.iter() {
                    map.entry(k, v);
                }
            }
            Inner::Gaps(m) => {
                for (k, v) in m.iter() {
                    map.entry(k, v);
                }
            }
            Inner::Ipo64(m) => {
                for (k, v) in m.iter() {
                    map.entry(k, v);
                }
            }
            Inner::FlatBTree(m) => {
                for (k, v) in m.iter() {
                    map.entry(k, v);
                }
            }
        }
        map.finish()
    }
}

impl<K: Hash + Eq + Ord + Clone, V: Clone> Clone for OptiMap<K, V> {
    fn clone(&self) -> Self {
        OptiMap {
            inner: match &self.inner {
                Inner::Ufm(m) => Inner::Ufm(m.clone()),
                Inner::Splitsies(m) => Inner::Splitsies(m.clone()),
                Inner::Ipo(m) => Inner::Ipo(m.clone()),
                Inner::Gaps(m) => Inner::Gaps(m.clone()),
                Inner::Ipo64(m) => Inner::Ipo64(m.clone()),
                Inner::FlatBTree(m) => Inner::FlatBTree(m.clone()),
            },
            backend: self.backend,
        }
    }
}

impl<K: Hash + Eq + Ord + Clone, V: PartialEq> PartialEq for OptiMap<K, V> {
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
                        _ => {
                            eq = false;
                            break;
                        }
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
            Inner::FlatBTree(m) => check_eq!(m),
        }
        eq
    }
}

impl<K: Hash + Eq + Ord + Clone, V: Eq> Eq for OptiMap<K, V> {}

impl<K: Hash + Eq + Ord + Clone, V> FromIterator<(K, V)> for OptiMap<K, V> {
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

impl<K: Hash + Eq + Ord + Clone, V> Extend<(K, V)> for OptiMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K: Hash + Eq + Ord + Clone, V> IntoIterator for OptiMap<K, V> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        match self.inner {
            Inner::Ufm(m) => IntoIter::Ufm(m.into_iter()),
            Inner::Splitsies(m) => IntoIter::Splitsies(m.into_iter()),
            Inner::Ipo(m) => IntoIter::Ipo(m.into_iter()),
            Inner::Gaps(m) => IntoIter::Gaps(m.into_iter()),
            Inner::Ipo64(m) => IntoIter::Ipo64(m.into_iter()),
            Inner::FlatBTree(m) => IntoIter::FlatBTree(m.into_iter()),
        }
    }
}

impl<K, Q, V> std::ops::Index<&Q> for OptiMap<K, V>
where
    K: Hash + Eq + Ord + Clone + Borrow<Q>,
    Q: Hash + Eq + Ord + ?Sized,
{
    type Output = V;

    fn index(&self, key: &Q) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

// ── Map facade trait impl ──────────────────────────────────────────────────
//
// `HashedMap` is intentionally NOT implemented for `OptiMap`. With FlatBTree
// in the variant set, dispatch into FlatBTree requires `K: Ord + Clone` and
// `Q: Ord` — bounds the `HashedMap` trait can't express. Generic code that
// wants a universal `<M: ...>` bound for OptiMap should use `Map` instead.


impl<K: Hash + Eq + Ord + Clone, V> crate::Map<K, V> for OptiMap<K, V> {
    fn new() -> Self {
        OptiMap::new()
    }
    fn with_capacity(capacity: usize) -> Self {
        OptiMap::with_capacity(capacity)
    }
    #[inline]
    fn insert(&mut self, key: K, value: V) -> Option<V> {
        OptiMap::insert(self, key, value)
    }
    #[inline]
    fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        OptiMap::get(self, key)
    }
    #[inline]
    fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        OptiMap::get_key_value(self, key)
    }
    #[inline]
    fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        OptiMap::get_mut(self, key)
    }
    #[inline]
    fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        OptiMap::remove(self, key)
    }
    #[inline]
    fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        OptiMap::remove_entry(self, key)
    }
    #[inline]
    fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + Ord + ?Sized,
    {
        OptiMap::contains_key(self, key)
    }
    fn try_insert(&mut self, key: K, value: V) -> Result<(), crate::traits::OccupiedError<K, V>> {
        OptiMap::try_insert(self, key, value)
    }
    #[inline]
    fn len(&self) -> usize {
        OptiMap::len(self)
    }
    #[inline]
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
    fn into_keys(self) -> impl Iterator<Item = K> {
        OptiMap::into_keys(self)
    }
    fn into_values(self) -> impl Iterator<Item = V> {
        OptiMap::into_values(self)
    }
}

// ── Private helpers ────────────────────────────────────────────────────────

/// Cold path: panics with a clear message when a sorted-only method is called
/// on a non-FlatBTree backend. The `!` return type coerces to any concrete
/// iterator or value type.
#[cold]
#[track_caller]
fn not_flat_btree(method: &str) -> ! {
    panic!(
        "{method} requires a FlatBTree backend. \
         Use OptiMap::flat_btree() or Hint::Sorted to guarantee a sorted backend."
    )
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
        for mt in [
            MapType::Ufm,
            MapType::Splitsies,
            MapType::Ipo,
            MapType::Gaps,
            MapType::Ipo64,
        ] {
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
        assert_eq!(
            OptiMap::<u64, u64>::splitsies().map_type(),
            MapType::Splitsies
        );
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
        // Reserve into the medium band — policy switches to IPO.
        map.reserve(10_000);
        assert_eq!(map.map_type(), MapType::Ipo);
        // Verify data survived the transition
        for i in 0..10u8 {
            assert_eq!(map.get(&i), Some(&i), "lost key {i} after transition");
        }
    }

    #[test]
    fn auto_transition_into_dram_band() {
        // Medium band: IPO. Resize beyond TOMBSTONE_DRAM_CLIFF → Splitsies.
        let mut map = OptiMap::<u64, u64>::with_capacity(10_000);
        assert_eq!(map.map_type(), MapType::Ipo);
        for i in 0..100u64 {
            map.insert(i, i);
        }
        map.reserve(2_000_000);
        assert_eq!(
            map.map_type(),
            MapType::Splitsies,
            "should fall back to Splitsies above DRAM cliff"
        );
        for i in 0..100u64 {
            assert_eq!(map.get(&i), Some(&i), "lost key {i} after transition");
        }
    }

    #[test]
    fn auto_band_thresholds() {
        // Boundary check: SMALL → Splitsies, MEDIUM → IPO, LARGE → Splitsies.
        let m = OptiMap::<u64, u64>::with_capacity(0);
        assert_eq!(m.map_type(), MapType::Splitsies);
        let m = OptiMap::<u64, u64>::with_capacity(100);
        assert_eq!(m.map_type(), MapType::Splitsies);
        let m = OptiMap::<u64, u64>::with_capacity(10_000);
        assert_eq!(m.map_type(), MapType::Ipo);
        let m = OptiMap::<u64, u64>::with_capacity(500_000);
        assert_eq!(m.map_type(), MapType::Ipo);
        let m = OptiMap::<u64, u64>::with_capacity(2_000_000);
        assert_eq!(m.map_type(), MapType::Splitsies);
    }

    #[test]
    fn read_heavy_hint_band() {
        // ReadHeavy: IPO at small/medium, Splitsies at DRAM scale.
        let m = OptiMap::<u64, u64>::with_capacity_and_hint(10_000, Hint::ReadHeavy);
        assert_eq!(m.map_type(), MapType::Ipo);
        let m = OptiMap::<u64, u64>::with_capacity_and_hint(5_000_000, Hint::ReadHeavy);
        assert_eq!(m.map_type(), MapType::Splitsies);
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

    #[test]
    fn flat_btree_backend() {
        // Sorted hint picks FlatBTree.
        let mut map: OptiMap<u32, u32> = OptiMap::with_hint(Hint::Sorted);
        assert_eq!(map.map_type(), MapType::FlatBTree);
        map.insert(3, 30);
        map.insert(1, 10);
        map.insert(2, 20);
        assert_eq!(map.get(&2), Some(&20));
        assert_eq!(map.len(), 3);
        // Iter on a FlatBTree yields sorted order.
        let pairs: Vec<_> = map.iter().map(|(&k, &v)| (k, v)).collect();
        assert_eq!(pairs, vec![(1, 10), (2, 20), (3, 30)]);

        // Pinned constructor.
        let pinned: OptiMap<u32, u32> = OptiMap::flat_btree();
        assert_eq!(pinned.map_type(), MapType::FlatBTree);

        // Entry API works through the FlatBTree variant.
        let mut m: OptiMap<u32, u32> = OptiMap::flat_btree();
        *m.entry(7).or_insert(0) += 1;
        *m.entry(7).or_insert(0) += 1;
        assert_eq!(m.get(&7), Some(&2));
    }

    #[test]
    fn into_iterator() {
        let map: OptiMap<u64, u64> = (0..50).map(|i| (i, i * 3)).collect();
        let mut pairs: Vec<(u64, u64)> = map.into_iter().collect();
        pairs.sort();
        assert_eq!(pairs.len(), 50);
        assert_eq!(pairs[0], (0, 0));
        assert_eq!(pairs[49], (49, 147));
    }

    #[test]
    fn for_loop() {
        let map: OptiMap<u64, u64> = vec![(1, 10), (2, 20), (3, 30)].into_iter().collect();
        let mut sum = 0u64;
        for (k, v) in map {
            sum += k + v;
        }
        assert_eq!(sum, 1 + 10 + 2 + 20 + 3 + 30);
    }

    #[test]
    fn index() {
        let mut map = OptiMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        assert_eq!(map[&"a"], 1);
        assert_eq!(map[&"b"], 2);
    }

    #[test]
    #[should_panic(expected = "no entry found for key")]
    fn index_missing_panics() {
        let map = OptiMap::<u64, u64>::new();
        let _ = map[&42];
    }

    #[test]
    fn try_insert_success() {
        let mut map = OptiMap::new();
        assert_eq!(map.try_insert(1u64, 10u64), Ok(()));
        assert_eq!(map.get(&1), Some(&10));
    }

    #[test]
    fn try_insert_occupied() {
        let mut map = OptiMap::new();
        map.insert(1u64, 10u64);
        let err = map.try_insert(1, 20).unwrap_err();
        assert_eq!(err.key, 1);
        assert_eq!(err.value, 20);
        // Original value unchanged
        assert_eq!(map.get(&1), Some(&10));
    }

    #[test]
    fn into_keys_values() {
        let map: OptiMap<u64, u64> = vec![(1, 10), (2, 20), (3, 30)].into_iter().collect();
        let mut keys: Vec<u64> = map.clone().into_keys().collect();
        keys.sort();
        assert_eq!(keys, vec![1, 2, 3]);

        let mut values: Vec<u64> = map.into_values().collect();
        values.sort();
        assert_eq!(values, vec![10, 20, 30]);
    }

    #[test]
    fn entry_or_insert() {
        for mt in [
            MapType::Ufm,
            MapType::Splitsies,
            MapType::Ipo,
            MapType::Gaps,
            MapType::Ipo64,
        ] {
            let mut map = OptiMap::<u64, u64>::with_type(mt);
            map.entry(1).or_insert(10);
            assert_eq!(map.get(&1), Some(&10));
            map.entry(1).or_insert(20);
            assert_eq!(map.get(&1), Some(&10)); // not overwritten
        }
    }

    #[test]
    fn entry_or_insert_with() {
        let mut map = OptiMap::<u64, u64>::new();
        map.entry(1).or_insert_with(|| 42);
        assert_eq!(map.get(&1), Some(&42));
    }

    #[test]
    fn entry_or_default() {
        let mut map = OptiMap::<u64, u64>::new();
        map.entry(1).or_default();
        assert_eq!(map.get(&1), Some(&0));
    }

    #[test]
    fn entry_or_insert_with_key() {
        let mut map = OptiMap::<u64, u64>::new();
        map.entry(5).or_insert_with_key(|&k| k * 10);
        assert_eq!(map.get(&5), Some(&50));
    }

    #[test]
    fn entry_and_modify() {
        let mut map = OptiMap::<u64, u64>::new();
        map.insert(1, 10);
        map.entry(1).and_modify(|v| *v += 5).or_insert(0);
        assert_eq!(map.get(&1), Some(&15));
        map.entry(2).and_modify(|v| *v += 5).or_insert(0);
        assert_eq!(map.get(&2), Some(&0));
    }

    #[test]
    fn entry_key() {
        let mut map = OptiMap::<String, u64>::new();
        let e = map.entry("hello".to_string());
        assert_eq!(e.key(), "hello");
    }

    #[test]
    fn entry_occupied_get_insert() {
        let mut map = OptiMap::<u64, u64>::new();
        map.insert(1, 10);
        match map.entry(1) {
            Entry::Occupied(mut e) => {
                assert_eq!(*e.get(), 10);
                assert_eq!(e.key(), &1);
                let old = e.insert(20);
                assert_eq!(old, 10);
            }
            Entry::Vacant(_) => panic!("expected occupied"),
        }
        assert_eq!(map.get(&1), Some(&20));
    }

    #[test]
    fn entry_vacant_insert() {
        let mut map = OptiMap::<u64, u64>::new();
        match map.entry(1) {
            Entry::Occupied(_) => panic!("expected vacant"),
            Entry::Vacant(e) => {
                assert_eq!(e.key(), &1);
                e.insert(42);
            }
        }
        assert_eq!(map.get(&1), Some(&42));
    }

    #[test]
    fn entry_vacant_into_key() {
        let mut map = OptiMap::<String, u64>::new();
        match map.entry("hello".to_string()) {
            Entry::Vacant(e) => {
                let k = e.into_key();
                assert_eq!(k, "hello");
            }
            Entry::Occupied(_) => panic!("expected vacant"),
        }
    }

    #[test]
    fn entry_occupied_into_mut() {
        let mut map = OptiMap::<u64, u64>::new();
        map.insert(1, 10);
        match map.entry(1) {
            Entry::Occupied(e) => {
                let v = e.into_mut();
                *v = 99;
            }
            Entry::Vacant(_) => panic!("expected occupied"),
        }
        assert_eq!(map.get(&1), Some(&99));
    }

    #[test]
    fn entry_counting_all_backends() {
        for mt in [
            MapType::Ufm,
            MapType::Splitsies,
            MapType::Ipo,
            MapType::Gaps,
            MapType::Ipo64,
        ] {
            let mut map = OptiMap::<u64, u64>::with_type(mt);
            for &key in &[1, 2, 3, 1, 2, 1] {
                map.entry(key).and_modify(|c| *c += 1).or_insert(1);
            }
            assert_eq!(map.get(&1), Some(&3), "failed for {mt:?}");
            assert_eq!(map.get(&2), Some(&2), "failed for {mt:?}");
            assert_eq!(map.get(&3), Some(&1), "failed for {mt:?}");
        }
    }

    #[test]
    fn try_insert_all_backends() {
        for mt in [
            MapType::Ufm,
            MapType::Splitsies,
            MapType::Ipo,
            MapType::Gaps,
            MapType::Ipo64,
        ] {
            let mut map = OptiMap::<u64, u64>::with_type(mt);
            assert_eq!(map.try_insert(1, 10), Ok(()));
            assert_eq!(map.try_insert(2, 20), Ok(()));
            assert!(map.try_insert(1, 30).is_err());
            assert_eq!(map.get(&1), Some(&10)); // unchanged
            assert_eq!(map.len(), 2);
        }
    }

    mod sorted_ops {
        use super::*;

        fn sorted_map() -> OptiMap<i32, i32> {
            OptiMap::flat_btree()
        }

        fn sorted_map_str() -> OptiMap<i32, &'static str> {
            OptiMap::flat_btree()
        }

        #[test]
        fn first_last_key_value() {
            let mut map = sorted_map_str();
            map.insert(1, "a");
            map.insert(5, "e");
            map.insert(3, "c");
            assert_eq!(map.first_key_value(), Some((&1, &"a")));
            assert_eq!(map.last_key_value(), Some((&5, &"e")));
        }

        #[test]
        fn first_last_key_value_empty() {
            let map: OptiMap<i32, i32> = OptiMap::flat_btree();
            assert_eq!(map.first_key_value(), None);
            assert_eq!(map.last_key_value(), None);
        }

        #[test]
        fn pop_first_last() {
            let mut map = sorted_map();
            for i in 1..=5 {
                map.insert(i, i);
            }
            assert_eq!(map.pop_first(), Some((1, 1)));
            assert_eq!(map.pop_last(), Some((5, 5)));
            assert_eq!(map.len(), 3);
            // Remaining keys are 2,3,4 in sorted order
            assert_eq!(map.first_key_value(), Some((&2, &2)));
            assert_eq!(map.last_key_value(), Some((&4, &4)));
        }

        #[test]
        fn iter_sorted() {
            let mut map = OptiMap::flat_btree();
            for i in [5, 3, 1, 4, 2] {
                map.insert(i, i * 10);
            }
            let keys: Vec<_> = map.iter_sorted().map(|(k, _)| *k).collect();
            assert_eq!(keys, vec![1, 2, 3, 4, 5]);
        }

        #[test]
        fn iter_sorted_is_sorted_even_after_mutations() {
            let mut map = OptiMap::flat_btree();
            map.insert(100, ());
            map.insert(50, ());
            map.insert(75, ());
            map.remove(&50);
            map.insert(25, ());
            let keys: Vec<_> = map.iter_sorted().map(|(k, _)| *k).collect();
            assert_eq!(keys, vec![25, 75, 100]);
        }

        #[test]
        fn range_query() {
            let mut map = sorted_map();
            for i in 0..10 {
                map.insert(i, i * 10);
            }
            let range: Vec<_> = map.range(3..7).map(|(k, _)| *k).collect();
            assert_eq!(range, vec![3, 4, 5, 6]);
        }

        #[test]
        fn range_mut() {
            let mut map = sorted_map();
            for i in 0..30 {
                map.insert(i, i);
            }
            for (_, v) in map.range_mut(10..20) {
                *v += 1000;
            }
            for i in 0..30 {
                let want = if (10..20).contains(&i) { i + 1000 } else { i };
                assert_eq!(map.get(&i), Some(&want));
            }
        }

        #[test]
        #[should_panic(expected = "first_key_value requires a FlatBTree backend")]
        fn first_key_value_panics_on_hash_backend() {
            let map: OptiMap<u64, u64> = OptiMap::splitsies();
            let _ = map.first_key_value();
        }

        #[test]
        #[should_panic(expected = "pop_first requires a FlatBTree backend")]
        fn pop_first_panics_on_hash_backend() {
            let mut map: OptiMap<u64, u64> = OptiMap::ipo();
            let _ = map.pop_first();
        }

        #[test]
        #[should_panic(expected = "iter_sorted requires a FlatBTree backend")]
        fn iter_sorted_panics_on_hash_backend() {
            let map: OptiMap<u64, u64> = OptiMap::gaps();
            let _ = map.iter_sorted();
        }

        #[test]
        #[should_panic(expected = "range requires a FlatBTree backend")]
        fn range_panics_on_hash_backend() {
            let map: OptiMap<u64, u64> = OptiMap::ufm();
            let _ = map.range(..);
        }

        #[test]
        #[should_panic(expected = "range_mut requires a FlatBTree backend")]
        fn range_mut_panics_on_hash_backend() {
            let mut map: OptiMap<u64, u64> = OptiMap::ipo64();
            let _ = map.range_mut(..);
        }

        #[test]
        fn hint_sorted_works() {
            let mut map: OptiMap<u32, u32> = OptiMap::with_hint(Hint::Sorted);
            map.insert(3, 30);
            map.insert(1, 10);
            map.insert(2, 20);
            let keys: Vec<_> = map.iter_sorted().map(|(k, _)| *k).collect();
            assert_eq!(keys, vec![1, 2, 3]);
        }

        #[test]
        fn split_off() {
            let mut map = sorted_map();
            for i in 0..100 {
                map.insert(i, i);
            }
            let upper = map.split_off(&50);
            assert_eq!(map.len(), 50);
            assert_eq!(upper.len(), 50);
            assert_eq!(map.last_key_value(), Some((&49, &49)));
            assert_eq!(upper.first_key_value(), Some((&50, &50)));
            for i in 0..50 {
                assert_eq!(map.get(&i), Some(&i));
            }
            for i in 50..100 {
                assert!(map.get(&i).is_none());
                assert_eq!(upper.get(&i), Some(&i));
            }
        }

        #[test]
        fn split_off_empty_left() {
            let mut map = sorted_map();
            for i in 0..10 {
                map.insert(i, i);
            }
            let upper = map.split_off(&0);
            assert!(map.is_empty());
            assert_eq!(upper.len(), 10);
        }

        #[test]
        fn split_off_empty_right() {
            let mut map = sorted_map();
            for i in 0..10 {
                map.insert(i, i);
            }
            let upper = map.split_off(&100);
            assert!(upper.is_empty());
            assert_eq!(map.len(), 10);
        }

        #[test]
        fn append_disjoint() {
            let mut a = sorted_map();
            let mut b = sorted_map();
            for i in 0..50 {
                a.insert(i, i);
            }
            for i in 50..100 {
                b.insert(i, i);
            }
            a.append(&mut b);
            assert_eq!(a.len(), 100);
            assert!(b.is_empty());
            for i in 0..100 {
                assert_eq!(a.get(&i), Some(&i));
            }
        }

        #[test]
        fn append_collision_other_wins() {
            let mut a: OptiMap<i32, &str> = OptiMap::flat_btree();
            let mut b: OptiMap<i32, &str> = OptiMap::flat_btree();
            a.insert(1, "original");
            b.insert(1, "replacement");
            a.append(&mut b);
            assert_eq!(a.get(&1), Some(&"replacement"));
        }

        #[test]
        fn append_empty_self() {
            let mut a: OptiMap<i32, i32> = OptiMap::flat_btree();
            let mut b = sorted_map();
            b.insert(1, 10);
            b.insert(2, 20);
            a.append(&mut b);
            assert_eq!(a.len(), 2);
            assert!(b.is_empty());
            assert_eq!(a.get(&1), Some(&10));
        }

        #[test]
        fn append_empty_other() {
            let mut a = sorted_map();
            let mut b: OptiMap<i32, i32> = OptiMap::flat_btree();
            a.insert(1, 10);
            let len_before = a.len();
            a.append(&mut b);
            assert_eq!(a.len(), len_before);
            assert!(b.is_empty());
        }

        #[test]
        #[should_panic(expected = "split_off requires a FlatBTree backend")]
        fn split_off_panics_on_hash_backend() {
            let mut map: OptiMap<u64, u64> = OptiMap::splitsies();
            let _ = map.split_off(&42);
        }

        #[test]
        #[should_panic(expected = "append requires a FlatBTree backend")]
        fn append_panics_on_hash_backend() {
            let mut a: OptiMap<u64, u64> = OptiMap::splitsies();
            let mut b: OptiMap<u64, u64> = OptiMap::flat_btree();
            a.append(&mut b);
        }

        #[test]
        #[should_panic(expected = "append requires a FlatBTree backend")]
        fn append_panics_if_other_is_hash() {
            let mut a: OptiMap<u64, u64> = OptiMap::flat_btree();
            let mut b: OptiMap<u64, u64> = OptiMap::splitsies();
            a.append(&mut b);
        }

        #[test]
        fn split_off_then_append_roundtrip() {
            let mut map = sorted_map();
            for i in 0..200 {
                map.insert(i, i * 2);
            }
            let mut upper = map.split_off(&100);
            assert_eq!(map.len(), 100);
            assert_eq!(upper.len(), 100);
            map.append(&mut upper);
            assert_eq!(map.len(), 200);
            assert!(upper.is_empty());
            for i in 0..200 {
                assert_eq!(map.get(&i), Some(&(i * 2)));
            }
        }
    }
}
