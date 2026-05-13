//! Raw entry API — lookup and insert by hash + custom equality.
//!
//! Mirrors hashbrown's `RawEntry` API. Works on any [`GenericMap`] backend
//! (UFM, Splitsies, IPO, IPO64, Gaps, SoA, matrix variants).
//!
//! Three lookup forms are supported:
//! - [`RawEntryBuilder::from_key`] — borrow-form lookup, hashes the key for you.
//! - [`RawEntryBuilder::from_key_hashed_nocheck`] — caller supplies the hash.
//! - [`RawEntryBuilder::from_hash`] — caller supplies hash + a `Fn(&K) -> bool`
//!   equality closure. Useful for interning, prefix lookups, etc.
//!
//! The mutable side ([`RawEntryBuilderMut`]) returns a [`RawEntryMut`] enum
//! with `Occupied` / `Vacant` variants matching the regular entry API but
//! exposing key mutation and per-call hash control.
//!
//! Mutating a key via [`RawOccupiedEntryMut::key_mut`] in a way that changes
//! its hash or equality class leaves the table in a logically-inconsistent
//! state — subsequent lookups for the new key value will not find it. Same
//! caveat as `std::collections::HashMap::raw_entry_mut`.

use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

use crate::generic_map::GenericMap;
use crate::raw::hash;
use crate::raw::table_api::RawTableApi;

// ── Read-only builder ──────────────────────────────────────────────────────

/// Builder returned by [`GenericMap::raw_entry`](crate::generic_map::GenericMap::raw_entry).
pub struct RawEntryBuilder<'a, K, V, S, R: RawTableApi<K, V>> {
    pub(crate) map: &'a GenericMap<K, V, S, R>,
}

impl<'a, K, V, S, R> RawEntryBuilder<'a, K, V, S, R>
where
    R: RawTableApi<K, V>,
    S: BuildHasher,
{
    /// Look up by a borrowed form of the key. Hashes `k` with the map's hasher.
    pub fn from_key<Q>(self, k: &Q) -> Option<(&'a K, &'a V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let h = hash::hash_no_mix(k, &self.map.hash_builder);
        self.from_key_hashed_nocheck(h, k)
    }

    /// Look up by a borrowed form of the key, using a caller-supplied hash.
    ///
    /// The hash must match what the map's hasher would produce for an equal
    /// key; otherwise lookup will spuriously miss.
    pub fn from_key_hashed_nocheck<Q>(self, hash: u64, k: &Q) -> Option<(&'a K, &'a V)>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.from_hash(hash, |stored| stored.borrow() == k)
    }

    /// Look up by hash + a custom equality closure.
    pub fn from_hash<F>(self, hash: u64, is_match: F) -> Option<(&'a K, &'a V)>
    where
        F: Fn(&K) -> bool,
    {
        let (gi, si) = self.map.table.find_by_hash(hash, is_match)?;
        unsafe {
            Some((
                &*self.map.table.key_ptr(gi, si),
                &*self.map.table.value_ptr(gi, si),
            ))
        }
    }
}

// ── Mutable builder ────────────────────────────────────────────────────────

/// Builder returned by [`GenericMap::raw_entry_mut`](crate::generic_map::GenericMap::raw_entry_mut).
pub struct RawEntryBuilderMut<'a, K, V, S, R: RawTableApi<K, V>> {
    pub(crate) map: &'a mut GenericMap<K, V, S, R>,
}

impl<'a, K, V, S, R> RawEntryBuilderMut<'a, K, V, S, R>
where
    R: RawTableApi<K, V>,
    S: BuildHasher,
{
    /// Find or prepare-to-insert by a borrowed form of the key.
    pub fn from_key<Q>(self, k: &Q) -> RawEntryMut<'a, K, V, S, R>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let h = hash::hash_no_mix(k, &self.map.hash_builder);
        self.from_key_hashed_nocheck(h, k)
    }

    /// Find or prepare-to-insert with a caller-supplied hash.
    pub fn from_key_hashed_nocheck<Q>(self, hash: u64, k: &Q) -> RawEntryMut<'a, K, V, S, R>
    where
        K: Borrow<Q>,
        Q: Eq + ?Sized,
    {
        self.from_hash(hash, |stored| stored.borrow() == k)
    }

    /// Find or prepare-to-insert by hash + custom equality closure.
    pub fn from_hash<F>(self, hash: u64, is_match: F) -> RawEntryMut<'a, K, V, S, R>
    where
        F: Fn(&K) -> bool,
    {
        // find_by_hash on an unallocated table returns None, which is correct
        // for Vacant; we defer allocation to insert time (ensure_capacity).
        match self.map.table.find_by_hash(hash, is_match) {
            Some((gi, si)) => RawEntryMut::Occupied(RawOccupiedEntryMut {
                table: &mut self.map.table,
                gi,
                si,
                hash,
                _marker: PhantomData,
            }),
            None => RawEntryMut::Vacant(RawVacantEntryMut {
                map: self.map,
                hash,
            }),
        }
    }
}

// ── Mutable entry enum ─────────────────────────────────────────────────────

/// Mutable entry returned by [`RawEntryBuilderMut`].
pub enum RawEntryMut<'a, K, V, S, R: RawTableApi<K, V>> {
    Occupied(RawOccupiedEntryMut<'a, K, V, R>),
    Vacant(RawVacantEntryMut<'a, K, V, S, R>),
}

/// Occupied half of [`RawEntryMut`].
pub struct RawOccupiedEntryMut<'a, K, V, R: RawTableApi<K, V>> {
    table: &'a mut R,
    gi: usize,
    si: usize,
    /// The hash used to find this slot — needed by `erase_slot` so overflow
    /// bookkeeping (overflow-bit family) or tombstone marking (tombstone
    /// family) can be applied correctly.
    hash: u64,
    _marker: PhantomData<(K, V)>,
}

/// Vacant half of [`RawEntryMut`].
pub struct RawVacantEntryMut<'a, K, V, S, R: RawTableApi<K, V>> {
    map: &'a mut GenericMap<K, V, S, R>,
    hash: u64,
}

impl<'a, K, V, S, R> RawEntryMut<'a, K, V, S, R>
where
    R: RawTableApi<K, V>,
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Ensure the entry is occupied, inserting `(default_key, default_value)`
    /// if it was vacant. Returns mutable references to the stored key and value.
    pub fn or_insert(self, default_key: K, default_value: V) -> (&'a mut K, &'a mut V) {
        match self {
            RawEntryMut::Occupied(e) => e.into_key_value(),
            RawEntryMut::Vacant(e) => e.insert(default_key, default_value),
        }
    }

    /// As [`or_insert`](Self::or_insert), but the default `(K, V)` is computed
    /// lazily by the supplied closure.
    pub fn or_insert_with<F>(self, default: F) -> (&'a mut K, &'a mut V)
    where
        F: FnOnce() -> (K, V),
    {
        match self {
            RawEntryMut::Occupied(e) => e.into_key_value(),
            RawEntryMut::Vacant(e) => {
                let (k, v) = default();
                e.insert(k, v)
            }
        }
    }

    /// Run `f` on the occupied entry's key and value, then return `self`.
    /// No-op when vacant.
    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut K, &mut V),
    {
        match self {
            RawEntryMut::Occupied(mut e) => {
                let (k, v) = e.get_key_value_mut();
                f(k, v);
                RawEntryMut::Occupied(e)
            }
            other => other,
        }
    }
}

// ── Occupied entry methods ─────────────────────────────────────────────────

impl<'a, K, V, R: RawTableApi<K, V>> RawOccupiedEntryMut<'a, K, V, R> {
    /// Reference to the stored key.
    pub fn key(&self) -> &K {
        unsafe { &*self.table.key_ptr(self.gi, self.si) }
    }

    /// Mutable reference to the stored key.
    ///
    /// Mutating the key in a way that changes its hash or equality class
    /// breaks the table invariant — subsequent lookups for the new key value
    /// will not find this entry.
    pub fn key_mut(&mut self) -> &mut K {
        unsafe { &mut *(self.table.key_ptr(self.gi, self.si) as *mut K) }
    }

    /// Consume the entry, returning a mutable reference to the stored key
    /// tied to the original map borrow.
    pub fn into_key(self) -> &'a mut K {
        unsafe { &mut *(self.table.key_ptr(self.gi, self.si) as *mut K) }
    }

    pub fn get(&self) -> &V {
        unsafe { &*self.table.value_ptr(self.gi, self.si) }
    }

    pub fn get_mut(&mut self) -> &mut V {
        unsafe { &mut *self.table.value_ptr(self.gi, self.si) }
    }

    pub fn into_mut(self) -> &'a mut V {
        unsafe { &mut *self.table.value_ptr(self.gi, self.si) }
    }

    pub fn get_key_value(&self) -> (&K, &V) {
        unsafe {
            (
                &*self.table.key_ptr(self.gi, self.si),
                &*self.table.value_ptr(self.gi, self.si),
            )
        }
    }

    pub fn get_key_value_mut(&mut self) -> (&mut K, &mut V) {
        // Extract both raw pointers under shared borrows of `self.table`,
        // *then* convert to `&mut` — calling `key_ptr` after constructing a
        // `&mut K` would re-borrow `self.table` and violate tree borrows.
        let (kp, vp) = unsafe {
            (
                self.table.key_ptr(self.gi, self.si) as *mut K,
                self.table.value_ptr(self.gi, self.si),
            )
        };
        unsafe { (&mut *kp, &mut *vp) }
    }

    pub fn into_key_value(self) -> (&'a mut K, &'a mut V) {
        let (kp, vp) = unsafe {
            (
                self.table.key_ptr(self.gi, self.si) as *mut K,
                self.table.value_ptr(self.gi, self.si),
            )
        };
        unsafe { (&mut *kp, &mut *vp) }
    }

    /// Replace the stored value, returning the old one.
    pub fn insert(&mut self, value: V) -> V {
        std::mem::replace(self.get_mut(), value)
    }

    /// Replace the stored key, returning the old one. Same caveat as
    /// [`key_mut`](Self::key_mut) — replacing with a key in a different
    /// equality class breaks the table invariant.
    pub fn insert_key(&mut self, key: K) -> K {
        std::mem::replace(self.key_mut(), key)
    }

    /// Remove the entry, returning its value.
    pub fn remove(self) -> V {
        self.remove_entry().1
    }

    /// Remove the entry, returning its `(key, value)` pair.
    pub fn remove_entry(self) -> (K, V) {
        unsafe {
            let k = std::ptr::read(self.table.key_ptr(self.gi, self.si));
            let v = std::ptr::read(self.table.value_ptr(self.gi, self.si));
            self.table.erase_slot(self.hash, self.gi, self.si);
            (k, v)
        }
    }
}

// ── Vacant entry methods ───────────────────────────────────────────────────

impl<'a, K, V, S, R> RawVacantEntryMut<'a, K, V, S, R>
where
    R: RawTableApi<K, V>,
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Insert `(key, value)`, hashing `key` with the map's hasher. Returns
    /// mutable references to the freshly-stored key and value.
    pub fn insert(self, key: K, value: V) -> (&'a mut K, &'a mut V) {
        let h = hash::hash_no_mix(&key, &self.map.hash_builder);
        self.insert_hashed_nocheck(h, key, value)
    }
}

impl<'a, K, V, S, R> RawVacantEntryMut<'a, K, V, S, R>
where
    R: RawTableApi<K, V>,
    K: Hash,
    S: BuildHasher,
{
    /// Insert `(key, value)` with a caller-supplied hash. The hash must equal
    /// what the map's hasher would produce; otherwise the entry will not be
    /// findable by `get`.
    pub fn insert_hashed_nocheck(
        self,
        hash: u64,
        key: K,
        value: V,
    ) -> (&'a mut K, &'a mut V) {
        self.map.table.ensure_capacity(&self.map.hash_builder);
        let (gi, si) = self.map.table.insert_no_check(hash, key, value);
        // Extract both raw pointers under shared borrows of `self.map.table`,
        // *then* convert to `&mut` — see `into_key_value` for the rationale.
        let (kp, vp) = unsafe {
            (
                self.map.table.key_ptr(gi, si) as *mut K,
                self.map.table.value_ptr(gi, si),
            )
        };
        unsafe { (&mut *kp, &mut *vp) }
    }

    /// Hash used by the builder to locate this vacant slot. Useful when the
    /// caller wants to reuse the builder's hash for insertion.
    pub fn hash(&self) -> u64 {
        self.hash
    }
}
