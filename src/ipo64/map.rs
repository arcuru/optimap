//! IPO64 — tombstone-based Swiss-table design with 64-slot groups.
//!
//! This is a type alias for `GenericMap` with the IPO64 raw table.

use crate::generic_map::{self, GenericMap};
use super::raw::RawTable;

pub type DefaultHashBuilder = generic_map::DefaultHashBuilder;

/// A hash map using 64-slot cache-line groups with AVX-512/AVX2/SSE2 dispatch.
///
/// Tombstone-based deletion. Best at high-load resilience.
pub type IPO64<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V>>;

// Re-export entry types
pub type Entry<'a, K, V, S = DefaultHashBuilder> =
    generic_map::Entry<'a, K, V, S, RawTable<K, V>>;
pub type OccupiedEntry<'a, K, V> = generic_map::OccupiedEntry<'a, K, V>;
pub type VacantEntry<'a, K, V, S = DefaultHashBuilder> =
    generic_map::VacantEntry<'a, K, V, S, RawTable<K, V>>;

// Re-export iterator types
pub type Iter<'a, K, V> = generic_map::Iter<'a, K, V, RawTable<K, V>>;
pub type IterMut<'a, K, V> = generic_map::IterMut<'a, K, V, RawTable<K, V>>;
pub type IntoIter<K, V> = super::raw::IntoIter<K, V>;
pub type Keys<'a, K, V> = generic_map::Keys<'a, K, V, RawTable<K, V>>;
pub type Values<'a, K, V> = generic_map::Values<'a, K, V, RawTable<K, V>>;
pub type ValuesMut<'a, K, V> = generic_map::ValuesMut<'a, K, V, RawTable<K, V>>;

crate::traits::impl_map_trait!(IPO64);
