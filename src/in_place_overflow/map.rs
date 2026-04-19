//! InPlaceOverflow — tombstone-based Swiss-table design (no overflow bytes).
//!
//! This is a type alias for `GenericMap` with the IPO raw table.

use crate::generic_map::{self, GenericMap};
use super::raw::RawTable;

pub type DefaultHashBuilder = generic_map::DefaultHashBuilder;

/// A hash map using 16-slot groups with tombstone-based deletion.
///
/// Similar to hashbrown's Swiss table but with 8-bit hash values (254 values)
/// instead of 7-bit (128). Best at lookup hit and insert.
pub type InPlaceOverflow<K, V, S = DefaultHashBuilder> =
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

crate::traits::impl_map_trait!(InPlaceOverflow);
