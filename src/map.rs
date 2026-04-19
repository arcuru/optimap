//! UnorderedFlatMap — 15-slot groups with embedded overflow byte.
//!
//! This is a type alias for `GenericMap` with `UfmLayout`.

use crate::generic_map::{self, GenericMap};
use crate::raw::group_layout::UfmLayout;
use crate::raw::overflow_table::RawTable;

pub type DefaultHashBuilder = generic_map::DefaultHashBuilder;

/// A hash map using open addressing with SIMD-accelerated group probing,
/// inspired by `boost::unordered_flat_map`.
///
/// Uses 15-slot groups with an embedded overflow byte at position 15.
/// Tombstone-free deletion with O(1) miss termination.
///
/// The maximum load factor is fixed at 0.875 and cannot be changed.
pub type UnorderedFlatMap<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, UfmLayout>>;

// Re-export entry types for backwards compatibility
pub type Entry<'a, K, V, S = DefaultHashBuilder> =
    generic_map::Entry<'a, K, V, S, RawTable<K, V, UfmLayout>>;
pub type OccupiedEntry<'a, K, V> = generic_map::OccupiedEntry<'a, K, V>;
pub type VacantEntry<'a, K, V, S = DefaultHashBuilder> =
    generic_map::VacantEntry<'a, K, V, S, RawTable<K, V, UfmLayout>>;

// Re-export iterator types
pub type Iter<'a, K, V> = generic_map::Iter<'a, K, V, RawTable<K, V, UfmLayout>>;
pub type IterMut<'a, K, V> = generic_map::IterMut<'a, K, V, RawTable<K, V, UfmLayout>>;
pub type IntoIter<K, V> = crate::raw::overflow_table::IntoIter<K, V, UfmLayout>;
pub type Keys<'a, K, V> = generic_map::Keys<'a, K, V, RawTable<K, V, UfmLayout>>;
pub type Values<'a, K, V> = generic_map::Values<'a, K, V, RawTable<K, V, UfmLayout>>;
pub type ValuesMut<'a, K, V> = generic_map::ValuesMut<'a, K, V, RawTable<K, V, UfmLayout>>;

// Keep the impl_map_trait invocation for the Map trait
crate::traits::impl_map_trait!(UnorderedFlatMap);
