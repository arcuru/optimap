//! Gaps — 15-slot groups with power-of-2 bucket stride.
//!
//! This is a type alias for `GenericMap` with `GapsLayout`.

use crate::generic_map::{self, GenericMap};
use crate::raw::group_layout::GapsLayout;
use crate::raw::overflow_table::RawTable;

pub type DefaultHashBuilder = generic_map::DefaultHashBuilder;

/// A hash map using 15-slot groups with power-of-2 bucket stride.
///
/// Like UFM but wastes 1 slot per group for faster bucket indexing (shift
/// instead of multiply-by-15). Tombstone-free deletion. Best at iteration.
pub type Gaps<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, GapsLayout>>;

// Re-export entry types
pub type Entry<'a, K, V, S = DefaultHashBuilder> =
    generic_map::Entry<'a, K, V, S, RawTable<K, V, GapsLayout>>;
pub type OccupiedEntry<'a, K, V> = generic_map::OccupiedEntry<'a, K, V>;
pub type VacantEntry<'a, K, V, S = DefaultHashBuilder> =
    generic_map::VacantEntry<'a, K, V, S, RawTable<K, V, GapsLayout>>;

// Re-export iterator types
pub type Iter<'a, K, V> = generic_map::Iter<'a, K, V, RawTable<K, V, GapsLayout>>;
pub type IterMut<'a, K, V> = generic_map::IterMut<'a, K, V, RawTable<K, V, GapsLayout>>;
pub type IntoIter<K, V> = crate::raw::overflow_table::IntoIter<K, V, GapsLayout>;
pub type Keys<'a, K, V> = generic_map::Keys<'a, K, V, RawTable<K, V, GapsLayout>>;
pub type Values<'a, K, V> = generic_map::Values<'a, K, V, RawTable<K, V, GapsLayout>>;
pub type ValuesMut<'a, K, V> = generic_map::ValuesMut<'a, K, V, RawTable<K, V, GapsLayout>>;

crate::traits::impl_map_trait!(Gaps);
