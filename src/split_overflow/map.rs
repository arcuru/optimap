//! Splitsies — 16-slot groups with separate overflow array.
//!
//! This is a type alias for `GenericMap` with `SplitsiesLayout`.

use crate::generic_map::{self, GenericMap};
use crate::raw::group_layout::SplitsiesLayout;
use crate::raw::overflow_table::RawTable;

pub type DefaultHashBuilder = generic_map::DefaultHashBuilder;

/// A hash map using 16-slot groups with a separate overflow byte array.
///
/// All 16 SIMD bits are usable (no embedded overflow byte taking a slot).
/// Power-of-2 bucket addressing. Tombstone-free deletion.
pub type Splitsies<K, V, S = DefaultHashBuilder> =
    GenericMap<K, V, S, RawTable<K, V, SplitsiesLayout>>;

// Re-export entry types
pub type Entry<'a, K, V, S = DefaultHashBuilder> =
    generic_map::Entry<'a, K, V, S, RawTable<K, V, SplitsiesLayout>>;
pub type OccupiedEntry<'a, K, V> = generic_map::OccupiedEntry<'a, K, V>;
pub type VacantEntry<'a, K, V, S = DefaultHashBuilder> =
    generic_map::VacantEntry<'a, K, V, S, RawTable<K, V, SplitsiesLayout>>;

// Re-export iterator types
pub type Iter<'a, K, V> = generic_map::Iter<'a, K, V, RawTable<K, V, SplitsiesLayout>>;
pub type IterMut<'a, K, V> = generic_map::IterMut<'a, K, V, RawTable<K, V, SplitsiesLayout>>;
pub type IntoIter<K, V> = crate::raw::overflow_table::IntoIter<K, V, SplitsiesLayout>;
pub type Keys<'a, K, V> = generic_map::Keys<'a, K, V, RawTable<K, V, SplitsiesLayout>>;
pub type Values<'a, K, V> = generic_map::Values<'a, K, V, RawTable<K, V, SplitsiesLayout>>;
pub type ValuesMut<'a, K, V> = generic_map::ValuesMut<'a, K, V, RawTable<K, V, SplitsiesLayout>>;

crate::traits::impl_map_trait!(Splitsies);
