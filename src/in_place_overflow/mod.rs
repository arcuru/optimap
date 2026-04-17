//! InPlaceOverflow — Swiss-table-style design with 8-bit hash values.
//!
//! Like Splitsies (16-slot groups, power-of-2 addressing) but without
//! the separate overflow array. Uses tombstones for deletion and EMPTY
//! slots for probe termination, similar to hashbrown's Swiss table.
//!
//! Key difference from hashbrown: 8-bit reduced hash (254 values, [2-255])
//! instead of 7-bit h2 (128 values), giving fewer false-positive SIMD matches.
//!
//! Trade-off vs Splitsies: loses tombstone-free deletion and O(1) miss
//! termination, but gains simpler memory layout (no overflow array) and
//! potentially faster lookup hit (no overflow byte access on hit path).

pub(crate) mod map;
pub(crate) mod raw;

pub use map::InPlaceOverflow;
