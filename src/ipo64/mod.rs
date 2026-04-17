//! IPO64 — Swiss-table-style design with 64-slot groups (one cache line of metadata).
//!
//! Like InPlaceOverflow but with 64 slots per group instead of 16.
//! Each group's metadata occupies one 64-byte cache line, so a single
//! cache fetch gives all metadata for 64 slots.

pub(crate) mod map;
pub(crate) mod raw;

pub use map::IPO64;
