//! Experimental 16-slot group design with separate overflow array.
//!
//! This module provides an alternative `UnorderedFlatMap` implementation
//! that uses 16-slot groups (all 16 SIMD bytes are valid slot metadata)
//! with overflow bytes stored in a separate contiguous array.
//!
//! Advantages over the 15-slot design:
//! - Power-of-2 bucket addressing: `(gi << 4) | si` instead of `gi * 15 + si`
//! - No `& 0x7FFF` mask on SIMD results — all 16 bits valid
//! - Simpler capacity arithmetic: `num_groups << 4` instead of `num_groups * 15`
//!
//! Trade-off: overflow byte is in a separate array (not adjacent to metadata),
//! requiring a prefetch to hide the latency of the separate access.

mod map;
pub(crate) mod raw;

pub use map::Splitsies;
