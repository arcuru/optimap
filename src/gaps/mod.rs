//! Gaps — Boost-style 15-slot groups with power-of-2 bucket addressing.
//!
//! Same as UnorderedFlatMap (15-slot groups, overflow byte in position 15,
//! tombstone-free deletion) but with a gap (unused 16th slot) in the bucket
//! array. This makes `bucket_ptr` use `(gi << 4) | si` instead of
//! `gi * 15 + si`, eliminating the multiply-by-15 on every operation.
//!
//! Trade-off: ~6.25% wasted memory in the bucket array (1/16 slots unused).
//! Same SIMD operations as UFM (including `& 0x7FFF` mask for 15-slot groups).

pub(crate) mod raw;
mod map;

pub use map::Gaps;
