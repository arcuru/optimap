//! Hash tag extraction strategies.
//!
//! A `TagStrategy` determines how the per-slot hash tag byte and the
//! per-hash overflow channel are derived from the full 64-bit hash.
//!
//! These two values can come from the same or different bits of the hash.
//! Decorrelating them (different source bits) reduces false-positive
//! probe continuation in overflow-bit designs.

/// Strategy for extracting hash tag and overflow channel from a hash value.
pub trait TagStrategy: 'static + Copy {
    /// Extract a non-zero tag byte for metadata storage.
    /// Must never return 0x00 (EMPTY sentinel).
    fn tag(h: u64) -> u8;

    /// Compute the overflow channel bitmask.
    /// For 8-bit overflow: `1 << (h & 7)` — one of 8 channels.
    /// For 1-bit overflow: this value is ignored (but still computed for API uniformity).
    fn overflow_channel(h: u64) -> u8;
}

// ── LowByte255 ─────────────────────────────────────────────────────────────

/// Tag from low byte (255 distinct values), overflow channel from bits 0-2.
///
/// This is the current default. Tag and overflow channel are correlated
/// (both from low byte): a miss matching the overflow channel has only
/// 32 possible tag values (bits 3-7), not the full 255.
#[derive(Clone, Copy)]
pub struct LowByte255;

impl TagStrategy for LowByte255 {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        crate::hash_tag(h)
    }

    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << (h & 7)
    }
}

// ── HighByte255 ────────────────────────────────────────────────────────────

/// Tag from byte 1 (bits 8-15, 255 distinct values), overflow channel from bits 0-2.
///
/// Tag and overflow channel are fully decorrelated: tag uses bits 8-15,
/// overflow uses bits 0-2. A miss matching the overflow channel has the
/// full 1/255 chance of also matching the tag, not the correlated 1/32.
#[derive(Clone, Copy)]
pub struct HighByte255;

impl TagStrategy for HighByte255 {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        crate::hash_tag(h >> 8)
    }

    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << (h & 7)
    }
}

// ── LowByte128 ─────────────────────────────────────────────────────────────

/// Tag from low byte (128 distinct values, fastest), overflow channel from bits 0-2.
///
/// Uses `(h as u8) | 1` — a single OR instruction. Only 128 distinct values
/// (odd numbers 1..=255), doubling the false-match rate vs 255 values.
/// Correlated with overflow channel (same low byte).
#[derive(Clone, Copy)]
pub struct LowByte128;

impl TagStrategy for LowByte128 {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        (h as u8) | 1
    }

    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << (h & 7)
    }
}
