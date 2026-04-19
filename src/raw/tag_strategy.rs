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

// ── Tombstone tag strategies ──────────────────────────────────────────────

/// Strategy for extracting a hash tag in tombstone-based designs.
///
/// Like `TagStrategy` but tags must avoid both 0x00 (EMPTY) and 0x01 (TOMBSTONE),
/// so valid range is [2, 255].
pub trait TombstoneTag: 'static + Copy {
    /// Extract a tag byte for metadata storage.
    /// Must never return 0x00 (EMPTY) or 0x01 (TOMBSTONE).
    fn reduced_hash(h: u64) -> u8;
}

// ── LowByte254 ────────────────────────────────────────────────────────────

/// Tag from low byte, 254 distinct values (range [2, 255]).
///
/// Maps 0→2, 1→3, everything else unchanged. This is the default IPO tag
/// strategy — maximum discrimination with minimal overhead (branchless cmov).
#[derive(Clone, Copy)]
pub struct LowByte254;

impl TombstoneTag for LowByte254 {
    #[inline(always)]
    fn reduced_hash(h: u64) -> u8 {
        let low = (h & 0xFF) as u8;
        if low < 2 { low + 2 } else { low }
    }
}

// ── HighByte128 ───────────────────────────────────────────────────────────

/// Tag from bits 8-15 with high bit forced, 128 distinct values (range [128, 255]).
///
/// Matches hashbrown's design philosophy: 128 tag values from bits decorrelated
/// with the group index (which uses high bits of the hash). All values are
/// naturally ≥ 128, avoiding both 0x00 (EMPTY) and 0x01 (TOMBSTONE).
#[derive(Clone, Copy)]
pub struct HighByte128;

impl TombstoneTag for HighByte128 {
    #[inline(always)]
    fn reduced_hash(h: u64) -> u8 {
        ((h >> 8) as u8) | 0x80
    }
}
