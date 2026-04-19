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

// ── Top-bit tag strategies (for AND-based group indexing) ─────────────────
//
// AND-based group indexing uses low hash bits for the group index.
// These strategies extract tags from the TOP bits (57+), which are maximally
// decorrelated from the group index regardless of table size.
//
// With shift-based indexing (h >> shift), the top bits ARE the group index
// so using them for tags would be catastrophic. But with AND-based indexing,
// top bits are completely free — same trick hashbrown uses for h2.

/// Tag from top 7 bits with high bit forced, 128 values in [128, 255].
///
/// Uses `(h >> 57) | 0x80` — the same bits as hashbrown's h2 function.
/// Safe with AND-based group indexing because group index uses low bits.
/// NOT safe with shift-based indexing (top bits = group index → correlation).
#[derive(Clone, Copy)]
pub struct TopTag128;

impl TagStrategy for TopTag128 {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        ((h >> 57) as u8) | 0x80
    }

    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << (h & 7)
    }
}

/// Tag from top byte (bits 56-63), 255 values.
///
/// Maximum discrimination from the top of the hash. Decorrelated from
/// AND-based group index (low bits). NOT safe with shift-based indexing.
#[derive(Clone, Copy)]
pub struct TopTag255;

impl TagStrategy for TopTag255 {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        crate::hash_tag(h >> 56)
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

/// Tag from bits 16-23, 254 distinct values (range [2, 255]).
///
/// Uses bits 16-23 of the hash, mapping values 0→2 and 1→3 to avoid
/// EMPTY (0x00) and TOMBSTONE (0x01). Bits 16+ are fully decorrelated
/// from the group index (which uses low bits via AND).
#[derive(Clone, Copy)]
pub struct LowByte254;

impl TombstoneTag for LowByte254 {
    #[inline(always)]
    fn reduced_hash(h: u64) -> u8 {
        let b = ((h >> 16) & 0xFF) as u8;
        if b < 2 { b + 2 } else { b }
    }
}

// ── HighByte128 ───────────────────────────────────────────────────────────

/// Tag from bits 24-30 with high bit forced, 128 distinct values (range [128, 255]).
///
/// Uses bits 24-30, fully decorrelated from the group index (low bits via AND).
/// The `| 0x80` forces values into [128, 255], avoiding EMPTY (0x00) and
/// TOMBSTONE (0x01). Only 7 bits of entropy.
#[derive(Clone, Copy)]
pub struct HighByte128;

impl TombstoneTag for HighByte128 {
    #[inline(always)]
    fn reduced_hash(h: u64) -> u8 {
        ((h >> 24) as u8) | 0x80
    }
}

// ── TopByte128 ────────────────────────────────────────────────────────────

/// Tag from bits 25-31, 128 distinct values (range [2, 129]).
///
/// Inspired by hashbrown's h2 (`h >> 57`), but adjusted for our group
/// index scheme. hashbrown uses low bits for group_index so top bits are
/// free for tags. We use HIGH bits for group_index (`h >> shift`), so
/// bits 57-63 overlap with group_index — using them as tags causes every
/// entry in a group to have the same tag, degrading to linear scan.
///
/// Instead we use bits 25-31 (middle of the hash), which are decorrelated
/// from both the low byte (used by other tag strategies) and the high
/// bits (used by group_index). Still 128 values, still a single shift+add.
#[derive(Clone, Copy)]
pub struct TopByte128;

impl TombstoneTag for TopByte128 {
    #[inline(always)]
    fn reduced_hash(h: u64) -> u8 {
        ((h >> 25) as u8) | 0x80
    }
}
