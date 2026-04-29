//! Hash tag extraction strategies.
//!
//! A `TagStrategy` determines how the per-slot hash tag byte and the
//! per-hash overflow channel are derived from the full 64-bit hash.
//!
//! These two values can come from the same or different bits of the hash.
//! Decorrelating them (different source bits) reduces false-positive
//! probe continuation in overflow-bit designs.
//!
//! # Naming
//!
//! Strategies are named `ByteN_VVV` where `N` is the byte index into the
//! 64-bit hash (0 = lowest, 7 = highest) and `VVV` is the count of
//! distinct tag values produced.
//!
//! # Choosing tag bits vs group-index bits
//!
//! Tag bits and group-index bits MUST come from different parts of the
//! hash. If they overlap, every key in a group shares the overlapping
//! bits, and SIMD tag matches lose discrimination.
//!
//! - Shift-based indexing (`h >> shift`) uses the **top** hash bits — pick
//!   tag bits from the **bottom** (`Byte0_*`, `Byte0_254`). `Byte1_*`
//!   exists for tag-channel decorrelation in 8-bit-channel overflow
//!   designs (channel uses bits 0-2; Byte1 sources tag from bits 8-15).
//! - AND-based indexing (`h & mask`) uses the **bottom** hash bits — pick
//!   tag bits from the **top** (`Byte7_*`).

#![allow(non_camel_case_types)]

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

// ── Byte0_255 ──────────────────────────────────────────────────────────────

/// Tag from byte 0 (low byte, 255 distinct values), overflow channel from bits 0-2.
///
/// Tag and overflow channel are correlated (both from low byte): a miss
/// matching the overflow channel has only 32 possible tag values
/// (bits 3-7), not the full 255. Safe with shift-based group indexing.
#[derive(Clone, Copy)]
pub struct Byte0_255;

impl TagStrategy for Byte0_255 {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        crate::hash_tag(h)
    }

    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << (h & 7)
    }
}

// ── Byte1_255 ──────────────────────────────────────────────────────────────

/// Tag from byte 1 (bits 8-15, 255 distinct values), overflow channel from bits 0-2.
///
/// Tag and overflow channel are fully decorrelated: tag uses bits 8-15,
/// overflow uses bits 0-2. A miss matching the overflow channel has the
/// full 1/255 chance of also matching the tag, not the correlated 1/32.
/// Safe with shift-based group indexing.
#[derive(Clone, Copy)]
pub struct Byte1_255;

impl TagStrategy for Byte1_255 {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        crate::hash_tag(h >> 8)
    }

    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << (h & 7)
    }
}

// ── Byte0_128 ──────────────────────────────────────────────────────────────

/// Tag from byte 0 (128 distinct values, fastest), overflow channel from bits 0-2.
///
/// Uses `(h as u8) | 1` — a single OR instruction. Only 128 distinct values
/// (odd numbers 1..=255), doubling the false-match rate vs 255 values.
/// Correlated with overflow channel (same low byte). Safe with shift-based
/// group indexing.
#[derive(Clone, Copy)]
pub struct Byte0_128;

impl TagStrategy for Byte0_128 {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        (h as u8) | 1
    }

    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << (h & 7)
    }
}

// ── Byte7 strategies (for AND-based group indexing) ───────────────────────
//
// AND-based group indexing uses low hash bits for the group index.
// These strategies extract tags from byte 7 (bits 56-63) — maximally
// decorrelated from the group index regardless of table size.
//
// With shift-based indexing (h >> shift), the top bits ARE the group index
// so using them for tags would be catastrophic. But with AND-based indexing,
// the top byte is completely free — same trick hashbrown uses for h2.

/// Tag from byte 7 with high bit forced, 128 values in [128, 255].
///
/// Uses `((h >> 56) as u8) | 0x80` — same byte as `Byte7_255`/`Byte7_254`,
/// but with bit 7 forced high to guarantee non-zero (avoids EMPTY) and
/// non-one (avoids TOMBSTONE). 7 bits of entropy from bits 56-62.
///
/// Implements both `TagStrategy` (for overflow-bit designs) and
/// `TombstoneTag` (for tombstone designs). Safe with AND-based group
/// indexing because group index uses low bits. NOT safe with shift-based
/// indexing (top bits = group index → correlation).
#[derive(Clone, Copy)]
pub struct Byte7_128;

impl TagStrategy for Byte7_128 {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        ((h >> 56) as u8) | 0x80
    }

    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << (h & 7)
    }
}

/// Tag from byte 7 (bits 56-63), 255 values.
///
/// Maximum discrimination from the top of the hash. Decorrelated from
/// AND-based group index (low bits). NOT safe with shift-based indexing.
#[derive(Clone, Copy)]
pub struct Byte7_255;

impl TagStrategy for Byte7_255 {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        crate::hash_tag(h >> 56)
    }

    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << (h & 7)
    }
}

// ── Byte7 strategies with shifted channels (AND index + 8-bit overflow) ───

/// Tag from byte 7 | 0x80, channel from `1 << ((h >> 45) & 7)`.
///
/// Both tag and channel use upper hash bits — fully decorrelated from
/// AND-based group indexing (low bits). This is the first strategy that
/// enables 8-bit (channeled) overflow with AND indexing. The standard
/// strategies use `1 << (h & 7)` for channels, which correlates with the
/// AND group index.
///
/// Channel uses bits 45-47, tag uses bits 56-62 with bit 7 forced.
#[derive(Clone, Copy)]
pub struct Byte7_128Ch;

impl TagStrategy for Byte7_128Ch {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        ((h >> 56) as u8) | 0x80
    }

    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << ((h >> 45) & 7)
    }
}

/// Tag from byte 7 (bits 56-63, 255 values), channel from bits 45-47.
///
/// Maximum tag discrimination + shifted channel. Both decorrelated from
/// AND group index. Channel uses bits 45-47 to avoid overlap with the
/// tag bits (56-63).
#[derive(Clone, Copy)]
pub struct Byte7_255Ch;

impl TagStrategy for Byte7_255Ch {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        crate::hash_tag(h >> 56)
    }

    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << ((h >> 45) & 7)
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

// ── Byte0_254 ─────────────────────────────────────────────────────────────

/// Tag from byte 0 (bits 0-7), 254 distinct values (range [2, 255]).
///
/// Uses the low byte of the hash, mapping values 0→2 and 1→3 to avoid
/// EMPTY (0x00) and TOMBSTONE (0x01). One shift cheaper than `Byte2_254`.
///
/// **Safety constraints:**
/// - With shift indexing (IPO64): safe at any size — bits 0-7 are
///   never reached by `h >> shift`.
/// - With AND indexing (IPO): NOT safe — the AND mask covers bits 0-7
///   for any non-trivial table, directly correlating tag with group index.
///   Use `Byte7_254` for AND-indexed tombstone designs.
#[derive(Clone, Copy)]
pub struct Byte0_254;

impl TombstoneTag for Byte0_254 {
    #[inline(always)]
    fn reduced_hash(h: u64) -> u8 {
        let b = h as u8;
        if b < 2 { b + 2 } else { b }
    }
}

// ── Byte2_254 ─────────────────────────────────────────────────────────────

/// Tag from byte 2 (bits 16-23), 254 distinct values (range [2, 255]).
///
/// Uses bits 16-23 of the hash, mapping values 0→2 and 1→3 to avoid
/// EMPTY (0x00) and TOMBSTONE (0x01). Kept as a labelled benchmark
/// variant for the IPO collision A/B test (`Byte2_254_TombMap`).
///
/// **Safety constraints:**
/// - With shift indexing (IPO64): safe at any size — bits 16-23 are
///   never reached by `h >> shift`. Prefer `Byte0_254` (one shift cheaper).
/// - With AND indexing (IPO): safe only while `num_groups ≤ 2¹⁶`. Above
///   that, the AND mask reaches into bits 16+, correlating tag bits with
///   group-index bits and degrading SIMD discrimination.
#[derive(Clone, Copy)]
pub struct Byte2_254;

impl TombstoneTag for Byte2_254 {
    #[inline(always)]
    fn reduced_hash(h: u64) -> u8 {
        let b = ((h >> 16) & 0xFF) as u8;
        if b < 2 { b + 2 } else { b }
    }
}

// ── Byte7_128 (TombstoneTag impl) ─────────────────────────────────────────

impl TombstoneTag for Byte7_128 {
    #[inline(always)]
    fn reduced_hash(h: u64) -> u8 {
        ((h >> 56) as u8) | 0x80
    }
}

// ── Byte7_254 ─────────────────────────────────────────────────────────────

/// Tag from byte 7 (bits 56-63), 254 distinct values (range [2, 255]).
///
/// Uses bits 56-63 of the hash, mapping values 0→2 and 1→3 to avoid
/// EMPTY (0x00) and TOMBSTONE (0x01). Safe with AND-based group indexing
/// at any size. NOT safe with shift-based indexing (top bits = group
/// index → correlation).
#[derive(Clone, Copy)]
pub struct Byte7_254;

impl TombstoneTag for Byte7_254 {
    #[inline(always)]
    fn reduced_hash(h: u64) -> u8 {
        let b = (h >> 56) as u8;
        if b < 2 { b + 2 } else { b }
    }
}
