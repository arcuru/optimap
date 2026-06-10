//! Hash tag extraction strategies.
//!
//! A `TagStrategy` determines how the per-slot hash tag byte and the
//! per-hash overflow channel are derived from the full 64-bit hash.
//!
//! # The one pairing rule
//!
//! The tag and the group index MUST read **opposite regions** of the hash.
//! If they overlap, every key in a group shares the overlapping bits and
//! SIMD tag matches lose discrimination. So:
//!
//! - `GROUP_INDEX_REGION = High` (shift indexing, `h >> shift`) ⟹ pair with a
//!   `LowTag*` / `LowTomb` tag (reads the bottom of the hash).
//! - `GROUP_INDEX_REGION = Low` (AND indexing, `h & mask`) ⟹ pair with a
//!   `HighTag*` / `HighTomb` tag (reads the top of the hash).
//!
//! The pairing is convention, not enforced (see [`super::group_layout::HashRegion`]).
//! Names are chosen so a safe pairing reads as **opposite** words (`High` index
//! ↔ `LowTag`) and a broken one as **matching** words — the deliberately-broken
//! regression fixtures (`LowTomb_TombMap`, `HighTomb_Tomb64Map`) are exactly the
//! matching-word combos.
//!
//! # Naming
//!
//! `<Region><Kind><Width>[ChSafe][Pure]`:
//! - **Region** = `Low` (bottom of the hash) or `High` (top).
//! - **Kind** = `Tag` (overflow-bit family, 1 sentinel reserved) or `Tomb`
//!   (tombstone family, 2 sentinels EMPTY+TOMBSTONE — a different trait,
//!   [`TombstoneTag`]; `Width` is the implied 254 and omitted).
//! - **Width** = `255` or `128` (the distinct-tag-value count / reduction).
//! - **ChSafe** = tag ⊥ channel ⊥ group index: full 1/255 (or 1/128)
//!   discrimination even on channel-correlated misses. Only meaningful for
//!   8-bit-channel designs. It unifies what used to be the `Byte1` trick (low
//!   side: move the tag up a byte) and the `Ch` suffix (high side: move the
//!   channel down) — same contract from opposite ends, same one-shift cost.
//! - **Pure** = pure-Rust reduction ([`reduce_255_pure`]) instead of the
//!   feature-gated `crate::hash_tag`.
//!
//! | strategy | trait | tag bits | values | channel bits |
//! |---|---|---|---|---|
//! | `LowTag255` | TagStrategy | 0-7 | 255 | 0-2 (**overlaps tag**) |
//! | `LowTag255ChSafe` | TagStrategy | 8-15 | 255 | 0-2 |
//! | `LowTag128` | TagStrategy | 0-7, `\|1` | 128 | 0-2 (**overlaps tag**) |
//! | `LowTomb` | TombstoneTag | 0-7 | 254 | — |
//! | `HighTag255` | TagStrategy | 56-63 | 255 | 0-2 |
//! | `HighTag128` | TagStrategy + TombstoneTag | 56-63, `\|0x80` | 128 | 0-2 |
//! | `HighTomb` | TombstoneTag | 56-63 | 254 | — |
//! | `HighTag255ChSafe` | TagStrategy | 56-63 | 255 | 45-47 |
//! | `HighTag128ChSafe` | TagStrategy | 56-63, `\|0x80` | 128 | 45-47 |
//! | `LowTag255Pure` | TagStrategy | 0-7 | 255 | 0-2 |
//! | `LowTag255ChSafePure` | TagStrategy | 8-15 | 255 | 0-2 |
//! | `HighTag255Pure` | TagStrategy | 56-63 | 255 | 0-2 |
//! | `HighTag255ChSafePure` | TagStrategy | 56-63 | 255 | 45-47 |
//!
//! The non-ChSafe low strategies stay channel-correlated **by design** — they
//! are the cheap arm of an open A/B (does tag↔channel correlation cost
//! anything?), and `LowTag255` is also the real UFM / Gaps / Splitsies default.
//! `HighTag128`'s `| 0x80` forces values ≥ 0x80, which never collide with EMPTY
//! (0x00) or TOMBSTONE (0x01) — that is why it can implement both traits.

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

// ── LowTag255 ──────────────────────────────────────────────────────────────

/// Tag from byte 0 (low byte, 255 distinct values), overflow channel from bits 0-2.
///
/// Tag and overflow channel are correlated (both from low byte): a miss
/// matching the overflow channel has only 32 possible tag values
/// (bits 3-7), not the full 255. Safe with shift-based group indexing.
#[derive(Clone, Copy)]
pub struct LowTag255;

impl TagStrategy for LowTag255 {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        crate::hash_tag(h)
    }

    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << (h & 7)
    }
}

// ── LowTag255ChSafe ──────────────────────────────────────────────────────────────

/// Tag from byte 1 (bits 8-15, 255 distinct values), overflow channel from bits 0-2.
///
/// Tag and overflow channel are fully decorrelated: tag uses bits 8-15,
/// overflow uses bits 0-2. A miss matching the overflow channel has the
/// full 1/255 chance of also matching the tag, not the correlated 1/32.
/// Safe with shift-based group indexing.
#[derive(Clone, Copy)]
pub struct LowTag255ChSafe;

impl TagStrategy for LowTag255ChSafe {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        crate::hash_tag(h >> 8)
    }

    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << (h & 7)
    }
}

// ── LowTag128 ──────────────────────────────────────────────────────────────

/// Tag from byte 0 (128 distinct values, fastest), overflow channel from bits 0-2.
///
/// Uses `(h as u8) | 1` — a single OR instruction. Only 128 distinct values
/// (odd numbers 1..=255), doubling the false-match rate vs 255 values.
/// Correlated with overflow channel (same low byte). Safe with shift-based
/// group indexing.
#[derive(Clone, Copy)]
pub struct LowTag128;

impl TagStrategy for LowTag128 {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        (h as u8) | 1
    }

    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << (h & 7)
    }
}

// ── High strategies (for AND-based group indexing) ────────────────────────
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
/// Uses `((h >> 56) as u8) | 0x80` — same byte as `HighTag255`/`HighTomb`,
/// but with bit 7 forced high to guarantee non-zero (avoids EMPTY) and
/// non-one (avoids TOMBSTONE). 7 bits of entropy from bits 56-62.
///
/// Implements both `TagStrategy` (for overflow-bit designs) and
/// `TombstoneTag` (for tombstone designs). Safe with AND-based group
/// indexing because group index uses low bits. NOT safe with shift-based
/// indexing (top bits = group index → correlation).
#[derive(Clone, Copy)]
pub struct HighTag128;

impl TagStrategy for HighTag128 {
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
pub struct HighTag255;

impl TagStrategy for HighTag255 {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        crate::hash_tag(h >> 56)
    }

    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << (h & 7)
    }
}

// ── High strategies with shifted channels (AND index + 8-bit overflow) ────

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
pub struct HighTag128ChSafe;

impl TagStrategy for HighTag128ChSafe {
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
pub struct HighTag255ChSafe;

impl TagStrategy for HighTag255ChSafe {
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

// ── LowTomb ─────────────────────────────────────────────────────────────

/// Tag from byte 0 (bits 0-7), 254 distinct values (range [2, 255]).
///
/// Uses the low byte of the hash, mapping values 0→2 and 1→3 to avoid
/// EMPTY (0x00) and TOMBSTONE (0x01). The cheapest tombstone tag — no shift.
///
/// **Safety constraints:**
/// - With shift indexing (IPO64): safe at any size — bits 0-7 are
///   never reached by `h >> shift`.
/// - With AND indexing (IPO): NOT safe — the AND mask covers bits 0-7
///   for any non-trivial table, directly correlating tag with group index.
///   Use `HighTomb` for AND-indexed tombstone designs.
#[derive(Clone, Copy)]
pub struct LowTomb;

impl TombstoneTag for LowTomb {
    #[inline(always)]
    fn reduced_hash(h: u64) -> u8 {
        let b = h as u8;
        if b < 2 { b + 2 } else { b }
    }
}

// ── HighTag128 (TombstoneTag impl) ─────────────────────────────────────────

impl TombstoneTag for HighTag128 {
    #[inline(always)]
    fn reduced_hash(h: u64) -> u8 {
        ((h >> 56) as u8) | 0x80
    }
}

// ── HighTomb ─────────────────────────────────────────────────────────────

/// Tag from byte 7 (bits 56-63), 254 distinct values (range [2, 255]).
///
/// Uses bits 56-63 of the hash, mapping values 0→2 and 1→3 to avoid
/// EMPTY (0x00) and TOMBSTONE (0x01). Safe with AND-based group indexing
/// at any size. NOT safe with shift-based indexing (top bits = group
/// index → correlation).
#[derive(Clone, Copy)]
pub struct HighTomb;

impl TombstoneTag for HighTomb {
    #[inline(always)]
    fn reduced_hash(h: u64) -> u8 {
        let b = (h >> 56) as u8;
        if b < 2 { b + 2 } else { b }
    }
}

// ── Hash reduction helpers (inlined, independent of crate features) ───────

/// Map a byte to a non-zero value using the pure-Rust fallback.
///
/// `b | (b == 0) as u8` — 3 instructions on x86_64:
///   test al, al
///   sete cl
///   or  al, cl
///
/// 255 distinct values; only collision is {0, 1} → 1.
#[inline(always)]
fn reduce_255_pure(b: u8) -> u8 {
    b | (b == 0) as u8
}

/// Map a byte to a non-zero value using the `cmp; adc` asm idiom (2 instructions).
/// Falls back to [`reduce_255_pure`] on non-x86_64 or under miri.
///
/// 255 distinct values; only collision is {254, 255} → 255.
#[inline(always)]
fn reduce_255_asm(b: u8) -> u8 {
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        let result: u8;
        unsafe {
            core::arch::asm!(
                "cmp {h}, 0xFF",
                "adc {h}, 0",
                h = inout(reg_byte) b => result,
            );
        }
        result
    }
    #[cfg(not(all(target_arch = "x86_64", not(miri))))]
    {
        reduce_255_pure(b)
    }
}

// ── *Pure variants (always pure Rust, independent of features) ────────────
//
// These mirror the existing 255-value strategies but use `reduce_255_pure`
// instead of `crate::hash_tag`. This gives per-strategy control over hash
// reduction instead of the crate-wide feature flag.

/// Tag from byte 0, 255 values, always pure Rust.
#[derive(Clone, Copy)]
pub struct LowTag255Pure;

impl TagStrategy for LowTag255Pure {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        reduce_255_pure(h as u8)
    }
    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << (h & 7)
    }
}

/// Tag from byte 1 (bits 8-15), 255 values, always pure Rust.
#[derive(Clone, Copy)]
pub struct LowTag255ChSafePure;

impl TagStrategy for LowTag255ChSafePure {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        reduce_255_pure((h >> 8) as u8)
    }
    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << (h & 7)
    }
}

/// Tag from byte 7 (bits 56-63), 255 values, always pure Rust.
#[derive(Clone, Copy)]
pub struct HighTag255Pure;

impl TagStrategy for HighTag255Pure {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        reduce_255_pure((h >> 56) as u8)
    }
    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << (h & 7)
    }
}

/// Tag from byte 7 (bits 56-63), 255 values, shifted channel, always pure Rust.
#[derive(Clone, Copy)]
pub struct HighTag255ChSafePure;

impl TagStrategy for HighTag255ChSafePure {
    #[inline(always)]
    fn tag(h: u64) -> u8 {
        reduce_255_pure((h >> 56) as u8)
    }
    #[inline(always)]
    fn overflow_channel(h: u64) -> u8 {
        1u8 << ((h >> 45) & 7)
    }
}
