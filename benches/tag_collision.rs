//! Tag/group-index collision regression bench.
//!
//! Validates the IPO/IPO64 tag/group-index collision fix from commit 8992137
//! (and the IPO64 default flip to Byte0_254 in 7815e8e).
//!
//! Two A/B comparisons are run:
//!
//! ## IPO (16-slot, AND-indexed)
//!
//! IPO uses `h & mask` for the group index, where `mask = num_groups - 1`.
//! Tag bytes that overlap with the mask correlate tag with group-index, which
//! degrades SIMD `match_byte` discrimination (more false positives → more
//! wasted key compares per probe).
//!
//! - `IPO`              — default, uses `Byte7_254` (bits 56-63). Safe at any
//!   size: the AND mask never reaches the top byte for realistic capacities.
//! - `IPO_Byte2_254`    — pre-fix default. Uses bits 16-23. Collides once
//!   `num_groups > 2^16` (≈ 1.05M slots / ~735K entries at 70% load).
//! - `IPO_Byte0_254`    — maximally collision-prone. Uses bits 0-7. Collides
//!   at any non-trivial size (the AND mask covers byte 0 once num_groups ≥ 2^1).
//!
//! ## IPO64 (64-slot, shift-indexed)
//!
//! IPO64 uses `h >> shift` where `shift = 64 - log2(num_groups)`. The group
//! index uses TOP bits, so any tag from the top byte correlates immediately.
//!
//! - `IPO64`              — default, uses `Byte0_254` (bits 0-7). Safe: the
//!   shift never reaches the bottom byte.
//! - `IPO64_Byte7_254`    — uses bits 56-63 = top byte = group index bits.
//!   Collides at any non-trivial size (shift always uses bits in byte 7).
//!
//! ## Sizes
//!
//! - `100K`  — below IPO collision threshold; both IPO variants should match.
//!   IPO64 collision is already active.
//! - `1M`    — IPO `Byte2_254` enters partial collision (num_groups ≈ 2^17,
//!   1 bit of overlap). `Byte0_254` is fully collided.
//! - `4M`    — IPO `Byte2_254` collision is significant (num_groups ≈ 2^19,
//!   3 bits of overlap). The default `Byte7_254` should clearly win.

mod bench_helpers;

use bench_helpers::*;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use optimap::matrix_types::{Byte0_254_TombMap, Byte2_254_TombMap, Byte7_254_Tomb64Map};
use optimap::{IPO64, InPlaceOverflow};

// ── Sizes ──────────────────────────────────────────────────────────────────

const LOAD_PCT: usize = 70;

/// Compute capacity that produces approximately `target_entries` at LOAD_PCT.
/// Both IPO and IPO64 round up to power-of-two groups internally; passing the
/// raw target as the capacity arg is fine — `with_capacity` does the rounding.
fn target_capacity(target_entries: usize) -> usize {
    // Slight overshoot so we don't hit the rehash boundary mid-bench.
    target_entries * 100 / LOAD_PCT
}

struct TestSize {
    name: &'static str,
    num_entries: usize,
    capacity: usize,
}

fn ipo_sizes() -> Vec<TestSize> {
    vec![
        TestSize {
            name: "100K",
            num_entries: 100_000,
            capacity: target_capacity(100_000),
        },
        TestSize {
            name: "1M",
            num_entries: 1_000_000,
            capacity: target_capacity(1_000_000),
        },
        TestSize {
            name: "4M",
            num_entries: 4_000_000,
            capacity: target_capacity(4_000_000),
        },
    ]
}

/// IPO64 collisions are visible at any non-trivial size; the largest size is
/// expensive (4M × 64-byte groups → significant memory) so we cap at 1M.
fn ipo64_sizes() -> Vec<TestSize> {
    vec![
        TestSize {
            name: "100K",
            num_entries: 100_000,
            capacity: target_capacity(100_000),
        },
        TestSize {
            name: "1M",
            num_entries: 1_000_000,
            capacity: target_capacity(1_000_000),
        },
    ]
}

// ── IPO A/B (AND indexing) ─────────────────────────────────────────────────

macro_rules! ipo_variants {
    ($helper:ident, $group:expr, $($args:expr),*) => {
        $helper::<InPlaceOverflow<u64, u64>>($group, "IPO_Byte7_254_default", $($args),*);
        $helper::<Byte2_254_TombMap<u64, u64>>($group, "IPO_Byte2_254_prefix", $($args),*);
        $helper::<Byte0_254_TombMap<u64, u64>>($group, "IPO_Byte0_254_always", $($args),*);
    };
}

fn bench_ipo_lookup_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("tag_collision/ipo/lookup_hit");
    for sz in ipo_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        ipo_variants!(bench_lookup_hit_for, &mut group, sz.name, &keys, sz.capacity);
    }
    group.finish();
}

fn bench_ipo_lookup_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("tag_collision/ipo/lookup_miss");
    for sz in ipo_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        let miss_keys = make_miss_keys(sz.num_entries);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        ipo_variants!(bench_lookup_miss_for, &mut group, sz.name, &keys, &miss_keys, sz.capacity);
    }
    group.finish();
}

fn bench_ipo_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("tag_collision/ipo/insert");
    // Skip 4M for insert — clear+refill per iteration is too slow at that size.
    for sz in ipo_sizes().into_iter().take(2) {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        ipo_variants!(bench_insert_for, &mut group, sz.name, &keys, sz.capacity);
    }
    group.finish();
}

fn bench_ipo_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("tag_collision/ipo/remove");
    for sz in ipo_sizes().into_iter().take(2) {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        ipo_variants!(bench_remove_for, &mut group, sz.name, &keys, sz.capacity);
    }
    group.finish();
}

// ── IPO64 A/B (shift indexing) ─────────────────────────────────────────────

macro_rules! ipo64_variants {
    ($helper:ident, $group:expr, $($args:expr),*) => {
        $helper::<IPO64<u64, u64>>($group, "IPO64_Byte0_254_default", $($args),*);
        $helper::<Byte7_254_Tomb64Map<u64, u64>>($group, "IPO64_Byte7_254_collide", $($args),*);
    };
}

fn bench_ipo64_lookup_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("tag_collision/ipo64/lookup_hit");
    for sz in ipo64_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        ipo64_variants!(bench_lookup_hit_for, &mut group, sz.name, &keys, sz.capacity);
    }
    group.finish();
}

fn bench_ipo64_lookup_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("tag_collision/ipo64/lookup_miss");
    for sz in ipo64_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        let miss_keys = make_miss_keys(sz.num_entries);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        ipo64_variants!(bench_lookup_miss_for, &mut group, sz.name, &keys, &miss_keys, sz.capacity);
    }
    group.finish();
}

fn bench_ipo64_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("tag_collision/ipo64/insert");
    for sz in ipo64_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        ipo64_variants!(bench_insert_for, &mut group, sz.name, &keys, sz.capacity);
    }
    group.finish();
}

fn bench_ipo64_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("tag_collision/ipo64/remove");
    for sz in ipo64_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        ipo64_variants!(bench_remove_for, &mut group, sz.name, &keys, sz.capacity);
    }
    group.finish();
}

criterion_group!(
    tag_collision,
    bench_ipo_lookup_hit,
    bench_ipo_lookup_miss,
    bench_ipo_insert,
    bench_ipo_remove,
    bench_ipo64_lookup_hit,
    bench_ipo64_lookup_miss,
    bench_ipo64_insert,
    bench_ipo64_remove,
);
criterion_main!(tag_collision);
