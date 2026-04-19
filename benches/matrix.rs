//! Design space matrix benchmark.
//!
//! Tests all combinations of tag strategy × overflow strategy to find
//! optimal configurations. Each entry benchmarks hit, miss, insert, and
//! remove at medium and large sizes.

mod bench_helpers;

use bench_helpers::*;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use optimap::matrix_types::*;
use optimap::{InPlaceOverflow, Splitsies};

// ── Table geometry ─────────────────────────────────────────────────────────

const MEDIUM_CAPACITY: usize = 13_440;
const LARGE_CAPACITY: usize = 107_520;
const LOAD_PCT: usize = 70;

struct TestSize {
    name: &'static str,
    capacity: usize,
    num_entries: usize,
}

fn test_sizes() -> Vec<TestSize> {
    vec![
        TestSize {
            name: "medium",
            capacity: MEDIUM_CAPACITY,
            num_entries: entries_for_load(MEDIUM_CAPACITY, LOAD_PCT),
        },
        TestSize {
            name: "large",
            capacity: LARGE_CAPACITY,
            num_entries: entries_for_load(LARGE_CAPACITY, LOAD_PCT),
        },
    ]
}

// ── Matrix macro ───────────────────────────────────────────────────────────

macro_rules! matrix_maps {
    ($helper:ident, $group:expr, $($args:expr),*) => {
        // Baselines
        $helper::<Splitsies<u64, u64>>($group, "Lo8_8bit", $($args),*);
        $helper::<InPlaceOverflow<u64, u64>>($group, "Tombstone", $($args),*);
        // 8-bit overflow variants
        $helper::<Hi8_8bitMap<u64, u64>>($group, "Hi8_8bit", $($args),*);
        $helper::<Lo128_8bitMap<u64, u64>>($group, "Lo128_8bit", $($args),*);
        // 1-bit overflow variants
        $helper::<Lo8_1bitMap<u64, u64>>($group, "Lo8_1bit", $($args),*);
        $helper::<Hi8_1bitMap<u64, u64>>($group, "Hi8_1bit", $($args),*);
        $helper::<Lo128_1bitMap<u64, u64>>($group, "Lo128_1bit", $($args),*);
        // AND-indexed variants
        $helper::<Top128_1bitAndMap<u64, u64>>($group, "Top128_1bitAnd", $($args),*);
        $helper::<Top255_1bitAndMap<u64, u64>>($group, "Top255_1bitAnd", $($args),*);
        $helper::<Top128_8bitAndMap<u64, u64>>($group, "Top128_8bitAnd", $($args),*);
        $helper::<Top255_8bitAndMap<u64, u64>>($group, "Top255_8bitAnd", $($args),*);
        // Tombstone variants
        $helper::<Hi128_TombMap<u64, u64>>($group, "Hi128_Tomb", $($args),*);
        $helper::<Top128_TombMap<u64, u64>>($group, "Top128_Tomb", $($args),*);
        // External control
        $helper::<hashbrown::HashMap<u64, u64>>($group, "hashbrown", $($args),*);
    };
}

// ── Benchmarks ─────────────────────────────────────────────────────────────

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix/insert");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        matrix_maps!(bench_insert_for, &mut group, sz.name, &keys, sz.capacity);
    }
    group.finish();
}

fn bench_lookup_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix/lookup_hit");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        matrix_maps!(bench_lookup_hit_for, &mut group, sz.name, &keys, sz.capacity);
    }
    group.finish();
}

fn bench_lookup_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix/lookup_miss");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        let miss_keys = make_miss_keys(sz.num_entries);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        matrix_maps!(bench_lookup_miss_for, &mut group, sz.name, &keys, &miss_keys, sz.capacity);
    }
    group.finish();
}

fn bench_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix/remove");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        matrix_maps!(bench_remove_for, &mut group, sz.name, &keys, sz.capacity);
    }
    group.finish();
}

criterion_group!(matrix, bench_insert, bench_lookup_hit, bench_lookup_miss, bench_remove);
criterion_main!(matrix);
