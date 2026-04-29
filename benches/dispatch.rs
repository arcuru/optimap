//! OptiMap dispatch overhead micro-bench.
//!
//! Isolates enum-dispatch cost vs the raw backend for every `OptiMap` variant.
//! The recent inlining work (commits a66f50b, ae1e4d5) was meant to close the
//! gap on hot ops; this bench ensures it stays closed.
//!
//! Pairs measured (each row in the criterion plot):
//!
//! | Raw backend          | OptiMap-pinned wrapper      |
//! |----------------------|-----------------------------|
//! | `UnorderedFlatMap`   | `OptiMap_Ufm`               |
//! | `Splitsies`          | `OptiMap_Splitsies`         |
//! | `InPlaceOverflow`    | `OptiMap_Ipo`               |
//! | `Gaps`               | `OptiMap_Gaps`              |
//! | `IPO64`              | `OptiMap_Ipo64`             |
//!
//! Same key set, same capacity, fixed backend pinning, identical helpers via
//! the `Map` trait → the *only* delta is the enum match in `OptiMap::get` /
//! `insert` / `remove`.
//!
//! Hot ops (expected delta: ≤ 5% per the inlining commit message):
//! - lookup hit
//! - lookup miss
//! - insert
//! - remove
//!
//! Cold-ish op kept for visibility (memory: `Box<dyn Iterator>` causes ~2x
//! slowdown):
//! - iter

mod bench_helpers;

use bench_helpers::*;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use optimap::{Gaps, IPO64, InPlaceOverflow, Splitsies, UnorderedFlatMap};

// ── Sizes ──────────────────────────────────────────────────────────────────

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

// ── Pair macro ─────────────────────────────────────────────────────────────

/// Run `$helper` for each (raw, OptiMap-pinned) pair.
///
/// The label is repeated so the criterion plot has a clear "raw vs wrapped"
/// pair per backend. Each pair lets you read the dispatch delta directly.
macro_rules! dispatch_pairs {
    ($helper:ident, $group:expr, $($args:expr),*) => {
        // Pair: UFM
        $helper::<UnorderedFlatMap<u64, u64>>($group, "Ufm_raw",     $($args),*);
        $helper::<OptiMapBenchBackend<u64, u64, OptiUfm>>($group, "Ufm_opti",    $($args),*);
        // Pair: Splitsies
        $helper::<Splitsies<u64, u64>>($group, "Splitsies_raw",      $($args),*);
        $helper::<OptiMapBenchBackend<u64, u64, OptiSplitsies>>($group, "Splitsies_opti", $($args),*);
        // Pair: IPO  (the primary case the inlining work targeted)
        $helper::<InPlaceOverflow<u64, u64>>($group, "Ipo_raw",      $($args),*);
        $helper::<OptiMapBenchBackend<u64, u64, OptiIpo>>($group, "Ipo_opti",    $($args),*);
        // Pair: Gaps
        $helper::<Gaps<u64, u64>>($group, "Gaps_raw",                $($args),*);
        $helper::<OptiMapBenchBackend<u64, u64, OptiGaps>>($group, "Gaps_opti",  $($args),*);
        // Pair: IPO64
        $helper::<IPO64<u64, u64>>($group, "Ipo64_raw",              $($args),*);
        $helper::<OptiMapBenchBackend<u64, u64, OptiIpo64>>($group, "Ipo64_opti", $($args),*);
    };
}

// ── Hot-op benches ─────────────────────────────────────────────────────────

fn bench_lookup_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("dispatch/lookup_hit");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        dispatch_pairs!(bench_lookup_hit_for, &mut group, sz.name, &keys, sz.capacity);
    }
    group.finish();
}

fn bench_lookup_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("dispatch/lookup_miss");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        let miss_keys = make_miss_keys(sz.num_entries);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        dispatch_pairs!(bench_lookup_miss_for, &mut group, sz.name, &keys, &miss_keys, sz.capacity);
    }
    group.finish();
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("dispatch/insert");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        dispatch_pairs!(bench_insert_for, &mut group, sz.name, &keys, sz.capacity);
    }
    group.finish();
}

fn bench_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("dispatch/remove");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        dispatch_pairs!(bench_remove_for, &mut group, sz.name, &keys, sz.capacity);
    }
    group.finish();
}

fn bench_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("dispatch/iter");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        dispatch_pairs!(bench_iteration_for, &mut group, sz.name, &keys, sz.capacity);
    }
    group.finish();
}

criterion_group!(
    dispatch,
    bench_lookup_hit,
    bench_lookup_miss,
    bench_insert,
    bench_remove,
    bench_iter,
);
criterion_main!(dispatch);
