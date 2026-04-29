//! Core single-operation throughput benchmarks.
//!
//! Uses generic helpers via the Map trait to benchmark all designs
//! with minimal boilerplate. Adding a new design = one line per function.

mod bench_helpers;

use bench_helpers::*;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use optimap::matrix_types::*;
use optimap::{Gaps, IPO64, InPlaceOverflow, Splitsies, UnorderedFlatMap};

// ── Table geometry ──────────────────────────────────────────────────────────

const GROUP_SIZE: usize = 15;

fn entries_for_load(capacity: usize, load_pct: usize) -> usize {
    let min_slots = (capacity * 8).div_ceil(7);
    let min_groups = min_slots.div_ceil(GROUP_SIZE);
    let mut num_groups = 1;
    while num_groups < min_groups {
        num_groups *= 2;
    }
    let total_slots = num_groups * GROUP_SIZE;
    total_slots * load_pct / 100
}

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

// ── Macro to run a bench helper for all designs ��────────────────────────────

macro_rules! all_maps {
    ($helper:ident, $group:expr, $($args:expr),*) => {
        // Original designs
        $helper::<UnorderedFlatMap<u64, u64>>($group, "UFM", $($args),*);
        $helper::<Gaps<u64, u64>>($group, "Gaps", $($args),*);
        $helper::<Splitsies<u64, u64>>($group, "Splitsies", $($args),*);
        $helper::<InPlaceOverflow<u64, u64>>($group, "IPO", $($args),*);
        $helper::<IPO64<u64, u64>>($group, "IPO64", $($args),*);
        $helper::<hashbrown::HashMap<u64, u64>>($group, "hashbrown", $($args),*);
        $helper::<OptiMapBench<u64, u64>>($group, "OptiMap", $($args),*);
        // Matrix variants
        $helper::<Byte1_8bitMap<u64, u64>>($group, "Byte1_8bit", $($args),*);
        $helper::<Byte0_128_8bitMap<u64, u64>>($group, "Byte0_128_8bit", $($args),*);
        $helper::<Byte0_1bitMap<u64, u64>>($group, "Byte0_1bit", $($args),*);
        $helper::<Byte0_128_1bitMap<u64, u64>>($group, "Byte0_128_1bit", $($args),*);
        // AND-indexed variants
        $helper::<Byte7_128_1bitAndMap<u64, u64>>($group, "Byte7_128_1bitAnd", $($args),*);
        $helper::<Byte7_128_8bitAndMap<u64, u64>>($group, "Byte7_128_8bitAnd", $($args),*);
        // Tombstone variant
        $helper::<Byte2_254_TombMap<u64, u64>>($group, "Byte2_254_Tomb", $($args),*);
        $helper::<Byte7_128_TombMap<u64, u64>>($group, "Byte7_128_Tomb", $($args),*);
        // IPO64 tombstone variants
        $helper::<Byte7_254_Tomb64Map<u64, u64>>($group, "Byte7_254_Tomb64", $($args),*);
        // SoA variants
        $helper::<optimap::SoaMap<u64, u64>>($group, "SoaMap", $($args),*);
        $helper::<optimap::soa::SoaByte0_128<u64, u64>>($group, "SoaByte0_128", $($args),*);
        $helper::<optimap::soa::SoaByte1<u64, u64>>($group, "SoaByte1", $($args),*);
        $helper::<optimap::soa::SoaByte0_1bit<u64, u64>>($group, "SoaByte0_1bit", $($args),*);
        $helper::<optimap::soa::SoaByte0_128_1bit<u64, u64>>($group, "SoaByte0_128_1bit", $($args),*);
        $helper::<optimap::soa::SoaByte7_128And<u64, u64>>($group, "SoaByte7_128And", $($args),*);
        $helper::<optimap::soa::SoaByte7_255And<u64, u64>>($group, "SoaByte7_255And", $($args),*);
        $helper::<optimap::soa::SoaByte7_128_8bitAnd<u64, u64>>($group, "SoaByte7_128_8bitAnd", $($args),*);
        $helper::<optimap::soa::SoaByte7_255_8bitAnd<u64, u64>>($group, "SoaByte7_255_8bitAnd", $($args),*);
        // SoA tombstone variants
        $helper::<optimap::soa::SoaIpo<u64, u64>>($group, "SoaIpo", $($args),*);
        $helper::<optimap::soa::SoaByte7_128_Tomb<u64, u64>>($group, "SoaByte7_128_Tomb", $($args),*);
    };
}

// ── Throughput: Insert ──────────────────────────────────────────────────────

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/insert");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        all_maps!(bench_insert_for, &mut group, sz.name, &keys, sz.capacity);
    }
    group.finish();
}

// ── Throughput: Lookup Hit ──────────────────────────────────────────────────

fn bench_lookup_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/lookup_hit");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        all_maps!(
            bench_lookup_hit_for,
            &mut group,
            sz.name,
            &keys,
            sz.capacity
        );
    }
    group.finish();
}

// ── Throughput: Lookup Miss ─────────────────────────────────────────────────

fn bench_lookup_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/lookup_miss");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        let miss_keys = make_miss_keys(sz.num_entries);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        all_maps!(
            bench_lookup_miss_for,
            &mut group,
            sz.name,
            &keys,
            &miss_keys,
            sz.capacity
        );
    }
    group.finish();
}

// ── Throughput: Remove ────────────────────────────��─────────────────────────

fn bench_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/remove");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        all_maps!(bench_remove_for, &mut group, sz.name, &keys, sz.capacity);
    }
    group.finish();
}

// ── Throughput: Iteration ────────────────────────────────────────────────────

fn bench_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/iteration");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        all_maps!(bench_iteration_for, &mut group, sz.name, &keys, sz.capacity);
    }
    group.finish();
}

// ── Throughput: Size Scaling ────────────────────────────────────────────────

fn bench_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/size_scaling");

    for &n in &[256, 4_000, 32_000, 256_000, 2_000_000, 20_000_000] {
        let keys = make_random_keys(n, 42);
        let miss_keys = make_miss_keys(n);
        let label = n.to_string();
        group.throughput(Throughput::Elements(n as u64));
        if n >= 2_000_000 {
            group.sample_size(10);
        }

        // Hit — all designs
        bench_lookup_hit_for::<UnorderedFlatMap<u64, u64>>(&mut group, "UFM_hit", &label, &keys, n);
        bench_lookup_hit_for::<Splitsies<u64, u64>>(&mut group, "Splitsies_hit", &label, &keys, n);
        bench_lookup_hit_for::<InPlaceOverflow<u64, u64>>(&mut group, "IPO_hit", &label, &keys, n);
        bench_lookup_hit_for::<IPO64<u64, u64>>(&mut group, "IPO64_hit", &label, &keys, n);
        bench_lookup_hit_for::<hashbrown::HashMap<u64, u64>>(
            &mut group,
            "hashbrown_hit",
            &label,
            &keys,
            n,
        );
        bench_lookup_hit_for::<OptiMapBench<u64, u64>>(&mut group, "OptiMap_hit", &label, &keys, n);

        // Miss — IPO variants + hashbrown + OptiMap
        bench_lookup_miss_for::<InPlaceOverflow<u64, u64>>(
            &mut group, "IPO_miss", &label, &keys, &miss_keys, n,
        );
        bench_lookup_miss_for::<IPO64<u64, u64>>(
            &mut group,
            "IPO64_miss",
            &label,
            &keys,
            &miss_keys,
            n,
        );
        bench_lookup_miss_for::<hashbrown::HashMap<u64, u64>>(
            &mut group,
            "hashbrown_miss",
            &label,
            &keys,
            &miss_keys,
            n,
        );
        bench_lookup_miss_for::<OptiMapBench<u64, u64>>(
            &mut group,
            "OptiMap_miss",
            &label,
            &keys,
            &miss_keys,
            n,
        );
    }
    group.finish();
}

criterion_group!(
    throughput,
    bench_insert,
    bench_lookup_hit,
    bench_lookup_miss,
    bench_remove,
    bench_iteration,
    bench_size_scaling,
);
criterion_main!(throughput);
