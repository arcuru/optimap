//! Allocation, growth, clone, and construction cost benchmarks.
//!
//! These benchmarks intentionally include allocation overhead — they measure
//! costs that happen once per table lifetime. Use these to understand the
//! cost of creating, growing, and copying hash maps.

mod bench_helpers;

use bench_helpers::*;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

use optimap::matrix_types::*;
use optimap::{Gaps, IPO64, InPlaceOverflow, Splitsies, UnorderedFlatMap};

macro_rules! all_maps {
    ($helper:ident, $group:expr, $($args:expr),*) => {
        $helper::<UnorderedFlatMap<u64, u64>>($group, "UFM", $($args),*);
        $helper::<Gaps<u64, u64>>($group, "Gaps", $($args),*);
        $helper::<Splitsies<u64, u64>>($group, "Splitsies", $($args),*);
        $helper::<InPlaceOverflow<u64, u64>>($group, "IPO", $($args),*);
        $helper::<IPO64<u64, u64>>($group, "IPO64", $($args),*);
        $helper::<hashbrown::HashMap<u64, u64>>($group, "hashbrown", $($args),*);
        $helper::<OptiMapBench<u64, u64>>($group, "OptiMap", $($args),*);
        // Matrix variants
        $helper::<LowTag255ChSafe_8bitMap<u64, u64>>($group, "LowTag255ChSafe_8bit", $($args),*);
        $helper::<LowTag128_8bitMap<u64, u64>>($group, "LowTag128_8bit", $($args),*);
        $helper::<LowTag255_1bitMap<u64, u64>>($group, "LowTag255_1bit", $($args),*);
        $helper::<LowTag128_1bitMap<u64, u64>>($group, "LowTag128_1bit", $($args),*);
        // AND-indexed variants
        $helper::<HighTag128_1bitAndMap<u64, u64>>($group, "HighTag128_1bitAnd", $($args),*);
        $helper::<HighTag128_8bitAndMap<u64, u64>>($group, "HighTag128_8bitAnd", $($args),*);
        // Tombstone variant
        $helper::<HighTag128_TombMap<u64, u64>>($group, "HighTag128_Tomb", $($args),*);
        // IPO64 tombstone variants
        $helper::<HighTomb_Tomb64Map<u64, u64>>($group, "HighTomb_Tomb64", $($args),*);
        // SoA variants
        $helper::<optimap::SoaMap<u64, u64>>($group, "SoaMap", $($args),*);
        $helper::<optimap::soa::SoaLowTag128<u64, u64>>($group, "SoaLowTag128", $($args),*);
        $helper::<optimap::soa::SoaLowTag255ChSafe<u64, u64>>($group, "SoaLowTag255ChSafe", $($args),*);
        $helper::<optimap::soa::SoaLowTag255_1bit<u64, u64>>($group, "SoaLowTag255_1bit", $($args),*);
        $helper::<optimap::soa::SoaLowTag128_1bit<u64, u64>>($group, "SoaLowTag128_1bit", $($args),*);
        $helper::<optimap::soa::SoaByte7_128And<u64, u64>>($group, "SoaByte7_128And", $($args),*);
        $helper::<optimap::soa::SoaByte7_255And<u64, u64>>($group, "SoaByte7_255And", $($args),*);
        $helper::<optimap::soa::SoaHighTag128_8bitAnd<u64, u64>>($group, "SoaHighTag128_8bitAnd", $($args),*);
        $helper::<optimap::soa::SoaHighTag255_8bitAnd<u64, u64>>($group, "SoaHighTag255_8bitAnd", $($args),*);
        $helper::<optimap::soa::SoaIpo<u64, u64>>($group, "SoaIpo", $($args),*);
        $helper::<optimap::soa::SoaHighTag128_Tomb<u64, u64>>($group, "SoaHighTag128_Tomb", $($args),*);
    };
}

// ── Grow from empty (no pre-allocation) ─────────────────────────────────────

fn bench_grow_from_empty(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction/grow_from_empty");

    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        let keys = make_random_keys(n, 42);
        group.throughput(Throughput::Elements(n as u64));
        if n >= 1_000_000 {
            group.sample_size(10);
        }

        all_maps!(bench_grow_for, &mut group, &keys, n);
    }
    group.finish();
}

// ── Insert with pre-allocation (cold pages) ─────────────────────────────────

fn bench_insert_with_capacity(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction/with_capacity");

    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        let keys = make_random_keys(n, 42);
        group.throughput(Throughput::Elements(n as u64));
        if n >= 1_000_000 {
            group.sample_size(10);
        }

        all_maps!(bench_with_capacity_for, &mut group, &keys, n);
    }
    group.finish();
}

// ── Clone ───────────────────────────────────────────────────────────────────

fn bench_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction/clone");

    for &n in &[1_000, 100_000, 1_000_000] {
        let keys = make_random_keys(n, 42);
        if n >= 1_000_000 {
            group.sample_size(10);
        }

        all_maps!(bench_clone_for, &mut group, &keys, n);
    }
    group.finish();
}

// ── FromIterator (collect) ──────────────────────────────────────────────────

macro_rules! bench_from_iter_for {
    ($group:expr, $name:expr, $map_type:ty, $pairs:expr, $n:expr) => {
        $group.bench_with_input(BenchmarkId::new($name, $n), $pairs, |b, pairs| {
            b.iter(|| {
                let map: $map_type = pairs.iter().copied().collect();
                black_box(map.len());
            });
        });
    };
}

fn bench_from_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction/from_iter");

    for &n in &[10_000, 100_000] {
        let pairs: Vec<(u64, u64)> = {
            let mut rng = Sfc64::new(42);
            (0..n).map(|i| (rng.next_u64(), i as u64)).collect()
        };
        group.throughput(Throughput::Elements(n as u64));

        bench_from_iter_for!(&mut group, "UFM", UnorderedFlatMap<u64, u64>, &pairs, n);
        bench_from_iter_for!(&mut group, "Gaps", Gaps<u64, u64>, &pairs, n);
        bench_from_iter_for!(&mut group, "Splitsies", Splitsies<u64, u64>, &pairs, n);
        bench_from_iter_for!(&mut group, "IPO", InPlaceOverflow<u64, u64>, &pairs, n);
        bench_from_iter_for!(&mut group, "IPO64", IPO64<u64, u64>, &pairs, n);
        bench_from_iter_for!(&mut group, "hashbrown", hashbrown::HashMap<u64, u64>, &pairs, n);
        bench_from_iter_for!(&mut group, "OptiMap", optimap::OptiMap<u64, u64>, &pairs, n);
    }
    group.finish();
}

criterion_group!(
    construction,
    bench_grow_from_empty,
    bench_insert_with_capacity,
    bench_clone,
    bench_from_iter,
);
criterion_main!(construction);
