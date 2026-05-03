//! Load-factor sweep benchmarks.
//!
//! Tests lookup hit, miss, insert, and mixed performance across
//! a range of load factors from ~45% to ~87%.
//!
//! Methodology: allocate a table with a fixed number of groups,
//! fill it to a target load factor, then benchmark operations.
//! This isolates load factor from table size effects.

mod bench_helpers;

use bench_helpers::*;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use optimap::{InPlaceOverflow, Map, Splitsies, UnorderedFlatMap};

// ── Load factor helpers ─────────────────────────────────────────────────────

fn compute_load_info(capacity: usize, load_pct: usize) -> (usize, f64) {
    let min_slots = (capacity * 8).div_ceil(7);
    let min_groups = min_slots.div_ceil(15);
    let mut num_groups = 1;
    while num_groups < min_groups {
        num_groups *= 2;
    }
    let total_slots = num_groups * 15;
    let num_entries = total_slots * load_pct / 100;
    let actual_lf = num_entries as f64 / total_slots as f64 * 100.0;
    (num_entries, actual_lf)
}

/// Generic helper: benchmark lookup hit at a specific load level with label.
fn bench_lf_hit_for<M: Map<u64, u64>>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    name: &str,
    label: &str,
    capacity: usize,
    num_entries: usize,
    ops: u64,
    seed: u64,
) {
    let (map, keys) = build_map_at_load::<M>(capacity, num_entries, seed);
    group.bench_with_input(
        criterion::BenchmarkId::new(format!("{name}_{label}"), num_entries),
        &keys,
        |b, keys| {
            b.iter(|| {
                let mut sum = 0u64;
                for i in 0..ops as usize {
                    sum = sum.wrapping_add(*map.get(&keys[i % keys.len()]).unwrap_or(&0));
                }
                criterion::black_box(sum);
            });
        },
    );
}

fn bench_lf_miss_for<M: Map<u64, u64>>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    name: &str,
    label: &str,
    capacity: usize,
    num_entries: usize,
    miss_keys: &[u64],
    seed: u64,
) {
    let (map, _) = build_map_at_load::<M>(capacity, num_entries, seed);
    group.bench_with_input(
        criterion::BenchmarkId::new(format!("{name}_{label}"), num_entries),
        miss_keys,
        |b, miss_keys| {
            b.iter(|| {
                let mut count = 0u64;
                for k in miss_keys {
                    if map.get(k).is_some() {
                        count += 1;
                    }
                }
                criterion::black_box(count);
            });
        },
    );
}

fn bench_lf_mixed_for<M: Map<u64, u64>>(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    name: &str,
    label: &str,
    capacity: usize,
    num_entries: usize,
    op_keys: &[(u8, u64)],
    seed: u64,
) {
    let (mut map, _) = build_map_at_load::<M>(capacity, num_entries, seed);
    group.bench_with_input(
        criterion::BenchmarkId::new(format!("{name}_{label}"), num_entries),
        op_keys,
        |b, ops| {
            b.iter(|| {
                let mut checksum = 0u64;
                for &(op, key) in ops {
                    match op {
                        0..=4 => {
                            map.insert(key, key);
                        }
                        5..=7 => {
                            if let Some(&v) = map.get(&key) {
                                checksum = checksum.wrapping_add(v);
                            }
                        }
                        _ => {
                            map.remove(&key);
                        }
                    }
                }
                criterion::black_box(checksum);
            });
        },
    );
}

// ── Benchmark: Lookup hit at varying load factors ───────────────────────

fn bench_lookup_hit_by_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_factor_hit");

    let capacity = 100_000;
    let ops = 100_000u64;

    for load_pct in [45, 55, 65, 75, 85] {
        let (num_entries, actual_lf) = compute_load_info(capacity, load_pct);
        group.throughput(Throughput::Elements(ops));
        let label = format!("{:.0}pct", actual_lf);

        bench_lf_hit_for::<UnorderedFlatMap<u64, u64>>(
            &mut group,
            "UFM",
            &label,
            capacity,
            num_entries,
            ops,
            42,
        );
        bench_lf_hit_for::<Splitsies<u64, u64>>(
            &mut group,
            "Splitsies",
            &label,
            capacity,
            num_entries,
            ops,
            42,
        );
        bench_lf_hit_for::<InPlaceOverflow<u64, u64>>(
            &mut group,
            "IPO",
            &label,
            capacity,
            num_entries,
            ops,
            42,
        );
        bench_lf_hit_for::<hashbrown::HashMap<u64, u64>>(
            &mut group,
            "hashbrown",
            &label,
            capacity,
            num_entries,
            ops,
            42,
        );
        bench_lf_hit_for::<OptiMapBench<u64, u64>>(
            &mut group,
            "OptiMap",
            &label,
            capacity,
            num_entries,
            ops,
            42,
        );
        bench_lf_hit_for::<optimap::SoaMap<u64, u64>>(
            &mut group,
            "SoaMap",
            &label,
            capacity,
            num_entries,
            ops,
            42,
        );
    }
    group.finish();
}

// ── Benchmark: Lookup miss at varying load factors ──────────────────────

fn bench_lookup_miss_by_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_factor_miss");

    let capacity = 100_000;
    let ops = 100_000u64;

    let mut miss_rng = Sfc64::new(9999);
    let miss_keys: Vec<u64> = (0..ops as usize).map(|_| miss_rng.next_u64()).collect();

    for load_pct in [45, 55, 65, 75, 85] {
        let (num_entries, actual_lf) = compute_load_info(capacity, load_pct);
        group.throughput(Throughput::Elements(ops));
        let label = format!("{:.0}pct", actual_lf);

        bench_lf_miss_for::<UnorderedFlatMap<u64, u64>>(
            &mut group,
            "UFM",
            &label,
            capacity,
            num_entries,
            &miss_keys,
            42,
        );
        bench_lf_miss_for::<Splitsies<u64, u64>>(
            &mut group,
            "Splitsies",
            &label,
            capacity,
            num_entries,
            &miss_keys,
            42,
        );
        bench_lf_miss_for::<InPlaceOverflow<u64, u64>>(
            &mut group,
            "IPO",
            &label,
            capacity,
            num_entries,
            &miss_keys,
            42,
        );
        bench_lf_miss_for::<hashbrown::HashMap<u64, u64>>(
            &mut group,
            "hashbrown",
            &label,
            capacity,
            num_entries,
            &miss_keys,
            42,
        );
        bench_lf_miss_for::<OptiMapBench<u64, u64>>(
            &mut group,
            "OptiMap",
            &label,
            capacity,
            num_entries,
            &miss_keys,
            42,
        );
        bench_lf_miss_for::<optimap::SoaMap<u64, u64>>(
            &mut group,
            "SoaMap",
            &label,
            capacity,
            num_entries,
            &miss_keys,
            42,
        );
    }
    group.finish();
}

// ── Benchmark: Mixed hit+miss at varying load factors ───────────────────

fn bench_mixed_by_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_factor_mixed");

    let capacity = 100_000;
    let ops = 100_000u64;

    for load_pct in [45, 55, 65, 75, 85] {
        let (num_entries, actual_lf) = compute_load_info(capacity, load_pct);
        group.throughput(Throughput::Elements(ops));
        let label = format!("{:.0}pct", actual_lf);

        // Build keys for this load level to generate op sequence
        let (_, ours_keys) =
            build_map_at_load::<UnorderedFlatMap<u64, u64>>(capacity, num_entries, 42);

        let mut mix_rng = Sfc64::new(777);
        let miss_keys: Vec<u64> = (0..ops as usize).map(|_| mix_rng.next_u64()).collect();

        let op_keys: Vec<(u8, u64)> = {
            let mut rng = Sfc64::new(555);
            (0..ops as usize)
                .map(|i| {
                    let op = (rng.next_u64() % 10) as u8;
                    let key = if op < 8 {
                        ours_keys[i % ours_keys.len()]
                    } else {
                        miss_keys[i % miss_keys.len()]
                    };
                    (op, key)
                })
                .collect()
        };

        bench_lf_mixed_for::<UnorderedFlatMap<u64, u64>>(
            &mut group,
            "UFM",
            &label,
            capacity,
            num_entries,
            &op_keys,
            42,
        );
        bench_lf_mixed_for::<Splitsies<u64, u64>>(
            &mut group,
            "Splitsies",
            &label,
            capacity,
            num_entries,
            &op_keys,
            42,
        );
        bench_lf_mixed_for::<InPlaceOverflow<u64, u64>>(
            &mut group,
            "IPO",
            &label,
            capacity,
            num_entries,
            &op_keys,
            42,
        );
        bench_lf_mixed_for::<hashbrown::HashMap<u64, u64>>(
            &mut group,
            "hashbrown",
            &label,
            capacity,
            num_entries,
            &op_keys,
            42,
        );
        bench_lf_mixed_for::<OptiMapBench<u64, u64>>(
            &mut group,
            "OptiMap",
            &label,
            capacity,
            num_entries,
            &op_keys,
            42,
        );
        bench_lf_mixed_for::<optimap::SoaMap<u64, u64>>(
            &mut group,
            "SoaMap",
            &label,
            capacity,
            num_entries,
            &op_keys,
            42,
        );
    }
    group.finish();
}

// ── Benchmark: At 1M scale with load factor sweep ───────────────────────

fn bench_load_factor_1m(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_factor_1m");
    group.sample_size(10);

    let capacity = 1_000_000;
    let ops = 500_000u64;

    for load_pct in [45, 65, 85] {
        let (num_entries, actual_lf) = compute_load_info(capacity, load_pct);
        group.throughput(Throughput::Elements(ops));
        let label = format!("{:.0}pct", actual_lf);

        let mut miss_rng = Sfc64::new(9999);
        let miss_keys: Vec<u64> = (0..ops as usize).map(|_| miss_rng.next_u64()).collect();

        // Hit
        bench_lf_hit_for::<UnorderedFlatMap<u64, u64>>(
            &mut group,
            "UFM_hit",
            &label,
            capacity,
            num_entries,
            ops,
            42,
        );
        bench_lf_hit_for::<Splitsies<u64, u64>>(
            &mut group,
            "Splitsies_hit",
            &label,
            capacity,
            num_entries,
            ops,
            42,
        );
        bench_lf_hit_for::<InPlaceOverflow<u64, u64>>(
            &mut group,
            "IPO_hit",
            &label,
            capacity,
            num_entries,
            ops,
            42,
        );
        bench_lf_hit_for::<hashbrown::HashMap<u64, u64>>(
            &mut group,
            "hashbrown_hit",
            &label,
            capacity,
            num_entries,
            ops,
            42,
        );
        bench_lf_hit_for::<OptiMapBench<u64, u64>>(
            &mut group,
            "OptiMap_hit",
            &label,
            capacity,
            num_entries,
            ops,
            42,
        );
        bench_lf_hit_for::<optimap::SoaMap<u64, u64>>(
            &mut group,
            "SoaMap_hit",
            &label,
            capacity,
            num_entries,
            ops,
            42,
        );

        // Miss
        bench_lf_miss_for::<UnorderedFlatMap<u64, u64>>(
            &mut group,
            "UFM_miss",
            &label,
            capacity,
            num_entries,
            &miss_keys,
            42,
        );
        bench_lf_miss_for::<Splitsies<u64, u64>>(
            &mut group,
            "Splitsies_miss",
            &label,
            capacity,
            num_entries,
            &miss_keys,
            42,
        );
        bench_lf_miss_for::<InPlaceOverflow<u64, u64>>(
            &mut group,
            "IPO_miss",
            &label,
            capacity,
            num_entries,
            &miss_keys,
            42,
        );
        bench_lf_miss_for::<hashbrown::HashMap<u64, u64>>(
            &mut group,
            "hashbrown_miss",
            &label,
            capacity,
            num_entries,
            &miss_keys,
            42,
        );
        bench_lf_miss_for::<OptiMapBench<u64, u64>>(
            &mut group,
            "OptiMap_miss",
            &label,
            capacity,
            num_entries,
            &miss_keys,
            42,
        );
        bench_lf_miss_for::<optimap::SoaMap<u64, u64>>(
            &mut group,
            "SoaMap_miss",
            &label,
            capacity,
            num_entries,
            &miss_keys,
            42,
        );
    }
    group.finish();
}

criterion_group!(
    load_factor,
    bench_lookup_hit_by_load,
    bench_lookup_miss_by_load,
    bench_mixed_by_load,
    bench_load_factor_1m,
);
criterion_main!(load_factor);
