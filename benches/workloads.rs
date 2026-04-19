//! Realistic mixed-operation workload benchmarks.
//!
//! These benchmarks combine multiple operations in a single measurement
//! to test how the map performs under realistic conditions. Maps may
//! grow during the benchmark; this is intentional.

mod bench_helpers;

use bench_helpers::*;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

use optimap::matrix_types::*;
use optimap::{InPlaceOverflow, Splitsies, UnorderedFlatMap};

// ── Macro for main designs ──────────────────────────────────────────────────

macro_rules! main_maps {
    ($helper:ident, $group:expr, $($args:expr),*) => {
        $helper::<UnorderedFlatMap<u64, u64>>($group, "UFM", $($args),*);
        $helper::<Splitsies<u64, u64>>($group, "Splitsies", $($args),*);
        $helper::<InPlaceOverflow<u64, u64>>($group, "IPO", $($args),*);
        $helper::<hashbrown::HashMap<u64, u64>>($group, "hashbrown", $($args),*);
        $helper::<OptiMapBench<u64, u64>>($group, "OptiMap", $($args),*);
        // Matrix top contenders
        $helper::<Lo128_1bitMap<u64, u64>>($group, "Lo128_1bit", $($args),*);
        $helper::<Lo128_8bitMap<u64, u64>>($group, "Lo128_8bit", $($args),*);
        $helper::<Hi128_TombMap<u64, u64>>($group, "Hi128_Tomb", $($args),*);
        $helper::<Top128_TombMap<u64, u64>>($group, "Top128_Tomb", $($args),*);
    };
}

const LARGE_CAPACITY: usize = 107_520;
const LOAD_PCT: usize = 70;

// ── Workload: Equilibrium Churn ─────────────────────────────────────────────

fn bench_equilibrium_churn(c: &mut Criterion) {
    let mut group = c.benchmark_group("workload/churn");
    let ops = 2_000_000u64;

    for &(name, mask) in &[("4K", 0xFFFu64), ("64K", 0xFFFFu64), ("1M", 0xF_FFFFu64)] {
        group.throughput(Throughput::Elements(ops));
        if mask >= 0xF_FFFF {
            group.sample_size(10);
        }

        main_maps!(bench_churn_for, &mut group, name, ops, mask);
    }
    group.finish();
}

// ── Workload: Read-Heavy (95% read, 5% write) ──────────────────────────────

fn bench_read_heavy(c: &mut Criterion) {
    let mut group = c.benchmark_group("workload/read_heavy");
    let n = entries_for_load(LARGE_CAPACITY, LOAD_PCT);
    let ops = 500_000u64;
    group.throughput(Throughput::Elements(ops));

    let keys = make_random_keys(n, 42);
    let miss_keys = make_random_keys(n, 9999);

    // Pre-generate operation sequence
    let op_seq: Vec<(u8, u64)> = {
        let mut rng = Sfc64::new(777);
        (0..ops as usize)
            .map(|i| {
                let op = (rng.next_u64() % 100) as u8;
                let key = if op < 80 {
                    keys[i % keys.len()] // 80% hit
                } else if op < 95 {
                    miss_keys[i % miss_keys.len()] // 15% miss
                } else if op < 98 {
                    rng.next_u64() // 3% insert new
                } else {
                    keys[i % keys.len()] // 2% remove existing
                };
                (op, key)
            })
            .collect()
    };

    let n_str = n.to_string();
    main_maps!(
        bench_mixed_workload_for,
        &mut group,
        &n_str,
        &keys,
        &op_seq,
        LARGE_CAPACITY
    );
    group.finish();
}

// ── Workload: Write-Heavy (50% read, 30% insert, 20% remove) ───────────────

fn bench_write_heavy(c: &mut Criterion) {
    let mut group = c.benchmark_group("workload/write_heavy");
    let n = entries_for_load(LARGE_CAPACITY, LOAD_PCT);
    let ops = 500_000u64;
    group.throughput(Throughput::Elements(ops));

    let keys = make_random_keys(n, 42);

    let op_seq: Vec<(u8, u64)> = {
        let mut rng = Sfc64::new(777);
        (0..ops as usize)
            .map(|i| {
                let op = (rng.next_u64() % 10) as u8;
                let key = if op < 5 {
                    keys[i % keys.len()] // lookup existing
                } else {
                    rng.next_u64() // insert/remove random
                };
                (op, key)
            })
            .collect()
    };

    let n_str = n.to_string();
    main_maps!(
        bench_write_heavy_for,
        &mut group,
        &n_str,
        &keys,
        &op_seq,
        LARGE_CAPACITY
    );
    group.finish();
}

// ── Workload: Counting / Aggregation (entry API — manual, no trait) ─────────

fn bench_counting(c: &mut Criterion) {
    let mut group = c.benchmark_group("workload/counting");
    let ops = 5_000_000u64;
    group.sample_size(10);

    for &(name, distinct_pct) in &[("5pct", 5u64), ("50pct", 50), ("100pct", 100)] {
        let distinct = (ops * distinct_pct / 100).max(1);
        group.throughput(Throughput::Elements(ops));

        group.bench_function(BenchmarkId::new("UFM", name), |b| {
            b.iter(|| {
                let mut map = UnorderedFlatMap::new();
                let mut rng = Sfc64::new(42);
                for _ in 0..ops {
                    let k = rng.next_u64() % distinct;
                    *map.entry(k).or_insert(0u64) += 1;
                }
                black_box(map.len());
            });
        });

        group.bench_function(BenchmarkId::new("Splitsies", name), |b| {
            b.iter(|| {
                let mut map = Splitsies::new();
                let mut rng = Sfc64::new(42);
                for _ in 0..ops {
                    let k = rng.next_u64() % distinct;
                    *map.entry(k).or_insert(0u64) += 1;
                }
                black_box(map.len());
            });
        });

        group.bench_function(BenchmarkId::new("hashbrown", name), |b| {
            b.iter(|| {
                let mut map = hashbrown::HashMap::new();
                let mut rng = Sfc64::new(42);
                for _ in 0..ops {
                    let k = rng.next_u64() % distinct;
                    *map.entry(k).or_insert(0u64) += 1;
                }
                black_box(map.len());
            });
        });
    }
    group.finish();
}

// ── Workload: Post-Delete Lookup ────────────────────────────────────────────

fn bench_post_delete_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("workload/post_delete_lookup");

    for &(name, capacity) in &[("medium", 13_440usize), ("large", LARGE_CAPACITY)] {
        let n = entries_for_load(capacity, LOAD_PCT);
        let keys = make_random_keys(n, 42);
        group.throughput(Throughput::Elements(n as u64));

        main_maps!(bench_post_delete_for, &mut group, name, &keys, capacity);
    }
    group.finish();
}

// ── Workload: Miss Ratio Sweep ──────────────────────────────────────────────

fn bench_miss_ratio_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("workload/miss_ratio");
    let n = entries_for_load(LARGE_CAPACITY, LOAD_PCT);
    let ops = n;
    group.throughput(Throughput::Elements(ops as u64));

    let hit_keys = make_random_keys(n, 42);
    let miss_keys = make_random_keys(n, 9999);

    for &miss_pct in &[0, 25, 50, 75, 100] {
        let lookup_keys: Vec<u64> = (0..ops)
            .map(|i| {
                if (i * 100 / ops) < miss_pct {
                    miss_keys[i % miss_keys.len()]
                } else {
                    hit_keys[i % hit_keys.len()]
                }
            })
            .collect();

        let label = format!("{}miss", miss_pct);
        bench_miss_ratio_for::<UnorderedFlatMap<u64, u64>>(
            &mut group,
            &format!("UFM_{label}"),
            &hit_keys,
            &lookup_keys,
            LARGE_CAPACITY,
        );
        bench_miss_ratio_for::<Splitsies<u64, u64>>(
            &mut group,
            &format!("Splitsies_{label}"),
            &hit_keys,
            &lookup_keys,
            LARGE_CAPACITY,
        );
        bench_miss_ratio_for::<InPlaceOverflow<u64, u64>>(
            &mut group,
            &format!("IPO_{label}"),
            &hit_keys,
            &lookup_keys,
            LARGE_CAPACITY,
        );
        bench_miss_ratio_for::<hashbrown::HashMap<u64, u64>>(
            &mut group,
            &format!("hashbrown_{label}"),
            &hit_keys,
            &lookup_keys,
            LARGE_CAPACITY,
        );
        bench_miss_ratio_for::<OptiMapBench<u64, u64>>(
            &mut group,
            &format!("OptiMap_{label}"),
            &hit_keys,
            &lookup_keys,
            LARGE_CAPACITY,
        );
    }
    group.finish();
}

// ── Workload: Remove + Reinsert (tombstone-free advantage) ──────────────────

fn bench_remove_reinsert(c: &mut Criterion) {
    let mut group = c.benchmark_group("workload/remove_reinsert");
    let n = entries_for_load(LARGE_CAPACITY, LOAD_PCT);
    let ops = 100_000u64;
    group.throughput(Throughput::Elements(ops));

    let keys = make_random_keys(n, 42);

    let op_keys: Vec<u64> = {
        let mut rng = Sfc64::new(777);
        (0..ops as usize)
            .map(|_| keys[rng.next_u64() as usize % keys.len()])
            .collect()
    };

    let n_str = n.to_string();
    main_maps!(
        bench_remove_reinsert_for,
        &mut group,
        &n_str,
        &keys,
        &op_keys,
        LARGE_CAPACITY
    );
    group.finish();
}

// ── Workload: High-Load Stress (overflow path + prefetch effectiveness) ─────

fn bench_high_load_stress(c: &mut Criterion) {
    let mut group = c.benchmark_group("workload/high_load_stress");

    // Test at 85% load — near max_load, many groups full, overflow common
    let capacity = 107_520;
    let min_slots = (capacity * 8_usize).div_ceil(7);
    let min_groups = min_slots.div_ceil(15);
    let mut num_groups = 1;
    while num_groups < min_groups {
        num_groups *= 2;
    }
    let total_slots = num_groups * 15;
    let num_entries = total_slots * 85 / 100;
    let ops = 100_000u64;

    group.throughput(Throughput::Elements(ops));

    let keys = make_random_keys(num_entries, 42);
    let miss_keys = make_random_keys(ops as usize, 9999);

    // Hit at 85% load
    bench_high_load_hit_for::<UnorderedFlatMap<u64, u64>>(
        &mut group,
        "UFM_hit85",
        &keys,
        capacity,
        ops,
    );
    bench_high_load_hit_for::<Splitsies<u64, u64>>(
        &mut group,
        "Splitsies_hit85",
        &keys,
        capacity,
        ops,
    );
    bench_high_load_hit_for::<hashbrown::HashMap<u64, u64>>(
        &mut group,
        "hashbrown_hit85",
        &keys,
        capacity,
        ops,
    );
    bench_high_load_hit_for::<OptiMapBench<u64, u64>>(
        &mut group,
        "OptiMap_hit85",
        &keys,
        capacity,
        ops,
    );

    // Miss at 85% load
    bench_high_load_miss_for::<UnorderedFlatMap<u64, u64>>(
        &mut group,
        "UFM_miss85",
        num_entries,
        &miss_keys,
        &keys,
        capacity,
    );
    bench_high_load_miss_for::<Splitsies<u64, u64>>(
        &mut group,
        "Splitsies_miss85",
        num_entries,
        &miss_keys,
        &keys,
        capacity,
    );
    bench_high_load_miss_for::<hashbrown::HashMap<u64, u64>>(
        &mut group,
        "hashbrown_miss85",
        num_entries,
        &miss_keys,
        &keys,
        capacity,
    );
    bench_high_load_miss_for::<OptiMapBench<u64, u64>>(
        &mut group,
        "OptiMap_miss85",
        num_entries,
        &miss_keys,
        &keys,
        capacity,
    );

    group.finish();
}

criterion_group!(
    workloads,
    bench_equilibrium_churn,
    bench_read_heavy,
    bench_write_heavy,
    bench_counting,
    bench_post_delete_lookup,
    bench_miss_ratio_sweep,
    bench_remove_reinsert,
    bench_high_load_stress,
);
criterion_main!(workloads);
