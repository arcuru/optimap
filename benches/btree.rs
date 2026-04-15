//! FlatBTree benchmarks.
//!
//! Compares FlatBTree against std::collections::BTreeMap on sorted map
//! operations: insert, lookup, remove, iteration, range queries, and
//! mixed workloads. Uses inherent methods (O(log n)) rather than the
//! Map trait (which has O(n) fallback for get/remove).

mod bench_helpers;

use bench_helpers::*;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use optimap::FlatBTree;
use std::collections::BTreeMap;

// ── Helpers ────────────────────────────────────────────────────────────────

fn build_flat<const N: usize>(keys: &[u64]) -> FlatBTree<u64, u64> {
    let mut map = FlatBTree::with_capacity(N);
    for &k in keys {
        map.insert(k, k);
    }
    map
}

fn build_std<const N: usize>(keys: &[u64]) -> BTreeMap<u64, u64> {
    let mut map = BTreeMap::new();
    for &k in keys {
        map.insert(k, k);
    }
    map
}

// ── Insert ─────────────────────────────────────────────────────────────────

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/insert");

    for &n in &[1_000, 10_000, 100_000] {
        let keys = make_random_keys(n, 42);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("FlatBTree", n), &keys, |b, keys| {
            b.iter(|| {
                let mut map = FlatBTree::with_capacity(n);
                for &k in keys {
                    map.insert(k, k);
                }
                black_box(&map);
            });
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap", n), &keys, |b, keys| {
            b.iter(|| {
                let mut map = BTreeMap::new();
                for &k in keys {
                    map.insert(k, k);
                }
                black_box(&map);
            });
        });
    }
    group.finish();
}

// ── Lookup Hit ─────────────────────────────────────────────────────────────

fn bench_lookup_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/lookup_hit");

    for &n in &[1_000, 10_000, 100_000] {
        let keys = make_random_keys(n, 42);
        let flat = build_flat::<0>(&keys);
        let std_map = build_std::<0>(&keys);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("FlatBTree", n), &keys, |b, keys| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in keys {
                    sum = sum.wrapping_add(*flat.get(&k).unwrap());
                }
                black_box(sum);
            });
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap", n), &keys, |b, keys| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in keys {
                    sum = sum.wrapping_add(*std_map.get(&k).unwrap());
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

// ── Lookup Miss ────────────────────────────────────────────────────────────

fn bench_lookup_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/lookup_miss");

    for &n in &[1_000, 10_000, 100_000] {
        let keys = make_random_keys(n, 42);
        let miss_keys = make_miss_keys(n);
        let flat = build_flat::<0>(&keys);
        let std_map = build_std::<0>(&keys);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("FlatBTree", n), &miss_keys, |b, miss| {
            b.iter(|| {
                let mut count = 0u64;
                for &k in miss {
                    if flat.contains_key(&k) {
                        count += 1;
                    }
                }
                black_box(count);
            });
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap", n), &miss_keys, |b, miss| {
            b.iter(|| {
                let mut count = 0u64;
                for &k in miss {
                    if std_map.contains_key(&k) {
                        count += 1;
                    }
                }
                black_box(count);
            });
        });
    }
    group.finish();
}

// ── Remove ─────────────────────────────────────────────────────────────────

fn bench_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/remove");

    for &n in &[1_000, 10_000, 100_000] {
        let keys = make_random_keys(n, 42);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("FlatBTree", n), &keys, |b, keys| {
            b.iter_batched(
                || build_flat::<0>(keys),
                |mut map| {
                    for &k in keys {
                        map.remove(&k);
                    }
                    black_box(&map);
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap", n), &keys, |b, keys| {
            b.iter_batched(
                || build_std::<0>(keys),
                |mut map| {
                    for &k in keys {
                        map.remove(&k);
                    }
                    black_box(&map);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// ── Sorted Iteration ───────────────────────────────────────────────────────

fn bench_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/iteration");

    for &n in &[1_000, 10_000, 100_000] {
        let keys = make_random_keys(n, 42);
        let flat = build_flat::<0>(&keys);
        let std_map = build_std::<0>(&keys);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_function(BenchmarkId::new("FlatBTree", n), |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for (_, &v) in flat.iter() {
                    sum = sum.wrapping_add(v);
                }
                black_box(sum);
            });
        });

        group.bench_function(BenchmarkId::new("BTreeMap", n), |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for (_, &v) in std_map.iter() {
                    sum = sum.wrapping_add(v);
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

// ── Range Queries ──────────────────────────────────────────────────────────

fn bench_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/range");

    for &n in &[10_000, 100_000] {
        let keys = make_random_keys(n, 42);
        let flat = build_flat::<0>(&keys);
        let std_map = build_std::<0>(&keys);

        // Find min/max to construct sensible ranges
        let min_key = *keys.iter().min().unwrap();
        let max_key = *keys.iter().max().unwrap();
        let range_size = (max_key - min_key) / 10; // ~10% of keyspace

        // Build a set of range start points
        let mut rng = Sfc64::new(123);
        let range_starts: Vec<u64> = (0..1000)
            .map(|_| min_key + (rng.next_u64() % (max_key - min_key - range_size)))
            .collect();

        group.bench_function(BenchmarkId::new("FlatBTree", n), |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &start in &range_starts {
                    for (_, &v) in flat.range(start..start + range_size) {
                        sum = sum.wrapping_add(v);
                    }
                }
                black_box(sum);
            });
        });

        group.bench_function(BenchmarkId::new("BTreeMap", n), |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &start in &range_starts {
                    for (_, &v) in std_map.range(start..start + range_size) {
                        sum = sum.wrapping_add(v);
                    }
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

// ── First / Last Key ───────────────────────────────────────────────────────

fn bench_first_last(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/first_last");

    for &n in &[1_000, 100_000] {
        let keys = make_random_keys(n, 42);
        let flat = build_flat::<0>(&keys);
        let std_map = build_std::<0>(&keys);

        group.bench_function(BenchmarkId::new("FlatBTree_first", n), |b| {
            b.iter(|| black_box(flat.first_key_value()));
        });

        group.bench_function(BenchmarkId::new("BTreeMap_first", n), |b| {
            b.iter(|| black_box(std_map.iter().next()));
        });

        group.bench_function(BenchmarkId::new("FlatBTree_last", n), |b| {
            b.iter(|| black_box(flat.last_key_value()));
        });

        group.bench_function(BenchmarkId::new("BTreeMap_last", n), |b| {
            b.iter(|| black_box(std_map.iter().next_back()));
        });
    }
    group.finish();
}

// ── Entry API (Counting Pattern) ───────────────────────────────────────────

fn bench_counting(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/counting");

    // 5% distinct keys out of 1M operations
    let n_ops = 1_000_000;
    let n_distinct = n_ops / 20;
    let mut rng = Sfc64::new(42);
    let ops: Vec<u64> = (0..n_ops)
        .map(|_| rng.next_u64() % n_distinct as u64)
        .collect();

    group.throughput(Throughput::Elements(n_ops as u64));

    group.bench_function("FlatBTree", |b| {
        b.iter(|| {
            let mut map = FlatBTree::new();
            for &k in &ops {
                *map.entry(k).or_insert(0u64) += 1;
            }
            black_box(&map);
        });
    });

    group.bench_function("BTreeMap", |b| {
        b.iter(|| {
            let mut map = BTreeMap::new();
            for &k in &ops {
                *map.entry(k).or_insert(0u64) += 1;
            }
            black_box(&map);
        });
    });

    group.finish();
}

// ── Mixed Workload (Read-Heavy) ────────────────────────────────────────────

fn bench_mixed_read_heavy(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/mixed_read_heavy");
    let n = 50_000;
    let n_ops = 500_000;
    let keys = make_random_keys(n, 42);
    let miss_keys = make_miss_keys(n);
    group.throughput(Throughput::Elements(n_ops as u64));

    // 80% hit, 15% miss, 5% insert
    let mut rng = Sfc64::new(77);
    let ops: Vec<(u8, u64)> = (0..n_ops)
        .map(|_| {
            let op = (rng.next_u64() % 100) as u8;
            let key = if op < 80 {
                keys[rng.next_u64() as usize % keys.len()]
            } else if op < 95 {
                miss_keys[rng.next_u64() as usize % miss_keys.len()]
            } else {
                rng.next_u64()
            };
            (op, key)
        })
        .collect();

    group.bench_function("FlatBTree", |b| {
        b.iter_batched(
            || build_flat::<0>(&keys),
            |mut map| {
                let mut sum = 0u64;
                for &(op, key) in &ops {
                    if op < 95 {
                        if let Some(&v) = map.get(&key) {
                            sum = sum.wrapping_add(v);
                        }
                    } else {
                        map.insert(key, key);
                    }
                }
                black_box(sum);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.bench_function("BTreeMap", |b| {
        b.iter_batched(
            || build_std::<0>(&keys),
            |mut map| {
                let mut sum = 0u64;
                for &(op, key) in &ops {
                    if op < 95 {
                        if let Some(&v) = map.get(&key) {
                            sum = sum.wrapping_add(v);
                        }
                    } else {
                        map.insert(key, key);
                    }
                }
                black_box(sum);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ── Sorted Insert (best case for B-trees) ──────────────────────────────────

fn bench_sorted_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/sorted_insert");

    for &n in &[1_000, 10_000, 100_000] {
        let keys: Vec<u64> = (0..n as u64).collect();
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("FlatBTree", n), &keys, |b, keys| {
            b.iter(|| {
                let mut map = FlatBTree::with_capacity(n);
                for &k in keys {
                    map.insert(k, k);
                }
                black_box(&map);
            });
        });

        group.bench_with_input(BenchmarkId::new("BTreeMap", n), &keys, |b, keys| {
            b.iter(|| {
                let mut map = BTreeMap::new();
                for &k in keys {
                    map.insert(k, k);
                }
                black_box(&map);
            });
        });
    }
    group.finish();
}

// ── Clone ──────────────────────────────────────────────────────────────────

fn bench_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/clone");

    for &n in &[1_000, 10_000, 100_000] {
        let keys = make_random_keys(n, 42);
        let flat = build_flat::<0>(&keys);
        let std_map = build_std::<0>(&keys);

        group.bench_function(BenchmarkId::new("FlatBTree", n), |b| {
            b.iter(|| black_box(flat.clone()));
        });

        group.bench_function(BenchmarkId::new("BTreeMap", n), |b| {
            b.iter(|| black_box(std_map.clone()));
        });
    }
    group.finish();
}

criterion_group!(
    btree_benches,
    bench_insert,
    bench_lookup_hit,
    bench_lookup_miss,
    bench_remove,
    bench_iteration,
    bench_range,
    bench_first_last,
    bench_counting,
    bench_mixed_read_heavy,
    bench_sorted_insert,
    bench_clone,
);
criterion_main!(btree_benches);
