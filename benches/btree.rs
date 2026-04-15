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

// ── Large Scale (DRAM-bound) ───────────────────────────────────────────

fn bench_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/large_scale");
    group.sample_size(10);

    for &n in &[1_000_000, 5_000_000, 20_000_000] {
        let keys = make_random_keys(n, 42);
        let label = format!("{}M", n / 1_000_000);

        // Build maps once (expensive)
        let flat = build_flat::<0>(&keys);
        let std_map = build_std::<0>(&keys);

        // Lookup hit
        group.throughput(Throughput::Elements(n as u64));
        group.bench_function(BenchmarkId::new("FlatBTree_hit", &label), |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in &keys[..n.min(100_000)] {
                    sum = sum.wrapping_add(*flat.get(&k).unwrap());
                }
                black_box(sum);
            });
        });
        group.bench_function(BenchmarkId::new("BTreeMap_hit", &label), |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in &keys[..n.min(100_000)] {
                    sum = sum.wrapping_add(*std_map.get(&k).unwrap());
                }
                black_box(sum);
            });
        });

        // Lookup miss
        let miss_keys = make_miss_keys(100_000);
        group.bench_function(BenchmarkId::new("FlatBTree_miss", &label), |b| {
            b.iter(|| {
                let mut count = 0u64;
                for &k in &miss_keys {
                    if flat.contains_key(&k) {
                        count += 1;
                    }
                }
                black_box(count);
            });
        });
        group.bench_function(BenchmarkId::new("BTreeMap_miss", &label), |b| {
            b.iter(|| {
                let mut count = 0u64;
                for &k in &miss_keys {
                    if std_map.contains_key(&k) {
                        count += 1;
                    }
                }
                black_box(count);
            });
        });

        // Iteration (full scan)
        group.bench_function(BenchmarkId::new("FlatBTree_iter", &label), |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for (_, &v) in flat.iter() {
                    sum = sum.wrapping_add(v);
                }
                black_box(sum);
            });
        });
        group.bench_function(BenchmarkId::new("BTreeMap_iter", &label), |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for (_, &v) in std_map.iter() {
                    sum = sum.wrapping_add(v);
                }
                black_box(sum);
            });
        });

        // Range query (1% of keyspace, 100 queries)
        let min_key = *keys.iter().min().unwrap();
        let max_key = *keys.iter().max().unwrap();
        let range_size = (max_key - min_key) / 100;
        let mut rng = Sfc64::new(456);
        let range_starts: Vec<u64> = (0..100)
            .map(|_| min_key + (rng.next_u64() % (max_key - min_key - range_size)))
            .collect();

        group.bench_function(BenchmarkId::new("FlatBTree_range", &label), |b| {
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
        group.bench_function(BenchmarkId::new("BTreeMap_range", &label), |b| {
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

// ── Equilibrium Churn ──────────────────────────────────────────────────

fn bench_churn(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/churn");

    for &n in &[4_000, 64_000, 1_000_000] {
        let label = format!("{}", n);
        let n_ops = 2_000_000usize;
        let keys = make_random_keys(n, 42);

        // Pre-build maps to equilibrium size
        let mut flat = FlatBTree::with_capacity(n);
        let mut std_map = BTreeMap::new();
        for &k in &keys {
            flat.insert(k, k);
            std_map.insert(k, k);
        }

        // Build op sequence: 50% insert random, 50% remove random existing
        let mut rng = Sfc64::new(77);
        let ops: Vec<(bool, u64)> = (0..n_ops)
            .map(|_| {
                let is_insert = rng.next_u64() % 2 == 0;
                let key = if is_insert {
                    rng.next_u64()
                } else {
                    keys[rng.next_u64() as usize % keys.len()]
                };
                (is_insert, key)
            })
            .collect();

        group.throughput(Throughput::Elements(n_ops as u64));
        if n >= 1_000_000 {
            group.sample_size(10);
        }

        group.bench_function(BenchmarkId::new("FlatBTree", &label), |b| {
            b.iter_batched(
                || flat.clone(),
                |mut map| {
                    for &(is_insert, key) in &ops {
                        if is_insert {
                            map.insert(key, key);
                        } else {
                            map.remove(&key);
                        }
                    }
                    black_box(&map);
                },
                criterion::BatchSize::LargeInput,
            );
        });

        group.bench_function(BenchmarkId::new("BTreeMap", &label), |b| {
            b.iter_batched(
                || std_map.clone(),
                |mut map| {
                    for &(is_insert, key) in &ops {
                        if is_insert {
                            map.insert(key, key);
                        } else {
                            map.remove(&key);
                        }
                    }
                    black_box(&map);
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ── Large Values ──────────────────────────────────────────────────────

/// A value type that occupies `N` bytes.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct BigVal<const N: usize>([u8; N]);

impl<const N: usize> BigVal<N> {
    fn new(seed: u64) -> Self {
        let mut arr = [0u8; N];
        // Fill with deterministic bytes
        let bytes = seed.to_le_bytes();
        for (i, b) in arr.iter_mut().enumerate() {
            *b = bytes[i % 8];
        }
        BigVal(arr)
    }
}

fn bench_large_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/large_values");

    // Test with 64B, 128B, 256B, 512B values
    macro_rules! bench_value_size {
        ($name:expr, $n:expr, $size:expr, $ty:ty) => {
            let keys = make_random_keys($n, 42);

            // Insert
            group.bench_function(
                BenchmarkId::new(concat!("FlatBTree_insert_", $name), $n),
                |b| {
                    b.iter(|| {
                        let mut map: FlatBTree<u64, $ty> = FlatBTree::with_capacity($n);
                        for &k in &keys {
                            map.insert(k, <$ty>::new(k));
                        }
                        black_box(&map);
                    });
                },
            );
            group.bench_function(
                BenchmarkId::new(concat!("BTreeMap_insert_", $name), $n),
                |b| {
                    b.iter(|| {
                        let mut map: BTreeMap<u64, $ty> = BTreeMap::new();
                        for &k in &keys {
                            map.insert(k, <$ty>::new(k));
                        }
                        black_box(&map);
                    });
                },
            );

            // Lookup hit
            {
                let mut flat: FlatBTree<u64, $ty> = FlatBTree::with_capacity($n);
                let mut std_map: BTreeMap<u64, $ty> = BTreeMap::new();
                for &k in &keys {
                    flat.insert(k, <$ty>::new(k));
                    std_map.insert(k, <$ty>::new(k));
                }

                group.bench_function(
                    BenchmarkId::new(concat!("FlatBTree_hit_", $name), $n),
                    |b| {
                        b.iter(|| {
                            let mut sum = 0u8;
                            for &k in &keys {
                                sum = sum.wrapping_add(flat.get(&k).unwrap().0[0]);
                            }
                            black_box(sum);
                        });
                    },
                );
                group.bench_function(BenchmarkId::new(concat!("BTreeMap_hit_", $name), $n), |b| {
                    b.iter(|| {
                        let mut sum = 0u8;
                        for &k in &keys {
                            sum = sum.wrapping_add(std_map.get(&k).unwrap().0[0]);
                        }
                        black_box(sum);
                    });
                });
            }

            // Iteration
            {
                let mut flat: FlatBTree<u64, $ty> = FlatBTree::with_capacity($n);
                let mut std_map: BTreeMap<u64, $ty> = BTreeMap::new();
                for &k in &keys {
                    flat.insert(k, <$ty>::new(k));
                    std_map.insert(k, <$ty>::new(k));
                }

                group.bench_function(
                    BenchmarkId::new(concat!("FlatBTree_iter_", $name), $n),
                    |b| {
                        b.iter(|| {
                            let mut sum = 0u8;
                            for (_, v) in flat.iter() {
                                sum = sum.wrapping_add(v.0[0]);
                            }
                            black_box(sum);
                        });
                    },
                );
                group.bench_function(
                    BenchmarkId::new(concat!("BTreeMap_iter_", $name), $n),
                    |b| {
                        b.iter(|| {
                            let mut sum = 0u8;
                            for (_, v) in std_map.iter() {
                                sum = sum.wrapping_add(v.0[0]);
                            }
                            black_box(sum);
                        });
                    },
                );
            }
        };
    }

    let n = 10_000;
    group.throughput(Throughput::Elements(n as u64));
    bench_value_size!("64B", n, 64, BigVal<64>);
    bench_value_size!("128B", n, 128, BigVal<128>);
    bench_value_size!("200B", n, 200, BigVal<200>);

    group.finish();
}

// ── String Keys ───────────────────────────────────────────────────────

fn bench_string_keys(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/string_keys");
    let n = 10_000;
    group.throughput(Throughput::Elements(n as u64));

    let mut rng = Sfc64::new(42);
    let keys: Vec<String> = (0..n)
        .map(|_| format!("key_{:016x}", rng.next_u64()))
        .collect();
    let miss_keys: Vec<String> = (0..n)
        .map(|_| format!("miss_{:016x}", rng.next_u64()))
        .collect();

    // Insert
    group.bench_function("FlatBTree_insert", |b| {
        b.iter(|| {
            let mut map = FlatBTree::new();
            for k in &keys {
                map.insert(k.clone(), 1u64);
            }
            black_box(&map);
        });
    });
    group.bench_function("BTreeMap_insert", |b| {
        b.iter(|| {
            let mut map = BTreeMap::new();
            for k in &keys {
                map.insert(k.clone(), 1u64);
            }
            black_box(&map);
        });
    });

    // Lookup hit
    let mut flat = FlatBTree::new();
    let mut std_map = BTreeMap::new();
    for k in &keys {
        flat.insert(k.clone(), 1u64);
        std_map.insert(k.clone(), 1u64);
    }

    group.bench_function("FlatBTree_hit", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for k in &keys {
                sum += flat.get(k.as_str()).unwrap();
            }
            black_box(sum);
        });
    });
    group.bench_function("BTreeMap_hit", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for k in &keys {
                sum += std_map.get(k.as_str()).unwrap();
            }
            black_box(sum);
        });
    });

    // Lookup miss
    group.bench_function("FlatBTree_miss", |b| {
        b.iter(|| {
            let mut count = 0u64;
            for k in &miss_keys {
                if flat.contains_key(k.as_str()) {
                    count += 1;
                }
            }
            black_box(count);
        });
    });
    group.bench_function("BTreeMap_miss", |b| {
        b.iter(|| {
            let mut count = 0u64;
            for k in &miss_keys {
                if std_map.contains_key(k.as_str()) {
                    count += 1;
                }
            }
            black_box(count);
        });
    });

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
    bench_large_scale,
    bench_churn,
    bench_large_values,
    bench_string_keys,
);
criterion_main!(btree_benches);
