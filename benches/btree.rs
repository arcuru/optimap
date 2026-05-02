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

        group.bench_with_input(
            BenchmarkId::new("FlatBTree_from_sorted", n),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let map: FlatBTree<u64, u64> =
                        FlatBTree::from_sorted_iter(keys.iter().map(|&k| (k, k)));
                    black_box(map);
                });
            },
        );

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
                let is_insert = rng.next_u64().is_multiple_of(2);
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

// ── range_mut ──────────────────────────────────────────────────────────

fn bench_range_mut(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/range_mut");

    for &n in &[10_000, 100_000] {
        let keys = make_random_keys(n, 42);
        let min_key = *keys.iter().min().unwrap();
        let max_key = *keys.iter().max().unwrap();
        let range_size = (max_key - min_key) / 10;

        let mut rng = Sfc64::new(123);
        let range_starts: Vec<u64> = (0..1000)
            .map(|_| min_key + (rng.next_u64() % (max_key - min_key - range_size)))
            .collect();

        group.bench_function(BenchmarkId::new("FlatBTree", n), |b| {
            let mut flat = build_flat::<0>(&keys);
            b.iter(|| {
                for &start in &range_starts {
                    for (_, v) in flat.range_mut(start..start + range_size) {
                        *v = v.wrapping_add(1);
                    }
                }
                black_box(&flat);
            });
        });

        group.bench_function(BenchmarkId::new("BTreeMap", n), |b| {
            let mut std_map = build_std::<0>(&keys);
            b.iter(|| {
                for &start in &range_starts {
                    for (_, v) in std_map.range_mut(start..start + range_size) {
                        *v = v.wrapping_add(1);
                    }
                }
                black_box(&std_map);
            });
        });
    }
    group.finish();
}

// ── Sparse-after-remove ────────────────────────────────────────────────
//
// Build N entries, remove every other, then measure lookup_hit on the
// surviving keys. Stresses the rebalance/merge path: if rebalance leaves
// the tree well-utilized, lookup cost ≈ original; if it doesn't, the
// tree walks more nodes than necessary.

fn bench_sparse_after_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/sparse_after_remove");

    for &n in &[10_000, 100_000] {
        let keys = make_random_keys(n, 42);
        let surviving: Vec<u64> = keys.iter().step_by(2).copied().collect();

        let mut flat = FlatBTree::with_capacity(n);
        for &k in &keys {
            flat.insert(k, k);
        }
        for k in keys.iter().skip(1).step_by(2) {
            flat.remove(k);
        }

        let mut std_map = BTreeMap::new();
        for &k in &keys {
            std_map.insert(k, k);
        }
        for k in keys.iter().skip(1).step_by(2) {
            std_map.remove(k);
        }

        group.throughput(Throughput::Elements(surviving.len() as u64));

        group.bench_function(BenchmarkId::new("FlatBTree", n), |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for k in &surviving {
                    sum = sum.wrapping_add(*flat.get(k).unwrap_or(&0));
                }
                black_box(sum);
            });
        });

        group.bench_function(BenchmarkId::new("BTreeMap", n), |b| {
            b.iter(|| {
                let mut sum = 0u64;
                for k in &surviving {
                    sum = sum.wrapping_add(*std_map.get(k).unwrap_or(&0));
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

// ── split_off / append ─────────────────────────────────────────────────

fn bench_split_off(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/split_off");

    // Exercise multiple split positions per N so we can see the asymmetric
    // cost: drain is O(n) regardless; surgical is O(min_subtree_size); std
    // structural split is O(log n). The 1% / 99% pivots specifically expose
    // whether always-copy-right hurts vs always-copy-left.
    let positions: &[(&str, f64)] = &[
        ("p001", 0.01), // peel off the smallest 1% — left small, right huge
        ("p010", 0.10), // 10% / 90%
        ("p050", 0.50), // half / half
        ("p090", 0.90), // 90% / 10%
        ("p099", 0.99), // peel off the largest 1% — left huge, right small
    ];

    for &n in &[1_000usize, 10_000, 100_000, 1_000_000] {
        // Use sequential keys so the pivot fraction maps directly to the
        // intended split position (random keys would make 0.5 of the keyspace
        // not coincide with 0.5 of the entries).
        let keys: Vec<u64> = (0..n as u64).collect();
        group.throughput(Throughput::Elements(n as u64));
        if n >= 1_000_000 {
            group.sample_size(20);
        }

        for &(label, frac) in positions {
            let pivot = ((n as f64) * frac) as u64;

            group.bench_with_input(
                BenchmarkId::new(format!("FlatBTree-drain-{label}"), n),
                &keys,
                |b, keys| {
                    b.iter_batched(
                        || build_flat::<0>(keys),
                        |mut map| {
                            let right = map.split_off_drain(&pivot);
                            black_box((map, right));
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("FlatBTree-surgical_right-{label}"), n),
                &keys,
                |b, keys| {
                    b.iter_batched(
                        || build_flat::<0>(keys),
                        |mut map| {
                            let right = map.split_off_surgical_right(&pivot);
                            black_box((map, right));
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("FlatBTree-surgical_left-{label}"), n),
                &keys,
                |b, keys| {
                    b.iter_batched(
                        || build_flat::<0>(keys),
                        |mut map| {
                            let right = map.split_off_surgical_left(&pivot);
                            black_box((map, right));
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );

            // Dispatcher: picks among drain / surgical_left / surgical_right.
            group.bench_with_input(
                BenchmarkId::new(format!("FlatBTree-dispatch-{label}"), n),
                &keys,
                |b, keys| {
                    b.iter_batched(
                        || build_flat::<0>(keys),
                        |mut map| {
                            let right = map.split_off(&pivot);
                            black_box((map, right));
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("std::BTreeMap-{label}"), n),
                &keys,
                |b, keys| {
                    b.iter_batched(
                        || build_std::<0>(keys),
                        |mut map| {
                            let right = map.split_off(&pivot);
                            black_box((map, right));
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    group.finish();
}

fn bench_append(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/append");

    // Two disjoint adjacent ranges of size n each → result of size 2n.
    // self holds keys in [0, 2^33), other holds [2^33, 2^34) — random within
    // each range, but `self.last_key < other.first_key` always.
    //
    // Append size sweep: 1K and 10M added so we see the small-N regime
    // (where fixed costs dominate) and the very-large-N regime (where the
    // graft savings of NOT touching self's existing nodes should compound).
    for &n in &[1_000usize, 10_000, 100_000, 1_000_000] {
        // Bounded random keys — keys_a in [0, 2^33), keys_b in [2^33, 2^34) so
        // self.last_key < other.first_key is GUARANTEED (the dispatcher routes
        // these to the disjoint-adjacent fast path).
        let keys_a: Vec<u64> = make_random_keys(n, 11)
            .into_iter()
            .map(|k| k & ((1u64 << 33) - 1))
            .collect();
        let keys_b: Vec<u64> = make_random_keys(n, 22)
            .into_iter()
            .map(|k| (k & ((1u64 << 33) - 1)) | (1u64 << 33))
            .collect();
        group.throughput(Throughput::Elements((2 * n) as u64));
        if n >= 1_000_000 {
            group.sample_size(20);
        }

        // Public dispatcher: detects disjointness and picks `append_extend`.
        group.bench_with_input(
            BenchmarkId::new("FlatBTree-dispatch", n),
            &(&keys_a, &keys_b),
            |b, (ka, kb)| {
                b.iter_batched(
                    || (build_flat::<0>(ka), build_flat::<0>(kb)),
                    |(mut a, mut b)| {
                        a.append(&mut b);
                        black_box((a, b));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // Drain + merge + bulk_load (former default). O(n + m).
        group.bench_with_input(
            BenchmarkId::new("FlatBTree-drain", n),
            &(&keys_a, &keys_b),
            |b, (ka, kb)| {
                b.iter_batched(
                    || (build_flat::<0>(ka), build_flat::<0>(kb)),
                    |(mut a, mut b)| {
                        a.append_drain(&mut b);
                        black_box((a, b));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // Drain + chain + bulk_load (no merge step, requires disjoint adjacent).
        group.bench_with_input(
            BenchmarkId::new("FlatBTree-concat", n),
            &(&keys_a, &keys_b),
            |b, (ka, kb)| {
                b.iter_batched(
                    || (build_flat::<0>(ka), build_flat::<0>(kb)),
                    |(mut a, mut b)| {
                        a.append_concat(&mut b);
                        black_box((a, b));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // Drain `other` only, insert each into `self` (exploits tail fast path).
        group.bench_with_input(
            BenchmarkId::new("FlatBTree-extend", n),
            &(&keys_a, &keys_b),
            |b, (ka, kb)| {
                b.iter_batched(
                    || (build_flat::<0>(ka), build_flat::<0>(kb)),
                    |(mut a, mut b)| {
                        a.append_extend(&mut b);
                        black_box((a, b));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // Tree-surgery graft (byte-copy other's nodes, splice chain, bridge spines).
        group.bench_with_input(
            BenchmarkId::new("FlatBTree-graft", n),
            &(&keys_a, &keys_b),
            |b, (ka, kb)| {
                b.iter_batched(
                    || (build_flat::<0>(ka), build_flat::<0>(kb)),
                    |(mut a, mut b)| {
                        a.append_graft(&mut b);
                        black_box((a, b));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("std::BTreeMap", n),
            &(&keys_a, &keys_b),
            |b, (ka, kb)| {
                b.iter_batched(
                    || (build_std::<0>(ka), build_std::<0>(kb)),
                    |(mut a, mut b)| {
                        a.append(&mut b);
                        black_box((a, b));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

/// Asymmetric append: small `other` (always 100 entries) appended to a large `self`.
/// This is where tree-surgery should shine — `self` doesn't need to be touched.
/// Drain pays O(n + m) for both even though m ≪ n.
fn bench_append_asymmetric(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree/append_asymmetric");

    let m: usize = 100; // small `other`
    let keys_b: Vec<u64> = make_random_keys(m, 99)
        .into_iter()
        .map(|k| (k & ((1u64 << 33) - 1)) | (1u64 << 33))
        .collect();

    for &n in &[10_000usize, 100_000, 1_000_000] {
        let keys_a: Vec<u64> = make_random_keys(n, 11)
            .into_iter()
            .map(|k| k & ((1u64 << 33) - 1))
            .collect();
        group.throughput(Throughput::Elements((n + m) as u64));
        if n >= 1_000_000 {
            group.sample_size(20);
        }

        group.bench_with_input(
            BenchmarkId::new("FlatBTree-dispatch", n),
            &(&keys_a, &keys_b),
            |b, (ka, kb)| {
                b.iter_batched(
                    || (build_flat::<0>(ka), build_flat::<0>(kb)),
                    |(mut a, mut b)| {
                        a.append(&mut b);
                        black_box((a, b));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("FlatBTree-drain", n),
            &(&keys_a, &keys_b),
            |b, (ka, kb)| {
                b.iter_batched(
                    || (build_flat::<0>(ka), build_flat::<0>(kb)),
                    |(mut a, mut b)| {
                        a.append_drain(&mut b);
                        black_box((a, b));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("FlatBTree-extend", n),
            &(&keys_a, &keys_b),
            |b, (ka, kb)| {
                b.iter_batched(
                    || (build_flat::<0>(ka), build_flat::<0>(kb)),
                    |(mut a, mut b)| {
                        a.append_extend(&mut b);
                        black_box((a, b));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("FlatBTree-graft", n),
            &(&keys_a, &keys_b),
            |b, (ka, kb)| {
                b.iter_batched(
                    || (build_flat::<0>(ka), build_flat::<0>(kb)),
                    |(mut a, mut b)| {
                        a.append_graft(&mut b);
                        black_box((a, b));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("std::BTreeMap", n),
            &(&keys_a, &keys_b),
            |b, (ka, kb)| {
                b.iter_batched(
                    || (build_std::<0>(ka), build_std::<0>(kb)),
                    |(mut a, mut b)| {
                        a.append(&mut b);
                        black_box((a, b));
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
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
    bench_large_scale,
    bench_churn,
    bench_large_values,
    bench_string_keys,
    bench_split_off,
    bench_append,
    bench_append_asymmetric,
    bench_range_mut,
    bench_sparse_after_remove,
);
criterion_main!(btree_benches);
