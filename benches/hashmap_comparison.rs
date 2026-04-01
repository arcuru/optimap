use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use unordered_flat_map::UnorderedFlatMap;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn make_keys(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.r#gen()).collect()
}

fn make_string_keys(n: usize, seed: u64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let len = rng.r#gen_range(8..24);
            (0..len)
                .map(|_| (b'a' + rng.r#gen_range(0..26)) as char)
                .collect()
        })
        .collect()
}

// ── Benchmark: Sequential insert (u64 keys) ────────────────────────────────

fn bench_insert_u64(c: &mut Criterion) {
    let sizes = [1_000, 10_000, 100_000, 1_000_000];
    let mut group = c.benchmark_group("insert_u64");

    for &n in &sizes {
        let keys = make_keys(n, 42);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("ours", n), &keys, |b, keys| {
            b.iter(|| {
                let mut map = UnorderedFlatMap::with_capacity(n);
                for (i, &k) in keys.iter().enumerate() {
                    map.insert(k, i);
                }
                black_box(&map);
            });
        });

        group.bench_with_input(BenchmarkId::new("std_HashMap", n), &keys, |b, keys| {
            b.iter(|| {
                let mut map = HashMap::with_capacity(n);
                for (i, &k) in keys.iter().enumerate() {
                    map.insert(k, i);
                }
                black_box(&map);
            });
        });

        group.bench_with_input(BenchmarkId::new("hashbrown", n), &keys, |b, keys| {
            b.iter(|| {
                let mut map = hashbrown::HashMap::with_capacity(n);
                for (i, &k) in keys.iter().enumerate() {
                    map.insert(k, i);
                }
                black_box(&map);
            });
        });

        group.bench_with_input(BenchmarkId::new("indexmap", n), &keys, |b, keys| {
            b.iter(|| {
                let mut map = indexmap::IndexMap::with_capacity(n);
                for (i, &k) in keys.iter().enumerate() {
                    map.insert(k, i);
                }
                black_box(&map);
            });
        });
    }
    group.finish();
}

// ── Benchmark: Successful lookup (u64 keys) ─────────────────────────────────

fn bench_lookup_hit_u64(c: &mut Criterion) {
    let sizes = [1_000, 10_000, 100_000, 1_000_000];
    let mut group = c.benchmark_group("lookup_hit_u64");

    for &n in &sizes {
        let keys = make_keys(n, 42);
        group.throughput(Throughput::Elements(n as u64));

        // Pre-build maps
        let mut ours = UnorderedFlatMap::with_capacity(n);
        let mut std_map = HashMap::with_capacity(n);
        let mut hb_map = hashbrown::HashMap::with_capacity(n);
        let mut idx_map = indexmap::IndexMap::with_capacity(n);
        for (i, &k) in keys.iter().enumerate() {
            ours.insert(k, i);
            std_map.insert(k, i);
            hb_map.insert(k, i);
            idx_map.insert(k, i);
        }

        group.bench_with_input(BenchmarkId::new("ours", n), &keys, |b, keys| {
            b.iter(|| {
                let mut sum = 0usize;
                for k in keys {
                    sum += ours.get(k).unwrap_or(&0);
                }
                black_box(sum);
            });
        });

        group.bench_with_input(BenchmarkId::new("std_HashMap", n), &keys, |b, keys| {
            b.iter(|| {
                let mut sum = 0usize;
                for k in keys {
                    sum += std_map.get(k).unwrap_or(&0);
                }
                black_box(sum);
            });
        });

        group.bench_with_input(BenchmarkId::new("hashbrown", n), &keys, |b, keys| {
            b.iter(|| {
                let mut sum = 0usize;
                for k in keys {
                    sum += hb_map.get(k).unwrap_or(&0);
                }
                black_box(sum);
            });
        });

        group.bench_with_input(BenchmarkId::new("indexmap", n), &keys, |b, keys| {
            b.iter(|| {
                let mut sum = 0usize;
                for k in keys {
                    sum += idx_map.get(k).unwrap_or(&0);
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

// ── Benchmark: Failed lookup (u64 keys) ─────────────────────────────────────

fn bench_lookup_miss_u64(c: &mut Criterion) {
    let sizes = [1_000, 10_000, 100_000, 1_000_000];
    let mut group = c.benchmark_group("lookup_miss_u64");

    for &n in &sizes {
        let keys = make_keys(n, 42);
        let miss_keys = make_keys(n, 9999); // different seed → mostly misses
        group.throughput(Throughput::Elements(n as u64));

        let mut ours = UnorderedFlatMap::with_capacity(n);
        let mut std_map = HashMap::with_capacity(n);
        let mut hb_map = hashbrown::HashMap::with_capacity(n);
        for (i, &k) in keys.iter().enumerate() {
            ours.insert(k, i);
            std_map.insert(k, i);
            hb_map.insert(k, i);
        }

        group.bench_with_input(BenchmarkId::new("ours", n), &miss_keys, |b, miss| {
            b.iter(|| {
                let mut count = 0usize;
                for k in miss {
                    if ours.get(k).is_some() {
                        count += 1;
                    }
                }
                black_box(count);
            });
        });

        group.bench_with_input(
            BenchmarkId::new("std_HashMap", n),
            &miss_keys,
            |b, miss| {
                b.iter(|| {
                    let mut count = 0usize;
                    for k in miss {
                        if std_map.get(k).is_some() {
                            count += 1;
                        }
                    }
                    black_box(count);
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("hashbrown", n), &miss_keys, |b, miss| {
            b.iter(|| {
                let mut count = 0usize;
                for k in miss {
                    if hb_map.get(k).is_some() {
                        count += 1;
                    }
                }
                black_box(count);
            });
        });
    }
    group.finish();
}

// ── Benchmark: Mixed insert/lookup/remove workload ──────────────────────────

fn bench_mixed_workload(c: &mut Criterion) {
    let sizes = [10_000, 100_000];
    let mut group = c.benchmark_group("mixed_workload");

    for &n in &sizes {
        let mut rng = StdRng::seed_from_u64(123);
        // Pre-generate operations: 0=insert, 1=lookup, 2=remove
        let ops: Vec<(u8, u64, usize)> = (0..n)
            .map(|i| {
                let op = rng.r#gen_range(0..10u8); // 50% insert, 30% lookup, 20% remove
                let key = rng.r#gen_range(0..n as u64 / 2);
                let op_type = if op < 5 { 0 } else if op < 8 { 1 } else { 2 };
                (op_type, key, i)
            })
            .collect();
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("ours", n), &ops, |b, ops| {
            b.iter(|| {
                let mut map = UnorderedFlatMap::with_capacity(n / 2);
                for &(op, key, val) in ops {
                    match op {
                        0 => {
                            map.insert(key, val);
                        }
                        1 => {
                            black_box(map.get(&key));
                        }
                        _ => {
                            map.remove(&key);
                        }
                    }
                }
                black_box(map.len());
            });
        });

        group.bench_with_input(BenchmarkId::new("std_HashMap", n), &ops, |b, ops| {
            b.iter(|| {
                let mut map = HashMap::with_capacity(n / 2);
                for &(op, key, val) in ops {
                    match op {
                        0 => {
                            map.insert(key, val);
                        }
                        1 => {
                            black_box(map.get(&key));
                        }
                        _ => {
                            map.remove(&key);
                        }
                    }
                }
                black_box(map.len());
            });
        });

        group.bench_with_input(BenchmarkId::new("hashbrown", n), &ops, |b, ops| {
            b.iter(|| {
                let mut map = hashbrown::HashMap::with_capacity(n / 2);
                for &(op, key, val) in ops {
                    match op {
                        0 => {
                            map.insert(key, val);
                        }
                        1 => {
                            black_box(map.get(&key));
                        }
                        _ => {
                            map.remove(&key);
                        }
                    }
                }
                black_box(map.len());
            });
        });
    }
    group.finish();
}

// ── Benchmark: String key insert + lookup ───────────────────────────────────

fn bench_string_keys(c: &mut Criterion) {
    let sizes = [1_000, 10_000, 100_000];
    let mut group = c.benchmark_group("string_insert_lookup");

    for &n in &sizes {
        let keys = make_string_keys(n, 42);
        group.throughput(Throughput::Elements(n as u64));

        // Insert benchmark
        group.bench_with_input(
            BenchmarkId::new("ours_insert", n),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut map = UnorderedFlatMap::with_capacity(n);
                    for (i, k) in keys.iter().enumerate() {
                        map.insert(k.clone(), i);
                    }
                    black_box(&map);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("std_insert", n),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut map = HashMap::with_capacity(n);
                    for (i, k) in keys.iter().enumerate() {
                        map.insert(k.clone(), i);
                    }
                    black_box(&map);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hashbrown_insert", n),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut map = hashbrown::HashMap::with_capacity(n);
                    for (i, k) in keys.iter().enumerate() {
                        map.insert(k.clone(), i);
                    }
                    black_box(&map);
                });
            },
        );

        // Lookup benchmark (pre-build the maps)
        let mut ours = UnorderedFlatMap::with_capacity(n);
        let mut std_map = HashMap::with_capacity(n);
        let mut hb_map = hashbrown::HashMap::with_capacity(n);
        for (i, k) in keys.iter().enumerate() {
            ours.insert(k.clone(), i);
            std_map.insert(k.clone(), i);
            hb_map.insert(k.clone(), i);
        }

        group.bench_with_input(
            BenchmarkId::new("ours_lookup", n),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for k in keys {
                        sum += ours.get(k.as_str()).unwrap_or(&0);
                    }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("std_lookup", n),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for k in keys {
                        sum += std_map.get(k.as_str()).unwrap_or(&0);
                    }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hashbrown_lookup", n),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for k in keys {
                        sum += hb_map.get(k.as_str()).unwrap_or(&0);
                    }
                    black_box(sum);
                });
            },
        );
    }
    group.finish();
}

// ── Benchmark: Iteration ────────────────────────────────────────────────────

fn bench_iteration(c: &mut Criterion) {
    let sizes = [1_000, 10_000, 100_000, 1_000_000];
    let mut group = c.benchmark_group("iteration");

    for &n in &sizes {
        let keys = make_keys(n, 42);
        group.throughput(Throughput::Elements(n as u64));

        let mut ours = UnorderedFlatMap::with_capacity(n);
        let mut std_map = HashMap::with_capacity(n);
        let mut hb_map = hashbrown::HashMap::with_capacity(n);
        for (i, &k) in keys.iter().enumerate() {
            ours.insert(k, i);
            std_map.insert(k, i);
            hb_map.insert(k, i);
        }

        group.bench_with_input(BenchmarkId::new("ours", n), &n, |b, _| {
            b.iter(|| {
                let mut sum = 0usize;
                for (_, &v) in ours.iter() {
                    sum = sum.wrapping_add(v);
                }
                black_box(sum);
            });
        });

        group.bench_with_input(BenchmarkId::new("std_HashMap", n), &n, |b, _| {
            b.iter(|| {
                let mut sum = 0usize;
                for (_, &v) in std_map.iter() {
                    sum = sum.wrapping_add(v);
                }
                black_box(sum);
            });
        });

        group.bench_with_input(BenchmarkId::new("hashbrown", n), &n, |b, _| {
            b.iter(|| {
                let mut sum = 0usize;
                for (_, &v) in hb_map.iter() {
                    sum = sum.wrapping_add(v);
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

// ── Benchmark: Grow from empty (no pre-allocation) ──────────────────────────

fn bench_grow_from_empty(c: &mut Criterion) {
    let sizes = [1_000, 10_000, 100_000];
    let mut group = c.benchmark_group("grow_from_empty");

    for &n in &sizes {
        let keys = make_keys(n, 42);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("ours", n), &keys, |b, keys| {
            b.iter(|| {
                let mut map = UnorderedFlatMap::new();
                for (i, &k) in keys.iter().enumerate() {
                    map.insert(k, i);
                }
                black_box(&map);
            });
        });

        group.bench_with_input(BenchmarkId::new("std_HashMap", n), &keys, |b, keys| {
            b.iter(|| {
                let mut map = HashMap::new();
                for (i, &k) in keys.iter().enumerate() {
                    map.insert(k, i);
                }
                black_box(&map);
            });
        });

        group.bench_with_input(BenchmarkId::new("hashbrown", n), &keys, |b, keys| {
            b.iter(|| {
                let mut map = hashbrown::HashMap::new();
                for (i, &k) in keys.iter().enumerate() {
                    map.insert(k, i);
                }
                black_box(&map);
            });
        });
    }
    group.finish();
}

// ── Benchmark: Remove half then lookup ──────────────────────────────────────

fn bench_remove_then_lookup(c: &mut Criterion) {
    let sizes = [10_000, 100_000];
    let mut group = c.benchmark_group("remove_half_then_lookup");

    for &n in &sizes {
        let keys = make_keys(n, 42);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("ours", n), &keys, |b, keys| {
            b.iter_batched(
                || {
                    let mut map = UnorderedFlatMap::with_capacity(n);
                    for (i, &k) in keys.iter().enumerate() {
                        map.insert(k, i);
                    }
                    map
                },
                |mut map| {
                    // Remove first half
                    for &k in &keys[..n / 2] {
                        map.remove(&k);
                    }
                    // Lookup all (half hit, half miss)
                    let mut sum = 0usize;
                    for &k in keys {
                        sum += map.get(&k).unwrap_or(&0);
                    }
                    black_box(sum);
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("std_HashMap", n), &keys, |b, keys| {
            b.iter_batched(
                || {
                    let mut map = HashMap::with_capacity(n);
                    for (i, &k) in keys.iter().enumerate() {
                        map.insert(k, i);
                    }
                    map
                },
                |mut map| {
                    for &k in &keys[..n / 2] {
                        map.remove(&k);
                    }
                    let mut sum = 0usize;
                    for &k in keys {
                        sum += map.get(&k).unwrap_or(&0);
                    }
                    black_box(sum);
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("hashbrown", n), &keys, |b, keys| {
            b.iter_batched(
                || {
                    let mut map = hashbrown::HashMap::with_capacity(n);
                    for (i, &k) in keys.iter().enumerate() {
                        map.insert(k, i);
                    }
                    map
                },
                |mut map| {
                    for &k in &keys[..n / 2] {
                        map.remove(&k);
                    }
                    let mut sum = 0usize;
                    for &k in keys {
                        sum += map.get(&k).unwrap_or(&0);
                    }
                    black_box(sum);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// ── Benchmark: With ahash (fast hasher) ─────────────────────────────────────

fn bench_with_ahash(c: &mut Criterion) {
    let sizes = [10_000, 100_000, 1_000_000];
    let mut group = c.benchmark_group("ahash_insert_lookup");

    for &n in &sizes {
        let keys = make_keys(n, 42);
        group.throughput(Throughput::Elements(n as u64));

        // Insert with ahash
        group.bench_with_input(BenchmarkId::new("ours_ahash_insert", n), &keys, |b, keys| {
            b.iter(|| {
                let mut map = UnorderedFlatMap::with_capacity_and_hasher(n, ahash::RandomState::new());
                for (i, &k) in keys.iter().enumerate() {
                    map.insert(k, i);
                }
                black_box(&map);
            });
        });

        group.bench_with_input(
            BenchmarkId::new("hashbrown_ahash_insert", n),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut map: hashbrown::HashMap<u64, usize, ahash::RandomState> =
                        hashbrown::HashMap::with_capacity_and_hasher(n, ahash::RandomState::new());
                    for (i, &k) in keys.iter().enumerate() {
                        map.insert(k, i);
                    }
                    black_box(&map);
                });
            },
        );

        // Lookup with ahash
        let hasher = ahash::RandomState::new();
        let mut ours_ah = UnorderedFlatMap::with_capacity_and_hasher(n, hasher.clone());
        let mut hb_ah: hashbrown::HashMap<u64, usize, ahash::RandomState> =
            hashbrown::HashMap::with_capacity_and_hasher(n, hasher.clone());
        for (i, &k) in keys.iter().enumerate() {
            ours_ah.insert(k, i);
            hb_ah.insert(k, i);
        }

        group.bench_with_input(
            BenchmarkId::new("ours_ahash_lookup", n),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for k in keys {
                        sum += ours_ah.get(k).unwrap_or(&0);
                    }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hashbrown_ahash_lookup", n),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for k in keys {
                        sum += hb_ah.get(k).unwrap_or(&0);
                    }
                    black_box(sum);
                });
            },
        );
    }
    group.finish();
}

// ── Benchmark: High load factor (insert to ~85% full, then lookup) ───────

fn bench_high_load_lookup(c: &mut Criterion) {
    let sizes = [10_000, 100_000, 1_000_000];
    let mut group = c.benchmark_group("high_load_lookup");

    for &n in &sizes {
        // Insert exactly enough to hit ~85% load factor
        let keys = make_keys(n, 42);
        let miss_keys = make_keys(n, 9999);
        group.throughput(Throughput::Elements(n as u64));

        // Build maps WITHOUT extra capacity — let them grow naturally to high load
        let mut ours = UnorderedFlatMap::new();
        let mut hb_map = hashbrown::HashMap::new();
        for (i, &k) in keys.iter().enumerate() {
            ours.insert(k, i);
            hb_map.insert(k, i);
        }

        // Hit lookups at high load
        group.bench_with_input(BenchmarkId::new("ours_hit", n), &keys, |b, keys| {
            b.iter(|| {
                let mut sum = 0usize;
                for k in keys {
                    sum += ours.get(k).unwrap_or(&0);
                }
                black_box(sum);
            });
        });

        group.bench_with_input(BenchmarkId::new("hashbrown_hit", n), &keys, |b, keys| {
            b.iter(|| {
                let mut sum = 0usize;
                for k in keys {
                    sum += hb_map.get(k).unwrap_or(&0);
                }
                black_box(sum);
            });
        });

        // Miss lookups at high load
        group.bench_with_input(BenchmarkId::new("ours_miss", n), &miss_keys, |b, miss| {
            b.iter(|| {
                let mut count = 0usize;
                for k in miss {
                    if ours.get(k).is_some() {
                        count += 1;
                    }
                }
                black_box(count);
            });
        });

        group.bench_with_input(BenchmarkId::new("hashbrown_miss", n), &miss_keys, |b, miss| {
            b.iter(|| {
                let mut count = 0usize;
                for k in miss {
                    if hb_map.get(k).is_some() {
                        count += 1;
                    }
                }
                black_box(count);
            });
        });
    }
    group.finish();
}

// ── Benchmark: Mixed hit/miss at various ratios ─────────────────────────

fn bench_miss_ratio(c: &mut Criterion) {
    let n = 100_000usize;
    let mut group = c.benchmark_group("miss_ratio_100k");

    let keys = make_keys(n, 42);
    let miss_keys = make_keys(n, 9999);

    let mut ours = UnorderedFlatMap::with_capacity(n);
    let mut hb_map = hashbrown::HashMap::with_capacity(n);
    for (i, &k) in keys.iter().enumerate() {
        ours.insert(k, i);
        hb_map.insert(k, i);
    }

    // Build mixed lookup arrays with different hit/miss ratios
    for miss_pct in [0, 25, 50, 75, 100] {
        let lookup_keys: Vec<u64> = (0..n)
            .map(|i| {
                if (i % 100) < miss_pct {
                    miss_keys[i % miss_keys.len()]
                } else {
                    keys[i % keys.len()]
                }
            })
            .collect();

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("ours_{}pct_miss", miss_pct), n),
            &lookup_keys,
            |b, lk| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for k in lk {
                        sum += ours.get(k).unwrap_or(&0);
                    }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("hashbrown_{}pct_miss", miss_pct), n),
            &lookup_keys,
            |b, lk| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for k in lk {
                        sum += hb_map.get(k).unwrap_or(&0);
                    }
                    black_box(sum);
                });
            },
        );
    }
    group.finish();
}

// ── Benchmark: Insert into pre-allocated (no alloc in measurement) ──────────

fn bench_insert_prealloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_prealloc");
    let n = 1_000_000;
    let keys = make_keys(n, 42);
    group.throughput(Throughput::Elements(n as u64));

    let mut ours = UnorderedFlatMap::with_capacity(n);
    group.bench_function("ours", |b| {
        b.iter(|| {
            ours.clear();
            for (i, &k) in keys.iter().enumerate() {
                ours.insert(k, i);
            }
            black_box(ours.len());
        });
    });

    let mut hb = hashbrown::HashMap::with_capacity(n);
    group.bench_function("hashbrown", |b| {
        b.iter(|| {
            hb.clear();
            for (i, &k) in keys.iter().enumerate() {
                hb.insert(k, i);
            }
            black_box(hb.len());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_insert_u64,
    bench_insert_prealloc,
    bench_lookup_hit_u64,
    bench_lookup_miss_u64,
    bench_mixed_workload,
    bench_string_keys,
    bench_iteration,
    bench_grow_from_empty,
    bench_remove_then_lookup,
    bench_with_ahash,
    bench_high_load_lookup,
    bench_miss_ratio,
);
criterion_main!(benches);
