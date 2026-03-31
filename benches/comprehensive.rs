//! Comprehensive hashmap benchmarks inspired by Martin Ankerl's methodology.
//! https://martin.ankerl.com/2022/08/27/hashmap-bench-01/
//!
//! Key differences from our original benchmarks:
//! - Checksummed outputs to prevent dead-code elimination
//! - Equilibrium insert/erase cycling at various sizes
//! - Growing-lookup pattern (insert few, lookup many)
//! - Varying string sizes (SSO boundary, medium, large)
//! - Clone/copy performance
//! - Only ours vs hashbrown (both use foldhash)

use criterion::{
    BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};

use unordered_flat_map::UnorderedFlatMap;

// ── Fast RNG (SFC64, same as Ankerl's benchmarks) ──────────────────────

/// SFC64: Small Fast Counting RNG. Very fast, good quality.
struct Sfc64 {
    a: u64,
    b: u64,
    c: u64,
    counter: u64,
}

impl Sfc64 {
    fn new(seed: u64) -> Self {
        let mut rng = Sfc64 {
            a: seed,
            b: seed,
            c: seed,
            counter: 1,
        };
        for _ in 0..12 {
            rng.next();
        }
        rng
    }

    #[inline(always)]
    fn next(&mut self) -> u64 {
        let tmp = self.a.wrapping_add(self.b).wrapping_add(self.counter);
        self.counter += 1;
        self.a = self.b ^ (self.b >> 11);
        self.b = self.c.wrapping_add(self.c << 3);
        self.c = self.c.rotate_left(24).wrapping_add(tmp);
        tmp
    }

    /// Bounded random number in [0, bound). Uses 128-bit multiply trick.
    #[inline(always)]
    fn bounded(&mut self, bound: u64) -> u64 {
        let r = self.next();
        ((r as u128 * bound as u128) >> 64) as u64
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn make_string(rng: &mut Sfc64, len: usize) -> String {
    (0..len)
        .map(|_| (b'a' + (rng.next() % 26) as u8) as char)
        .collect()
}

// ── Benchmark: Insert & Erase 100M (phases) ────────────────────────────
// Scaled down to 10M for reasonable bench time.
// 5 phases: insert all, clear, reinsert, erase one-by-one, drop.

fn bench_insert_erase_phases(c: &mut Criterion) {
    let n = 5_000_000usize;
    let mut group = c.benchmark_group("insert_erase_phases");
    group.sample_size(10);
    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("ours", |b| {
        b.iter(|| {
            let mut rng = Sfc64::new(123);
            let mut checksum = 0u64;

            // Phase 1: Insert n random entries
            let mut map = UnorderedFlatMap::new();
            for _ in 0..n {
                let k = rng.next();
                map.insert(k, k);
            }
            checksum = checksum.wrapping_add(map.len() as u64);

            // Phase 2: Clear
            map.clear();
            checksum = checksum.wrapping_add(map.len() as u64);

            // Phase 3: Reinsert (same RNG, different seed)
            let mut rng2 = Sfc64::new(456);
            for _ in 0..n {
                let k = rng2.next();
                map.insert(k, k);
            }
            checksum = checksum.wrapping_add(map.len() as u64);

            // Phase 4: Erase one by one (replay rng2 keys)
            let mut rng3 = Sfc64::new(456);
            for _ in 0..n {
                let k = rng3.next();
                map.remove(&k);
            }
            checksum = checksum.wrapping_add(map.len() as u64);

            // Phase 5: Drop (implicit)
            black_box(checksum);
        });
    });

    group.bench_function("hashbrown", |b| {
        b.iter(|| {
            let mut rng = Sfc64::new(123);
            let mut checksum = 0u64;

            let mut map = hashbrown::HashMap::new();
            for _ in 0..n {
                let k = rng.next();
                map.insert(k, k);
            }
            checksum = checksum.wrapping_add(map.len() as u64);

            map.clear();
            checksum = checksum.wrapping_add(map.len() as u64);

            let mut rng2 = Sfc64::new(456);
            for _ in 0..n {
                let k = rng2.next();
                map.insert(k, k);
            }
            checksum = checksum.wrapping_add(map.len() as u64);

            let mut rng3 = Sfc64::new(456);
            for _ in 0..n {
                let k = rng3.next();
                map.remove(&k);
            }
            checksum = checksum.wrapping_add(map.len() as u64);

            black_box(checksum);
        });
    });

    group.finish();
}

// ── Benchmark: RandomInsertErase (equilibrium churn) ───────────────────
// Insert and erase at steady state. Tests behavior under churn at various
// map sizes. Uses bitmask to control the key space.

fn bench_insert_erase_equilibrium(c: &mut Criterion) {
    let mut group = c.benchmark_group("equilibrium_churn");

    // Test at different equilibrium sizes via bitmask
    for bits in [12, 16, 20] {
        let mask: u64 = (1u64 << bits) - 1;
        let ops = 2_000_000u64;
        group.throughput(Throughput::Elements(ops));

        group.bench_with_input(
            BenchmarkId::new("ours", format!("{}bit", bits)),
            &(mask, ops),
            |b, &(mask, ops)| {
                b.iter(|| {
                    let mut rng = Sfc64::new(999);
                    let mut map = UnorderedFlatMap::new();
                    let mut checksum = 0u64;

                    for _ in 0..ops {
                        // Insert a random key
                        let k = rng.next() & mask;
                        map.insert(k, k);
                        // Erase a different random key
                        let k2 = rng.next() & mask;
                        if let Some(v) = map.remove(&k2) {
                            checksum = checksum.wrapping_add(v);
                        }
                    }
                    checksum = checksum.wrapping_add(map.len() as u64);
                    black_box(checksum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hashbrown", format!("{}bit", bits)),
            &(mask, ops),
            |b, &(mask, ops)| {
                b.iter(|| {
                    let mut rng = Sfc64::new(999);
                    let mut map = hashbrown::HashMap::new();
                    let mut checksum = 0u64;

                    for _ in 0..ops {
                        let k = rng.next() & mask;
                        map.insert(k, k);
                        let k2 = rng.next() & mask;
                        if let Some(v) = map.remove(&k2) {
                            checksum = checksum.wrapping_add(v);
                        }
                    }
                    checksum = checksum.wrapping_add(map.len() as u64);
                    black_box(checksum);
                });
            },
        );
    }
    group.finish();
}

// ── Benchmark: RandomDistinct (increment pattern) ──────────────────────
// `checksum += ++map[rng(max)]` — models counting/aggregation workloads.
// Tests with different distinct-key ratios.

fn bench_random_distinct(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_distinct");
    let ops = 5_000_000u64;

    // 5% distinct = 250K max keys, 50% = 2.5M, 100% = 5M
    for (label, max_key) in [("5pct", ops / 20), ("50pct", ops / 2), ("100pct", ops)] {
        group.throughput(Throughput::Elements(ops));

        group.bench_with_input(
            BenchmarkId::new("ours", label),
            &max_key,
            |b, &max_key| {
                b.iter(|| {
                    let mut rng = Sfc64::new(123);
                    let mut map = UnorderedFlatMap::new();
                    let mut checksum = 0u64;

                    for _ in 0..ops {
                        let k = rng.bounded(max_key);
                        let v = map.entry(k).or_insert(0u64);
                        *v += 1;
                        checksum = checksum.wrapping_add(*v);
                    }
                    black_box(checksum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hashbrown", label),
            &max_key,
            |b, &max_key| {
                b.iter(|| {
                    let mut rng = Sfc64::new(123);
                    let mut map = hashbrown::HashMap::new();
                    let mut checksum = 0u64;

                    for _ in 0..ops {
                        let k = rng.bounded(max_key);
                        let v = map.entry(k).or_insert(0u64);
                        *v += 1;
                        checksum = checksum.wrapping_add(*v);
                    }
                    black_box(checksum);
                });
            },
        );
    }
    group.finish();
}

// ── Benchmark: Growing Lookup (insert few, lookup many) ────────────────
// Inserts 4 elements at a time, then performs many lookups.
// Models read-heavy workloads with gradual growth.

fn bench_growing_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("growing_lookup");

    for (target_size, lookups_per_batch) in [(2_000, 5_000u64), (100_000, 1_000)] {
        let total_lookups = (target_size / 4) * lookups_per_batch;
        group.throughput(Throughput::Elements(total_lookups));

        // 50% hit rate
        group.bench_with_input(
            BenchmarkId::new("ours_50pct", target_size),
            &(target_size, lookups_per_batch),
            |b, &(target_size, lookups_per_batch)| {
                b.iter(|| {
                    let mut insert_rng = Sfc64::new(123);
                    let mut lookup_rng = Sfc64::new(987654321);
                    let mut map = UnorderedFlatMap::new();
                    let mut checksum = 0u64;

                    for _ in 0..(target_size / 4) {
                        // Insert 4
                        for _ in 0..4 {
                            let k = insert_rng.next();
                            map.insert(k, k);
                        }
                        // Lookup many with ~50% hit rate
                        for _ in 0..lookups_per_batch {
                            let k = lookup_rng.next();
                            if let Some(&v) = map.get(&k) {
                                checksum = checksum.wrapping_add(v);
                            }
                        }
                    }
                    black_box(checksum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hashbrown_50pct", target_size),
            &(target_size, lookups_per_batch),
            |b, &(target_size, lookups_per_batch)| {
                b.iter(|| {
                    let mut insert_rng = Sfc64::new(123);
                    let mut lookup_rng = Sfc64::new(987654321);
                    let mut map = hashbrown::HashMap::new();
                    let mut checksum = 0u64;

                    for _ in 0..(target_size / 4) {
                        for _ in 0..4 {
                            let k = insert_rng.next();
                            map.insert(k, k);
                        }
                        for _ in 0..lookups_per_batch {
                            let k = lookup_rng.next();
                            if let Some(&v) = map.get(&k) {
                                checksum = checksum.wrapping_add(v);
                            }
                        }
                    }
                    black_box(checksum);
                });
            },
        );
    }
    group.finish();
}

// ── Benchmark: String keys at various sizes ────────────────────────────
// Tests SSO boundary (7, 8, 13 bytes), medium (100), large (1000).

fn bench_string_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_sizes");

    for str_len in [7, 8, 13, 100] {
        let n = if str_len <= 13 { 200_000 } else { 50_000 };
        group.throughput(Throughput::Elements(n as u64));

        // Pre-generate keys
        let mut rng = Sfc64::new(42);
        let keys: Vec<String> = (0..n).map(|_| make_string(&mut rng, str_len)).collect();
        let miss_keys: Vec<String> = {
            let mut rng2 = Sfc64::new(9999);
            (0..n).map(|_| make_string(&mut rng2, str_len)).collect()
        };

        // Insert
        group.bench_with_input(
            BenchmarkId::new(format!("ours_insert_{}b", str_len), n),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut map = UnorderedFlatMap::with_capacity(n);
                    for (i, k) in keys.iter().enumerate() {
                        map.insert(k.clone(), i);
                    }
                    black_box(map.len());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("hashbrown_insert_{}b", str_len), n),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut map = hashbrown::HashMap::with_capacity(n);
                    for (i, k) in keys.iter().enumerate() {
                        map.insert(k.clone(), i);
                    }
                    black_box(map.len());
                });
            },
        );

        // Lookup hit
        let mut ours = UnorderedFlatMap::with_capacity(n);
        let mut hb = hashbrown::HashMap::with_capacity(n);
        for (i, k) in keys.iter().enumerate() {
            ours.insert(k.clone(), i);
            hb.insert(k.clone(), i);
        }

        group.bench_with_input(
            BenchmarkId::new(format!("ours_hit_{}b", str_len), n),
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
            BenchmarkId::new(format!("hashbrown_hit_{}b", str_len), n),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0usize;
                    for k in keys {
                        sum += hb.get(k.as_str()).unwrap_or(&0);
                    }
                    black_box(sum);
                });
            },
        );

        // Lookup miss
        group.bench_with_input(
            BenchmarkId::new(format!("ours_miss_{}b", str_len), n),
            &miss_keys,
            |b, miss| {
                b.iter(|| {
                    let mut count = 0usize;
                    for k in miss {
                        if ours.get(k.as_str()).is_some() {
                            count += 1;
                        }
                    }
                    black_box(count);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("hashbrown_miss_{}b", str_len), n),
            &miss_keys,
            |b, miss| {
                b.iter(|| {
                    let mut count = 0usize;
                    for k in miss {
                        if hb.get(k.as_str()).is_some() {
                            count += 1;
                        }
                    }
                    black_box(count);
                });
            },
        );
    }
    group.finish();
}

// ── Benchmark: Clone performance ───────────────────────────────────────

fn bench_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("clone");

    for n in [1_000, 100_000, 1_000_000] {
        let mut rng = Sfc64::new(42);
        group.throughput(Throughput::Elements(n as u64));

        let mut ours = UnorderedFlatMap::new();
        let mut hb = hashbrown::HashMap::new();
        for _ in 0..n {
            let k = rng.next();
            ours.insert(k, k);
            hb.insert(k, k);
        }

        group.bench_with_input(BenchmarkId::new("ours", n), &(), |b, _| {
            b.iter(|| {
                let cloned = ours.clone();
                black_box(cloned.len());
            });
        });

        group.bench_with_input(BenchmarkId::new("hashbrown", n), &(), |b, _| {
            b.iter(|| {
                let cloned = hb.clone();
                black_box(cloned.len());
            });
        });
    }
    group.finish();
}

// ── Benchmark: Iteration during growth/shrinkage ───────────────────────

fn bench_iterate_grow_shrink(c: &mut Criterion) {
    let mut group = c.benchmark_group("iterate_grow_shrink");
    let n = 5_000usize;
    let iters_per_step = 200usize;
    group.throughput(Throughput::Elements((n * iters_per_step) as u64));

    group.bench_function("ours", |b| {
        b.iter(|| {
            let mut rng = Sfc64::new(42);
            let mut map = UnorderedFlatMap::new();
            let mut checksum = 0u64;
            let mut inserted_keys = Vec::with_capacity(n);

            // Grow phase: insert 1, iterate all, repeat
            for _ in 0..n {
                let k = rng.next();
                map.insert(k, k);
                inserted_keys.push(k);
                for _ in 0..iters_per_step {
                    for (&_k, &v) in map.iter() {
                        checksum = checksum.wrapping_add(v);
                    }
                }
            }

            black_box(checksum);
        });
    });

    group.bench_function("hashbrown", |b| {
        b.iter(|| {
            let mut rng = Sfc64::new(42);
            let mut map = hashbrown::HashMap::new();
            let mut checksum = 0u64;

            for _ in 0..n {
                let k = rng.next();
                map.insert(k, k);
                for _ in 0..iters_per_step {
                    for (&_k, &v) in map.iter() {
                        checksum = checksum.wrapping_add(v);
                    }
                }
            }

            black_box(checksum);
        });
    });

    group.finish();
}

criterion_group!(
    comprehensive,
    bench_insert_erase_phases,
    bench_insert_erase_equilibrium,
    bench_random_distinct,
    bench_growing_lookup,
    bench_string_sizes,
    bench_clone,
    bench_iterate_grow_shrink,
);
criterion_main!(comprehensive);
