//! Realistic mixed-operation workload benchmarks.
//!
//! These benchmarks combine multiple operations in a single measurement
//! to test how the map performs under realistic conditions. Maps may
//! grow during the benchmark; this is intentional.

use criterion::{
    BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};

use unordered_flat_map::UnorderedFlatMap;

// ── Fast deterministic RNG ──────────────────────────────────────────────────

struct Sfc64 {
    a: u64, b: u64, c: u64, counter: u64,
}

impl Sfc64 {
    fn new(seed: u64) -> Self {
        let mut rng = Sfc64 { a: seed, b: seed, c: seed, counter: 1 };
        for _ in 0..12 { rng.next(); }
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
}

// ── Table geometry ──────────────────────────────────────────────────────────

const GROUP_SIZE: usize = 15;

fn entries_for_load(capacity: usize, load_pct: usize) -> usize {
    let min_slots = (capacity * 8 + 6) / 7;
    let min_groups = (min_slots + GROUP_SIZE - 1) / GROUP_SIZE;
    let mut num_groups = 1;
    while num_groups < min_groups { num_groups *= 2; }
    let total_slots = num_groups * GROUP_SIZE;
    total_slots * load_pct / 100
}

fn make_random_keys(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = Sfc64::new(seed);
    (0..n).map(|_| rng.next()).collect()
}

const LARGE_CAPACITY: usize = 107_520;
const LOAD_PCT: usize = 70;

// ── Workload: Equilibrium Churn ─────────────────────────────────────────────

fn bench_equilibrium_churn(c: &mut Criterion) {
    let mut group = c.benchmark_group("workload/churn");
    let ops = 2_000_000u64;

    for &(name, mask) in &[("4K", 0xFFFu64), ("64K", 0xFFFFu64), ("1M", 0xF_FFFFu64)] {
        group.throughput(Throughput::Elements(ops));
        if mask >= 0xF_FFFF { group.sample_size(10); }

        group.bench_function(BenchmarkId::new("ours", name), |b| {
            b.iter(|| {
                let mut map = UnorderedFlatMap::new();
                let mut rng = Sfc64::new(42);
                let mut checksum = 0u64;
                for _ in 0..ops {
                    let k = rng.next() & mask;
                    map.insert(k, k);
                    let k2 = rng.next() & mask;
                    if let Some(v) = map.remove(&k2) {
                        checksum = checksum.wrapping_add(v);
                    }
                }
                black_box(checksum);
            });
        });

        group.bench_function(BenchmarkId::new("hashbrown", name), |b| {
            b.iter(|| {
                let mut map = hashbrown::HashMap::new();
                let mut rng = Sfc64::new(42);
                let mut checksum = 0u64;
                for _ in 0..ops {
                    let k = rng.next() & mask;
                    map.insert(k, k);
                    let k2 = rng.next() & mask;
                    if let Some(v) = map.remove(&k2) {
                        checksum = checksum.wrapping_add(v);
                    }
                }
                black_box(checksum);
            });
        });
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
                let op = (rng.next() % 100) as u8;
                let key = if op < 80 {
                    keys[i % keys.len()] // 80% hit
                } else if op < 95 {
                    miss_keys[i % miss_keys.len()] // 15% miss
                } else if op < 98 {
                    rng.next() // 3% insert new
                } else {
                    keys[i % keys.len()] // 2% remove existing
                };
                (op, key)
            })
            .collect()
    };

    let mut ours = UnorderedFlatMap::with_capacity(LARGE_CAPACITY);
    let mut hb = hashbrown::HashMap::with_capacity(LARGE_CAPACITY);
    for (i, &k) in keys.iter().enumerate() {
        ours.insert(k, i as u64);
        hb.insert(k, i as u64);
    }

    group.bench_with_input(BenchmarkId::new("ours", n), &op_seq, |b, ops| {
        b.iter(|| {
            let mut checksum = 0u64;
            for &(op, key) in ops {
                match op {
                    0..=94 => { // 95% lookup (80% hit + 15% miss)
                        if let Some(&v) = ours.get(&key) {
                            checksum = checksum.wrapping_add(v);
                        }
                    }
                    95..=97 => { ours.insert(key, key); } // 3% insert
                    _ => { ours.remove(&key); } // 2% remove
                }
            }
            black_box(checksum);
        });
    });

    group.bench_with_input(BenchmarkId::new("hashbrown", n), &op_seq, |b, ops| {
        b.iter(|| {
            let mut checksum = 0u64;
            for &(op, key) in ops {
                match op {
                    0..=94 => {
                        if let Some(&v) = hb.get(&key) {
                            checksum = checksum.wrapping_add(v);
                        }
                    }
                    95..=97 => { hb.insert(key, key); }
                    _ => { hb.remove(&key); }
                }
            }
            black_box(checksum);
        });
    });
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
                let op = (rng.next() % 10) as u8;
                let key = if op < 5 {
                    keys[i % keys.len()] // lookup existing
                } else {
                    rng.next() // insert/remove random
                };
                (op, key)
            })
            .collect()
    };

    let mut ours = UnorderedFlatMap::with_capacity(LARGE_CAPACITY);
    let mut hb = hashbrown::HashMap::with_capacity(LARGE_CAPACITY);
    for (i, &k) in keys.iter().enumerate() {
        ours.insert(k, i as u64);
        hb.insert(k, i as u64);
    }

    group.bench_with_input(BenchmarkId::new("ours", n), &op_seq, |b, ops| {
        b.iter(|| {
            let mut checksum = 0u64;
            for &(op, key) in ops {
                match op {
                    0..=4 => { // 50% lookup
                        if let Some(&v) = ours.get(&key) {
                            checksum = checksum.wrapping_add(v);
                        }
                    }
                    5..=7 => { ours.insert(key, key); } // 30% insert
                    _ => { ours.remove(&key); } // 20% remove
                }
            }
            black_box(checksum);
        });
    });

    group.bench_with_input(BenchmarkId::new("hashbrown", n), &op_seq, |b, ops| {
        b.iter(|| {
            let mut checksum = 0u64;
            for &(op, key) in ops {
                match op {
                    0..=4 => {
                        if let Some(&v) = hb.get(&key) {
                            checksum = checksum.wrapping_add(v);
                        }
                    }
                    5..=7 => { hb.insert(key, key); }
                    _ => { hb.remove(&key); }
                }
            }
            black_box(checksum);
        });
    });
    group.finish();
}

// ── Workload: Counting / Aggregation (entry API) ────────────────────────────

fn bench_counting(c: &mut Criterion) {
    let mut group = c.benchmark_group("workload/counting");
    let ops = 5_000_000u64;
    group.sample_size(10);

    for &(name, distinct_pct) in &[("5pct", 5u64), ("50pct", 50), ("100pct", 100)] {
        let distinct = (ops * distinct_pct / 100).max(1);
        group.throughput(Throughput::Elements(ops));

        group.bench_function(BenchmarkId::new("ours", name), |b| {
            b.iter(|| {
                let mut map = UnorderedFlatMap::new();
                let mut rng = Sfc64::new(42);
                for _ in 0..ops {
                    let k = rng.next() % distinct;
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
                    let k = rng.next() % distinct;
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
        let half = n / 2;
        group.throughput(Throughput::Elements(n as u64));

        // Build, remove half, measure lookup of all N keys (half hit, half miss)
        let mut ours = UnorderedFlatMap::with_capacity(capacity);
        let mut hb = hashbrown::HashMap::with_capacity(capacity);
        for (i, &k) in keys.iter().enumerate() {
            ours.insert(k, i as u64);
            hb.insert(k, i as u64);
        }
        for &k in &keys[..half] {
            ours.remove(&k);
            hb.remove(&k);
        }

        group.bench_with_input(BenchmarkId::new("ours", name), &keys, |b, keys| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in keys {
                    if let Some(&v) = ours.get(&k) {
                        sum = sum.wrapping_add(v);
                    }
                }
                black_box(sum);
            });
        });

        group.bench_with_input(BenchmarkId::new("hashbrown", name), &keys, |b, keys| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in keys {
                    if let Some(&v) = hb.get(&k) {
                        sum = sum.wrapping_add(v);
                    }
                }
                black_box(sum);
            });
        });
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

    let mut ours = UnorderedFlatMap::with_capacity(LARGE_CAPACITY);
    let mut hb = hashbrown::HashMap::with_capacity(LARGE_CAPACITY);
    for (i, &k) in hit_keys.iter().enumerate() {
        ours.insert(k, i as u64);
        hb.insert(k, i as u64);
    }

    for &miss_pct in &[0, 25, 50, 75, 100] {
        // Build lookup sequence with target miss ratio
        let lookup_keys: Vec<u64> = (0..ops)
            .map(|i| {
                if (i * 100 / ops) < miss_pct {
                    miss_keys[i % miss_keys.len()]
                } else {
                    hit_keys[i % hit_keys.len()]
                }
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new(format!("ours_{}miss", miss_pct), n),
            &lookup_keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &k in keys {
                        if let Some(&v) = ours.get(&k) {
                            sum = sum.wrapping_add(v);
                        }
                    }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("hb_{}miss", miss_pct), n),
            &lookup_keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &k in keys {
                        if let Some(&v) = hb.get(&k) {
                            sum = sum.wrapping_add(v);
                        }
                    }
                    black_box(sum);
                });
            },
        );
    }
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
);
criterion_main!(workloads);
