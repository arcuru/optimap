//! Generic benchmark helpers using the Map trait.
//!
//! Each helper takes a map type via generics and runs a standard benchmark.
//! Adding a new design = one line per benchmark function.

use criterion::{BenchmarkId, BenchmarkGroup, Throughput, black_box, measurement::WallTime};
use optimap::Map;

// ── Fast deterministic RNG (shared across all benchmark files) ──────────────

pub struct Sfc64 {
    a: u64, b: u64, c: u64, counter: u64,
}

impl Sfc64 {
    pub fn new(seed: u64) -> Self {
        let mut rng = Sfc64 { a: seed, b: seed, c: seed, counter: 1 };
        for _ in 0..12 { rng.next(); }
        rng
    }
    #[inline(always)]
    pub fn next(&mut self) -> u64 {
        let tmp = self.a.wrapping_add(self.b).wrapping_add(self.counter);
        self.counter += 1;
        self.a = self.b ^ (self.b >> 11);
        self.b = self.c.wrapping_add(self.c << 3);
        self.c = self.c.rotate_left(24).wrapping_add(tmp);
        tmp
    }
}

pub fn make_random_keys(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = Sfc64::new(seed);
    (0..n).map(|_| rng.next()).collect()
}

pub fn make_miss_keys(n: usize) -> Vec<u64> {
    make_random_keys(n, 9999)
}

// ── Generic benchmark functions ─────────────────────────────────────────────

/// Benchmark insert: clear + re-insert into a pre-warmed map.
pub fn bench_insert_for<M: Map<u64, u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    label: &str,
    keys: &[u64],
    capacity: usize,
) {
    let mut map = M::with_capacity(capacity);
    for (i, &k) in keys.iter().enumerate() { map.insert(k, i as u64); }

    group.bench_with_input(
        BenchmarkId::new(name, label),
        keys,
        |b, keys| {
            b.iter(|| {
                map.clear();
                for (i, &k) in keys.iter().enumerate() {
                    map.insert(k, i as u64);
                }
                black_box(map.len());
            });
        },
    );
}

/// Benchmark lookup hit on a pre-built map.
pub fn bench_lookup_hit_for<M: Map<u64, u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    label: &str,
    keys: &[u64],
    capacity: usize,
) {
    let mut map = M::with_capacity(capacity);
    for (i, &k) in keys.iter().enumerate() { map.insert(k, i as u64); }

    group.bench_with_input(
        BenchmarkId::new(name, label),
        keys,
        |b, keys| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in keys {
                    sum = sum.wrapping_add(*map.get(&k).unwrap_or(&0));
                }
                black_box(sum);
            });
        },
    );
}

/// Benchmark lookup miss on a pre-built map.
pub fn bench_lookup_miss_for<M: Map<u64, u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    label: &str,
    keys: &[u64],
    miss_keys: &[u64],
    capacity: usize,
) {
    let mut map = M::with_capacity(capacity);
    for (i, &k) in keys.iter().enumerate() { map.insert(k, i as u64); }

    group.bench_with_input(
        BenchmarkId::new(name, label),
        miss_keys,
        |b, miss_keys| {
            b.iter(|| {
                let mut count = 0u64;
                for &k in miss_keys {
                    if map.get(&k).is_some() { count += 1; }
                }
                black_box(count);
            });
        },
    );
}

/// Benchmark remove: fill then remove all keys.
pub fn bench_remove_for<M: Map<u64, u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    label: &str,
    keys: &[u64],
    capacity: usize,
) {
    let mut map = M::with_capacity(capacity);
    for (i, &k) in keys.iter().enumerate() { map.insert(k, i as u64); }

    group.bench_with_input(
        BenchmarkId::new(name, label),
        keys,
        |b, keys| {
            b.iter(|| {
                map.clear();
                for (i, &k) in keys.iter().enumerate() { map.insert(k, i as u64); }
                for &k in keys {
                    black_box(map.remove(&k));
                }
            });
        },
    );
}

/// Benchmark grow from empty (no pre-allocation).
pub fn bench_grow_for<M: Map<u64, u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    keys: &[u64],
    n: usize,
) {
    group.bench_with_input(BenchmarkId::new(name, n), keys, |b, keys| {
        b.iter(|| {
            let mut map = M::new();
            for (i, &k) in keys.iter().enumerate() {
                map.insert(k, i as u64);
            }
            black_box(map.len());
        });
    });
}

/// Benchmark with_capacity + fill.
pub fn bench_with_capacity_for<M: Map<u64, u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    keys: &[u64],
    n: usize,
) {
    group.bench_with_input(BenchmarkId::new(name, n), keys, |b, keys| {
        b.iter(|| {
            let mut map = M::with_capacity(n);
            for (i, &k) in keys.iter().enumerate() {
                map.insert(k, i as u64);
            }
            black_box(map.len());
        });
    });
}
