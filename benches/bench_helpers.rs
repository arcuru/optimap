//! Generic benchmark helpers using the Map trait.
//!
//! Each helper takes a map type via generics and runs a standard benchmark.
//! Adding a new design = one line per benchmark function.

use criterion::{BenchmarkId, BenchmarkGroup, black_box, measurement::WallTime};
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

// ── Table geometry (GROUP_SIZE=15 designs: UFM, Splitsies, Gaps) ────────────

pub const GROUP_SIZE: usize = 15;

pub fn entries_for_load(capacity: usize, load_pct: usize) -> usize {
    let min_slots = (capacity * 8 + 6) / 7;
    let min_groups = (min_slots + GROUP_SIZE - 1) / GROUP_SIZE;
    let mut num_groups = 1;
    while num_groups < min_groups { num_groups *= 2; }
    let total_slots = num_groups * GROUP_SIZE;
    total_slots * load_pct / 100
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

/// Benchmark clone on a pre-built map.
pub fn bench_clone_for<M: Map<u64, u64> + Clone>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    keys: &[u64],
    n: usize,
) {
    let mut map = M::with_capacity(n);
    for (i, &k) in keys.iter().enumerate() { map.insert(k, i as u64); }
    group.bench_with_input(BenchmarkId::new(name, n), &(), |b, _| {
        b.iter(|| black_box(map.clone()));
    });
}

/// Build a map at a specific load level, returning (map, keys).
pub fn build_map_at_load<M: Map<u64, u64>>(
    target_capacity: usize,
    num_entries: usize,
    seed: u64,
) -> (M, Vec<u64>) {
    let mut rng = Sfc64::new(seed);
    let mut map = M::with_capacity(target_capacity);
    let mut keys = Vec::with_capacity(num_entries);
    for _ in 0..num_entries {
        let k = rng.next();
        map.insert(k, k);
        keys.push(k);
    }
    (map, keys)
}

/// Benchmark lookup hit at a specific load level (fixed ops count, cycling keys).
pub fn bench_load_hit_for<M: Map<u64, u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    capacity: usize,
    num_entries: usize,
    ops: u64,
    seed: u64,
) {
    let (map, keys) = build_map_at_load::<M>(capacity, num_entries, seed);
    group.bench_with_input(
        BenchmarkId::new(name, num_entries),
        &keys,
        |b, keys| {
            b.iter(|| {
                let mut sum = 0u64;
                for i in 0..ops as usize {
                    sum = sum.wrapping_add(*map.get(&keys[i % keys.len()]).unwrap_or(&0));
                }
                black_box(sum);
            });
        },
    );
}

/// Benchmark lookup miss at a specific load level.
pub fn bench_load_miss_for<M: Map<u64, u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    capacity: usize,
    num_entries: usize,
    miss_keys: &[u64],
    seed: u64,
) {
    let (map, _) = build_map_at_load::<M>(capacity, num_entries, seed);
    group.bench_with_input(
        BenchmarkId::new(name, num_entries),
        miss_keys,
        |b, miss_keys| {
            b.iter(|| {
                let mut count = 0u64;
                for k in miss_keys {
                    if map.get(k).is_some() { count += 1; }
                }
                black_box(count);
            });
        },
    );
}

/// Benchmark mixed ops (50% insert, 30% lookup, 20% remove) at a specific load.
pub fn bench_load_mixed_for<M: Map<u64, u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    capacity: usize,
    num_entries: usize,
    op_keys: &[(u8, u64)],
    seed: u64,
) {
    let (mut map, _) = build_map_at_load::<M>(capacity, num_entries, seed);
    group.bench_with_input(
        BenchmarkId::new(name, num_entries),
        op_keys,
        |b, ops| {
            b.iter(|| {
                let mut checksum = 0u64;
                for &(op, key) in ops {
                    match op {
                        0..=4 => { map.insert(key, key); }
                        5..=7 => {
                            if let Some(&v) = map.get(&key) {
                                checksum = checksum.wrapping_add(v);
                            }
                        }
                        _ => { map.remove(&key); }
                    }
                }
                black_box(checksum);
            });
        },
    );
}

/// Benchmark post-delete lookup: build, remove half, measure lookup of all keys.
pub fn bench_post_delete_for<M: Map<u64, u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    label: &str,
    keys: &[u64],
    capacity: usize,
) {
    let half = keys.len() / 2;
    let mut map = M::with_capacity(capacity);
    for (i, &k) in keys.iter().enumerate() { map.insert(k, i as u64); }
    for &k in &keys[..half] { map.remove(&k); }

    group.bench_with_input(BenchmarkId::new(name, label), keys, |b, keys| {
        b.iter(|| {
            let mut sum = 0u64;
            for &k in keys {
                if let Some(&v) = map.get(&k) {
                    sum = sum.wrapping_add(v);
                }
            }
            black_box(sum);
        });
    });
}

/// Benchmark remove+reinsert pattern on a pre-built map.
pub fn bench_remove_reinsert_for<M: Map<u64, u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    label: &str,
    keys: &[u64],
    op_keys: &[u64],
    capacity: usize,
) {
    let mut map = M::with_capacity(capacity);
    for (i, &k) in keys.iter().enumerate() { map.insert(k, i as u64); }

    group.bench_with_input(BenchmarkId::new(name, label), op_keys, |b, op_keys| {
        b.iter(|| {
            let mut checksum = 0u64;
            for &k in op_keys {
                if let Some(v) = map.remove(&k) { checksum = checksum.wrapping_add(v); }
                map.insert(k, checksum);
            }
            black_box(checksum);
        });
    });
}

/// Benchmark miss ratio sweep: lookup with a mix of hit/miss keys.
pub fn bench_miss_ratio_for<M: Map<u64, u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    keys: &[u64],
    lookup_keys: &[u64],
    capacity: usize,
) {
    let mut map = M::with_capacity(capacity);
    for (i, &k) in keys.iter().enumerate() { map.insert(k, i as u64); }

    group.bench_with_input(
        BenchmarkId::new(name, keys.len()),
        lookup_keys,
        |b, keys| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in keys {
                    if let Some(&v) = map.get(&k) {
                        sum = sum.wrapping_add(v);
                    }
                }
                black_box(sum);
            });
        },
    );
}

/// Benchmark equilibrium churn: insert + remove in a loop.
pub fn bench_churn_for<M: Map<u64, u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    label: &str,
    ops: u64,
    mask: u64,
) {
    group.bench_function(BenchmarkId::new(name, label), |b| {
        b.iter(|| {
            let mut map = M::new();
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

/// Benchmark read-heavy workload: mixed ops on a pre-built map.
pub fn bench_mixed_workload_for<M: Map<u64, u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    label: &str,
    keys: &[u64],
    op_seq: &[(u8, u64)],
    capacity: usize,
) {
    let mut map = M::with_capacity(capacity);
    for (i, &k) in keys.iter().enumerate() { map.insert(k, i as u64); }

    group.bench_with_input(BenchmarkId::new(name, label), op_seq, |b, ops| {
        b.iter(|| {
            let mut checksum = 0u64;
            for &(op, key) in ops {
                match op {
                    0..=94 => {
                        if let Some(&v) = map.get(&key) {
                            checksum = checksum.wrapping_add(v);
                        }
                    }
                    95..=97 => { map.insert(key, key); }
                    _ => { map.remove(&key); }
                }
            }
            black_box(checksum);
        });
    });
}

/// Benchmark write-heavy workload: 50% read, 30% insert, 20% remove.
pub fn bench_write_heavy_for<M: Map<u64, u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    label: &str,
    keys: &[u64],
    op_seq: &[(u8, u64)],
    capacity: usize,
) {
    let mut map = M::with_capacity(capacity);
    for (i, &k) in keys.iter().enumerate() { map.insert(k, i as u64); }

    group.bench_with_input(BenchmarkId::new(name, label), op_seq, |b, ops| {
        b.iter(|| {
            let mut checksum = 0u64;
            for &(op, key) in ops {
                match op {
                    0..=4 => {
                        if let Some(&v) = map.get(&key) {
                            checksum = checksum.wrapping_add(v);
                        }
                    }
                    5..=7 => { map.insert(key, key); }
                    _ => { map.remove(&key); }
                }
            }
            black_box(checksum);
        });
    });
}

/// Benchmark high-load hit: lookup on a pre-built map (fixed ops, cycling).
pub fn bench_high_load_hit_for<M: Map<u64, u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    keys: &[u64],
    capacity: usize,
    ops: u64,
) {
    let mut map = M::with_capacity(capacity);
    for (i, &k) in keys.iter().enumerate() { map.insert(k, i as u64); }

    group.bench_with_input(BenchmarkId::new(name, keys.len()), keys, |b, keys| {
        b.iter(|| {
            let mut sum = 0u64;
            for i in 0..ops as usize {
                sum = sum.wrapping_add(*map.get(&keys[i % keys.len()]).unwrap_or(&0));
            }
            black_box(sum);
        });
    });
}

/// Benchmark high-load miss.
pub fn bench_high_load_miss_for<M: Map<u64, u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    num_entries: usize,
    miss_keys: &[u64],
    keys: &[u64],
    capacity: usize,
) {
    let mut map = M::with_capacity(capacity);
    for (i, &k) in keys.iter().enumerate() { map.insert(k, i as u64); }

    group.bench_with_input(BenchmarkId::new(name, num_entries), miss_keys, |b, mkeys| {
        b.iter(|| {
            let mut count = 0u64;
            for &k in mkeys { if map.get(&k).is_some() { count += 1; } }
            black_box(count);
        });
    });
}
