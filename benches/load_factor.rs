//! Load-factor sweep benchmarks.
//!
//! Tests lookup hit, miss, insert, and mixed performance across
//! a range of load factors from ~45% to ~87%.
//!
//! Methodology: allocate a table with a fixed number of groups,
//! fill it to a target load factor, then benchmark operations.
//! This isolates load factor from table size effects.

use criterion::{
    BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};

use optimap::UnorderedFlatMap;
use optimap::Splitsies;

// ── Fast RNG ────────────────────────────────────────────────────────────

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

// ── Helpers ─────────────────────────────────────────────────────────────

/// Pre-build a map with exactly `n` entries, using with_capacity to
/// control the internal table size. Returns (map, keys_in_map).
fn build_at_load(
    target_capacity: usize,
    num_entries: usize,
    seed: u64,
) -> (UnorderedFlatMap<u64, u64>, Vec<u64>) {
    let mut rng = Sfc64::new(seed);
    let mut map = UnorderedFlatMap::with_capacity(target_capacity);
    let mut keys = Vec::with_capacity(num_entries);
    for _ in 0..num_entries {
        let k = rng.next();
        map.insert(k, k);
        keys.push(k);
    }
    (map, keys)
}

fn build_split_at_load(
    target_capacity: usize,
    num_entries: usize,
    seed: u64,
) -> (Splitsies<u64, u64>, Vec<u64>) {
    let mut rng = Sfc64::new(seed);
    let mut map = Splitsies::with_capacity(target_capacity);
    let mut keys = Vec::with_capacity(num_entries);
    for _ in 0..num_entries {
        let k = rng.next();
        map.insert(k, k);
        keys.push(k);
    }
    (map, keys)
}

fn build_hb_at_load(
    target_capacity: usize,
    num_entries: usize,
    seed: u64,
) -> (hashbrown::HashMap<u64, u64>, Vec<u64>) {
    let mut rng = Sfc64::new(seed);
    let mut map = hashbrown::HashMap::with_capacity(target_capacity);
    let mut keys = Vec::with_capacity(num_entries);
    for _ in 0..num_entries {
        let k = rng.next();
        map.insert(k, k);
        keys.push(k);
    }
    (map, keys)
}

// ── Benchmark: Lookup hit at varying load factors ───────────────────────

fn bench_lookup_hit_by_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_factor_hit");

    // Use a fixed capacity that gives us ~8K groups (120K slots)
    // Then fill to various percentages.
    let capacity = 100_000; // will allocate enough groups for 100K
    let ops = 100_000u64;

    for load_pct in [45, 55, 65, 75, 85] {
        // Compute how many entries to get this load factor
        // First figure out how many slots we actually get
        let min_slots = (capacity * 8 + 6) / 7;
        let min_groups = (min_slots + 14) / 15;
        let mut num_groups = 1;
        while num_groups < min_groups { num_groups *= 2; }
        let total_slots = num_groups * 15;
        let num_entries = total_slots * load_pct / 100;

        let actual_lf = num_entries as f64 / total_slots as f64 * 100.0;
        group.throughput(Throughput::Elements(ops));

        let (ours, keys) = build_at_load(capacity, num_entries, 42);
        let (split, _) = build_split_at_load(capacity, num_entries, 42);
        let (hb, _) = build_hb_at_load(capacity, num_entries, 42);

        group.bench_with_input(
            BenchmarkId::new(format!("UFM_{:.0}pct", actual_lf), num_entries),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for i in 0..ops as usize {
                        let k = &keys[i % keys.len()];
                        sum = sum.wrapping_add(*ours.get(k).unwrap_or(&0));
                    }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("Splitsies_{:.0}pct", actual_lf), num_entries),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for i in 0..ops as usize {
                        let k = &keys[i % keys.len()];
                        sum = sum.wrapping_add(*split.get(k).unwrap_or(&0));
                    }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("hashbrown_{:.0}pct", actual_lf), num_entries),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for i in 0..ops as usize {
                        let k = &keys[i % keys.len()];
                        sum = sum.wrapping_add(*hb.get(k).unwrap_or(&0));
                    }
                    black_box(sum);
                });
            },
        );
    }
    group.finish();
}

// ── Benchmark: Lookup miss at varying load factors ──────────────────────

fn bench_lookup_miss_by_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_factor_miss");

    let capacity = 100_000;
    let ops = 100_000u64;

    // Generate miss keys (different seed)
    let mut miss_rng = Sfc64::new(9999);
    let miss_keys: Vec<u64> = (0..ops as usize).map(|_| miss_rng.next()).collect();

    for load_pct in [45, 55, 65, 75, 85] {
        let min_slots = (capacity * 8 + 6) / 7;
        let min_groups = (min_slots + 14) / 15;
        let mut num_groups = 1;
        while num_groups < min_groups { num_groups *= 2; }
        let total_slots = num_groups * 15;
        let num_entries = total_slots * load_pct / 100;

        let actual_lf = num_entries as f64 / total_slots as f64 * 100.0;
        group.throughput(Throughput::Elements(ops));

        let (ours, _) = build_at_load(capacity, num_entries, 42);
        let (split, _) = build_split_at_load(capacity, num_entries, 42);
        let (hb, _) = build_hb_at_load(capacity, num_entries, 42);

        group.bench_with_input(
            BenchmarkId::new(format!("UFM_{:.0}pct", actual_lf), num_entries),
            &miss_keys,
            |b, miss_keys| {
                b.iter(|| {
                    let mut count = 0u64;
                    for k in miss_keys {
                        if ours.get(k).is_some() { count += 1; }
                    }
                    black_box(count);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("Splitsies_{:.0}pct", actual_lf), num_entries),
            &miss_keys,
            |b, miss_keys| {
                b.iter(|| {
                    let mut count = 0u64;
                    for k in miss_keys {
                        if split.get(k).is_some() { count += 1; }
                    }
                    black_box(count);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("hashbrown_{:.0}pct", actual_lf), num_entries),
            &miss_keys,
            |b, miss_keys| {
                b.iter(|| {
                    let mut count = 0u64;
                    for k in miss_keys {
                        if hb.get(k).is_some() {
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

// ── Benchmark: Mixed hit+miss at varying load factors ───────────────────

fn bench_mixed_by_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_factor_mixed");

    let capacity = 100_000;
    let ops = 100_000u64;

    for load_pct in [45, 55, 65, 75, 85] {
        let min_slots = (capacity * 8 + 6) / 7;
        let min_groups = (min_slots + 14) / 15;
        let mut num_groups = 1;
        while num_groups < min_groups { num_groups *= 2; }
        let total_slots = num_groups * 15;
        let num_entries = total_slots * load_pct / 100;

        let actual_lf = num_entries as f64 / total_slots as f64 * 100.0;
        group.throughput(Throughput::Elements(ops));

        // 50% insert (into occupied keys = update), 30% lookup hit, 20% lookup miss
        let (ours_keys, _) = {
            let (m, k) = build_at_load(capacity, num_entries, 42);
            (k, m)
        };

        let mut mix_rng = Sfc64::new(777);
        let miss_keys: Vec<u64> = (0..ops as usize).map(|_| mix_rng.next()).collect();

        // Build operation sequence
        let op_keys: Vec<(u8, u64)> = {
            let mut rng = Sfc64::new(555);
            (0..ops as usize)
                .map(|i| {
                    let op = (rng.next() % 10) as u8;
                    let key = if op < 8 {
                        // 80% existing keys (hit)
                        ours_keys[i % ours_keys.len()]
                    } else {
                        // 20% miss keys
                        miss_keys[i % miss_keys.len()]
                    };
                    (op, key)
                })
                .collect()
        };

        let (mut ours, _) = build_at_load(capacity, num_entries, 42);
        let (mut split, _) = build_split_at_load(capacity, num_entries, 42);
        let (mut hb, _) = build_hb_at_load(capacity, num_entries, 42);

        group.bench_with_input(
            BenchmarkId::new(format!("UFM_{:.0}pct", actual_lf), num_entries),
            &op_keys,
            |b, ops| {
                b.iter(|| {
                    let mut checksum = 0u64;
                    for &(op, key) in ops {
                        match op {
                            0..=4 => { // 50% insert/update
                                ours.insert(key, key);
                            }
                            5..=7 => { // 30% lookup
                                if let Some(&v) = ours.get(&key) {
                                    checksum = checksum.wrapping_add(v);
                                }
                            }
                            _ => { // 20% remove
                                ours.remove(&key);
                            }
                        }
                    }
                    black_box(checksum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("Splitsies_{:.0}pct", actual_lf), num_entries),
            &op_keys,
            |b, ops| {
                b.iter(|| {
                    let mut checksum = 0u64;
                    for &(op, key) in ops {
                        match op {
                            0..=4 => {
                                split.insert(key, key);
                            }
                            5..=7 => {
                                if let Some(&v) = split.get(&key) {
                                    checksum = checksum.wrapping_add(v);
                                }
                            }
                            _ => {
                                split.remove(&key);
                            }
                        }
                    }
                    black_box(checksum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("hashbrown_{:.0}pct", actual_lf), num_entries),
            &op_keys,
            |b, ops| {
                b.iter(|| {
                    let mut checksum = 0u64;
                    for &(op, key) in ops {
                        match op {
                            0..=4 => {
                                hb.insert(key, key);
                            }
                            5..=7 => {
                                if let Some(&v) = hb.get(&key) {
                                    checksum = checksum.wrapping_add(v);
                                }
                            }
                            _ => {
                                hb.remove(&key);
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

// ── Benchmark: At 1M scale with load factor sweep ───────────────────────

fn bench_load_factor_1m(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_factor_1m");
    group.sample_size(10);

    // ~1M capacity → large enough to see cache effects
    let capacity = 1_000_000;
    let ops = 500_000u64;

    for load_pct in [45, 65, 85] {
        let min_slots = (capacity * 8 + 6) / 7;
        let min_groups = (min_slots + 14) / 15;
        let mut num_groups = 1;
        while num_groups < min_groups { num_groups *= 2; }
        let total_slots = num_groups * 15;
        let num_entries = total_slots * load_pct / 100;

        let actual_lf = num_entries as f64 / total_slots as f64 * 100.0;
        group.throughput(Throughput::Elements(ops));

        let (ours, keys) = build_at_load(capacity, num_entries, 42);
        let (split, _) = build_split_at_load(capacity, num_entries, 42);
        let (hb, _) = build_hb_at_load(capacity, num_entries, 42);

        let mut miss_rng = Sfc64::new(9999);
        let miss_keys: Vec<u64> = (0..ops as usize).map(|_| miss_rng.next()).collect();

        // Hit
        group.bench_with_input(
            BenchmarkId::new(format!("UFM_hit_{:.0}pct", actual_lf), num_entries),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for i in 0..ops as usize {
                        sum = sum.wrapping_add(*ours.get(&keys[i % keys.len()]).unwrap_or(&0));
                    }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("Splitsies_hit_{:.0}pct", actual_lf), num_entries),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for i in 0..ops as usize {
                        sum = sum.wrapping_add(*split.get(&keys[i % keys.len()]).unwrap_or(&0));
                    }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("hashbrown_hit_{:.0}pct", actual_lf), num_entries),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for i in 0..ops as usize {
                        sum = sum.wrapping_add(*hb.get(&keys[i % keys.len()]).unwrap_or(&0));
                    }
                    black_box(sum);
                });
            },
        );

        // Miss
        group.bench_with_input(
            BenchmarkId::new(format!("UFM_miss_{:.0}pct", actual_lf), num_entries),
            &miss_keys,
            |b, miss_keys| {
                b.iter(|| {
                    let mut count = 0u64;
                    for k in miss_keys {
                        if ours.get(k).is_some() { count += 1; }
                    }
                    black_box(count);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("Splitsies_miss_{:.0}pct", actual_lf), num_entries),
            &miss_keys,
            |b, miss_keys| {
                b.iter(|| {
                    let mut count = 0u64;
                    for k in miss_keys {
                        if split.get(k).is_some() { count += 1; }
                    }
                    black_box(count);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("hashbrown_miss_{:.0}pct", actual_lf), num_entries),
            &miss_keys,
            |b, miss_keys| {
                b.iter(|| {
                    let mut count = 0u64;
                    for k in miss_keys {
                        if hb.get(k).is_some() { count += 1; }
                    }
                    black_box(count);
                });
            },
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
