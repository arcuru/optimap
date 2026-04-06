//! Key distribution and value size sensitivity benchmarks.
//!
//! Tests how different key patterns and value sizes affect performance.
//! All benchmarks use pre-allocated, pre-warmed tables at 70% load factor
//! to isolate the distribution/size effect from allocation noise.
//!
//! Key distributions tested:
//! - Random: uniformly distributed u64 (baseline)
//! - Sequential: 0, 1, 2, ... N (tests hash quality on low-entropy input)
//! - Byte-swapped: sequential with bytes reversed (high bits vary, low bits constant)
//!
//! Value sizes tested: u64 (8B), [u8;64], [u8;128], [u8;256]

use criterion::{
    BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};

use optimap::UnorderedFlatMap;
use optimap::Splitsies;

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

// Large: 8192 groups, 122880 slots
const LARGE_CAPACITY: usize = 107_520;
// Medium: 1024 groups, 15360 slots
const MEDIUM_CAPACITY: usize = 13_440;
const LOAD_PCT: usize = 70;

// ── Key generators ──────────────────────────────────────────────────────────

fn make_random_keys(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = Sfc64::new(seed);
    (0..n).map(|_| rng.next()).collect()
}

fn make_sequential_keys(n: usize) -> Vec<u64> {
    (0..n as u64).collect()
}

fn make_byteswapped_keys(n: usize) -> Vec<u64> {
    (0..n as u64).map(|i| i.swap_bytes()).collect()
}

// ── Lookup hit by key distribution ──────────────────────────────────────────

fn bench_lookup_hit_by_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("distribution/lookup_hit");
    let n = entries_for_load(LARGE_CAPACITY, LOAD_PCT);
    group.throughput(Throughput::Elements(n as u64));

    let distributions: Vec<(&str, Vec<u64>)> = vec![
        ("random", make_random_keys(n, 42)),
        ("sequential", make_sequential_keys(n)),
        ("byteswapped", make_byteswapped_keys(n)),
    ];

    for (dist_name, keys) in &distributions {
        let mut ours = UnorderedFlatMap::with_capacity(LARGE_CAPACITY);
        let mut split = Splitsies::with_capacity(LARGE_CAPACITY);
        let mut hb = hashbrown::HashMap::with_capacity(LARGE_CAPACITY);
        for (i, &k) in keys.iter().enumerate() {
            ours.insert(k, i as u64);
            split.insert(k, i as u64);
            hb.insert(k, i as u64);
        }

        group.bench_with_input(
            BenchmarkId::new(format!("UFM_{dist_name}"), n),
            keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &k in keys { sum = sum.wrapping_add(*ours.get(&k).unwrap_or(&0)); }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("Splitsies_{dist_name}"), n),
            keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &k in keys { sum = sum.wrapping_add(*split.get(&k).unwrap_or(&0)); }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("hashbrown_{dist_name}"), n),
            keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &k in keys { sum = sum.wrapping_add(*hb.get(&k).unwrap_or(&0)); }
                    black_box(sum);
                });
            },
        );
    }
    group.finish();
}

// ── Lookup miss by key distribution ─────────────────────────────────────────

fn bench_lookup_miss_by_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("distribution/lookup_miss");
    let n = entries_for_load(LARGE_CAPACITY, LOAD_PCT);
    let num_ops = n;
    group.throughput(Throughput::Elements(num_ops as u64));

    let distributions: Vec<(&str, Vec<u64>, Vec<u64>)> = vec![
        (
            "random",
            make_random_keys(n, 42),
            make_random_keys(num_ops, 9999), // miss keys from different seed
        ),
        (
            "sequential",
            make_sequential_keys(n),
            (n as u64..n as u64 + num_ops as u64).collect(), // keys above inserted range
        ),
        (
            "byteswapped",
            make_byteswapped_keys(n),
            (n as u64..n as u64 + num_ops as u64).map(|i| i.swap_bytes()).collect(),
        ),
    ];

    for (dist_name, insert_keys, miss_keys) in &distributions {
        let mut ours = UnorderedFlatMap::with_capacity(LARGE_CAPACITY);
        let mut split = Splitsies::with_capacity(LARGE_CAPACITY);
        let mut hb = hashbrown::HashMap::with_capacity(LARGE_CAPACITY);
        for (i, &k) in insert_keys.iter().enumerate() {
            ours.insert(k, i as u64);
            split.insert(k, i as u64);
            hb.insert(k, i as u64);
        }

        group.bench_with_input(
            BenchmarkId::new(format!("UFM_{dist_name}"), n),
            miss_keys,
            |b, miss_keys| {
                b.iter(|| {
                    let mut count = 0u64;
                    for &k in miss_keys { if ours.get(&k).is_some() { count += 1; } }
                    black_box(count);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("Splitsies_{dist_name}"), n),
            miss_keys,
            |b, miss_keys| {
                b.iter(|| {
                    let mut count = 0u64;
                    for &k in miss_keys { if split.get(&k).is_some() { count += 1; } }
                    black_box(count);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("hashbrown_{dist_name}"), n),
            miss_keys,
            |b, miss_keys| {
                b.iter(|| {
                    let mut count = 0u64;
                    for &k in miss_keys { if hb.get(&k).is_some() { count += 1; } }
                    black_box(count);
                });
            },
        );
    }
    group.finish();
}

// ── Insert by key distribution ──────────────────────────────────────────────

fn bench_insert_by_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("distribution/insert");
    let n = entries_for_load(LARGE_CAPACITY, LOAD_PCT);
    group.throughput(Throughput::Elements(n as u64));

    let distributions: Vec<(&str, Vec<u64>)> = vec![
        ("random", make_random_keys(n, 42)),
        ("sequential", make_sequential_keys(n)),
        ("byteswapped", make_byteswapped_keys(n)),
    ];

    for (dist_name, keys) in &distributions {
        let mut ours = UnorderedFlatMap::with_capacity(LARGE_CAPACITY);
        for (i, &k) in keys.iter().enumerate() { ours.insert(k, i as u64); }

        group.bench_with_input(
            BenchmarkId::new(format!("UFM_{dist_name}"), n),
            keys,
            |b, keys| {
                b.iter(|| {
                    ours.clear();
                    for (i, &k) in keys.iter().enumerate() { ours.insert(k, i as u64); }
                    black_box(ours.len());
                });
            },
        );

        let mut split = Splitsies::with_capacity(LARGE_CAPACITY);
        for (i, &k) in keys.iter().enumerate() { split.insert(k, i as u64); }

        group.bench_with_input(
            BenchmarkId::new(format!("Splitsies_{dist_name}"), n),
            keys,
            |b, keys| {
                b.iter(|| {
                    split.clear();
                    for (i, &k) in keys.iter().enumerate() { split.insert(k, i as u64); }
                    black_box(split.len());
                });
            },
        );

        let mut hb = hashbrown::HashMap::with_capacity(LARGE_CAPACITY);
        for (i, &k) in keys.iter().enumerate() { hb.insert(k, i as u64); }

        group.bench_with_input(
            BenchmarkId::new(format!("hashbrown_{dist_name}"), n),
            keys,
            |b, keys| {
                b.iter(|| {
                    hb.clear();
                    for (i, &k) in keys.iter().enumerate() { hb.insert(k, i as u64); }
                    black_box(hb.len());
                });
            },
        );
    }
    group.finish();
}

// ── String key sizes ────────────────────────────────────────────────────────

fn bench_string_key_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("distribution/string_keys");
    let n = entries_for_load(MEDIUM_CAPACITY, LOAD_PCT);
    group.throughput(Throughput::Elements(n as u64));

    for &len in &[7, 8, 13, 24, 100] {
        let keys: Vec<String> = {
            let mut rng = Sfc64::new(42);
            (0..n)
                .map(|_| {
                    (0..len)
                        .map(|_| (b'a' + (rng.next() % 26) as u8) as char)
                        .collect()
                })
                .collect()
        };

        let mut ours: UnorderedFlatMap<String, u64> =
            UnorderedFlatMap::with_capacity(MEDIUM_CAPACITY);
        let mut split: Splitsies<String, u64> =
            Splitsies::with_capacity(MEDIUM_CAPACITY);
        let mut hb: hashbrown::HashMap<String, u64> =
            hashbrown::HashMap::with_capacity(MEDIUM_CAPACITY);
        for (i, k) in keys.iter().enumerate() {
            ours.insert(k.clone(), i as u64);
            split.insert(k.clone(), i as u64);
            hb.insert(k.clone(), i as u64);
        }

        group.bench_with_input(
            BenchmarkId::new(format!("UFM_{len}b"), n),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for k in keys {
                        sum = sum.wrapping_add(*ours.get(k.as_str()).unwrap_or(&0));
                    }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("Splitsies_{len}b"), n),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for k in keys {
                        sum = sum.wrapping_add(*split.get(k.as_str()).unwrap_or(&0));
                    }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("hashbrown_{len}b"), n),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for k in keys {
                        sum = sum.wrapping_add(*hb.get(k.as_str()).unwrap_or(&0));
                    }
                    black_box(sum);
                });
            },
        );
    }
    group.finish();
}

// ── Value size sensitivity ──────────────────────────────────────────────────

macro_rules! bench_value_size {
    ($c:expr, $group:expr, $name:expr, $val_type:ty, $val:expr, $keys:expr, $capacity:expr) => {{
        let keys = $keys;
        let val: $val_type = $val;

        let mut ours: UnorderedFlatMap<u64, $val_type> =
            UnorderedFlatMap::with_capacity($capacity);
        for &k in &keys { ours.insert(k, val); }

        let mut split: Splitsies<u64, $val_type> =
            Splitsies::with_capacity($capacity);
        for &k in &keys { split.insert(k, val); }

        let mut hb: hashbrown::HashMap<u64, $val_type> =
            hashbrown::HashMap::with_capacity($capacity);
        for &k in &keys { hb.insert(k, val); }

        // Insert (clear + refill)
        $group.bench_with_input(
            BenchmarkId::new(format!("UFM_insert_{}", $name), keys.len()),
            &keys,
            |b, keys| {
                b.iter(|| {
                    ours.clear();
                    for &k in keys.iter() { ours.insert(k, val); }
                    black_box(ours.len());
                });
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(format!("Splitsies_insert_{}", $name), keys.len()),
            &keys,
            |b, keys| {
                b.iter(|| {
                    split.clear();
                    for &k in keys.iter() { split.insert(k, val); }
                    black_box(split.len());
                });
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(format!("hashbrown_insert_{}", $name), keys.len()),
            &keys,
            |b, keys| {
                b.iter(|| {
                    hb.clear();
                    for &k in keys.iter() { hb.insert(k, val); }
                    black_box(hb.len());
                });
            },
        );

        // Lookup hit
        $group.bench_with_input(
            BenchmarkId::new(format!("UFM_hit_{}", $name), keys.len()),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &k in keys { sum = sum.wrapping_add(ours.get(&k).map(|v| v[0] as u64).unwrap_or(0)); }
                    black_box(sum);
                });
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(format!("Splitsies_hit_{}", $name), keys.len()),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &k in keys { sum = sum.wrapping_add(split.get(&k).map(|v| v[0] as u64).unwrap_or(0)); }
                    black_box(sum);
                });
            },
        );

        $group.bench_with_input(
            BenchmarkId::new(format!("hashbrown_hit_{}", $name), keys.len()),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &k in keys { sum = sum.wrapping_add(hb.get(&k).map(|v| v[0] as u64).unwrap_or(0)); }
                    black_box(sum);
                });
            },
        );
    }};
}

fn bench_value_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("distribution/value_size");
    let n = entries_for_load(MEDIUM_CAPACITY, LOAD_PCT);
    let keys = make_random_keys(n, 42);
    group.throughput(Throughput::Elements(n as u64));

    bench_value_size!(c, group, "64B", [u8; 64], [0u8; 64], keys.clone(), MEDIUM_CAPACITY);
    bench_value_size!(c, group, "128B", [u8; 128], [0u8; 128], keys.clone(), MEDIUM_CAPACITY);
    bench_value_size!(c, group, "256B", [u8; 256], [0u8; 256], keys.clone(), MEDIUM_CAPACITY);

    group.finish();
}

criterion_group!(
    distributions,
    bench_lookup_hit_by_distribution,
    bench_lookup_miss_by_distribution,
    bench_insert_by_distribution,
    bench_string_key_sizes,
    bench_value_sizes,
);
criterion_main!(distributions);
