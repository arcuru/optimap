//! Allocation, growth, clone, and construction cost benchmarks.
//!
//! These benchmarks intentionally include allocation overhead — they measure
//! costs that happen once per table lifetime. Use these to understand the
//! cost of creating, growing, and copying hash maps.

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

fn make_random_keys(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = Sfc64::new(seed);
    (0..n).map(|_| rng.next()).collect()
}

// ── Grow from empty (no pre-allocation) ─────────────────────────────────────

fn bench_grow_from_empty(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction/grow_from_empty");

    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        let keys = make_random_keys(n, 42);
        group.throughput(Throughput::Elements(n as u64));

        if n >= 1_000_000 { group.sample_size(10); }

        group.bench_with_input(BenchmarkId::new("ours", n), &keys, |b, keys| {
            b.iter(|| {
                let mut map = UnorderedFlatMap::new();
                for (i, &k) in keys.iter().enumerate() {
                    map.insert(k, i as u64);
                }
                black_box(map.len());
            });
        });

        group.bench_with_input(BenchmarkId::new("hashbrown", n), &keys, |b, keys| {
            b.iter(|| {
                let mut map = hashbrown::HashMap::new();
                for (i, &k) in keys.iter().enumerate() {
                    map.insert(k, i as u64);
                }
                black_box(map.len());
            });
        });
    }
    group.finish();
}

// ── Insert with pre-allocation (cold pages) ─────────────────────────────────

fn bench_insert_with_capacity(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction/with_capacity");

    for &n in &[1_000, 10_000, 100_000, 1_000_000] {
        let keys = make_random_keys(n, 42);
        group.throughput(Throughput::Elements(n as u64));

        if n >= 1_000_000 { group.sample_size(10); }

        group.bench_with_input(BenchmarkId::new("ours", n), &keys, |b, keys| {
            b.iter(|| {
                let mut map = UnorderedFlatMap::with_capacity(n);
                for (i, &k) in keys.iter().enumerate() {
                    map.insert(k, i as u64);
                }
                black_box(map.len());
            });
        });

        group.bench_with_input(BenchmarkId::new("hashbrown", n), &keys, |b, keys| {
            b.iter(|| {
                let mut map = hashbrown::HashMap::with_capacity(n);
                for (i, &k) in keys.iter().enumerate() {
                    map.insert(k, i as u64);
                }
                black_box(map.len());
            });
        });
    }
    group.finish();
}

// ── Clone ───────────────────────────────────────────────────────────────────

fn bench_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction/clone");

    for &n in &[1_000, 100_000, 1_000_000] {
        let keys = make_random_keys(n, 42);

        if n >= 1_000_000 { group.sample_size(10); }

        let mut ours = UnorderedFlatMap::with_capacity(n);
        let mut hb = hashbrown::HashMap::with_capacity(n);
        for (i, &k) in keys.iter().enumerate() {
            ours.insert(k, i as u64);
            hb.insert(k, i as u64);
        }

        group.bench_with_input(BenchmarkId::new("ours", n), &(), |b, _| {
            b.iter(|| black_box(ours.clone()));
        });

        group.bench_with_input(BenchmarkId::new("hashbrown", n), &(), |b, _| {
            b.iter(|| black_box(hb.clone()));
        });
    }
    group.finish();
}

// ── FromIterator (collect) ──────────────────────────────────────────────────

fn bench_from_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction/from_iter");

    for &n in &[10_000, 100_000] {
        let pairs: Vec<(u64, u64)> = {
            let mut rng = Sfc64::new(42);
            (0..n).map(|i| (rng.next(), i as u64)).collect()
        };
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("ours", n), &pairs, |b, pairs| {
            b.iter(|| {
                let map: UnorderedFlatMap<u64, u64> = pairs.iter().copied().collect();
                black_box(map.len());
            });
        });

        group.bench_with_input(BenchmarkId::new("hashbrown", n), &pairs, |b, pairs| {
            b.iter(|| {
                let map: hashbrown::HashMap<u64, u64> = pairs.iter().copied().collect();
                black_box(map.len());
            });
        });
    }
    group.finish();
}

criterion_group!(
    construction,
    bench_grow_from_empty,
    bench_insert_with_capacity,
    bench_clone,
    bench_from_iter,
);
criterion_main!(construction);
