//! Core single-operation throughput benchmarks.
//!
//! Every benchmark uses pre-allocated, pre-warmed tables to measure pure
//! operation throughput without OS page fault or allocator overhead. Maps
//! are created once with `with_capacity()`, filled to warm all pages, then
//! each criterion iteration uses `clear()` + re-insert.
//!
//! Table sizes are chosen to produce specific load factors, not arbitrary
//! round numbers. Default load factor: 70% (representative mid-point).
//!
//! Compares against hashbrown only (the relevant competitor).

use criterion::{
    BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};

use unordered_flat_map::UnorderedFlatMap;
use unordered_flat_map::Splitsies;
use unordered_flat_map::InPlaceOverflow;

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

// Group size for capacity calculation. Uses the 15-slot design's geometry.
// Splitsies (16-slot) gets slightly lower actual load at the same entry count
// (~65.6% vs 70% for medium), but relative comparisons are valid since all
// implementations insert the same number of entries.
const GROUP_SIZE: usize = 15;

/// Compute (total_slots, max_load) for a given requested capacity.
/// Replicates the 15-slot map's internal group-count calculation.
fn table_geometry(capacity: usize) -> (usize, usize) {
    let min_slots = (capacity * 8 + 6) / 7;
    let min_groups = (min_slots + GROUP_SIZE - 1) / GROUP_SIZE;
    let mut num_groups = 1;
    while num_groups < min_groups {
        num_groups *= 2;
    }
    let total_slots = num_groups * GROUP_SIZE;
    let max_load = total_slots * 7 / 8;
    (total_slots, max_load)
}

/// Compute how many entries to insert for a target load factor percentage.
fn entries_for_load(capacity: usize, load_pct: usize) -> usize {
    let (total_slots, _) = table_geometry(capacity);
    total_slots * load_pct / 100
}

// ── Key generators ──────────────────────────────────────────────────────────

fn make_random_keys(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = Sfc64::new(seed);
    (0..n).map(|_| rng.next()).collect()
}

fn make_miss_keys(n: usize) -> Vec<u64> {
    // Different seed guarantees no overlap with insert keys (probabilistically)
    make_random_keys(n, 9999)
}

// ── Standard sizes ──────────────────────────────────────────────────────────

// Medium: 1024 groups, 15360 slots, max_load = 13440
const MEDIUM_CAPACITY: usize = 13_440;
// Large: 8192 groups, 122880 slots, max_load = 107520
const LARGE_CAPACITY: usize = 107_520;

const LOAD_PCT: usize = 70;

struct TestSize {
    name: &'static str,
    capacity: usize,
    num_entries: usize,
}

fn test_sizes() -> Vec<TestSize> {
    vec![
        TestSize {
            name: "medium",
            capacity: MEDIUM_CAPACITY,
            num_entries: entries_for_load(MEDIUM_CAPACITY, LOAD_PCT),
        },
        TestSize {
            name: "large",
            capacity: LARGE_CAPACITY,
            num_entries: entries_for_load(LARGE_CAPACITY, LOAD_PCT),
        },
    ]
}

// ── Throughput: Insert ──────────────────────────────────────────────────────

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/insert");

    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));

        // Ours: pre-warm, then clear + re-insert
        let mut ours = UnorderedFlatMap::with_capacity(sz.capacity);
        for (i, &k) in keys.iter().enumerate() { ours.insert(k, i as u64); }

        group.bench_with_input(
            BenchmarkId::new("UFM", sz.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    ours.clear();
                    for (i, &k) in keys.iter().enumerate() {
                        ours.insert(k, i as u64);
                    }
                    black_box(ours.len());
                });
            },
        );

        // split_overflow (16-slot groups)
        let mut split = Splitsies::with_capacity(sz.capacity);
        for (i, &k) in keys.iter().enumerate() { split.insert(k, i as u64); }

        group.bench_with_input(
            BenchmarkId::new("Splitsies", sz.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    split.clear();
                    for (i, &k) in keys.iter().enumerate() {
                        split.insert(k, i as u64);
                    }
                    black_box(split.len());
                });
            },
        );

        // in-place overflow (tombstone-based, no overflow bytes)
        let mut ipo = InPlaceOverflow::with_capacity(sz.capacity);
        for (i, &k) in keys.iter().enumerate() { ipo.insert(k, i as u64); }

        group.bench_with_input(
            BenchmarkId::new("IPO", sz.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    ipo.clear();
                    for (i, &k) in keys.iter().enumerate() {
                        ipo.insert(k, i as u64);
                    }
                    black_box(ipo.len());
                });
            },
        );

        // hashbrown
        let mut hb = hashbrown::HashMap::with_capacity(sz.capacity);
        for (i, &k) in keys.iter().enumerate() { hb.insert(k, i as u64); }

        group.bench_with_input(
            BenchmarkId::new("hashbrown", sz.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    hb.clear();
                    for (i, &k) in keys.iter().enumerate() {
                        hb.insert(k, i as u64);
                    }
                    black_box(hb.len());
                });
            },
        );
    }
    group.finish();
}

// ── Throughput: Insert with large values ─────────────────────────────────────

fn bench_insert_large_value(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/insert_val128");

    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        let val = [0u8; 128];
        group.throughput(Throughput::Elements(sz.num_entries as u64));

        let mut ours: UnorderedFlatMap<u64, [u8; 128]> =
            UnorderedFlatMap::with_capacity(sz.capacity);
        for &k in &keys { ours.insert(k, val); }

        group.bench_with_input(
            BenchmarkId::new("UFM", sz.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    ours.clear();
                    for &k in keys.iter() {
                        ours.insert(k, val);
                    }
                    black_box(ours.len());
                });
            },
        );

        // split_overflow (16-slot groups)
        let mut split: Splitsies<u64, [u8; 128]> =
            Splitsies::with_capacity(sz.capacity);
        for &k in &keys { split.insert(k, val); }

        group.bench_with_input(
            BenchmarkId::new("Splitsies", sz.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    split.clear();
                    for &k in keys.iter() {
                        split.insert(k, val);
                    }
                    black_box(split.len());
                });
            },
        );

        let mut hb: hashbrown::HashMap<u64, [u8; 128]> =
            hashbrown::HashMap::with_capacity(sz.capacity);
        for &k in &keys { hb.insert(k, val); }

        group.bench_with_input(
            BenchmarkId::new("hashbrown", sz.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    hb.clear();
                    for &k in keys.iter() {
                        hb.insert(k, val);
                    }
                    black_box(hb.len());
                });
            },
        );
    }
    group.finish();
}

// ── Throughput: Lookup Hit ───────────────────────────────────────────────────

fn bench_lookup_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/lookup_hit");

    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));

        let mut ours = UnorderedFlatMap::with_capacity(sz.capacity);
        let mut split = Splitsies::with_capacity(sz.capacity);
        let mut ipo = InPlaceOverflow::with_capacity(sz.capacity);
        let mut hb = hashbrown::HashMap::with_capacity(sz.capacity);
        for (i, &k) in keys.iter().enumerate() {
            ours.insert(k, i as u64);
            split.insert(k, i as u64);
            ipo.insert(k, i as u64);
            hb.insert(k, i as u64);
        }

        group.bench_with_input(
            BenchmarkId::new("UFM", sz.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &k in keys {
                        sum = sum.wrapping_add(*ours.get(&k).unwrap_or(&0));
                    }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Splitsies", sz.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &k in keys {
                        sum = sum.wrapping_add(*split.get(&k).unwrap_or(&0));
                    }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("IPO", sz.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &k in keys {
                        sum = sum.wrapping_add(*ipo.get(&k).unwrap_or(&0));
                    }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hashbrown", sz.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for &k in keys {
                        sum = sum.wrapping_add(*hb.get(&k).unwrap_or(&0));
                    }
                    black_box(sum);
                });
            },
        );
    }
    group.finish();
}

// ── Throughput: Lookup Miss ─────────────────────────────────────────────────

fn bench_lookup_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/lookup_miss");

    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        let miss_keys = make_miss_keys(sz.num_entries);
        group.throughput(Throughput::Elements(sz.num_entries as u64));

        let mut ours = UnorderedFlatMap::with_capacity(sz.capacity);
        let mut split = Splitsies::with_capacity(sz.capacity);
        let mut ipo = InPlaceOverflow::with_capacity(sz.capacity);
        let mut hb = hashbrown::HashMap::with_capacity(sz.capacity);
        for (i, &k) in keys.iter().enumerate() {
            ours.insert(k, i as u64);
            split.insert(k, i as u64);
            ipo.insert(k, i as u64);
            hb.insert(k, i as u64);
        }

        group.bench_with_input(
            BenchmarkId::new("UFM", sz.name),
            &miss_keys,
            |b, miss_keys| {
                b.iter(|| {
                    let mut count = 0u64;
                    for &k in miss_keys {
                        if ours.get(&k).is_some() { count += 1; }
                    }
                    black_box(count);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Splitsies", sz.name),
            &miss_keys,
            |b, miss_keys| {
                b.iter(|| {
                    let mut count = 0u64;
                    for &k in miss_keys {
                        if split.get(&k).is_some() { count += 1; }
                    }
                    black_box(count);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("IPO", sz.name),
            &miss_keys,
            |b, miss_keys| {
                b.iter(|| {
                    let mut count = 0u64;
                    for &k in miss_keys {
                        if ipo.get(&k).is_some() { count += 1; }
                    }
                    black_box(count);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hashbrown", sz.name),
            &miss_keys,
            |b, miss_keys| {
                b.iter(|| {
                    let mut count = 0u64;
                    for &k in miss_keys {
                        if hb.get(&k).is_some() { count += 1; }
                    }
                    black_box(count);
                });
            },
        );
    }
    group.finish();
}

// ── Throughput: Remove ──────────────────────────────────────────────────────

fn bench_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/remove");

    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));

        let mut ours = UnorderedFlatMap::with_capacity(sz.capacity);
        for (i, &k) in keys.iter().enumerate() { ours.insert(k, i as u64); }

        group.bench_with_input(
            BenchmarkId::new("UFM", sz.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    // Restore full table
                    ours.clear();
                    for (i, &k) in keys.iter().enumerate() { ours.insert(k, i as u64); }
                    // Measure removes
                    for &k in keys {
                        black_box(ours.remove(&k));
                    }
                });
            },
        );

        // split_overflow (16-slot groups)
        let mut split = Splitsies::with_capacity(sz.capacity);
        for (i, &k) in keys.iter().enumerate() { split.insert(k, i as u64); }

        group.bench_with_input(
            BenchmarkId::new("Splitsies", sz.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    split.clear();
                    for (i, &k) in keys.iter().enumerate() { split.insert(k, i as u64); }
                    for &k in keys {
                        black_box(split.remove(&k));
                    }
                });
            },
        );

        let mut hb = hashbrown::HashMap::with_capacity(sz.capacity);
        for (i, &k) in keys.iter().enumerate() { hb.insert(k, i as u64); }

        group.bench_with_input(
            BenchmarkId::new("hashbrown", sz.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    hb.clear();
                    for (i, &k) in keys.iter().enumerate() { hb.insert(k, i as u64); }
                    for &k in keys {
                        black_box(hb.remove(&k));
                    }
                });
            },
        );
    }
    group.finish();
}

// ── Throughput: Insert Existing (overwrite) ─────────────────────────────────

fn bench_insert_existing(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/insert_existing");

    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));

        let mut ours = UnorderedFlatMap::with_capacity(sz.capacity);
        let mut split = Splitsies::with_capacity(sz.capacity);
        let mut hb = hashbrown::HashMap::with_capacity(sz.capacity);
        for (i, &k) in keys.iter().enumerate() {
            ours.insert(k, i as u64);
            split.insert(k, i as u64);
            hb.insert(k, i as u64);
        }

        group.bench_with_input(
            BenchmarkId::new("UFM", sz.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    for (i, &k) in keys.iter().enumerate() {
                        ours.insert(k, i as u64 + 1);
                    }
                    black_box(ours.len());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Splitsies", sz.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    for (i, &k) in keys.iter().enumerate() {
                        split.insert(k, i as u64 + 1);
                    }
                    black_box(split.len());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hashbrown", sz.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    for (i, &k) in keys.iter().enumerate() {
                        hb.insert(k, i as u64 + 1);
                    }
                    black_box(hb.len());
                });
            },
        );
    }
    group.finish();
}

// ── Throughput: Iteration ───────────────────────────────────────────────────

fn bench_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/iteration");

    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));

        let mut ours = UnorderedFlatMap::with_capacity(sz.capacity);
        let mut split = Splitsies::with_capacity(sz.capacity);
        let mut ipo = InPlaceOverflow::with_capacity(sz.capacity);
        let mut hb = hashbrown::HashMap::with_capacity(sz.capacity);
        for (i, &k) in keys.iter().enumerate() {
            ours.insert(k, i as u64);
            split.insert(k, i as u64);
            ipo.insert(k, i as u64);
            hb.insert(k, i as u64);
        }

        group.bench_with_input(
            BenchmarkId::new("UFM", sz.name),
            &(),
            |b, _| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, &v) in ours.iter() {
                        sum = sum.wrapping_add(v);
                    }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Splitsies", sz.name),
            &(),
            |b, _| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, &v) in split.iter() {
                        sum = sum.wrapping_add(v);
                    }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("IPO", sz.name),
            &(),
            |b, _| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, &v) in ipo.iter() {
                        sum = sum.wrapping_add(v);
                    }
                    black_box(sum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hashbrown", sz.name),
            &(),
            |b, _| {
                b.iter(|| {
                    let mut sum = 0u64;
                    for (_, &v) in hb.iter() {
                        sum = sum.wrapping_add(v);
                    }
                    black_box(sum);
                });
            },
        );
    }
    group.finish();
}

// ── Throughput: Entry API ───────────────────────────────────────────────────

fn bench_entry(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/entry");
    let sz = &test_sizes()[0]; // medium only
    let keys = make_random_keys(sz.num_entries, 42);
    group.throughput(Throughput::Elements(sz.num_entries as u64));

    let mut ours = UnorderedFlatMap::with_capacity(sz.capacity);
    let mut split = Splitsies::with_capacity(sz.capacity);
    let mut hb = hashbrown::HashMap::with_capacity(sz.capacity);
    for &k in &keys {
        ours.insert(k, 0u64);
        split.insert(k, 0u64);
        hb.insert(k, 0u64);
    }

    group.bench_function("UFM", |b| {
        b.iter(|| {
            for &k in &keys {
                *ours.entry(k).or_insert(0) += 1;
            }
            black_box(ours.len());
        });
    });

    group.bench_function("Splitsies", |b| {
        b.iter(|| {
            for &k in &keys {
                *split.entry(k).or_insert(0) += 1;
            }
            black_box(split.len());
        });
    });

    group.bench_function("hashbrown", |b| {
        b.iter(|| {
            for &k in &keys {
                *hb.entry(k).or_insert(0) += 1;
            }
            black_box(hb.len());
        });
    });

    group.finish();
}

// ── Throughput: Size Scaling (find cache boundaries) ────────────────────────

fn bench_size_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput/size_scaling");

    // Test lookup hit at 70% load across sizes spanning L1→L2→L3→DRAM.
    // For u64/u64 (16B per bucket + metadata overhead):
    //   256 entries ≈ 4KB         (L1, 32-48KB typical)
    //   4K entries ≈ 64KB         (L1/L2 boundary)
    //   32K entries ≈ 512KB       (L2, 256KB-2MB typical)
    //   256K entries ≈ 4MB        (L2/L3 boundary)
    //   2M entries ≈ 32MB         (L3, may fit in large L3)
    //   20M entries ≈ 320MB       (guaranteed DRAM on any CPU)
    for &n in &[256, 4_000, 32_000, 256_000, 2_000_000, 20_000_000] {
        let capacity = n;
        let keys = make_random_keys(n, 42);
        group.throughput(Throughput::Elements(n as u64));

        if n >= 2_000_000 { group.sample_size(10); }

        let miss_keys = make_miss_keys(n);

        let mut ours = UnorderedFlatMap::with_capacity(capacity);
        let mut split = Splitsies::with_capacity(capacity);
        let mut ipo = InPlaceOverflow::with_capacity(capacity);
        let mut hb = hashbrown::HashMap::with_capacity(capacity);
        for (i, &k) in keys.iter().enumerate() {
            ours.insert(k, i as u64);
            split.insert(k, i as u64);
            ipo.insert(k, i as u64);
            hb.insert(k, i as u64);
        }

        // Lookup hit
        group.bench_with_input(BenchmarkId::new("UFM_hit", n), &keys, |b, keys| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in keys { sum = sum.wrapping_add(*ours.get(&k).unwrap_or(&0)); }
                black_box(sum);
            });
        });
        group.bench_with_input(BenchmarkId::new("Splitsies_hit", n), &keys, |b, keys| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in keys { sum = sum.wrapping_add(*split.get(&k).unwrap_or(&0)); }
                black_box(sum);
            });
        });
        group.bench_with_input(BenchmarkId::new("IPO_hit", n), &keys, |b, keys| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in keys { sum = sum.wrapping_add(*ipo.get(&k).unwrap_or(&0)); }
                black_box(sum);
            });
        });
        group.bench_with_input(BenchmarkId::new("hashbrown_hit", n), &keys, |b, keys| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in keys { sum = sum.wrapping_add(*hb.get(&k).unwrap_or(&0)); }
                black_box(sum);
            });
        });

        // Lookup miss
        group.bench_with_input(BenchmarkId::new("IPO_miss", n), &miss_keys, |b, mkeys| {
            b.iter(|| {
                let mut count = 0u64;
                for &k in mkeys { if ipo.get(&k).is_some() { count += 1; } }
                black_box(count);
            });
        });
        group.bench_with_input(BenchmarkId::new("hashbrown_miss", n), &miss_keys, |b, mkeys| {
            b.iter(|| {
                let mut count = 0u64;
                for &k in mkeys { if hb.get(&k).is_some() { count += 1; } }
                black_box(count);
            });
        });
    }
    group.finish();
}

criterion_group!(
    throughput,
    bench_insert,
    bench_insert_large_value,
    bench_lookup_hit,
    bench_lookup_miss,
    bench_remove,
    bench_insert_existing,
    bench_iteration,
    bench_size_scaling,
    bench_entry,
);
criterion_main!(throughput);
