//! Set benchmarks — mirrors the map throughput benchmarks for all Set types.

mod bench_helpers;

use bench_helpers::*;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, Throughput, black_box, criterion_group,
    criterion_main, measurement::WallTime};
use optimap::Set;

const N_MEDIUM: usize = 10_000;
const N_LARGE: usize = 100_000;

struct TestSize {
    name: &'static str,
    n: usize,
}

fn test_sizes() -> Vec<TestSize> {
    vec![
        TestSize { name: "10k", n: N_MEDIUM },
        TestSize { name: "100k", n: N_LARGE },
    ]
}

// ── Generic set benchmark helpers ──────────────────────────────────────────

fn bench_set_insert_for<S: Set<u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    label: &str,
    keys: &[u64],
) {
    let mut set = S::with_capacity(keys.len());
    for &k in keys {
        set.insert(k);
    }

    group.bench_with_input(BenchmarkId::new(name, label), keys, |b, keys| {
        b.iter(|| {
            set.clear();
            for &k in keys {
                set.insert(k);
            }
            black_box(set.len());
        });
    });
}

fn bench_set_contains_hit_for<S: Set<u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    label: &str,
    keys: &[u64],
) {
    let mut set = S::with_capacity(keys.len());
    for &k in keys {
        set.insert(k);
    }

    group.bench_with_input(BenchmarkId::new(name, label), keys, |b, keys| {
        b.iter(|| {
            let mut count = 0u64;
            for &k in keys {
                if set.contains(&k) {
                    count += 1;
                }
            }
            black_box(count);
        });
    });
}

fn bench_set_contains_miss_for<S: Set<u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    label: &str,
    keys: &[u64],
    miss_keys: &[u64],
) {
    let mut set = S::with_capacity(keys.len());
    for &k in keys {
        set.insert(k);
    }

    group.bench_with_input(BenchmarkId::new(name, label), miss_keys, |b, miss_keys| {
        b.iter(|| {
            let mut count = 0u64;
            for &k in miss_keys {
                if set.contains(&k) {
                    count += 1;
                }
            }
            black_box(count);
        });
    });
}

fn bench_set_remove_for<S: Set<u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    label: &str,
    keys: &[u64],
) {
    let mut set = S::with_capacity(keys.len());
    for &k in keys {
        set.insert(k);
    }

    group.bench_with_input(BenchmarkId::new(name, label), keys, |b, keys| {
        b.iter(|| {
            set.clear();
            for &k in keys {
                set.insert(k);
            }
            for &k in keys {
                black_box(set.remove(&k));
            }
        });
    });
}

fn bench_set_iter_for<S: Set<u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    label: &str,
    keys: &[u64],
) {
    let mut set = S::with_capacity(keys.len());
    for &k in keys {
        set.insert(k);
    }

    group.bench_with_input(BenchmarkId::new(name, label), &(), |b, _| {
        b.iter(|| {
            let mut sum = 0u64;
            for &v in set.iter() {
                sum = sum.wrapping_add(v);
            }
            black_box(sum);
        });
    });
}

fn bench_set_churn_for<S: Set<u64>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    label: &str,
    ops: u64,
    mask: u64,
) {
    group.bench_function(BenchmarkId::new(name, label), |b| {
        b.iter(|| {
            let mut set = S::new();
            let mut rng = Sfc64::new(42);
            let mut checksum = 0u64;
            for _ in 0..ops {
                let k = rng.next_u64() & mask;
                set.insert(k);
                let k2 = rng.next_u64() & mask;
                if set.remove(&k2) {
                    checksum += 1;
                }
            }
            black_box(checksum);
        });
    });
}

// ── Macro to run a bench helper for all set designs ────────────────────────

macro_rules! all_sets {
    ($helper:ident, $group:expr, $($args:expr),*) => {
        $helper::<optimap::UnorderedFlatSet<u64>>($group, "UFM", $($args),*);
        $helper::<optimap::SplitsiesSet<u64>>($group, "Splitsies", $($args),*);
        $helper::<optimap::IpoSet<u64>>($group, "IPO", $($args),*);
        $helper::<optimap::GapsSet<u64>>($group, "Gaps", $($args),*);
        $helper::<optimap::Ipo64Set<u64>>($group, "IPO64", $($args),*);
        $helper::<optimap::FlatBTreeSet<u64>>($group, "FlatBTree", $($args),*);
        $helper::<hashbrown::HashSet<u64>>($group, "hashbrown", $($args),*);
        $helper::<OptiSetBench<u64>>($group, "OptiSet", $($args),*);
    };
}

// ── Benchmark functions ────────────────────────────────────────────────────

fn bench_set_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("set/insert");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.n, 42);
        group.throughput(Throughput::Elements(sz.n as u64));
        all_sets!(bench_set_insert_for, &mut group, sz.name, &keys);
    }
    group.finish();
}

fn bench_set_contains_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("set/contains_hit");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.n, 42);
        group.throughput(Throughput::Elements(sz.n as u64));
        all_sets!(bench_set_contains_hit_for, &mut group, sz.name, &keys);
    }
    group.finish();
}

fn bench_set_contains_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("set/contains_miss");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.n, 42);
        let miss_keys = make_miss_keys(sz.n);
        group.throughput(Throughput::Elements(sz.n as u64));
        all_sets!(bench_set_contains_miss_for, &mut group, sz.name, &keys, &miss_keys);
    }
    group.finish();
}

fn bench_set_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("set/remove");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.n, 42);
        group.throughput(Throughput::Elements(sz.n as u64));
        all_sets!(bench_set_remove_for, &mut group, sz.name, &keys);
    }
    group.finish();
}

fn bench_set_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("set/iter");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.n, 42);
        group.throughput(Throughput::Elements(sz.n as u64));
        all_sets!(bench_set_iter_for, &mut group, sz.name, &keys);
    }
    group.finish();
}

fn bench_set_churn(c: &mut Criterion) {
    let mut group = c.benchmark_group("set/churn");
    all_sets!(bench_set_churn_for, &mut group, "16k_slots", 50_000, 0xFFFF);
    all_sets!(bench_set_churn_for, &mut group, "256k_slots", 50_000, 0x3_FFFF);
    group.finish();
}

criterion_group!(
    benches,
    bench_set_insert,
    bench_set_contains_hit,
    bench_set_contains_miss,
    bench_set_remove,
    bench_set_iter,
    bench_set_churn,
);
criterion_main!(benches);
