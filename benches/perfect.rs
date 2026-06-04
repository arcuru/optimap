//! Perfect-hash family benchmarks vs hashbrown and `Byte7_128_TombMap`.
//!
//! Measures:
//!
//! 1. **Construction cost** — building the map/set from N entries. The
//!    perfect-hash side pays an O(N · λ) displacement search up front;
//!    mutable hash maps pay incremental insert + resize.
//! 2. **Hit-lookup cost** — looking up keys that are in the set. The
//!    perfect side does one PHF computation + one indirect (+ one key
//!    compare for `PerfectMap` / `PerfectSet`); the mutable side does a
//!    probe loop.
//! 3. **Miss-lookup cost** — looking up keys NOT in the set. Skipped for
//!    `PerfectMapUnchecked`: its contract returns garbage on miss, so
//!    "miss" isn't a measurable distinct path.
//!
//! Variants:
//!
//! - `PerfectMap` — dense, stored keys, miss-safe
//! - `PerfectMapUnchecked` — no key compare, hit-only
//! - `PerfectSet` — exact membership
//! - `hashbrown::HashMap` — the reference incumbent
//! - `Byte7_128_TombMap` — `MapType::Tomb`, OptiMap's `Auto` default
//!
//! All u64 → u64 (or u64-only for `PerfectSet`), N ∈ {10K, 100K, 1M}.

mod bench_helpers;

use bench_helpers::{make_miss_keys, make_random_keys};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

use optimap::matrix_types::Byte7_128_TombMap;
use optimap::{PerfectMap, PerfectMapUnchecked, PerfectSet};

const SIZES: &[usize] = &[10_000, 100_000, 1_000_000];

fn sample_size_for(n: usize) -> usize {
    if n >= 1_000_000 {
        10
    } else if n >= 100_000 {
        20
    } else {
        30
    }
}

// ── Construction ──────────────────────────────────────────────────────────

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("perfect/construction");

    for &n in SIZES {
        let keys = make_random_keys(n, 42);
        let entries: Vec<(u64, u64)> = keys.iter().map(|&k| (k, k.wrapping_mul(31))).collect();

        group.throughput(Throughput::Elements(n as u64));
        group.sample_size(sample_size_for(n));

        group.bench_with_input(BenchmarkId::new("PerfectMap", n), &entries, |b, e| {
            b.iter(|| {
                let m = PerfectMap::<u64, u64>::from_iter_perfect(e.iter().copied()).unwrap();
                black_box(m);
            });
        });

        group.bench_with_input(BenchmarkId::new("PerfectMapUnchecked", n), &entries, |b, e| {
            b.iter(|| {
                let m =
                    PerfectMapUnchecked::<u64, u64>::from_iter_perfect(e.iter().copied()).unwrap();
                black_box(m);
            });
        });

        group.bench_with_input(BenchmarkId::new("PerfectSet", n), &keys, |b, k| {
            b.iter(|| {
                let s = PerfectSet::<u64>::from_iter_perfect(k.iter().copied()).unwrap();
                black_box(s);
            });
        });

        group.bench_with_input(BenchmarkId::new("hashbrown", n), &entries, |b, e| {
            b.iter(|| {
                let m: hashbrown::HashMap<u64, u64> = e.iter().copied().collect();
                black_box(m);
            });
        });

        group.bench_with_input(BenchmarkId::new("Byte7_128_Tomb", n), &entries, |b, e| {
            b.iter(|| {
                let mut m = Byte7_128_TombMap::<u64, u64>::with_capacity(e.len());
                for &(k, v) in e {
                    m.insert(k, v);
                }
                black_box(m);
            });
        });
    }

    group.finish();
}

// ── Hit lookup ────────────────────────────────────────────────────────────

fn bench_lookup_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("perfect/lookup_hit");

    for &n in SIZES {
        let keys = make_random_keys(n, 42);
        let entries: Vec<(u64, u64)> = keys.iter().map(|&k| (k, k.wrapping_mul(31))).collect();

        // Pre-build all containers once per N.
        let pm = PerfectMap::<u64, u64>::from_iter_perfect(entries.iter().copied()).unwrap();
        let pmu =
            PerfectMapUnchecked::<u64, u64>::from_iter_perfect(entries.iter().copied()).unwrap();
        let ps = PerfectSet::<u64>::from_iter_perfect(keys.iter().copied()).unwrap();
        let hb: hashbrown::HashMap<u64, u64> = entries.iter().copied().collect();
        let mut tomb = Byte7_128_TombMap::<u64, u64>::with_capacity(n);
        for &(k, v) in &entries {
            tomb.insert(k, v);
        }

        group.throughput(Throughput::Elements(n as u64));
        group.sample_size(sample_size_for(n));

        group.bench_with_input(BenchmarkId::new("PerfectMap", n), &keys, |b, ks| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in ks {
                    sum = sum.wrapping_add(*pm.get(&k).unwrap_or(&0));
                }
                black_box(sum);
            });
        });

        group.bench_with_input(BenchmarkId::new("PerfectMapUnchecked", n), &keys, |b, ks| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in ks {
                    sum = sum.wrapping_add(*pmu.get_unchecked(&k));
                }
                black_box(sum);
            });
        });

        group.bench_with_input(BenchmarkId::new("PerfectSet", n), &keys, |b, ks| {
            b.iter(|| {
                let mut count = 0u64;
                for &k in ks {
                    if ps.contains(&k) {
                        count += 1;
                    }
                }
                black_box(count);
            });
        });

        group.bench_with_input(BenchmarkId::new("hashbrown", n), &keys, |b, ks| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in ks {
                    sum = sum.wrapping_add(*hb.get(&k).unwrap_or(&0));
                }
                black_box(sum);
            });
        });

        group.bench_with_input(BenchmarkId::new("Byte7_128_Tomb", n), &keys, |b, ks| {
            b.iter(|| {
                let mut sum = 0u64;
                for &k in ks {
                    sum = sum.wrapping_add(*tomb.get(&k).unwrap_or(&0));
                }
                black_box(sum);
            });
        });
    }

    group.finish();
}

// ── Miss lookup ───────────────────────────────────────────────────────────

fn bench_lookup_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("perfect/lookup_miss");

    for &n in SIZES {
        let keys = make_random_keys(n, 42);
        let miss_keys = make_miss_keys(n);
        let entries: Vec<(u64, u64)> = keys.iter().map(|&k| (k, k.wrapping_mul(31))).collect();

        let pm = PerfectMap::<u64, u64>::from_iter_perfect(entries.iter().copied()).unwrap();
        let ps = PerfectSet::<u64>::from_iter_perfect(keys.iter().copied()).unwrap();
        let hb: hashbrown::HashMap<u64, u64> = entries.iter().copied().collect();
        let mut tomb = Byte7_128_TombMap::<u64, u64>::with_capacity(n);
        for &(k, v) in &entries {
            tomb.insert(k, v);
        }

        group.throughput(Throughput::Elements(n as u64));
        group.sample_size(sample_size_for(n));

        group.bench_with_input(BenchmarkId::new("PerfectMap", n), &miss_keys, |b, ks| {
            b.iter(|| {
                let mut count = 0u64;
                for &k in ks {
                    if pm.get(&k).is_none() {
                        count += 1;
                    }
                }
                black_box(count);
            });
        });

        group.bench_with_input(BenchmarkId::new("PerfectSet", n), &miss_keys, |b, ks| {
            b.iter(|| {
                let mut count = 0u64;
                for &k in ks {
                    if !ps.contains(&k) {
                        count += 1;
                    }
                }
                black_box(count);
            });
        });

        group.bench_with_input(BenchmarkId::new("hashbrown", n), &miss_keys, |b, ks| {
            b.iter(|| {
                let mut count = 0u64;
                for &k in ks {
                    if hb.get(&k).is_none() {
                        count += 1;
                    }
                }
                black_box(count);
            });
        });

        group.bench_with_input(BenchmarkId::new("Byte7_128_Tomb", n), &miss_keys, |b, ks| {
            b.iter(|| {
                let mut count = 0u64;
                for &k in ks {
                    if tomb.get(&k).is_none() {
                        count += 1;
                    }
                }
                black_box(count);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_construction, bench_lookup_hit, bench_lookup_miss);
criterion_main!(benches);
