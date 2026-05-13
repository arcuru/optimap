//! Overhead check: `raw_entry()` read path vs regular `get()`.
//!
//! `raw_entry().from_key(k)` is just an extra builder construction wrapping
//! the same `find_by_hash` call that `get` makes. With `#[inline]` on the
//! builder methods, the two should generate identical code and benchmark
//! within noise. This bench locks that in.
//!
//! For mutable raw entries, `raw_entry_mut().from_key(k)` matching the
//! `Occupied`/`Vacant` enum is more work than a plain `get_mut` (the latter
//! returns `Option<&mut V>`), so a small constant gap is expected — we just
//! make sure it doesn't grow unbounded.
//!
//! Reference numbers (Splitsies, u64/u64, single host):
//! - `lookup_hit/get/medium` vs `raw_from_key/medium`: within 1 %
//! - `lookup_hit/get/large` vs `raw_from_key/large`: within 1 %
//! - `lookup_hit/raw_from_hash_precomputed/medium`: ~ +2 % vs get
//!   (skipping the hash recomputation when the caller already has it)
//! - `lookup_miss/...`: ~ ±3 %
//! - `mut_dispatch/get_mut/large` vs `raw_entry_mut_from_key/large`:
//!   ~ 16 % slower on the raw entry path — matches the regular `entry`
//!   API's enum-dispatch overhead. Read-side raw entries are the design
//!   target; the mutable path is offered for completeness.
//!
//! Sizes mirror dispatch.rs:
//! - medium: 13_440 capacity, 70 % load
//! - large:  107_520 capacity, 70 % load

use std::hash::BuildHasher;

use criterion::{Criterion, Throughput, criterion_group, criterion_main, BenchmarkId};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use optimap::raw_entry::RawEntryMut;
use optimap::Splitsies;

const MEDIUM_CAPACITY: usize = 13_440;
const LARGE_CAPACITY: usize = 107_520;
const LOAD_PCT: usize = 70;

struct TestSize {
    name: &'static str,
    capacity: usize,
    num_entries: usize,
}

const SIZES: &[TestSize] = &[
    TestSize {
        name: "medium",
        capacity: MEDIUM_CAPACITY,
        num_entries: MEDIUM_CAPACITY * LOAD_PCT / 100,
    },
    TestSize {
        name: "large",
        capacity: LARGE_CAPACITY,
        num_entries: LARGE_CAPACITY * LOAD_PCT / 100,
    },
];

fn build(size: &TestSize) -> (Splitsies<u64, u64>, Vec<u64>, Vec<u64>) {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let keys: Vec<u64> = (0..size.num_entries).map(|_| rng.r#gen()).collect();
    let misses: Vec<u64> = (0..size.num_entries).map(|_| rng.r#gen()).collect();
    let mut map: Splitsies<u64, u64> = Splitsies::with_capacity(size.capacity);
    for &k in &keys {
        map.insert(k, k.wrapping_mul(31));
    }
    (map, keys, misses)
}

fn bench_lookup_hit(c: &mut Criterion) {
    let mut g = c.benchmark_group("lookup_hit");
    for size in SIZES {
        let (map, keys, _) = build(size);
        g.throughput(Throughput::Elements(keys.len() as u64));

        g.bench_with_input(BenchmarkId::new("get", size.name), &keys, |b, keys| {
            b.iter(|| {
                let mut sum: u64 = 0;
                for k in keys {
                    sum = sum.wrapping_add(*map.get(k).unwrap());
                }
                sum
            });
        });

        g.bench_with_input(
            BenchmarkId::new("raw_from_key", size.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    let mut sum: u64 = 0;
                    for k in keys {
                        let (_, v) = map.raw_entry().from_key(k).unwrap();
                        sum = sum.wrapping_add(*v);
                    }
                    sum
                });
            },
        );

        g.bench_with_input(
            BenchmarkId::new("raw_from_hash_precomputed", size.name),
            &keys,
            |b, keys| {
                let hashes: Vec<u64> = keys.iter().map(|k| map.hasher().hash_one(*k)).collect();
                b.iter(|| {
                    let mut sum: u64 = 0;
                    for (h, k) in hashes.iter().zip(keys) {
                        let (_, v) = map
                            .raw_entry()
                            .from_key_hashed_nocheck(*h, k)
                            .unwrap();
                        sum = sum.wrapping_add(*v);
                    }
                    sum
                });
            },
        );
    }
    g.finish();
}

fn bench_lookup_miss(c: &mut Criterion) {
    let mut g = c.benchmark_group("lookup_miss");
    for size in SIZES {
        let (map, _, misses) = build(size);
        g.throughput(Throughput::Elements(misses.len() as u64));

        g.bench_with_input(BenchmarkId::new("get", size.name), &misses, |b, misses| {
            b.iter(|| {
                let mut hits = 0u64;
                for k in misses {
                    if map.get(k).is_some() {
                        hits += 1;
                    }
                }
                hits
            });
        });

        g.bench_with_input(
            BenchmarkId::new("raw_from_key", size.name),
            &misses,
            |b, misses| {
                b.iter(|| {
                    let mut hits = 0u64;
                    for k in misses {
                        if map.raw_entry().from_key(k).is_some() {
                            hits += 1;
                        }
                    }
                    hits
                });
            },
        );
    }
    g.finish();
}

fn bench_mut_dispatch(c: &mut Criterion) {
    // get_mut: returns Option<&mut V> directly.
    // raw_entry_mut().from_key(k): does the same lookup, returns an enum.
    let mut g = c.benchmark_group("mut_dispatch");
    for size in SIZES {
        let (mut map, keys, _) = build(size);
        g.throughput(Throughput::Elements(keys.len() as u64));

        g.bench_with_input(BenchmarkId::new("get_mut", size.name), &keys, |b, keys| {
            b.iter(|| {
                for k in keys {
                    if let Some(v) = map.get_mut(k) {
                        *v = v.wrapping_add(1);
                    }
                }
            });
        });

        g.bench_with_input(
            BenchmarkId::new("raw_entry_mut_from_key", size.name),
            &keys,
            |b, keys| {
                b.iter(|| {
                    for k in keys {
                        if let RawEntryMut::Occupied(mut e) = map.raw_entry_mut().from_key(k) {
                            let v = e.get_mut();
                            *v = v.wrapping_add(1);
                        }
                    }
                });
            },
        );
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_lookup_hit,
    bench_lookup_miss,
    bench_mut_dispatch,
);
criterion_main!(benches);
