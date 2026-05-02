// Index loops here are clearer than .iter().enumerate().take(n).skip(prev_n).
#![allow(clippy::needless_range_loop)]

//! Sorted-map sweep — measures FlatBTree vs `std::BTreeMap` ns/op as N grows.
//!
//! Mirrors `sweep.rs`'s harness (log-spaced N points, median-of-trials,
//! CSV stdout) but for ordered maps. Cannot use the hash-map sweep harness
//! because the `Map` trait's `Q: Hash + Eq` bound would route FlatBTree's
//! `get`/`remove` through the O(n) leaf-walk fallback (see comment at
//! `flat_btree::map.rs::Map for FlatBTree::get`). This bench dispatches
//! through a private `BTreeBench` trait that calls the inherent
//! `Ord`-based methods, so both implementations get O(log n).
//!
//! Usage:
//!   cargo bench --bench sweep_btree                         # full run
//!   cargo bench --bench sweep_btree -- --op insert          # one operation
//!   cargo bench --bench sweep_btree -- --design FlatBTree   # one design
//!   cargo bench --bench sweep_btree -- --max-n 100000       # cap N range
//!   cargo bench --bench sweep_btree -- --trials 3           # fewer trials

mod bench_helpers;

use bench_helpers::Sfc64;
use optimap::FlatBTree;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

// ── Configuration ───────────────────────────────────────────────────────────

const DEFAULT_MAX_N: usize = 10_000_000;
const DEFAULT_TRIALS: usize = 5;

/// Target minimum wall time per measurement point. If a single pass is shorter
/// than this, we increase the ops count to compensate.
const MIN_MEASUREMENT_NS: u64 = 500_000; // 0.5ms

// ── BTreeBench trait ────────────────────────────────────────────────────────

/// Static-dispatch trait over ordered-map implementations. Implemented by
/// calling each map's *inherent* `Ord`-based methods (not the `Map<K, V>`
/// trait, whose `Hash + Eq` bound would force FlatBTree into an O(n) scan).
trait BTreeBench {
    fn new() -> Self;
    fn insert(&mut self, k: u64, v: u64) -> Option<u64>;
    fn get(&self, k: &u64) -> Option<&u64>;
    fn remove(&mut self, k: &u64) -> Option<u64>;
    fn iter(&self) -> impl Iterator<Item = (&u64, &u64)> + '_;
}

impl BTreeBench for FlatBTree<u64, u64> {
    fn new() -> Self {
        FlatBTree::new()
    }
    fn insert(&mut self, k: u64, v: u64) -> Option<u64> {
        FlatBTree::insert(self, k, v)
    }
    fn get(&self, k: &u64) -> Option<&u64> {
        FlatBTree::get(self, k)
    }
    fn remove(&mut self, k: &u64) -> Option<u64> {
        FlatBTree::remove(self, k)
    }
    fn iter(&self) -> impl Iterator<Item = (&u64, &u64)> + '_ {
        FlatBTree::iter(self)
    }
}

impl BTreeBench for BTreeMap<u64, u64> {
    fn new() -> Self {
        BTreeMap::new()
    }
    fn insert(&mut self, k: u64, v: u64) -> Option<u64> {
        BTreeMap::insert(self, k, v)
    }
    fn get(&self, k: &u64) -> Option<&u64> {
        BTreeMap::get(self, k)
    }
    fn remove(&mut self, k: &u64) -> Option<u64> {
        BTreeMap::remove(self, k)
    }
    fn iter(&self) -> impl Iterator<Item = (&u64, &u64)> + '_ {
        BTreeMap::iter(self)
    }
}

// ── CLI ─────────────────────────────────────────────────────────────────────

struct Config {
    max_n: usize,
    trials: usize,
    filter_op: Option<String>,
    filter_design: Option<String>,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config {
        max_n: DEFAULT_MAX_N,
        trials: DEFAULT_TRIALS,
        filter_op: None,
        filter_design: None,
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--op" => {
                i += 1;
                config.filter_op = Some(args[i].clone());
            }
            "--design" => {
                i += 1;
                config.filter_design = Some(args[i].clone());
            }
            "--max-n" => {
                i += 1;
                config.max_n = args[i].parse().expect("--max-n must be a number");
            }
            "--trials" => {
                i += 1;
                config.trials = args[i].parse().expect("--trials must be a number");
            }
            _ => {}
        }
        i += 1;
    }
    config
}

// ── N-point generation ──────────────────────────────────────────────────────

/// Logarithmic spacing (~3% growth per step, minimum step of 16). Same shape
/// as `sweep.rs` so the two CSVs are directly comparable.
fn sweep_points(max_n: usize) -> Vec<usize> {
    let mut points = Vec::new();
    let mut n = 100usize;
    while n <= max_n {
        points.push(n);
        let step = ((n as f64 * 0.03) as usize).max(16);
        n += step;
    }
    points
}

// ── Key generation ──────────────────────────────────────────────────────────

fn make_random_keys(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = Sfc64::new(seed);
    (0..n).map(|_| rng.next_u64()).collect()
}

fn make_miss_keys(n: usize) -> Vec<u64> {
    make_random_keys(n, 9999)
}

// ── Measurement helpers ─────────────────────────────────────────────────────

fn median(samples: &mut [Duration]) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

/// Compute how many times to repeat `ops` to reach `MIN_MEASUREMENT_NS`.
/// Calibrates with a single pilot run using the provided closure.
fn calibrate_repeats(ops: usize, mut f: impl FnMut(usize)) -> usize {
    let start = Instant::now();
    f(ops);
    let pilot_ns = start.elapsed().as_nanos() as u64;
    if pilot_ns >= MIN_MEASUREMENT_NS {
        return 1;
    }
    ((MIN_MEASUREMENT_NS / pilot_ns.max(1)) as usize).max(1)
}

// ── Sweep functions ─────────────────────────────────────────────────────────

/// Insert sweep: grow from empty, measure each batch (incremental, can't
/// repeat the same batch — instead run the full sweep `trials` times and
/// take the median per point).
fn sweep_insert<M: BTreeBench>(design: &str, points: &[usize], keys: &[u64], trials: usize) {
    let num_points = points.len();
    let mut all_ns: Vec<Vec<f64>> = vec![Vec::with_capacity(trials); num_points];

    for _trial in 0..trials {
        let mut map = M::new();
        let mut prev_n = 0;
        for (pi, &n) in points.iter().enumerate() {
            let batch = &keys[prev_n..n];
            let start = Instant::now();
            for (i, &k) in batch.iter().enumerate() {
                black_box(map.insert(k, (prev_n + i) as u64));
            }
            let elapsed = start.elapsed();
            let ns_per_op = elapsed.as_nanos() as f64 / batch.len() as f64;
            all_ns[pi].push(ns_per_op);
            prev_n = n;
        }
        drop(map);
    }

    for (pi, &n) in points.iter().enumerate() {
        all_ns[pi].sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = all_ns[pi][trials / 2];
        println!("insert,{design},{n},{med:.2}");
    }
}

/// Lookup hit: grow incrementally, measure lookups at each size with
/// calibrated op count and trials.
fn sweep_lookup_hit<M: BTreeBench>(design: &str, points: &[usize], keys: &[u64], trials: usize) {
    let mut map = M::new();
    let mut prev_n = 0;

    for &n in points {
        for i in prev_n..n {
            map.insert(keys[i], i as u64);
        }
        prev_n = n;

        let ops = n.min(50_000);
        let repeats = calibrate_repeats(ops, |count| {
            let mut sum = 0u64;
            for i in 0..count {
                sum = sum.wrapping_add(*black_box(map.get(&keys[i % n]).unwrap_or(&0)));
            }
            black_box(sum);
        });

        let total_ops = ops * repeats;
        let mut samples = Vec::with_capacity(trials);
        for _ in 0..trials {
            let start = Instant::now();
            let mut sum = 0u64;
            for i in 0..total_ops {
                sum = sum.wrapping_add(*black_box(map.get(&keys[i % n]).unwrap_or(&0)));
            }
            black_box(sum);
            samples.push(start.elapsed());
        }
        let med = median(&mut samples);
        let ns_per_op = med.as_nanos() as f64 / total_ops as f64;
        println!("lookup_hit,{design},{n},{ns_per_op:.2}");
    }
}

/// Lookup miss: grow incrementally, measure misses at each size.
fn sweep_lookup_miss<M: BTreeBench>(
    design: &str,
    points: &[usize],
    keys: &[u64],
    miss_keys: &[u64],
    trials: usize,
) {
    let mut map = M::new();
    let mut prev_n = 0;

    for &n in points {
        for i in prev_n..n {
            map.insert(keys[i], i as u64);
        }
        prev_n = n;

        let ops = 50_000.min(miss_keys.len());
        let repeats = calibrate_repeats(ops, |count| {
            let mut c = 0u64;
            for i in 0..count {
                if map.get(&miss_keys[i % miss_keys.len()]).is_some() {
                    c += 1;
                }
            }
            black_box(c);
        });

        let total_ops = ops * repeats;
        let mut samples = Vec::with_capacity(trials);
        for _ in 0..trials {
            let start = Instant::now();
            let mut count = 0u64;
            for i in 0..total_ops {
                if map.get(&miss_keys[i % miss_keys.len()]).is_some() {
                    count += 1;
                }
            }
            black_box(count);
            samples.push(start.elapsed());
        }
        let med = median(&mut samples);
        let ns_per_op = med.as_nanos() as f64 / total_ops as f64;
        println!("lookup_miss,{design},{n},{ns_per_op:.2}");
    }
}

/// Remove: build to size N, then remove a batch. Rebuilds per trial since
/// remove is destructive.
fn sweep_remove<M: BTreeBench>(design: &str, points: &[usize], keys: &[u64], trials: usize) {
    for &n in points {
        let ops = n.min(50_000);
        let mut samples = Vec::with_capacity(trials);

        for _ in 0..trials {
            let mut map = M::new();
            for i in 0..n {
                map.insert(keys[i], i as u64);
            }

            let start = Instant::now();
            for i in 0..ops {
                black_box(map.remove(&keys[i]));
            }
            samples.push(start.elapsed());
        }
        let med = median(&mut samples);
        let ns_per_op = med.as_nanos() as f64 / ops as f64;
        println!("remove,{design},{n},{ns_per_op:.2}");
    }
}

/// Iterate: grow incrementally, measure full scan at each size.
fn sweep_iterate<M: BTreeBench>(design: &str, points: &[usize], keys: &[u64], trials: usize) {
    let mut map = M::new();
    let mut prev_n = 0;

    for &n in points {
        for i in prev_n..n {
            map.insert(keys[i], i as u64);
        }
        prev_n = n;

        let repeats = calibrate_repeats(1, |_| {
            let mut sum = 0u64;
            for (_, &v) in map.iter() {
                sum = sum.wrapping_add(v);
            }
            black_box(sum);
        });

        let mut samples = Vec::with_capacity(trials);
        for _ in 0..trials {
            let start = Instant::now();
            for _ in 0..repeats {
                let mut sum = 0u64;
                for (_, &v) in map.iter() {
                    sum = sum.wrapping_add(v);
                }
                black_box(sum);
            }
            samples.push(start.elapsed());
        }
        let med = median(&mut samples);
        let total_elements = n * repeats;
        let ns_per_op = med.as_nanos() as f64 / total_elements as f64;
        println!("iterate,{design},{n},{ns_per_op:.2}");
    }
}

// ── Dispatch ────────────────────────────────────────────────────────────────

macro_rules! for_each_design {
    ($config:expr, $callback:ident $(, $arg:expr)*) => {
        for_each_design!(@run $config, $callback, FlatBTree<u64, u64>, "FlatBTree" $(, $arg)*);
        for_each_design!(@run $config, $callback, BTreeMap<u64, u64>, "BTreeMap" $(, $arg)*);
    };
    (@run $config:expr, $callback:ident, $ty:ty, $name:expr $(, $arg:expr)*) => {
        if $config.filter_design.as_ref().is_none_or(|f| f.eq_ignore_ascii_case($name)) {
            eprintln!("  {} ...", $name);
            $callback::<$ty>($name $(, $arg)*);
        }
    };
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let config = parse_args();
    let points = sweep_points(config.max_n);
    let keys = make_random_keys(config.max_n, 42);
    let miss_keys = make_miss_keys(100_000);

    eprintln!(
        "Sorted-map sweep: max_n={}, {} points, {} trials, {} designs",
        config.max_n,
        points.len(),
        config.trials,
        if config.filter_design.is_some() { "1" } else { "2" }
    );

    println!("operation,design,n,ns_per_op");

    macro_rules! run_op {
        ($op_name:expr, $sweep_fn:ident, $($extra:expr),*) => {
            if config.filter_op.as_ref().is_none_or(|f| f.eq_ignore_ascii_case($op_name)) {
                eprintln!("[{}]", $op_name);
                for_each_design!(config, $sweep_fn, &points, &keys $(, $extra)*, config.trials);
            }
        };
    }

    run_op!("insert", sweep_insert,);
    run_op!("lookup_hit", sweep_lookup_hit,);
    run_op!("lookup_miss", sweep_lookup_miss, &miss_keys);
    run_op!("remove", sweep_remove,);
    run_op!("iterate", sweep_iterate,);
}
