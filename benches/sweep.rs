//! Sweep benchmark harness — measures operation throughput as a function of N.
//!
//! Produces CSV to stdout for plotting. Not criterion — raw `Instant` timing
//! with multiple trials per measurement point (reports median).
//!
//! Usage:
//!   cargo bench --bench sweep                           # full run
//!   cargo bench --bench sweep -- --op insert            # one operation
//!   cargo bench --bench sweep -- --design hashbrown     # one design
//!   cargo bench --bench sweep -- --max-n 100000         # cap N range
//!   cargo bench --bench sweep -- --trials 3             # fewer trials (faster)

mod bench_helpers;

use bench_helpers::{OptiMapBench, Sfc64};
use optimap::matrix_types::*;
use optimap::{Gaps, IPO64, InPlaceOverflow, Map, Splitsies, UnorderedFlatMap};
use std::hint::black_box;
use std::time::{Duration, Instant};

// ── Configuration ───────────────────────────────────────────────────────────

const DEFAULT_MAX_N: usize = 10_000_000;
const DEFAULT_TRIALS: usize = 5;

/// Target minimum wall time per measurement point. If a single pass is shorter
/// than this, we increase the ops count to compensate.
const MIN_MEASUREMENT_NS: u64 = 500_000; // 0.5ms

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

/// Logarithmic spacing (~3% growth per step, minimum step of 16).
/// ~400 points from 100 to 10M — dense enough for smooth curves.
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

/// Return the median of a slice of Durations.
fn median(samples: &mut [Duration]) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

/// Compute how many times to repeat `ops` operations to reach MIN_MEASUREMENT_NS.
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

/// Insert sweep: grow from empty, measure each batch.
///
/// Insert is special: it's incremental and each batch changes the table state,
/// so we can't repeat the same batch. Instead we run the full sweep `trials`
/// times and take the median per point.
fn sweep_insert<M: Map<u64, u64>>(design: &str, points: &[usize], keys: &[u64], trials: usize) {
    // Collect all trials
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
        // Reset for next trial
        drop(map);
    }

    // Report median per point
    for (pi, &n) in points.iter().enumerate() {
        all_ns[pi].sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = all_ns[pi][trials / 2];
        println!("insert,{design},{n},{med:.2}");
    }
}

/// Lookup hit sweep: grow table incrementally, measure lookups at each size.
/// Multiple trials per point with calibrated op count.
fn sweep_lookup_hit<M: Map<u64, u64>>(
    design: &str,
    points: &[usize],
    keys: &[u64],
    trials: usize,
) {
    let mut map = M::new();
    let mut prev_n = 0;

    for &n in points {
        for i in prev_n..n {
            map.insert(keys[i], i as u64);
        }
        prev_n = n;

        // Calibrate: how many full passes over `ops` keys to fill MIN_MEASUREMENT_NS?
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

/// Lookup miss sweep: grow table incrementally, measure misses at each size.
fn sweep_lookup_miss<M: Map<u64, u64>>(
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

/// Remove sweep: build table to size N, then remove a batch.
/// Rebuilds per trial since remove is destructive.
fn sweep_remove<M: Map<u64, u64>>(
    design: &str,
    points: &[usize],
    keys: &[u64],
    trials: usize,
) {
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

/// Iteration sweep: grow table incrementally, measure full scan at each size.
fn sweep_iterate<M: Map<u64, u64>>(
    design: &str,
    points: &[usize],
    keys: &[u64],
    trials: usize,
) {
    let mut map = M::new();
    let mut prev_n = 0;

    for &n in points {
        for i in prev_n..n {
            map.insert(keys[i], i as u64);
        }
        prev_n = n;

        // Calibrate repeats so we get enough wall time
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
        // Original designs
        for_each_design!(@run $config, $callback, UnorderedFlatMap<u64,u64>, "UFM" $(, $arg)*);
        for_each_design!(@run $config, $callback, Gaps<u64,u64>, "Gaps" $(, $arg)*);
        for_each_design!(@run $config, $callback, Splitsies<u64,u64>, "Splitsies" $(, $arg)*);
        for_each_design!(@run $config, $callback, InPlaceOverflow<u64,u64>, "IPO" $(, $arg)*);
        for_each_design!(@run $config, $callback, IPO64<u64,u64>, "IPO64" $(, $arg)*);
        for_each_design!(@run $config, $callback, hashbrown::HashMap<u64,u64>, "hashbrown" $(, $arg)*);
        for_each_design!(@run $config, $callback, OptiMapBench<u64,u64>, "OptiMap" $(, $arg)*);
        // Matrix variants
        for_each_design!(@run $config, $callback, Hi8_8bitMap<u64,u64>, "Hi8_8bit" $(, $arg)*);
        for_each_design!(@run $config, $callback, Lo128_8bitMap<u64,u64>, "Lo128_8bit" $(, $arg)*);
        for_each_design!(@run $config, $callback, Lo8_1bitMap<u64,u64>, "Lo8_1bit" $(, $arg)*);
        for_each_design!(@run $config, $callback, Hi8_1bitMap<u64,u64>, "Hi8_1bit" $(, $arg)*);
        for_each_design!(@run $config, $callback, Lo128_1bitMap<u64,u64>, "Lo128_1bit" $(, $arg)*);
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
        "Sweep benchmark: max_n={}, {} points, {} trials, {} designs",
        config.max_n,
        points.len(),
        config.trials,
        if config.filter_design.is_some() {
            "1"
        } else {
            "12"
        }
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
