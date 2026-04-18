//! Sweep benchmark harness — measures operation throughput as a function of N.
//!
//! Produces CSV to stdout for plotting. Not criterion — raw `Instant` timing.
//!
//! Usage:
//!   cargo bench --bench sweep                           # full run
//!   cargo bench --bench sweep -- --op insert            # one operation
//!   cargo bench --bench sweep -- --design hashbrown     # one design
//!   cargo bench --bench sweep -- --max-n 100000         # cap N range

mod bench_helpers;

use bench_helpers::{OptiMapBench, Sfc64};
use optimap::{Gaps, IPO64, InPlaceOverflow, Map, Splitsies, UnorderedFlatMap};
use std::hint::black_box;
use std::time::Instant;

// ── Configuration ───────────────────────────────────────────────────────────

const DEFAULT_MAX_N: usize = 10_000_000;
const LOOKUP_OPS_CAP: usize = 100_000;

// ── CLI ─────────────────────────────────────────────────────────────────────

struct Config {
    max_n: usize,
    filter_op: Option<String>,
    filter_design: Option<String>,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config {
        max_n: DEFAULT_MAX_N,
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
            _ => {}
        }
        i += 1;
    }
    config
}

// ── N-point generation ──────────────────────────────────────────────────────

/// Logarithmic spacing (~15% growth per step, minimum step of 64).
/// ~200 points from 100 to 10M.
fn sweep_points(max_n: usize) -> Vec<usize> {
    let mut points = Vec::new();
    let mut n = 100usize;
    while n <= max_n {
        points.push(n);
        let step = ((n as f64 * 0.15) as usize).max(64);
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

// ── Sweep functions ─────────────────────────────────────────────────────────

/// Insert sweep: grow from empty, measure each batch.
/// Captures rehash spikes naturally.
fn sweep_insert<M: Map<u64, u64>>(design: &str, points: &[usize], keys: &[u64]) {
    // Warm-up: full insert cycle (discarded)
    {
        let mut m = M::new();
        for (i, &k) in keys.iter().enumerate() {
            m.insert(k, i as u64);
        }
        black_box(m.len());
    }

    let mut map = M::new();
    let mut prev_n = 0;
    for &n in points {
        let batch = &keys[prev_n..n];
        let start = Instant::now();
        for (i, &k) in batch.iter().enumerate() {
            black_box(map.insert(k, (prev_n + i) as u64));
        }
        let elapsed = start.elapsed();
        let ns_per_op = elapsed.as_nanos() as f64 / batch.len() as f64;
        println!("insert,{design},{n},{ns_per_op:.2}");
        prev_n = n;
    }
}

/// Lookup hit sweep: grow table incrementally, measure lookups at each size.
fn sweep_lookup_hit<M: Map<u64, u64>>(design: &str, points: &[usize], keys: &[u64]) {
    let mut map = M::new();
    let mut prev_n = 0;

    for &n in points {
        // Grow table to size n
        for i in prev_n..n {
            map.insert(keys[i], i as u64);
        }
        prev_n = n;

        let ops = n.min(LOOKUP_OPS_CAP);
        let start = Instant::now();
        let mut sum = 0u64;
        for i in 0..ops {
            sum = sum.wrapping_add(*black_box(map.get(&keys[i % n]).unwrap_or(&0)));
        }
        black_box(sum);
        let elapsed = start.elapsed();
        let ns_per_op = elapsed.as_nanos() as f64 / ops as f64;
        println!("lookup_hit,{design},{n},{ns_per_op:.2}");
    }
}

/// Lookup miss sweep: grow table incrementally, measure misses at each size.
fn sweep_lookup_miss<M: Map<u64, u64>>(
    design: &str,
    points: &[usize],
    keys: &[u64],
    miss_keys: &[u64],
) {
    let mut map = M::new();
    let mut prev_n = 0;

    for &n in points {
        for i in prev_n..n {
            map.insert(keys[i], i as u64);
        }
        prev_n = n;

        let ops = LOOKUP_OPS_CAP.min(miss_keys.len());
        let start = Instant::now();
        let mut count = 0u64;
        for i in 0..ops {
            if map.get(&miss_keys[i]).is_some() {
                count += 1;
            }
        }
        black_box(count);
        let elapsed = start.elapsed();
        let ns_per_op = elapsed.as_nanos() as f64 / ops as f64;
        println!("lookup_miss,{design},{n},{ns_per_op:.2}");
    }
}

/// Remove sweep: build table to size N, then remove a batch.
/// Rebuilds at each point since remove is destructive.
fn sweep_remove<M: Map<u64, u64>>(design: &str, points: &[usize], keys: &[u64]) {
    for &n in points {
        let mut map = M::new();
        for i in 0..n {
            map.insert(keys[i], i as u64);
        }

        let ops = n.min(LOOKUP_OPS_CAP);
        let start = Instant::now();
        for i in 0..ops {
            black_box(map.remove(&keys[i]));
        }
        let elapsed = start.elapsed();
        let ns_per_op = elapsed.as_nanos() as f64 / ops as f64;
        println!("remove,{design},{n},{ns_per_op:.2}");
    }
}

/// Iteration sweep: grow table incrementally, measure full scan at each size.
fn sweep_iterate<M: Map<u64, u64>>(design: &str, points: &[usize], keys: &[u64]) {
    let mut map = M::new();
    let mut prev_n = 0;

    for &n in points {
        for i in prev_n..n {
            map.insert(keys[i], i as u64);
        }
        prev_n = n;

        // Repeat small scans to get above timer noise
        let repeats = (100_000 / n).max(1);
        let start = Instant::now();
        for _ in 0..repeats {
            let mut sum = 0u64;
            for (_, &v) in map.iter() {
                sum = sum.wrapping_add(v);
            }
            black_box(sum);
        }
        let elapsed = start.elapsed();
        let total_elements = n * repeats;
        let ns_per_op = elapsed.as_nanos() as f64 / total_elements as f64;
        println!("iterate,{design},{n},{ns_per_op:.2}");
    }
}

// ── Dispatch ────────────────────────────────────────────────────────────────

macro_rules! for_each_design {
    ($config:expr, $callback:ident $(, $arg:expr)*) => {
        for_each_design!(@run $config, $callback, UnorderedFlatMap<u64,u64>, "UFM" $(, $arg)*);
        for_each_design!(@run $config, $callback, Gaps<u64,u64>, "Gaps" $(, $arg)*);
        for_each_design!(@run $config, $callback, Splitsies<u64,u64>, "Splitsies" $(, $arg)*);
        for_each_design!(@run $config, $callback, InPlaceOverflow<u64,u64>, "IPO" $(, $arg)*);
        for_each_design!(@run $config, $callback, IPO64<u64,u64>, "IPO64" $(, $arg)*);
        for_each_design!(@run $config, $callback, hashbrown::HashMap<u64,u64>, "hashbrown" $(, $arg)*);
        for_each_design!(@run $config, $callback, OptiMapBench<u64,u64>, "OptiMap" $(, $arg)*);
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
    let miss_keys = make_miss_keys(LOOKUP_OPS_CAP);

    eprintln!(
        "Sweep benchmark: max_n={}, {} points, {} designs",
        config.max_n,
        points.len(),
        if config.filter_design.is_some() {
            "1"
        } else {
            "7"
        }
    );

    println!("operation,design,n,ns_per_op");

    let ops: &[(&str, fn(&str, &[usize], &[u64]) -> bool)] = &[];
    let _ = ops; // suppress warning

    macro_rules! run_op {
        ($op_name:expr, $sweep_fn:ident, $($extra:expr),*) => {
            if config.filter_op.as_ref().is_none_or(|f| f.eq_ignore_ascii_case($op_name)) {
                eprintln!("[{}]", $op_name);
                for_each_design!(config, $sweep_fn, &points, &keys $(, $extra)*);
            }
        };
    }

    run_op!("insert", sweep_insert,);
    run_op!("lookup_hit", sweep_lookup_hit,);
    run_op!("lookup_miss", sweep_lookup_miss, &miss_keys);
    run_op!("remove", sweep_remove,);
    run_op!("iterate", sweep_iterate,);
}
