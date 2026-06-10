// Index loops here are clearer than .iter().enumerate().take(n).skip(prev_n).
#![allow(clippy::needless_range_loop)]

//! Sweep benchmark harness — measures operation throughput as a function of N.
//!
//! Produces CSV to stdout for plotting. Not criterion — raw `Instant` timing
//! with multiple trials per measurement point (reports median).
//!
//! Usage:
//!   cargo bench --bench sweep                            # curated set (default), 10M N
//!   cargo bench --bench sweep -- --op insert             # one operation
//!   cargo bench --bench sweep -- --design hashbrown      # one design (overrides set)
//!   cargo bench --bench sweep -- --design-set tomb       # one cluster
//!   cargo bench --bench sweep -- --design-set all        # everything (~70 designs)
//!   cargo bench --bench sweep -- --width 32              # restrict to 32-slot SIMD
//!   cargo bench --bench sweep -- --max-n 100000          # cap N range
//!   cargo bench --bench sweep -- --trials 3              # fewer trials (faster)
//!
//! Design sets (cluster-based):
//!   curated  — one representative per perf cluster (default; 12 designs)
//!   tomb     — tombstone family (IPO, IPO64, *_Tomb*)
//!   overflow — overflow-bit family (UFM, Splitsies, Gaps, matrix/embedded/AND variants)
//!   soa      — Structure-of-Arrays family
//!   sorted   — tree designs (FlatBTree, std::BTreeMap)
//!   baseline — external comparators (hashbrown)
//!   opti     — OptiMap (Auto policy)
//!   all      — every design in the harness
//!
//! Widths (sub-filter): 16, 32, 64, or all (default). Width filter is applied
//! AFTER set selection — `--design-set overflow --width 32` gives overflow-bit
//! variants at 32-slot only.

mod bench_helpers;

use bench_helpers::{OptiMapAuto, Sfc64};
use optimap::matrix_types::*;
use optimap::{FlatBTree, Gaps, IPO64, InPlaceOverflow, Map, Splitsies, UnorderedFlatMap};
use std::hint::black_box;
use std::time::{Duration, Instant};

// ── Configuration ───────────────────────────────────────────────────────────

const DEFAULT_MAX_N: usize = 10_000_000;
const DEFAULT_TRIALS: usize = 5;

/// Target minimum wall time per measurement point. If a single pass is shorter
/// than this, we increase the ops count to compensate.
const MIN_MEASUREMENT_NS: u64 = 500_000; // 0.5ms

// ── CLI ─────────────────────────────────────────────────────────────────────

/// Perf-cluster grouping for the design list. Each design is tagged with
/// exactly one cluster; `--design-set <cluster>` selects all designs in it.
/// `Curated` is the default and picks one representative from every other
/// cluster (plus the 5 headline designs and the OptiMap wrapper).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cluster {
    /// Tombstone family — fast hit + remove cache-resident; lookup_miss
    /// cliff at DRAM (IPO, IPO64, *_Tomb*).
    Tomb,
    /// Overflow-bit family — O(1) miss termination; tombstone-free; flat
    /// at DRAM scale (UFM, Splitsies, Gaps, matrix/embedded/AND variants).
    Overflow,
    /// Structure-of-Arrays — fast iterate + miss; slower insert.
    Soa,
    /// Tree designs — FlatBTree, std::BTreeMap. O(log n) shape; sorted ops.
    Sorted,
    /// External baselines (hashbrown, std::HashMap).
    Baseline,
    /// OptiMap with default Auto policy (the wrapper we're measuring).
    Opti,
}

/// SIMD group width (in slots). Sub-filter applied after cluster selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Width {
    W16,
    W32,
    W64,
    /// Not applicable (trees, baselines, OptiMap-Auto since it varies).
    NA,
}

/// Which cluster preset to run. Resolved into a predicate over (cluster, width).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DesignSet {
    /// One representative per cluster + headline 5 + OptiMap. Default.
    Curated,
    Cluster(Cluster),
    All,
}

impl DesignSet {
    fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "curated" => Ok(DesignSet::Curated),
            "all" => Ok(DesignSet::All),
            "tomb" | "tombstone" => Ok(DesignSet::Cluster(Cluster::Tomb)),
            "overflow" | "overflow-bit" => Ok(DesignSet::Cluster(Cluster::Overflow)),
            "soa" => Ok(DesignSet::Cluster(Cluster::Soa)),
            "sorted" | "tree" => Ok(DesignSet::Cluster(Cluster::Sorted)),
            "baseline" => Ok(DesignSet::Cluster(Cluster::Baseline)),
            "opti" | "optimap" => Ok(DesignSet::Cluster(Cluster::Opti)),
            _ => Err(format!(
                "unknown --design-set '{s}' (valid: curated, all, tomb, overflow, soa, sorted, baseline, opti)"
            )),
        }
    }
}

/// Width sub-filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WidthFilter {
    Only(Width),
    All,
}

impl WidthFilter {
    fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "all" => Ok(WidthFilter::All),
            "16" => Ok(WidthFilter::Only(Width::W16)),
            "32" => Ok(WidthFilter::Only(Width::W32)),
            "64" => Ok(WidthFilter::Only(Width::W64)),
            _ => Err(format!("--width must be 16, 32, 64, or all (got '{s}')")),
        }
    }
}

struct Config {
    max_n: usize,
    trials: usize,
    filter_op: Option<String>,
    filter_design: Option<String>,
    design_set: DesignSet,
    width: WidthFilter,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut config = Config {
        max_n: DEFAULT_MAX_N,
        trials: DEFAULT_TRIALS,
        filter_op: None,
        filter_design: None,
        design_set: DesignSet::Curated,
        width: WidthFilter::All,
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
            "--design-set" => {
                i += 1;
                config.design_set = DesignSet::parse(&args[i]).unwrap_or_else(|e| panic!("{e}"));
            }
            "--width" => {
                i += 1;
                config.width = WidthFilter::parse(&args[i]).unwrap_or_else(|e| panic!("{e}"));
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

/// Membership of `(name, cluster, width)` in the active set.
///
/// Resolution order: an explicit `--design <name>` wins over everything else
/// (single-design filter, case-insensitive). Otherwise the `--design-set`
/// preset picks the cluster pool, `--width` narrows it, and `Curated`
/// hand-picks the representatives by name.
fn design_active(name: &str, cluster: Cluster, width: Width, config: &Config) -> bool {
    if let Some(filter) = &config.filter_design {
        return filter.eq_ignore_ascii_case(name);
    }
    let in_width = match config.width {
        WidthFilter::All => true,
        // Trees, baselines, and OptiMap aren't width-tagged; let them through
        // any width filter so a `--width 32` run still has reference lines.
        WidthFilter::Only(_) if width == Width::NA => true,
        WidthFilter::Only(w) => width == w,
    };
    if !in_width {
        return false;
    }
    match config.design_set {
        DesignSet::All => true,
        DesignSet::Cluster(c) => cluster == c,
        DesignSet::Curated => CURATED_NAMES.contains(&name),
    }
}

/// Curated default: cluster representatives, named for the "settled-on best"
/// of each family. Display labels are decoupled from the underlying Rust type
/// names (UFM, IPO, etc. are still the brand types in `src/`).
///
/// The Tomb cluster gets **two** representatives because the top two designs
/// across the project (`Byte7_128_Tomb` and `IPO`) differ only in tag strategy
/// — 128 vs 254 distinct tag values — and both routinely top the benchmarks
/// at cache-resident sizes. Tomb (128) is hashbrown-tag-equivalent; TombWide
/// (254) has one more bit of tag entropy. Other clusters use a single rep.
const CURATED_NAMES: &[&str] = &[
    // Tomb cluster (16-slot tombstone, 4 reps)
    "Tomb",     // was Byte7_128_Tomb — 128-value tag (hashbrown-equivalent)
    "TombWide", // was IPO           — 254-value tag (wider, more entropy)
    "Tomb64",   // was IPO64         — 64-slot AVX-512 tomb
    "TombSoa",  // was SoaIpo        — SoA flavor of Tomb
    // OvInline family (15-slot embedded overflow byte)
    "OvInline",     // was UFM           — canonical embedded-overflow
    "OvInlineGaps", // was Gaps          — UFM variant w/ power-of-2 stride
    "OvInline32",   // was Ufm32         — 32-slot AVX2 variant
    // OvSplit (16-slot separate-overflow rep)
    "OvSplit", // was Splitsies     — separate-overflow channel array
    // Sorted cluster
    "FlatBTree",     // arena-based B+ tree
    "std::BTreeMap", // std baseline
    // External baseline + wrapper
    "hashbrown",
    "OptiMap", // Auto policy
];

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
fn sweep_lookup_hit<M: Map<u64, u64>>(design: &str, points: &[usize], keys: &[u64], trials: usize) {
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
fn sweep_remove<M: Map<u64, u64>>(design: &str, points: &[usize], keys: &[u64], trials: usize) {
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
fn sweep_iterate<M: Map<u64, u64>>(design: &str, points: &[usize], keys: &[u64], trials: usize) {
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
        // ── Headline brands (renamed to cluster-representative labels) ──────
        // Source-level type names (UnorderedFlatMap, InPlaceOverflow, etc.)
        // are unchanged; only the display labels here use the cluster names.
        for_each_design!(@run $config, $callback, UnorderedFlatMap<u64,u64>, "OvInline", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Gaps<u64,u64>, "OvInlineGaps", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Splitsies<u64,u64>, "OvSplit", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, InPlaceOverflow<u64,u64>, "TombWide", Cluster::Tomb, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, IPO64<u64,u64>, "Tomb64", Cluster::Tomb, Width::W64 $(, $arg)*);
        // ── External baselines ──────────────────────────────────────────────
        for_each_design!(@run $config, $callback, hashbrown::HashMap<u64,u64>, "hashbrown", Cluster::Baseline, Width::NA $(, $arg)*);
        // ── OptiMap wrapper (default Auto policy) ───────────────────────────
        for_each_design!(@run $config, $callback, OptiMapAuto<u64,u64>, "OptiMap", Cluster::Opti, Width::NA $(, $arg)*);
        // ── Tree designs (sorted cluster) ───────────────────────────────────
        for_each_design!(@run $config, $callback, FlatBTree<u64,u64>, "FlatBTree", Cluster::Sorted, Width::NA $(, $arg)*);
        for_each_design!(@run $config, $callback, std::collections::BTreeMap<u64,u64>, "std::BTreeMap", Cluster::Sorted, Width::NA $(, $arg)*);
        // ── 16-slot matrix variants (overflow-bit, separate) ────────────────
        for_each_design!(@run $config, $callback, Byte1_8bitMap<u64,u64>, "Byte1_8bit", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte0_128_8bitMap<u64,u64>, "Byte0_128_8bit", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte0_1bitMap<u64,u64>, "Byte0_1bit", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte0_128_1bitMap<u64,u64>, "Byte0_128_1bit", Cluster::Overflow, Width::W16 $(, $arg)*);
        // ── 16-slot AND-indexed (overflow) ──────────────────────────────────
        for_each_design!(@run $config, $callback, Byte7_128_1bitAndMap<u64,u64>, "Byte7_128_1bitAnd", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_128_8bitAndMap<u64,u64>, "Byte7_128_8bitAnd", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_255_1bitAndMap<u64,u64>, "Byte7_255_1bitAnd", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_255_8bitAndMap<u64,u64>, "Byte7_255_8bitAnd", Cluster::Overflow, Width::W16 $(, $arg)*);
        // ── 16-slot Pure-Rust tag variants (overflow) ───────────────────────
        for_each_design!(@run $config, $callback, Byte7_255Pure_1bitAndMap<u64,u64>, "Byte7_255Pure_1bitAnd", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_255Pure_8bitAndMap<u64,u64>, "Byte7_255Pure_8bitAnd", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte0_255PureLayoutMap<u64,u64>, "Byte0_255Pure", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte1_255PureLayoutMap<u64,u64>, "Byte1_255Pure", Cluster::Overflow, Width::W16 $(, $arg)*);
        // ── 16-slot embedded-overflow (overflow) ────────────────────────────
        for_each_design!(@run $config, $callback, Byte1_EmbMap<u64,u64>, "Byte1_Emb", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte1_EmbP2Map<u64,u64>, "Byte1_EmbP2", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte0_128_EmbMap<u64,u64>, "Byte0_128_Emb", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte0_128_EmbP2Map<u64,u64>, "Byte0_128_EmbP2", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_128Ch_EmbAndMap<u64,u64>, "Byte7_128Ch_EmbAnd", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_128Ch_EmbP2AndMap<u64,u64>, "Byte7_128Ch_EmbP2And", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_255Ch_EmbAndMap<u64,u64>, "Byte7_255Ch_EmbAnd", Cluster::Overflow, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_255Ch_EmbP2AndMap<u64,u64>, "Byte7_255Ch_EmbP2And", Cluster::Overflow, Width::W16 $(, $arg)*);
        // ── 32-slot separate-overflow (AVX2) ────────────────────────────────
        for_each_design!(@run $config, $callback, Splitsies32Map<u64,u64>, "Splitsies32", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Splitsies32_1bitMap<u64,u64>, "Splitsies32_1bit", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte1_8bit32Map<u64,u64>, "Byte1_8bit32", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte0_128_1bit32Map<u64,u64>, "Byte0_128_1bit32", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte0_128_8bit32Map<u64,u64>, "Byte0_128_8bit32", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_128_1bitAnd32Map<u64,u64>, "Byte7_128_1bitAnd32", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_255_1bitAnd32Map<u64,u64>, "Byte7_255_1bitAnd32", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_128_8bitAnd32Map<u64,u64>, "Byte7_128_8bitAnd32", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_255_8bitAnd32Map<u64,u64>, "Byte7_255_8bitAnd32", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_255Pure_1bitAnd32Map<u64,u64>, "Byte7_255Pure_1bitAnd32", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_255Pure_8bitAnd32Map<u64,u64>, "Byte7_255Pure_8bitAnd32", Cluster::Overflow, Width::W32 $(, $arg)*);
        // ── 32-slot embedded-overflow ───────────────────────────────────────
        for_each_design!(@run $config, $callback, Ufm32Map<u64,u64>, "OvInline32", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Gaps32Map<u64,u64>, "Gaps32", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte1_Emb32Map<u64,u64>, "Byte1_Emb32", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte1_EmbP232Map<u64,u64>, "Byte1_EmbP232", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte0_128_Emb32Map<u64,u64>, "Byte0_128_Emb32", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte0_128_EmbP232Map<u64,u64>, "Byte0_128_EmbP232", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_128Ch_EmbAnd32Map<u64,u64>, "Byte7_128Ch_EmbAnd32", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_128Ch_EmbP2And32Map<u64,u64>, "Byte7_128Ch_EmbP2And32", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_255Ch_EmbAnd32Map<u64,u64>, "Byte7_255Ch_EmbAnd32", Cluster::Overflow, Width::W32 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_255Ch_EmbP2And32Map<u64,u64>, "Byte7_255Ch_EmbP2And32", Cluster::Overflow, Width::W32 $(, $arg)*);
        // ── 64-slot separate-overflow (AVX-512) ─────────────────────────────
        for_each_design!(@run $config, $callback, Splitsies64Map<u64,u64>, "Splitsies64", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Splitsies64_1bitMap<u64,u64>, "Splitsies64_1bit", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte1_8bit64Map<u64,u64>, "Byte1_8bit64", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte0_128_1bit64Map<u64,u64>, "Byte0_128_1bit64", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte0_128_8bit64Map<u64,u64>, "Byte0_128_8bit64", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_128_1bitAnd64Map<u64,u64>, "Byte7_128_1bitAnd64", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_255_1bitAnd64Map<u64,u64>, "Byte7_255_1bitAnd64", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_128_8bitAnd64Map<u64,u64>, "Byte7_128_8bitAnd64", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_255_8bitAnd64Map<u64,u64>, "Byte7_255_8bitAnd64", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_255Pure_1bitAnd64Map<u64,u64>, "Byte7_255Pure_1bitAnd64", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_255Pure_8bitAnd64Map<u64,u64>, "Byte7_255Pure_8bitAnd64", Cluster::Overflow, Width::W64 $(, $arg)*);
        // ── 64-slot embedded-overflow ───────────────────────────────────────
        for_each_design!(@run $config, $callback, Ufm64Map<u64,u64>, "Ufm64", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Gaps64Map<u64,u64>, "Gaps64", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte1_Emb64Map<u64,u64>, "Byte1_Emb64", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte1_EmbP264Map<u64,u64>, "Byte1_EmbP264", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte0_128_Emb64Map<u64,u64>, "Byte0_128_Emb64", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte0_128_EmbP264Map<u64,u64>, "Byte0_128_EmbP264", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_128Ch_EmbAnd64Map<u64,u64>, "Byte7_128Ch_EmbAnd64", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_128Ch_EmbP2And64Map<u64,u64>, "Byte7_128Ch_EmbP2And64", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_255Ch_EmbAnd64Map<u64,u64>, "Byte7_255Ch_EmbAnd64", Cluster::Overflow, Width::W64 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_255Ch_EmbP2And64Map<u64,u64>, "Byte7_255Ch_EmbP2And64", Cluster::Overflow, Width::W64 $(, $arg)*);
        // ── Tombstone variants ──────────────────────────────────────────────
        for_each_design!(@run $config, $callback, Byte7_128_TombMap<u64,u64>, "Tomb", Cluster::Tomb, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, Byte7_254_Tomb64Map<u64,u64>, "Byte7_254_Tomb64", Cluster::Tomb, Width::W64 $(, $arg)*);
        // ── SoA variants ────────────────────────────────────────────────────
        for_each_design!(@run $config, $callback, optimap::SoaMap<u64,u64>, "SoaMap", Cluster::Soa, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, optimap::soa::SoaByte0_128<u64,u64>, "SoaByte0_128", Cluster::Soa, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, optimap::soa::SoaByte1<u64,u64>, "SoaByte1", Cluster::Soa, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, optimap::soa::SoaByte0_1bit<u64,u64>, "SoaByte0_1bit", Cluster::Soa, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, optimap::soa::SoaByte0_128_1bit<u64,u64>, "SoaByte0_128_1bit", Cluster::Soa, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, optimap::soa::SoaByte7_128And<u64,u64>, "SoaByte7_128And", Cluster::Soa, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, optimap::soa::SoaByte7_255And<u64,u64>, "SoaByte7_255And", Cluster::Soa, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, optimap::soa::SoaByte7_128_8bitAnd<u64,u64>, "SoaByte7_128_8bitAnd", Cluster::Soa, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, optimap::soa::SoaByte7_255_8bitAnd<u64,u64>, "SoaByte7_255_8bitAnd", Cluster::Soa, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, optimap::soa::SoaIpo<u64,u64>, "TombSoa", Cluster::Soa, Width::W16 $(, $arg)*);
        for_each_design!(@run $config, $callback, optimap::soa::SoaByte7_128_Tomb<u64,u64>, "SoaByte7_128_Tomb", Cluster::Soa, Width::W16 $(, $arg)*);
    };
    (@run $config:expr, $callback:ident, $ty:ty, $name:expr, $cluster:expr, $width:expr $(, $arg:expr)*) => {
        if design_active($name, $cluster, $width, &$config) {
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
        "Sweep benchmark: max_n={}, {} points, {} trials, set={:?}, width={:?}{}",
        config.max_n,
        points.len(),
        config.trials,
        config.design_set,
        config.width,
        match &config.filter_design {
            Some(d) => format!(", design={d}"),
            None => String::new(),
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
