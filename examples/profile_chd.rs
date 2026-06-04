//! Per-phase breakdown of `ChdPhf` construction at several N. Built to
//! answer "where do the ~740 ms at N=1M go?".
//!
//! Run with `cargo run --release --example profile_chd`. Optionally pass
//! a comma-separated list of sizes, e.g. `... --example profile_chd -- 10000,100000,1000000,2000000`.
//!
//! For each N the harness:
//!   1. Generates `n` distinct u64 hashes from an SFC64 stream (no
//!      collisions in practice; if any do appear, the run aborts so the
//!      profile reflects a clean minimal-perfect build).
//!   2. Calls `ChdPhf::build_with_profile(&hashes, n)`.
//!   3. Prints total wall time and the per-phase split alongside the
//!      attempt-counter diagnostics.

use optimap::{ChdPhf, PerfectHashFunction};
use std::time::Duration;

#[path = "../benches/bench_helpers.rs"]
mod bench_helpers;
use bench_helpers::Sfc64;

fn parse_sizes(arg: Option<&str>) -> Vec<usize> {
    match arg {
        Some(s) => s
            .split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .collect(),
        None => vec![10_000, 100_000, 1_000_000],
    }
}

fn fmt_ms(d: Duration) -> String {
    let us = d.as_micros();
    if us >= 1_000 {
        format!("{:>9.3} ms", us as f64 / 1_000.0)
    } else {
        format!("{:>9.3} µs", us as f64)
    }
}

fn pct(part: Duration, whole: Duration) -> String {
    if whole.is_zero() {
        return "  --".to_string();
    }
    let p = part.as_nanos() as f64 / whole.as_nanos() as f64 * 100.0;
    format!("{:>5.1}%", p)
}

fn unique_hashes(n: usize, seed: u64) -> Vec<u64> {
    let mut rng = Sfc64::new(seed);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(rng.next_u64());
    }
    // Sanity check: u64 collisions on a random stream are vanishingly rare
    // at the sizes we care about, but bail explicitly if they do happen so
    // the profile reflects a clean build path.
    let mut sorted = v.clone();
    sorted.sort_unstable();
    for pair in sorted.windows(2) {
        if pair[0] == pair[1] {
            panic!("input stream produced a u64 collision at n={n} — pick a new seed");
        }
    }
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sizes = parse_sizes(args.get(1).map(String::as_str));
    // Second arg: m / n load factor. 1.0 (default) is minimal-perfect; > 1
    // gives the displacement search slack and slashes the last buckets'
    // attempt counts (see roadmap "Build-path profiling" section).
    let m_factor: f64 = args
        .get(2)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0);

    for &n in &sizes {
        let m = ((n as f64) * m_factor).ceil() as usize;
        println!();
        println!(
            "─── ChdPhf::build_with_profile  n = {n}  m = {m}  (m/n = {m_factor:.3}) ───"
        );
        let hashes = unique_hashes(n, 0x9E37_79B9_7F4A_7C15);
        let (phf, p) = ChdPhf::build_with_profile(&hashes, m)
            .expect("build at the default λ=5 should succeed");
        debug_assert!(phf.bits_per_key().is_finite());

        // Sum of the per-retry phase Durations. `total` covers everything
        // (duplicate check + this sum + leftover scheduling), so the
        // residual exposes any work outside the instrumented phases.
        let inner_sum = p.bucket_assign
            + p.counting_sort
            + p.order_sort
            + p.displacement_search;
        let accounted = p.duplicate_check + inner_sum;
        let residual = p.total.saturating_sub(accounted);

        println!("  total              {}             100.0%", fmt_ms(p.total));
        println!(
            "    duplicate check  {}             {}",
            fmt_ms(p.duplicate_check),
            pct(p.duplicate_check, p.total)
        );
        println!(
            "    bucket assign    {}             {}",
            fmt_ms(p.bucket_assign),
            pct(p.bucket_assign, p.total)
        );
        println!(
            "    counting sort    {}             {}",
            fmt_ms(p.counting_sort),
            pct(p.counting_sort, p.total)
        );
        println!(
            "    order sort       {}             {}",
            fmt_ms(p.order_sort),
            pct(p.order_sort, p.total)
        );
        println!(
            "    displacement     {}             {}",
            fmt_ms(p.displacement_search),
            pct(p.displacement_search, p.total)
        );
        println!(
            "    residual         {}             {}",
            fmt_ms(residual),
            pct(residual, p.total)
        );

        println!();
        println!(
            "  seed_retries           {:>10}",
            p.seed_retries
        );
        println!(
            "  bucket_count           {:>10}    (n / λ = {} / 5)",
            p.bucket_count, n
        );
        println!(
            "  max_bucket_size        {:>10}",
            p.max_bucket_size
        );
        println!(
            "  displacement_attempts  {:>10}    ({:.2} avg per bucket)",
            p.displacement_attempts,
            p.displacement_attempts as f64 / p.bucket_count.max(1) as f64
        );
        println!(
            "  max_displacement_used  {:>10}",
            p.max_displacement_used
        );
        println!(
            "  bits_per_key (PHF)     {:>10.3}",
            phf.bits_per_key()
        );
    }
}
