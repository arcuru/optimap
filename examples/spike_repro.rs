//! One-off: reproduce the lookup_hit spike at Tomb's resize boundary near N=937K.
//! Times multiple back-to-back passes to distinguish cold-touch from steady state.

use optimap::Map;
use optimap::matrix_types::HighTag128_TombMap;
use std::time::Instant;

#[path = "../benches/bench_helpers.rs"]
mod bench_helpers;
use bench_helpers::Sfc64;

fn time_pass<M: Map<u64, u64>>(map: &M, keys: &[u64], n_lookups: usize) -> f64 {
    let start = Instant::now();
    let mut sum = 0u64;
    for &k in &keys[..n_lookups] {
        sum = sum.wrapping_add(*map.get(&k).unwrap_or(&0));
    }
    std::hint::black_box(sum);
    let elapsed = start.elapsed().as_nanos() as f64;
    elapsed / n_lookups as f64
}

fn run<M: Map<u64, u64>>(label: &str, keys: &[u64], warmup: bool) {
    // Inserting growth points that span Tomb's resize at ~917K
    let growth_points: Vec<usize> = vec![
        900_000, 920_000, 940_000, 960_000, 990_000, 1_020_000, 1_060_000, 1_100_000,
    ];
    let n_lookups = 50_000;
    println!(
        "\n=== {label} {} ===",
        if warmup {
            "(2K-key I-cache warmup)"
        } else {
            "(no warmup)"
        }
    );
    println!(
        "{:>9} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "N", "P1(ns)", "P2(ns)", "P3(ns)", "P4(ns)", "P5(ns)"
    );

    let mut map = M::new();
    let mut prev = 0;
    for &n in &growth_points {
        for (i, &k) in (prev..n).zip(&keys[prev..n]) {
            map.insert(k, i as u64);
        }
        prev = n;
        // I-cache warmup: call time_pass itself so the warmup primes the
        // exact same monomorphized function the measurement uses.
        if warmup {
            let _ = time_pass(&map, keys, 2000);
        }
        let passes: Vec<f64> = (0..5).map(|_| time_pass(&map, keys, n_lookups)).collect();
        println!(
            "{:>9} {:>7.2}  {:>7.2}  {:>7.2}  {:>7.2}  {:>7.2}",
            n, passes[0], passes[1], passes[2], passes[3], passes[4]
        );
    }
}

fn main() {
    let mut rng = Sfc64::new(42);
    let keys: Vec<u64> = (0..1_200_000).map(|_| rng.next_u64()).collect();
    run::<HighTag128_TombMap<u64, u64>>("Tomb (HighTag128)", &keys, false);
    run::<HighTag128_TombMap<u64, u64>>("Tomb (HighTag128)", &keys, true);
    run::<hashbrown::HashMap<u64, u64>>("hashbrown", &keys, false);
    run::<hashbrown::HashMap<u64, u64>>("hashbrown", &keys, true);
}
