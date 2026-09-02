//! Minimal binary for perf stat: pre-seed map to just past resize boundary,
//! then do N lookup passes. Per-pass timings printed to stderr so perf stat
//! captures total over all passes.

use optimap::matrix_types::HighTag128_TombMap;
use optimap::{IPO64, InPlaceOverflow, Map, Splitsies, UnorderedFlatMap};
use std::time::Instant;

#[path = "../benches/bench_helpers.rs"]
mod bench_helpers;
use bench_helpers::Sfc64;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kind = args.get(1).map(|s| s.as_str()).unwrap_or("tomb");
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(920_000);
    let lookups: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50_000);
    let passes: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);

    let mut rng = Sfc64::new(42);
    let keys: Vec<u64> = (0..(n.max(lookups))).map(|_| rng.next_u64()).collect();

    match kind {
        "tomb" => run::<HighTag128_TombMap<u64, u64>>(&keys, n, lookups, passes),
        "hb" => run::<hashbrown::HashMap<u64, u64>>(&keys, n, lookups, passes),
        "ufm" => run::<UnorderedFlatMap<u64, u64>>(&keys, n, lookups, passes),
        "ipo" => run::<InPlaceOverflow<u64, u64>>(&keys, n, lookups, passes),
        "ipo64" => run::<IPO64<u64, u64>>(&keys, n, lookups, passes),
        "split" => run::<Splitsies<u64, u64>>(&keys, n, lookups, passes),
        _ => panic!("kind must be tomb|hb|ufm|ipo|ipo64|split"),
    }
}

/// Isolated lookup loop — marked #[inline(never)] AND called through a
/// function pointer (via black_box) so LLVM can't inline it into the caller.
/// Required for perf-annotate work: this function needs to be its own
/// symbol in the binary.
#[inline(never)]
fn do_lookups<M: Map<u64, u64>>(map: &M, keys: &[u64], lookups: usize) -> u64 {
    let mut sum = 0u64;
    for &k in keys.iter().take(lookups) {
        sum = sum.wrapping_add(*map.get(&k).unwrap_or(&0));
    }
    sum
}

fn run<M: Map<u64, u64>>(keys: &[u64], n: usize, lookups: usize, passes: usize) {
    let mut map = M::new();
    for (i, &k) in keys.iter().enumerate().take(n) {
        map.insert(k, i as u64);
    }
    eprintln!(
        "inserted {} keys; running {} passes of {} lookups",
        n, passes, lookups
    );
    // Force do_lookups to remain a separate symbol by routing through a
    // function pointer that LLVM cannot statically resolve.
    let lookup_fn: fn(&M, &[u64], usize) -> u64 =
        std::hint::black_box(do_lookups::<M> as fn(&M, &[u64], usize) -> u64);
    for p in 0..passes {
        let start = Instant::now();
        let sum = lookup_fn(&map, keys, lookups);
        std::hint::black_box(sum);
        let ns_per_op = start.elapsed().as_nanos() as f64 / lookups as f64;
        eprintln!("pass {}: {:.2} ns/op", p + 1, ns_per_op);
    }
}
