//! CHD perfect-hash construction (Belazzougui, Botelho, Dietzfelbinger 2009).
//!
//! Given `n` pre-hashed keys and target table size `m ≥ n`, CHD assigns each
//! key to one of `r = n / λ` buckets via a primary hash, sorts buckets by
//! size (largest first), and for each bucket searches for a per-bucket
//! displacement `d` such that every key in the bucket lands in a currently-
//! empty slot via `(secondary_hash(k) + d) mod m`. Lookup is two derived
//! hashes + one displacement-table read + one modular add.
//!
//! Per-key storage: `~32 / λ` bits in the displacement table (default λ=5 →
//! ~6.4 bits/key for the table itself; including padding and the seed, the
//! reported `bits_per_key` is closer to 8).
//!
//! The two derived hashes come from mixing the input u64 hash with an
//! internal CHD seed. The seed is iterated on construction failure to give
//! the algorithm a fresh hash family — so the input `BuildHasher` only
//! needs to be called once per key (at `PerfectMap` construction time),
//! and the same hash is reused across CHD retries.

use super::phf::{BuildError, PerfectHashFunction};
use super::util::{bucket_of, has_duplicate};
use std::time::{Duration, Instant};

/// Default average bucket size. CHD's theoretical lower bound is around
/// λ ≈ 4 for minimal PHFs; λ = 5 is a common practical default — fewer
/// buckets, smaller displacement table, harder construction.
const DEFAULT_LAMBDA: usize = 5;

/// How many independent (seed) hash families to try before giving up.
/// Construction failure within MAX_SEED_RETRIES is extremely rare for
/// reasonable input distributions; bumping it is cheap so set generously.
const MAX_SEED_RETRIES: u32 = 32;

/// Per-bucket displacement search ceiling. CHD's expected per-bucket cost
/// is O(m / (m - occupied)) — at minimal load the last buckets can require
/// many attempts. Cap loosely; if we hit this, the seed family is bad and
/// we retry from scratch.
const MAX_DISPLACEMENT: u32 = 1 << 24;

#[inline(always)]
fn slot_of(h: u64, seed: u64, d: u32, m: u64) -> usize {
    // `d` is mixed *into* the hash, not added after. This gives each
    // displacement value an effectively independent hash family — varying
    // `d` permutes a bucket's keys across `[0, m)` instead of shifting
    // them together (which would fail whenever two keys in the bucket
    // shared a slot residue mod m).
    let mut x = h ^ seed.wrapping_add(0x9E3779B97F4A7C15);
    x = x.wrapping_add((d as u64).wrapping_mul(0xBF58476D1CE4E5B9));
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 32;
    (x % m) as usize
}

/// Per-phase timing + counter breakdown of one CHD construction.
///
/// Returned alongside the PHF from [`ChdPhf::build_with_profile`]. All
/// `Duration` fields under the seed loop (`bucket_assign`, `counting_sort`,
/// `order_sort`, `displacement_search`) are summed across every retry that
/// ran, so the parts add up to roughly `total − duplicate_check`. The
/// `displacement_attempts` counter is incremented once per inner-loop `d`
/// trial, again summed across retries.
///
/// Overhead on the construction path is ~10 `Instant::now` calls per build
/// plus one `u64 += 1` per displacement attempt — sub-microsecond against
/// any non-trivial build, so this is always-on; the trait
/// [`PerfectHashFunction::build`] simply discards the profile.
#[derive(Debug, Default, Clone)]
pub struct ChdBuildProfile {
    /// Wall-clock time from entry to return of `build_with_profile`.
    pub total: Duration,
    /// Time spent in the up-front sort-based duplicate-hash check.
    pub duplicate_check: Duration,
    /// Time spent assigning each key index to its bucket (first pass over
    /// the input hash slice each retry).
    pub bucket_assign: Duration,
    /// Time spent building the flat bucket-major key layout via counting
    /// sort (prefix sums + scatter).
    pub counting_sort: Duration,
    /// Time spent ordering buckets by descending size (and computing
    /// `max_bucket_size`).
    pub order_sort: Duration,
    /// Time spent inside the per-bucket displacement search — the loop
    /// that tries `d = 0, 1, 2, …` until every key in a bucket lands on a
    /// distinct empty slot.
    pub displacement_search: Duration,
    /// Number of independent seed families that actually ran (1 on the
    /// usual happy path; up to `MAX_SEED_RETRIES` when CHD has to back off
    /// after a stuck bucket).
    pub seed_retries: u32,
    /// Number of buckets `r = ceil(n / λ)` the algorithm partitioned the
    /// key set into.
    pub bucket_count: usize,
    /// Largest bucket observed across the retries that ran. CHD's
    /// per-bucket cost grows steeply in `sz`, so this dominates the worst
    /// inner loops.
    pub max_bucket_size: u32,
    /// Total displacement values tried across every retry. Divide by
    /// `bucket_count` to get an effective average; large values usually
    /// indicate a near-saturated table.
    pub displacement_attempts: u64,
    /// Largest `d` actually accepted for a bucket. A useful sanity probe
    /// against `MAX_DISPLACEMENT` and a hint at the hardest bucket.
    pub max_displacement_used: u32,
}

/// CHD perfect hash function. Compact runtime form: one seed + one
/// displacement table.
#[derive(Debug, Clone)]
pub struct ChdPhf {
    seed: u64,
    /// Per-bucket displacements. Length is the bucket count `r`.
    displacements: Box<[u32]>,
    /// Table size `m` (slot domain).
    m: u32,
}

impl ChdPhf {
    /// Build the PHF and return a per-phase timing + counter breakdown
    /// alongside it. The trait-level [`PerfectHashFunction::build`] is a
    /// thin wrapper that calls this and drops the profile.
    pub fn build_with_profile(
        hashes: &[u64],
        m: usize,
    ) -> Result<(Self, ChdBuildProfile), BuildError> {
        let n = hashes.len();
        assert!(m >= n, "PHF table size m={m} must be >= key count n={n}");
        assert!(
            m <= u32::MAX as usize,
            "PerfectMap does not support tables larger than u32::MAX slots"
        );

        let t_total = Instant::now();
        let mut profile = ChdBuildProfile::default();

        if n == 0 {
            profile.total = t_total.elapsed();
            return Ok((
                ChdPhf {
                    seed: 0,
                    displacements: Box::new([]),
                    m: m as u32,
                },
                profile,
            ));
        }

        let t_dup = Instant::now();
        let dup = has_duplicate(hashes);
        let dup_elapsed = t_dup.elapsed();
        if dup {
            return Err(BuildError::DuplicateHash);
        }

        // r = ceil(n / λ), at least 1.
        let r = ((n + DEFAULT_LAMBDA - 1) / DEFAULT_LAMBDA).max(1);

        #[cfg(feature = "parallel-build")]
        {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicBool, Ordering};

            let cancelled = AtomicBool::new(false);
            let duplicate = AtomicBool::new(false);

            let result: Option<(ChdPhf, ChdBuildProfile)> = (0..MAX_SEED_RETRIES)
                .into_par_iter()
                .find_map_any(|seed_try| {
                    if cancelled.load(Ordering::Relaxed) {
                        return None;
                    }
                    let seed = SEED_BASE.wrapping_add(seed_try as u64);
                    let mut local_profile = ChdBuildProfile {
                        duplicate_check: dup_elapsed,
                        bucket_count: r,
                        ..Default::default()
                    };

                    match try_build_seed(hashes, m, r, seed, &mut local_profile, Some(&cancelled)) {
                        Ok(disp) => {
                            cancelled.store(true, Ordering::Release);
                            local_profile.seed_retries = seed_try + 1;
                            local_profile.total = t_total.elapsed();
                            Some((
                                ChdPhf {
                                    seed,
                                    displacements: disp.into_boxed_slice(),
                                    m: m as u32,
                                },
                                local_profile,
                            ))
                        }
                        Err(BuildError::DuplicateHash) => {
                            duplicate.store(true, Ordering::Release);
                            cancelled.store(true, Ordering::Release);
                            None
                        }
                        Err(BuildError::Exhausted) => None,
                    }
                });

            if duplicate.load(Ordering::Acquire) {
                return Err(BuildError::DuplicateHash);
            }
            if let Some((phf, prof)) = result {
                return Ok((phf, prof));
            }
        }

        #[cfg(not(feature = "parallel-build"))]
        {
            profile.duplicate_check = dup_elapsed;
            profile.bucket_count = r;
            for seed_try in 0..MAX_SEED_RETRIES {
                profile.seed_retries = seed_try + 1;
                let seed = SEED_BASE.wrapping_add(seed_try as u64);
                match try_build_seed(hashes, m, r, seed, &mut profile, None) {
                    Ok(disp) => {
                        profile.total = t_total.elapsed();
                        return Ok((
                            ChdPhf {
                                seed,
                                displacements: disp.into_boxed_slice(),
                                m: m as u32,
                            },
                            profile,
                        ));
                    }
                    Err(BuildError::DuplicateHash) => return Err(BuildError::DuplicateHash),
                    Err(BuildError::Exhausted) => continue,
                }
            }
        }
        Err(BuildError::Exhausted)
    }
}

impl PerfectHashFunction for ChdPhf {
    fn build(hashes: &[u64], m: usize) -> Result<Self, BuildError> {
        Self::build_with_profile(hashes, m).map(|(phf, _)| phf)
    }

    #[inline]
    fn index(&self, hash: u64) -> usize {
        if self.m == 0 {
            // Empty key set — no slot can ever be requested validly.
            // Return 0 as a benign sentinel; callers (PerfectMap::get)
            // must check `slots.is_empty()` before indexing.
            return 0;
        }
        let r = self.displacements.len() as u64;
        let m = self.m as u64;
        let bucket = bucket_of(hash, self.seed, r);
        // SAFETY: bucket_of returns a value in [0, r) by construction.
        let d = unsafe { *self.displacements.get_unchecked(bucket) };
        slot_of(hash, self.seed, d, m)
    }

    fn capacity(&self) -> usize {
        self.m as usize
    }

    fn bits_per_key(&self) -> f64 {
        if self.displacements.is_empty() {
            return 0.0;
        }
        // n is unknown to the PHF after build — approximate from λ.
        let n_approx = (self.displacements.len() * DEFAULT_LAMBDA) as f64;
        let bits = (self.displacements.len() * 32 + 64/* seed */) as f64;
        bits / n_approx
    }
}

/// A non-trivial starting seed (splitmix64-style constant) keeps the first
/// attempt's hash family distinct from the input hasher's identity.
const SEED_BASE: u64 = 0xD6E8FEB86659FD93;

/// One CHD construction attempt under a fixed `seed`.
fn try_build_seed(
    hashes: &[u64],
    m: usize,
    r: usize,
    seed: u64,
    profile: &mut ChdBuildProfile,
    cancelled: Option<&std::sync::atomic::AtomicBool>,
) -> Result<Vec<u32>, BuildError> {
    let n = hashes.len();
    let r_u64 = r as u64;
    let m_u64 = m as u64;

    // Phase 1: assign each key index to its bucket.
    let t_assign = Instant::now();
    let mut bucket_of_key: Vec<u32> = Vec::with_capacity(n);
    let mut bucket_sizes: Vec<u32> = vec![0u32; r];
    for &h in hashes {
        let b = bucket_of(h, seed, r_u64);
        bucket_of_key.push(b as u32);
        bucket_sizes[b] += 1;
    }
    profile.bucket_assign += t_assign.elapsed();

    // Phase 2: flat (bucket-major) list of key indices via counting-sort.
    // bucket_start[b..b+1] delimits the slice of keys in bucket b.
    let t_cs = Instant::now();
    let mut bucket_start: Vec<u32> = Vec::with_capacity(r + 1);
    let mut acc: u32 = 0;
    for &sz in &bucket_sizes {
        bucket_start.push(acc);
        acc += sz;
    }
    bucket_start.push(acc);

    let mut bucket_keys: Vec<u32> = vec![0u32; n];
    let mut cursor: Vec<u32> = bucket_start[..r].to_vec();
    for (i, &b) in bucket_of_key.iter().enumerate() {
        let pos = cursor[b as usize] as usize;
        bucket_keys[pos] = i as u32;
        cursor[b as usize] += 1;
    }
    profile.counting_sort += t_cs.elapsed();

    // Phase 3: process buckets in descending size order. Ties broken by
    // bucket id for determinism.
    let t_order = Instant::now();
    let mut order: Vec<u32> = (0..r as u32).collect();
    order.sort_unstable_by(|&a, &b| bucket_sizes[b as usize].cmp(&bucket_sizes[a as usize]));
    let max_bucket_this_retry = bucket_sizes.iter().copied().max().unwrap_or(0);
    if max_bucket_this_retry > profile.max_bucket_size {
        profile.max_bucket_size = max_bucket_this_retry;
    }
    profile.order_sort += t_order.elapsed();

    // Phase 4: per-bucket displacement search. `occupied` is a dense
    // bitset over `m` slots — cheap to set/test in the inner loop, and
    // lets us reset between bucket attempts by re-zeroing the tentative
    // slots we wrote.
    let t_disp = Instant::now();
    let mut occupied: Vec<u64> = vec![0u64; m.div_ceil(64)];
    let mut tentative: Vec<u32> = Vec::with_capacity(16);
    let mut displacements: Vec<u32> = vec![0u32; r];

    for &b in &order {
        let b_us = b as usize;
        let sz = bucket_sizes[b_us] as usize;
        if sz == 0 {
            continue;
        }
        let keys_in_bucket =
            &bucket_keys[bucket_start[b_us] as usize..bucket_start[b_us + 1] as usize];

        // Try displacements 0, 1, 2, … until a value lands every key in
        // this bucket on currently-unoccupied, mutually-distinct slots.
        let mut d: u32 = 0;
        loop {
            if d > MAX_DISPLACEMENT {
                profile.displacement_search += t_disp.elapsed();
                return Err(BuildError::Exhausted);
            }
            // Cooperative cancellation: check whether another seed attempt
            // already succeeded. Sampled every 64 `d` ticks to keep the
            // check overhead negligible (one relaxed atomic load ≈ 0).
            if d & 63 == 0
                && let Some(cancelled) = cancelled
                && cancelled.load(std::sync::atomic::Ordering::Relaxed)
            {
                profile.displacement_search += t_disp.elapsed();
                return Err(BuildError::Exhausted);
            }
            profile.displacement_attempts += 1;
            tentative.clear();
            let mut ok = true;
            for &ki in keys_in_bucket {
                let h = hashes[ki as usize];
                let s = slot_of(h, seed, d, m_u64);
                if bitset_test(&occupied, s) || tentative.contains(&(s as u32)) {
                    ok = false;
                    break;
                }
                tentative.push(s as u32);
            }
            if ok {
                debug_assert_eq!(
                    tentative.len(),
                    sz,
                    "CHD inner loop accepted a displacement but produced the wrong slot count"
                );
                for &s in &tentative {
                    bitset_set(&mut occupied, s as usize);
                }
                displacements[b_us] = d;
                if d > profile.max_displacement_used {
                    profile.max_displacement_used = d;
                }
                break;
            }
            d = d.wrapping_add(1);
        }
    }
    profile.displacement_search += t_disp.elapsed();

    Ok(displacements)
}

#[inline(always)]
fn bitset_test(bits: &[u64], i: usize) -> bool {
    bits[i >> 6] & (1u64 << (i & 63)) != 0
}

#[inline(always)]
fn bitset_set(bits: &mut [u64], i: usize) {
    bits[i >> 6] |= 1u64 << (i & 63);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perfect::util::mix;

    fn make_hashes(n: usize) -> Vec<u64> {
        // Deterministic, well-mixed input. The PHF doesn't care what the
        // upstream hasher is, only that values are unique.
        (0..n as u64)
            .map(|i| mix(i.wrapping_mul(0x9E3779B97F4A7C15), 0x6A09E667F3BCC908))
            .collect()
    }

    #[test]
    fn build_empty() {
        let phf = ChdPhf::build(&[], 0).unwrap();
        assert_eq!(phf.capacity(), 0);
    }

    #[test]
    fn build_single() {
        let phf = ChdPhf::build(&[42], 1).unwrap();
        assert_eq!(phf.index(42), 0);
    }

    #[test]
    fn minimal_small_round_trip() {
        for &n in &[2usize, 8, 64, 256, 1024] {
            let hashes = make_hashes(n);
            let phf = ChdPhf::build(&hashes, n).expect("build should succeed at λ=5, m=n");
            let mut seen = vec![false; n];
            for &h in &hashes {
                let s = phf.index(h);
                assert!(s < n, "slot {s} out of range at n={n}");
                assert!(!seen[s], "collision at slot {s} for n={n}");
                seen[s] = true;
            }
            assert!(seen.iter().all(|&b| b));
        }
    }

    #[test]
    fn non_minimal_round_trip() {
        let n = 10_000;
        let m = (n as f64 * 1.23) as usize;
        let hashes = make_hashes(n);
        let phf = ChdPhf::build(&hashes, m).expect("non-minimal build is easier than minimal");
        assert_eq!(phf.capacity(), m);
        let mut seen = vec![false; m];
        for &h in &hashes {
            let s = phf.index(h);
            assert!(s < m);
            assert!(!seen[s]);
            seen[s] = true;
        }
    }

    #[test]
    fn minimal_medium_round_trip() {
        let n = 50_000;
        let hashes = make_hashes(n);
        let phf = ChdPhf::build(&hashes, n).expect("minimal build at N=50K");
        let mut seen = vec![false; n];
        for &h in &hashes {
            let s = phf.index(h);
            assert!(!seen[s]);
            seen[s] = true;
        }
        assert_eq!(seen.iter().filter(|b| **b).count(), n);
    }

    #[test]
    fn duplicate_hash_is_reported() {
        // Two genuinely-colliding hashes can never be perfect-hashed.
        let hashes = vec![1, 2, 3, 4, 1];
        let err = ChdPhf::build(&hashes, 5).unwrap_err();
        assert_eq!(err, BuildError::DuplicateHash);
    }

    #[test]
    fn build_with_profile_populates_phases() {
        let n = 10_000;
        let hashes = make_hashes(n);
        let (_phf, profile) =
            ChdPhf::build_with_profile(&hashes, n).expect("minimal build at N=10K");

        // Bucket count = ceil(n/λ) = ceil(10000/5) = 2000.
        assert_eq!(profile.bucket_count, 2_000);
        // At λ=5 with well-mixed input the first seed almost always works.
        assert!(profile.seed_retries >= 1);
        // Every phase under the seed loop ran at least once.
        assert!(profile.bucket_assign > Duration::ZERO);
        assert!(profile.counting_sort > Duration::ZERO);
        assert!(profile.order_sort > Duration::ZERO);
        assert!(profile.displacement_search > Duration::ZERO);
        // total covers everything else; should dominate each component.
        assert!(profile.total >= profile.displacement_search);
        // At least r attempts (one per non-empty bucket); usually many more.
        assert!(profile.displacement_attempts >= profile.bucket_count as u64);
        // Largest bucket can't exceed n.
        assert!(profile.max_bucket_size > 0 && profile.max_bucket_size as usize <= n);
    }

    #[test]
    fn build_with_profile_duplicate_reported_with_phase_data() {
        let hashes = vec![1u64, 2, 3, 4, 1];
        let (_dup_check_ran, err) = match ChdPhf::build_with_profile(&hashes, 5) {
            Ok(_) => panic!("duplicate must fail"),
            Err(e) => (true, e),
        };
        assert_eq!(err, BuildError::DuplicateHash);
        assert!(_dup_check_ran);
    }
}
