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

impl PerfectHashFunction for ChdPhf {
    fn build(hashes: &[u64], m: usize) -> Result<Self, BuildError> {
        let n = hashes.len();
        assert!(
            m >= n,
            "PHF table size m={m} must be >= key count n={n}"
        );
        assert!(
            m <= u32::MAX as usize,
            "PerfectMap does not support tables larger than u32::MAX slots"
        );

        if n == 0 {
            return Ok(ChdPhf {
                seed: 0,
                displacements: Box::new([]),
                m: m as u32,
            });
        }

        if has_duplicate(hashes) {
            return Err(BuildError::DuplicateHash);
        }

        // r = ceil(n / λ), at least 1.
        let r = ((n + DEFAULT_LAMBDA - 1) / DEFAULT_LAMBDA).max(1);

        for seed_try in 0..MAX_SEED_RETRIES {
            let seed = SEED_BASE.wrapping_add(seed_try as u64);
            match try_build_seed(hashes, m, r, seed) {
                Ok(disp) => {
                    return Ok(ChdPhf {
                        seed,
                        displacements: disp.into_boxed_slice(),
                        m: m as u32,
                    });
                }
                Err(BuildError::DuplicateHash) => return Err(BuildError::DuplicateHash),
                Err(BuildError::Exhausted) => continue,
            }
        }
        Err(BuildError::Exhausted)
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
        let bits = (self.displacements.len() * 32 + 64 /* seed */) as f64;
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
) -> Result<Vec<u32>, BuildError> {
    let n = hashes.len();
    let r_u64 = r as u64;
    let m_u64 = m as u64;

    // Assign each key index to its bucket.
    let mut bucket_of_key: Vec<u32> = Vec::with_capacity(n);
    let mut bucket_sizes: Vec<u32> = vec![0u32; r];
    for &h in hashes {
        let b = bucket_of(h, seed, r_u64);
        bucket_of_key.push(b as u32);
        bucket_sizes[b] += 1;
    }

    // Build a flat (bucket-major) list of key indices via counting-sort:
    // bucket_start[b..b+1] delimits the slice of keys in bucket b.
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

    // Process buckets in descending size order. Ties broken by bucket id
    // for determinism.
    let mut order: Vec<u32> = (0..r as u32).collect();
    order.sort_unstable_by(|&a, &b| bucket_sizes[b as usize].cmp(&bucket_sizes[a as usize]));

    // `occupied` is a dense bitset over `m` slots — cheap to set/test in the
    // inner loop, and lets us reset between bucket attempts by re-zeroing
    // the tentative slots we wrote.
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
                return Err(BuildError::Exhausted);
            }
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
                break;
            }
            d = d.wrapping_add(1);
        }
    }

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
}
