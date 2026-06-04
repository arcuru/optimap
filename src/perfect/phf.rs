//! Perfect-hash function trait.
//!
//! A `PerfectHashFunction` takes a fixed set of u64 key hashes and a target
//! table size `m`, and produces a structure whose `index(hash)` method maps
//! each of the original hashes to a unique slot in `[0, m)`. Concrete
//! algorithms (CHD, PTHash, BBHash, …) plug into this trait so `PerfectMap`
//! is parameterized over algorithm choice without runtime dispatch.

use std::fmt;

/// A perfect hash function over a fixed key set.
///
/// Built from a slice of pre-hashed keys. Once built, `index(hash)` returns
/// the slot for that key — for any key in the original set, the slot is
/// unique. For any key not in the original set, the slot is arbitrary
/// (still in `[0, m)`, but may collide with an original slot).
///
/// Implementors should be cheap to clone if used in a `Send` context.
pub trait PerfectHashFunction: Sized {
    /// Build a PHF over `hashes`, mapping into `[0, m)`.
    ///
    /// `m` must be ≥ `hashes.len()`. `m == hashes.len()` requests a minimal
    /// perfect hash (densest packing, slowest build); `m > hashes.len()`
    /// gives the construction more slack and tends to finish faster.
    ///
    /// Returns `Err(BuildError::Exhausted)` if construction failed within
    /// its internal retry budget (typically: `m` too small or λ-parameter
    /// too aggressive for this key set). Returns
    /// `Err(BuildError::DuplicateHash)` if two input hashes collide — these
    /// genuinely cannot be perfect-hashed by any algorithm.
    fn build(hashes: &[u64], m: usize) -> Result<Self, BuildError>;

    /// Map a u64 hash to its slot in `[0, m)`.
    ///
    /// For hashes that were in the build set: returns the unique pre-assigned
    /// slot. For hashes that were not: returns some slot in `[0, m)` that
    /// may collide with an in-set slot — callers needing miss-safety must
    /// verify membership independently (e.g. via a stored-key compare).
    fn index(&self, hash: u64) -> usize;

    /// Table size `m` the PHF was built for.
    fn capacity(&self) -> usize;

    /// Approximate space overhead, in bits per built key. Diagnostic only.
    fn bits_per_key(&self) -> f64;
}

/// Reasons a PHF build can fail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildError {
    /// The construction algorithm exhausted its retry budget. Try a larger
    /// `m` (load factor > 1.0) or a different algorithm.
    Exhausted,
    /// Two input hashes collided. No PHF can resolve this — the upstream
    /// hash function is producing collisions on the input key set, or the
    /// caller passed duplicate keys.
    DuplicateHash,
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::Exhausted => {
                f.write_str("perfect-hash construction exhausted its retry budget")
            }
            BuildError::DuplicateHash => {
                f.write_str("input contained duplicate u64 hashes (true hasher collision)")
            }
        }
    }
}

impl std::error::Error for BuildError {}
