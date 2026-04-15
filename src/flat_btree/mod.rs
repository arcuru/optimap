//! FlatBTree — Cache-line-optimized B+ tree.
//!
//! A sorted map storing generic key-value pairs in 256-byte nodes (4 cache lines).
//! Internal nodes hold only keys and child indices to maximize fan-out. Leaf nodes
//! hold keys and values, linked in a doubly-linked chain for O(n) sorted iteration.
//!
//! Linear scan within nodes — the CPU prefetcher loads adjacent cache lines while
//! scanning the first, making this competitive with binary search for typical fan-outs.

mod map;
mod node;
mod raw;

pub use map::FlatBTree;
