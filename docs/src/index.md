# OptiMap

A Rust library providing multiple SIMD-accelerated hash map implementations
with different performance trade-offs, benchmarked against hashbrown (Rust's std HashMap).

## Designs at a Glance

| Design | Key Idea | Best At |
|--------|----------|---------|
| **UnorderedFlatMap** | 15-slot groups, overflow byte | High-load miss, churn |
| **Splitsies** | 16-slot, separate overflow array | Balanced (miss + insert), tombstone-free |
| **InPlaceOverflow** | 16-slot Swiss-table style | Lookup hit, insert |
| **IPO64** | 64-slot cache-line, AVX-512 | Specialty: high-load resilience |
| **Gaps** | 15-slot + power-of-2 buckets | Iteration |

## Common Properties

- All designs use **foldhash** by default (avalanching, fast)
- Overflow-bit designs (UFM, Splitsies, Gaps) are **tombstone-free** with O(1) miss termination
- IPO/IPO64 use **tombstones** like hashbrown but with 254 hash values (vs hashbrown's 128)
- 70% default load factor across all designs
- Generic `Map` trait allows benchmarking all implementations + hashbrown uniformly

## Building

```bash
# Uses flake.nix devShell (direnv auto-activates)
cargo test
cargo bench
```

Requires Rust nightly (for SIMD intrinsics).
