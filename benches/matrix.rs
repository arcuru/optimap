//! Design space matrix benchmark.
//!
//! Tests all combinations of tag strategy × overflow strategy to find
//! optimal configurations. Each entry benchmarks hit, miss, insert, and
//! remove at medium and large sizes.

mod bench_helpers;

use bench_helpers::*;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use optimap::matrix_types::*;
use optimap::{Gaps, InPlaceOverflow, Splitsies, UnorderedFlatMap};

// ── Table geometry ─────────────────────────────────────────────────────────

const MEDIUM_CAPACITY: usize = 13_440;
const LARGE_CAPACITY: usize = 107_520;
const LOAD_PCT: usize = 70;

struct TestSize {
    name: &'static str,
    capacity: usize,
    num_entries: usize,
}

fn test_sizes() -> Vec<TestSize> {
    vec![
        TestSize {
            name: "medium",
            capacity: MEDIUM_CAPACITY,
            num_entries: entries_for_load(MEDIUM_CAPACITY, LOAD_PCT),
        },
        TestSize {
            name: "large",
            capacity: LARGE_CAPACITY,
            num_entries: entries_for_load(LARGE_CAPACITY, LOAD_PCT),
        },
    ]
}

// ── Matrix macro ───────────────────────────────────────────────────────────

macro_rules! matrix_maps {
    ($helper:ident, $group:expr, $($args:expr),*) => {
        // Baselines
        $helper::<Splitsies<u64, u64>>($group, "Lo8_8bit", $($args),*);
        $helper::<UnorderedFlatMap<u64, u64>>($group, "Ufm", $($args),*);
        $helper::<Gaps<u64, u64>>($group, "Gaps", $($args),*);
        // 16-slot embedded-overflow matrix entries (other tags)
        $helper::<LowTag255ChSafe_EmbMap<u64, u64>>($group, "LowTag255ChSafe_Emb", $($args),*);
        $helper::<LowTag255ChSafe_EmbP2Map<u64, u64>>($group, "LowTag255ChSafe_EmbP2", $($args),*);
        $helper::<LowTag128_EmbMap<u64, u64>>($group, "LowTag128_Emb", $($args),*);
        $helper::<LowTag128_EmbP2Map<u64, u64>>($group, "LowTag128_EmbP2", $($args),*);
        $helper::<HighTag128ChSafe_EmbAndMap<u64, u64>>($group, "HighTag128ChSafe_EmbAnd", $($args),*);
        $helper::<HighTag128ChSafe_EmbP2AndMap<u64, u64>>($group, "HighTag128ChSafe_EmbP2And", $($args),*);
        $helper::<HighTag255ChSafe_EmbAndMap<u64, u64>>($group, "HighTag255ChSafe_EmbAnd", $($args),*);
        $helper::<HighTag255ChSafe_EmbP2AndMap<u64, u64>>($group, "HighTag255ChSafe_EmbP2And", $($args),*);
        $helper::<InPlaceOverflow<u64, u64>>($group, "Tombstone", $($args),*);
        // 8-bit overflow variants.
        // tag↔channel decorrelation A/B (255 values + 8-bit channel held
        // constant, only tag location differs): the byte-0 control is the
        // `Splitsies` baseline above (= `LowTag255_8bit`, tag bits 0-7,
        // correlated with the channel) vs `LowTag255ChSafe_8bit` (tag bits 8-15,
        // decorrelated). Run `cargo bench --bench matrix` and diff the two.
        $helper::<LowTag255ChSafe_8bitMap<u64, u64>>($group, "LowTag255ChSafe_8bit", $($args),*);
        $helper::<LowTag128_8bitMap<u64, u64>>($group, "LowTag128_8bit", $($args),*);
        // 1-bit overflow variants
        $helper::<LowTag255_1bitMap<u64, u64>>($group, "LowTag255_1bit", $($args),*);
        $helper::<LowTag128_1bitMap<u64, u64>>($group, "LowTag128_1bit", $($args),*);
        // AND-indexed variants
        $helper::<HighTag128_1bitAndMap<u64, u64>>($group, "HighTag128_1bitAnd", $($args),*);
        $helper::<HighTag255_1bitAndMap<u64, u64>>($group, "HighTag255_1bitAnd", $($args),*);
        $helper::<HighTag128_8bitAndMap<u64, u64>>($group, "HighTag128_8bitAnd", $($args),*);
        $helper::<HighTag255_8bitAndMap<u64, u64>>($group, "HighTag255_8bitAnd", $($args),*);
        // 32-slot (AVX2) overflow-bit variants
        $helper::<Splitsies32Map<u64, u64>>($group, "Splitsies32", $($args),*);
        $helper::<Splitsies32_1bitMap<u64, u64>>($group, "Splitsies32_1bit", $($args),*);
        $helper::<LowTag255ChSafe_8bit32Map<u64, u64>>($group, "LowTag255ChSafe_8bit32", $($args),*);
        $helper::<LowTag128_8bit32Map<u64, u64>>($group, "LowTag128_8bit32", $($args),*);
        $helper::<LowTag128_1bit32Map<u64, u64>>($group, "LowTag128_1bit32", $($args),*);
        $helper::<Ufm32Map<u64, u64>>($group, "Ufm32", $($args),*);
        $helper::<Gaps32Map<u64, u64>>($group, "Gaps32", $($args),*);
        // Embedded-overflow matrix entries (other tags)
        $helper::<LowTag255ChSafe_Emb32Map<u64, u64>>($group, "LowTag255ChSafe_Emb32", $($args),*);
        $helper::<LowTag255ChSafe_EmbP232Map<u64, u64>>($group, "LowTag255ChSafe_EmbP232", $($args),*);
        $helper::<LowTag128_Emb32Map<u64, u64>>($group, "LowTag128_Emb32", $($args),*);
        $helper::<LowTag128_EmbP232Map<u64, u64>>($group, "LowTag128_EmbP232", $($args),*);
        $helper::<HighTag128ChSafe_EmbAnd32Map<u64, u64>>($group, "HighTag128ChSafe_EmbAnd32", $($args),*);
        $helper::<HighTag128ChSafe_EmbP2And32Map<u64, u64>>($group, "HighTag128ChSafe_EmbP2And32", $($args),*);
        $helper::<HighTag255ChSafe_EmbAnd32Map<u64, u64>>($group, "HighTag255ChSafe_EmbAnd32", $($args),*);
        $helper::<HighTag255ChSafe_EmbP2And32Map<u64, u64>>($group, "HighTag255ChSafe_EmbP2And32", $($args),*);
        $helper::<HighTag128_1bitAnd32Map<u64, u64>>($group, "HighTag128_1bitAnd32", $($args),*);
        $helper::<HighTag255_1bitAnd32Map<u64, u64>>($group, "HighTag255_1bitAnd32", $($args),*);
        $helper::<HighTag128_8bitAnd32Map<u64, u64>>($group, "HighTag128_8bitAnd32", $($args),*);
        $helper::<HighTag255_8bitAnd32Map<u64, u64>>($group, "HighTag255_8bitAnd32", $($args),*);
        // 64-slot (AVX-512) overflow-bit variants
        $helper::<Splitsies64Map<u64, u64>>($group, "Splitsies64", $($args),*);
        $helper::<Splitsies64_1bitMap<u64, u64>>($group, "Splitsies64_1bit", $($args),*);
        $helper::<LowTag255ChSafe_8bit64Map<u64, u64>>($group, "LowTag255ChSafe_8bit64", $($args),*);
        $helper::<LowTag128_8bit64Map<u64, u64>>($group, "LowTag128_8bit64", $($args),*);
        $helper::<LowTag128_1bit64Map<u64, u64>>($group, "LowTag128_1bit64", $($args),*);
        $helper::<Ufm64Map<u64, u64>>($group, "Ufm64", $($args),*);
        $helper::<Gaps64Map<u64, u64>>($group, "Gaps64", $($args),*);
        $helper::<LowTag255ChSafe_Emb64Map<u64, u64>>($group, "LowTag255ChSafe_Emb64", $($args),*);
        $helper::<LowTag255ChSafe_EmbP264Map<u64, u64>>($group, "LowTag255ChSafe_EmbP264", $($args),*);
        $helper::<LowTag128_Emb64Map<u64, u64>>($group, "LowTag128_Emb64", $($args),*);
        $helper::<LowTag128_EmbP264Map<u64, u64>>($group, "LowTag128_EmbP264", $($args),*);
        $helper::<HighTag128ChSafe_EmbAnd64Map<u64, u64>>($group, "HighTag128ChSafe_EmbAnd64", $($args),*);
        $helper::<HighTag128ChSafe_EmbP2And64Map<u64, u64>>($group, "HighTag128ChSafe_EmbP2And64", $($args),*);
        $helper::<HighTag255ChSafe_EmbAnd64Map<u64, u64>>($group, "HighTag255ChSafe_EmbAnd64", $($args),*);
        $helper::<HighTag255ChSafe_EmbP2And64Map<u64, u64>>($group, "HighTag255ChSafe_EmbP2And64", $($args),*);
        $helper::<HighTag128_1bitAnd64Map<u64, u64>>($group, "HighTag128_1bitAnd64", $($args),*);
        $helper::<HighTag255_1bitAnd64Map<u64, u64>>($group, "HighTag255_1bitAnd64", $($args),*);
        $helper::<HighTag128_8bitAnd64Map<u64, u64>>($group, "HighTag128_8bitAnd64", $($args),*);
        $helper::<HighTag255_8bitAnd64Map<u64, u64>>($group, "HighTag255_8bitAnd64", $($args),*);
        // Tombstone variants
        $helper::<HighTag128_TombMap<u64, u64>>($group, "HighTag128_Tomb", $($args),*);
        // IPO64 tombstone variants
        $helper::<HighTomb_Tomb64Map<u64, u64>>($group, "HighTomb_Tomb64", $($args),*);
        // External control
        $helper::<hashbrown::HashMap<u64, u64>>($group, "hashbrown", $($args),*);
        // SoA variants
        $helper::<optimap::SoaMap<u64, u64>>($group, "SoaMap", $($args),*);
        $helper::<optimap::soa::SoaLowTag128<u64, u64>>($group, "SoaLowTag128", $($args),*);
        $helper::<optimap::soa::SoaLowTag255ChSafe<u64, u64>>($group, "SoaLowTag255ChSafe", $($args),*);
        $helper::<optimap::soa::SoaLowTag255_1bit<u64, u64>>($group, "SoaLowTag255_1bit", $($args),*);
        $helper::<optimap::soa::SoaLowTag128_1bit<u64, u64>>($group, "SoaLowTag128_1bit", $($args),*);
        $helper::<optimap::soa::SoaByte7_128And<u64, u64>>($group, "SoaByte7_128And", $($args),*);
        $helper::<optimap::soa::SoaByte7_255And<u64, u64>>($group, "SoaByte7_255And", $($args),*);
        $helper::<optimap::soa::SoaHighTag128_8bitAnd<u64, u64>>($group, "SoaHighTag128_8bitAnd", $($args),*);
        $helper::<optimap::soa::SoaHighTag255_8bitAnd<u64, u64>>($group, "SoaHighTag255_8bitAnd", $($args),*);
        // SoA tombstone variants
        $helper::<optimap::soa::SoaIpo<u64, u64>>($group, "SoaIpo", $($args),*);
        $helper::<optimap::soa::SoaHighTag128_Tomb<u64, u64>>($group, "SoaHighTag128_Tomb", $($args),*);
    };
}

// ── Benchmarks ─────────────────────────────────────────────────────────────

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix/insert");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        matrix_maps!(bench_insert_for, &mut group, sz.name, &keys, sz.capacity);
    }
    group.finish();
}

fn bench_lookup_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix/lookup_hit");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        matrix_maps!(
            bench_lookup_hit_for,
            &mut group,
            sz.name,
            &keys,
            sz.capacity
        );
    }
    group.finish();
}

fn bench_lookup_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix/lookup_miss");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        let miss_keys = make_miss_keys(sz.num_entries);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        matrix_maps!(
            bench_lookup_miss_for,
            &mut group,
            sz.name,
            &keys,
            &miss_keys,
            sz.capacity
        );
    }
    group.finish();
}

fn bench_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix/remove");
    for sz in test_sizes() {
        let keys = make_random_keys(sz.num_entries, 42);
        group.throughput(Throughput::Elements(sz.num_entries as u64));
        matrix_maps!(bench_remove_for, &mut group, sz.name, &keys, sz.capacity);
    }
    group.finish();
}

criterion_group!(
    matrix,
    bench_insert,
    bench_lookup_hit,
    bench_lookup_miss,
    bench_remove
);
criterion_main!(matrix);
