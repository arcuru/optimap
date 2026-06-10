#![no_main]

mod hashmap_harness;

use libfuzzer_sys::fuzz_target;
use optimap::matrix_types::HighTomb_Tomb64Map;

fuzz_target!(|ops: Vec<hashmap_harness::Op>| {
    hashmap_harness::run_differential::<HighTomb_Tomb64Map<u16, u16>>(&ops);
});
