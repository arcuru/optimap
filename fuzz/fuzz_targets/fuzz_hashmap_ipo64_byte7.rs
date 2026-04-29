#![no_main]

mod hashmap_harness;

use libfuzzer_sys::fuzz_target;
use optimap::matrix_types::Byte7_254_Tomb64Map;

fuzz_target!(|ops: Vec<hashmap_harness::Op>| {
    hashmap_harness::run_differential::<Byte7_254_Tomb64Map<u16, u16>>(&ops);
});
