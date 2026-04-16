#![no_main]

mod hashmap_harness;

use libfuzzer_sys::fuzz_target;
use optimap::InPlaceOverflow;

fuzz_target!(|ops: Vec<hashmap_harness::Op>| {
    hashmap_harness::run_differential::<InPlaceOverflow<u16, u16>>(&ops);
});
