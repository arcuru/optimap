#![no_main]

mod hashmap_harness;

use libfuzzer_sys::fuzz_target;
use optimap::Splitsies;

fuzz_target!(|ops: Vec<hashmap_harness::Op>| {
    hashmap_harness::run_differential::<Splitsies<u16, u16>>(&ops);
});
