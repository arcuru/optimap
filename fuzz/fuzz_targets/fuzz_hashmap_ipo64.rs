#![no_main]

mod hashmap_harness;

use libfuzzer_sys::fuzz_target;
use optimap::IPO64;

fuzz_target!(|ops: Vec<hashmap_harness::Op>| {
    hashmap_harness::run_differential::<IPO64<u16, u16>>(&ops);
});
