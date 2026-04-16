#![no_main]

mod hashmap_harness;

use libfuzzer_sys::fuzz_target;
use optimap::Gaps;

fuzz_target!(|ops: Vec<hashmap_harness::Op>| {
    hashmap_harness::run_differential::<Gaps<u16, u16>>(&ops);
});
