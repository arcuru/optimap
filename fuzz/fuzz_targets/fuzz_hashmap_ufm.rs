#![no_main]

mod hashmap_harness;

use libfuzzer_sys::fuzz_target;
use optimap::UnorderedFlatMap;

fuzz_target!(|ops: Vec<hashmap_harness::Op>| {
    hashmap_harness::run_differential::<UnorderedFlatMap<u16, u16>>(&ops);
});
