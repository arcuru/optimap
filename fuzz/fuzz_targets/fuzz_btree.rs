#![no_main]

mod btree_harness;

use libfuzzer_sys::fuzz_target;

fuzz_target!(|ops: Vec<btree_harness::Op>| {
    btree_harness::run_differential(&ops);
});
