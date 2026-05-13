//! Property-based tests for `raw_entry` / `raw_entry_mut`.
//!
//! Cross-validates random sequences of mixed `insert` / `remove` / `get` and
//! raw-entry variants against `std::collections::HashMap`. Each backend gets
//! its own test fn so a failing seed names which design tripped.
//!
//! Op set:
//!
//! - regular `insert` / `remove` / `get` / `get_mut`
//! - `raw_entry().from_key(k)` — read
//! - `raw_entry().from_key_hashed_nocheck(h, k)` — read with caller hash
//! - `raw_entry_mut().from_key(k)` + Vacant::insert / Occupied::insert
//! - `raw_entry_mut().from_key(k)` + Occupied::remove_entry
//!
//! Keys are `u16` to force collisions in a small space. The raw-table types
//! behind each map alias are crate-private; instead of being generic over
//! `R: RawTableApi`, the differential loop is emitted once per map type by
//! macro so it works against the public alias.

use std::collections::HashMap;
use std::hash::BuildHasher;

use optimap::raw_entry::RawEntryMut;
use optimap::{Gaps, IPO64, InPlaceOverflow, SoaMap, Splitsies, UnorderedFlatMap};
use proptest::prelude::*;

#[derive(Debug, Clone)]
enum Op {
    Insert(u16, u16),
    Remove(u16),
    Get(u16),
    GetMut(u16, u16),
    RawFromKey(u16),
    RawFromKeyHashed(u16),
    RawMutSet(u16, u16),
    RawMutRemove(u16),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        5 => (any::<u16>(), any::<u16>()).prop_map(|(k, v)| Op::Insert(k, v)),
        3 => any::<u16>().prop_map(Op::Remove),
        3 => any::<u16>().prop_map(Op::Get),
        2 => (any::<u16>(), any::<u16>()).prop_map(|(k, v)| Op::GetMut(k, v)),
        4 => any::<u16>().prop_map(Op::RawFromKey),
        4 => any::<u16>().prop_map(Op::RawFromKeyHashed),
        3 => (any::<u16>(), any::<u16>()).prop_map(|(k, v)| Op::RawMutSet(k, v)),
        2 => any::<u16>().prop_map(Op::RawMutRemove),
    ]
}

/// Emit a function `$name(&[Op])` that drives the differential loop on map
/// type `$map`. Generated this way (rather than over a `RawTableApi` generic)
/// because the raw-table types behind each backend are crate-private.
macro_rules! differential_fn {
    ($name:ident, $map:ty) => {
        fn $name(ops: &[Op]) {
            let mut test: $map = <$map>::new();
            let mut reference: HashMap<u16, u16> = HashMap::new();

            for (i, op) in ops.iter().enumerate() {
                match op {
                    Op::Insert(k, v) => {
                        let t = test.insert(*k, *v);
                        let r = reference.insert(*k, *v);
                        assert_eq!(t, r, "op {i}: insert({k}, {v})");
                    }
                    Op::Remove(k) => {
                        let t = test.remove(k);
                        let r = reference.remove(k);
                        assert_eq!(t, r, "op {i}: remove({k})");
                    }
                    Op::Get(k) => {
                        let t = test.get(k).copied();
                        let r = reference.get(k).copied();
                        assert_eq!(t, r, "op {i}: get({k})");
                    }
                    Op::GetMut(k, v) => {
                        let t = test.get_mut(k);
                        let r = reference.get_mut(k);
                        match (t, r) {
                            (Some(tv), Some(rv)) => {
                                assert_eq!(*tv, *rv, "op {i}: get_mut({k}) value differs");
                                *tv = *v;
                                *rv = *v;
                            }
                            (None, None) => {}
                            _ => panic!("op {i}: get_mut({k}) presence mismatch"),
                        }
                    }
                    Op::RawFromKey(k) => {
                        let t = test.raw_entry().from_key(k).map(|(_, v)| *v);
                        let r = reference.get(k).copied();
                        assert_eq!(t, r, "op {i}: raw from_key({k})");
                    }
                    Op::RawFromKeyHashed(k) => {
                        let h = test.hasher().hash_one(*k);
                        let t = test
                            .raw_entry()
                            .from_key_hashed_nocheck(h, k)
                            .map(|(_, v)| *v);
                        let r = reference.get(k).copied();
                        assert_eq!(t, r, "op {i}: raw from_key_hashed_nocheck({k})");
                    }
                    Op::RawMutSet(k, v) => {
                        match test.raw_entry_mut().from_key(k) {
                            RawEntryMut::Occupied(mut e) => {
                                e.insert(*v);
                            }
                            RawEntryMut::Vacant(e) => {
                                e.insert(*k, *v);
                            }
                        }
                        reference.insert(*k, *v);
                    }
                    Op::RawMutRemove(k) => {
                        let t = match test.raw_entry_mut().from_key(k) {
                            RawEntryMut::Occupied(e) => Some(e.remove_entry()),
                            RawEntryMut::Vacant(_) => None,
                        };
                        let r = reference.remove_entry(k);
                        assert_eq!(t, r, "op {i}: raw remove_entry({k})");
                    }
                }

                assert_eq!(test.len(), reference.len(), "op {i}: len after {op:?}");
            }

            let mut t: Vec<_> = test.iter().map(|(&k, &v)| (k, v)).collect();
            let mut r: Vec<_> = reference.iter().map(|(&k, &v)| (k, v)).collect();
            t.sort();
            r.sort();
            assert_eq!(t, r, "final contents differ");
        }
    };
}

differential_fn!(ufm_run, UnorderedFlatMap<u16, u16>);
differential_fn!(splitsies_run, Splitsies<u16, u16>);
differential_fn!(ipo_run, InPlaceOverflow<u16, u16>);
differential_fn!(ipo64_run, IPO64<u16, u16>);
differential_fn!(gaps_run, Gaps<u16, u16>);
differential_fn!(soa_run, SoaMap<u16, u16>);

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn ufm_raw_entry_vs_std(ops in proptest::collection::vec(op_strategy(), 0..400)) {
        ufm_run(&ops);
    }

    #[test]
    fn splitsies_raw_entry_vs_std(ops in proptest::collection::vec(op_strategy(), 0..400)) {
        splitsies_run(&ops);
    }

    #[test]
    fn ipo_raw_entry_vs_std(ops in proptest::collection::vec(op_strategy(), 0..400)) {
        ipo_run(&ops);
    }

    #[test]
    fn ipo64_raw_entry_vs_std(ops in proptest::collection::vec(op_strategy(), 0..400)) {
        ipo64_run(&ops);
    }

    #[test]
    fn gaps_raw_entry_vs_std(ops in proptest::collection::vec(op_strategy(), 0..400)) {
        gaps_run(&ops);
    }

    #[test]
    fn soa_raw_entry_vs_std(ops in proptest::collection::vec(op_strategy(), 0..400)) {
        soa_run(&ops);
    }
}
