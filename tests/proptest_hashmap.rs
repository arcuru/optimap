use optimap::{
    Gaps, InPlaceOverflow, Map, OccupiedError, Splitsies, UnorderedFlatMap, IPO64,
};
use proptest::prelude::*;
use std::collections::HashMap;

/// Operation on a hash map. Keys are u16 to force collisions in a small space.
#[derive(Debug, Clone)]
enum Op {
    Insert(u16, u16),
    Remove(u16),
    Get(u16),
    GetKeyValue(u16),
    GetMut(u16, u16),
    RemoveEntry(u16),
    ContainsKey(u16),
    Clear,
    Reserve(u8),
    ShrinkToFit,
    Retain(u16),
    Drain,
    IterCollect,
    TryInsert(u16, u16),
    IntoKeys,
    IntoValues,
}

fn op_strategy() -> impl Strategy<Value = Op> {
    // Weight insert/remove/get heavily, other ops less so
    prop_oneof![
        8 => (any::<u16>(), any::<u16>()).prop_map(|(k, v)| Op::Insert(k, v)),
        4 => any::<u16>().prop_map(Op::Remove),
        4 => any::<u16>().prop_map(Op::Get),
        2 => any::<u16>().prop_map(Op::GetKeyValue),
        2 => (any::<u16>(), any::<u16>()).prop_map(|(k, v)| Op::GetMut(k, v)),
        2 => any::<u16>().prop_map(Op::RemoveEntry),
        2 => any::<u16>().prop_map(Op::ContainsKey),
        2 => (any::<u16>(), any::<u16>()).prop_map(|(k, v)| Op::TryInsert(k, v)),
        1 => Just(Op::Clear),
        1 => any::<u8>().prop_map(Op::Reserve),
        1 => Just(Op::ShrinkToFit),
        1 => any::<u16>().prop_map(Op::Retain),
        1 => Just(Op::Drain),
        1 => Just(Op::IterCollect),
        1 => Just(Op::IntoKeys),
        1 => Just(Op::IntoValues),
    ]
}

fn run_differential<M: Map<u16, u16>>(ops: &[Op]) {
    let mut test: M = Map::new();
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
                let t = test.get(k);
                let r = reference.get(k);
                assert_eq!(t, r, "op {i}: get({k})");
            }
            Op::GetKeyValue(k) => {
                let t = test.get_key_value(k);
                let r = reference.get_key_value(k);
                assert_eq!(t, r, "op {i}: get_key_value({k})");
            }
            Op::GetMut(k, v) => {
                let t = test.get_mut(k);
                let r = reference.get_mut(k);
                match (t, r) {
                    (Some(tv), Some(rv)) => {
                        assert_eq!(*tv, *rv, "op {i}: get_mut({k}) values differ");
                        *tv = *v;
                        *rv = *v;
                    }
                    (None, None) => {}
                    _ => panic!("op {i}: get_mut({k}) presence mismatch"),
                }
            }
            Op::RemoveEntry(k) => {
                let t = test.remove_entry(k);
                let r = reference.remove_entry(k);
                assert_eq!(t, r, "op {i}: remove_entry({k})");
            }
            Op::ContainsKey(k) => {
                let t = test.contains_key(k);
                let r = reference.contains_key(k);
                assert_eq!(t, r, "op {i}: contains_key({k})");
            }
            Op::Clear => {
                test.clear();
                reference.clear();
            }
            Op::Reserve(n) => {
                let n = *n as usize;
                test.reserve(n);
                reference.reserve(n);
                // Don't compare capacity — implementations differ
            }
            Op::ShrinkToFit => {
                test.shrink_to_fit();
                reference.shrink_to_fit();
            }
            Op::Retain(threshold) => {
                let t = *threshold;
                test.retain(|_, v| *v >= t);
                reference.retain(|_, v| *v >= t);
            }
            Op::Drain => {
                let mut t: Vec<_> = test.drain().collect();
                let mut r: Vec<_> = reference.drain().collect();
                t.sort();
                r.sort();
                assert_eq!(t, r, "op {i}: drain contents differ");
            }
            Op::IterCollect => {
                let mut t: Vec<_> = test.iter().map(|(&k, &v)| (k, v)).collect();
                let mut r: Vec<_> = reference.iter().map(|(&k, &v)| (k, v)).collect();
                t.sort();
                r.sort();
                assert_eq!(t, r, "op {i}: iter contents differ");
            }
            Op::TryInsert(k, v) => {
                let t = test.try_insert(*k, *v);
                let existed = reference.contains_key(k);
                if existed {
                    assert_eq!(
                        t,
                        Err(OccupiedError { key: *k, value: *v }),
                        "op {i}: try_insert({k}, {v}) should fail"
                    );
                } else {
                    assert_eq!(t, Ok(()), "op {i}: try_insert({k}, {v}) should succeed");
                    reference.insert(*k, *v);
                }
            }
            Op::IntoKeys => {
                let mut t: Vec<_> = test.keys().copied().collect();
                let mut r: Vec<_> = reference.keys().copied().collect();
                t.sort();
                r.sort();
                assert_eq!(t, r, "op {i}: into_keys contents differ");
                // Don't actually consume — just verify keys() matches
                // (into_keys would consume the map, ending the test)
            }
            Op::IntoValues => {
                let mut t: Vec<_> = test.values().copied().collect();
                let mut r: Vec<_> = reference.values().copied().collect();
                t.sort();
                r.sort();
                assert_eq!(t, r, "op {i}: into_values contents differ");
            }
        }

        assert_eq!(test.len(), reference.len(), "op {i}: len mismatch after {op:?}");
    }

    // Final full verification
    let mut t: Vec<_> = test.iter().map(|(&k, &v)| (k, v)).collect();
    let mut r: Vec<_> = reference.iter().map(|(&k, &v)| (k, v)).collect();
    t.sort();
    r.sort();
    assert_eq!(t, r, "final contents differ");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn ufm_vs_hashmap(ops in proptest::collection::vec(op_strategy(), 0..500)) {
        run_differential::<UnorderedFlatMap<u16, u16>>(&ops);
    }

    #[test]
    fn splitsies_vs_hashmap(ops in proptest::collection::vec(op_strategy(), 0..500)) {
        run_differential::<Splitsies<u16, u16>>(&ops);
    }

    #[test]
    fn ipo_vs_hashmap(ops in proptest::collection::vec(op_strategy(), 0..500)) {
        run_differential::<InPlaceOverflow<u16, u16>>(&ops);
    }

    #[test]
    fn ipo64_vs_hashmap(ops in proptest::collection::vec(op_strategy(), 0..500)) {
        run_differential::<IPO64<u16, u16>>(&ops);
    }

    #[test]
    fn gaps_vs_hashmap(ops in proptest::collection::vec(op_strategy(), 0..500)) {
        run_differential::<Gaps<u16, u16>>(&ops);
    }
}
