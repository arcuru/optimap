use optimap::{FlatBTree, SortedMap};
use proptest::prelude::*;
use std::collections::BTreeMap;

/// Operations for a sorted map. u8 keys to force node splits/merges with only 256 possible keys.
#[derive(Debug, Clone)]
enum Op {
    Insert(u8, u8),
    Remove(u8),
    Get(u8),
    GetKeyValue(u8),
    GetMut(u8, u8),
    RemoveEntry(u8),
    ContainsKey(u8),
    Clear,
    Reserve(u8),
    ShrinkToFit,
    Retain(u8),
    Drain,
    IterCollect,
    IterSorted,
    FirstKeyValue,
    LastKeyValue,
    PopFirst,
    PopLast,
    Range(u8, u8),
    EntryOrInsert(u8, u8),
    EntryOrDefault(u8),
    EntryAndModify(u8, u8),
    /// Split off everything >= `at` into a new map; verify both halves match the reference.
    SplitOff(u8),
    /// Append a freshly-built map of (k, v) pairs into self; verify against reference.
    Append(Vec<(u8, u8)>),
    /// range_mut: mutate every value in [lo, hi] by adding a delta.
    RangeMut(u8, u8, u8),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        8 => (any::<u8>(), any::<u8>()).prop_map(|(k, v)| Op::Insert(k, v)),
        4 => any::<u8>().prop_map(Op::Remove),
        4 => any::<u8>().prop_map(Op::Get),
        2 => any::<u8>().prop_map(Op::GetKeyValue),
        2 => (any::<u8>(), any::<u8>()).prop_map(|(k, v)| Op::GetMut(k, v)),
        2 => any::<u8>().prop_map(Op::RemoveEntry),
        2 => any::<u8>().prop_map(Op::ContainsKey),
        1 => Just(Op::Clear),
        1 => any::<u8>().prop_map(Op::Reserve),
        1 => Just(Op::ShrinkToFit),
        1 => any::<u8>().prop_map(Op::Retain),
        1 => Just(Op::Drain),
        1 => Just(Op::IterCollect),
        1 => Just(Op::IterSorted),
        1 => Just(Op::FirstKeyValue),
        1 => Just(Op::LastKeyValue),
        1 => Just(Op::PopFirst),
        1 => Just(Op::PopLast),
        2 => (any::<u8>(), any::<u8>()).prop_map(|(a, b)| {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            Op::Range(lo, hi)
        }),
        2 => (any::<u8>(), any::<u8>()).prop_map(|(k, v)| Op::EntryOrInsert(k, v)),
        1 => any::<u8>().prop_map(Op::EntryOrDefault),
        2 => (any::<u8>(), any::<u8>()).prop_map(|(k, v)| Op::EntryAndModify(k, v)),
        1 => any::<u8>().prop_map(Op::SplitOff),
        1 => proptest::collection::vec((any::<u8>(), any::<u8>()), 0..40).prop_map(Op::Append),
        2 => (any::<u8>(), any::<u8>(), any::<u8>()).prop_map(|(a, b, d)| {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            Op::RangeMut(lo, hi, d)
        }),
    ]
}

fn run_differential(ops: &[Op]) {
    let mut test: FlatBTree<u8, u8> = FlatBTree::new();
    let mut reference: BTreeMap<u8, u8> = BTreeMap::new();

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
                test.reserve(*n as usize);
                // BTreeMap has no reserve — just verify test map still works
            }
            Op::ShrinkToFit => {
                test.shrink_to_fit();
                // BTreeMap doesn't have shrink_to_fit
            }
            Op::Retain(threshold) => {
                let t = *threshold;
                test.retain(|_, v| *v >= t);
                reference.retain(|_, v| *v >= t);
            }
            Op::Drain => {
                let mut t: Vec<_> = test.drain().collect();
                t.sort();
                let r: Vec<_> = reference.iter().map(|(&k, &v)| (k, v)).collect();
                assert_eq!(t, r, "op {i}: drain contents differ");
                reference.clear();
            }
            Op::IterCollect => {
                let mut t: Vec<_> = test.iter().map(|(&k, &v)| (k, v)).collect();
                let mut r: Vec<_> = reference.iter().map(|(&k, &v)| (k, v)).collect();
                t.sort();
                r.sort();
                assert_eq!(t, r, "op {i}: iter contents differ");
            }
            Op::IterSorted => {
                let t: Vec<_> = test.iter_sorted().map(|(&k, &v)| (k, v)).collect();
                let r: Vec<_> = reference.iter().map(|(&k, &v)| (k, v)).collect();
                assert_eq!(t, r, "op {i}: iter_sorted order differs");
            }
            Op::FirstKeyValue => {
                let t = test.first_key_value();
                let r = reference.first_key_value();
                assert_eq!(t, r, "op {i}: first_key_value");
            }
            Op::LastKeyValue => {
                let t = test.last_key_value();
                let r = reference.last_key_value();
                assert_eq!(t, r, "op {i}: last_key_value");
            }
            Op::PopFirst => {
                let t = test.pop_first();
                let r = reference.pop_first();
                assert_eq!(t, r, "op {i}: pop_first");
            }
            Op::PopLast => {
                let t = test.pop_last();
                let r = reference.pop_last();
                assert_eq!(t, r, "op {i}: pop_last");
            }
            Op::Range(lo, hi) => {
                let t: Vec<_> = test.range(*lo..=*hi).map(|(&k, &v)| (k, v)).collect();
                let r: Vec<_> = reference.range(*lo..=*hi).map(|(&k, &v)| (k, v)).collect();
                assert_eq!(t, r, "op {i}: range({lo}..={hi})");
            }
            Op::EntryOrInsert(k, v) => {
                let tv = *test.entry(*k).or_insert(*v);
                let rv = *reference.entry(*k).or_insert(*v);
                assert_eq!(tv, rv, "op {i}: entry({k}).or_insert({v})");
            }
            Op::EntryOrDefault(k) => {
                let tv = *test.entry(*k).or_default();
                let rv = *reference.entry(*k).or_default();
                assert_eq!(tv, rv, "op {i}: entry({k}).or_default()");
            }
            Op::EntryAndModify(k, v) => {
                test.entry(*k)
                    .and_modify(|e| *e = e.wrapping_add(*v))
                    .or_insert(*v);
                reference
                    .entry(*k)
                    .and_modify(|e| *e = e.wrapping_add(*v))
                    .or_insert(*v);
            }
            Op::SplitOff(at) => {
                let t_right = test.split_off(at);
                let r_right = reference.split_off(at);
                let tr: Vec<_> = t_right.iter_sorted().map(|(&k, &v)| (k, v)).collect();
                let rr: Vec<_> = r_right.iter().map(|(&k, &v)| (k, v)).collect();
                assert_eq!(tr, rr, "op {i}: split_off({at}) right side");
                let tl: Vec<_> = test.iter_sorted().map(|(&k, &v)| (k, v)).collect();
                let rl: Vec<_> = reference.iter().map(|(&k, &v)| (k, v)).collect();
                assert_eq!(tl, rl, "op {i}: split_off({at}) left side");
            }
            Op::RangeMut(lo, hi, d) => {
                for (_, v) in test.range_mut(*lo..=*hi) {
                    *v = v.wrapping_add(*d);
                }
                for (_, v) in reference.range_mut(*lo..=*hi) {
                    *v = v.wrapping_add(*d);
                }
            }
            Op::Append(pairs) => {
                let mut t_other: FlatBTree<u8, u8> = FlatBTree::new();
                let mut r_other: BTreeMap<u8, u8> = BTreeMap::new();
                for (k, v) in pairs {
                    t_other.insert(*k, *v);
                    r_other.insert(*k, *v);
                }
                test.append(&mut t_other);
                reference.append(&mut r_other);
                assert!(t_other.is_empty(), "op {i}: append left other non-empty");
                assert!(
                    r_other.is_empty(),
                    "op {i}: ref append left other non-empty"
                );
                let tl: Vec<_> = test.iter_sorted().map(|(&k, &v)| (k, v)).collect();
                let rl: Vec<_> = reference.iter().map(|(&k, &v)| (k, v)).collect();
                assert_eq!(tl, rl, "op {i}: append result");
            }
        }

        assert_eq!(
            test.len(),
            reference.len(),
            "op {i}: len mismatch after {op:?}"
        );
    }

    // Final full verification — sorted order must match
    let t: Vec<_> = test.iter_sorted().map(|(&k, &v)| (k, v)).collect();
    let r: Vec<_> = reference.iter().map(|(&k, &v)| (k, v)).collect();
    assert_eq!(t, r, "final sorted contents differ");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn flatbtree_vs_btreemap(ops in proptest::collection::vec(op_strategy(), 0..500)) {
        run_differential(&ops);
    }
}
