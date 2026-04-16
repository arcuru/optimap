use arbitrary::Arbitrary;
use optimap::{FlatBTree, SortedMap};
use std::collections::BTreeMap;

#[derive(Arbitrary, Debug)]
pub enum Op {
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
}

pub fn run_differential(ops: &[Op]) {
    let mut test: FlatBTree<u8, u8> = FlatBTree::new();
    let mut reference: BTreeMap<u8, u8> = BTreeMap::new();

    for op in ops {
        match op {
            Op::Insert(k, v) => {
                let t = test.insert(*k, *v);
                let r = reference.insert(*k, *v);
                assert_eq!(t, r);
            }
            Op::Remove(k) => {
                let t = test.remove(k);
                let r = reference.remove(k);
                assert_eq!(t, r);
            }
            Op::Get(k) => {
                let t = test.get(k);
                let r = reference.get(k);
                assert_eq!(t, r);
            }
            Op::GetKeyValue(k) => {
                let t = test.get_key_value(k);
                let r = reference.get_key_value(k);
                assert_eq!(t, r);
            }
            Op::GetMut(k, v) => {
                let t = test.get_mut(k);
                let r = reference.get_mut(k);
                match (t, r) {
                    (Some(tv), Some(rv)) => {
                        assert_eq!(*tv, *rv);
                        *tv = *v;
                        *rv = *v;
                    }
                    (None, None) => {}
                    _ => panic!("get_mut presence mismatch"),
                }
            }
            Op::RemoveEntry(k) => {
                let t = test.remove_entry(k);
                let r = reference.remove_entry(k);
                assert_eq!(t, r);
            }
            Op::ContainsKey(k) => {
                let t = test.contains_key(k);
                let r = reference.contains_key(k);
                assert_eq!(t, r);
            }
            Op::Clear => {
                test.clear();
                reference.clear();
            }
            Op::Reserve(n) => {
                test.reserve(*n as usize);
                // BTreeMap has no reserve
            }
            Op::ShrinkToFit => {
                test.shrink_to_fit();
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
                assert_eq!(t, r);
                reference.clear();
            }
            Op::IterCollect => {
                let mut t: Vec<_> = test.iter().map(|(&k, &v)| (k, v)).collect();
                let mut r: Vec<_> = reference.iter().map(|(&k, &v)| (k, v)).collect();
                t.sort();
                r.sort();
                assert_eq!(t, r);
            }
            Op::IterSorted => {
                let t: Vec<_> = test.iter_sorted().map(|(&k, &v)| (k, v)).collect();
                let r: Vec<_> = reference.iter().map(|(&k, &v)| (k, v)).collect();
                assert_eq!(t, r);
            }
            Op::FirstKeyValue => {
                let t = test.first_key_value();
                let r = reference.first_key_value();
                assert_eq!(t, r);
            }
            Op::LastKeyValue => {
                let t = test.last_key_value();
                let r = reference.last_key_value();
                assert_eq!(t, r);
            }
            Op::PopFirst => {
                let t = test.pop_first();
                let r = reference.pop_first();
                assert_eq!(t, r);
            }
            Op::PopLast => {
                let t = test.pop_last();
                let r = reference.pop_last();
                assert_eq!(t, r);
            }
            Op::Range(lo, hi) => {
                let (lo, hi) = if lo <= hi { (*lo, *hi) } else { (*hi, *lo) };
                let t: Vec<_> = test.range(lo..=hi).map(|(&k, &v)| (k, v)).collect();
                let r: Vec<_> = reference.range(lo..=hi).map(|(&k, &v)| (k, v)).collect();
                assert_eq!(t, r);
            }
            Op::EntryOrInsert(k, v) => {
                let tv = *test.entry(*k).or_insert(*v);
                let rv = *reference.entry(*k).or_insert(*v);
                assert_eq!(tv, rv);
            }
            Op::EntryOrDefault(k) => {
                let tv = *test.entry(*k).or_default();
                let rv = *reference.entry(*k).or_default();
                assert_eq!(tv, rv);
            }
            Op::EntryAndModify(k, v) => {
                test.entry(*k).and_modify(|e| *e = e.wrapping_add(*v)).or_insert(*v);
                reference.entry(*k).and_modify(|e| *e = e.wrapping_add(*v)).or_insert(*v);
            }
        }

        assert_eq!(test.len(), reference.len());
    }

    // Final verification — sorted order must match
    let t: Vec<_> = test.iter_sorted().map(|(&k, &v)| (k, v)).collect();
    let r: Vec<_> = reference.iter().map(|(&k, &v)| (k, v)).collect();
    assert_eq!(t, r);
}
