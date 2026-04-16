use arbitrary::Arbitrary;
use optimap::Map;
use std::collections::HashMap;

#[derive(Arbitrary, Debug)]
pub enum Op {
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
}

pub fn run_differential<M: Map<u16, u16>>(ops: &[Op]) {
    let mut test: M = Map::new();
    let mut reference: HashMap<u16, u16> = HashMap::new();

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
                reference.reserve(*n as usize);
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
                assert_eq!(t, r);
            }
            Op::IterCollect => {
                let mut t: Vec<_> = test.iter().map(|(&k, &v)| (k, v)).collect();
                let mut r: Vec<_> = reference.iter().map(|(&k, &v)| (k, v)).collect();
                t.sort();
                r.sort();
                assert_eq!(t, r);
            }
        }

        assert_eq!(test.len(), reference.len());
    }

    // Final verification
    let mut t: Vec<_> = test.iter().map(|(&k, &v)| (k, v)).collect();
    let mut r: Vec<_> = reference.iter().map(|(&k, &v)| (k, v)).collect();
    t.sort();
    r.sort();
    assert_eq!(t, r);
}
