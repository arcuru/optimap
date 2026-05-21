//! Tests for the `Map` facade trait — exercises generic dispatch across
//! both hash-dispatched and ord-dispatched backends through a single bound.

use optimap::{FlatBTree, Gaps, IPO64, InPlaceOverflow, Map, OptiMap, Splitsies, UnorderedFlatMap};
use std::collections::{BTreeMap, HashMap};

fn exercise<M: Map<i32, i32>>() {
    let mut m: M = Map::new();
    assert!(m.is_empty());
    assert_eq!(Map::len(&m), 0);

    // Insert + len + contains
    assert_eq!(Map::insert(&mut m, 2, 20), None);
    assert_eq!(Map::insert(&mut m, 1, 10), None);
    assert_eq!(Map::insert(&mut m, 3, 30), None);
    assert_eq!(Map::insert(&mut m, 2, 200), Some(20)); // overwrite
    assert_eq!(Map::len(&m), 3);
    assert!(Map::contains_key(&m, &1));
    assert!(!Map::contains_key(&m, &99));

    // get / get_mut / get_key_value
    assert_eq!(Map::get(&m, &1), Some(&10));
    assert_eq!(Map::get_key_value(&m, &2), Some((&2, &200)));
    *Map::get_mut(&mut m, &1).unwrap() = 11;
    assert_eq!(Map::get(&m, &1), Some(&11));

    // try_insert
    assert!(Map::try_insert(&mut m, 4, 40).is_ok());
    let err = Map::try_insert(&mut m, 4, 400).unwrap_err();
    assert_eq!(err.value, 400);

    // remove / remove_entry
    assert_eq!(Map::remove(&mut m, &3), Some(30));
    assert_eq!(Map::remove_entry(&mut m, &1), Some((1, 11)));
    assert_eq!(Map::len(&m), 2);

    // iter sees remaining (order-agnostic)
    let mut pairs: Vec<_> = Map::iter(&m).map(|(&k, &v)| (k, v)).collect();
    pairs.sort();
    assert_eq!(pairs, vec![(2, 200), (4, 40)]);

    // retain
    Map::retain(&mut m, |k, _| *k == 2);
    assert_eq!(Map::len(&m), 1);

    // clear + drain on a refilled map
    Map::clear(&mut m);
    for i in 0..5 {
        m.insert(i, i * 10);
    }
    let mut drained: Vec<_> = Map::drain(&mut m).collect();
    drained.sort();
    assert_eq!(drained, vec![(0, 0), (1, 10), (2, 20), (3, 30), (4, 40)]);
    assert!(m.is_empty());
}

// ── Hash-dispatched backends ────────────────────────────────────────────────

#[test]
fn map_facade_unordered_flat() {
    exercise::<UnorderedFlatMap<i32, i32>>();
}

#[test]
fn map_facade_splitsies() {
    exercise::<Splitsies<i32, i32>>();
}

#[test]
fn map_facade_ipo() {
    exercise::<InPlaceOverflow<i32, i32>>();
}

#[test]
fn map_facade_ipo64() {
    exercise::<IPO64<i32, i32>>();
}

#[test]
fn map_facade_gaps() {
    exercise::<Gaps<i32, i32>>();
}

#[test]
fn map_facade_optimap() {
    exercise::<OptiMap<i32, i32>>();
}

#[test]
fn map_facade_hashbrown() {
    exercise::<hashbrown::HashMap<i32, i32>>();
}

#[test]
fn map_facade_std_hashmap() {
    exercise::<HashMap<i32, i32>>();
}

// ── Ord-dispatched backends ─────────────────────────────────────────────────

#[test]
fn map_facade_flat_btree() {
    exercise::<FlatBTree<i32, i32>>();
}

#[test]
fn map_facade_std_btreemap() {
    exercise::<BTreeMap<i32, i32>>();
}

#[test]
fn map_facade_optimap_flat_btree() {
    // OptiMap pinned to FlatBTree exercises the Map facade through the
    // sorted dispatch path. Mirrors the hash-backend exercise above but
    // constructs via `OptiMap::flat_btree()` instead of `Map::new()`,
    // which would otherwise route through the auto policy.
    let mut m: OptiMap<i32, i32> = OptiMap::flat_btree();
    assert!(m.is_empty());
    assert_eq!(Map::insert(&mut m, 2, 20), None);
    assert_eq!(Map::insert(&mut m, 1, 10), None);
    assert_eq!(Map::insert(&mut m, 3, 30), None);
    assert_eq!(Map::insert(&mut m, 2, 200), Some(20));
    assert_eq!(Map::len(&m), 3);
    assert!(Map::contains_key(&m, &1));
    assert_eq!(Map::get(&m, &1), Some(&10));
    assert_eq!(Map::get_key_value(&m, &2), Some((&2, &200)));
    *Map::get_mut(&mut m, &1).unwrap() = 11;
    assert_eq!(Map::remove(&mut m, &3), Some(30));
    assert_eq!(Map::remove_entry(&mut m, &1), Some((1, 11)));
    Map::retain(&mut m, |k, _| *k == 2);
    assert_eq!(Map::len(&m), 1);
    Map::clear(&mut m);
    assert!(m.is_empty());
}
