use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use unordered_flat_map::{UnorderedFlatMap, UnorderedFlatSet};

/// Compare our map against std::collections::HashMap for correctness.
#[test]
fn map_matches_std_hashmap() {
    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
    let mut ours = UnorderedFlatMap::new();
    let mut std_map = HashMap::new();

    for _ in 0..10_000 {
        let op = rng.gen_range(0..3);
        let key = rng.gen_range(0..500i32);
        let value = rng.gen_range(0..10000i32);

        match op {
            0 => {
                // Insert
                let ours_old = ours.insert(key, value);
                let std_old = std_map.insert(key, value);
                assert_eq!(ours_old, std_old, "insert mismatch for key={key}");
            }
            1 => {
                // Remove
                let ours_removed = ours.remove(&key);
                let std_removed = std_map.remove(&key);
                assert_eq!(ours_removed, std_removed, "remove mismatch for key={key}");
            }
            2 => {
                // Get
                let ours_val = ours.get(&key);
                let std_val = std_map.get(&key);
                assert_eq!(ours_val, std_val, "get mismatch for key={key}");
            }
            _ => unreachable!(),
        }

        assert_eq!(ours.len(), std_map.len(), "length mismatch");
    }

    // Final verification: all keys match
    for (k, v) in &std_map {
        assert_eq!(ours.get(k), Some(v), "final check: key {k} missing or wrong");
    }
}

/// Same approach for the set.
#[test]
fn set_matches_std_hashset() {
    let mut rng = StdRng::seed_from_u64(0xCAFEBABE);
    let mut ours = UnorderedFlatSet::new();
    let mut std_set = HashSet::new();

    for _ in 0..10_000 {
        let op = rng.gen_range(0..3);
        let value = rng.gen_range(0..500i32);

        match op {
            0 => {
                let ours_new = ours.insert(value);
                let std_new = std_set.insert(value);
                assert_eq!(ours_new, std_new, "insert mismatch for value={value}");
            }
            1 => {
                let ours_removed = ours.remove(&value);
                let std_removed = std_set.remove(&value);
                assert_eq!(ours_removed, std_removed, "remove mismatch for value={value}");
            }
            2 => {
                let ours_has = ours.contains(&value);
                let std_has = std_set.contains(&value);
                assert_eq!(ours_has, std_has, "contains mismatch for value={value}");
            }
            _ => unreachable!(),
        }

        assert_eq!(ours.len(), std_set.len(), "length mismatch");
    }
}

/// Stress test with many insert/delete cycles to exercise anti-drift rehashing.
#[test]
fn insert_delete_cycles_stress() {
    let mut rng = StdRng::seed_from_u64(42);
    let mut map = UnorderedFlatMap::new();

    for cycle in 0..20 {
        let n = rng.gen_range(50..200);
        let keys: Vec<i64> = (0..n).map(|_| rng.gen_range(0..1000)).collect();

        for &k in &keys {
            map.insert(k, cycle);
        }

        // Remove a random subset
        let remove_count = rng.gen_range(0..keys.len());
        for &k in &keys[..remove_count] {
            map.remove(&k);
        }
    }

    // Verify integrity: every key we can find has a valid value
    let mut count = 0;
    for (k, v) in map.iter() {
        assert!(map.contains_key(k), "iterator yielded key not found via get");
        assert_eq!(map.get(k), Some(v));
        count += 1;
    }
    assert_eq!(count, map.len());
}

/// Test with all elements hashing to the same group (worst-case collision).
#[test]
fn high_collision_keys() {
    // Use keys that all hash to similar values
    let mut map = UnorderedFlatMap::new();
    for i in 0..200u64 {
        // Shift by 16 to keep low bits similar, stressing overflow handling
        map.insert(i << 16, i);
    }
    assert_eq!(map.len(), 200);
    for i in 0..200u64 {
        assert_eq!(map.get(&(i << 16)), Some(&i));
    }
    // Remove half
    for i in 0..100u64 {
        assert_eq!(map.remove(&(i << 16)), Some(i));
    }
    assert_eq!(map.len(), 100);
    for i in 100..200u64 {
        assert_eq!(map.get(&(i << 16)), Some(&i));
    }
}

/// Edge case: empty map operations.
#[test]
fn empty_map_operations() {
    let map: UnorderedFlatMap<i32, i32> = UnorderedFlatMap::new();
    assert!(map.is_empty());
    assert_eq!(map.len(), 0);
    assert_eq!(map.get(&0), None);
    assert_eq!(map.iter().count(), 0);
    assert_eq!(map.keys().count(), 0);
    assert_eq!(map.values().count(), 0);
}

/// Edge case: single element.
#[test]
fn single_element() {
    let mut map = UnorderedFlatMap::new();
    map.insert(42, "hello");
    assert_eq!(map.len(), 1);
    assert_eq!(map.get(&42), Some(&"hello"));
    assert_eq!(map.remove(&42), Some("hello"));
    assert!(map.is_empty());
    assert_eq!(map.get(&42), None);
}

/// Edge case: insert and remove the same key repeatedly.
#[test]
fn repeated_insert_remove_same_key() {
    let mut map = UnorderedFlatMap::new();
    for i in 0..1000 {
        // Map is empty at start of each iteration (removed at end of previous)
        assert_eq!(map.insert(0, i), None);
        assert_eq!(map.get(&0), Some(&i));
        // Replace with a new value
        assert_eq!(map.insert(0, i + 1000), Some(i));
        assert_eq!(map.get(&0), Some(&(i + 1000)));
        assert_eq!(map.remove(&0), Some(i + 1000));
        assert!(map.is_empty());
    }
}

/// Test that clone produces an independent copy.
#[test]
fn clone_independence() {
    let mut map = UnorderedFlatMap::new();
    for i in 0..100 {
        map.insert(i, i * 10);
    }

    let mut cloned = map.clone();

    // Modify original
    map.insert(999, 9990);
    map.remove(&50);

    // Clone should be unaffected
    assert_eq!(cloned.len(), 100);
    assert_eq!(cloned.get(&50), Some(&500));
    assert_eq!(cloned.get(&999), None);

    // Modify clone
    cloned.insert(888, 8880);
    assert_eq!(map.get(&888), None);
}

/// Test with String keys (heap-allocated, tests proper Drop).
#[test]
fn string_keys_stress() {
    let mut map = UnorderedFlatMap::new();
    let mut rng = StdRng::seed_from_u64(123);

    for _ in 0..5000 {
        let key: String = (0..rng.gen_range(1..20))
            .map(|_| (b'a' + rng.gen_range(0..26)) as char)
            .collect();
        let value = rng.gen_range(0..10000i32);
        map.insert(key, value);
    }

    // No crash = Drop works correctly for all allocations
    drop(map);
}

/// Test set operations correctness.
#[test]
fn set_operations_correctness() {
    let mut rng = StdRng::seed_from_u64(777);
    let a_vals: Vec<i32> = (0..100).map(|_| rng.gen_range(0..200)).collect();
    let b_vals: Vec<i32> = (0..100).map(|_| rng.gen_range(0..200)).collect();

    let a: UnorderedFlatSet<i32> = a_vals.iter().copied().collect();
    let b: UnorderedFlatSet<i32> = b_vals.iter().copied().collect();
    let a_std: HashSet<i32> = a_vals.iter().copied().collect();
    let b_std: HashSet<i32> = b_vals.iter().copied().collect();

    // Union
    let union = a.union(&b);
    let std_union: HashSet<i32> = a_std.union(&b_std).copied().collect();
    assert_eq!(union.len(), std_union.len());
    for v in &std_union {
        assert!(union.contains(v), "union missing {v}");
    }

    // Intersection
    let inter = a.intersection(&b);
    let std_inter: HashSet<i32> = a_std.intersection(&b_std).copied().collect();
    assert_eq!(inter.len(), std_inter.len());
    for v in &std_inter {
        assert!(inter.contains(v), "intersection missing {v}");
    }

    // Difference
    let diff = a.difference(&b);
    let std_diff: HashSet<i32> = a_std.difference(&b_std).copied().collect();
    assert_eq!(diff.len(), std_diff.len());
    for v in &std_diff {
        assert!(diff.contains(v), "difference missing {v}");
    }

    // Symmetric difference
    let sym_diff = a.symmetric_difference(&b);
    let std_sym: HashSet<i32> = a_std.symmetric_difference(&b_std).copied().collect();
    assert_eq!(sym_diff.len(), std_sym.len());
    for v in &std_sym {
        assert!(sym_diff.contains(v), "symmetric_difference missing {v}");
    }
}

/// Test Borrow support: String keys looked up with &str.
#[test]
fn borrow_string_str() {
    let mut map: UnorderedFlatMap<String, i32> = UnorderedFlatMap::new();
    map.insert("hello".to_string(), 1);
    map.insert("world".to_string(), 2);

    // Lookup with &str (Borrow<str> for String)
    assert_eq!(map.get("hello"), Some(&1));
    assert_eq!(map.get("world"), Some(&2));
    assert_eq!(map.get("missing"), None);
    assert!(map.contains_key("hello"));
    assert!(!map.contains_key("missing"));

    // Remove with &str
    assert_eq!(map.remove("hello"), Some(1));
    assert_eq!(map.get("hello"), None);
    assert_eq!(map.len(), 1);

    // Set with String keys, contains with &str
    let mut set: UnorderedFlatSet<String> = UnorderedFlatSet::new();
    set.insert("alpha".to_string());
    set.insert("beta".to_string());
    assert!(set.contains("alpha"));
    assert!(!set.contains("gamma"));
    assert!(set.remove("alpha"));
    assert!(!set.contains("alpha"));
}

/// Test large capacity to exercise multiple rehashes.
#[test]
fn large_scale() {
    let mut map = UnorderedFlatMap::new();
    let n = 100_000;

    for i in 0..n {
        map.insert(i, i);
    }
    assert_eq!(map.len(), n);

    for i in 0..n {
        assert_eq!(map.get(&i), Some(&i));
    }

    // Remove every other
    for i in (0..n).step_by(2) {
        map.remove(&i);
    }
    assert_eq!(map.len(), n / 2);

    for i in 0..n {
        if i % 2 == 0 {
            assert_eq!(map.get(&i), None);
        } else {
            assert_eq!(map.get(&i), Some(&i));
        }
    }
}
