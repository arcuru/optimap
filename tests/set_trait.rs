//! Tests for the Set and SortedSet traits, exercised across all concrete types.

use optimap::Set;

// ── Generic test harness ────────────────────────────────────────────────────

fn test_basic_ops<S: Set<i32>>() {
    let mut s = S::new();
    assert!(s.is_empty());
    assert_eq!(s.len(), 0);

    assert!(s.insert(1));
    assert!(s.insert(2));
    assert!(s.insert(3));
    assert!(!s.insert(1));
    assert_eq!(s.len(), 3);

    assert!(s.contains(&1));
    assert!(s.contains(&2));
    assert!(s.contains(&3));
    assert!(!s.contains(&4));

    assert_eq!(s.get(&1), Some(&1));
    assert_eq!(s.get(&99), None);

    assert!(s.remove(&2));
    assert!(!s.remove(&2));
    assert_eq!(s.len(), 2);
    assert!(!s.contains(&2));

    assert_eq!(s.take(&3), Some(3));
    assert_eq!(s.take(&3), None);
    assert_eq!(s.len(), 1);

    s.clear();
    assert!(s.is_empty());
}

fn test_with_capacity<S: Set<i32>>() {
    let s = S::with_capacity(100);
    // capacity() is a hint — B-tree styles may not guarantee exact capacity
    assert!(s.is_empty());
}

fn test_reserve_shrink<S: Set<i32>>() {
    let mut s = S::new();
    s.reserve(200);

    for i in 0..10 {
        s.insert(i);
    }
    s.shrink_to_fit();
    // After shrink, data must still be intact
    assert_eq!(s.len(), 10);
    for i in 0..10 {
        assert!(s.contains(&i));
    }
}

fn test_iter<S: Set<i32>>() {
    let mut s = S::new();
    for i in 0..100 {
        s.insert(i);
    }

    let mut items: Vec<i32> = s.iter().copied().collect();
    items.sort();
    assert_eq!(items, (0..100).collect::<Vec<_>>());
}

fn test_retain<S: Set<i32>>() {
    let mut s = S::new();
    for i in 0..20 {
        s.insert(i);
    }
    s.retain(|&x| x % 3 == 0);
    assert_eq!(s.len(), 7); // 0, 3, 6, 9, 12, 15, 18
    assert!(s.contains(&0));
    assert!(s.contains(&9));
    assert!(!s.contains(&1));
    assert!(!s.contains(&10));
}

fn test_drain<S: Set<i32>>() {
    let mut s = S::new();
    for i in 0..50 {
        s.insert(i);
    }

    let mut drained: Vec<i32> = s.drain().collect();
    drained.sort();
    assert_eq!(drained, (0..50).collect::<Vec<_>>());
    assert!(s.is_empty());
}

fn test_large_scale<S: Set<i32>>() {
    let mut s = S::new();
    for i in 0..5000 {
        assert!(s.insert(i));
    }
    assert_eq!(s.len(), 5000);

    for i in 0..5000 {
        assert!(s.contains(&i));
    }

    for i in 0..2500 {
        assert!(s.remove(&i));
    }
    assert_eq!(s.len(), 2500);

    for i in 0..2500 {
        assert!(s.insert(i));
    }
    assert_eq!(s.len(), 5000);
}

fn test_string_keys<S: Set<String>>() {
    let mut s = S::new();
    s.insert("hello".to_string());
    s.insert("world".to_string());
    assert!(s.contains("hello"));
    assert!(s.contains("world"));
    assert!(!s.contains("foo"));
    assert_eq!(s.get("hello").map(|s| s.as_str()), Some("hello"));
    assert_eq!(s.take("world"), Some("world".to_string()));
    assert_eq!(s.len(), 1);
}

// ── Run all tests for each concrete type ────────────────────────────────────

macro_rules! set_trait_tests {
    ($mod_name:ident, $int_type:ty, $str_type:ty) => {
        mod $mod_name {
            use super::*;

            #[test]
            fn basic_ops() { test_basic_ops::<$int_type>(); }
            #[test]
            fn with_capacity() { test_with_capacity::<$int_type>(); }
            #[test]
            fn reserve_shrink() { test_reserve_shrink::<$int_type>(); }
            #[test]
            fn iter() { test_iter::<$int_type>(); }
            #[test]
            fn retain() { test_retain::<$int_type>(); }
            #[test]
            fn drain() { test_drain::<$int_type>(); }
            #[test]
            fn large_scale() { test_large_scale::<$int_type>(); }
            #[test]
            fn string_keys() { test_string_keys::<$str_type>(); }
        }
    };
}

set_trait_tests!(ufm_set, optimap::UnorderedFlatSet<i32>, optimap::UnorderedFlatSet<String>);
set_trait_tests!(splitsies_set, optimap::SplitsiesSet<i32>, optimap::SplitsiesSet<String>);
set_trait_tests!(ipo_set, optimap::IpoSet<i32>, optimap::IpoSet<String>);
set_trait_tests!(gaps_set, optimap::GapsSet<i32>, optimap::GapsSet<String>);
set_trait_tests!(ipo64_set, optimap::Ipo64Set<i32>, optimap::Ipo64Set<String>);
set_trait_tests!(flat_btree_set, optimap::FlatBTreeSet<i32>, optimap::FlatBTreeSet<String>);
set_trait_tests!(std_hashset, std::collections::HashSet<i32>, std::collections::HashSet<String>);
set_trait_tests!(hashbrown_hashset, hashbrown::HashSet<i32>, hashbrown::HashSet<String>);

// ── SortedSet tests ─────────────────────────────────────────────────────────

mod sorted_set {
    use optimap::SortedSet;

    fn test_sorted_basics<S: SortedSet<i32> + Default + Extend<i32>>() {
        let mut s = S::default();
        s.extend([5, 3, 8, 1, 9, 2, 7]);

        assert_eq!(s.first(), Some(&1));
        assert_eq!(s.last(), Some(&9));

        let sorted: Vec<i32> = s.iter_sorted().copied().collect();
        assert_eq!(sorted, vec![1, 2, 3, 5, 7, 8, 9]);

        let range: Vec<i32> = s.range(3..=7).copied().collect();
        assert_eq!(range, vec![3, 5, 7]);

        assert_eq!(s.pop_first(), Some(1));
        assert_eq!(s.pop_last(), Some(9));
        assert_eq!(s.first(), Some(&2));
        assert_eq!(s.last(), Some(&8));
    }

    #[test]
    fn btree_set() {
        test_sorted_basics::<std::collections::BTreeSet<i32>>();
    }

    #[test]
    fn flat_btree_set() {
        test_sorted_basics::<optimap::FlatBTreeSet<i32>>();
    }
}
