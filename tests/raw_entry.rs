//! Integration tests for the `raw_entry` / `raw_entry_mut` API.
//!
//! Each scenario runs against all five `GenericMap`-backed hash maps so the
//! shared surface stays in lockstep.

use std::hash::{BuildHasher, BuildHasherDefault, Hasher};

use optimap::raw_entry::RawEntryMut;
use optimap::{Gaps, IPO64, InPlaceOverflow, Splitsies, UnorderedFlatMap};

fn hash_one<K: std::hash::Hash, S: BuildHasher>(k: &K, s: &S) -> u64 {
    s.hash_one(k)
}

/// Run `$body` once per backend. `$map` names the binding (`&mut M`) in scope
/// inside the block.
macro_rules! for_each_map {
    ($map:ident, $body:block) => {{
        {
            let $map = &mut UnorderedFlatMap::<String, i32>::new();
            $body
        }
        {
            let $map = &mut Splitsies::<String, i32>::new();
            $body
        }
        {
            let $map = &mut InPlaceOverflow::<String, i32>::new();
            $body
        }
        {
            let $map = &mut IPO64::<String, i32>::new();
            $body
        }
        {
            let $map = &mut Gaps::<String, i32>::new();
            $body
        }
    }};
}

#[test]
fn from_key_read_hit_and_miss() {
    for_each_map!(map, {
        map.insert("apple".to_string(), 1);
        map.insert("banana".to_string(), 2);

        let got = map.raw_entry().from_key("apple");
        assert_eq!(got.map(|(k, v)| (k.as_str(), *v)), Some(("apple", 1)));

        assert!(map.raw_entry().from_key("cherry").is_none());
    });
}

#[test]
fn from_key_hashed_nocheck_read() {
    for_each_map!(map, {
        // Pre-populate so the table grows past a single group — that way the
        // wrong-hash probe below has a real chance to land elsewhere.
        for i in 0..256u32 {
            map.insert(format!("k{i}"), i as i32);
        }
        map.insert("hello".to_string(), 42);
        let h = hash_one(&"hello".to_string(), map.hasher());

        let got = map.raw_entry().from_key_hashed_nocheck(h, "hello");
        assert_eq!(got.map(|(_, v)| *v), Some(42));

        // Wrong hash → spurious miss is the documented contract once the
        // table is large enough that the wrong hash lands in a different
        // group than the real key.
        let miss = map
            .raw_entry()
            .from_key_hashed_nocheck(h ^ 0xDEAD_BEEF, "hello");
        assert!(miss.is_none(), "wrong hash must miss on a multi-group table");
    });
}

#[test]
fn from_hash_with_custom_eq() {
    for_each_map!(map, {
        map.insert("alpha".to_string(), 10);
        map.insert("beta".to_string(), 20);

        // Same hash the map uses, custom eq matches by first byte.
        let probe = "alpha".to_string();
        let h = hash_one(&probe, map.hasher());
        let got = map
            .raw_entry()
            .from_hash(h, |stored| stored.as_bytes().first() == Some(&b'a'));
        assert_eq!(got.map(|(_, v)| *v), Some(10));
    });
}

#[test]
fn mut_occupied_get_and_replace() {
    for_each_map!(map, {
        map.insert("k".to_string(), 1);

        match map.raw_entry_mut().from_key("k") {
            RawEntryMut::Occupied(mut e) => {
                assert_eq!(e.key(), "k");
                assert_eq!(*e.get(), 1);
                assert_eq!(e.insert(99), 1);
                assert_eq!(*e.get(), 99);
            }
            RawEntryMut::Vacant(_) => panic!("expected Occupied"),
        }

        assert_eq!(map.get("k"), Some(&99));
    });
}

#[test]
fn mut_vacant_insert() {
    for_each_map!(map, {
        match map.raw_entry_mut().from_key("new") {
            RawEntryMut::Vacant(e) => {
                let (k, v) = e.insert("new".to_string(), 7);
                assert_eq!(k.as_str(), "new");
                assert_eq!(*v, 7);
            }
            RawEntryMut::Occupied(_) => panic!("expected Vacant"),
        }

        assert_eq!(map.get("new"), Some(&7));
    });
}

#[test]
fn mut_vacant_insert_hashed_nocheck() {
    for_each_map!(map, {
        let key = "precomputed".to_string();
        let h = hash_one(&key, map.hasher());

        match map.raw_entry_mut().from_key_hashed_nocheck(h, &key) {
            RawEntryMut::Vacant(e) => {
                let (_, v) = e.insert_hashed_nocheck(h, key.clone(), 123);
                assert_eq!(*v, 123);
            }
            RawEntryMut::Occupied(_) => panic!("expected Vacant"),
        }

        // The map's normal get-path must find the entry — confirms the
        // user-supplied hash matched what the map's hasher would compute.
        assert_eq!(map.get(key.as_str()), Some(&123));
    });
}

#[test]
fn mut_occupied_remove_entry() {
    for_each_map!(map, {
        map.insert("doomed".to_string(), 1);
        map.insert("survivor".to_string(), 2);

        let removed = match map.raw_entry_mut().from_key("doomed") {
            RawEntryMut::Occupied(e) => Some(e.remove_entry()),
            RawEntryMut::Vacant(_) => None,
        };
        assert_eq!(removed, Some(("doomed".to_string(), 1)));

        assert_eq!(map.len(), 1);
        assert_eq!(map.get("doomed"), None);
        assert_eq!(map.get("survivor"), Some(&2));
    });
}

#[test]
fn or_insert_branches() {
    for_each_map!(map, {
        let (_, v) = map.raw_entry_mut().from_key("a").or_insert("a".to_string(), 1);
        assert_eq!(*v, 1);
        let (_, v) = map.raw_entry_mut().from_key("a").or_insert("a".to_string(), 999);
        assert_eq!(*v, 1, "or_insert must not overwrite occupied");
    });
}

#[test]
fn and_modify_then_or_insert() {
    for_each_map!(map, {
        // Vacant → and_modify is a no-op → or_insert runs the default.
        let (_, v) = map
            .raw_entry_mut()
            .from_key("counter")
            .and_modify(|_, n| *n += 100)
            .or_insert("counter".to_string(), 0);
        assert_eq!(*v, 0);

        // Now Occupied → and_modify bumps the stored value.
        let (_, v) = map
            .raw_entry_mut()
            .from_key("counter")
            .and_modify(|_, n| *n += 100)
            .or_insert("counter".to_string(), 999);
        assert_eq!(*v, 100);
    });
}

#[test]
fn insert_key_replaces_in_place() {
    for_each_map!(map, {
        // Replace with a logically-equal key that has different capacity.
        // String's Hash + Eq ignore capacity, so the table stays consistent.
        map.insert("orig".to_string(), 1);
        if let RawEntryMut::Occupied(mut e) = map.raw_entry_mut().from_key("orig") {
            let mut tight = String::from("orig");
            tight.shrink_to_fit();
            assert_eq!(e.insert_key(tight), "orig");
        } else {
            panic!("expected Occupied");
        }
        assert_eq!(map.get("orig"), Some(&1));
    });
}

#[test]
fn insert_into_empty_table() {
    for_each_map!(map, {
        match map.raw_entry_mut().from_key("first") {
            RawEntryMut::Vacant(e) => {
                e.insert("first".to_string(), 7);
            }
            RawEntryMut::Occupied(_) => panic!("empty map must yield Vacant"),
        }
        assert_eq!(map.len(), 1);
        assert_eq!(map.get("first"), Some(&7));
    });
}

#[test]
fn interning_pattern() {
    for_each_map!(map, {
        // The headline use case: lookup by a non-owned form (&str), only
        // allocate an owned String when the key is absent.
        let q = "interned";
        let h = hash_one(&q.to_string(), map.hasher());
        let (k, v) = match map.raw_entry_mut().from_key_hashed_nocheck(h, q) {
            RawEntryMut::Occupied(e) => e.into_key_value(),
            RawEntryMut::Vacant(e) => e.insert_hashed_nocheck(h, q.to_string(), 1),
        };
        assert_eq!(k, q);
        assert_eq!(*v, 1);

        match map.raw_entry().from_key_hashed_nocheck(h, q) {
            Some((k, v)) => {
                assert_eq!(k, q);
                assert_eq!(*v, 1);
            }
            None => panic!("second lookup must hit"),
        }
    });
}

// Deterministic hasher so the drop-safety test below is stable.
#[derive(Default)]
struct FixedHasher(u64);
impl Hasher for FixedHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.wrapping_mul(0x100000001b3).wrapping_add(b as u64);
        }
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

#[test]
fn remove_entry_drops_exactly_once() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static DROPS: AtomicUsize = AtomicUsize::new(0);
    struct Trace(u32);
    impl Drop for Trace {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }

    DROPS.store(0, Ordering::SeqCst);
    let mut map: Splitsies<u32, Trace, BuildHasherDefault<FixedHasher>> =
        Splitsies::with_hasher(BuildHasherDefault::default());
    map.insert(1, Trace(10));
    map.insert(2, Trace(20));

    if let RawEntryMut::Occupied(e) = map.raw_entry_mut().from_key(&1) {
        let (k, v) = e.remove_entry();
        assert_eq!(k, 1);
        assert_eq!(v.0, 10);
        // v drops at end of scope → +1 drop.
    }
    assert_eq!(DROPS.load(Ordering::SeqCst), 1, "removed value dropped once");

    drop(map); // remaining entry drops → +1 drop.
    assert_eq!(DROPS.load(Ordering::SeqCst), 2, "remaining value dropped on map drop");
}
