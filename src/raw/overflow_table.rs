//! Generic overflow-bit raw hash table.
//!
//! `RawTable<K, V, L: GroupLayout>` is the single implementation that replaces
//! the three separate overflow-bit raw tables (UFM, Splitsies, Gaps).
//! The `GroupLayout` trait parameterizes overflow storage, bucket stride, and
//! SIMD mask — all resolved at compile time via monomorphization.

use std::alloc::{self, Layout};
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::ptr;

use super::bitmask::BitMask;
use super::generic_group::EMPTY;
use super::group_layout::{GroupLayout, GroupOps};
use super::hash;
use super::overflow_strategy::OverflowStrategy;
use super::table_api::{EntryProbe, RawTableApi};
use super::tag_strategy::TagStrategy;

/// Maximum load factor (fixed at 7/8 = 0.875).
const MAX_LOAD_FACTOR_NUM: usize = 7;
const MAX_LOAD_FACTOR_DEN: usize = 8;

#[inline(always)]
fn max_load_for_capacity(capacity: usize) -> usize {
    capacity * MAX_LOAD_FACTOR_NUM / MAX_LOAD_FACTOR_DEN
}

/// Static sentinel for empty tables. Sized for the largest layout (32 bytes).
#[repr(align(16))]
struct EmptySentinel([u8; 32]);
static EMPTY_SENTINEL: EmptySentinel = EmptySentinel([0; 32]);

/// The core overflow-bit hash table engine, generic over group layout.
///
/// Replaces the three separate implementations (UFM, Splitsies, Gaps).
/// The `L: GroupLayout` parameter controls:
/// - Overflow storage (embedded byte 15 vs separate array)
/// - Bucket stride (15 vs 16)
/// - SIMD bitmask (0x7FFF vs 0xFFFF)
pub struct RawTable<K, V, L: GroupLayout> {
    pub(crate) mask: usize,
    pub(crate) metadata: *mut u8,
    buckets: *mut u8,
    pub(crate) len: usize,
    pub(crate) max_load: usize,
    pub(crate) shift: u32,
    _marker: PhantomData<(K, V, L)>,
}

impl<K, V, L: GroupLayout> RawTable<K, V, L> {
    /// Map hash to group index. Uses AND (low bits) or shift (high bits) depending
    /// on the layout's AND_INDEX const. The branch is eliminated at compile time.
    #[inline(always)]
    pub(crate) fn group_index(&self, h: u64) -> usize {
        if L::AND_INDEX {
            (h as usize) & self.mask
        } else {
            (h.wrapping_shr(self.shift) as usize) & self.mask
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn meta_ptr(&self, gi: usize) -> *mut u8 {
        unsafe { self.metadata.add(gi * 16) }
    }

    #[inline(always)]
    unsafe fn overflow_ptr(&self, gi: usize) -> *mut u8 {
        unsafe { L::Overflow::overflow_ptr(self.metadata, self.mask, gi) }
    }

    #[inline(always)]
    unsafe fn has_overflow_bit(&self, gi: usize, bit: u8) -> bool {
        unsafe { L::Overflow::has_overflow(self.overflow_ptr(gi), gi, bit) }
    }

    #[inline(always)]
    unsafe fn set_overflow_bit(&self, gi: usize, bit: u8) {
        unsafe { L::Overflow::set_overflow(self.overflow_ptr(gi), gi, bit); }
    }

    #[inline(always)]
    fn bytes_to_copy_total(num_groups: usize) -> usize {
        num_groups * 16 + L::Overflow::overflow_bytes_to_copy(num_groups)
    }

    #[inline(always)]
    fn hash_key<S: BuildHasher>(key: &K, hb: &S) -> u64
    where
        K: Hash,
    {
        hash::hash_no_mix(key, hb)
    }

    fn bucket_offset(num_groups: usize) -> usize {
        let meta_size = num_groups * 16;
        let before_buckets = meta_size + L::Overflow::extra_alloc_bytes(num_groups);
        let bucket_align = std::mem::align_of::<(K, V)>().max(1);
        (before_buckets + bucket_align - 1) & !(bucket_align - 1)
    }

    fn combined_layout(num_groups: usize) -> Layout {
        let bucket_size = std::mem::size_of::<(K, V)>();
        let total_buckets = num_groups * L::BUCKET_STRIDE;
        let bucket_offset = Self::bucket_offset(num_groups);
        let total_size = bucket_offset + total_buckets * bucket_size;
        Layout::from_size_align(total_size.max(16), 16).unwrap()
    }

    pub(crate) fn allocate(&mut self, num_groups: usize) {
        debug_assert!(num_groups.is_power_of_two());
        let layout = Self::combined_layout(num_groups);
        let bucket_offset = Self::bucket_offset(num_groups);
        let total_buckets = num_groups * L::GROUP_SIZE;

        unsafe {
            let ptr = alloc::alloc(layout);
            if ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }
            self.metadata = ptr;
            self.buckets = ptr.add(bucket_offset);
            ptr::write_bytes(self.metadata, 0, num_groups * 16 + L::Overflow::overflow_bytes_to_zero(num_groups));
        }

        self.mask = num_groups - 1;
        self.max_load = max_load_for_capacity(total_buckets);
        self.shift = 64u32.wrapping_sub(num_groups.trailing_zeros());
    }

    unsafe fn deallocate(&mut self) {
        if self.max_load == 0 {
            return;
        }
        let layout = Self::combined_layout(self.mask + 1);
        unsafe { alloc::dealloc(self.metadata, layout); }
        self.metadata = EMPTY_SENTINEL.0.as_ptr() as *mut u8;
        self.buckets = ptr::null_mut();
        self.max_load = 0;
    }

    #[inline(always)]
    pub(crate) unsafe fn bucket_ptr_impl(&self, gi: usize, si: usize) -> *mut (K, V) {
        let bucket_size = std::mem::size_of::<(K, V)>();
        let idx = L::bucket_index(gi, si);
        unsafe { self.buckets.add(idx * bucket_size).cast::<(K, V)>() }
    }

    // ── Core lookup ────────────────────────────────────────────────

    #[inline(always)]
    fn find_by_hash_impl<F>(&self, h: u64, eq: F) -> Option<(usize, usize)>
    where
        F: Fn(&K) -> bool,
    {
        let reduced = L::Tag::tag(h);
        let mut gi = self.group_index(h);
        let ofw_bit = L::Tag::overflow_channel(h);
        let mut probe = 0usize;

        if L::SEPARATE_OVERFLOW {
            unsafe { L::Grp::prefetch_read(self.overflow_ptr(gi) as *const u8); }
        }

        loop {
            let meta = unsafe { self.meta_ptr(gi) };

            for si in unsafe { L::Grp::match_byte(meta, reduced) } {
                let bucket = unsafe { &*self.bucket_ptr_impl(gi, si) };
                if eq(&bucket.0) {
                    return Some((gi, si));
                }
            }

            if !unsafe { self.has_overflow_bit(gi, ofw_bit) } {
                return None;
            }

            probe += 1;
            gi = (gi.wrapping_add(probe)) & self.mask;

            unsafe {
                L::Grp::prefetch_read(self.meta_ptr(gi) as *const u8);
                L::Grp::prefetch_read(self.bucket_ptr_impl(gi, 0) as *const u8);
                if L::SEPARATE_OVERFLOW {
                    L::Grp::prefetch_read(self.overflow_ptr(gi) as *const u8);
                }
            }
        }
    }

    #[inline(always)]
    fn find_bucket_impl<F>(&self, h: u64, eq: F) -> Option<*mut (K, V)>
    where
        F: Fn(&K) -> bool,
    {
        let reduced = L::Tag::tag(h);
        let mut gi = self.group_index(h);
        let ofw_bit = L::Tag::overflow_channel(h);
        let mut probe = 0usize;

        if L::SEPARATE_OVERFLOW {
            unsafe { L::Grp::prefetch_read(self.overflow_ptr(gi) as *const u8); }
        }

        loop {
            let meta = unsafe { self.meta_ptr(gi) };

            for si in unsafe { L::Grp::match_byte(meta, reduced) } {
                let bucket = unsafe { self.bucket_ptr_impl(gi, si) };
                if eq(unsafe { &(*bucket).0 }) {
                    return Some(bucket);
                }
            }

            if !unsafe { self.has_overflow_bit(gi, ofw_bit) } {
                return None;
            }

            probe += 1;
            gi = (gi.wrapping_add(probe)) & self.mask;

            unsafe {
                L::Grp::prefetch_read(self.meta_ptr(gi) as *const u8);
                L::Grp::prefetch_read(self.bucket_ptr_impl(gi, 0) as *const u8);
                if L::SEPARATE_OVERFLOW {
                    L::Grp::prefetch_read(self.overflow_ptr(gi) as *const u8);
                }
            }
        }
    }

    // ── Insert ─────────────────────────────────────────────────────

    #[inline(always)]
    fn insert_no_check_impl(&mut self, h: u64, key: K, value: V) -> (usize, usize) {
        let reduced = L::Tag::tag(h);
        let mut gi = self.group_index(h);
        let mut probe = 0usize;

        loop {
            let meta = unsafe { self.meta_ptr(gi) };

            if let Some(si) = unsafe { L::Grp::match_empty(meta) }.lowest_set_bit() {
                unsafe {
                    L::Grp::set_meta(meta, si, reduced);
                    self.bucket_ptr_impl(gi, si).write((key, value));
                }
                self.len += 1;
                return (gi, si);
            }

            let ofw_bit = L::Tag::overflow_channel(h);
            unsafe { self.set_overflow_bit(gi, ofw_bit); }

            probe += 1;
            gi = (gi.wrapping_add(probe)) & self.mask;
        }
    }

    fn insert_at_impl(&mut self, h: u64, gi: usize, si: usize, key: K, value: V, full_mask: u8) {
        let reduced = L::Tag::tag(h);
        let ofw_bit = L::Tag::overflow_channel(h);

        if full_mask != 0 {
            let home_gi = self.group_index(h);
            let mut set_probe = 0usize;
            let mut set_gi = home_gi;
            let mut mask = full_mask;
            while mask != 0 {
                if mask & 1 != 0 {
                    unsafe { self.set_overflow_bit(set_gi, ofw_bit); }
                }
                mask >>= 1;
                set_probe += 1;
                set_gi = (set_gi.wrapping_add(set_probe)) & self.mask;
            }
        }

        unsafe {
            let meta = self.meta_ptr(gi);
            L::Grp::set_meta(meta, si, reduced);
            self.bucket_ptr_impl(gi, si).write((key, value));
        }
        self.len += 1;
    }

    // ── Find-or-locate ─────────────────────────────────────────────

    #[inline(always)]
    fn find_or_locate_impl<F>(&self, h: u64, eq: F) -> FindOrLocateResult
    where
        F: Fn(&K) -> bool,
    {
        let reduced = L::Tag::tag(h);
        let ofw_bit = L::Tag::overflow_channel(h);
        let gi = self.group_index(h);

        let meta = unsafe { self.meta_ptr(gi) };
        let (matches, empties) = unsafe { L::Grp::match_byte_and_empty(meta, reduced) };

        for si in matches {
            let bucket = unsafe { &*self.bucket_ptr_impl(gi, si) };
            if eq(&bucket.0) {
                return FindOrLocateResult::Found(gi, si);
            }
        }

        if let Some(si) = empties.lowest_set_bit() {
            if !unsafe { self.has_overflow_bit(gi, ofw_bit) } {
                return FindOrLocateResult::InsertSlot(gi, si, 0);
            }
            return self.find_or_locate_overflow(h, eq, reduced, ofw_bit, gi, Some((gi, si)), 0);
        }

        if !unsafe { self.has_overflow_bit(gi, ofw_bit) } {
            return FindOrLocateResult::NotFound;
        }

        self.find_or_locate_overflow(h, eq, reduced, ofw_bit, gi, None, 1)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn find_or_locate_overflow<F>(
        &self,
        _h: u64,
        eq: F,
        reduced: u8,
        ofw_bit: u8,
        home_gi: usize,
        mut first_empty: Option<(usize, usize)>,
        mut full_mask: u8,
    ) -> FindOrLocateResult
    where
        F: Fn(&K) -> bool,
    {
        let mut probe = 1usize;
        let mut gi = (home_gi.wrapping_add(probe)) & self.mask;

        loop {
            let meta = unsafe { self.meta_ptr(gi) };
            let (matches, empties) = unsafe { L::Grp::match_byte_and_empty(meta, reduced) };

            for si in matches {
                let bucket = unsafe { &*self.bucket_ptr_impl(gi, si) };
                if eq(&bucket.0) {
                    return FindOrLocateResult::Found(gi, si);
                }
            }

            if first_empty.is_none() {
                if let Some(si) = empties.lowest_set_bit() {
                    first_empty = Some((gi, si));
                } else if probe < 8 {
                    full_mask |= 1 << probe;
                }
            }

            if !unsafe { self.has_overflow_bit(gi, ofw_bit) } {
                return match first_empty {
                    Some((ins_gi, ins_si)) => FindOrLocateResult::InsertSlot(ins_gi, ins_si, full_mask),
                    None => FindOrLocateResult::NotFound,
                };
            }

            probe += 1;
            gi = (gi.wrapping_add(probe)) & self.mask;

            unsafe {
                L::Grp::prefetch_read(self.meta_ptr(gi) as *const u8);
                L::Grp::prefetch_read(self.bucket_ptr_impl(gi, 0) as *const u8);
                if L::SEPARATE_OVERFLOW {
                    L::Grp::prefetch_read(self.overflow_ptr(gi) as *const u8);
                }
            }
        }
    }

    fn rehash_with_impl<S: BuildHasher>(&mut self, new_num_groups: usize, hash_builder: &S)
    where
        K: Hash,
    {
        let was_allocated = self.max_load > 0;
        let old_num_groups = self.mask + 1;
        let old_metadata = self.metadata;
        let old_buckets = self.buckets;
        let old_layout = if was_allocated {
            Some(Self::combined_layout(old_num_groups))
        } else {
            None
        };

        let bucket_size = std::mem::size_of::<(K, V)>();

        self.metadata = EMPTY_SENTINEL.0.as_ptr() as *mut u8;
        self.buckets = ptr::null_mut();
        self.mask = 0;
        self.len = 0;
        self.max_load = 0;
        self.allocate(new_num_groups);

        if !was_allocated {
            return;
        }

        unsafe {
            for gi in 0..old_num_groups {
                let group_meta = old_metadata.add(gi * 16);
                for si in L::Grp::match_non_empty(group_meta) {
                    let old_bucket = old_buckets
                        .add(L::bucket_index(gi, si) * bucket_size)
                        .cast::<(K, V)>();
                    let (key, value) = old_bucket.read();
                    let h = Self::hash_key(&key, hash_builder);
                    self.insert_no_check_impl(h, key, value);
                }
            }
            alloc::dealloc(old_metadata, old_layout.unwrap());
        }
    }

    fn first_non_empty_mask(&self) -> BitMask {
        if self.max_load == 0 {
            BitMask(0)
        } else {
            unsafe { L::Grp::match_non_empty(self.metadata) }
        }
    }
}

// ── Internal enum ──────────────────────────────────────────────────────────

enum FindOrLocateResult {
    Found(usize, usize),
    InsertSlot(usize, usize, u8),
    NotFound,
}

// ── Cold insert paths ──────────────────────────────────────────────────────

impl<K: Hash + Eq, V, L: GroupLayout> RawTable<K, V, L> {
    #[cold]
    #[inline(never)]
    fn insert_overflow<S: BuildHasher>(
        &mut self,
        h: u64,
        key: K,
        value: V,
        hb: &S,
    ) -> Option<V> {
        if let Some((gi, si)) = self.find_by_hash_impl(h, |k| k == &key) {
            let bucket = unsafe { &mut *self.bucket_ptr_impl(gi, si) };
            return Some(std::mem::replace(&mut bucket.1, value));
        }
        if self.len >= self.max_load {
            self.grow_and_rehash(hb);
        }
        self.insert_no_check_impl(h, key, value);
        None
    }

    #[cold]
    #[inline(never)]
    fn insert_at_capacity<S: BuildHasher>(
        &mut self,
        h: u64,
        key: K,
        value: V,
        hb: &S,
    ) -> Option<V> {
        if let Some((gi, si)) = self.find_by_hash_impl(h, |k| k == &key) {
            let bucket = unsafe { &mut *self.bucket_ptr_impl(gi, si) };
            return Some(std::mem::replace(&mut bucket.1, value));
        }
        self.grow_and_rehash(hb);
        self.insert_no_check_impl(h, key, value);
        None
    }

    #[cold]
    #[inline(never)]
    fn grow_and_rehash<S: BuildHasher>(&mut self, hb: &S) {
        let new_groups = if self.max_load == 0 { 1 } else { (self.mask + 1) * 2 };
        self.rehash_with_impl(new_groups, hb);
    }
}

// ── RawTableApi implementation ─────────────────────────────────────────────

impl<K, V, L: GroupLayout> RawTableApi<K, V> for RawTable<K, V, L> {
    type SlotIter<'a> = SlotIter<'a, K, V, L> where K: 'a, V: 'a;
    type IntoIter = IntoIter<K, V, L>;

    fn new() -> Self {
        RawTable {
            mask: 0,
            metadata: EMPTY_SENTINEL.0.as_ptr() as *mut u8,
            buckets: ptr::null_mut(),
            len: 0,
            max_load: 0,
            shift: 64,
            _marker: PhantomData,
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 { return Self::new(); }
        let mut table = Self::new();
        let num_groups = Self::groups_for_capacity(capacity);
        table.allocate(num_groups);
        table
    }

    #[inline(always)]
    fn len(&self) -> usize { self.len }

    #[inline(always)]
    fn capacity(&self) -> usize {
        if self.max_load > 0 { (self.mask + 1) * L::GROUP_SIZE } else { 0 }
    }

    #[inline(always)]
    fn is_allocated(&self) -> bool { self.max_load > 0 }

    #[inline(always)]
    fn num_groups(&self) -> usize { self.mask + 1 }

    fn groups_for_capacity(capacity: usize) -> usize {
        let min_slots = (capacity * MAX_LOAD_FACTOR_DEN + MAX_LOAD_FACTOR_NUM - 1) / MAX_LOAD_FACTOR_NUM;
        let min_groups = (min_slots + L::GROUP_SIZE - 1) / L::GROUP_SIZE;
        min_groups.next_power_of_two()
    }

    fn clear(&mut self) {
        if self.max_load == 0 { return; }
        unsafe {
            if std::mem::needs_drop::<(K, V)>() {
                for gi in 0..self.mask + 1 {
                    let group_meta = self.metadata.add(gi * 16);
                    for si in L::Grp::match_non_empty(group_meta) {
                        ptr::drop_in_place(self.bucket_ptr_impl(gi, si));
                    }
                }
            }
            ptr::write_bytes(self.metadata, 0, (self.mask + 1) * 16 + L::Overflow::overflow_bytes_to_zero(self.mask + 1));
        }
        self.len = 0;
        self.max_load = max_load_for_capacity((self.mask + 1) * L::GROUP_SIZE);
    }

    #[inline(always)]
    unsafe fn bucket_ptr(&self, gi: usize, si: usize) -> *mut (K, V) {
        unsafe { self.bucket_ptr_impl(gi, si) }
    }

    #[inline(always)]
    fn find_bucket<F: Fn(&K) -> bool>(&self, h: u64, eq: F) -> Option<*mut (K, V)> {
        self.find_bucket_impl(h, eq)
    }

    #[inline(always)]
    fn find_by_hash<F: Fn(&K) -> bool>(&self, h: u64, eq: F) -> Option<(usize, usize)> {
        self.find_by_hash_impl(h, eq)
    }

    fn insert_or_replace<S: BuildHasher>(&mut self, key: K, value: V, hb: &S) -> Option<V>
    where
        K: Hash + Eq,
    {
        if self.max_load == 0 {
            self.allocate(1);
        }

        let h = Self::hash_key(&key, hb);

        if self.len >= self.max_load {
            return self.insert_at_capacity(h, key, value, hb);
        }

        let reduced = L::Tag::tag(h);
        let gi = self.group_index(h);
        let meta = unsafe { self.meta_ptr(gi) };
        let (matches, empties) = unsafe { L::Grp::match_byte_and_empty(meta, reduced) };

        for si in matches {
            let bucket = unsafe { &mut *self.bucket_ptr_impl(gi, si) };
            if bucket.0 == key {
                return Some(std::mem::replace(&mut bucket.1, value));
            }
        }

        let ofw_bit = L::Tag::overflow_channel(h);
        if let Some(si) = empties.lowest_set_bit()
            && !unsafe { self.has_overflow_bit(gi, ofw_bit) }
        {
            unsafe {
                L::Grp::set_meta(meta, si, reduced);
                self.bucket_ptr_impl(gi, si).write((key, value));
            }
            self.len += 1;
            return None;
        }

        self.insert_overflow(h, key, value, hb)
    }

    fn find_for_entry(&self, h: u64, key: &K) -> EntryProbe
    where
        K: Eq,
    {
        if self.len >= self.max_load {
            if let Some((gi, si)) = self.find_by_hash_impl(h, |k| k == key) {
                return EntryProbe::Found(gi, si);
            }
            return EntryProbe::Vacant(None);
        }

        let reduced = L::Tag::tag(h);
        let gi = self.group_index(h);
        let meta = unsafe { self.meta_ptr(gi) };
        let (matches, empties) = unsafe { L::Grp::match_byte_and_empty(meta, reduced) };

        for si in matches {
            let bucket = unsafe { &*self.bucket_ptr_impl(gi, si) };
            if bucket.0 == *key {
                return EntryProbe::Found(gi, si);
            }
        }

        let ofw_bit = L::Tag::overflow_channel(h);
        if let Some(si) = empties.lowest_set_bit()
            && !unsafe { self.has_overflow_bit(gi, ofw_bit) }
        {
            return EntryProbe::Vacant(Some((gi, si, 0)));
        }

        match self.find_or_locate_impl(h, |k| k == key) {
            FindOrLocateResult::Found(gi, si) => EntryProbe::Found(gi, si),
            FindOrLocateResult::InsertSlot(gi, si, mask) => EntryProbe::Vacant(Some((gi, si, mask))),
            FindOrLocateResult::NotFound => EntryProbe::Vacant(None),
        }
    }

    #[inline(always)]
    fn insert_at(&mut self, h: u64, gi: usize, si: usize, k: K, v: V, mask: u8) {
        self.insert_at_impl(h, gi, si, k, v, mask);
    }

    #[inline(always)]
    fn insert_no_check(&mut self, h: u64, k: K, v: V) -> (usize, usize) {
        self.insert_no_check_impl(h, k, v)
    }

    fn ensure_capacity<S: BuildHasher>(&mut self, hb: &S) where K: Hash {
        if self.len >= self.max_load {
            let new_groups = if self.max_load == 0 { 1 } else { (self.mask + 1) * 2 };
            self.rehash_with_impl(new_groups, hb);
        }
    }

    fn remove_by_hash<F: Fn(&K) -> bool>(&mut self, h: u64, eq: F) -> Option<(K, V)> {
        let (gi, si) = self.find_by_hash_impl(h, eq)?;
        unsafe {
            let bucket = self.bucket_ptr_impl(gi, si).read();
            let meta = self.meta_ptr(gi);
            L::Grp::set_meta(meta, si, EMPTY);
            self.len -= 1;

            let initial_gi = self.group_index(h);
            let ofw_bit = L::Tag::overflow_channel(h);
            if self.has_overflow_bit(initial_gi, ofw_bit) {
                self.max_load = self.max_load.saturating_sub(1);
            }
            Some(bucket)
        }
    }

    unsafe fn erase_slot(&mut self, h: u64, gi: usize, si: usize) {
        unsafe {
            let meta = self.meta_ptr(gi);
            L::Grp::set_meta(meta, si, EMPTY);
            self.len -= 1;

            let initial_gi = self.group_index(h);
            let ofw_bit = L::Tag::overflow_channel(h);
            if self.has_overflow_bit(initial_gi, ofw_bit) {
                self.max_load = self.max_load.saturating_sub(1);
            }
        }
    }

    fn reserve<S: BuildHasher>(&mut self, additional: usize, hb: &S) where K: Hash {
        let needed = self.len.checked_add(additional).expect("capacity overflow");
        if self.max_load == 0 {
            if additional > 0 {
                self.allocate(Self::groups_for_capacity(needed));
            }
            return;
        }
        if needed > self.max_load {
            let new_groups = Self::groups_for_capacity(needed);
            if new_groups > self.mask + 1 {
                self.rehash_with_impl(new_groups, hb);
            }
        }
    }

    fn shrink_to_fit<S: BuildHasher>(&mut self, hb: &S) where K: Hash {
        if self.len == 0 {
            let mut empty = Self::new();
            std::mem::swap(self, &mut empty);
            return;
        }
        let min_groups = Self::groups_for_capacity(self.len);
        if min_groups < self.mask + 1 {
            self.rehash_with_impl(min_groups, hb);
        }
    }

    fn iter_slots(&self) -> SlotIter<'_, K, V, L> {
        SlotIter {
            table: self,
            group: 0,
            current_mask: self.first_non_empty_mask(),
        }
    }

    fn into_iter_impl(self) -> IntoIter<K, V, L> {
        let mask = self.first_non_empty_mask();
        let table = unsafe { ptr::read(&self) };
        std::mem::forget(self);
        IntoIter { table, group: 0, current_mask: mask }
    }

    fn drain_impl(&mut self) -> IntoIter<K, V, L> {
        let table = std::mem::replace(self, Self::new());
        table.into_iter_impl()
    }

    fn rehash_with<S: BuildHasher>(&mut self, new_num_groups: usize, hb: &S) where K: Hash {
        self.rehash_with_impl(new_num_groups, hb);
    }

    fn clone_table(&self) -> Self where K: Clone, V: Clone {
        if self.max_load == 0 {
            return Self::new();
        }

        let mut new_table = Self::new();
        new_table.allocate(self.mask + 1);

        unsafe {
            let copy_size = Self::bytes_to_copy_total(self.mask + 1);
            ptr::copy_nonoverlapping(self.metadata, new_table.metadata, copy_size);

            let bucket_size = std::mem::size_of::<(K, V)>();
            if bucket_size > 0 {
                for gi in 0..self.mask + 1 {
                    let group_meta = self.metadata.add(gi * 16);
                    for si in L::Grp::match_non_empty(group_meta) {
                        let src = &*self.bucket_ptr_impl(gi, si);
                        new_table.bucket_ptr_impl(gi, si).write(src.clone());
                    }
                }
            }
        }

        new_table.len = self.len;
        new_table.max_load = self.max_load;
        new_table
    }
}

// ── Drop ───────────────────────────────────────────────────────────────────

impl<K, V, L: GroupLayout> Drop for RawTable<K, V, L> {
    fn drop(&mut self) {
        if self.max_load == 0 { return; }
        if std::mem::needs_drop::<(K, V)>() {
            unsafe {
                for gi in 0..self.mask + 1 {
                    let group_meta = self.metadata.add(gi * 16);
                    for si in L::Grp::match_non_empty(group_meta) {
                        ptr::drop_in_place(self.bucket_ptr_impl(gi, si));
                    }
                }
            }
        }
        unsafe { self.deallocate(); }
    }
}

unsafe impl<K: Send, V: Send, L: GroupLayout> Send for RawTable<K, V, L> {}
unsafe impl<K: Sync, V: Sync, L: GroupLayout> Sync for RawTable<K, V, L> {}

// ── SlotIter ───────────────────────────────────────────────────────────────

pub struct SlotIter<'a, K, V, L: GroupLayout> {
    pub(crate) table: &'a RawTable<K, V, L>,
    group: usize,
    current_mask: BitMask,
}

impl<'a, K, V, L: GroupLayout> Iterator for SlotIter<'a, K, V, L> {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<(usize, usize)> {
        loop {
            if let Some(si) = self.current_mask.next() {
                return Some((self.group, si));
            }
            self.group += 1;
            if self.group > self.table.mask {
                return None;
            }
            self.current_mask = unsafe { L::Grp::match_non_empty(self.table.meta_ptr(self.group)) };
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.table.len))
    }
}

// ── IntoIter ───────────────────────────────────────────────────────────────

pub struct IntoIter<K, V, L: GroupLayout> {
    table: RawTable<K, V, L>,
    group: usize,
    current_mask: BitMask,
}

impl<K, V, L: GroupLayout> Iterator for IntoIter<K, V, L> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(si) = self.current_mask.next() {
                let gi = self.group;
                unsafe {
                    let ptr = self.table.bucket_ptr_impl(gi, si);
                    let kv = ptr.read();
                    let meta = self.table.metadata.add(gi * 16 + si);
                    *meta = EMPTY;
                    self.table.len -= 1;
                    return Some(kv);
                }
            }
            self.group += 1;
            if self.group > self.table.mask {
                return None;
            }
            self.current_mask = unsafe { L::Grp::match_non_empty(self.table.metadata.add(self.group * 16)) };
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.table.len, Some(self.table.len))
    }
}

impl<K, V, L: GroupLayout> ExactSizeIterator for IntoIter<K, V, L> {}
impl<K, V, L: GroupLayout> std::iter::FusedIterator for IntoIter<K, V, L> {}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::group_layout::{GapsLayout, SplitsiesLayout, UfmLayout};
    use std::hash::RandomState;

    fn test_basic<L: GroupLayout>() {
        let hb = RandomState::new();
        let mut table: RawTable<u64, u64, L> = RawTable::new();
        assert!(table.is_empty());
        assert_eq!(table.capacity(), 0);

        assert_eq!(table.insert_or_replace(42, 100, &hb), None);
        assert_eq!(table.len(), 1);

        let h = hash::hash_no_mix(&42u64, &hb);
        let bucket = table.find_bucket(h, |k| *k == 42).unwrap();
        assert_eq!(unsafe { &*bucket }, &(42, 100));

        assert_eq!(table.insert_or_replace(42, 200, &hb), Some(100));
        assert_eq!(table.len(), 1);

        let h = hash::hash_no_mix(&42u64, &hb);
        let kv = table.remove_by_hash(h, |k| *k == 42);
        assert_eq!(kv, Some((42, 200)));
        assert!(table.is_empty());
    }

    fn test_grow<L: GroupLayout>() {
        let hb = RandomState::new();
        let mut table: RawTable<u64, u64, L> = RawTable::new();
        for i in 0..200 {
            table.insert_or_replace(i, i * 10, &hb);
        }
        assert_eq!(table.len(), 200);
        for i in 0..200 {
            let h = hash::hash_no_mix(&i, &hb);
            let bucket = table.find_bucket(h, |k| *k == i).unwrap();
            assert_eq!(unsafe { (*bucket).1 }, i * 10);
        }
    }

    fn test_clone<L: GroupLayout>() {
        let hb = RandomState::new();
        let mut table: RawTable<u64, u64, L> = RawTable::new();
        for i in 0..50 {
            table.insert_or_replace(i, i * 10, &hb);
        }
        let cloned = table.clone_table();
        assert_eq!(cloned.len(), 50);
        for i in 0..50 {
            let h = hash::hash_no_mix(&i, &hb);
            assert!(cloned.find_bucket(h, |k| *k == i).is_some());
        }
    }

    fn test_into_iter<L: GroupLayout>() {
        let hb = RandomState::new();
        let mut table: RawTable<u64, u64, L> = RawTable::new();
        for i in 0..50 {
            table.insert_or_replace(i, i * 10, &hb);
        }
        let items: Vec<_> = table.into_iter_impl().collect();
        assert_eq!(items.len(), 50);
    }

    // Existing layouts
    #[test] fn ufm_basic() { test_basic::<UfmLayout>(); }
    #[test] fn splitsies_basic() { test_basic::<SplitsiesLayout>(); }
    #[test] fn gaps_basic() { test_basic::<GapsLayout>(); }
    #[test] fn ufm_grow() { test_grow::<UfmLayout>(); }
    #[test] fn splitsies_grow() { test_grow::<SplitsiesLayout>(); }
    #[test] fn gaps_grow() { test_grow::<GapsLayout>(); }
    #[test] fn ufm_clone() { test_clone::<UfmLayout>(); }
    #[test] fn splitsies_clone() { test_clone::<SplitsiesLayout>(); }
    #[test] fn gaps_clone() { test_clone::<GapsLayout>(); }
    #[test] fn ufm_into_iter() { test_into_iter::<UfmLayout>(); }
    #[test] fn splitsies_into_iter() { test_into_iter::<SplitsiesLayout>(); }
    #[test] fn gaps_into_iter() { test_into_iter::<GapsLayout>(); }

    // Matrix entries
    use crate::raw::group_layout::{Hi8_8bit, Hi8_1bit, Lo128_8bit, Lo128_1bit, Lo8_1bit};

    #[test] fn hi8_8bit_basic() { test_basic::<Hi8_8bit>(); }
    #[test] fn hi8_8bit_grow() { test_grow::<Hi8_8bit>(); }
    #[test] fn hi8_1bit_basic() { test_basic::<Hi8_1bit>(); }
    #[test] fn hi8_1bit_grow() { test_grow::<Hi8_1bit>(); }
    #[test] fn lo128_8bit_basic() { test_basic::<Lo128_8bit>(); }
    #[test] fn lo128_8bit_grow() { test_grow::<Lo128_8bit>(); }
    #[test] fn lo128_1bit_basic() { test_basic::<Lo128_1bit>(); }
    #[test] fn lo128_1bit_grow() { test_grow::<Lo128_1bit>(); }
    #[test] fn lo8_1bit_basic() { test_basic::<Lo8_1bit>(); }
    #[test] fn lo8_1bit_grow() { test_grow::<Lo8_1bit>(); }
}
