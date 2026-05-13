//! Allocator-instrumented stress tests.
//!
//! Wraps `System` with a counting / alignment-asserting allocator and exercises
//! every map design through a full insert / mutate / remove / drop cycle,
//! verifying that:
//!
//! - allocator-returned pointers honour the requested alignment,
//! - the `Layout` passed back to `dealloc` matches the one used at `alloc`
//!   time (re-checked by `System` internally — we just forward),
//! - dropping a map releases every byte it allocated (no leaks),
//! - oversized capacity hints still result in symmetric alloc/free pairs.
//!
//! Everything runs from a single `#[test]` so the harness has no parallel
//! threads making the live-alloc counter drift while a scenario is in flight.
//! Each scenario is encoded as its own helper function; on failure the panic
//! message names the scenario.

#![cfg(not(miri))] // miri has its own UB / leak tracker — this would just slow it down.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use optimap::{Gaps, IPO64, InPlaceOverflow, Splitsies, UnorderedFlatMap};

// ── Global tracking allocator ──────────────────────────────────────────────

#[global_allocator]
static TRACING: TracingAllocator = TracingAllocator;

struct TracingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static MAX_ALIGN_SEEN: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TracingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        assert!(
            layout.align().is_power_of_two(),
            "non-pow2 align requested: {}",
            layout.align()
        );
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            assert_eq!(
                (p as usize) % layout.align(),
                0,
                "alloc returned misaligned ptr: align={}, ptr=0x{:x}",
                layout.align(),
                p as usize
            );
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            MAX_ALIGN_SEEN.fetch_max(layout.align(), Ordering::Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Belt-and-braces: any misaligned ptr handed back would imply a logic
        // bug in our bucket-layout math, not in the system allocator.
        assert_eq!(
            (ptr as usize) % layout.align(),
            0,
            "dealloc misaligned ptr: align={}, ptr=0x{:x}",
            layout.align(),
            ptr as usize
        );
        unsafe { System.dealloc(ptr, layout) };
        DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        DEALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { self.alloc(layout) };
        if !p.is_null() {
            unsafe { std::ptr::write_bytes(p, 0, layout.size()) };
        }
        p
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_layout = match Layout::from_size_align(new_size, layout.align()) {
            Ok(l) => l,
            Err(_) => return std::ptr::null_mut(),
        };
        let new_ptr = unsafe { self.alloc(new_layout) };
        if !new_ptr.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, new_ptr, layout.size().min(new_size));
                self.dealloc(ptr, layout);
            }
        }
        new_ptr
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AllocSnapshot {
    count: usize,
    bytes: usize,
}

impl AllocSnapshot {
    fn now() -> Self {
        Self {
            count: ALLOC_COUNT
                .load(Ordering::Relaxed)
                .wrapping_sub(DEALLOC_COUNT.load(Ordering::Relaxed)),
            bytes: ALLOC_BYTES
                .load(Ordering::Relaxed)
                .wrapping_sub(DEALLOC_BYTES.load(Ordering::Relaxed)),
        }
    }
}

/// Run `f` and assert that its live-allocation snapshot returns to baseline.
fn assert_no_leak(label: &str, f: impl FnOnce()) {
    let before = AllocSnapshot::now();
    f();
    let after = AllocSnapshot::now();
    assert_eq!(before, after, "{label}: alloc drift {before:?} -> {after:?}");
}

// ── Shared exercise bodies ─────────────────────────────────────────────────

trait Cap: Default {
    fn ins(&mut self, k: String, v: Vec<u32>);
    fn rm(&mut self, k: &str);
    fn cap(capacity: usize) -> Self;
    fn rsv(&mut self, additional: usize);
    fn shrink(&mut self);
    fn ln(&self) -> usize;
}

macro_rules! impl_cap {
    ($t:ident) => {
        impl Cap for $t<String, Vec<u32>> {
            fn ins(&mut self, k: String, v: Vec<u32>) { self.insert(k, v); }
            fn rm(&mut self, k: &str) { self.remove(k); }
            fn cap(capacity: usize) -> Self { Self::with_capacity(capacity) }
            fn rsv(&mut self, additional: usize) { self.reserve(additional); }
            fn shrink(&mut self) { self.shrink_to_fit(); }
            fn ln(&self) -> usize { self.len() }
        }
    };
}
impl_cap!(UnorderedFlatMap);
impl_cap!(Splitsies);
impl_cap!(InPlaceOverflow);
impl_cap!(IPO64);
impl_cap!(Gaps);

fn lifecycle<M: Cap>(label: &str) {
    assert_no_leak(label, || {
        let mut map = M::default();
        for i in 0..2_048u32 {
            map.ins(format!("k{i:06}"), (0..(i % 16)).collect());
        }
        for i in 0..1_024u32 {
            map.rm(&format!("k{i:06}"));
        }
        map.rsv(4_096);
        map.shrink();
        assert_eq!(map.ln(), 1_024);
    });
}

fn cap_no_inserts<M: Cap>(label: &str) {
    assert_no_leak(label, || {
        let map = M::cap(10_000);
        assert_eq!(map.ln(), 0);
    });
}

fn cap_partial_fill<M: Cap>(label: &str) {
    assert_no_leak(label, || {
        let mut map = M::cap(1_000);
        for i in 0..100u32 {
            map.ins(format!("v{i}"), vec![i; 4]);
        }
    });
}

fn churn_cycles<M: Cap>(label: &str) {
    assert_no_leak(label, || {
        let mut map = M::default();
        for cycle in 0..8u32 {
            for i in 0..256u32 {
                map.ins(format!("c{cycle}-k{i}"), vec![i; (cycle as usize % 4) + 1]);
            }
            for i in 0..256u32 {
                map.rm(&format!("c{cycle}-k{i}"));
            }
        }
        assert_eq!(map.ln(), 0);
    });
}

// ── Single test driving every scenario ─────────────────────────────────────

#[test]
fn all_scenarios() {
    // Lifecycle: insert, partial remove, reserve, shrink, full drop.
    lifecycle::<UnorderedFlatMap<String, Vec<u32>>>("lifecycle: UnorderedFlatMap");
    lifecycle::<Splitsies<String, Vec<u32>>>("lifecycle: Splitsies");
    lifecycle::<InPlaceOverflow<String, Vec<u32>>>("lifecycle: InPlaceOverflow");
    lifecycle::<IPO64<String, Vec<u32>>>("lifecycle: IPO64");
    lifecycle::<Gaps<String, Vec<u32>>>("lifecycle: Gaps");

    // Oversized capacity hint with zero inserts.
    cap_no_inserts::<UnorderedFlatMap<String, Vec<u32>>>("cap_no_inserts: UFM");
    cap_no_inserts::<Splitsies<String, Vec<u32>>>("cap_no_inserts: Splitsies");
    cap_no_inserts::<InPlaceOverflow<String, Vec<u32>>>("cap_no_inserts: IPO");
    cap_no_inserts::<IPO64<String, Vec<u32>>>("cap_no_inserts: IPO64");
    cap_no_inserts::<Gaps<String, Vec<u32>>>("cap_no_inserts: Gaps");

    // Capacity hint with partial fill.
    cap_partial_fill::<UnorderedFlatMap<String, Vec<u32>>>("cap_partial_fill: UFM");
    cap_partial_fill::<Splitsies<String, Vec<u32>>>("cap_partial_fill: Splitsies");
    cap_partial_fill::<InPlaceOverflow<String, Vec<u32>>>("cap_partial_fill: IPO");
    cap_partial_fill::<IPO64<String, Vec<u32>>>("cap_partial_fill: IPO64");
    cap_partial_fill::<Gaps<String, Vec<u32>>>("cap_partial_fill: Gaps");

    // Insert/remove churn over multiple cycles.
    churn_cycles::<UnorderedFlatMap<String, Vec<u32>>>("churn_cycles: UnorderedFlatMap");
    churn_cycles::<Splitsies<String, Vec<u32>>>("churn_cycles: Splitsies");
    churn_cycles::<InPlaceOverflow<String, Vec<u32>>>("churn_cycles: InPlaceOverflow");
    churn_cycles::<IPO64<String, Vec<u32>>>("churn_cycles: IPO64");
    churn_cycles::<Gaps<String, Vec<u32>>>("churn_cycles: Gaps");

    // Force at least one allocation per backend so the global allocator
    // observes the bucket alignment each design wants. The bucket layout for
    // these backends includes the `*mut (K, V)` slot array (8B alignment for
    // u64 KV) plus metadata; at minimum we expect ≥ 8.
    {
        let mut m = UnorderedFlatMap::<u64, u64>::with_capacity(16);
        m.insert(1, 1);
    }
    {
        let mut m = Splitsies::<u64, u64>::with_capacity(16);
        m.insert(1, 1);
    }
    {
        let mut m = InPlaceOverflow::<u64, u64>::with_capacity(16);
        m.insert(1, 1);
    }
    {
        let mut m = IPO64::<u64, u64>::with_capacity(16);
        m.insert(1, 1);
    }
    {
        let mut m = Gaps::<u64, u64>::with_capacity(16);
        m.insert(1, 1);
    }

    let observed = MAX_ALIGN_SEEN.load(Ordering::Relaxed);
    assert!(
        observed >= 8,
        "expected ≥ 8-byte alignment from some bucket allocation, got {observed}"
    );
}
