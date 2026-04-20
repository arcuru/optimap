//! Key-value storage layout strategies.
//!
//! `KvStorage` abstracts how keys and values are laid out in memory:
//! - `AoS` (Array of Structs): interleaved `(K, V)` tuples — the default
//! - `SoA` (Struct of Arrays): separate key and value arrays
//!
//! The probe loop only needs key access. SoA keeps values out of cache
//! during probing, which helps when values are large.

use std::ptr;

/// How keys and values are stored in the hash table's backing memory.
///
/// Both implementations are fully monomorphized — zero-cost abstraction.
pub trait KvStorage<K, V>: 'static + Copy {
    /// Extra state beyond `ctrl`. AoS: `()`. SoA: `*mut u8` (values base).
    type Extra: Copy;

    fn extra_null() -> Self::Extra;

    /// Size of the region backward from ctrl (keys for SoA, (K,V) tuples for AoS).
    /// Rounded up to 16-byte alignment for SIMD ctrl access.
    fn backward_size(num_slots: usize) -> usize;

    /// Size of the values region placed after metadata+overflow.
    /// AoS: 0 (values are interleaved with keys). SoA: sizeof(V) * num_slots.
    fn values_region_size(num_slots: usize) -> usize;

    /// Alignment required for the values region. AoS: 1. SoA: align_of::<V>().
    fn values_align() -> usize;

    /// Minimum allocation alignment.
    fn alloc_align() -> usize;

    /// Initialize the extra pointer after allocation.
    /// `ctrl` is the metadata pointer. `values_offset` is the byte offset
    /// from ctrl to the start of the values region (after metadata+overflow, aligned).
    unsafe fn init_extra(ctrl: *mut u8, values_offset: usize) -> Self::Extra;

    /// Pointer to the key at bucket index `idx`. Backward from ctrl.
    unsafe fn key_ptr(ctrl: *mut u8, idx: usize) -> *mut K;

    /// Pointer to the value at bucket index `idx`.
    unsafe fn value_ptr(ctrl: *mut u8, extra: Self::Extra, idx: usize) -> *mut V;

    /// Write a key-value pair to bucket index `idx`.
    #[inline(always)]
    unsafe fn write(ctrl: *mut u8, extra: Self::Extra, idx: usize, key: K, value: V) {
        unsafe {
            Self::key_ptr(ctrl, idx).write(key);
            Self::value_ptr(ctrl, extra, idx).write(value);
        }
    }

    /// Read (move out) the key-value pair at bucket index `idx`.
    #[inline(always)]
    unsafe fn read(ctrl: *mut u8, extra: Self::Extra, idx: usize) -> (K, V) {
        unsafe {
            let k = Self::key_ptr(ctrl, idx).read();
            let v = Self::value_ptr(ctrl, extra, idx).read();
            (k, v)
        }
    }

    /// Drop the key-value pair at bucket index `idx` in place.
    #[inline(always)]
    unsafe fn drop_slot(ctrl: *mut u8, extra: Self::Extra, idx: usize) {
        unsafe {
            ptr::drop_in_place(Self::key_ptr(ctrl, idx));
            ptr::drop_in_place(Self::value_ptr(ctrl, extra, idx));
        }
    }

    /// Clone a slot from src to dst.
    #[inline(always)]
    unsafe fn clone_slot(
        src_ctrl: *mut u8, src_extra: Self::Extra,
        dst_ctrl: *mut u8, dst_extra: Self::Extra,
        idx: usize,
    ) where K: Clone, V: Clone {
        unsafe {
            let sk = &*Self::key_ptr(src_ctrl, idx);
            let sv = &*Self::value_ptr(src_ctrl, src_extra, idx);
            Self::key_ptr(dst_ctrl, idx).write(sk.clone());
            Self::value_ptr(dst_ctrl, dst_extra, idx).write(sv.clone());
        }
    }

    /// Whether K or V need dropping.
    fn needs_drop() -> bool {
        std::mem::needs_drop::<K>() || std::mem::needs_drop::<V>()
    }

    /// Prefetch data for bucket index `idx`.
    #[inline(always)]
    unsafe fn prefetch(_ctrl: *mut u8, _extra: Self::Extra, _idx: usize) {}
}

// ── AoS: Array of Structs (default) ───────────────────────────────────────

/// Interleaved (K, V) tuples. The current default layout.
#[derive(Clone, Copy)]
pub struct AoS;

impl<K, V> KvStorage<K, V> for AoS {
    type Extra = ();

    fn extra_null() -> () {}

    fn backward_size(num_slots: usize) -> usize {
        let raw = num_slots * std::mem::size_of::<(K, V)>();
        (raw + 15) & !15
    }

    fn values_region_size(_num_slots: usize) -> usize { 0 }
    fn values_align() -> usize { 1 }

    fn alloc_align() -> usize {
        16usize.max(std::mem::align_of::<(K, V)>())
    }

    unsafe fn init_extra(_ctrl: *mut u8, _values_offset: usize) -> () {}

    #[inline(always)]
    unsafe fn key_ptr(ctrl: *mut u8, idx: usize) -> *mut K {
        unsafe { ctrl.cast::<(K, V)>().sub(idx + 1).cast::<K>() }
    }

    #[inline(always)]
    unsafe fn value_ptr(ctrl: *mut u8, _extra: (), idx: usize) -> *mut V {
        unsafe {
            let tuple_ptr = ctrl.cast::<(K, V)>().sub(idx + 1);
            &raw mut (*tuple_ptr).1
        }
    }

    #[inline(always)]
    unsafe fn write(ctrl: *mut u8, _extra: (), idx: usize, key: K, value: V) {
        unsafe { ctrl.cast::<(K, V)>().sub(idx + 1).write((key, value)); }
    }

    #[inline(always)]
    unsafe fn read(ctrl: *mut u8, _extra: (), idx: usize) -> (K, V) {
        unsafe { ctrl.cast::<(K, V)>().sub(idx + 1).read() }
    }

    #[inline(always)]
    unsafe fn drop_slot(ctrl: *mut u8, _extra: (), idx: usize) {
        unsafe { ptr::drop_in_place(ctrl.cast::<(K, V)>().sub(idx + 1)); }
    }

    #[inline(always)]
    unsafe fn clone_slot(
        src_ctrl: *mut u8, _src_extra: (),
        dst_ctrl: *mut u8, _dst_extra: (),
        idx: usize,
    ) where K: Clone, V: Clone {
        unsafe {
            let src = &*src_ctrl.cast::<(K, V)>().sub(idx + 1);
            dst_ctrl.cast::<(K, V)>().sub(idx + 1).write(src.clone());
        }
    }

    fn needs_drop() -> bool {
        std::mem::needs_drop::<(K, V)>()
    }
}

// ── SoA: Struct of Arrays ─────────────────────────────────────────────────

/// Separate key and value arrays. Probe loop only touches keys.
#[derive(Clone, Copy)]
pub struct SoA;

impl<K, V> KvStorage<K, V> for SoA {
    type Extra = *mut u8;

    fn extra_null() -> *mut u8 { ptr::null_mut() }

    fn backward_size(num_slots: usize) -> usize {
        let raw = num_slots * std::mem::size_of::<K>();
        (raw + 15) & !15
    }

    fn values_region_size(num_slots: usize) -> usize {
        num_slots * std::mem::size_of::<V>()
    }

    fn values_align() -> usize {
        std::mem::align_of::<V>().max(1)
    }

    fn alloc_align() -> usize {
        16usize
            .max(std::mem::align_of::<K>())
            .max(std::mem::align_of::<V>())
    }

    unsafe fn init_extra(ctrl: *mut u8, values_offset: usize) -> *mut u8 {
        unsafe { ctrl.add(values_offset) }
    }

    #[inline(always)]
    unsafe fn key_ptr(ctrl: *mut u8, idx: usize) -> *mut K {
        unsafe { ctrl.cast::<K>().sub(idx + 1) }
    }

    #[inline(always)]
    unsafe fn value_ptr(_ctrl: *mut u8, extra: *mut u8, idx: usize) -> *mut V {
        unsafe { extra.cast::<V>().add(idx) }
    }
}

unsafe impl Send for SoA {}
unsafe impl Sync for SoA {}
