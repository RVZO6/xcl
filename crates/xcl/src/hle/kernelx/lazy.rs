//! Physical pages mapped into the title's own reservations, committed on first use.
//!
//! A title maps its physical pages wherever it reserved address space and places resources
//! there – render targets and depth buffers too, which on the PC live in video memory
//! (`d3d12x::vram`); the CPU never touches those pages. Committed at map time they doubled
//! that memory (a retail game: 5.4 GB of mapped pages next to its video-memory targets, past
//! the system's commit). So a page whose contents are still zero is mapped as reserved address
//! space only: the CPU's first access faults, the page is committed (zeros, as the page holds)
//! and the access goes on. Kernel calls do not fault in the title – they fail – so the buffers
//! of the file calls XCL routes, and DirectStorage's destinations, are committed first
//! (`ensure`).

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::{Once, RwLock};

use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
    PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY,
};
use windows_sys::Win32::System::Memory::{PAGE_EXECUTE, PAGE_NOACCESS};

const PAGE: usize = 0x1_0000;
const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

/// pages (64 KiB) mapped but not committed yet
static LAZY: RwLock<BTreeMap<usize, u32>> = RwLock::new(BTreeMap::new());

fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(handler));
    });
}

fn access_allowed(access: usize, protect: u32) -> bool {
    if protect == 0 || protect & (PAGE_NOACCESS | PAGE_GUARD) != 0 { return false; }
    let w = PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    let r = w | PAGE_READONLY | PAGE_EXECUTE_READ;
    let x = PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    match access { 0 => protect & r != 0, 1 => protect & w != 0, 8 => protect & x != 0, _ => false }
}

/// Commit `page` if it is a lazy one (true: it was).
unsafe fn commit(page: usize) -> bool {
    let mut g = LAZY.write().unwrap_or_else(|e| e.into_inner());
    let Some(&prot) = g.get(&page) else { return false };
    if VirtualAlloc(page as *const c_void, PAGE, MEM_COMMIT, prot).is_null() { return false; }
    g.remove(&page);
    true
}

unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
    let r = &*(*info).ExceptionRecord;
    if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    let (access, addr) = (r.ExceptionInformation[0], r.ExceptionInformation[1]);
    let page = addr & !(PAGE - 1);
    let prot = LAZY.read().unwrap_or_else(|e| e.into_inner()).get(&page).copied();
    if prot.is_some_and(|p| access_allowed(access, p)) && commit(page) || allows(access, addr) {
        EXCEPTION_CONTINUE_EXECUTION
    } else {
        EXCEPTION_CONTINUE_SEARCH
    }
}

unsafe fn allows(access: usize, at: usize) -> bool {
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    VirtualQuery(at as *const c_void, &mut m, std::mem::size_of_val(&m)) != 0 && m.State == MEM_COMMIT && access_allowed(access, m.Protect)
}

/// `page` (reserved, 64 KiB aligned) is mapped: committed on first use.
pub(crate) fn mark(page: usize) { mark_protect(page, PAGE_READWRITE); }

pub(crate) fn mark_protect(page: usize, protect: u32) {
    install();
    LAZY.write().unwrap_or_else(|e| e.into_inner()).insert(page & !(PAGE - 1), super::sanitize_protect(protect));
}

/// `page` is unmapped: true if it was never used (nothing to keep of it).
pub(super) fn unmark(page: usize) -> bool {
    LAZY.write().unwrap_or_else(|e| e.into_inner()).remove(&(page & !(PAGE - 1))).is_some()
}

pub(crate) fn unmark_range(p: usize, len: usize) -> usize {
    let Some(r) = (len > 0).then(|| p & !(PAGE - 1)..p.saturating_add(len)) else { return 0 };
    if LAZY.read().unwrap_or_else(|e| e.into_inner()).range(r.clone()).next().is_none() { return 0; }
    let mut g = LAZY.write().unwrap_or_else(|e| e.into_inner());
    let keys: Vec<usize> = g.range(r).map(|(&k, _)| k).collect();
    keys.iter().for_each(|k| { g.remove(k); });
    keys.len()
}

pub(crate) fn is_lazy(page: usize) -> bool { LAZY.read().unwrap_or_else(|e| e.into_inner()).contains_key(&(page & !(PAGE - 1))) }

pub(crate) fn query(addr: usize) -> Option<(usize, usize, u32)> {
    let (page, g) = (addr & !(PAGE - 1), LAZY.read().unwrap_or_else(|e| e.into_inner()));
    let (&prot, mut end) = (g.get(&page)?, page.checked_add(PAGE)?);
    for (&k, &p) in g.range(end..) {
        if k != end || p != prot { break; }
        end = match end.checked_add(PAGE) { Some(n) => n, None => break };
    }
    Some((page, end - page, prot))
}

fn ensure_locked(g: &mut BTreeMap<usize, u32>, lo: usize, hi: usize) -> bool {
    let pages: Vec<(usize, u32)> = g.range(lo..hi).map(|(&k, &v)| (k, v)).collect();
    let mut i = 0;
    while i < pages.len() {
        let ((start, prot), mut j) = (pages[i], i + 1);
        let mut end = start.saturating_add(PAGE);
        while j < pages.len() && pages[j] == (end, prot) { end = end.saturating_add(PAGE); j += 1; }
        let ok = |a, n| !unsafe { VirtualAlloc(a as *const c_void, n, MEM_COMMIT, prot) }.is_null();
        if ok(start, end - start) { pages[i..j].iter().for_each(|&(p, _)| { g.remove(&p); }); }
        else if j - i == 1 || pages[i..j].iter().any(|&(p, _)| !ok(p, PAGE) || { g.remove(&p); false }) { return false; }
        i = j;
    }
    true
}

/// The lazy pages of `[p, p + len)` committed – memory a kernel call or XCL writes.
pub fn ensure(p: usize, len: usize) -> bool {
    let Some(r) = (len > 0).then(|| p & !(PAGE - 1)..p.saturating_add(len)) else { return true };
    LAZY.read().unwrap_or_else(|e| e.into_inner()).range(r.clone()).next().is_none()
        || ensure_locked(&mut LAZY.write().unwrap_or_else(|e| e.into_inner()), r.start, r.end)
}

pub(crate) fn reprotect(p: usize, len: usize, new_prot: u32) -> Option<u32> {
    let r = (len > 0).then(|| p & !(PAGE - 1)..p.saturating_add(len))?;
    if LAZY.read().unwrap_or_else(|e| e.into_inner()).range(r.clone()).next().is_none() { return None; }
    let mut g = LAZY.write().unwrap_or_else(|e| e.into_inner());
    if p & (PAGE - 1) == 0 && len & (PAGE - 1) == 0 && (new_prot & 0xFF).is_power_of_two() && new_prot & (PAGE_NOACCESS | PAGE_GUARD) == 0 {
        if let Some(end) = p.checked_add(len).filter(|&end| g.range(p..end).count() == len / PAGE) {
            let old = *g.get(&p).unwrap();
            g.range_mut(p..end).for_each(|(_, prot)| *prot = new_prot);
            return Some(old);
        }
    }
    ensure_locked(&mut g, r.start, r.end);
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{VirtualFree, MEM_RELEASE, MEM_RESERVE};

    /// A lazy page reads as zeros on first touch, keeps what is written; `ensure` commits it
    /// without a fault.
    #[test]
    fn a_lazy_page_is_committed_on_first_use() {
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            mark(r);
            mark(r + PAGE);
            assert_eq!(std::ptr::read_volatile((r + 8) as *const u32), 0);
            std::ptr::write_volatile((r + 8) as *mut u32, 7);
            assert_eq!(std::ptr::read_volatile((r + 8) as *const u32), 7);
            assert!(!unmark(r), "used: no longer lazy");
            ensure(r + PAGE + 4, 4);
            assert!(!is_lazy(r + PAGE));
            assert!(crate::gpu::shader::readable(r + PAGE, 4));
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }

    unsafe fn vq(a: usize) -> MEMORY_BASIC_INFORMATION {
        let mut m = std::mem::zeroed(); assert_ne!(VirtualQuery(a as *const c_void, &mut m, std::mem::size_of_val(&m)), 0); m
    }

    #[test]
    fn readonly_lazy_page_commits_on_read_and_stays_readonly() {
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            mark_protect(r, PAGE_READONLY);
            assert!(access_allowed(0, PAGE_READONLY) && !access_allowed(1, PAGE_READONLY) && !access_allowed(0, PAGE_NOACCESS) && !access_allowed(0, PAGE_READWRITE | PAGE_GUARD) && access_allowed(8, PAGE_EXECUTE_READ));
            let mut rec: windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_RECORD = std::mem::zeroed();
            (rec.ExceptionCode, rec.NumberParameters, rec.ExceptionInformation[0], rec.ExceptionInformation[1]) = (EXCEPTION_ACCESS_VIOLATION as NTSTATUS, 2, 1, r + 8);
            assert_eq!(handler(&mut EXCEPTION_POINTERS { ExceptionRecord: &mut rec, ContextRecord: std::ptr::null_mut() }), EXCEPTION_CONTINUE_SEARCH);
            assert!(is_lazy(r) && std::ptr::read_volatile((r + 8) as *const u32) == 0 && !is_lazy(r) && allows(0, r + 8) && !allows(1, r + 8));
            assert_eq!((vq(r).State, vq(r).Protect), (MEM_COMMIT, PAGE_READONLY));
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn ensure_batches_contiguous_runs_and_unmark_range_clears() {
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            for (i, p) in [PAGE_READWRITE, PAGE_READWRITE, PAGE_READONLY, PAGE_READWRITE].into_iter().enumerate() { mark_protect(r + i * PAGE, p); }
            assert_eq!(query(r), Some((r, 2 * PAGE, PAGE_READWRITE)));
            assert!(ensure(r + 0x100, 2 * PAGE) && !is_lazy(r) && !is_lazy(r + PAGE) && !is_lazy(r + 2 * PAGE) && is_lazy(r + 3 * PAGE));
            let (m0, m2) = (vq(r), vq(r + 2 * PAGE));
            assert_eq!((m0.State, m0.Protect, m0.RegionSize, m2.State, m2.Protect, m2.RegionSize), (MEM_COMMIT, PAGE_READWRITE, 2 * PAGE, MEM_COMMIT, PAGE_READONLY, PAGE));
            mark_protect(0x7FFF_FFFF_0000, PAGE_READWRITE);
            assert!(unmark_range(r, 4 * PAGE) == 1 && !is_lazy(r + 3 * PAGE) && !ensure(0x7FFF_FFFF_0000, PAGE) && is_lazy(0x7FFF_FFFF_0000) && unmark(0x7FFF_FFFF_0000));
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn reprotect_updates_lazy_page_without_committing() {
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            mark_protect(r, PAGE_READWRITE);
            mark_protect(r + PAGE, PAGE_READWRITE);
            assert_eq!((reprotect(r, 2 * PAGE, PAGE_READONLY), query(r), vq(r).State), (Some(PAGE_READWRITE), Some((r, 2 * PAGE, PAGE_READONLY)), MEM_RESERVE));
            assert!(reprotect(r, 0x1000, PAGE_READWRITE).is_none() && !is_lazy(r) && is_lazy(r + PAGE));
            assert!(reprotect(r + PAGE, PAGE, PAGE_NOACCESS).is_none() && !is_lazy(r + PAGE));
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }

    #[test]
    fn threads_first_touching_one_lazy_page_together_all_go_on() {
        const THREADS: usize = 8;
        const PAGES: usize = 256;
        unsafe {
            let r = VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
            for i in 0..PAGES {
                mark(r + i * PAGE);
            }
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(THREADS));
            let handles: Vec<_> = (0..THREADS)
                .map(|t| {
                    let barrier = barrier.clone();
                    std::thread::spawn(move || {
                        for i in 0..PAGES {
                            barrier.wait();
                            std::ptr::write_volatile((r + i * PAGE + t * 64) as *mut u64, (i * THREADS + t) as u64);
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
            for i in 0..PAGES {
                for t in 0..THREADS {
                    assert_eq!(std::ptr::read_volatile((r + i * PAGE + t * 64) as *const u64), (i * THREADS + t) as u64);
                }
            }
            VirtualFree(r as *mut c_void, 0, MEM_RELEASE);
        }
    }
}
