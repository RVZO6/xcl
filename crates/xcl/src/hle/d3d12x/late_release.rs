//! The title's last `Release` of a resource, put off until the GPU is past the work in flight.
//!
//! D3D12 on the PC asks that an object outlives the GPU work that uses it. A console title
//! keeps to that by its own count of finished frames – but XCL's submits are not the title's:
//! texture uploads go out with a later submit, tile mappings on a queue of XCL's own. A retail
//! title's final `Release` of a resource then came while a queue still had work on it (debug
//! layer: "CORRUPTION: … is referenced by GPU operations in-flight", id 921).
//!
//! The last reference is held here instead. Each batch of held resources gets a fence value
//! signalled on every queue with work on it; the batch goes once all of them have passed it.
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// a batch is not closed more often than this (a `Signal` a queue each time)
const SEAL_EVERY: Duration = Duration::from_millis(10);
/// how often the thread here looks for batches that can go
const TICK: Duration = Duration::from_millis(4);

struct Batch {
    dead: Vec<usize>,
    /// (fence, value) a queue: all reached = nothing submitted before the batch closed is left
    marks: Vec<(ID3D12Fence, u64)>,
}

#[derive(Default)]
struct Held {
    /// queue → its fence here and the last value signalled on it
    fences: HashMap<usize, (ID3D12Fence, u64)>,
    open: Vec<usize>,
    sealed: VecDeque<Batch>,
    last_seal: Option<Instant>,
}

// (the interface pointers are only used under the lock)
unsafe impl Send for Held {}

impl Held {
    fn is_empty(&self) -> bool {
        self.open.is_empty() && self.sealed.is_empty()
    }

    fn needs_seal(&self, seal_now: bool) -> bool {
        !self.open.is_empty() && (seal_now || self.last_seal.is_none_or(|t| t.elapsed() >= SEAL_EVERY))
    }

    unsafe fn drain_completed(&mut self) -> Vec<usize> {
        let mut gone = Vec::new();
        while self.sealed.front().is_some_and(|b| b.marks.iter().all(|(f, v)| f.GetCompletedValue() >= *v)) {
            let dead = self.sealed.pop_front().unwrap().dead;
            if gone.is_empty() { gone = dead; } else { gone.extend(dead); }
        }
        gone
    }

    /// Close the open batch (at most every `SEAL_EVERY`, or `seal_now`) with a fence value on each of `queues`,
    /// and take out the batches every queue is past: the resources to release now.
    unsafe fn drain(&mut self, queues: &[ID3D12CommandQueue], seal_now: bool) -> Vec<usize> {
        if self.needs_seal(seal_now) {
            let mut marks = Vec::with_capacity(queues.len());
            let mut ok = true;
            for q in queues {
                let key = q.as_raw() as usize;
                if !self.fences.contains_key(&key) {
                    let mut dev: Option<ID3D12Device> = None;
                    let made = q.GetDevice(&mut dev).ok().and(dev).and_then(|d| d.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE).ok());
                    let Some(f) = made else {
                        ok = false;
                        break;
                    };
                    self.fences.insert(key, (f, 0));
                }
                let (fence, value) = self.fences.get_mut(&key).unwrap();
                if q.Signal(&*fence, *value + 1).is_err() {
                    ok = false;
                    break;
                }
                *value += 1;
                marks.push((fence.clone(), *value));
            }
            // (no fence or no signal: the device is gone – the batch stays open, and held)
            if ok {
                self.last_seal = Some(Instant::now());
                let dead = std::mem::take(&mut self.open);
                self.sealed.push_back(Batch { dead, marks });
            }
        }
        self.drain_completed()
    }
}

static HELD: Mutex<Option<Held>> = Mutex::new(None);

/// Whether last releases are held (`XCL_LATE_RELEASE=0` turns it off).
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XCL_LATE_RELEASE").is_ok_and(|v| v.trim() == "0"))
}

/// how long a last `Release` itself waits for queues with nothing to do to say so
const IDLE_WAIT: Duration = Duration::from_micros(300);

unsafe fn hold_with<F: FnMut() -> Vec<ID3D12CommandQueue>>(held: &Mutex<Option<Held>>, res: usize, collect_queues: F) -> bool {
    let (gone, gpu_busy) = {
        let mut g = held.lock().unwrap();
        let h = g.get_or_insert_with(Held::default);
        let gone = h.drain_completed();
        h.open.push(res);
        (gone, !h.sealed.is_empty())
    };
    release_all(gone);
    if gpu_busy { return true; }
    let start = Instant::now();
    drain_with(held, true, collect_queues);
    loop {
        let (gone, sealed_empty, still_held) = held.lock().unwrap().as_mut().map_or((Vec::new(), true, false), |h| {
            (h.drain_completed(), h.sealed.is_empty(), !h.is_empty())
        });
        release_all(gone);
        if sealed_empty { return still_held; }
        if start.elapsed() >= IDLE_WAIT { return true; }
        std::hint::spin_loop();
    }
}

/// `res` (an adopted resource) holds one reference that is now this module's. With no work in
/// flight it goes before this returns (the queues pass a fence value at once): a title that
/// frees the memory under a resource right after its last `Release` finds it unpinned, as
/// before. Otherwise the thread here lets it go.
pub unsafe fn hold(res: usize) {
    if !hold_with(&HELD, res, || unsafe { collect_known_queues() }) { return; }
    // A thread of its own lets the held ones go: the title may release a resource and then
    // wait for its memory without another submit (an imported heap pins the pages under its
    // resources – unmapping them stood still while one was held "until the next submit").
    static THREAD: std::sync::Once = std::sync::Once::new();
    THREAD.call_once(|| {
        let _ = std::thread::Builder::new().name("xcl-late-release".into()).spawn(|| loop {
            std::thread::sleep(TICK);
            unsafe { drain_known(false) };
        });
    });
}

/// Release what every queue XCL knows of is past: the title's, the tile-mapping one, the
/// upload ones.
pub unsafe fn drain_known(seal_now: bool) {
    drain_with(&HELD, seal_now, || unsafe { collect_known_queues() });
}

unsafe fn collect_known_queues() -> Vec<ID3D12CommandQueue> {
    let mut queues: Vec<ID3D12CommandQueue> = Vec::new();
    for raw in super::present::known_queues() {
        let raw = raw as *mut c_void;
        if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&raw) {
            queues.push(q.clone());
        }
    }
    queues.extend(super::sparse::queue());
    queues.extend(super::title_textures::upload_queues());
    queues
}

unsafe fn drain_with<F: FnMut() -> Vec<ID3D12CommandQueue>>(held: &Mutex<Option<Held>>, seal_now: bool, mut collect_queues: F) {
    let (mut gone, need_seal) = {
        let mut g = held.lock().unwrap();
        let Some(h) = g.as_mut().filter(|h| !h.is_empty()) else { return };
        (h.drain_completed(), h.needs_seal(seal_now))
    };
    if need_seal {
        let queues = collect_queues();
        let more = held.lock().unwrap().as_mut().map_or_else(Vec::new, |h| h.drain(&queues, seal_now));
        drop(queues);
        if gone.is_empty() { gone = more; } else { gone.extend(more); }
    }
    release_all(gone);
}

unsafe fn release_all(gone: Vec<usize>) {
    // (outside the lock: a release may end other objects, whose releases come back here)
    for res in gone {
        let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(super::real::orig_fn(res as *mut c_void, 2));
        f(res as *mut c_void);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

    /// A held resource stays while a queue has not got past the work submitted before it was
    /// released, and goes once the queue has.
    #[test]
    fn a_held_resource_goes_when_the_queue_is_past_its_batch() {
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            // the queue stands still, as with work in flight: it waits for a fence nobody has signalled
            let gate: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            queue.Wait(&gate, 1).unwrap();
            let mut res: Option<ID3D12Resource> = None;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 4096,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut res).unwrap();
            let raw = res.unwrap().into_raw() as usize;

            let mut held = Held::default();
            held.open.push(raw);
            let queues = [queue.clone()];
            assert!(held.drain(&queues, false).is_empty(), "the queue has not passed the batch's fence value");
            std::thread::sleep(Duration::from_millis(100));
            assert!(held.drain(&queues, false).is_empty(), "still in flight: still held");
            assert!(!held.is_empty());

            gate.Signal(1).unwrap();
            let start = Instant::now();
            let mut gone = Vec::new();
            while gone.is_empty() && start.elapsed() < Duration::from_secs(10) {
                std::thread::sleep(Duration::from_millis(5));
                gone = held.drain(&queues, false);
            }
            assert_eq!(gone, vec![raw], "released once the queue is past the batch");
            assert!(held.is_empty());
            drop(ID3D12Resource::from_raw(raw as *mut c_void));
        }
    }

    /// With no queue at all nothing can be in flight: the next drain lets the batch go.
    #[test]
    fn without_queues_a_batch_goes_at_once() {
        let mut held = Held::default();
        held.open.push(0x1000);
        assert_eq!(unsafe { held.drain(&[], false) }, vec![0x1000]);
        assert!(held.is_empty());
    }

    use std::mem::{offset_of, size_of};
    use std::sync::{atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering}, OnceLock};
    use windows::core::{GUID, HRESULT};

    #[repr(C)] struct FakeFence { vt: *const usize, completed: AtomicU64, signaled: AtomicU64, complete_after: AtomicU32, polls: AtomicU32 }
    #[repr(C)] struct FakeDevice { vt: *const usize, fence: *const FakeFence }
    #[repr(C)] struct FakeQueue { vt: *const usize, device: *const FakeDevice, signals: AtomicU32 }
    #[repr(C)] struct FakeRes { vt: *const usize, held: *const Mutex<Option<Held>>, releases: AtomicU32, unlocked: AtomicBool }

    extern "system" fn unexpected_call() { std::process::abort(); }
    unsafe extern "system" fn fake_add_ref(_: *mut c_void) -> u32 { 2 }
    unsafe extern "system" fn fake_release(_: *mut c_void) -> u32 { 1 }
    unsafe extern "system" fn fake_fence_completed(this: *mut c_void) -> u64 {
        let f = &*this.cast::<FakeFence>();
        f.polls.fetch_add(1, Ordering::Relaxed);
        if f.complete_after.load(Ordering::Relaxed) > 0 && f.complete_after.fetch_sub(1, Ordering::Relaxed) == 1 {
            f.completed.store(f.signaled.load(Ordering::Relaxed), Ordering::Relaxed);
        }
        f.completed.load(Ordering::Relaxed)
    }
    unsafe extern "system" fn fake_create_fence(this: *mut c_void, init: u64, _: D3D12_FENCE_FLAGS, _: *const GUID, out: *mut *mut c_void) -> HRESULT {
        let fence = (*this.cast::<FakeDevice>()).fence;
        (*fence).completed.store(init, Ordering::Relaxed); *out = fence.cast_mut().cast(); HRESULT(0)
    }
    unsafe extern "system" fn fake_get_device(this: *mut c_void, _: *const GUID, out: *mut *mut c_void) -> HRESULT {
        *out = (*this.cast::<FakeQueue>()).device.cast_mut().cast(); HRESULT(0)
    }
    unsafe extern "system" fn fake_queue_signal(this: *mut c_void, fence: *mut c_void, value: u64) -> HRESULT {
        (*this.cast::<FakeQueue>()).signals.fetch_add(1, Ordering::Relaxed);
        (*fence.cast::<FakeFence>()).signaled.store(value, Ordering::Relaxed); HRESULT(0)
    }
    unsafe extern "system" fn fake_res_release(this: *mut c_void) -> u32 {
        let r = &*this.cast::<FakeRes>();
        r.unlocked.store((*r.held).try_lock().is_ok(), Ordering::Relaxed); r.releases.fetch_add(1, Ordering::Relaxed); 0
    }
    fn com_vt<const N: usize>(extra: &[(usize, usize)]) -> [usize; N] {
        let mut v = [unexpected_call as usize; N];
        (v[1], v[2]) = (fake_add_ref as usize, fake_release as usize);
        for &(slot, f) in extra { v[slot] = f; }
        v
    }

    struct FakeEnv { queue: Box<FakeQueue>, _device: Box<FakeDevice>, fence: Box<FakeFence>, collects: AtomicU32 }
    impl FakeEnv {
        fn new(complete_after: u32) -> Self {
            const S: usize = size_of::<usize>();
            type Vts = ([usize; size_of::<ID3D12Fence_Vtbl>() / S], [usize; size_of::<ID3D12Device_Vtbl>() / S], [usize; size_of::<ID3D12CommandQueue_Vtbl>() / S]);
            static VTS: OnceLock<Vts> = OnceLock::new();
            let (f_vt, d_vt, q_vt) = VTS.get_or_init(|| (
                com_vt(&[(offset_of!(ID3D12Fence_Vtbl, GetCompletedValue) / S, fake_fence_completed as usize)]),
                com_vt(&[(offset_of!(ID3D12Device_Vtbl, CreateFence) / S, fake_create_fence as usize)]),
                com_vt(&[(offset_of!(ID3D12CommandQueue_Vtbl, base__.base__.GetDevice) / S, fake_get_device as usize), (offset_of!(ID3D12CommandQueue_Vtbl, Signal) / S, fake_queue_signal as usize)]),
            ));
            let fence = Box::new(FakeFence { vt: f_vt.as_ptr(), completed: AtomicU64::new(0), signaled: AtomicU64::new(0), complete_after: AtomicU32::new(complete_after), polls: AtomicU32::new(0) });
            let device = Box::new(FakeDevice { vt: d_vt.as_ptr(), fence: &*fence });
            let queue = Box::new(FakeQueue { vt: q_vt.as_ptr(), device: &*device, signals: AtomicU32::new(0) });
            Self { queue, _device: device, fence, collects: AtomicU32::new(0) }
        }
        fn collect(&self) -> Vec<ID3D12CommandQueue> {
            self.collects.fetch_add(1, Ordering::Relaxed);
            let raw = (&*self.queue as *const FakeQueue).cast_mut().cast::<c_void>();
            vec![unsafe { ID3D12CommandQueue::from_raw_borrowed(&raw).unwrap().clone() }]
        }
        fn stats(&self) -> (u32, u32) { (self.collects.load(Ordering::Relaxed), self.queue.signals.load(Ordering::Relaxed)) }
    }
    impl FakeRes {
        fn new(held: &Mutex<Option<Held>>) -> Box<Self> {
            static ORIG: OnceLock<[usize; 3]> = OnceLock::new();
            static EXT: OnceLock<[usize; 4]> = OnceLock::new();
            let ext = EXT.get_or_init(|| { let o = ORIG.get_or_init(|| [unexpected_call as usize, fake_add_ref as usize, fake_res_release as usize]); [o.as_ptr() as usize, o[0], o[1], o[2]] });
            Box::new(Self { vt: unsafe { ext.as_ptr().add(1) }, held, releases: AtomicU32::new(0), unlocked: AtomicBool::new(false) })
        }
        fn ptr(&self) -> usize { self as *const Self as usize }
        fn released_unlocked(&self) -> bool { self.releases.load(Ordering::Relaxed) == 1 && self.unlocked.load(Ordering::Relaxed) }
    }

    #[test]
    fn drain_skips_queue_collection_when_no_seal_is_due() {
        let env = FakeEnv::new(0);
        let held = Mutex::new(None);
        let (r1, r2) = (FakeRes::new(&held), FakeRes::new(&held));
        held.lock().unwrap().get_or_insert_with(Held::default).open.push(r1.ptr());
        unsafe { drain_with(&held, false, || env.collect()) };
        assert_eq!(env.stats(), (1, 1));
        unsafe { drain_with(&held, false, || env.collect()); drain_with(&held, true, || env.collect()); }
        held.lock().unwrap().as_mut().unwrap().open.push(r2.ptr());
        unsafe { drain_with(&held, false, || env.collect()) };
        assert_eq!(env.stats(), (1, 1));
        env.fence.completed.store(1, Ordering::Relaxed);
        unsafe { drain_with(&held, false, || env.collect()) };
        assert_eq!((env.stats(), r1.released_unlocked(), r2.releases.load(Ordering::Relaxed)), ((1, 1), true, 0));
        unsafe { drain_with(&held, true, || env.collect()) };
        assert_eq!(env.stats(), (2, 2));
        env.fence.completed.store(2, Ordering::Relaxed);
        unsafe { drain_with(&held, false, || env.collect()) };
        assert_eq!((env.stats(), r2.released_unlocked()), ((2, 2), true));
        assert!(held.lock().unwrap().as_ref().is_none_or(Held::is_empty));
    }

    #[test]
    fn hold_batches_without_spinning_or_signaling_while_gpu_is_busy() {
        let env = FakeEnv::new(0);
        let held = Mutex::new(None);
        let (r1, rest) = (FakeRes::new(&held), (0..16).map(|_| FakeRes::new(&held)).collect::<Vec<_>>());
        assert!(unsafe { hold_with(&held, r1.ptr(), || env.collect()) });
        assert_eq!((env.stats(), r1.releases.load(Ordering::Relaxed)), ((1, 1), 0));
        let polls_before = env.fence.polls.load(Ordering::Relaxed);
        for r in &rest { assert!(unsafe { hold_with(&held, r.ptr(), || env.collect()) }); }
        assert_eq!((env.stats(), env.fence.polls.load(Ordering::Relaxed) - polls_before), ((1, 1), rest.len() as u32));
        assert_eq!(held.lock().unwrap().as_ref().unwrap().open.len(), rest.len());
        env.fence.completed.store(1, Ordering::Relaxed);
        unsafe { drain_with(&held, true, || env.collect()) };
        assert_eq!((env.stats(), r1.released_unlocked()), ((2, 2), true));
        env.fence.completed.store(2, Ordering::Relaxed);
        unsafe { drain_with(&held, false, || env.collect()) };
        assert_eq!(env.stats(), (2, 2));
        assert!(rest.iter().all(|r| r.released_unlocked()) && held.lock().unwrap().as_ref().is_none_or(Held::is_empty));
    }

    #[test]
    fn hold_releases_synchronously_when_queues_are_idle_without_recollecting() {
        let env = FakeEnv::new(2);
        let held = Mutex::new(None);
        let r = FakeRes::new(&held);
        assert!(!unsafe { hold_with(&held, r.ptr(), || env.collect()) });
        assert_eq!((env.stats(), env.fence.polls.load(Ordering::Relaxed), r.released_unlocked()), ((1, 1), 2, true));
        assert!(held.lock().unwrap().as_ref().is_none_or(Held::is_empty));
    }
}
