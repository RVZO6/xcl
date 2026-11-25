//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A graphics pipeline draws nothing until then; a compute pipeline is waited
//! for at its first bind (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

struct Pending {
    /// the build, until a worker – or a bind that cannot wait for the queue – takes it
    job: Mutex<Option<Job>>,
    /// the build went to the workers (a lazy one goes on its first `resolve`)
    sent: std::sync::atomic::AtomicBool,
    /// the real pipeline state (raw pointer), 0 while building or when building failed
    real: AtomicUsize,
    done: Mutex<bool>,
    cv: Condvar,
    compute: bool,
    /// of its shaders (`used`); 0 = none
    key: u64,
    /// it was used (`used::note` is told once)
    used: std::sync::atomic::AtomicBool,
    /// the stand-in was released before the real pipeline was built
    orphan: std::sync::atomic::AtomicBool,
    /// a bind waited for it longer than `wait_budget`: binds go on without it until it is built
    late: std::sync::atomic::AtomicBool,
    demand: Mutex<Demand>,
}

/// Whether one of the title's threads waits for the pipeline, and the preload worker building it.
#[derive(Default)]
struct Demand {
    wanted: bool,
    /// thread id of the preload worker (lowest priority) building it, 0 = none
    preload_thread: u32,
}

thread_local! {
    /// what this worker is building
    static BUILDING: std::cell::RefCell<Option<Arc<Pending>>> = const { std::cell::RefCell::new(None) };
}

/// This thread is a worker and one of the title's threads waits for what it is building: the
/// build must not stand back for the title's sake (`real::big_compile_turn`).
pub(super) fn build_is_wanted() -> bool {
    BUILDING.with(|b| b.borrow().as_ref().is_some_and(|p| p.demand.lock().unwrap().wanted))
}

/// Whether one of the title's threads waits for what this thread is doing, to be asked later and
/// from another thread: a worker's build as `build_is_wanted`; a thread that is not XCL's is the
/// title's own.
pub fn wanted_here() -> crate::gpu::recomp::dxc::Wanted {
    if let Some(p) = BUILDING.with(|b| b.borrow().clone()) {
        return Box::new(move || p.demand.lock().unwrap().wanted);
    }
    let own = crate::state::boot().is_some() && !std::thread::current().name().is_some_and(|n| n.starts_with("xcl-"));
    Box::new(move || own)
}

/// A worker's build of `p` (`preload`: by one of the lowest-priority workers).
fn build(p: &Arc<Pending>, job: Job, preload: bool) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetCurrentThreadId, SetThreadPriority, THREAD_PRIORITY_LOWEST, THREAD_PRIORITY_NORMAL};
    BUILDING.with(|b| *b.borrow_mut() = Some(p.clone()));
    if preload {
        let mut d = p.demand.lock().unwrap();
        d.preload_thread = unsafe { GetCurrentThreadId() };
        if d.wanted {
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL) };
        }
    }
    complete(p, job());
    if preload {
        // (under the lock: `want` does not raise this thread after it)
        let mut d = p.demand.lock().unwrap();
        d.preload_thread = 0;
        if d.wanted {
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST) };
        }
    }
    BUILDING.with(|b| *b.borrow_mut() = None);
}

impl Pending {
    /// One of the title's threads is about to wait for it: the worker building it goes on now,
    /// at a normal priority.
    fn want(&self) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Threading::{OpenThread, SetThreadPriority, THREAD_PRIORITY_NORMAL, THREAD_SET_LIMITED_INFORMATION};
        let mut d = self.demand.lock().unwrap();
        if !std::mem::replace(&mut d.wanted, true) && d.preload_thread != 0 {
            unsafe {
                let h = OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, d.preload_thread);
                if !h.is_null() {
                    SetThreadPriority(h, THREAD_PRIORITY_NORMAL);
                    CloseHandle(h);
                }
            }
        }
    }

    fn note_use(&self) {
        if self.key != 0 && !self.used.swap(true, Ordering::Relaxed) {
            used::note(self.key);
        }
    }
}

/// stand-in handed to the title → its real pipeline
static PENDING: Mutex<Option<HashMap<usize, Arc<Pending>>>> = Mutex::new(None);

/// The build is done (`real`: its pipeline, 0 = failed).
fn complete(p: &Pending, real: usize) {
    if real == 0 {
        FAILED.fetch_add(1, Ordering::Relaxed);
    } else {
        BUILT.fetch_add(1, Ordering::Relaxed);
    }
    let orphan = {
        let mut done = p.done.lock().unwrap();
        let orphan = p.orphan.load(Ordering::Acquire);
        if !orphan {
            p.real.store(real, Ordering::Release);
        }
        *done = true;
        p.cv.notify_all();
        orphan
    };
    // outside `done`: the Release override takes PENDING (`counts` takes PENDING, then `done`)
    if orphan {
        release(real);
    }
}
static BUILT: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);
/// binds waiting right now for a pipeline still being built (`title_held`)
static HELD: AtomicUsize = AtomicUsize::new(0);

/// Whether a bind is waiting for a pipeline still being built: one of the title's threads is
/// held where a console title never waits (`kernelx::Sleep` does not count that time).
pub fn title_held() -> bool {
    HELD.load(Ordering::Acquire) != 0
}

#[cfg(test)]
thread_local! {
    static HELD_ACQUIRES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEST_WAIT_BUDGET: std::cell::Cell<Option<Option<std::time::Duration>>> = const { std::cell::Cell::new(None) };
}

/// A bind's wait for a pipeline still being built (`title_held`).
struct Held;

impl Held {
    fn new() -> Held {
        #[cfg(test)]
        HELD_ACQUIRES.with(|c| c.set(c.get() + 1));
        HELD.fetch_add(1, Ordering::AcqRel);
        Held
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        HELD.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn enabled() -> bool {
    std::env::var("XCL_ASYNC_PSO").map_or(true, |v| v != "0")
}

/// Builds the real pipeline: its raw pointer, 0 = failed.
pub type Job = Box<dyn FnOnce() -> usize + Send>;

#[derive(Default)]
struct QueueState {
    wanted_gfx: VecDeque<Arc<Pending>>,
    wanted_cs: VecDeque<Arc<Pending>>,
    gfx: VecDeque<Arc<Pending>>,
    cs: VecDeque<Arc<Pending>>,
    active_compute: usize,
    max_compute: usize,
    closed: bool,
}

struct WorkQueue { state: Mutex<QueueState>, cv: Condvar }
struct ComputeSlotGuard<'a>(&'a WorkQueue);

impl Drop for ComputeSlotGuard<'_> {
    fn drop(&mut self) {
        { let mut g = self.0.state.lock().unwrap_or_else(|e| e.into_inner()); g.active_compute = g.active_compute.saturating_sub(1); }
        self.0.cv.notify_all();
    }
}

fn pop_live(deque: &mut VecDeque<Arc<Pending>>) -> Option<Arc<Pending>> {
    while let Some(p) = deque.pop_front() {
        if !p.orphan.load(Ordering::Acquire) && p.real.load(Ordering::Acquire) == 0 { return Some(p); }
    }
    None
}

impl WorkQueue {
    fn new(workers: usize) -> Self {
        let graphics_reserve = if workers >= 8 { 2 } else if workers >= 2 { 1 } else { 0 };
        let max_compute = workers.saturating_sub(graphics_reserve).max(1);
        Self { state: Mutex::new(QueueState { max_compute, ..Default::default() }), cv: Condvar::new() }
    }
    fn push(&self, p: Arc<Pending>, wanted: bool) {
        let mut g = self.state.lock().unwrap();
        match (p.compute, wanted) {
            (false, true) => g.wanted_gfx.push_back(p),
            (true, true) => g.wanted_cs.push_back(p),
            (false, false) => g.gfx.push_back(p),
            (true, false) => g.cs.push_back(p),
        }
        drop(g);
        self.cv.notify_one();
    }
    fn promote(&self, p: Arc<Pending>) {
        let mut g = self.state.lock().unwrap();
        if p.compute { g.wanted_cs.push_back(p); } else { g.wanted_gfx.push_back(p); }
        drop(g);
        self.cv.notify_all();
    }
    fn pop(&self) -> Option<(Arc<Pending>, Option<ComputeSlotGuard<'_>>)> {
        let mut g = self.state.lock().unwrap();
        loop {
            if let Some(p) = pop_live(&mut g.wanted_gfx) { return Some((p, None)); }
            if let Some(p) = (g.active_compute < g.max_compute).then(|| pop_live(&mut g.wanted_cs)).flatten() {
                g.active_compute += 1; return Some((p, Some(ComputeSlotGuard(self))));
            }
            if let Some(p) = pop_live(&mut g.gfx) { return Some((p, None)); }
            if let Some(p) = (g.active_compute < g.max_compute).then(|| pop_live(&mut g.cs)).flatten() {
                g.active_compute += 1; return Some((p, Some(ComputeSlotGuard(self))));
            }
            if g.closed { return None; }
            g = self.cv.wait(g).unwrap();
        }
    }
    #[cfg(test)]
    fn close(&self) { self.state.lock().unwrap().closed = true; self.cv.notify_all(); }
}

fn queue() -> &'static WorkQueue {
    static Q: OnceLock<WorkQueue> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    // translation recurses deeply on big programs; DXC itself runs on its own pool
    let n = crate::gpu::recomp::dxc::compile_threads();
    let q = Q.get_or_init(|| WorkQueue::new(n));
    STARTED.call_once(|| {
        for k in 0..n {
            let _ = std::thread::Builder::new().name(format!("xcl-pso-{k}")).stack_size(64 << 20).spawn(move || loop {
                let Some((p, _guard)) = q.pop() else { return };
                // (None: a bind took it and builds it itself)
                let job = p.job.lock().unwrap().take();
                if let Some(job) = job {
                    build(&p, job, false);
                }
            });
        }
    });
    q
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_as(standin, compute, job, true, 0);
}

/// Lazy pipeline creation (`XCL_LAZY_PSO=0`: off): a title's pipelines with console shaders get
/// their stand-in when made and are built when first used – the first bind of a graphics one
/// sends it to the workers, a compute one is built at its first bind. A retail game makes
/// ~51 000 pipelines in its first 30 s and uses few; each one built stays in the driver
/// (~400 KB), and the process ran out of commit.
pub fn lazy() -> bool {
    static L: OnceLock<bool> = OnceLock::new();
    *L.get_or_init(|| enabled() && std::env::var("XCL_LAZY_PSO").map_or(true, |v| v != "0"))
}

/// `submit`, sent to the workers only on its first `resolve` (`lazy`) – or now, at a low
/// priority, when an earlier run used a pipeline of these shaders (`key`, 0 = none; `used`).
pub fn submit_lazy(standin: usize, compute: bool, job: Job, key: u64) {
    submit_as(standin, compute, job, false, key);
}

fn submit_as(standin: usize, compute: bool, job: Job, send: bool, key: u64) {
    let p = Arc::new(Pending {
        job: Mutex::new(Some(job)),
        sent: std::sync::atomic::AtomicBool::new(send),
        real: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
        key,
        used: Default::default(),
        orphan: Default::default(),
        late: Default::default(),
        demand: Default::default(),
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        queue().push(p, false);
    } else if used::known(key) {
        if let Some(q) = preload() {
            q.push(p);
        }
    }
}

/// The keys of the pipelines earlier runs used (`used-pipelines.txt` beside the DXIL cache):
/// what `xcl shaders --precompile --used-only` builds.
pub fn earlier_used_keys() -> std::collections::HashSet<u64> {
    used::parse(&std::fs::read_to_string(crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt")).unwrap_or_default())
}

/// The pipelines a title used, by a hash of their shaders, kept between runs
/// (`used-pipelines.txt` beside the DXIL cache, a key a line).
///
/// A lazy pipeline is built at its first use: until then a graphics one draws nothing, and a
/// compute one is waited for on the title's thread. A retail title's first frames of its world
/// use hundreds of pipelines for the first time at once. The pipelines of an earlier run are
/// built while the title loads instead (`preload`): measured on a build that skips a dispatch
/// instead of waiting for its pipeline, ~2 170 dispatches were skipped in those frames without
/// this and 94 – 150 with it.
mod used {
    use std::collections::HashSet;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock, RwLock};

    struct Used {
        earlier: RwLock<HashSet<u64>>,
        now: Mutex<(HashSet<u64>, Option<std::fs::File>)>,
    }

    /// The keys in a `used-pipelines.txt` (lines that are not a hexadecimal key are passed over).
    pub fn parse(text: &str) -> HashSet<u64> {
        text.lines().filter_map(|l| u64::from_str_radix(l.trim(), 16).ok()).filter(|&k| k != 0).collect()
    }

    fn get() -> &'static Used {
        static U: OnceLock<Used> = OnceLock::new();
        U.get_or_init(|| {
            // (a test neither reads nor writes the user's file)
            if cfg!(test) {
                return Used { earlier: RwLock::new(HashSet::new()), now: Mutex::new((HashSet::new(), None)) };
            }
            let path = crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt");
            let earlier = parse(&std::fs::read_to_string(&path).unwrap_or_default());
            if !earlier.is_empty() {
                crate::xlog_info!("D3D12.X: the pipelines of {} shader sets an earlier run used are built ahead of their first use", earlier.len());
            }
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok();
            Used { now: Mutex::new((earlier.clone(), file)), earlier: RwLock::new(earlier) }
        })
    }

    /// An earlier run used a pipeline of these shaders.
    pub fn known(key: u64) -> bool {
        key != 0 && get().earlier.read().unwrap().contains(&key)
    }

    /// A pipeline of these shaders is used now: kept for the next run.
    pub fn note(key: u64) {
        if key == 0 {
            return;
        }
        let mut g = get().now.lock().unwrap();
        if g.0.insert(key) {
            if let Some(f) = g.1.as_mut() {
                let _ = writeln!(f, "{key:016x}");
            }
        }
    }

    /// (tests) This run used a pipeline of these shaders.
    #[cfg(test)]
    pub fn noted(key: u64) -> bool {
        get().now.lock().unwrap().0.contains(&key)
    }

    #[cfg(test)]
    pub fn earlier_run_used(key: u64) {
        get().earlier.write().unwrap().insert(key);
    }
}

/// The workers that build the pipelines of an earlier run ahead of their first use
/// (`XCL_PSO_PRELOAD=0`: off). Apart from the ones a bind sends to (`queue`) and at the lowest
/// priority: a pipeline that is bound is not held up by the ones made ahead (whoever takes
/// `job` first builds it), and the title's own threads are not either.
#[derive(Default)]
struct PreloadState {
    graphics: VecDeque<Arc<Pending>>,
    compute: VecDeque<Arc<Pending>>,
    active_compute: usize,
    max_compute: usize,
    closed: bool,
}

struct Preload { state: Mutex<PreloadState>, cv: Condvar }
struct PreloadComputeGuard<'a>(&'a Preload);

impl Drop for PreloadComputeGuard<'_> {
    fn drop(&mut self) {
        { let mut g = self.0.state.lock().unwrap_or_else(|e| e.into_inner()); g.active_compute = g.active_compute.saturating_sub(1); }
        self.0.cv.notify_all();
    }
}

impl Preload {
    fn new(workers: usize) -> Self {
        Self { state: Mutex::new(PreloadState { max_compute: workers.saturating_sub(1).max(1), ..Default::default() }), cv: Condvar::new() }
    }
    fn push(&self, p: Arc<Pending>) {
        let mut g = self.state.lock().unwrap();
        if p.compute { g.compute.push_back(p); } else { g.graphics.push_back(p); }
        drop(g);
        self.cv.notify_one();
    }
    fn pop(&self) -> Option<(Arc<Pending>, Option<PreloadComputeGuard<'_>>)> {
        let mut g = self.state.lock().unwrap();
        loop {
            if let Some(p) = pop_live(&mut g.graphics) { return Some((p, None)); }
            if let Some(p) = (g.active_compute < g.max_compute).then(|| pop_live(&mut g.compute)).flatten() {
                g.active_compute += 1; return Some((p, Some(PreloadComputeGuard(self))));
            }
            if g.closed { return None; }
            g = self.cv.wait(g).unwrap();
        }
    }
    #[cfg(test)]
    fn close(&self) { self.state.lock().unwrap().closed = true; self.cv.notify_all(); }
}

fn preload() -> Option<&'static Preload> {
    static Q: OnceLock<Option<Preload>> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let n = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
    let q = Q
        .get_or_init(|| {
            if std::env::var("XCL_PSO_PRELOAD").is_ok_and(|v| v.trim() == "0") {
                return None;
            }
            Some(Preload::new(n))
        })
        .as_ref()?;
    STARTED.call_once(|| {
        for k in 0..n {
            let _ = std::thread::Builder::new().name(format!("xcl-pso-pre-{k}")).stack_size(64 << 20).spawn(move || {
                unsafe {
                    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                }
                static DONE: AtomicUsize = AtomicUsize::new(0);
                while let Some((p, _guard)) = q.pop() {
                    // (released by the title meanwhile)
                    if p.orphan.load(Ordering::Acquire) {
                        continue;
                    }
                    // (None: a bind took it and builds it itself)
                    let job = p.job.lock().unwrap().take();
                    if let Some(job) = job {
                        build(&p, job, true);
                        let n = DONE.fetch_add(1, Ordering::Relaxed) + 1;
                        if n.is_power_of_two() {
                            crate::xlog_info!("D3D12.X: {n} pipelines built ahead of their first use");
                        }
                    }
                }
            });
        }
    });
    Some(q)
}

#[cfg(test)]
mod used_tests {
    use super::*;
    use std::sync::mpsc;

    /// While a bind waits for a pipeline still being built, the title is held (`title_held`):
    /// `kernelx::Sleep` does not count that time.
    #[test]
    fn a_bind_waiting_for_a_build_holds_the_title() {
        let standin = 0x7E57_B002usize;
        let (started, building) = mpsc::channel();
        let (go, finish) = mpsc::channel::<()>();
        submit_lazy(standin, true, Box::new(move || {
            let _ = started.send(());
            let _ = finish.recv();
            0x5252
        }), 0);
        let bind = std::thread::spawn(move || resolve_wait_for(standin, None));
        building.recv_timeout(std::time::Duration::from_secs(10)).expect("the pipeline is being built");
        // (built in the bind, or by a worker with the bind about to wait for it)
        let t = std::time::Instant::now();
        while !title_held() && t.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(title_held(), "a bind waits for a build, but the title is not held");
        go.send(()).unwrap();
        assert_eq!(bind.join().unwrap(), 0x5252);
        let _ = released(standin);
    }

    /// A bind waits for a pipeline still being built no longer than its budget and goes on with
    /// the stand-in; later binds do not wait for it again; once built it is the real one.
    #[test]
    fn a_bind_waits_for_a_slow_pipeline_no_longer_than_its_budget() {
        let standin = 0x7E57_B001usize;
        submit(standin, true, Box::new(|| {
            std::thread::sleep(std::time::Duration::from_millis(600));
            0x5151
        }));
        let budget = Some(std::time::Duration::from_millis(30));
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin, "not built within the budget: the stand-in");
        assert!(t.elapsed() < std::time::Duration::from_millis(400), "waited {:?}", t.elapsed());
        let t = std::time::Instant::now();
        assert_eq!(resolve_wait_for(standin, budget), standin);
        assert!(t.elapsed() < std::time::Duration::from_millis(20), "a late pipeline is not waited for again ({:?})", t.elapsed());
        let start = std::time::Instant::now();
        while is_pending(standin) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(resolve_wait_for(standin, budget), 0x5151, "built by a worker");
        let _ = released(standin);
    }

    /// A compile a worker's build starts is wanted once a bind waits for the build – asked from
    /// another thread, as the DXC workers do; a thread that builds nothing (no title runs here)
    /// wants nothing.
    #[test]
    fn a_waiting_bind_makes_the_builds_compiles_wanted() {
        let standin = 0x7E57_B005usize;
        let (tx, rx) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            tx.send(wanted_here()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            0x5454
        }));
        let wanted = rx.recv().unwrap();
        assert!(!wanted(), "nobody waits yet");
        assert_eq!(resolve_wait_for(standin, None), 0x5454);
        assert!(wanted());
        assert!(!wanted_here()());
        let _ = released(standin);
    }

    /// A worker's build hears that a bind waits for it (and only its own build does).
    #[test]
    fn a_waiting_bind_makes_the_build_wanted() {
        let standin = 0x7E57_B004usize;
        let (started_tx, started) = mpsc::channel();
        submit(standin, true, Box::new(move || {
            let unwanted_first = !build_is_wanted();
            started_tx.send(()).unwrap();
            let t = std::time::Instant::now();
            while !build_is_wanted() && t.elapsed() < std::time::Duration::from_secs(10) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            if unwanted_first && build_is_wanted() { 0x5353 } else { 0 }
        }));
        started.recv().unwrap();
        assert!(!build_is_wanted(), "this thread builds nothing");
        assert_eq!(resolve_wait_for(standin, None), 0x5353, "the bind's wait reached the build");
        let _ = released(standin);
    }

    #[test]
    fn the_keys_of_a_file_are_read_and_other_lines_passed_over() {
        let keys = used::parse("00000000000000ff\n\nnot a key\n  1a2b \n0\n");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&0xff) && keys.contains(&0x1a2b));
    }

    /// A lazy pipeline of shaders an earlier run used is built without a bind; another one is
    /// built only at its first use, and that use is kept for the next run.
    #[test]
    fn a_pipeline_an_earlier_run_used_is_built_ahead_of_its_first_use() {
        let (known, unknown) = (0x7E57_7001usize, 0x7E57_7002usize);
        let (known_key, unknown_key) = (0x5EED_0001u64, 0x5EED_0002u64);
        used::earlier_run_used(known_key);
        submit_lazy(known, false, Box::new(|| 0x4321), known_key);
        submit_lazy(unknown, false, Box::new(|| 0x4322), unknown_key);
        let start = std::time::Instant::now();
        while is_pending(known) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(known), "built by the workers that build ahead, with no bind");
        assert_eq!(resolve(known), 0x4321);
        assert!(is_pending(unknown), "nobody used it and no earlier run did");
        assert!(!used::noted(unknown_key));
        assert_eq!(resolve_wait(unknown), 0x4322);
        assert!(used::noted(unknown_key) && used::noted(known_key), "a first use is kept for the next run");
        released(known);
        released(unknown);
    }
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.lock().unwrap().as_ref()?.get(&pso).cloned()
}

/// The pipeline to use for `pso` now: the real one once built, else `pso` itself.
pub fn resolve(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
    match p.real.load(Ordering::Acquire) {
        0 => {
            // a lazy one is wanted now: to the workers
            if !p.sent.swap(true, Ordering::AcqRel) {
                queue().push(p, false);
            }
            pso
        }
        r => r,
    }
}

/// Atomic outcome of waiting for a pipeline's background build (`resolve_wait_checked`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolveOutcome { Untracked(usize), Ready(usize), Pending, Failed }

fn outcome_when_done(p: &Pending) -> ResolveOutcome {
    match p.real.load(Ordering::Acquire) { 0 => ResolveOutcome::Failed, r => ResolveOutcome::Ready(r) }
}

pub fn resolve_wait_checked(pso: usize) -> ResolveOutcome { resolve_wait_for_checked(pso, wait_budget()) }

/// `resolve`, waiting for a pipeline still being built – built here if no worker has started
/// it yet (a queue of a title's hundreds of pipelines must not hold up the one it binds now).
pub fn resolve_wait(pso: usize) -> usize {
    resolve_wait_for(pso, wait_budget())
}

/// `resolve_wait`, waiting at most `budget` (None: until the pipeline is built).
fn resolve_wait_for(pso: usize, budget: Option<std::time::Duration>) -> usize {
    match resolve_wait_for_checked(pso, budget) { ResolveOutcome::Ready(r) | ResolveOutcome::Untracked(r) => r, ResolveOutcome::Pending | ResolveOutcome::Failed => pso }
}

fn resolve_wait_for_checked(pso: usize, budget: Option<std::time::Duration>) -> ResolveOutcome {
    let Some(p) = entry(pso) else { return ResolveOutcome::Untracked(pso) };
    p.note_use();
    resolve_pending_for(&p, budget)
}

fn resolve_pending_for(p: &Arc<Pending>, budget: Option<std::time::Duration>) -> ResolveOutcome {
    let done = p.done.lock().unwrap();
    if *done { return outcome_when_done(p); }
    drop(done);
    if let Some(budget) = budget { return resolve_within(p, budget); }
    let _held = Held::new();
    let job = p.job.lock().unwrap().take();
    if let Some(job) = job {
        complete(p, job());
    }
    p.want();
    let mut done = p.done.lock().unwrap();
    while !*done {
        done = p.cv.wait(done).unwrap();
    }
    outcome_when_done(p)
}

#[cfg(test)]
pub(super) fn with_test_wait_budget<R>(budget: Option<std::time::Duration>, f: impl FnOnce() -> R) -> R {
    struct Reset(Option<Option<std::time::Duration>>);
    impl Drop for Reset { fn drop(&mut self) { TEST_WAIT_BUDGET.with(|b| b.set(self.0)); } }
    let _reset = Reset(TEST_WAIT_BUDGET.with(|b| b.replace(Some(budget))));
    f()
}

/// How long a bind waits for a pipeline still being built (`XCL_PSO_WAIT_MS`; unset or 0: until
/// it is built). A retail game builds compute programs on its own threads: the driver takes up to
/// 30 s for the biggest translated ones, and the title's watchdog ends it when its main thread
/// has been held 30 s.
fn wait_budget() -> Option<std::time::Duration> {
    #[cfg(test)]
    if let Some(budget) = TEST_WAIT_BUDGET.with(|b| b.get()) { return budget; }
    static B: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_PSO_WAIT_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&ms| ms > 0).map(std::time::Duration::from_millis))
}

/// `resolve_wait` for at most `budget`: the build goes to the workers (not this thread), and a
/// pipeline not built by then is `late` – it and later binds of it get the stand-in (their
/// dispatches are skipped) until a worker has built it.
fn resolve_within(p: &Arc<Pending>, budget: std::time::Duration) -> ResolveOutcome {
    let newly_sent = !p.sent.swap(true, Ordering::AcqRel);
    if p.late.load(Ordering::Acquire) {
        if newly_sent {
            queue().push(p.clone(), false);
        }
        return ResolveOutcome::Pending;
    }
    // (the title's thread is held from here until the pipeline is built or the budget is over)
    let _held = Held::new();
    p.want();
    if newly_sent {
        queue().push(p.clone(), true);
    } else {
        queue().promote(p.clone());
    }
    let (done, _) = p.cv.wait_timeout_while(p.done.lock().unwrap(), budget, |d| !*d).unwrap();
    if *done {
        return outcome_when_done(p);
    }
    drop(done);
    p.late.store(true, Ordering::Release);
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    if n < 16 || n.is_power_of_two() {
        crate::xlog_warn!("D3D12.X: a pipeline still being built after {} ms: its binds go on without it until it is built ({} so far)", budget.as_millis(), n + 1);
    }
    ResolveOutcome::Pending
}

/// For binding: compute pipelines are waited for, graphics ones are not.
pub fn resolve_for_bind(pso: usize) -> usize {
    match entry(pso) {
        Some(p) if p.compute => resolve_wait(pso),
        Some(_) => resolve(pso),
        None => pso,
    }
}

/// The title released stand-in `pso` for good: forget it (its address may come back for
/// another object) and return its real pipeline to release (0: none, or still being built –
/// the worker's result is then released when it arrives).
pub fn released(pso: usize) -> usize {
    let Some(p) = PENDING.lock().unwrap().as_mut().and_then(|m| m.remove(&pso)) else { return 0 };
    let done = p.done.lock().unwrap();
    if *done {
        p.real.swap(0, Ordering::AcqRel)
    } else {
        // the worker stores its result into a Pending nobody will read: mark it orphaned
        p.orphan.store(true, Ordering::Release);
        0
    }
}

/// `pso` is a stand-in whose real pipeline is still being built.
pub fn is_pending(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !*p.done.lock().unwrap())
}

/// (built in the background, failed, still building)
pub fn counts() -> (usize, usize, usize) {
    let pending = PENDING.lock().unwrap().as_ref().map_or(0, |m| m.values().filter(|p| !*p.done.lock().unwrap()).count());
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), pending)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, time::Duration};

    fn test_pending(compute: bool, key: u64, job: Option<Job>) -> Arc<Pending> {
        Arc::new(Pending {
            job: Mutex::new(job), sent: std::sync::atomic::AtomicBool::new(true), real: AtomicUsize::new(0),
            done: Mutex::new(false), cv: Condvar::new(), compute, key, used: Default::default(),
            orphan: Default::default(), late: Default::default(), demand: Default::default(),
        })
    }

    #[test]
    fn wanted_job_preempts_queued_background_jobs() {
        let q = WorkQueue::new(4);
        let gfx2 = test_pending(false, 2, None);
        for (c, k, w) in [(true, 3, false), (false, 1, false), (true, 4, false), (true, 5, true)] { q.push(test_pending(c, k, None), w); }
        q.push(gfx2.clone(), false);
        q.promote(gfx2);
        let (p, g) = q.pop().unwrap();
        assert_eq!((p.key, g.is_none()), (2, true));
        p.real.store(2, Ordering::Release);
        for (k, cs) in [(5, true), (1, false), (3, true), (4, true)] { let (p, g) = q.pop().unwrap(); assert_eq!((p.key, g.is_some()), (k, cs)); }
        q.close();
        assert!(q.pop().is_none());
    }

    #[test]
    fn graphics_jobs_do_not_starve_when_compute_workers_are_wedged() {
        for (w, cap) in [(1, 1), (2, 1), (4, 3), (8, 6)] { assert_eq!(WorkQueue::new(w).state.lock().unwrap().max_compute, cap); }
        let q = Arc::new(WorkQueue::new(4));
        let ((cs_tx, cs_rx), (gfx_tx, gfx_rx)) = (mpsc::channel(), mpsc::channel());
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let workers: Vec<_> = (0..4).map(|_| {
            let q = q.clone();
            std::thread::spawn(move || while let Some((p, _g)) = q.pop() { if let Some(j) = p.job.lock().unwrap().take() { let _ = j(); } })
        }).collect();
        for id in 1..=4u64 {
            let (tx, gate) = (cs_tx.clone(), gate.clone());
            q.push(test_pending(true, id, Some(Box::new(move || {
                let _ = tx.send(id);
                let mut open = gate.0.lock().unwrap();
                while !*open { open = gate.1.wait(open).unwrap(); }
                id as usize
            }))), false);
        }
        for _ in 0..3 { cs_rx.recv_timeout(Duration::from_secs(5)).expect("3 compute workers start"); }
        for id in 10..13u64 { let tx = gfx_tx.clone(); q.push(test_pending(false, id, Some(Box::new(move || { let _ = tx.send(id); id as usize }))), false); }
        for _ in 0..3 { gfx_rx.recv_timeout(Duration::from_secs(5)).expect("graphics runs on reserved lane"); }
        assert!(cs_rx.try_recv().is_err(), "4th compute waits while max_compute is wedged");
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
        cs_rx.recv_timeout(Duration::from_secs(5)).expect("4th compute runs once slot frees");
        q.close();
        for w in workers { w.join().unwrap(); }
    }

    #[test]
    fn orphaned_job_is_skipped_in_work_queue() {
        let q = WorkQueue::new(2);
        let (ow, oc, done, late_oc) = (test_pending(false, 1, None), test_pending(true, 2, None), test_pending(false, 3, None), test_pending(true, 4, None));
        ow.orphan.store(true, Ordering::Release);
        oc.orphan.store(true, Ordering::Release);
        done.real.store(3, Ordering::Release);
        for (p, w) in [(ow, true), (oc, true), (done, false), (test_pending(false, 5, None), false), (late_oc.clone(), false), (test_pending(true, 6, None), false)] { q.push(p, w); }
        late_oc.orphan.store(true, Ordering::Release);
        for (k, cs) in [(5, false), (6, true)] { let (p, g) = q.pop().unwrap(); assert_eq!((p.key, g.is_some()), (k, cs)); }
        q.close();
        assert!(q.pop().is_none());
        assert_eq!(q.state.lock().unwrap().active_compute, 0);
    }

    #[test]
    fn preload_drains_graphics_before_compute_and_reserves_graphics_lane() {
        let preload = Arc::new(Preload::new(2));
        let orphan = test_pending(false, 99, None);
        orphan.orphan.store(true, Ordering::Release);
        for p in [test_pending(true, 1, None), orphan, test_pending(false, 2, None), test_pending(true, 3, None)] { preload.push(p); }
        assert_eq!(preload.pop().unwrap().0.key, 2);
        let (p, cs_guard) = preload.pop().unwrap();
        assert_eq!((p.key, cs_guard.is_some()), (1, true));
        let (tx, rx) = mpsc::channel();
        let p2 = preload.clone();
        let worker = std::thread::spawn(move || while let Some((item, _g)) = p2.pop() { let _ = tx.send(item.key); });
        preload.push(test_pending(false, 4, None));
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)).expect("graphics preload runs while compute slot held"), 4);
        assert!(rx.try_recv().is_err(), "2nd compute preload waits while max_compute is active");
        drop(cs_guard);
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)).expect("compute preload wakes when slot frees"), 3);
        preload.close();
        worker.join().unwrap();
    }

    #[test]
    fn compute_slot_guard_releases_slot_on_panic() {
        let q = WorkQueue::new(2);
        for k in [1, 2] { q.push(test_pending(true, k, None), false); }
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (p, _g) = q.pop().unwrap();
            assert_eq!((p.key, q.state.lock().unwrap().active_compute), (1, 1));
            panic!("worker panic");
        })).is_err());
        assert_eq!(q.state.lock().unwrap().active_compute, 0);
        assert_eq!(q.pop().unwrap().0.key, 2);

        let preload = Preload::new(2);
        for k in [10, 11] { preload.push(test_pending(true, k, None)); }
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (p, _g) = preload.pop().unwrap();
            assert_eq!((p.key, preload.state.lock().unwrap().active_compute), (10, 1));
            panic!("preload panic");
        })).is_err());
        assert_eq!(preload.state.lock().unwrap().active_compute, 0);
        assert_eq!(preload.pop().unwrap().0.key, 11);
    }

    #[test]
    fn resolve_wait_checked_distinguishes_untracked_ready_pending_and_failed() {
        let (untracked, ready, pending, failed) = (0x7E57_C001usize, 0x7E57_C002usize, 0x7E57_C003usize, 0x7E57_C004usize);
        assert_eq!((resolve_wait_checked(untracked), resolve_wait_checked(0)), (ResolveOutcome::Untracked(untracked), ResolveOutcome::Untracked(0)));
        let (go_tx, go_rx) = mpsc::channel::<()>();
        submit_lazy(ready, false, Box::new(|| 0x7E57_C101), 0);
        submit_lazy(failed, false, Box::new(|| 0), 0);
        submit_lazy(pending, false, Box::new(move || { let _ = go_rx.recv_timeout(Duration::from_secs(5)); 0 }), 0);
        assert_eq!((resolve_wait_for_checked(ready, None), resolve_wait_for_checked(ready, Some(Duration::from_millis(5))), resolve_wait(ready)), (ResolveOutcome::Ready(0x7E57_C101), ResolveOutcome::Ready(0x7E57_C101), 0x7E57_C101));
        assert_eq!((resolve_wait_for_checked(failed, None), resolve_wait_for_checked(failed, Some(Duration::from_millis(5))), resolve_wait(failed)), (ResolveOutcome::Failed, ResolveOutcome::Failed, failed));
        assert_eq!(with_test_wait_budget(Some(Duration::from_millis(5)), || resolve_wait_checked(pending)), ResolveOutcome::Pending);
        assert!(is_pending(pending));
        for k in [ready, failed, pending] { let _ = released(k); }
        let _ = go_tx.send(());
        assert_eq!(resolve_wait_checked(ready), ResolveOutcome::Untracked(ready));
    }

    #[test]
    fn late_pipeline_does_not_hold_title_on_subsequent_binds() {
        let (p, held) = (test_pending(true, 0, None), || HELD_ACQUIRES.with(|c| c.get()));
        let b = held();
        assert_eq!(resolve_pending_for(&p, Some(Duration::from_millis(1))), ResolveOutcome::Pending);
        assert_eq!((held(), p.late.load(Ordering::Acquire)), (b + 1, true));
        let (b, t) = (held(), std::time::Instant::now());
        assert_eq!(resolve_pending_for(&p, Some(Duration::from_secs(10))), ResolveOutcome::Pending);
        assert_eq!(held(), b, "late pipeline must not construct Held");
        assert!(t.elapsed() < Duration::from_secs(1));
        complete(&p, 0x7E57_C201);
        let b = held();
        assert_eq!(resolve_pending_for(&p, Some(Duration::from_secs(10))), ResolveOutcome::Ready(0x7E57_C201));
        assert_eq!(held(), b, "completed pipeline must not construct Held");
    }
}
