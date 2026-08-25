//! The bridge that carries log records to Python's `logging`.
//!
//! Python arms the bridge by syncing the `"cafein"` logger's effective
//! level into an atomic threshold; a record below the threshold costs
//! one relaxed load and no GIL activity. Armed records are delivered
//! through the dispatch callable `cafein._log` registers at import,
//! and errors are swallowed — logging must never break a computation.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use pyo3::prelude::*;

pub const DEBUG: u32 = 10;
pub const INFO: u32 = 20;

/// The emission threshold; `u32::MAX` is the disarmed initial state,
/// above every Python level and never produced by the Python sync.
static THRESHOLD: AtomicU32 = AtomicU32::new(u32::MAX);
static DISPATCH: OnceLock<Py<PyAny>> = OnceLock::new();

/// Arm the bridge: records at `level` and above get delivered.
#[pyfunction]
pub fn set_log_level(level: u32) {
    THRESHOLD.store(level, Ordering::Relaxed);
}

/// Register the Python dispatch records are delivered through; the
/// first registration wins.
#[pyfunction]
pub fn install_log_dispatch(dispatch: Py<PyAny>) {
    let _ = DISPATCH.set(dispatch);
}

pub fn enabled(level: u32) -> bool {
    level >= THRESHOLD.load(Ordering::Relaxed)
}

/// Deliver one record to Python, re-acquiring the GIL. The message is
/// built lazily so a disarmed bridge costs one atomic load and no
/// allocation. Callers on worker threads may only emit inside
/// `allow_threads` sections.
pub fn emit<F>(target: &str, level: u32, message: F, phase: Option<&str>, seconds: Option<f64>)
where
    F: FnOnce() -> String,
{
    if !enabled(level) {
        return;
    }
    let Some(dispatch) = DISPATCH.get() else {
        return;
    };
    let message = message();
    Python::with_gil(|py| {
        let _ = dispatch.call1(py, (target, level, message, phase, seconds));
    });
}

/// Times one phase: a DEBUG starting line at construction, the INFO
/// completion with the structured attributes at `finish`. Dropping the
/// timer without finishing (an error path) emits no completion — the
/// raised error is the report.
pub struct PhaseTimer {
    target: &'static str,
    phase: &'static str,
    done: &'static str,
    started: Instant,
}

impl PhaseTimer {
    pub fn start(
        target: &'static str,
        phase: &'static str,
        doing: &'static str,
        done: &'static str,
    ) -> Self {
        emit(target, DEBUG, || doing.to_string(), None, None);
        PhaseTimer {
            target,
            phase,
            done,
            started: Instant::now(),
        }
    }

    pub fn finish(self) {
        let seconds = self.started.elapsed().as_secs_f64();
        let done = self.done;
        emit(
            self.target,
            INFO,
            || format!("{done} in {seconds:.1} s"),
            Some(self.phase),
            Some(seconds),
        );
    }
}

/// Emits throttled INFO progress lines from an origin fan-out: one
/// `tick` per completed origin, a line at each ~5% boundary (at most
/// ~20 GIL acquisitions per run). Constructed disarmed when logging is
/// off or the run is small; disarmed call sites pass the core `None`
/// so the fan-out runs exactly as before.
pub struct ProgressTicker {
    label: &'static str,
    total: usize,
    step: usize,
    completed: AtomicUsize,
    started: Instant,
    armed: bool,
}

impl ProgressTicker {
    /// Runs below this many origins never tick; their phase
    /// completion is report enough.
    const MIN_TOTAL: usize = 40;

    pub fn new(label: &'static str, total: usize) -> Self {
        let armed = total >= Self::MIN_TOTAL && enabled(INFO) && DISPATCH.get().is_some();
        ProgressTicker {
            label,
            total,
            step: total.div_ceil(20).max(1),
            completed: AtomicUsize::new(0),
            started: Instant::now(),
            armed,
        }
    }

    /// One completed origin. Call only where the GIL is released.
    pub fn tick(&self) {
        if !self.armed {
            return;
        }
        let done = self.completed.fetch_add(1, Ordering::Relaxed) + 1;
        if !done.is_multiple_of(self.step) {
            return;
        }
        let (label, total) = (self.label, self.total);
        let percent = done * 100 / total;
        let seconds = self.started.elapsed().as_secs_f64();
        emit(
            "cafein.matrix",
            INFO,
            || format!("{label} {percent}% ({done}/{total} origins, {seconds:.1} s elapsed)"),
            None,
            None,
        );
    }
}

/// The core-facing hook for a ticker: `None` when disarmed, so the
/// fan-out's hot path carries no counter at all.
pub fn progress_hook<'a>(
    ticker: &ProgressTicker,
    tick: &'a (dyn Fn() + Sync),
) -> cafein_core::progress::Progress<'a> {
    if ticker.armed {
        Some(tick)
    } else {
        None
    }
}
