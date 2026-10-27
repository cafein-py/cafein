//! The bridge that carries log records to Python's `logging`.
//!
//! Python arms the bridge by syncing the minimum effective level over
//! the `cafein.build`/`cafein.matrix`/`cafein.artifact` loggers (capped
//! at INFO while a timing collector is active) into an atomic
//! threshold; a record below the threshold costs one relaxed load and
//! no GIL activity. Armed records are delivered
//! through the dispatch callable `cafein._log` registers at import,
//! and errors are swallowed — logging must never break a computation.
//! Progress ticks have an arm of their own: Python sets it while it
//! draws progress, whatever the log level.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::Instant;

use pyo3::prelude::*;

pub const DEBUG: u32 = 10;
pub const INFO: u32 = 20;
pub const WARNING: u32 = 30;

/// The emission threshold; `u32::MAX` is the disarmed initial state,
/// above every Python level and never produced by the Python sync.
static THRESHOLD: AtomicU32 = AtomicU32::new(u32::MAX);
static DISPATCH: OnceLock<Py<PyAny>> = OnceLock::new();
/// Whether Python draws progress: ticks are then delivered at any
/// threshold.
static PROGRESS: AtomicBool = AtomicBool::new(false);
/// The next fan-out's run id, which keeps concurrent runs of one label
/// apart.
static NEXT_RUN: AtomicU64 = AtomicU64::new(1);

/// Arm the bridge: records at `level` and above get delivered.
#[pyfunction]
pub fn set_log_level(level: u32) {
    THRESHOLD.store(level, Ordering::Relaxed);
}

/// Arm or disarm progress ticks independently of the threshold.
#[pyfunction]
pub fn set_progress(on: bool) {
    PROGRESS.store(on, Ordering::Relaxed);
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
    emit_full(target, level, message, phase, seconds, None)
}

/// [`emit`] with the progress triple ticks carry: `(label, done,
/// total)`, `None` on every non-tick record.
pub fn emit_full<F>(
    target: &str,
    level: u32,
    message: F,
    phase: Option<&str>,
    seconds: Option<f64>,
    progress: Option<(&str, usize, usize)>,
) where
    F: FnOnce() -> String,
{
    if !enabled(level) {
        return;
    }
    deliver(target, level, message, phase, seconds, progress, None);
}

/// The progress of one fan-out run: its id, `"start"`/`"end"` for its
/// opening and closing (which Python presents but never logs) or `None`
/// on a tick.
type RunEvent<'a> = Option<(u64, Option<&'a str>)>;

/// Hand one record to the Python dispatch, unconditionally; `run` is
/// `None` on every record that is no fan-out progress.
fn deliver<F>(
    target: &str,
    level: u32,
    message: F,
    phase: Option<&str>,
    seconds: Option<f64>,
    progress: Option<(&str, usize, usize)>,
    run: RunEvent<'_>,
) where
    F: FnOnce() -> String,
{
    let Some(dispatch) = DISPATCH.get() else {
        return;
    };
    let message = message();
    let (run, event) = run.unzip();
    let event = event.flatten();
    Python::with_gil(|py| {
        let _ = dispatch.call1(
            py,
            (target, level, message, phase, seconds, progress, event, run),
        );
    });
}

/// A build diagnostic: a `UserWarning` attributed to the first caller
/// outside cafein, and the same text as a WARNING record on the build
/// logger.
pub fn build_warning(py: Python<'_>, message: String) -> PyResult<()> {
    emit("cafein.build", WARNING, || message.clone(), None, None);
    py.import("cafein._log")?.call_method1("warn", (message,))?;
    Ok(())
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

/// Emits throttled INFO progress lines from a fan-out: one `tick` per
/// completed unit (an origin, a stop, a step), a line at each ~5%
/// boundary (at most ~20 GIL acquisitions per run), between a start and
/// an end event. Constructed disarmed when neither logging nor progress
/// is on, or the fan-out is small; disarmed call sites pass the core
/// `None` so the fan-out runs exactly as before.
pub struct ProgressTicker {
    label: &'static str,
    unit: &'static str,
    total: usize,
    step: usize,
    completed: AtomicUsize,
    /// The last boundary reported; boundary holders emit every
    /// pending boundary in order under this lock, so lines never
    /// regress however the workers interleave.
    emitted: Mutex<usize>,
    started: Instant,
    armed: bool,
    run: u64,
}

impl ProgressTicker {
    /// Runs below this many origins never tick; their phase
    /// completion is report enough.
    const MIN_TOTAL: usize = 40;

    /// A fan-out over `total` origins.
    pub fn new(label: &'static str, total: usize) -> Self {
        Self::counting(label, total, "origins")
    }

    /// A fan-out over `total` units of the kind `unit`.
    pub fn counting(label: &'static str, total: usize, unit: &'static str) -> Self {
        Self::armed_from(label, total, unit, Self::MIN_TOTAL)
    }

    /// An operation of `total` steps, each ticked when done: it has
    /// progress however few its steps.
    pub fn steps(label: &'static str, total: usize) -> Self {
        Self::armed_from(label, total, "steps", 1)
    }

    fn armed_from(label: &'static str, total: usize, unit: &'static str, minimum: usize) -> Self {
        let armed = total >= minimum
            && (enabled(INFO) || PROGRESS.load(Ordering::Relaxed))
            && DISPATCH.get().is_some();
        let run = if armed {
            NEXT_RUN.fetch_add(1, Ordering::Relaxed)
        } else {
            0
        };
        if armed {
            Self::event(run, label, 0, total, "start");
        }
        ProgressTicker {
            label,
            unit,
            total,
            step: total.div_ceil(20).max(1),
            completed: AtomicUsize::new(0),
            emitted: Mutex::new(0),
            started: Instant::now(),
            armed,
            run,
        }
    }

    /// One completed unit. On a worker thread, call only where the GIL
    /// is released.
    pub fn tick(&self) {
        if !self.armed {
            return;
        }
        let done = self.completed.fetch_add(1, Ordering::Relaxed) + 1;
        let last = done == self.total;
        if !done.is_multiple_of(self.step) && !last {
            return;
        }
        let mut emitted = self.emitted.lock().unwrap_or_else(PoisonError::into_inner);
        // The last unit closes the run at its total, a boundary or not;
        // nothing is reported past the total.
        while *emitted < done.min(self.total) && (*emitted + self.step <= done || last) {
            *emitted = (*emitted + self.step).min(self.total);
            let at = *emitted;
            let (label, unit, total) = (self.label, self.unit, self.total);
            let percent = at * 100 / total;
            let seconds = self.started.elapsed().as_secs_f64();
            deliver(
                "cafein.matrix",
                INFO,
                || format!("{label} {percent}% ({at}/{total} {unit}, {seconds:.1} s elapsed)"),
                None,
                None,
                Some((label, at, total)),
                Some((self.run, None)),
            );
        }
    }

    fn event(run: u64, label: &str, done: usize, total: usize, event: &str) {
        deliver(
            "cafein.matrix",
            INFO,
            String::new,
            None,
            None,
            Some((label, done, total)),
            Some((run, Some(event))),
        );
    }
}

impl Drop for ProgressTicker {
    /// The end event, also when the fan-out stopped early.
    fn drop(&mut self) {
        if self.armed {
            let done = self.completed.load(Ordering::Relaxed);
            Self::event(self.run, self.label, done, self.total, "end");
        }
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
