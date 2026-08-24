//! The bridge that carries log records to Python's `logging`.
//!
//! Python arms the bridge by syncing the `"cafein"` logger's effective
//! level into an atomic threshold; a record below the threshold costs
//! one relaxed load and no GIL activity. Armed records are delivered
//! through the dispatch callable `cafein._log` registers at import,
//! and errors are swallowed — logging must never break a computation.

use std::sync::atomic::{AtomicU32, Ordering};
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

/// Deliver one record to Python, re-acquiring the GIL. Callers on
/// worker threads may only emit inside `allow_threads` sections.
pub fn emit(target: &str, level: u32, message: String, phase: Option<&str>, seconds: Option<f64>) {
    if !enabled(level) {
        return;
    }
    let Some(dispatch) = DISPATCH.get() else {
        return;
    };
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
        emit(target, DEBUG, doing.to_string(), None, None);
        PhaseTimer {
            target,
            phase,
            done,
            started: Instant::now(),
        }
    }

    pub fn finish(self) {
        let seconds = self.started.elapsed().as_secs_f64();
        let message = format!("{} in {:.1} s", self.done, seconds);
        emit(self.target, INFO, message, Some(self.phase), Some(seconds));
    }
}
