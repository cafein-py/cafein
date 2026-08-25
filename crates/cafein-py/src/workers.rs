//! Per-call fan-out width: a rayon local pool installed for one
//! computation, leaving the global pool (and `RAYON_NUM_THREADS`)
//! untouched.

use pyo3::prelude::*;

/// Run `work` on a local pool of `workers` threads (`None`: the
/// ambient pool). One DEBUG line reports the effective width through
/// the logging bridge — read inside the installed context, so it is
/// the width the fan-out actually rides. A pool-build failure falls
/// back to the ambient pool: width is a preference, never worth
/// failing a computation for.
pub fn with_workers<R: Send>(
    label: &'static str,
    workers: Option<usize>,
    work: impl FnOnce() -> R + Send,
) -> R {
    let run = || {
        crate::logging::emit(
            "cafein.matrix",
            crate::logging::DEBUG,
            || format!("{label} on {} workers", rayon::current_num_threads()),
            None,
            None,
        );
        work()
    };
    let Some(n) = workers else {
        return run();
    };
    match rayon::ThreadPoolBuilder::new().num_threads(n).build() {
        Ok(pool) => pool.install(run),
        Err(_) => run(),
    }
}

/// The width `with_workers` would run at — the test seam for the
/// mechanism itself.
#[pyfunction]
#[pyo3(signature = (workers=None))]
pub fn _probe_workers(workers: Option<usize>) -> usize {
    with_workers("probe", workers, rayon::current_num_threads)
}
