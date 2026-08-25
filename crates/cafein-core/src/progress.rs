//! Per-unit completion callbacks for long fan-outs.

/// Called once per completed unit (an origin, a request); `None` is
/// free. Callbacks must be cheap and thread-safe — they run on rayon
/// workers.
pub type Progress<'a> = Option<&'a (dyn Fn() + Sync)>;
