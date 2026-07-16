//! Trip-Based Transit Routing: the query-day trip universe and the
//! precomputed trip-to-trip transfer set.
//!
//! TBTR (Witt, ESA 2015) replaces RAPTOR's per-stop labels with trip
//! segments linked by precomputed transfers: alight a trip at a
//! position, walk (or stay at the stop), board another trip at a
//! position. Generation keeps, per reachable (line, position), only
//! the earliest boardable trip; a reduction pass then drops transfers
//! that improve no stop's arrival over staying on the trip or over the
//! transfers already kept — typically the large majority — leaving the
//! set the query engine scans. The reduction is tie-complete: a
//! transfer that exactly ties a kept competitor from a *different*
//! trip is retained too (as is each trip's earliest tied boarding), so
//! cost reconstruction can elect the same journey RAPTOR's
//! tie-breaking does; ties against staying on the trip still prune.
//!
//! Both passes run over a [`DayView`]: the virtual trips one query
//! date sees. Restricting to a date before the reduction is what keeps
//! the reduction exact — dropped transfers are judged against exactly
//! the trips that run — and it folds the previous service day's
//! over-midnight tails in as *lines of their own*, shifted back a day,
//! so no service check or day arithmetic is left inside the query
//! loop. The all-trips [`DayView::universal`] view serves calendar-free
//! uses (and the whole-feed diagnostics the tests pin).

use std::collections::HashMap;

use rayon::prelude::*;

use crate::journey::{Journey, Leg};
use crate::path_key::{challenger_wins, PathToken};
use crate::raptor::{CostInputs, CostRow};
use crate::router::{Request, TransitRouter};
use crate::timetable::{PatternIdx, StopIdx, StopTime, Timetable, TripIdx};
use crate::transfers::Transfers;

const UNREACHED: u32 = u32::MAX;

/// Seconds in a service day: the shift of previous-day lines.
const DAY_SECONDS: u32 = 86_400;

mod costs;
mod engine;
mod matrix;
mod set;
mod tokens;
mod view;

use engine::*;
pub use engine::{Tbtr, TbtrEngine};
pub use matrix::MatrixState;
use matrix::*;
pub(crate) use set::earliest_boardable;
use set::*;
pub use set::{TransferSet, TransferSetBuild, TripTransfer};
pub use view::{DayView, ViewTrip};

#[cfg(test)]
mod tests;
